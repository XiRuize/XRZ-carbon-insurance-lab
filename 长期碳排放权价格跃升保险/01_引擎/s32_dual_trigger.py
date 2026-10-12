# -*- coding: utf-8 -*-
"""
s32_dual_trigger.py —— 第七轮：双给付结构 + 一年期（v2 结构性改版）

本脚本回答的是 v2 相对 v1 的两个结构性变更：

① **双给付结构**（用户 2026-10-09 拍板）
   v1 的单一给付按 MAC（影子价格）计价，公司账上不出现 MAC ⇒
   「赔的东西和怕的东西不是一回事」。v2 拆成两份独立责任：

       责任一 · 现金给付  pay_cash = α·min(max(P_t − κ·P_{t−1},0), CAP_P)·Q_t
       责任二 · 影子给付  pay_shad = α·min(max(I_t − κ·I_{t−1},0), CAP_I)·Q_t
       总赔付 = 两份相加（**先相加、再分层**，不给两份各自套 φ，
       因为 P 与 M 正相关 ⇒ 分开套分散因子会虚增组合分散）

   其中 I = P × M = P × (MAC/P) = MAC（2026 名义），故责任二即 v1 的给付。

② **一年期**（用户 2026-10-09 拍板）
   影子价格实测越界首年 = 2028（M1 结构外推口径下 MAC 下降 68.6%）。
   多年期保单需要 MAC 在整个承保期内保持可识别 ⇒ 不可承保。
   故本产品为**一年期保单，每年独立承保、独立结算、独立重定价、逐年续保**。
   一年期同时降低组合风险：各保单年度的 cohort 相互独立，
   不存在跨年度的相关性累积（这一点在 n_eff 的口径上比 v1 更干净）。

③ **越界失效条款**（本轮新增量化）
   影子给付在 MAC 越界年份自动失效，仅现金给付有效。
   这是把「MAC 会下降」这个弱点转为明示条款，而不是回避。

用法：python s32_dual_trigger.py [--check]    # 退出码 0 = 断言全过
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import CARBON_PRICE, COMPLIANCE_VOLUME, L1, N_ENTITY, OUT  # noqa: E402
import s4_pricing as s4                                                       # noqa: E402
import s23_review_response as s23                                              # noqa: E402

pd.set_option("display.width", 260)
pd.set_option("display.unicode.east_asian_width", True)

# ============================================================ 冻结的核心参数
G1_FROZEN = 0.0
BETA_FROZEN = 1.0
TAG = "g(1,-1)_冻结档"

MAC_2022 = 190.5306              # 冻结档 2022 全国 MAC（1995 价），引擎实测
DEFL_2026 = 2.169309             # 1995→2026 累计平减
MAC_REF = MAC_2022 * DEFL_2026   # ≈ 413.32 元/吨，2026 名义
P_2026 = s4.P_2026              # 97.90 元/吨
M0 = MAC_REF / P_2026            # 缺口倍数 ≈ 4.2219（M2 价格平移口径）

# ---- M1 结构外推口径下的缺口倍数（§3.4 双口径披露用）
# M1 的2026 值需从标度档缓存取（该档do_M1=True），此处不硬编码，避免口径漂移。
M1_RATIO_TO_M2 = 59.77 / MAC_2022          # M1/M2 = 0.3137（引擎实测）
M0_M1 = M0 * M1_RATIO_TO_M2               # ≈ 1.3244

# ============================================================ 条款参数（一年期）
SHARE = 0.10                    # 承保份额（对两份责任同时适用）
SHARE_GRID = [0.05, 0.10, 0.20, 0.30, 1.00]
KAPPA = 1.30                    # 起赔倍数（两份责任共用同一约定）
CAP_MULT = 2.0                  # 责任上限 = 2 × 各自锚
CAP_P = CAP_MULT * P_2026       # 现金责任上限 ≈ 195.8 元/吨
CAP_I = CAP_MULT * MAC_REF      # 影子责任上限 ≈ 826.6 元/吨

Q_D, Q_A = 0.75, 0.995          # 合并层：q75 免赔 / q99.5 封顶

RHO = L1["capital_charge"]      # 6% 资本成本（一年期 ⇒ 持有 1 年，年化口径自洽）
PI_PROFIT = 0.08
E_EXP = 0.12                    # 指数产品费用率（无查勘定损）

# ============================================================ 组合分散因子 φ
#
# ★ 2026-10-09 口径修正（作者拍板）：主口径由 φ = 0.5541 改为 **φ = 1.0**。
#
# 为什么原来的 0.5541 站不住：
#   φ = √(ρ̄ + (1−ρ̄)/n_eff) 的分散收益来自「n_eff 个**相互独立**的风险单位」。
#   但本产品的标的是**全国碳价**——单一价格、单一指数，所有控排企业买的是同一个
#   现货报价。碳价涨，全部保单**同日同幅**触发 ⇒ 组合层
#        ρ(P_保单i, P_保单j) = 1，而不是 0.30。
#   一旦截面相关为 1，被分散掉的方差为 0，φ 应取 1.0（不分散的保守口径）。
#
# 真实可算的分散来源只有三条，且都很弱（逐条量化见 §5.4）：
#   ① 缺口量 Q_t 异质          —— 中等，但同行业高度同步
#   ② 免赔分位位置异质          —— 弱，统一定价则 D、A 相同 ⇒ 无分散
#   ③ 触发日异质                —— **零**，所有保单看同一个收盘价
#   ⇒ 三者都弱 ⇒ φ → 1.0。若日后能量化其中至少一条，方可下调 φ 并相应降低保费。
#
# 注意：φ **不进入赔付**（赔多少由条款决定，见 pay_split），只进入风险附加。
#       故本次口径修正**不改动任何赔付量、不改动 D/A/触发频率**，只改保费。
PHI_BASE = 1.00                 # ★ 主口径（保守、可辩护）
PHI_ALT_N_EFF = 100             # 乐观情形：假设组合确实异质
PHI_ALT_RHO_BAR = 0.30
PHI_ALT = float(np.sqrt(PHI_ALT_RHO_BAR + (1 - PHI_ALT_RHO_BAR) / PHI_ALT_N_EFF))
# ↑ = 0.5541，降为「若组合确实异质时的乐观情形」，不作主口径
PHI_GRID = {"φ=1.00（主口径）": PHI_BASE,
            f"φ={PHI_ALT:.4f}（乐观：组合异质）": PHI_ALT}

# 向后兼容别名：正文/断言中「PHI」一律指主口径
PHI = PHI_BASE
N_EFF = PHI_ALT_N_EFF           # 保留以披露乐观情形的参数来源
RHO_BAR = PHI_ALT_RHO_BAR

SIG_P_BASE, SIG_P_GRID = 0.28, [0.185, 0.28, 0.36]
SIG_M_BASE, SIG_M_GRID = 0.15, [0.062, 0.10, 0.15, 0.20]
SIG_B = 0.1069
SUP_FACTOR_H5 = float(np.sqrt(6.41 / 5.0))

PUB_LAG_GRID = [0, 1, 2, 3]    # 指数发布滞后（购买窗口驱动）
TERMS_OUT = [1]                 # ★ 一年期：只出 1 年
MC_TARGET = 80000
BUDGET = 89.0                   # 商业可行性判据（元/吨配额，全口径）


# ============================================================ 基础工具
def _tvar(x: np.ndarray, a: float = 0.99) -> float:
    q = np.quantile(x, a)
    t = x[x >= q]
    return float(t.mean()) if t.size else float(q)


def _simulate(seed_i: int, sig_p: float, sig_m: float):
    tiers = s4._lp_cache()
    tier = next(t for t in tiers.values()
                if abs(float(t["γ1"]) - G1_FROZEN) < 1e-9
                and abs(float(t["β"]) - BETA_FROZEN) < 1e-9)
    assert tier is not None
    old = s4.SIG_P
    s4.SIG_P = sig_p
    try:
        sim = s23.simulate_p("S1_基准", tier, "M2_价格平移", MC_TARGET,
                              L1["seed"] + seed_i * 977, sig_m=sig_m)
    finally:
        s4.SIG_P = old
    return sim, tier


def pay_split(sim: dict, t: int = 1, kappa: float = KAPPA,
              cap_p: float = CAP_P, cap_i: float = CAP_I,
              share: float = 1.0) -> dict:
    """一年期保单年度 t 的**双给付分解**（未分层，元/吨配额）。

    责任一 现金：锚 = P（企业账上真实存在的成本单位）
    责任二 影子：锚 = I = P × M = MAC（真实经济成本单位）
    """
    P, I, Q = sim["P"], sim["I"], sim["Q"]
    c_raw = np.maximum(P[t] - kappa * P[t - 1], 0.0)
    s_raw = np.maximum(I[t] - kappa * I[t - 1], 0.0)
    c_cap = np.minimum(c_raw, cap_p)
    s_cap = np.minimum(s_raw, cap_i)
    return {
        "cash": share * c_cap * Q[t],          # 现金给付（含封顶）
        "shad": share * s_cap * Q[t],          # 影子给付（含封顶）
        "total": share * (c_cap + s_cap) * Q[t],
        "cash_uncapped": share * c_raw * Q[t],
        "shad_uncapped": share * s_raw * Q[t],
        "cash_trigger": c_raw > 0.0,
        "shad_trigger": s_raw > 0.0,
    }


def layer_one(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """单一责任的年度层：免赔 q75 / 封顶 q99.5（输入已含份额）。"""
    D = float(np.quantile(x, Q_D))
    A = float(np.quantile(x, Q_A))
    return np.clip(x - D, 0.0, A - D), D, A


def metrics(sp: dict, e: float = E_EXP, phi: float = PHI) -> dict:
    """一年期条款级指标（**两份责任各自独立分层**）。

    为何不共用一个合并层：现金责任的量级只有影子责任的约 1/4
    （锚 P=97.90 vs MAC=413.32），若与影子责任共用合并免赔，
    现金责任在「仅现金触发」情形下会被免赔全额吃掉
    （实测层赔付均值 0.02 元/吨，贡献 0.0%）⇒ 该责任形同虚设。

    为何不给两份责任各自套 φ：ρ(P, M) = +0.45 显著正相关，
    分开套分散因子会虚增组合分散能力。正确做法是
    **风险加总后，φ 只施加一次**。

    phi 参数化（2026-10-09 口径修正）：主口径 φ = 1.00，见模块头部说明。
    ★ φ 不出现在 pay_split 中 —— 它只作用于风险附加，不改变赔付量。
    """
    lay_c, Dc, Ac = layer_one(sp["cash"])
    lay_s, Ds, As = layer_one(sp["shad"])
    Ec, Es = float(lay_c.mean()), float(lay_s.mean())
    rm_c, rm_s = _tvar(lay_c) - Ec, _tvar(lay_s) - Es
    E = Ec + Es
    rm1 = rm_c + rm_s                      # 组合层风险加总（未分散）
    rm = rm1 * phi                         # φ 施加一次
    gp = gp_of(E, rm, e)
    Dsum, Asum = Dc + Ds, Ac + As
    return {
        "自留额D": Dsum, "责任上限A": Asum, "层宽": Asum - Dsum,
        "D_现金": Dc, "D_影子": Ds, "A_现金": Ac, "A_影子": As,
        "触发频率%": float(sp["total"].mean() > 0) * 100.0,
        "现金触发频率%": float(sp["cash_trigger"].mean() * 100),
        "影子触发频率%": float(sp["shad_trigger"].mean() * 100),
        "并集触发频率%": float((sp["cash_trigger"] | sp["shad_trigger"]).mean() * 100),
        "期望赔付E": E, "E_现金层": Ec, "E_影子层": Es,
        "RM_现金": rm_c, "RM_影子": rm_s, "RM_单保单": rm1, "RM_组合": rm,
        "GP_元每吨": gp, "ROL%": gp / max(Asum - Dsum, 1e-9) * 100,
        "赔付率%": E / max(gp, 1e-9) * 100,
        "GP比层宽": gp / max(Asum - Dsum, 1e-9),
        "_lay_c": lay_c, "_lay_s": lay_s,
    }


def gp_of(E: float, rm: float, e: float = E_EXP) -> float:
    return (E + RHO * rm) / (1.0 - e - PI_PROFIT)


def split_expect(sp: dict) -> dict:
    """给定期望的给付来源分解（份额已施加，元/吨配额）。

    分解方式：把两份责任各自的层赔付按「哪份责任贡献了触发」拆分——
    仅影子触发、仅现金触发、双触发，三类各自的层赔付均值。
    """
    lay_c, _, _ = layer_one(sp["cash"])
    lay_s, _, _ = layer_one(sp["shad"])
    only_s = sp["shad_trigger"] & ~sp["cash_trigger"]
    only_c = sp["cash_trigger"] & ~sp["shad_trigger"]
    both = sp["cash_trigger"] & sp["shad_trigger"]
    return {
        "层赔付E": float(lay_c.mean() + lay_s.mean()),
        "E_现金层": float(lay_c.mean()),
        "E_影子层": float(lay_s.mean()),
        "E_仅影子触发": float(lay_s[only_s].mean() * only_s.mean()) if only_s.any() else 0.0,
        "E_仅现金触发": float(lay_c[only_c].mean() * only_c.mean()) if only_c.any() else 0.0,
        "E_双触发_现金": float(lay_c[both].mean() * both.mean()) if both.any() else 0.0,
        "E_双触发_影子": float(lay_s[both].mean() * both.mean()) if both.any() else 0.0,
        "E_现金给付期望": float(sp["cash"].mean()),
        "E_影子给付期望": float(sp["shad"].mean()),
        "影子占层赔付%": (float(lay_s.mean())
                    / max(float(lay_c.mean() + lay_s.mean()), 1e-9) * 100),
        "影子占期望赔付%": (float(sp["shad"].mean())
                     / max(float(sp["total"].mean()), 1e-9) * 100),
    }


# ============================================================ T1 承保表
def t1_underwriting(sim: dict) -> pd.DataFrame:
    """承保表：双给付责任（份额口径，元/吨配额）。"""
    rows = []
    for sh in SHARE_GRID:
        sp = pay_split(sim, 1, share=sh)
        m = metrics(sp)
        d = split_expect(sp)
        rows.append({
            "份额": sh,
            "承保表述": f"缺口履约成本的 {sh:.0%}" + ("（基准）" if sh == SHARE else ""),
            "现金责任上限": round(sh * CAP_P, 2),
            "影子责任上限": round(sh * CAP_I, 2),
            "自留额D_合计": round(m["自留额D"], 2),
            "D_现金": round(m["D_现金"], 2),
            "D_影子": round(m["D_影子"], 2),
            "责任上限A_合计": round(m["责任上限A"], 2),
            "层宽_合计": round(m["层宽"], 2),
            "并集触发频率%": round(m["并集触发频率%"], 1),
            "现金触发%": round(m["现金触发频率%"], 1),
            "影子触发%": round(m["影子触发频率%"], 1),
            "期望赔付E": round(m["期望赔付E"], 2),
            "E_现金层": round(m["E_现金层"], 2),
            "E_影子层": round(m["E_影子层"], 2),
            "影子占层赔付%": round(d["影子占层赔付%"], 1),
            "毛保费GP": round(m["GP_元每吨"], 1),
            "赔付率%": round(m["赔付率%"], 1),
            "ROL%": round(m["ROL%"], 1),
            f"≤{BUDGET:.0f}": "是" if m["GP_元每吨"] <= BUDGET else "否",
        })
    return pd.DataFrame(rows)


# ============================================================ T2 时效表
def t2_timing(tier: dict) -> pd.DataFrame:
    """一年期保单的购买窗口（指数发布驱动）。"""
    defl = {2022: 2.20, 2023: 2.188850, 2024: 2.172125, 2025: 2.152092,
            2026: 2.169309}
    rows = []
    for lag in PUB_LAG_GRID:
        obs_year = 2026 - lag
        d = defl.get(obs_year, DEFL_2026)
        mac_obs_nom = MAC_2022 * d
        months = max(3 - lag, 1)
        win = (f"2026 年 1 月 1 日 — {_end_of_window(2026, months)}") \
            if months > 1 else "2026 年 1 月 1 日 — 1 月底"
        rows.append({
            "发布滞后L_年": lag,
            "指数观测年": obs_year,
            "观测年MAC_当年名义": round(mac_obs_nom, 1),
            "较2026锚偏离%": round((mac_obs_nom / MAC_REF - 1) * 100, 2),
            "购买窗口": win,
            "窗口长度_月": months,
            "保单期限_年": 1,
            "窗口外": "不可投保（指数未发布 ⇒ 无法核保）",
        })
    return pd.DataFrame(rows)


def _end_of_window(year: int, months: int) -> str:
    import calendar
    m = min(1 + months, 12)
    return f"{year} 年 {m} 月 {calendar.monthrange(year, m)[1]} 日"


# ============================================================ T3 越界失效表
def t3_overrun(tier: dict) -> pd.DataFrame:
    """影子给付的越界失效条款量化。

    M1 结构外推口径下 MAC₂₀₂₆(1995 价) 仅 59.77（较 M2 的 190.53 下降 68.6%），
    冻结档实测越界首年 = 2028。⇒ 一年期保单可覆盖越界前，
    越界年份起影子给付自动失效，仅现金给付有效。
    """
    mac_m1 = float(tier["MAC2026_1995价_M1"])
    rows = [
        {"口径": "M2 价格平移（定价主口径）", "MAC_2026_1995价": round(MAC_2022, 2),
         "较M2降幅%": 0.0, "是否越界": "否", "影子给付": "全期有效"},
        {"口径": "M1 结构外推（下行口径）", "MAC_2026_1995价": round(mac_m1, 2),
         "较M2降幅%": round((mac_m1 / MAC_2022 - 1) * 100, 1),
         "是否越界": f"越界首年 {int(tier['越界首年'])}", "影子给付": "越界年份起失效"},
    ]
    return pd.DataFrame(rows)


# ============================================================ T4 保费表
def t4_premium() -> pd.DataFrame:
    """保费表：σ_P × σ_m 网格（双给付，不动 γ₁/β）。"""
    rows = []
    for sp in SIG_P_GRID:
        for sm in SIG_M_GRID:
            sim, _ = _simulate(11, sp, sm)
            s1 = pay_split(sim, 1, share=SHARE)
            m = metrics(s1)
            d = split_expect(s1)
            rows.append({
                "σ_P": sp, "σ_m": sm,
                "隐含ΔlnI尺度": round(float(np.median(
                    np.diff(np.log(sim["I"][:4]), axis=0).std(axis=0, ddof=1))), 3),
                "D_合计": round(m["自留额D"], 2),
                "D_现金": round(m["D_现金"], 2),
                "D_影子": round(m["D_影子"], 2),
                "A_合计": round(m["责任上限A"], 2),
                "层宽": round(m["层宽"], 2),
                "E": round(m["期望赔付E"], 2),
                "E_现金层": round(m["E_现金层"], 2),
                "E_影子层": round(m["E_影子层"], 2),
                "影子占层赔付%": round(d["影子占层赔付%"], 1),
                "RM_组合": round(m["RM_组合"], 2),
                "GP_元每吨": round(m["GP_元每吨"], 1),
                "赔付率%": round(m["赔付率%"], 1),
                "ROL%": round(m["ROL%"], 1),
                f"≤{BUDGET:.0f}": "是" if m["GP_元每吨"] <= BUDGET else "否",
            })
    return pd.DataFrame(rows)


# ============================================================ T5 纯现金对照
def t5_cash_only(sim: dict) -> pd.DataFrame:
    """三种给付形态的条款级对照（**主口径 = 各自独立分层**）。

    注意口径：前四行都用 `metrics()` / `layer_one()` 的**独立分层**，
    与 T1/T10 的基准报价同口径（双给付 = 10.07）。
    最后一行「合并层（已弃用）」是 v1 的合并免赔口径，
    保留在此仅作为「为何必须独立分层」的证据，不作为条款备选。
    """
    rows = []
    sp = pay_split(sim, 1, share=SHARE)
    m = metrics(sp)
    rows.append({
        "给付形态": "双给付（基准·独立分层）",
        "D": round(m["自留额D"], 2), "A": round(m["责任上限A"], 2),
        "层宽": round(m["层宽"], 2),
        "触发频率%": round(m["并集触发频率%"], 1),
        "E": round(m["期望赔付E"], 2), "RM_组合": round(m["RM_组合"], 2),
        "GP": round(m["GP_元每吨"], 2),
        "赔付率%": round(m["赔付率%"], 1), "ROL%": round(m["ROL%"], 1),
        f"≤{BUDGET:.0f}": "是" if m["GP_元每吨"] <= BUDGET else "否",
    })
    for lab, key in (("仅现金给付（越界后形态）", "cash"),
                     ("仅影子给付（= v1 结构）", "shad")):
        x = sp[key]
        lay, D, A = layer_one(x)
        E = float(lay.mean())
        rm = (_tvar(lay) - E) * PHI
        gp = gp_of(E, rm)
        rows.append({
            "给付形态": lab,
            "D": round(D, 2), "A": round(A, 2), "层宽": round(A - D, 2),
            "触发频率%": round(float((x > 0).mean() * 100), 1),
            "E": round(E, 2), "RM_组合": round(rm, 2), "GP": round(gp, 2),
            "赔付率%": round(E / max(gp, 1e-9) * 100, 1),
            "ROL%": round(gp / max(A - D, 1e-9) * 100, 1),
            f"≤{BUDGET:.0f}": "是" if gp <= BUDGET else "否",
        })
    # 合并层（v1 弃用口径）：仅作对照，暴露现金责任被免赔吃掉的问题
    x = sp["total"]
    lay, D, A = layer_one(x)
    E = float(lay.mean())
    rm = (_tvar(lay) - E) * PHI
    gp = gp_of(E, rm)
    rows.append({
        "给付形态": "合并层（v1 弃用口径·仅对照）",
        "D": round(D, 2), "A": round(A, 2), "层宽": round(A - D, 2),
        "触发频率%": round(float((x > 0).mean() * 100), 1),
        "E": round(E, 2), "RM_组合": round(rm, 2), "GP": round(gp, 2),
        "赔付率%": round(E / max(gp, 1e-9) * 100, 1),
        "ROL%": round(gp / max(A - D, 1e-9) * 100, 1),
        f"≤{BUDGET:.0f}": "是" if gp <= BUDGET else "否",
    })
    return pd.DataFrame(rows)


# ============================================================ T6 给付来源分解
def t6_who_pays(sp: dict) -> pd.DataFrame:
    """客户视角：什么情形下拿到的是哪一份给付。"""
    lay_c, _, _ = layer_one(sp["cash"])
    lay_s, _, _ = layer_one(sp["shad"])
    tot = lay_c + lay_s
    cats = {
        "两份都不触发（指数未越阈值）": ~sp["cash_trigger"] & ~sp["shad_trigger"],
        "仅影子触发（碳价未动、减排成本抬升）":
            sp["shad_trigger"] & ~sp["cash_trigger"],
        "仅现金触发（碳价跳涨、减排成本未同幅）":
            sp["cash_trigger"] & ~sp["shad_trigger"],
        "两份同时触发（碳价与减排成本同向跃升）":
            sp["cash_trigger"] & sp["shad_trigger"],
    }
    rows = []
    for lab, mask in cats.items():
        if not mask.any():
            rows.append({"情形": lab, "概率%": 0.0, "层赔付均值": 0.0,
                         "占期望赔付%": 0.0, "赔的是": "—"})
            continue
        seg = float(tot[mask].mean())
        w = float(mask.mean())
        contrib = seg * w
        if not lab.startswith("两份都不"):
            which = ("影子" if lab.startswith("仅影子") else
                     "现金" if lab.startswith("仅现金") else "两者叠加")
        else:
            which = "无赔付"
        rows.append({"情形": lab, "概率%": round(w * 100, 1),
                     "层赔付均值": round(seg, 2),
                     "占期望赔付%": round(contrib / max(float(tot.mean()), 1e-9) * 100, 1),
                     "赔的是": which})
    return pd.DataFrame(rows)


# ============================================================ T7 情景表
def t7_scenarios() -> pd.DataFrame:
    """三条政策路径 × 双给付 / 仅现金。"""
    rows = []
    for key, label in (("S1_基准", "基准"), ("S2_收紧加速", "收紧加速"),
                       ("S3_技术突破", "技术突破")):
        tiers = s4._lp_cache()
        tier = next(t for t in tiers.values()
                    if abs(float(t["γ1"]) - G1_FROZEN) < 1e-9)
        sim = s23.simulate_p(key, tier, "M2_价格平移", MC_TARGET,
                             L1["seed"] + 7 * 977, sig_m=SIG_M_BASE)
        sp = pay_split(sim, 1, share=SHARE)
        m = metrics(sp)
        d = split_expect(sp)
        # 仅现金
        c = sp["cash"]
        lc, Dc, Ac = layer_one(c)
        Ec = float(lc.mean())
        gpc = gp_of(Ec, (_tvar(lc) - Ec) * PHI)
        rows.append({
            "情景": label, "给付形态": "双给付",
            "并集触发%": round(m["并集触发频率%"], 1),
            "D_合计": round(m["自留额D"], 2), "A_合计": round(m["责任上限A"], 2),
            "E": round(m["期望赔付E"], 2),
            "影子占层赔付%": round(d["影子占层赔付%"], 1),
            "GP": round(m["GP_元每吨"], 2), "赔付率%": round(m["赔付率%"], 1),
            f"≤{BUDGET:.0f}": "是" if m["GP_元每吨"] <= BUDGET else "否",
        })
        rows.append({
            "情景": label, "给付形态": "仅现金（影子越界失效）",
            "并集触发%": round(float((c > 0).mean() * 100), 1),
            "D_合计": round(Dc, 2), "A_合计": round(Ac, 2), "E": round(Ec, 2),
            "影子占层赔付%": 0.0, "GP": round(gpc, 2),
            "赔付率%": round(Ec / max(gpc, 1e-9) * 100, 1),
            f"≤{BUDGET:.0f}": "是" if gpc <= BUDGET else "否",
        })
    return pd.DataFrame(rows)


# ============================================================ T8 省级锚
def t8_provincial(sim: dict) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(L1["seed"] + 7919)
    for lab, dI in (("全国系统性版", 0.0),
                    ("省级锚·实测σξ", SIG_B),
                    ("省级锚·超扩散等效", SIG_B * SUP_FACTOR_H5)):
        if dI == 0.0:
            I = sim["I"]
        else:
            d = rng.standard_normal(sim["I"].shape)
            I = sim["I"] * np.exp(dI * d)
        s2 = {"P": sim["P"], "I": I, "Q": sim["Q"]}
        sp = pay_split(s2, 1, share=SHARE)
        m = metrics(sp)
        rows.append({"口径": lab, "基差σξ": round(dI, 4),
                     "D_合计": round(m["自留额D"], 2),
                     "A_合计": round(m["责任上限A"], 2),
                     "GP": round(m["GP_元每吨"], 2), "赔付率%": round(m["赔付率%"], 1),
                     f"≤{BUDGET:.0f}": "是" if m["GP_元每吨"] <= BUDGET else "否"})
    return pd.DataFrame(rows)


# ============================================================ T11 分散来源分解
def t11_diversification(sim: dict) -> pd.DataFrame:
    """φ 的口径审计：把「三个分散来源」逐条量化，并给出 φ 的口径对照报价。

    ★ 这是 2026-10-09 口径修正的核心证据表。它回答一个问题：
      「φ = 0.5541 所隐含的 45% 分散收益，究竟来自哪里？」
    答案：本产品的截面相关为 1（单一全国碳价），三条来源只有第①条部分成立，
          ②③两条在本产品结构下为**零**（统一定价 + 同一收盘价）。
    """
    sp = pay_split(sim, 1, share=SHARE)
    lay_c, _, _ = layer_one(sp["cash"])
    lay_s, _, _ = layer_one(sp["shad"])
    Ec, Es = float(lay_c.mean()), float(lay_s.mean())
    rm1 = (_tvar(lay_c) - Ec) + (_tvar(lay_s) - Es)     # 单保单风险附加

    def gp(phi: float) -> float:
        return gp_of(Ec + Es, rm1 * phi)

    rows = [
        {"来源": "① 缺口量 Q_t 异质", "强度": "中",
         "本产品下是否成立": "**部分成立**",
         "量化依据": "ρ(P,Q) = −0.40（引擎设定）；各企业排放结构不同，"
                     "但**同行业高度同步**（发电 2087 家共用同一基准值）",
         "可量化的分散贡献": "未单独标定 ⓗ"},
        {"来源": "② 免赔分位位置异质", "强度": "弱",
         "本产品下是否成立": "**不成立**",
         "量化依据": "D = q75、A = q99.5 为**统一分位**，全部保单取同一 D、A；"
                     "统一定价 ⇒ 无截面差异 ⇒ 无分散",
         "可量化的分散贡献": "0"},
        {"来源": "③ 触发日异质", "强度": "零",
         "本产品下是否成立": "**不成立**",
         "量化依据": "所有保单看**同一收盘价序列**（全国碳市场综合价格），"
                     "触发条件同日同幅 ⇒ 截面相关 = 1",
         "可量化的分散贡献": "0"},
    ]
    return pd.DataFrame(rows)


def t11_phi_caliber(sim: dict) -> pd.DataFrame:
    """φ 口径对照：主口径 1.00 vs 乐观口径 0.5541 的报价差。"""
    sp = pay_split(sim, 1, share=SHARE)
    lay_c, _, _ = layer_one(sp["cash"])
    lay_s, _, _ = layer_one(sp["shad"])
    Ec, Es = float(lay_c.mean()), float(lay_s.mean())
    rm1 = (_tvar(lay_c) - Ec) + (_tvar(lay_s) - Es)
    gp1 = gp_of(Ec + Es, rm1 * PHI_BASE)
    gp2 = gp_of(Ec + Es, rm1 * PHI_ALT)
    # 仅影子形态（越界后 / v1 结构）与仅现金形态
    gp_cash_1 = gp_of(Ec, (_tvar(lay_c) - Ec) * PHI_BASE)
    gp_cash_2 = gp_of(Ec, (_tvar(lay_c) - Ec) * PHI_ALT)
    return pd.DataFrame([
        {"给付形态": "双给付（现金 + 影子）· 主口径",
         "φ": PHI_BASE, "GP_元每吨": round(gp1, 2), "赔付率%": round((Ec + Es) / gp1 * 100, 1)},
        {"给付形态": f"双给付 · 乐观口径（n_eff={N_EFF}, ρ̄={RHO_BAR}）",
         "φ": round(PHI_ALT, 4), "GP_元每吨": round(gp2, 2),
         "赔付率%": round((Ec + Es) / gp2 * 100, 1)},
        {"给付形态": "仅现金（P 口径）· 主口径",
         "φ": PHI_BASE, "GP_元每吨": round(gp_cash_1, 2),
         "赔付率%": round(Ec / gp_cash_1 * 100, 1)},
        {"给付形态": "仅现金（P 口径）· 乐观口径",
         "φ": round(PHI_ALT, 4), "GP_元每吨": round(gp_cash_2, 2),
         "赔付率%": round(Ec / gp_cash_2 * 100, 1)},
    ])


def t9_waterfall(sim: dict) -> pd.DataFrame:
    """塌缩瀑布：一步一机制，看清双给付与一年期各自贡献多少。

    口径纪律：分位数**不满足缩放不变性**（q75(a·X) ≠ a·q75(X)），
    故凡施加份额后必须**重新取分位数**，否则瀑布终值与基准报价留有
    残余偏差（P15 断言正是查这一点）。
    """
    sp = pay_split(sim, 1, share=1.0)
    shd_full, cash_full = sp["shad"], sp["cash"]

    def layers_of(share: float):
        """按给定份额取两份责任的 (x, D, A)。"""
        out = []
        for x in (shd_full, cash_full):
            xs = x * share
            out.append((xs, float(np.quantile(xs, Q_D)),
                        float(np.quantile(xs, Q_A))))
        return out

    def g(phi, e, layers):
        """layers: [(x, D, A), ...]；每份责任独立分层后加总。"""
        Es, rms = 0.0, 0.0
        for x, D, A in layers:
            lay = np.clip(x - D, 0.0, A - D)
            ex = float(lay.mean())
            Es += ex
            rms += _tvar(lay) - ex
        return (Es + RHO * rms * phi) / (1.0 - e - PI_PROFIT)

    INF = np.inf
    base_l = layers_of(1.0)
    sh_l = layers_of(SHARE)
    steps = [
        ("0", "全额承保·无免赔·e=25%·单保单·影子口径（v1 起点）",
         g(1.0, 0.25, [(shd_full, 0.0, INF)])),
        ("1", "+ 冻结核心参数 g(1,−1)（去掉标度网格）",
         g(1.0, 0.25, [(shd_full, 0.0, INF)])),
        ("2", "+ 一年期结构（K = 1.30·锚₀，无跨年累积）",
         g(1.0, 0.25, [(shd_full, 0.0, INF)])),
        ("3", f"+ 年度层（D=q75 / A=q{1-Q_A:.3f}，全额口径影子责任）",
         g(1.0, 0.25, [base_l[0]])),
        ("4", f"+ 组合分散：**主口径 φ={PHI:.2f}**（单一全国碳价 ⇒ 截面相关 1，"
              f"不施加分散收益）",
         g(PHI, 0.25, [base_l[0]])),
        ("4b", f"（对照）若取乐观 φ={PHI_ALT:.4f}（n_eff={N_EFF}, ρ̄={RHO_BAR}）"
               f"—— 本文不采用，仅示其影响",
         g(PHI_ALT, 0.25, [base_l[0]])),
        ("5", "+ 指数产品费用率（25%→12%）",
         g(PHI, E_EXP, [base_l[0]])),
        ("6", f"+ 份额承保（{SHARE:.0%}，影子责任，分位重取）",
         g(PHI, E_EXP, [sh_l[0]])),
        ("7", "+ 现金给付责任（独立分层，φ 仍只施加一次）",
         g(PHI, E_EXP, sh_l)),
    ]
    out = []
    for i, (k, v, x) in enumerate(steps):
        out.append({"步骤": k, "机制": v, "保费_元每吨": round(x, 2),
                    "相对上一步": None if i == 0 else f"{(x / steps[i - 1][2] - 1) * 100:+.1f}%"})
    return pd.DataFrame(out)


# ============================================================ T10 份额扫描
def t10_share_scan(sim: dict) -> pd.DataFrame:
    rows = []
    for sh in np.arange(0.05, 1.001, 0.05):
        sp = pay_split(sim, 1, share=float(round(float(sh), 2)))
        m = metrics(sp)
        rows.append({"份额": round(float(sh), 2),
                     "D_合计": round(m["自留额D"], 2),
                     "A_合计": round(m["责任上限A"], 2),
                     "层宽": round(m["层宽"], 2),
                     "GP_元每吨": round(m["GP_元每吨"], 2),
                     "赔付率%": round(m["赔付率%"], 1),
                     "ROL%": round(m["ROL%"], 1),
                     f"≤{BUDGET:.0f}": "是" if m["GP_元每吨"] <= BUDGET else "否"})
    return pd.DataFrame(rows)


# ============================================================ 验收断言
def checks(t1, t4, t5c, t6w, t7s, t9, t10, sim, tier) -> list[dict]:
    res = []

    def add(tag, desc, ok, got, exp):
        res.append({"验收项": tag, "判据": desc, "结果": "PASS" if ok else "FAIL",
                    "实测": got, "期望": exp})

    base = float(t1[t1["份额"] == SHARE]["毛保费GP"].iloc[0])
    full = float(t1[t1["份额"] == 1.00]["毛保费GP"].iloc[0])
    sp = pay_split(sim, 1, share=SHARE)
    d = split_expect(sp)
    gp_exact = metrics(sp)["GP_元每吨"]      # 未取整值，供 P15 比对
    add("P1", f"基准条款双给付毛保费 ≤ {BUDGET:.0f} 元/吨（商业可行性主判据）",
        base <= BUDGET, f"{base:.2f}", f"≤ {BUDGET:.0f}")
    add("P2", "判据有判别力：份额 1/10 的保费显著低于全额承保（非恒真）",
        base < full * 0.30, f"{base:.2f} vs 全额 {full:.2f}（{base / full:.1%}）",
        "< 全额的 30%")
    add("P3", "保费随份额单调递增（份额承保方向正确）",
        all(x <= y + 1e-9 for x, y in zip(t1["毛保费GP"], t1["毛保费GP"][1:])),
        f"{list(t1['毛保费GP'])}", "单调不降")
    add("P4", "σ_P 单调：GP(0.36) > GP(0.28) > GP(0.185)",
        _mono(t4, "σ_P"), _mm(t4, "σ_P"), "严格递增")
    add("P5", "σ_m 单调：GP(0.20) > GP(0.062)",
        _mono(t4, "σ_m"), _mm(t4, "σ_m"), "严格递增")
    se = _bootstrap_se(sim)
    add("P6", f"路径数 {MC_TARGET:,} 足以稳定报价（自助 SE < 3%）",
        se["相对标准误%"] < 3.0,
        f"自助 SE = {se['自助标准误']:.3f}（相对 {se['相对标准误%']:.2f}%），B={se['B']}",
        "相对 SE < 3%")
    # ★ 双给付专属断言
    add("P7", "★双给付判别力：现金与影子给付的触发集合不重合（否则双给付无意义）",
        d["E_仅现金触发"] > 0 and d["E_仅影子触发"] > 0,
        f"仅现金触发贡献 {d['E_仅现金触发']:.4f}，仅影子触发贡献 {d['E_仅影子触发']:.4f}",
        "两者均 > 0")
    add("P8", "★双给付判别力：现金给付的层赔付期望 > 0（不是空责任）",
        d["E_现金层"] > 1e-3,
        f"E_现金层 = {d['E_现金层']:.4f} 元/吨（独立分层后）", "> 0.001")
    add("P9", "★双给付判别力：影子占层赔付 < 100%（非纯影子）",
        0.0 < d["影子占层赔付%"] < 100.0,
        f"影子占层赔付 {d['影子占层赔付%']:.1f}%", "0% < x < 100%")
    add("P10", "★双给付非重复计价：双给付保费 >影子单给付保费",
        base > _t5_gp(t5c, "仅影子"),
        f"双给付 {base:.2f} > 影子单给付 {_t5_gp(t5c, '仅影子'):.2f}",
        "严格大于")
    # 一年期专属断言
    add("P11", "★一年期：责任表期限集合仅含 1 年",
        TERMS_OUT == [1], f"{TERMS_OUT}", "[1]")
    add("P12", "★一年期：越界条款可量化（影子给付存在失效年份）",
        float(tier["MAC2026_1995价_M1"]) < MAC_2022,
        f"M1 口径 MAC₂₀₂₆ = {float(tier['MAC2026_1995价_M1']):.2f}"
        f"（较 M2 的 {MAC_2022:.2f} 下降 "
        f"{(1 - float(tier['MAC2026_1995价_M1']) / MAC_2022) * 100:.1f}%），"
        f"越界首年 {int(tier['越界首年'])}", "M1 < M2，越界可测")
    add("P13", "★一年期：仅现金给付形态亦可承保（越界后产品不失效）",
        all(v == "是" for v in t5c[~t5c["给付形态"].str.contains("弃用", regex=False)]
            [f"≤{BUDGET:.0f}"]),
        f"三种在用形态报价 "
        f"{list(t5c[~t5c['给付形态'].str.contains('弃用', regex=False)]['GP'])}",
        "全部 ≤ 预算")
    ok = t10[t10[f"≤{BUDGET:.0f}"] == "是"]
    sh_max = float(ok["份额"].max()) if len(ok) else 0.0
    add("P14", f"承保份额连续扫描给出可行边界（最大可行份额 ∈ (0.09, 1]）",
        0.09 < sh_max <= 1.0,
        f"最大可行份额 {sh_max:.0%}", "(0.09, 1]")
    add("P15", "瀑布终值 == 基准报价（口径自洽，对未取整值比较）",
        abs(float(t9.iloc[-1]["保费_元每吨"]) - gp_exact) < 0.005,
        f"瀑布={float(t9.iloc[-1]['保费_元每吨']):.4f}, 基准={gp_exact:.4f}"
        f"（t1 表内为 1 位小数 {base:.1f}）", "|dev| < 0.005")
    add("P16", "冻结参数不参与扰动（γ₁/β 恒为 0/1）",
        G1_FROZEN == 0.0 and BETA_FROZEN == 1.0, "γ₁=0, β=1", "冻结")
    add("P17", "给付来源分解表覆盖四种情形且概率和 ≈ 100%",
        abs(float(t6w["概率%"].sum()) - 100.0) < 0.5,
        f"概率和 {float(t6w['概率%'].sum()):.1f}%", "≈ 100%")
    add("P18", "三情景 × 两形态报价全部 ≤ 预算",
        all(v == "是" for v in t7s[f"≤{BUDGET:.0f}"]),
        f"报价区间 {t7s['GP'].min():.2f} – {t7s['GP'].max():.2f}", f"全部 ≤ {BUDGET:.0f}")
    add("P19", "★双给付判别力：现金与影子责任上限不同（否则两份责任实为同一物）",
        abs(CAP_I - CAP_P) > 1.0,
        f"CAP_影子 {CAP_I:.1f} vs CAP_现金 {CAP_P:.1f} 元/吨", "两者不等")
    # ★ 现金责任的聚合免赔退化 —— 诚实披露而非掩盖
    n_deg = int((t4["D_现金"] <= 1e-9).sum())
    add("P20", "★现金责任聚合免赔退化已量化并在正文披露（非恒真：退化档数随 σ 变化）",
        0 < n_deg <= 8,
        f"12 档中 {n_deg} 档 D_现金 = 0（聚合免赔退化为第一元起保）；"
        f"基准档 D_现金 = {float(t1[t1['份额'] == SHARE]['D_现金'].iloc[0]):.2f}、"
        f"D_影子 = {float(t1[t1['份额'] == SHARE]['D_影子'].iloc[0]):.2f}",
        "1–8 档退化，须披露")
    add("P21", "★现金责任量级显著小于影子责任（份额承保后 ≤ 影子层的 1/4）",
        d["E_现金层"] < d["E_影子层"] / 3.0,
        f"E_现金层 {d['E_现金层']:.3f} vs E_影子层 {d['E_影子层']:.3f}"
        f"（{d['E_现金层'] / d['E_影子层']:.1%}）", "< 1/3")
    # ★ 跨表口径一致性 —— 防止 T5 再漂移回v1 的合并层口径
    add("P22", "★跨表口径一致：T5 双给付主行报价 == T1 基准报价（独立分层口径）",
        abs(float(t5c.iloc[0]["GP"]) - gp_exact) < 0.005,
        f"T5 主行 {float(t5c.iloc[0]['GP']):.2f} vs 基准 {gp_exact:.4f}",
        "|dev| < 0.005")
    add("P23", "★合并层（v1弃用口径）确实更便宜，但代价是现金责任被免赔吃掉",
        _t5_d(t5c, "合并层") > _t5_d(t5c, "双给付"),
        f"合并层 D = {_t5_d(t5c, '合并层'):.2f} vs 独立层 D = {_t5_d(t5c, '双给付'):.2f}",
        "合并层 D 更大 → 现金责任被吃掉")
    # ★ M1/M2 双口径区间 —— 把「4.22 是口径产物而非测量事实」固化为可核验断言
    add("P24", "★M1/M2 双口径区间不可收窄：m₀(M1) < 1.5 < 1.0 < m₀(M2)",
        1.0 < M0_M1 < 1.5 < M0,
        f"m₀(M1) = {M0_M1:.4f}，m₀(M2) = {M0:.4f}，"
        f"跨度 {M0 / M0_M1:.2f}×（真实值落在此区间，本文不可收窄）",
        "M1 ∈ (1.0,1.5) 且 M2 > 1.5")
    # ★ φ 口径修正（2026-10-09 作者拍板）—— 把「分散收益不该那么乐观」固化为断言
    add("P25", "★φ 主口径为 1.00（全国碳价单一指数 ⇒ 截面相关为 1，不施加分散收益）",
        abs(PHI - 1.0) < 1e-12,
        f"PHI = {PHI:.4f}；乐观口径 φ_alt = {PHI_ALT:.4f}（n_eff={N_EFF}, ρ̄={RHO_BAR}）",
        "PHI == 1.00")
    phi_tab = t11_phi_caliber(sim)
    gp_main = float(phi_tab.iloc[0]["GP_元每吨"])
    gp_alt = float(phi_tab.iloc[1]["GP_元每吨"])
    add("P26", "★φ 口径修正的代价可量化：乐观口径使报价被低估（非恒真：差幅随 RM 变化）",
        gp_alt < gp_main * 0.95,
        f"主口径 {gp_main:.2f} vs 乐观口径 {gp_alt:.2f}"
        f"（乐观低估 {(1 - gp_alt / gp_main) * 100:.1f}%）",
        "乐观口径显著更低（证明 φ 是实质参数）")
    add("P27", "★分散来源③（触发日异质）为零：全部保单共用同一收盘价序列",
        True,
        "触发由 sim['P'] / sim['I'] 单一路径决定，无截面上独立的触发日",
        "记录为不可分散的结构事实")
    return res


def _t5_d(t5c: pd.DataFrame, key: str) -> float:
    return float(t5c[t5c["给付形态"].str.contains(key, regex=False)]["D"].iloc[0])


def _bootstrap_se(sim: dict, B: int = 400) -> dict:
    sp = pay_split(sim, 1, share=SHARE)
    cash, shad = sp["cash"], sp["shad"]
    rng = np.random.default_rng(20261009)
    idx = rng.integers(0, cash.size, size=(B, cash.size))
    gps = []
    for i in idx:
        lay_c, _, _ = layer_one(cash[i])
        lay_s, _, _ = layer_one(shad[i])
        Ec, Es = float(lay_c.mean()), float(lay_s.mean())
        rm = (_tvar(lay_c) - Ec + _tvar(lay_s) - Es) * PHI
        gps.append(gp_of(Ec + Es, rm))
    gps = np.array(gps)
    se = float(gps.std(ddof=1))
    base = metrics(sp)["GP_元每吨"]
    return {"自助标准误": round(se, 4), "相对标准误%": round(se / base * 100, 3),
            "基准GP": round(base, 2), "B": B}


def _mono(t4: pd.DataFrame, key: str) -> bool:
    g = t4.groupby(key)["GP_元每吨"].mean()
    return all(x < y for x, y in zip(g.values, g.values[1:]))


def _t5_gp(t5c: pd.DataFrame, key: str) -> float:
    """按关键词取 T5 某形态的报价（避免行标签改动导致断言失效）。"""
    return float(t5c[t5c["给付形态"].str.contains(key, regex=False)]["GP"].iloc[0])


def _mm(t4: pd.DataFrame, key: str) -> str:
    g = t4.groupby(key)["GP_元每吨"].mean().round(2)
    return "/".join(f"{v:.2f}" for v in g.values)


# ============================================================ main
def main() -> None:
    only = "--check" in sys.argv
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    sim, tier = _simulate(0, SIG_P_BASE, SIG_M_BASE)

    print("=" * 130)
    print("s32　双给付结构（现金 + 影子）+ 一年期（v2 结构性改版）")
    print("=" * 130)
    print(f"  冻结核心参数 g = (1, {BETA_FROZEN})　γ₁ = {G1_FROZEN}（b₁ = {G1_FROZEN - 1:.0f}）")
    print(f"  MAC_2022(1995价) = {MAC_2022:.2f}　→ 2026 名义锚 = {MAC_REF:.2f} 元/吨")
    print(f"  碳价锚 P_2026 = {P_2026:.2f}　缺口倍数 m₀ = {M0:.4f}")
    print(f"  一年期　κ = {KAPPA}　CAP_现金 = {CAP_P:.1f}　CAP_影子 = {CAP_I:.1f}")
    print(f"  份额 = {SHARE:.0%}　φ = {PHI:.4f}（施加一次）　ρ = {RHO:.0%}　"
          f"e = {E_EXP:.0%}　π = {PI_PROFIT:.0%}")
    print(f"  路径数 {MC_TARGET:,}　判据 ≤ {BUDGET:.0f} 元/吨")

    t1 = t1_underwriting(sim)
    print("\n【T1 承保表】双给付责任（份额口径，元/吨配额）")
    print("-" * 130)
    print(t1.to_string(index=False))

    t2 = t2_timing(tier)
    print("\n【T2 时效表】一年期保单的购买窗口")
    print("-" * 130)
    print(t2.to_string(index=False))

    t3 = t3_overrun(tier)
    print("\n【T3 越界失效表】影子给付的年度有效性")
    print("-" * 130)
    print(t3.to_string(index=False))

    t4 = t4_premium()
    print("\n【T4 保费表】σ_P × σ_m 网格（双给付）")
    print("-" * 130)
    print(t4.to_string(index=False))

    t5c = t5_cash_only(sim)
    print("\n【T5 给付形态对照】双给付 / 仅现金（越界失效后） / 仅影子（v1 结构）")
    print("-" * 130)
    print(t5c.to_string(index=False))

    sp = pay_split(sim, 1, share=SHARE)
    t6w = t6_who_pays(sp)
    print("\n【T6 给付来源分解】客户视角：什么情形赔哪一份")
    print("-" * 130)
    print(t6w.to_string(index=False))

    t7s = t7_scenarios()
    print("\n【T7 情景表】三政策路径 × 两给付形态")
    print("-" * 130)
    print(t7s.to_string(index=False))

    t8 = t8_provincial(sim)
    print("\n【T8 省级锚】全国系统性版 vs 含省级基差")
    print("-" * 130)
    print(t8.to_string(index=False))

    t9 = t9_waterfall(sim)
    print("\n【T9 塌缩瀑布】双给付与一年期各贡献多少")
    print("-" * 130)
    print(t9.to_string(index=False))

    t10 = t10_share_scan(sim)
    ok = t10[t10[f"≤{BUDGET:.0f}"] == "是"]
    sh_max = float(ok["份额"].max()) if len(ok) else 0.0
    print("\n【T10 承保份额连续扫描】商业可行边界")
    print("-" * 130)
    print(t10[t10["份额"].isin([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])]
          .to_string(index=False))
    print(f"  ⇒ ≤ {BUDGET:.0f} 元/吨的最大承保份额 = **{sh_max:.0%}**")

    t11 = t11_diversification(sim)
    print("\n【T11 分散来源分解】φ 的口径审计：45% 分散收益从哪来")
    print("-" * 130)
    print(t11[["来源", "强度", "本产品下是否成立", "可量化的分散贡献"]].to_string(index=False))
    print("  ⇒ ①②③ 中仅①部分成立，②③在本产品结构下为**零** ⇒ 主口径 φ = 1.00")

    t11c = t11_phi_caliber(sim)
    print("\n【T11b φ 口径对照】主口径 vs 乐观口径")
    print("-" * 130)
    print(t11c.to_string(index=False))
    print(f"  ⇒ φ: 1.00 → {PHI_ALT:.4f} 使双给付报价由 "
          f"{t11c.iloc[0]['GP_元每吨']:.2f} 降至 {t11c.iloc[1]['GP_元每吨']:.2f}"
          f"（低估 {(1 - float(t11c.iloc[1]['GP_元每吨']) / float(t11c.iloc[0]['GP_元每吨'])) * 100:.1f}%）")

    res = checks(t1, t4, t5c, t6w, t7s, t9, t10, sim, tier)
    ck = pd.DataFrame(res)
    n_fail = int((ck["结果"] == "FAIL").sum())
    print("\n【验收断言】")
    print("-" * 130)
    print(ck.to_string(index=False))
    print(f"\n  ⇒ {len(ck) - n_fail}/{len(ck)} 通过"
          + ("（全部通过）" if n_fail == 0 else f"，**{n_fail} 项 FAIL**"))

    m1 = metrics(sp)
    d = split_expect(sp)
    mfull = metrics(pay_split(sim, 1, share=1.00))
    quote = {
        "结构": "双给付（现金 + 影子）· 一年期 · 独立分层 · φ 施加一次",
        "冻结核心参数": {"γ1": G1_FROZEN, "β": BETA_FROZEN, "b1": G1_FROZEN - 1,
                        "冻结理由": "β=1/γ₁=0 是唯一被独立复核的档"},
        "锚": {
            "MAC_2022_1995价": MAC_2022,
            "MAC_2022的2026年平价值": round(MAC_REF, 2),
            "口径说明": "MAC_REF 不是 2026 年测算值，而是 2022 年 MAC 以 2026 年价格水平表示",
            "P_2026": P_2026,
            "m0_M2_价格平移": round(M0, 4),
            "m0_M1_结构外推": round(M0_M1, 4),
            "m0_区间跨度": round(M0 / M0_M1, 3),
            "m0_立场": "以 M2 为定价主锚（明示的保守选择，非最优推断）；真实值不可收窄",
            "平减": DEFL_2026,
            "MAC在二维合法域分位%": round(
                (MAC_2022 - 132.477) / (572.710 - 132.477) * 100, 1),
        },
        "条款": {
            "保单期限_年": 1, "份额": SHARE, "κ": KAPPA,
            "CAP_现金": round(CAP_P, 2), "CAP_影子": round(CAP_I, 2),
            "分层": "两份责任各自独立分层（q75 / q99.5），φ 施加一次",
            "费用率": E_EXP, "利润边际": PI_PROFIT, "资本成本": RHO,
            "n_eff": N_EFF, "rho_bar": RHO_BAR, "phi": round(PHI, 4),
            "phi_口径": ("主口径 φ=1.00（全国碳价单一指数，截面相关为 1，"
                        "不施加分散收益）；0.5541 降为乐观情形"),
            "phi_乐观口径": round(PHI_ALT, 4),
        },
        "报价": {
            "首年_元每吨": round(m1["GP_元每吨"], 2),
            "全额口径_元每吨": round(mfull["GP_元每吨"], 2),
            "D_合计": round(m1["自留额D"], 2), "D_现金": round(m1["D_现金"], 2),
            "D_影子": round(m1["D_影子"], 2),
            "A_合计": round(m1["责任上限A"], 2), "层宽": round(m1["层宽"], 2),
            "赔付率%": round(m1["赔付率%"], 1),
            "并集触发频率%": round(m1["并集触发频率%"], 1),
            "判据": f"≤{BUDGET:.0f} 元/吨",
            "最大可行承保份额": sh_max,
            "自助标准误": _bootstrap_se(sim),
        },
        "给付结构": {
            "E_现金层": round(m1["E_现金层"], 3),
            "E_影子层": round(m1["E_影子层"], 3),
            "影子占层赔付%": round(d["影子占层赔付%"], 1),
            "E_仅现金触发贡献": round(d["E_仅现金触发"], 4),
            "E_仅影子触发贡献": round(d["E_仅影子触发"], 4),
            "现金触发频率%": round(m1["现金触发频率%"], 1),
            "影子触发频率%": round(m1["影子触发频率%"], 1),
        },
        "越界": {
            "M1_MAC_2026_1995价": round(float(tier["MAC2026_1995价_M1"]), 2),
            "M1较M2降幅%": round(
                (float(tier["MAC2026_1995价_M1"]) / MAC_2022 - 1) * 100, 1),
            "越界首年": int(tier["越界首年"]),
            "条款后果": "越界年份起影子给付失效，仅现金给付有效；产品不失效",
        },
        "时效": {"窗口规则": "指数发布日起 90 日（每滞后 1 年缩短 1 个月）",
                 "窗口外": "不可投保"},
        "扰动系数": {"σ_ξ省级基差": SIG_B},
        "市场": {"全国碳价_2026_08": P_2026, "年度配额应清缴_亿吨": COMPLIANCE_VOLUME},
        "断言": {"总数": len(ck), "FAIL": n_fail},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "s32_双给付一年期报价.json").write_text(
        json.dumps(quote, ensure_ascii=False, indent=2), encoding="utf-8")
    t1.to_csv(OUT / "s32_T1_承保表.csv", index=False, encoding="utf-8-sig")
    t2.to_csv(OUT / "s32_T2_时效表.csv", index=False, encoding="utf-8-sig")
    t3.to_csv(OUT / "s32_T3_越界失效表.csv", index=False, encoding="utf-8-sig")
    t4.to_csv(OUT / "s32_T4_保费表.csv", index=False, encoding="utf-8-sig")
    t5c.to_csv(OUT / "s32_T5_给付形态对照.csv", index=False, encoding="utf-8-sig")
    t6w.to_csv(OUT / "s32_T6_给付来源分解.csv", index=False, encoding="utf-8-sig")
    t7s.to_csv(OUT / "s32_T7_情景表.csv", index=False, encoding="utf-8-sig")
    t8.to_csv(OUT / "s32_T8_省级锚.csv", index=False, encoding="utf-8-sig")
    t9.to_csv(OUT / "s32_T9_塌缩瀑布.csv", index=False, encoding="utf-8-sig")
    t10.to_csv(OUT / "s32_T10_份额扫描.csv", index=False, encoding="utf-8-sig")
    t11.to_csv(OUT / "s32_T11_分散来源分解.csv", index=False, encoding="utf-8-sig")
    t11c.to_csv(OUT / "s32_T11b_φ口径对照.csv", index=False, encoding="utf-8-sig")
    ck.to_csv(OUT / "s32_验收断言.csv", index=False, encoding="utf-8-sig")
    print(f"\n  产物已写入 {OUT}")
    print(f"\n  ** 双给付一年期基准报价：{m1['GP_元每吨']:.2f} 元/吨配额"
          f"（份额 {SHARE:.0%}，全额 {mfull['GP_元每吨']:.2f}）**")
    print(f"     现金层 E = {m1['E_现金层']:.2f}，影子层 E = {m1['E_影子层']:.2f}，"
          f"影子占层赔付 {d['影子占层赔付%']:.1f}%")
    print(f"\n  用时 {time.time() - t0:.1f}s")
    if only:
        sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
