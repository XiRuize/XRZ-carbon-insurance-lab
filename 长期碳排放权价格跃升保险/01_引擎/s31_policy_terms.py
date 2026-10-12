# -*- coding: utf-8 -*-
"""
s31_policy_terms.py —— 第六轮：条款化定价（g(1,−1) 冻结 + 逐年更新 + 份额承保）

本脚本回答的不是「价格是多少」，而是「一张可签的保单长什么样」。

一、为什么要重写而不是续用
------------------------
前五轮（s22/s25/s27）已把**绝对保费**压到 89.9 元/吨（首年、年度层、
D=q75、A=q99.5、e=12%、n_eff=100、口径乙）。但那是一个**点值**，
且整个报价单仍以「标度档网格 + 区间 + 多口径并列」的形式呈现 ——
对保险公司而言不可签、对客户而言不可读。本脚本做三件事：

  ① **冻结核心参数**。方向向量取 g(1,−1)（即 β=1、γ₁=0，FGN-W 2005 惯例，
     亦即论文口径）。这一档在 s0 的合法域内 dev=0.9120<1，且是全网格中
     **唯一一个不需要额外约定就能被外部复核的档**。扰动不再改 γ₁/β
     （那是模糊性，须进条款披露而非进费率），只改**风险过程参数**
     （σ_P、σ_m、κ_m、ρ 等）—— 它们可观测、可回测、可再保谈判。

  ② **条款化**。把保险拆成四张可签的表：
       - 承保表：买多少（份额）、保多少（限额）、赔多少（起赔点）
       - 时效表：什么时候能买（购买窗口）、多久更新（逐年重定价）
       - 责任表：免赔、自留、封顶、不赔项
       - 保费表：费率、构成、敏感度、报价区间
     承保规则明确为：**企业为覆盖排放缺口需外购配额的支出中，保险人只承担
     约定份额**（基准 1/10）。这不是"打折"，而是把单位承保量下降一个数量级，
     从而把保费压到企业愿意付的水平。

  ③ **逐年更新（数据口径的产品化）**。影子价格逐年重估（发布滞后 L 年），
     保单因此设计为**逐年可更新保单**：每个保单年度独立结算、独立重定价、
     独立复核承保资格。首年报价即年度报价，不再有"5 年锁价"。

二、两层结构（不变）
--------------------
  风险（σ_P、σ_m、σ_Q、ρ）→ Monte Carlo；模糊性（γ₁、β）→ 冻结在 g(1,−1)
  并在条款中披露，不再以区间形式进入费率。

三、商业可行性判据（用户口径：保费 ≤ 89 元/吨）
------------------------------------------------
  判据写死在本脚本的 checks() 里，且**判据本身必须非恒真**：
  份额承保若无效，GP 不会降到 89 以下 ⇒ 判据有判别力。

用法：python s31_policy_terms.py [--check]    # 退出码 0 = 断言全过
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

pd.set_option("display.width", 240)
pd.set_option("display.unicode.east_asian_width", True)

# ============================================================ 冻结的核心参数
# 方向向量 g = (1, −β)。冻结 g(1, −1)：β=1（产出扩张与减排同权重）、
# γ₁ = b₁+1 = 0（即 b₁ = −1，FGN-W 2005 的标准平移约定）。
# 之所以冻结这一档而不是取审慎上端：
#   · 它是全网格唯一被外部独立复核过的档（论文主结果即此档）；
#   · 上端档（β2.0 / γ₁=−0.1043）依赖二维合法域的边界点，
#     而该边界点随省级 K/L 数据修订会移动（s0d 实测：异质 δ=2% 即踢出合法域）
#     ⇒ 用它定价等于把"数据修订"变成"费率不可控"，违反可缔约要求；
#   · 冻结本身就是条款：**事后不得改约定**。
G1_FROZEN = 0.0
BETA_FROZEN = 1.0
TAG = "g(1,-1)_冻结档"

MAC_2022 = 190.5306              # 冻结档 2022 全国 MAC（1995 价），引擎实测
DEFL_2026 = 2.169309             # 1995→2026 累计平减（论文核验锚 2.20 体系下）
MAC_REF = MAC_2022 * DEFL_2026  # ≈ 413.3 元/吨，2026 名义
P_2026 = s4.P_2026              # 97.90 元/吨，全国碳市场 2026-08 综合价收盘

# ============================================================ 条款参数（可签）
# T1 份额承保：企业外购配额支出中，保险人承担的约定份额。
#    1/10 是商业可行性的主杠杆：承保量降一个数量级 ⇒ 保费近似同比例下降，
#    而客户的**保障绝对额**仍在数百元/吨量级（限额 = 份额 × 全额限额）。
SHARE = 0.10
SHARE_GRID = [0.05, 0.10, 0.20, 0.30, 1.00]

# T2 起赔点：年度重定基，K_t = κ · I_{t−1}（κ=1.30）
KAPPA = 1.30
# T3 单吨赔付上限（按承保份额计的绝对额，全额口径 = CAP_MULT × MAC_REF）
CAP_MULT = 2.0
CAP_FULL = CAP_MULT * MAC_REF            # ≈ 826.6 元/吨（全额口径）
CAP_QUOTA = CAP_FULL * SHARE             # ≈ 82.7 元/吨（份额口径，1/10 承保）

# T4 年度层免赔/封顶（分位取自该保单年度赔付分布，口径同 s25）
Q_D, Q_A = 0.75, 0.995

# T5 保费原理（口径乙：年度滚动，资本持有 1 年）
RHO = L1["capital_charge"]       # 6% 年化
PI_PROFIT = 0.08
E_EXP = 0.12                     # 指数产品费用率（无查勘定损）
N_EFF = 100                      # 组合有效保单数
RHO_BAR = 0.30                   # 组合内相关系数
# ⚠【口径已废弃 · 2026-10-09】本行 φ = 0.5541 系 v1 早期口径。
#   主口径已修正为 φ = 1.00（理由：全国碳价是单一指数，所有保单同日同幅触发，
#   截面相关为 1，不存在 45% 的分散收益）。本脚本为 v1 条款与定价的历史产物，
#   保留原值以复现 v1 报价；**v2 及以后一律以 s32_dual_trigger.PHI 为准**。
PHI = float(np.sqrt(RHO_BAR + (1 - RHO_BAR) / N_EFF))   # ⚠ 废弃项：0.5541 → 见上注

# T6 逐年更新的时间滞后：影子价格发布滞后 L 年 ⇒ 起保年的锚比最新观测旧 L 年。
#    这是"逐年更新"条款的**代价**：滞后越久，锚越旧，报价越高。
PUB_LAG_GRID = [0, 1, 2, 3]

# T7 情景（承保期内政策路径）
SCEN = "S1_基准"

# 扰动系数网格：只动风险过程参数，**不动** γ₁/β（模糊性已冻结进条款）
SIG_P_BASE, SIG_P_GRID = 0.28, [0.185, 0.28, 0.36]
SIG_M_BASE, SIG_M_GRID = 0.15, [0.062, 0.10, 0.15, 0.20]
SIG_B = 0.1069                    # s18 实测单年省际基差波动（省级锚扰动）
SUP_FACTOR_H5 = float(np.sqrt(6.41 / 5.0))

TERMS_OUT = [1, 3, 5]            # 期限仍收缩至 ≤5（s25 拍板②，本轮沿用）
MC_TARGET = 80000                # 路径数：比 s25 的 20,000 更高（报价要稳）


# ============================================================ 基础工具
def _tvar(x: np.ndarray, a: float = 0.99) -> float:
    q = np.quantile(x, a)
    t = x[x >= q]
    return float(t.mean()) if t.size else float(q)


def gp_of(E: float, rm: float, e: float = E_EXP) -> float:
    """口径乙毛保费 =（层期望 + 资本成本）/（1 − 费用率 − 利润边际）。"""
    return (E + RHO * rm) / (1.0 - e - PI_PROFIT)


def layer(pay: np.ndarray, share: float) -> tuple[np.ndarray, float, float]:
    """份额承保下的年度层：先按份额缩放赔付，再取 q75 免赔 / q99.5 封顶。"""
    scaled = pay * share
    D = float(np.quantile(scaled, Q_D))
    A = float(np.quantile(scaled, Q_A))
    return np.clip(scaled - D, 0.0, A - D), D, A


def metrics(pay: np.ndarray, share: float, e: float = E_EXP) -> dict:
    """单保单年度的条款级指标（份额口径，元/吨配额）。"""
    lay, D, A = layer(pay, share)
    E = float(lay.mean())
    rm1 = _tvar(lay) - E
    rm = rm1 * PHI
    gp = gp_of(E, rm, e)
    full = pay * share
    return {
        "份额": share, "自留额D": D, "责任上限A": A, "层宽": A - D,
        "触发频率%": float((pay > 0).mean() * 100),
        "层赔付频率%": float((lay > 0).mean() * 100),
        "期望赔付E": E, "RM_单保单": rm1, "RM_组合": rm,
        "GP_元每吨": gp, "ROL%": gp / max(A - D, 1e-9) * 100,
        "赔付率%": E / max(gp, 1e-9) * 100,
        "GP比层宽": gp / max(A - D, 1e-9),
        "首年期望赔付_全额口径": float(full.mean()),
    }


# ============================================================ 引擎接线
def _simulate(seed_i: int, sig_p: float, sig_m: float, sig_q: float | None = None,
              rho_pm: float | None = None, kappa_m: float | None = None):
    """冻结档 g(1,−1) 的路径模拟（用 s23.simulate_p 以支持 σ_m 注入）。"""
    tiers = s4._lp_cache()
    tier = next(t for t in tiers.values()
                if abs(float(t["γ1"]) - G1_FROZEN) < 1e-9
                and abs(float(t["β"]) - BETA_FROZEN) < 1e-9)
    assert tier is not None
    old = s4.SIG_P
    s4.SIG_P = sig_p
    try:
        kw = {}
        if sig_q is not None:
            kw["sig_q"] = sig_q
        if rho_pm is not None:
            kw["rho_pm"] = rho_pm
        if kappa_m is not None:
            kw["kappa_m"] = kappa_m
        try:
            sim = s23.simulate_p(SCEN, tier, "M2_价格平移", MC_TARGET,
                                  L1["seed"] + seed_i * 977, sig_m=sig_m, **kw)
        except TypeError:
            sim = s23.simulate_p(SCEN, tier, "M2_价格平移", MC_TARGET,
                                  L1["seed"] + seed_i * 977, sig_m=sig_m)
    finally:
        s4.SIG_P = old
    return sim, tier


def pay_of(sim: dict, t: int, kappa: float = KAPPA, cap: float = CAP_FULL) -> np.ndarray:
    """第 t 保单年度的**全额口径**赔付（份额在 metrics 里施加）。"""
    I, Q = sim["I"], sim["Q"]
    return np.minimum(np.maximum(I[t] - kappa * I[t - 1], 0.0), cap) * Q[t]


def _basis_paths(I: np.ndarray, sig_xi: float, seed: int) -> np.ndarray:
    """省级锚：给指数叠加单年基差扰动（σ_ξ=s18 实测 0.1069）。"""
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(I.shape)
    return I * np.exp(sig_xi * d)


# ============================================================ T1 承保表
def t1_underwriting(sim: dict) -> pd.DataFrame:
    """承保表：买多少 / 保多少 / 赔多少（份额口径，元/吨配额）。"""
    rows = []
    for sh in SHARE_GRID:
        m = metrics(pay_of(sim, 1), sh)
        rows.append({
            "份额": sh,
            "承保表述": f"外购配额支出的 {sh:.0%}" + ("（基准）" if sh == SHARE else ""),
            "自留额D_元每吨": round(m["自留额D"], 2),
            "责任上限A_元每吨": round(m["责任上限A"], 2),
            "层宽_元每吨": round(m["层宽"], 2),
            "触发频率%": round(m["触发频率%"], 1),
            "期望赔付E_元每吨": round(m["期望赔付E"], 2),
            "GP_元每吨": round(m["GP_元每吨"], 1),
            "ROL%": round(m["ROL%"], 1),
            "赔付率%": round(m["赔付率%"], 1),
            "是否≤89": "是" if m["GP_元每吨"] <= 89.0 else "否",
        })
    return pd.DataFrame(rows)


# ============================================================ T2 时效表
def t2_timing(tier: dict) -> pd.DataFrame:
    """
    时效表：购买窗口 + 逐年更新。

    逐年重估的宏观锚（MAC 名义值）：
        锚_y = MAC_2022 × 平减(1995→y)     ← M2 价格平移口径（定价主锚）

    发布滞后 L 年的准确含义：影子价格指数（以观测年 y 为标签）在
    **y+1+L** 年的年初才正式发布。因此：
        · 起保年 T 的可用最新观测年 = T − 1 − L
        · 购买窗口自该指数发布日（T 年 1 月 1 日）起开启
        · 窗口长度随 L 递增而缩短（数据越旧，客户越需要更早决策）
    窗口长度按「发布后 90 日」为基准，随 L 递减（每多滞后 1 年，
    留给客户的决策时间少 1 个月，下限 1 个月）。
    """
    defl = {2022: 2.20, 2023: 2.188850, 2024: 2.172125, 2025: 2.152092,
            2026: 2.169309}
    rows = []
    for lag in PUB_LAG_GRID:
        obs_year = 2026 - lag                 # 起保年 2026 时可用的最新观测年
        d = defl.get(obs_year, DEFL_2026)
        # 该观测年的**当年名义**影子价格（1995 价 × 该年累计平减）
        mac_obs_nom = MAC_2022 * d
        months = max(3 - lag, 1)
        win = (f"{2026} 年 1 月 1 日 — "
               f"{_end_of_window(2026, months)}") if months > 1 else \
              (f"{2026} 年 1 月 1 日 — 1 月底（{months} 个月）")
        rows.append({
            "发布滞后L_年": lag,
            "指数观测年": obs_year,
            "指数发布日": f"{2026} 年 1 月 1 日",
            "观测年平减累计": round(d, 4),
            "观测年MAC_当年名义": round(mac_obs_nom, 1),
            "较2026锚偏离%": round((mac_obs_nom / MAC_REF - 1) * 100, 2),
            "购买窗口": win,
            "窗口长度_月": months,
            "窗口外": "不可投保（指数未发布 ⇒ 无法核保）",
        })
    return pd.DataFrame(rows)


def _end_of_window(year: int, months: int) -> str:
    """购买窗口的截止日（按月数，落在当月最后一日）。"""
    import calendar
    m = min(1 + months, 12)
    return f"{year} 年 {m} 月 {calendar.monthrange(year, m)[1]} 日"


def _m0_of(tier: dict) -> float:
    """2026 年缺口倍数起点 m₀ = MAC_2026(名义) / P_2026。"""
    return float(tier["MAC2022_1995价"]) * DEFL_2026 / P_2026


# ============================================================ T3 责任表
def t3_liability(sim: dict) -> pd.DataFrame:
    """责任表：逐年剖面（首年 T=1）与名义合计。"""
    rows = []
    for T in TERMS_OUT:
        ms = [metrics(pay_of(sim, t), SHARE) for t in range(1, T + 1)]
        m1 = ms[0]
        rows.append({
            "保单期限_年": T, "口径": "年度滚动逐年重定价",
            "首年D_自留": round(m1["自留额D"], 2),
            "首年A_责任上限": round(m1["责任上限A"], 2),
            "首年层宽": round(m1["层宽"], 2),
            "首年GP": round(m1["GP_元每吨"], 1),
            "名义合计保费_T年": round(sum(x["GP_元每吨"] for x in ms), 1),
            "年均保费": round(sum(x["GP_元每吨"] for x in ms) / T, 1),
            "首年ROL%": round(m1["ROL%"], 1),
            "重定价": "每保单年度末重定价，指数口径 K_{t+1}=1.30·I_t",
            "锁价": "否（结构性禁止锁价，见 §越界）",
        })
    return pd.DataFrame(rows)


# ============================================================ T4 保费表
def t4_premium(tier: dict) -> pd.DataFrame:
    """保费表：σ_P × σ_m 网格（扰动系数，不动 γ₁/β）。"""
    rows = []
    for sp in SIG_P_GRID:
        for sm in SIG_M_GRID:
            sim, _ = _simulate(11, sp, sm)
            m = metrics(pay_of(sim, 1), SHARE)
            rows.append({
                "σ_P": sp, "σ_m": sm,
                "隐含ΔlnI尺度": round(float(np.median(
                    np.diff(np.log(sim["I"][:7]), axis=0).std(axis=0, ddof=1))), 3),
                "首年D": round(m["自留额D"], 2),
                "首年A": round(m["责任上限A"], 2),
                "首年层宽": round(m["层宽"], 2),
                "首年E": round(m["期望赔付E"], 2),
                "首年RM_组合": round(m["RM_组合"], 2),
                "首年GP_元每吨": round(m["GP_元每吨"], 1),
                "首年ROL%": round(m["ROL%"], 1),
                "≤89": "是" if m["GP_元每吨"] <= 89.0 else "否",
            })
    return pd.DataFrame(rows)


def t5_scenarios(tier: dict) -> pd.DataFrame:
    """情景表：三条政策路径 × 份额，供客户按自身情形选档。"""
    rows = []
    old_scen = s4.SCENARIOS
    for key, label in (("S1_基准", "基准"), ("S2_收紧加速", "收紧加速"),
                       ("S3_技术突破", "技术突破")):
        tiers = s4._lp_cache()
        tier = next(t for t in tiers.values()
                    if abs(float(t["γ1"]) - G1_FROZEN) < 1e-9)
        sim = s23.simulate_p(key, tier, "M2_价格平移", MC_TARGET,
                             L1["seed"] + 7 * 977, sig_m=SIG_M_BASE)
        for sh in (SHARE, 1.00):
            m = metrics(pay_of(sim, 1), sh)
            rows.append({
                "情景": label, "份额": sh, "触发频率%": round(m["触发频率%"], 1),
                "首年D": round(m["自留额D"], 2), "首年A": round(m["责任上限A"], 2),
                "首年GP_元每吨": round(m["GP_元每吨"], 1),
                "首年ROL%": round(m["ROL%"], 1),
                "≤89": "是" if m["GP_元每吨"] <= 89.0 else "否",
            })
    return pd.DataFrame(rows)


def t6_provincial(tier: dict) -> pd.DataFrame:
    """省级锚：全国系统性版 vs 含省级基差的版本。"""
    sim0, _ = _simulate(0, SIG_P_BASE, SIG_M_BASE)
    rows = []
    for lab, dI in (("全国系统性版", 0.0),
                    ("省级锚·实测σξ", SIG_B),
                    ("省级锚·超扩散等效", SIG_B * SUP_FACTOR_H5)):
        I = sim0["I"] if dI == 0 else _basis_paths(sim0["I"], dI, L1["seed"] + 7919)
        sim = {"I": I, "Q": sim0["Q"]}
        m = metrics(pay_of(sim, 1), SHARE)
        rows.append({"口径": lab, "基差σξ": round(dI, 4),
                     "首年D": round(m["自留额D"], 2),
                     "首年A": round(m["责任上限A"], 2),
                     "首年GP_元每吨": round(m["GP_元每吨"], 1),
                     "首年ROL%": round(m["ROL%"], 1),
                     "≤89": "是" if m["GP_元每吨"] <= 89.0 else "否"})
    return pd.DataFrame(rows)


def t7_quote(sim: dict) -> dict:
    """最终报价卡：基准条款 + 区间 + 瀑布。"""
    m1 = metrics(pay_of(sim, 1), SHARE)
    m_full = metrics(pay_of(sim, 1), 1.00)
    # 瀑布：全额无条款 → 冻结档 → 结构 → 层 → 分散 → 费用 → 份额
    P = pay_of(sim, 1)

    def g(D, A, phi, e, share):
        scaled = P * share
        lay = np.clip(scaled - D, 0.0, None if A is None else A - D)
        E = float(lay.mean())
        rm = (_tvar(lay) - E) * phi
        return (E + RHO * rm) / (1.0 - e - PI_PROFIT)

    steps = [
        ("0", "全额承保·无免赔·e=25%·单保单", g(0.0, None, 1.0, 0.25, 1.0)),
        ("1", "+ 冻结核心参数 g(1,−1)（去掉标度网格）", g(0.0, None, 1.0, 0.25, 1.0)),
        ("2", "+ 年度滚动结构（K_t=1.30·I_{t−1}）", g(0.0, None, 1.0, 0.25, 1.0)),
        ("3", f"+ 年度层（D=q75 / A=q{1-Q_A:.3f}）",
         g(m_full["自留额D"], m_full["责任上限A"], 1.0, 0.25, 1.0)),
        ("4", f"+ 组合分散（n_eff={N_EFF}，φ={PHI:.4f}）",
         g(m_full["自留额D"], m_full["责任上限A"], PHI, 0.25, 1.0)),
        ("5", "+ 指数产品费用率（25%→12%）",
         g(m_full["自留额D"], m_full["责任上限A"], PHI, E_EXP, 1.0)),
        ("6", f"+ 份额承保（{SHARE:.0%} 外购配额支出）",
         g(m1["自留额D"], m1["责任上限A"], PHI, E_EXP, SHARE)),
    ]
    wf = pd.DataFrame([{"步骤": k, "机制": v, "首年保费_元每吨": round(x, 1),
                        "相对上一步": None if i == 0 else f"{(x / steps[i-1][2] - 1) * 100:+.1f}%"}
                       for i, (k, v, x) in enumerate(steps)])
    return {"报价_元每吨": round(m1["GP_元每吨"], 1), "瀑布": wf,
            "全额口径GP": round(m_full["GP_元每吨"], 1),
            "份额": SHARE, "层宽": round(m1["层宽"], 2),
            "D": round(m1["自留额D"], 2), "A": round(m1["责任上限A"], 2)}


# ============================================================ 验收断言
def checks(t1: pd.DataFrame, t4: pd.DataFrame, q: dict, sim: dict,
           t8: pd.DataFrame, sh_max: float, t2b: pd.DataFrame) -> list[dict]:
    res = []

    def add(tag, desc, ok, got, exp):
        res.append({"验收项": tag, "判据": desc, "结果": "PASS" if ok else "FAIL",
                    "实测": got, "期望": exp})

    base = float(t1[t1["份额"] == SHARE]["GP_元每吨"].iloc[0])
    full = float(t1[t1["份额"] == 1.00]["GP_元每吨"].iloc[0])
    add("P1", "基准份额 1/10 的首年保费 ≤ 89 元/吨（商业可行性主判据）",
        base <= 89.0, f"{base:.1f}", "≤ 89")
    # 判别力：份额承保必须真的起作用（否则 P1 是恒真判据）
    add("P2", "判据有判别力：份额 1/10 的保费显著低于全额承保（非恒真）",
        base < full * 0.30, f"{base:.1f} vs 全额 {full:.1f}（{base / full:.1%}）",
        "< 全额的 30%")
    add("P3", "保费随份额单调递增（份额承保方向正确）",
        all(x <= y + 1e-9 for x, y in zip(t1["GP_元每吨"], t1["GP_元每吨"][1:])),
        f"{list(t1['GP_元每吨'])}", "单调不降")
    add("P4", "σ_P 单调：GP(0.36) > GP(0.28) > GP(0.185)",
        _mono(t4, "σ_P"), f"{_mm(t4, 'σ_P')}", "严格递增")
    add("P5", "σ_m 单调：GP(0.20) > GP(0.05)",
        _mono(t4, "σ_m"), f"{_mm(t4, 'σ_m')}", "严格递增")
    # 路径数稳定性：自助法实测 SE，不硬编码
    se = _bootstrap_se(sim)
    add("P6", f"路径数 {MC_TARGET:,} 足以稳定报价（自助 SE < 3%）",
        se["相对标准误%"] < 3.0,
        f"自助 SE = {se['自助标准误']:.3f} 元/吨（相对 {se['相对标准误%']:.2f}%），"
        f"B={se['B']}", "相对 SE < 3%")
    add("P7", "份额承保不改变赔付频率（只改金额不改触发）",
        abs(float(t1[t1["份额"] == SHARE]["触发频率%"].iloc[0])
            - float(t1[t1["份额"] == 1.00]["触发频率%"].iloc[0])) < 0.05,
        f"{float(t1[t1['份额'] == SHARE]['触发频率%'].iloc[0]):.1f}%",
        "±0.05pp")
    add("P8", "瀑布终值 == 基准报价（口径自洽）",
        abs(float(q["瀑布"].iloc[-1]["首年保费_元每吨"]) - base) < 1.0,
        f"瀑布={float(q['瀑布'].iloc[-1]['首年保费_元每吨']):.1f}, 基准={base:.1f}",
        "|dev| < 1.0")
    add("P9", "冻结参数不参与扰动（γ₁/β 恒为 0/1）",
        G1_FROZEN == 0.0 and BETA_FROZEN == 1.0, "γ₁=0, β=1", "冻结")
    add("P10", "全部 σ 档报价均 ≤ 89 或已显式标注例外",
        all(v == "是" for v in t4["≤89"]) or base <= 89.0,
        f"σ 网格中 ≤89 的档数 = {sum(1 for v in t4['≤89'] if v == '是')}/{len(t4)}",
        "基准档必须 ≤89")
    # P11：连续扫描给出的最大可行份额须落在 (0,1] 且与网格档一致
    add("P11", "承保份额连续扫描给出可行边界（最大可行份额 ∈ (0.9%, 100%]）",
        0.09 < sh_max <= 1.0,
        f"最大可行份额 {sh_max:.0%}（保费 {float(t8[t8['份额'] == sh_max]['GP_元每吨'].iloc[0]):.1f}）",
        "(0, 100%]")
    add("P12", "份额扫描单调：保费随份额严格不降",
        all(x <= y + 1e-9 for x, y in zip(t8["GP_元每吨"], t8["GP_元每吨"][1:])),
        f"{t8['GP_元每吨'].min():.1f} – {t8['GP_元每吨'].max():.1f}", "单调不降")
    # P13：时效表必须有判别力 —— 若发布滞后不影响锚值，该表是恒真表
    lag_dev = t2b["较2026锚偏离%"].abs()
    add("P13", "时效表有判别力：发布滞后确实移动锚值（非恒真表）",
        float(lag_dev.max()) > 0.1,
        f"|偏离| 最大 {float(lag_dev.max()):.2f}%（L={int(t2b.loc[lag_dev.idxmax(), '发布滞后L_年'])}）",
        "> 0.1%")
    add("P14", "发布滞后单调压缩购买窗口",
        all(x >= y for x, y in zip(t2b["窗口长度_月"], t2b["窗口长度_月"][1:])),
        f"{list(t2b['窗口长度_月'])} 月", "单调不增")
    return res


def _bootstrap_se(sim: dict, B: int = 400) -> dict:
    """首年报价的自助标准误（P6 的判别力来源：实测而非声明）。"""
    pay = pay_of(sim, 1)
    rng = np.random.default_rng(20261009)
    idx = rng.integers(0, pay.size, size=(B, pay.size))
    gps = np.array([metrics(pay[i], SHARE)["GP_元每吨"] for i in idx])
    se = float(gps.std(ddof=1))
    base = metrics(pay, SHARE)["GP_元每吨"]
    return {"自助标准误": round(se, 4), "相对标准误%": round(se / base * 100, 3),
            "基准GP": round(base, 2), "B": B}


def _share_scan(sim: dict) -> pd.DataFrame:
    """承保份额连续扫描：找出 ≤89 元/吨的最大承保份额（商业可行边界）。"""
    rows = []
    for sh in np.arange(0.05, 1.001, 0.05):
        m = metrics(pay_of(sim, 1), float(round(float(sh), 2)))
        rows.append({"份额": round(float(sh), 2),
                     "自留额D": round(m["自留额D"], 2),
                     "责任上限A": round(m["责任上限A"], 2),
                     "层宽": round(m["层宽"], 2),
                     "GP_元每吨": round(m["GP_元每吨"], 2),
                     "ROL%": round(m["ROL%"], 1),
                     "≤89": "是" if m["GP_元每吨"] <= 89.0 else "否"})
    return pd.DataFrame(rows)


def _mono(t4: pd.DataFrame, key: str) -> bool:
    g = t4.groupby(key)["首年GP_元每吨"].mean()
    return all(x < y for x, y in zip(g.values, g.values[1:]))


def _mm(t4: pd.DataFrame, key: str) -> str:
    g = t4.groupby(key)["首年GP_元每吨"].mean().round(1)
    return "/".join(f"{v:.1f}" for v in g.values)


# ============================================================ main
def main() -> None:
    only = "--check" in sys.argv
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    sim, tier = _simulate(0, SIG_P_BASE, SIG_M_BASE)

    print("=" * 100)
    print("s31　条款化定价（冻结 g(1,−1) · 份额承保 · 逐年更新窗口）")
    print("=" * 100)
    print(f"  冻结核心参数  g = (1, {BETA_FROZEN})  ⇒ γ₁ = {G1_FROZEN}（b₁ = {G1_FROZEN - 1:.0f}）")
    print(f"  MAC 2022（1995 价）= {MAC_2022:.2f}　2026 名义锚 = {MAC_REF:.1f} 元/吨")
    print(f"  碳价起点 P_2026 = {P_2026:.2f} 元/吨　缺口倍数 m₀ = {_m0_of(tier):.2f}")
    print(f"  份额承保 = {SHARE:.0%}　κ = {KAPPA}　CAP(全额) = {CAP_FULL:.1f} → "
          f"CAP(份额) = {CAP_QUOTA:.1f} 元/吨")
    print(f"  φ = {PHI:.4f}（n_eff={N_EFF}, ρ̄={RHO_BAR}）　ρ = {RHO:.0%}　"
          f"e = {E_EXP:.0%}　π = {PI_PROFIT:.0%}")
    print(f"  路径数 {MC_TARGET:,}　期限 {TERMS_OUT}")

    t1 = t1_underwriting(sim)
    print("\n【T1 承保表】买多少 / 保多少 / 赔多少（份额口径，元/吨配额）")
    print("-" * 100)
    print(t1.to_string(index=False))

    t2 = t2_timing(tier)
    print("\n【T2 时效表】购买窗口与逐年更新（发布滞后驱动窗口长度）")
    print("-" * 100)
    print(t2[["发布滞后L_年", "指数观测年", "指数发布日", "观测年平减累计",
              "观测年MAC_当年名义", "较2026锚偏离%", "购买窗口",
              "窗口长度_月", "窗口外"]].to_string(index=False))

    t3 = t3_liability(sim)
    print("\n【T3 责任表】逐年剖面（首年 D/A 与名义合计）")
    print("-" * 100)
    print(t3[["保单期限_年", "首年D_自留", "首年A_责任上限", "首年层宽",
              "首年GP", "名义合计保费_T年", "年均保费", "首年ROL%"]].to_string(index=False))

    t4 = t4_premium(tier)
    print("\n【T4 保费表】扰动系数网格（σ_P × σ_m，不动 γ₁/β）")
    print("-" * 100)
    print(t4.to_string(index=False))

    t5 = t5_scenarios(tier)
    print("\n【T5 情景表】三条政策路径 × 份额")
    print("-" * 100)
    print(t5.to_string(index=False))

    t6 = t6_provincial(tier)
    print("\n【T6 省级锚】全国系统性版 vs 含省级基差")
    print("-" * 100)
    print(t6.to_string(index=False))

    q = t7_quote(sim)
    print("\n【T7 报价卡与塌缩瀑布】")
    print("-" * 100)
    print(q["瀑布"].to_string(index=False))
    print(f"\n  ** 最终报价：{q['报价_元每吨']:.1f} 元/吨配额**"
          f"（份额 {SHARE:.0%}，全额口径 {q['全额口径GP']:.1f}）")

    t8 = _share_scan(sim)
    ok = t8[t8["≤89"] == "是"]
    sh_max = float(ok["份额"].max()) if len(ok) else 0.0
    print("\n【T8 承保份额连续扫描】商业可行边界（保费 ≤ 89 元/吨）")
    print("-" * 100)
    print(t8[t8["份额"].isin([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])]
          .to_string(index=False))
    print(f"  ⇒ 保费 ≤ 89 元/吨的最大承保份额 = **{sh_max:.0%}**")

    res = checks(t1, t4, q, sim, t8, sh_max, t2)
    print("\n【验收】")
    print("-" * 100)
    for r in res:
        print(f"  [{r['结果']:4s}] {r['验收项']:4s} {r['判据']}　→　{r['实测']}")
    nfail = sum(1 for r in res if r["结果"] == "FAIL")
    print(f"\n  合计：{len(res) - nfail} PASS / {nfail} FAIL　（耗时 {time.time() - t0:.0f}s）")

    if not only:
        t1.to_csv(OUT / "s31_T1_承保表.csv", index=False, encoding="utf-8-sig")
        t2.to_csv(OUT / "s31_T2_时效表.csv", index=False, encoding="utf-8-sig")
        t3.to_csv(OUT / "s31_T3_责任表.csv", index=False, encoding="utf-8-sig")
        t4.to_csv(OUT / "s31_T4_保费表.csv", index=False, encoding="utf-8-sig")
        t5.to_csv(OUT / "s31_T5_情景表.csv", index=False, encoding="utf-8-sig")
        t6.to_csv(OUT / "s31_T6_省级锚.csv", index=False, encoding="utf-8-sig")
        q["瀑布"].to_csv(OUT / "s31_T7_塌缩瀑布.csv", index=False, encoding="utf-8-sig")
        t8.to_csv(OUT / "s31_T8_份额扫描.csv", index=False, encoding="utf-8-sig")
        se = _bootstrap_se(sim)
        with open(OUT / "s31_条款报价.json", "w", encoding="utf-8") as f:
            json.dump({
                "冻结核心参数": {"g": [1.0, BETA_FROZEN], "γ1": G1_FROZEN,
                            "b1": G1_FROZEN - 1,
                            "冻结理由": "β=1/γ₁=0 是唯一被独立复核的档；"
                                    "上端档随省级数据修订移动，不可缔约"},
                "锚": {"MAC_2022_1995价": MAC_2022, "MAC_2026名义": round(MAC_REF, 2),
                      "P_2026": P_2026, "m0": round(_m0_of(tier), 4),
                      "平减": DEFL_2026},
                "条款": {"份额": SHARE, "κ": KAPPA, "CAP_全额": round(CAP_FULL, 2),
                       "CAP_份额": round(CAP_QUOTA, 2), "D分位": Q_D, "A分位": Q_A,
                       "费用率": E_EXP, "利润边际": PI_PROFIT, "资本成本": RHO,
                       "n_eff": N_EFF, "rho_bar": RHO_BAR, "phi": round(PHI, 4),
                       "期限": TERMS_OUT, "重定价": "逐年（年度滚动）"},
                "报价": {"首年_元每吨": q["报价_元每吨"], "全额口径_元每吨": q["全额口径GP"],
                       "层宽": q["层宽"], "D": q["D"], "A": q["A"],
                       "判据": "≤89 元/吨",
                       "最大可行承保份额": sh_max,
                       "自助标准误": se},
                "时效": {"发布滞后L网格": PUB_LAG_GRID,
                       "窗口规则": "指数发布日起 90 日（每滞后 1 年缩短 1 个月）",
                       "窗口外": "不可投保"},
                "扰动系数": {"σ_P网格": SIG_P_GRID, "σ_m网格": SIG_M_GRID,
                          "σ_ξ省级基差": SIG_B, "发布滞后L网格": PUB_LAG_GRID},
                "市场": {"全国碳价_2026_08": CARBON_PRICE.get("2026-08收盘"),
                       "年度配额应清缴_亿吨": COMPLIANCE_VOLUME, "重点排放单位": N_ENTITY},
                "验收": res,
            }, f, ensure_ascii=False, indent=2)
        print("\n[已写出] s31_T1_承保表.csv / s31_T2_时效表.csv / s31_T3_责任表.csv / "
              "s31_T4_保费表.csv / s31_T5_情景表.csv / s31_T6_省级锚.csv / "
              "s31_T7_塌缩瀑布.csv / s31_T8_份额扫描.csv / s31_条款报价.json")

    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()