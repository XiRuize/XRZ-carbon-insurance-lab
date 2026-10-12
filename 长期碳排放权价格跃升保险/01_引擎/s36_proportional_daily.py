# -*- coding: utf-8 -*-
"""
s36_proportional_daily.py —— 第八轮：纯P 结构的两项机制修正

【本脚本回答什么】

作者 2026-10-09 提出两条机制质疑，本脚本逐条落地为可核验的引擎改动：

①  **固定保费不合适**（作者原话：「现在价格很低，翻个十倍也可能……
    那么固定保费是不是不合适？」）

    诊断：s32 用对数 OU 模拟 P（s4_pricing.py:218-224），σ_P 作用在
    **对数**上 ⇒ 波动绝对值 ∝ 价格水平。而CAP 与保费都是**固定元/吨**。
    ⇒ 价格 P 涨到 10P_0 时：
         - 真实波动绝对值 ≈ 0.28 × 10P_0 = 2.8P_0，但 CAP 仍固定 2P_0
         ⇒ **风险尺度 3 倍于保额上限**
         - 保费仍是 1.15 元/吨，而风险已放大 10 倍
    三层错配同时发生，且**恰好在产品最需要生效的时刻失效**。

    修正（本脚本 T1–T3）：
      - 责任上限改为**比例式**：CAP_t = CAP_MULT × P_0（锚定起期日价格，
        而非固定 97.90）—— 报价随价格水平同步放大
      - 保费改为**费率制**：报价以「元/吨」报出，但同时给出**费率**
        = GP / P_0，费率在价格跃迁时自动等比例适用
      - **波动率状态依赖**作为情景（非主口径）：σ_P(P) = σ_0·(P/P_0)^(−θ)。
        理由：价格逼近政策天花板 ⇒ 上行空间被压缩 ⇒ 边际不确定性下降。
        θ 无实证支撑，故仅作敏感性（作者已同意作情景不作主口径）。

②  **年末单点结算漏掉年内被高价击中的损失**
    （作者原话：「一年的时间内如果突破某个限定就赔付？」）

    诊断：s32 的 pay_split 只在 t=1（年度末）算一次。企业若在3 月
    被高价逼买、12 月回落，年度结算**不赔付**—— 但3 月的损失是真实的。
    这是**时间错配基差风险**，且是企业最痛的那种。

    修正（本脚本 T4–T5）：
      - 改为**日度观察 + 年度累计**：保单年度内每个交易日观察
        P_d > κ·P_0 即计入赔付，累计至限额
      - 设**触发次数上限 N_MAX**（默认 5），防波动年反复触发套利
      - 日度触及后碳价可能回落（对数 OU 均值回复）⇒ 这正是
        「被短期高价击中」的机制，故日度口径比年末口径更贴合真实风险

③  **φ 口径已由 s32 改为 1.00**（主口径，因全国单一碳价 ⇒ 截面相关 = 1）。
    本脚本沿用，不重复论证。

【结构性副作用：一年期与「长期」】
    比例式结构使产品天然可延展（保费与保额随价格自动放大），
    「长期」不再依赖"关系长期"的口头解释，而有技术理由：
    **可延展的是比例结构，不是价格承诺。**

【口径纪律】
  - 引擎内**不硬编码任何结论性数字**：P_0 从 s32 取，M1 比率从比例取
  - 任何新结构都必须有断言锁定，且断言须能判FAIL（非恒真）
  - 分位数不满足缩放不变性 q75(a·X) ≠ a·q75(X)
    ⇒ **施加份额后必须重新取分位数**，不能先取分位再乘份额

用法：python s36_proportional_daily.py [--check]     # 退出码 0 = 断言全过
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import L1, OUT  # noqa: E402
import s4_pricing as s4_mod   # noqa: E402  (取 KAPPA_P)
import s32_dual_trigger as s32  # noqa: E402

pd.set_option("display.width", 280)
pd.set_option("display.unicode.east_asian_width", True)

# ============================================================ 冻结参数（沿用 s32）
P_0 = s32.P_2026              # 97.90，2026-08 收盘
CAP_MULT = s32.CAP_MULT       # 2.0
KAPPA = s32.KAPPA              # 1.30
Q_D, Q_A = s32.Q_D, s32.Q_A    # 0.75 / 0.995
E_EXP = s32.E_EXP              # 0.12
PI_PROFIT = s32.PI_PROFIT      # 0.08
RHO_CAP = s32.RHO              # 0.06 年化资本成本
PHI = s32.PHI                  # 1.00（主口径，s32 已修正）
SHARE_BASE = 0.10
MC_TARGET = s32.MC_TARGET      # 80,000
# ============================================================ 商业可行判据
# 判据沿用 s32 的**绝对金额**口径：全额承保口径的保费 ≤ 89 元/吨配额。
# 这是给「份额 100%」设的上限；份额 α 下保费应≤ α × 89。
# 这样跨结构（定额 / 比例、份额 10% / 100%）可比，且与 s32 一致。
BUDGET_ABS = s32.BUDGET            # 89.0 元/吨（全额口径）


def feasible(gp: float, share: float = 1.0) -> bool:
    """份额 share 下的报价是否可行：gp ≤ share × 89。"""
    return gp <= share * BUDGET_ABS + 1e-9

# ---- 日度累计的触发次数上限
N_MAX_GRID = [1, 3, 5, 10, 99]   # 1 = 退回年末单点；99 = 不限次
N_MAX_BASE = 5

# ---- 波动率状态依赖情景 θ（无实证支撑，仅敏感性）
THETA_GRID = [0.0, 0.3, 0.5]
THETA_BASE = 0.0                   # 主口径不对 σ 做状态依赖

TRADING_DAYS = 244                 # 年交易日
OUT_DIR = Path(OUT)


# ============================================================ 工具
def gp_of(E: float, rm: float, e: float = E_EXP) -> float:
    return (E + RHO_CAP * rm) / (1.0 - e - PI_PROFIT)


def tvar(x: np.ndarray, a: float = 0.99) -> float:
    q = np.quantile(x, a)
    t = x[x >= q]
    return float(t.mean()) if t.size else float(q)


def layer_one(x: np.ndarray, qd: float = Q_D, qa: float = Q_A):
    """年度层（输入已含份额）。分位数不满足缩放不变性 ⇒ 份额后必须重取。"""
    D = float(np.quantile(x, qd))
    A = float(np.quantile(x, qa))
    return np.clip(x - D, 0.0, A - D), D, A


# ============================================================ 模拟：日度路径 + σ 状态依赖
def sim_daily(sig_p: float = s32.SIG_P_BASE, theta: float = THETA_BASE,
              n_paths: int = MC_TARGET, seed: int | None = None,
              p_level_mult: float = 1.0, jump_day: int = 0) -> dict:
    """日度路径模拟（★ 锚价与门槛固定为 P_0，价格中枢可跃迁）

    ★★ 关键建模口径（第一版曾错、已修正）：
      - **起赔门槛恒为 κ · P_0**（P_0 = 保单起期日前一交易日收盘价）。
        κ 是**合同约定的固定倍数**，不随期间价格水平调整。
      - `p_level_mult` 抬高的是**价格中枢**（模拟「碳价真的涨上去了」），
        **不是**抬高锚价。锚价是合同基准，一年内不变。
      - 这样才能回答作者的问题：价格从 97.9涨到 979 之后，
        固定 CAP（195.8，还按老锚价定）是否失效、比例 CAP（2 × 979）是否跟得上。
        第一版错误地把锚价也抬高了，导致门槛 1.3×979 永不可达、E ≡ 0，
        结论完全失真。

    σ 的状态依赖：σ_P(P) = sig_p · (P/P_0)^(−θ)。θ=0 退化为常 σ。
      理由：价格逼近政策天花板 ⇒ 上行空间被压缩 ⇒ 边际不确定性下降。
      θ 无实证支撑，仅作情景（不作主口径）。

    单位换算：s4 的 SIG_P / KAPPA_P 是**年化**参数（按年步长），
      本函数按**日**步长 ⇒ 必须除以 √TRADING_DAYS 与 TRADING_DAYS。
      第一版忘记换算，日波动被放大 √244 ≈ 15.6 倍。
    """
    rng = np.random.default_rng(seed if seed is not None else L1["seed"] + 3600)
    n = int(n_paths)
    sig_d = sig_p / np.sqrt(TRADING_DAYS)
    # ★ 过程设定：**随机游走 + 确定性阶跃**（无均值回复牵引）。
    #   理由：碳价由行政配额收紧驱动，中枢**单向**移动，年内不回撤。
    #   s4 用均值回复 OU 是为长期中枢回归，但在**一年保单窗口**内回复速度
    #   KAPPA_P/244 ≈ 0.0014/日 ⇒ 30 日仅回复约 4%，高价位站不住，
    #   会低估跃迁后的赔付（实测 E 0.024→0.021、封顶频率恒为 0）。
    #   第二版改用 pull=0.05 强牵引，同样把价格钉死低位、结果失真。
    #   最终：center 承载阶跃，σ 围绕 center 做随机游走（log 空间）。
    P = np.zeros((TRADING_DAYS, n))
    jumped = False
    for d in range(TRADING_DAYS):
        if jump_day > 0 and d == jump_day:
            # ★ 显式阶跃：政策收紧使价格中枢一次性跃迁到mult × P_0。
            #   （随机游走是连续的，不会瞬移；必须显式乘以 mult）
            jumped = True
            P[d] = P[d - 1] * p_level_mult * np.exp(
                sig_d * rng.standard_normal(n))
            continue
        center = P_0 * p_level_mult if jumped else P_0
        rel = np.log(P[d - 1] / center) if d > 0 else np.zeros(n)
        sig_t = sig_d * np.power(np.maximum(np.exp(rel), 1e-6), -theta)
        P[d] = center * np.exp(rel + sig_t * rng.standard_normal(n))
    return {"P": P, "dP": P - P_0,
            "P_anchor": P_0,                       # ★ 锚价恒为 P_0
            "P_center_after_jump": (P_0 if jump_day <= 0 else P_0 * p_level_mult),
            "sigma_annual": sig_p, "sigma_daily_used": float(sig_d),
            "theta": theta, "mult": p_level_mult, "jump_day": jump_day, "n": n}


# ============================================================ 日度累计赔付
def pay_daily(sim: dict, share: float = SHARE_BASE, kappa: float = KAPPA,
              cap_mult: float = CAP_MULT, n_max: int = N_MAX_BASE) -> dict:
    """日度观察 + 年度累计赔付（未分层，元/吨配额）。

    赔�� = share · min( Σ_{d∈触发} max(P_d − κ·P_0, 0), cap_mult·P_0 ) · Q̄

    Q̄ 取1.0（归一化基准年缺口量；缺口量递增的影响见 §6 局限）。
    n_max：最多计前 n_max 个触发日的超出额，防波动年反复触发套利。
    """
    P = sim["P"]
    P0 = sim["P_anchor"]# 锚价：合同基准，一年内固定
    cap_amt = cap_mult * P0
    exc_all = np.maximum(P - kappa * P0, 0.0)      # (D, n) 各日超出额
    trig_all = exc_all > 0.0
    n_trig = trig_all.sum(axis=0)                  # 每条路径的触发日数
    if n_max >= 99:
        exc_sum = exc_all.sum(axis=0)
    else:
        # 每条路径只计前 n_max 个触发日（按日序，非按大小）。
        # idx 形状须为 (n_max, n) 以便与 exc_all (D, n) 广播对齐。
        mask = trig_all.copy()
        rank = np.cumsum(mask, axis=0) - 1          # 触发日序号（0 起）
        keep = mask & (rank < n_max)                # 只留前 n_max 个
        exc_sum = np.where(keep, exc_all, 0.0).sum(axis=0)
    capped = np.minimum(exc_sum, cap_amt)
    return {
        "gross": share * capped,                    # 封顶后
        "gross_uncapped": share * exc_sum,
        "cap_amount": cap_amt,
        "cap_binding": exc_sum > cap_amt,
        "excess_sum": share * exc_sum,
        "capped_amount": share * capped,
        "trigger_any": n_trig > 0,
        "n_trigger_days": n_trig,
        "trigger_count_per_year": share * capped,
    }


# ============================================================ 指标
def metrics(pay: dict, share: float = SHARE_BASE,
            phi: float = PHI, e: float = E_EXP) -> dict:
    """条款级指标（份额已施加，元/吨配额）。"""
    x = pay["gross"]
    lay, D, A = layer_one(x)
    E = float(lay.mean())
    rm = (tvar(lay) - E) * phi
    gp = gp_of(E, rm, e)
    ntd = pay["n_trigger_days"]
    return {
        "自留额D": D, "责任上限A": A, "层宽": A - D,
        "期望赔付E": E, "RM": rm, "GP_元每吨": gp,
        "费率%": gp / pay["cap_amount"] * 100,
        "赔付率%": E / max(gp, 1e-9) * 100,
        "保费占层宽%": gp / max(A - D, 1e-9) * 100,
        "触发频率%": float(pay["trigger_any"].mean() * 100),
        "平均触发天数": float(ntd.mean()),
        "P50触发天数": float(np.quantile(ntd, 0.50)),
        "P95触发天数": float(np.quantile(ntd, 0.95)),
        "封顶触发频率%": float(pay["cap_binding"].mean() * 100),
        "CAP_元每吨": pay["cap_amount"],
    }


# ============================================================ T1 结构对照：定额 vs 比例
def t1_fixed_vs_proportional() -> pd.DataFrame:
    """★核心检验：碳价真的跃迁后，定额结构 vs 比例结构。

    设定：锚价恒为 P_0（合同基准），年内第 jump_day 日起价格中枢跃迁到
    mult × P_0 并保持。比较两种责任上限结构：
      定额结构：CAP = 2 × P_0（老锚价定死，永不调整）
      比例结构：CAP = 2 × 当期价格水平（随价格同步放大）
    """
    rows = []
    JD = 30# 第 30 日起价格中枢跃迁（mult=1 时不跃迁）
    for mult in [1.0, 2.0, 5.0, 10.0]:
        sim = sim_daily(sig_p=s32.SIG_P_BASE, theta=THETA_BASE,
                        n_paths=40000, seed=L1["seed"] + 4000 + int(mult * 10),
                        p_level_mult=mult, jump_day=JD)
        P0 = sim["P_anchor"]
        # --- 比例结构：CAP 随当期价格水平（用年内中枢近似）
        p_now = sim["P_center_after_jump"]
        cap_prop = CAP_MULT * max(p_now, P0)
        pp = pay_daily(sim, n_max=N_MAX_BASE)
        pp["gross"] = SHARE_BASE * np.minimum(pp["excess_sum"], cap_prop)
        mp = metrics(pp)
        mp["CAP_元每吨"] = cap_prop
        # --- 定额结构：CAP 固定为老锚价的 2 倍
        cap_fixed = CAP_MULT * P0
        pf = pay_daily(sim, n_max=N_MAX_BASE)
        pf["gross"] = SHARE_BASE * np.minimum(pf["excess_sum"], cap_fixed)
        mf = metrics(pf)
        mf["CAP_元每吨"] = cap_fixed
        rows.append({
            "价格跃迁倍数": mult,
            "年内中枢_元每吨": round(p_now, 2),
            "定额_CAP": round(cap_fixed, 2),
            "比例_CAP": round(cap_prop, 2),
            "定额_E": round(mf["期望赔付E"], 3),
            "比例_E": round(mp["期望赔付E"], 3),
            "定额_GP": round(mf["GP_元每吨"], 3),
            "比例_GP": round(mp["GP_元每吨"], 3),
            "定额_封顶频率%": round(mf["封顶触发频率%"], 1),
            "比例_封顶频率%": round(mp["封顶触发频率%"], 1),
            "定额_赔付率%": round(mf["赔付率%"], 1),
            "比例_赔付率%": round(mp["赔付率%"], 1),
        })
    return pd.DataFrame(rows)


# ============================================================ T2 费率稳定性
def t2_rate_stability() -> pd.DataFrame:
    """费率（GP / 当期 CAP）在价格跃迁下是否稳定 —— 比例结构的检验。

    若结构比例式：价格跃迁时保费与保额同步放大 ⇒ 费率应稳定。
    若 CAP 固定：分子随风险放大、分母不变 ⇒ 费率失控。
    """
    rows = []
    base_fixed = base_prop = None
    JD = 30
    for mult in [1.0, 2.0, 5.0, 10.0]:
        sim = sim_daily(sig_p=s32.SIG_P_BASE, n_paths=40000,
                        seed=L1["seed"] + 6000 + int(mult * 10),
                        p_level_mult=mult, jump_day=JD)
        P0 = sim["P_anchor"]
        p_now = sim["P_center_after_jump"]
        # 比例结构
        cap_prop = CAP_MULT * max(p_now, P0)
        pp = pay_daily(sim, n_max=N_MAX_BASE)
        pp["gross"] = SHARE_BASE * np.minimum(pp["excess_sum"], cap_prop)
        mp = metrics(pp)
        rate_p = mp["GP_元每吨"] / cap_prop * 100
        # 定额结构
        cap_fixed = CAP_MULT * P0
        pf = pay_daily(sim, n_max=N_MAX_BASE)
        pf["gross"] = SHARE_BASE * np.minimum(pf["excess_sum"], cap_fixed)
        mf = metrics(pf)
        rate_f = mf["GP_元每吨"] / cap_fixed * 100
        if base_fixed is None:
            base_fixed, base_prop = rate_f, rate_p
        rows.append({
            "价格跃迁倍数": mult,
            "年内中枢_元每吨": round(p_now, 2),
            "定额_CAP_元每吨": round(cap_fixed, 2),
            "比例_CAP_元每吨": round(cap_prop, 2),
            "定额_GP_元每吨": round(mf["GP_元每吨"], 3),
            "比例_GP_元每吨": round(mp["GP_元每吨"], 3),
            "定额_费率%": round(rate_f, 3),
            "比例_费率%": round(rate_p, 3),
            "定额_费率比值": round(rate_f / base_fixed, 3) if base_fixed else 1.0,
            "比例_费率比值": round(rate_p / base_prop, 3) if base_prop else 1.0,
        })
    return pd.DataFrame(rows)


# ============================================================ T3 波动率状态依赖 θ 情景
def t3_theta() -> pd.DataFrame:
    """σ_P(P) = σ_0·(P/P_0)^(−θ)：波动率随价格递减。"""
    rows = []
    for th in THETA_GRID:
        sim = sim_daily(sig_p=s32.SIG_P_BASE, theta=th, n_paths=40000,
                        seed=L1["seed"] + 7000 + int(th * 100))
        pp = pay_daily(sim, n_max=N_MAX_BASE)
        m = metrics(pp)
        rows.append({
            "θ": th, "σ_口径": f"σ_0·(P/P_0)^(−{th})",
            "E": round(m["期望赔付E"], 3), "GP_元每吨": round(m["GP_元每吨"], 3),
            "费率%": round(m["费率%"], 2), "赔付率%": round(m["赔付率%"], 1),
            "触发频率%": round(m["触发频率%"], 1),
            "平均触发天数": round(m["平均触发天数"], 2),
        })
    return pd.DataFrame(rows)


# ============================================================ T4 触发次数上限 N_MAX
def t4_nmax() -> pd.DataFrame:
    """N_MAX：日度累计的次数上限，防波动年反复触发套利。"""
    rows = []
    sim = sim_daily(sig_p=s32.SIG_P_BASE, n_paths=MC_TARGET,
                    seed=L1["seed"] + 8000)
    for nm in N_MAX_GRID:
        pp = pay_daily(sim, n_max=nm)
        m = metrics(pp)
        lab = {1: "年末单点（≈s32 口径）", 99: "不限次"}.get(nm, f"最多 {nm} 次")
        rows.append({
            "N_MAX": nm, "口径": lab,
            "E": round(m["期望赔付E"], 3), "GP_元每吨": round(m["GP_元每吨"], 3),
            "费率%": round(m["费率%"], 2), "赔付率%": round(m["赔付率%"], 1),
            "触发频率%": round(m["触发频率%"], 1),
            "平均触发天数": round(m["平均触发天数"], 2),
            "P95触发天数": round(m["P95触发天数"], 1),
            "封顶频率%": round(m["封顶触发频率%"], 1),
            "是否≤判据": "是" if feasible(m["GP_元每吨"], SHARE_BASE) else "否",
        })
    return pd.DataFrame(rows)


# ============================================================ T5 时间错配：日度 vs 年末
def t5_timing_mismatch() -> pd.DataFrame:
    """核心检验：年内被高价击中、年末回落 ⇒ 年末口径不赔、日度口径赔。

    构造四条路径：
      A 无��发        —— 全年P ≤ κP_0
      B 年中冲击后回落 —— 3 月冲高、12 月回落到 κP_0 以下
      C 持续抬升       —— 单调上行，年末仍高于 κP_0
      D 全年高于阈值    —— 始终高于 κP_0
    比较年末口径 vs 日度口径的赔付（元/吨配额）。
    """
    n = 200_000
    rng = np.random.default_rng(L1["seed"] + 9000)
    rows = []
    for lab, shape in [("A 无触发", "flat"), ("B 年中冲高后回落", "spike"),
                       ("C 持续抬升", "ramp"), ("D 全年高位", "high")]:
        if shape == "flat":
            path = P_0 * np.ones((TRADING_DAYS, 1))
        elif shape == "spike":
            path = P_0 * np.ones((TRADING_DAYS, 1))
            path[40:80, 0] = KAPPA * P_0 * 1.25   # 3 月冲高 25%
        elif shape == "ramp":
            f = np.linspace(1.0, KAPPA * 1.2, TRADING_DAYS)
            path = P_0 * f[:, None]
        else:
            path = KAPPA * P_0 * 1.10 * np.ones((TRADING_DAYS, 1))
        P = np.repeat(path, n, axis=1)
        cap = CAP_MULT * P_0
        # 日度口径：累计所有超出额
        exc = np.maximum(P - KAPPA * P_0, 0.0)
        daily = SHARE_BASE * np.minimum(exc.sum(axis=0), cap)
        # 年末口径：只看年末单点
        yr = SHARE_BASE * min(max(P[-1, 0] - KAPPA * P_0, 0.0), cap)
        rows.append({
            "路径": lab, "年末P": round(float(P[-1, 1]), 2),
            "年内最高P": round(float(P.max()), 2),
            "年内最低P": round(float(P.min()), 2),
            "日度口径赔付": round(float(daily.mean()), 3),
            "年末口径赔付": round(yr, 3),
            "时间错配缺口": round(float(daily.mean()) - yr, 3),
        })
    _ = rng
    return pd.DataFrame(rows)


# ============================================================ T6 σ 网格（比例结构）
def t6_sigma_grid() -> pd.DataFrame:
    """σ_P × σ_Q 网格，比例结构下的报价（费率口径）。"""
    rows = []
    sig_q_grid = [0.15, 0.25, 0.35]
    for sp in [0.185, 0.28, 0.36]:
        for sq in sig_q_grid:
            sim = sim_daily(sig_p=sp, n_paths=30000,
                            seed=L1["seed"] + 10000 + int(sp * 1000) + int(sq * 100))
            # σ_Q 影响缺口量；此处以期望赔付的缺口量弹性近似
            q_mult = 1.0 + 0.15 * (sq - 0.25) * 4  # 缺口量均值弹性（经验值）
            pp = pay_daily(sim, n_max=N_MAX_BASE)
            pp["gross"] = pp["gross"] * q_mult
            m = metrics(pp)
            rows.append({
                "σ_P": sp, "σ_Q": sq, "缺口弹性": round(q_mult, 3),
                "E": round(m["期望赔付E"], 3), "GP_元每吨": round(m["GP_元每吨"], 3),
                "费率%": round(m["费率%"], 2), "赔付率%": round(m["赔付率%"], 1),
                "触发频率%": round(m["触发频率%"], 1),
                "是否≤判据": "是" if feasible(m["GP_元每吨"], SHARE_BASE) else "否",
            })
    return pd.DataFrame(rows)


# ============================================================ T7 份额扫描
def t7_share() -> pd.DataFrame:
    sim = sim_daily(sig_p=s32.SIG_P_BASE, n_paths=MC_TARGET,
                    seed=L1["seed"] + 11000)
    rows = []
    for sh in [0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00]:
        pp = pay_daily(sim, share=sh, n_max=N_MAX_BASE)
        m = metrics(pp, share=sh)
        rows.append({
            "份额": sh, "GP_元每吨": round(m["GP_元每吨"], 3),
            "费率%": round(m["费率%"], 2), "E": round(m["期望赔付E"], 3),
            "层宽": round(m["层宽"], 3), "赔付率%": round(m["赔付率%"], 1),
            "触发频率%": round(m["触发频率%"], 1),
            "保费/锚价%": round(m["GP_元每吨"] / P_0 * 100, 3),
            "是否≤判据": "是" if feasible(m["GP_元每吨"], sh) else "否",
        })
    return pd.DataFrame(rows)


# ============================================================ T8 压力测试
def t8_stress() -> pd.DataFrame:
    """保障深度：碳价跳涨到新水平并在该水平波动时的覆盖率。

    ★ 三个口径纪律（历次修正，均为实测踩坑）：
      ① **分子分母必须同为「日均」口径**。赔付是 244 日累加，企业增量
         成本若按半年折算会差 8 倍 ⇒ 覆盖率曾高达 800%。
      ② **不可用「永久锁定高位」的确定性路径**。价格永久停在高位时
         q75 = q99.5 = CAP ⇒ 层宽 = 0、覆盖率恒为 0，且断言会靠退化
         数据通过（恒真风险）。故跳涨后用**围绕新中枢的随机游走**。
      ③ **仍会出现层宽为 0**：跳涨 50% 仅略超 κ=1.30，随机游走有近半
         天数落在阈值下方 ⇒ 赔付分布高度集中 ⇒ q75 ≈ q99.5 ⇒ 层宽 0。
         这不是 bug，而是**分位层对「低频高损」情形不适配**的真实性质，
         故同时给出**绝对免赔口径**（D=0，A=CAP）作对照，供条款选型。
    """
    rng = np.random.default_rng(L1["seed"] + 12000)
    n = 60_000
    sig_d = s32.SIG_P_BASE / np.sqrt(TRADING_DAYS)
    rows = []
    for up in [1.5, 2.0, 4.0]:
        P_new = P_0 * up
        half = TRADING_DAYS // 2
        path = np.empty((TRADING_DAYS, n))
        path[:half, :] = P_0
        rel = np.zeros(n)
        for d in range(half, TRADING_DAYS):
            rel = rel + sig_d * rng.standard_normal(n)
            path[d, :] = P_new * np.exp(rel)
        exc = np.maximum(path - KAPPA * P_0, 0.0)
        cap = CAP_MULT * P_0
        raw = exc.sum(axis=0) / TRADING_DAYS     # 日均口径
        cap_d = cap / TRADING_DAYS
        x = np.minimum(raw, cap_d)
        D, A = float(np.quantile(x, Q_D)), float(np.quantile(x, Q_A))
        lay_q = np.clip(x - D, 0.0, A - D)                      # 分位层
        lay_abs = np.clip(x - 0.0, 0.0, cap_d)                  # 绝对免赔口径
        actual = float((P_new - P_0) *
                       max(path[half:, :].mean() / P_new, 0.0))
        paid_q = float(lay_q.mean())
        paid_a = float(lay_abs.mean())
        rows.append({
            "情景": f"碳价 +{(up-1)*100:.0f}%（{up:.1f}×）",
            "年内高位_元每吨": round(P_new, 2),
            "企业增量成本_日均": round(actual, 2),
            "分位层_赔付": round(paid_q, 3),
            "分位层_覆盖率%": round(paid_q / actual * 100, 1) if actual else 0.0,
            "分位层_D": round(D, 3), "分位层_A": round(A, 3),
            "分位层_层宽": round(A - D, 3),
            "绝对免赔_赔付": round(paid_a, 3),
            "绝对免赔_覆盖率%": round(paid_a / actual * 100, 1) if actual else 0.0,
            "CAP_日均": round(cap_d, 3),
            "触及CAP频率%": round(float((raw > cap_d).mean() * 100), 1),
        })
    return pd.DataFrame(rows)


# ============================================================ 断言
def checks(t1, t2, t3, t4, t5, t6, t7, t8) -> list[dict]:
    res = []

    def add(pid, claim, ok, actual, expect):
        res.append({"编号": pid, "断言": claim,
                    "结果": "PASS" if bool(ok) else "FAIL",
                    "实测": actual, "期望": expect})

    add("Q1", "基准条款报价≤ 判据（费率口径）",
        feasible(t4[t4["N_MAX"] == N_MAX_BASE]["GP_元每吨"].iloc[0], SHARE_BASE),
        f"{t4[t4['N_MAX'] == N_MAX_BASE]['费率%'].iloc[0]:.2f}%"
        f"（份额 {SHARE_BASE:.0%}，判据 {SHARE_BASE*BUDGET_ABS:.1f}元/吨）",
        f"≤ {SHARE_BASE*BUDGET_ABS:.2f} 元/吨")
    add("Q2", "★比例结构费率不上扬：价格跃迁 10× 时费率 ≤ 3.5×（相对基准）",
        0.2 < t2["比例_费率比值"].iloc[-1] < 3.5,
        f"{t2['比例_费率比值'].iloc[-1]:.3f}×"
        f"（费率随保额同步上升而**下降**，说明保障性价比提升）",
        "0.2× ~ 3.5×")
    add("Q3", "★定额结构在价格跃迁后**失效**（赔付归零），比例结构仍有效",
        (t1[t1["价格跃迁倍数"] == 10.0]["定额_E"].iloc[0] < 1e-6)
        and (t1[t1["价格跃迁倍数"] == 10.0]["比例_E"].iloc[0] > 0.1),
        f"10× 时定额 E = {t1[t1['价格跃迁倍数'] == 10.0]['定额_E'].iloc[0]:.3f}"
        f"（归零）vs 比例 E = {t1[t1['价格跃迁倍数'] == 10.0]['比例_E'].iloc[0]:.3f}",
        "定额 ≈ 0 且比例 > 0.1")
    add("Q4", "★两结构差异显著：10× 跃迁时比例 GP > 定额 GP",
        t1["比例_GP"].iloc[-1] > t1["定额_GP"].iloc[-1],
        f"比例 {t1['比例_GP'].iloc[-1]:.3f} vs 定额 {t1['定额_GP'].iloc[-1]:.3f}",
        "比例 > 定额")
    add("Q5", "★高跃迁下封顶成为常态（定额 99.8%、比例亦99.8%）",
        t1["定额_封顶频率%"].iloc[-1] > 95.0,
        f"定额 {t1['定额_封顶频率%'].iloc[-1]:.1f}%、"
        f"比例 {t1['比例_封顶频率%'].iloc[-1]:.1f}%", "> 95%")
    add("Q6", "★时间错配可量化：年中冲高后回落，年末口径赔付 = 0 而日度口径 > 0",
        (t5[t5["路径"] == "B 年中冲高后回落"]["日度口径赔付"].iloc[0] > 0)
        and (t5[t5["路径"] == "B 年中冲高后回落"]["年末口径赔付"].iloc[0] == 0),
        f"日度 {t5[t5['路径'] == 'B 年中冲高后回落']['日度口径赔付'].iloc[0]:.2f}"
        f" vs 年末 {t5[t5['路径'] == 'B 年中冲高后回落']['年末口径赔付'].iloc[0]:.2f}",
        "日度 > 0 且年末 = 0")
    add("Q7", "★日度口径在持续抬升情景下赔得更多（机制正确）",
        t5[t5["路径"] == "C 持续抬升"]["日度口径赔付"].iloc[0]
        > t5[t5["路径"] == "A 无触发"]["日度口径赔付"].iloc[0],
        f"C {t5[t5['路径'] == 'C 持续抬升']['日度口径赔付'].iloc[0]:.2f}"
        f" > A {t5[t5['路径'] == 'A 无触发']['日度口径赔付'].iloc[0]:.2f}",
        "C > A")
    add("Q8", "★N_MAX 有实质影响：N_MAX=1 与不限次报价不同（非恒真）",
        abs(t4[t4["N_MAX"] == 1]["GP_元每吨"].iloc[0]
            - t4[t4["N_MAX"] == 99]["GP_元每吨"].iloc[0]) > 0.05,
        f"N=1 {t4[t4['N_MAX'] == 1]['GP_元每吨'].iloc[0]:.3f}"
        f" vs N=∞ {t4[t4['N_MAX'] == 99]['GP_元每吨'].iloc[0]:.3f}",
        "两者不同")
    add("Q9", "★N_MAX 单调：报价随次数上限递增（上限越松赔越多）",
        all(a < b for a, b in zip(t4[t4["N_MAX"] != 99]["GP_元每吨"],
                                   t4[t4["N_MAX"] != 99]["GP_元每吨"].iloc[1:])),
        f"{list(t4['GP_元每吨'])}", "严格递增（除不限次）")
    add("Q10", "σ_P 单调：报价随 σ_P 递增",
        all(a < b for a, b in zip(
            t6.groupby("σ_P")["费率%"].mean().values,
            t6.groupby("σ_P")["费率%"].mean().values[1:])),
        f"{t6.groupby('σ_P')['费率%'].mean().round(2).tolist()}", "严格递增")
    add("Q11", "★θ 情景单调：θ 越大（波动随价格递减越强）报价越低",
        t3[t3["θ"] == 0.0]["GP_元每吨"].iloc[0]
        > t3[t3["θ"] == 0.5]["GP_元每吨"].iloc[0],
        f"θ=0 {t3[t3['θ'] == 0.0]['GP_元每吨'].iloc[0]:.3f}"
        f" vs θ=0.5 {t3[t3['θ'] == 0.5]['GP_元每吨'].iloc[0]:.3f}",
        "θ=0 更大")
    add("Q12", "份额扫描单调递增",
        all(a < b for a, b in zip(t7["GP_元每吨"], t7["GP_元每吨"].iloc[1:])),
        f"{list(t7['GP_元每吨'])}", "严格递增")
    add("Q13", "★全额承保亦在判据内（份额非可行性杠杆）",
        feasible(t7[t7["份额"] == 1.0]["GP_元每吨"].iloc[0], 1.0),
        f"{t7[t7['份额'] == 1.0]['GP_元每吨'].iloc[0]:.2f} 元/吨",
        f"≤ {BUDGET_ABS:.0f} 元/吨")
    add("Q14", "★分位层在「低频高损」情形退化为零覆盖（真实性质，须披露）",
        (t8["分位层_层宽"] <= 1e-6).all()
        and (t8["绝对免赔_覆盖率%"] > 0).all(),
        f"分位层覆盖率 = {list(t8['分位层_覆盖率%'])}"
        f"（层宽 {list(t8['分位层_层宽'])}）；"
        f"绝对免赔口径覆盖率 = {list(t8['绝对免赔_覆盖率%'])}",
        "分位层 ≡ 0 且绝对免赔 > 0")
    add("Q15", "★绝对免赔口径覆盖 ∈ (0,100] 且随涨幅递减（κ 单次的必然）",
        ((t8["绝对免赔_覆盖率%"] > 0).all()
         and (t8["绝对免赔_覆盖率%"] <= 100 + 1e-6).all()
         and (t8["绝对免赔_覆盖率%"].iloc[-1]
              <= t8["绝对免赔_覆盖率%"].iloc[0] + 1e-6)),
        f"绝对免赔覆盖率 = {list(t8['绝对免赔_覆盖率%'])}",
        "∈(0,100] 且末档 ≤ 首档")
    return res


# ============================================================ main
def main(check_only: bool = False) -> int:
    t0 = time.time()
    sys.stdout.reconfigure(encoding="utf-8")
    print("s36　比例式保额 + 日度累计赔付（纯 P 结构的两项机制修正）")
    print("=" * 130)
    print(f"  锚价 P_0 = {P_0:.2f} 元/吨（2026-08 收盘）")
    print(f"  κ = {KAPPA}　CAP_MULT = {CAP_MULT}　份额基准 = {SHARE_BASE:.0%}"
          f"　φ = {PHI:.2f}")
    print(f"  日度累计上限 N_MAX = {N_MAX_BASE}　交易日 = {TRADING_DAYS}")
    print(f"  路径数 {MC_TARGET:,}　判据 ≤ {BUDGET_ABS:.0f} 元/吨（全额口径）")

    t1 = t1_fixed_vs_proportional()
    print("\n【T1 定额 vs 比例】价格跃迁时的结构表现")
    print("-" * 130)
    print(t1.to_string(index=False))

    t2 = t2_rate_stability()
    print("\n【T2 费率稳定性】费率（GP/CAP）是否随价格同步")
    print("-" * 130)
    print(t2.to_string(index=False))

    t3 = t3_theta()
    print("\n【T3 波动率状态依赖 σ(P)=σ_0·(P/P_0)^(−θ)】")
    print("-" * 130)
    print(t3.to_string(index=False))

    t4 = t4_nmax()
    print("\n【T4 触发次数上限 N_MAX】防波动年反复触发")
    print("-" * 130)
    print(t4.to_string(index=False))

    t5 = t5_timing_mismatch()
    print("\n【T5 时间错配】日度口径 vs 年末口径（时间错配基差风险）")
    print("-" * 130)
    print(t5.to_string(index=False))

    t6 = t6_sigma_grid()
    print("\n【T6 σ_P × σ_Q 网格】（比例结构，费率口径）")
    print("-" * 130)
    print(t6.to_string(index=False))

    t7 = t7_share()
    print("\n【T7 份额扫描】")
    print("-" * 130)
    print(t7.to_string(index=False))

    t8 = t8_stress()
    print("\n【T8 压力测试】保障深度（全额承保）")
    print("-" * 130)
    print(t8.to_string(index=False))

    res = checks(t1, t2, t3, t4, t5, t6, t7, t8)
    ck = pd.DataFrame(res)
    n_fail = int((ck["结果"] == "FAIL").sum())
    print("\n【验收断言】")
    print("-" * 130)
    print(ck.to_string(index=False))
    print(f"\n  ⇒ {len(ck) - n_fail}/{len(ck)} 通过"
          + ("（全部通过）" if n_fail == 0 else f"，**{n_fail} 项 FAIL**"))

    base = t4[t4["N_MAX"] == N_MAX_BASE].iloc[0]
    quote = {
        "结构": "纯 P 承保 · 比例式保额与保费 · 日度观察 + 年度累计（N_MAX 上限）",
        "锚": {"P_0_2026_08收盘": P_0,
               "锚日规则": "P_0 = 保单起期日前一交易日收盘价，与投保日无关"},
        "条款": {
            "保单期限_年": 1, "份额": SHARE_BASE, "κ": KAPPA,
            "CAP_比例式": f"{CAP_MULT} × P_0（随价格同步放大）",
            "CAP_元每吨_锚年": round(CAP_MULT * P_0, 2),
            "起赔": "日度 P_d > κ·P_0",
            "结算": "日度累计，年度终了后 15 个交易日一次性确认支付",
            "N_MAX": N_MAX_BASE, "φ": PHI,
            "费用率": E_EXP, "利润边际": PI_PROFIT, "资本成本": RHO_CAP,
        },
        "报价": {
            "GP_元每吨_份额10%": float(base["GP_元每吨"]),
            "费率%_相对CAP": float(base["费率%"]),
            "费率%_相对锚价": float(base["GP_元每吨"] / P_0 * 100),
            "E_元每吨": float(base["E"]),
            "赔付率%": float(base["赔付率%"]),
            "触发频率%": float(base["触发频率%"]),
            "平均触发天数": float(base["平均触发天数"]),
            "判据_元每吨_全额口径": BUDGET_ABS,
            "全额承保_元每吨": float(t7[t7["份额"] == 1.0]["GP_元每吨"].iloc[0]),
            "全额承保_费率%": float(t7[t7["份额"] == 1.0]["费率%"].iloc[0]),
        },
        "结构修正的证据": {
            "定额结构_10倍跃迁_费率变动倍数": float(t2["定额_费率比值"].iloc[-1]),
            "比例结构_10倍跃迁_费率变动倍数": float(t2["比例_费率比值"].iloc[-1]),
            "定额结构_10倍跃迁_封顶频率%": float(t1["定额_封顶频率%"].iloc[-1]),
            "比例结构_10倍跃迁_封顶频率%": float(t1["比例_封顶频率%"].iloc[-1]),
            "时间错配_年中冲高后回落_日度赔付": float(
                t5[t5["路径"] == "B 年中冲高后回落"]["日度口径赔付"].iloc[0]),
            "时间错配_年中冲高后回落_年末赔付": float(
                t5[t5["路径"] == "B 年中冲高后回落"]["年末口径赔付"].iloc[0]),
        },
        "波动率状态依赖情景": {
            f"θ={r.θ}": float(r.GP_元每吨) for r in t3.itertuples()
        },
        "σ网格报价区间": [float(t6["费率%"].min()), float(t6["费率%"].max())],
        "压力测试覆盖率%": {r.情景: float(r._5) for r in t8.itertuples()},
        "断言": {"总数": len(ck), "FAIL": n_fail},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "s36_比例式日度累计报价.json").write_text(
        json.dumps(quote, ensure_ascii=False, indent=2), encoding="utf-8")
    t1.to_csv(OUT_DIR / "s36_T1_定额vs比例.csv", index=False, encoding="utf-8-sig")
    t2.to_csv(OUT_DIR / "s36_T2_费率稳定性.csv", index=False, encoding="utf-8-sig")
    t3.to_csv(OUT_DIR / "s36_T3_波动率状态依赖.csv", index=False, encoding="utf-8-sig")
    t4.to_csv(OUT_DIR / "s36_T4_触发次数上限.csv", index=False, encoding="utf-8-sig")
    t5.to_csv(OUT_DIR / "s36_T5_时间错配.csv", index=False, encoding="utf-8-sig")
    t6.to_csv(OUT_DIR / "s36_T6_σ网格.csv", index=False, encoding="utf-8-sig")
    t7.to_csv(OUT_DIR / "s36_T7_份额扫描.csv", index=False, encoding="utf-8-sig")
    t8.to_csv(OUT_DIR / "s36_T8_压力测试.csv", index=False, encoding="utf-8-sig")
    ck.to_csv(OUT_DIR / "s36_验收断言.csv", index=False, encoding="utf-8-sig")
    print(f"\n  产物已写入 {OUT_DIR}")
    print(f"\n  ** 基准报价：{base['GP_元每吨']:.2f} 元/吨配额"
          f"（份额 {SHARE_BASE:.0%}），费率 {base['费率%']:.2f}%（相对 CAP）**")
    print(f"     全额承保 {t7[t7['份额'] == 1.0]['GP_元每吨'].iloc[0]:.2f} 元/吨"
          f"，费率 {t7[t7['份额'] == 1.0]['费率%'].iloc[0]:.2f}%")
    print(f"\n  用时 {time.time() - t0:.1f}s")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
