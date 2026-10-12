# -*- coding: utf-8 -*-
"""
s4_pricing.py —— L1 长期险定价引擎（v2：m 参数化 + 标度区间定价）

v1 的两个缺陷（本版修正）
-------------------------
✗ 缺陷一：指数被写成 I_t = I^ub_t × R_t，R_t ∈ (0,1] 且单调回复。
  后果：指数**从期初最大值起单调下降** → 任何虚值看涨期权必然零价值，
  L1b 赔付恒为 0（v1 实测即如此）。这不是参数没调好，是**结构设计错误**：
  「指数只会跌」的产品在数学上不可能有正保费。

✓ v2 改为按**缺口倍数**参数化：
      I_t = P_t × m_t ,  m_t = m_floor + (m_0 − m_floor)·2^(−(t−2026)/half_life)
  指数 = 碳价 × 缺口倍数。碳价按情景上行，缺口倍数向「市场成熟地板」收敛，
  两者方向相反 → 指数可以上行，看涨结构非退化。且 m 正是企业真正怕的东西
  （「想减排但减排成本比碳价还贵」），经济含义比「成本曲线下移比例 R」更清楚。

✗ 缺陷二：单一 γ₁ = 0 标度约定 → 单一保费点值。
  后果：把 190.53 元/吨这个「合法区间最低端 5.8% 处」的点值当锚，
  低估承保成本最高 2.92 倍。

✓ v2 按标度档分档定价，输出**保费区间**与**审慎上端**。

标度模糊性的处置原则（不可省）
------------------------------
γ₁ ∈ W* 与 β 都是方向向量的**约定性自由度**，数据不可识别，deliver2 只见
合法域、未见概率分布 → 属**模糊性（ambiguity）**而非**风险（risk）**。
故本引擎：
  · 不对 γ₁ 取先验做「求分位数」的伪概率运算；
  · 保费以**档位网格**呈现，审慎报价取上端；
  · 条款须**冻结并披露标度约定**，否则事后可选 γ₁ 操纵指数与赔付。

两层结构：风险的归 Monte Carlo，模糊性的归档位网格
--------------------------------------------------
  第一层（风险 / aleatory）：固定一个标度档，MC 模拟碳价、缺口倍数路径、
      承保吨数的联合分布 → 该档的保费与分位。
  第二层（模糊 / ambiguity）：换档重算，得到保费**区间**。
  两层不得混为一谈 —— 把档位当先验去平均，等于用一个假的概率分布
  掩盖「我们不知道」。

产品结构对比：水平式 vs 重定基式
--------------------------------
  · 设计A 水平式：起赔价 K 以**绝对元/吨**固定（K = κ × 参考水平）。
    标度模糊性直接改变合约的**虚实值状态**，保费区间极宽，事实上不可报价。
  · 设计B 重定基式：每个续保期把起赔价重设为 κ × 当期指数。
    把「水平模糊」转成「相对风险」，区间显著收窄 → 可报价。
  两者之差 = **定期重定价条款的价值**，这是 s2b「档位死亡」结论的产品落点。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from _config import GAP_MULTIPLE, L1, OUT, PRICE_PATH, SCENARIOS
from s0_gamma_window import ROOTS
from s2_nowcast import build_national_inputs, national_emission_path
from s3_term import deflator_to

RNG_SEED = L1["seed"]
N_PATH = L1["mc_paths"]
DF_R = L1["discount"]

START = 2026
TERMS = [3, 5, 10, 15]
KAPPA = [1.15, 1.30, 1.50, 1.80]      # 起赔倍数：K = κ × 起期基准水平
KAPPA_BASE = 1.30
RENEW_PERIODS = [3, 5]                 # 设计B 的续保周期
CAP_MULT = 2.0                         # 每吨赔付上限 = CAP_MULT × MAC_REF

# 参考水平：论文口径 γ₁=0 的 2026 名义 MAC（公开披露的基准数）
MAC_REF_1995 = 190.5306
MAC_REF = MAC_REF_1995 * deflator_to(2026)     # ≈ 413.3 元/吨
CAP_AMOUNT = CAP_MULT * MAC_REF                # ≈ 826.6 元/吨（每吨赔付上限）
P_2026 = float(PRICE_PATH["S1_基准"]["2026"])

# ------------------------------------------------------------ 标度档网格
# (标签, γ₁, β, 是否做 M1 结构外推)
# β=1 档与 s2b 同档，便于交叉核验：段1 左端（最保守）→ 段2 中点（最宽松）；
# β≠1 两档取自 s0c 二维模糊域的上下确界点 —— 用户提示「参数 g 的取值影响
# DDF 测算」，β 即方向向量的形状自由度，只扫 γ₁ 会低估上界 3.0%。
# β≠1 档只做 M2（价格平移）口径：M1 的闭式外推需要 β 一般的系数注入，
# 本版未接入（见 s1 的 dD_from_inputs 已备好接口），故如实标「—」。
TIERS_RAW = [
    ("段1左端_最保守", ROOTS[0] + 1e-4, 1.0, True),
    ("段1偏左", -0.25, 1.0, True),
    ("段1中段", -0.15, 1.0, True),
    ("段1偏右", -0.05, 1.0, True),
    ("论文口径", 0.0, 1.0, True),
    ("段1右端_最宽松", 0.06, 1.0, True),
    ("段2中点", 0.5 * (ROOTS[2] + ROOTS[3]), 1.0, True),
    ("β2.0_上确界", -0.1043, 2.0, False),
    ("β0.6_下确界", 0.0957, 0.6, False),
]
# 成本曲线口径：
#   M2 = 价格平移（实际成本曲线静止）—— s3 认定的定价主锚
#   M1 = 叠加结构外推（u 空间）—— 唯一有模型依据的下移路径，但脆弱且会越界
CALIBERS = ["M2_价格平移", "M1_结构外推"]

# ------------------------------------------------------------ 随机过程参数
SIG_P = 0.28          # 碳价对数波动率（年化）
KAPPA_P = 0.35        # 碳价均值回复速度
SIG_M_PATH = 0.15     # 缺口倍数的路径扰动
KAPPA_M = 0.30        # 缺口倍数扰动回复速度（回复到自身中枢，非回复到 1）
SIG_M_LEVEL = 0.12    # **档内**剩余的弱水平不确定性（数据修订等），刻意取小
SIG_Q = 0.25          # 承保吨数对数波动率
RHO_PM = 0.45         # 碳价 ↔ 缺口倍数：碳价高 → 减排需求大 → MAC 上抬
RHO_PQ = -0.40        # 碳价 ↔ 缺口吨数：碳价高 → 多减排 → 缺口小
RHO_MQ = 0.30         # 缺口倍数 ↔ 缺口吨数：减排成本高 → 减不动 → 缺口大
# 成本曲线年下移率（在缺口倍数中枢上叠加）。默认 0（不可识别，不假设）。
M_DRIFT_SENS = [0.0, 0.02, 0.04]


def _corr_chol() -> np.ndarray:
    C = np.array([[1.0, RHO_PM, RHO_PQ],
                  [RHO_PM, 1.0, RHO_MQ],
                  [RHO_PQ, RHO_MQ, 1.0]])
    w, V = np.linalg.eigh(C)
    if (w < -1e-9).any():
        raise ValueError(f"相关矩阵非正定，最小特征值 {w.min():.4f}")
    return V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T


CHOL = _corr_chol()


# ============================================================ 标度档标定

def _lp_cache() -> dict:
    """逐档解 LP（独立复算，见 s0/s0c），返回 MAC / 系数 / 越界年。"""
    from s0_gamma_window import run_at
    from s1_ddf import mac_scaled_from_u
    _, gE, gEF = national_emission_path()
    cache = {}
    for tag, g1, beta, do_m1 in TIERS_RAW:
        r = run_at(g1, beta=beta)
        if not r.get("ok"):
            print(f"  ⚠ 标度档 {tag} (γ₁={g1:.5f}, β={beta}) LP 无法求解，跳过")
            continue
        coef = r["_coef"]
        mac22 = float(r["MAC_2022均值"])
        t = {"tag": tag, "γ1": float(g1), "β": float(beta),
             "b1": float(r["b1"]), "MAC2022_1995价": mac22,
             "MAC2026_1995价_M1": None, "越界首年": None,
             "coef": coef, "do_M1": bool(do_m1)}

        if do_m1:
            # --- M1 口径：u 空间外推路径（自洽标度：2022 精确等于省级还原口径）
            nat = build_national_inputs(SCENARIOS["S1_基准"], gE, gEF, coef=coef)
            nat = nat.copy()
            nat["MAC_scaled_u"] = mac_scaled_from_u(nat["u"])
            ni = nat.set_index("年份")
            u22 = float(ni.loc[2022, "u"])
            scale_eff = mac22 / float(mac_scaled_from_u(u22))
            bad = nat[(nat["年份"] >= 2023) & (nat["u"] <= 0.005)]
            t["MAC2026_1995价_M1"] = float(ni.loc[2026, "MAC_scaled_u"]) * scale_eff
            t["越界首年"] = int(bad["年份"].min()) if len(bad) else None
            t["u2022"] = u22
        cache[tag] = t
    return cache


def m0_of(tier: dict, caliber: str):
    """2026 年缺口倍数起点 m₀ = MAC_2026(名义) / P_2026；口径不可用时返回 None。"""
    mac95 = (tier["MAC2022_1995价"] if caliber == "M2_价格平移"
             else tier["MAC2026_1995价_M1"])
    if mac95 is None:
        return float("nan")
    return mac95 * deflator_to(2026) / P_2026


# ============================================================ 路径与模拟

def _price_central(scen_key: str, years: list) -> np.ndarray:
    pp = PRICE_PATH[scen_key]
    ks = sorted(int(k) for k in pp)
    xs = np.array([2026] + ks, float)
    ys = np.array([float(pp["2026"])] + [float(pp[str(k)]) for k in ks], float)
    return np.interp(years, xs, ys)


def _m_central(m0: float, scen_key: str, years: list) -> np.ndarray:
    spec = GAP_MULTIPLE[scen_key]
    dt = np.maximum(np.array(years, float) - START, 0.0)
    return spec["floor"] + (m0 - spec["floor"]) * \
        np.exp(-np.log(2) / spec["half_life"] * dt)


def simulate(scen_key: str, tier: dict, caliber: str, n_paths: int, seed: int,
             m_drift: float = 0.0, sig_p: float | None = None) -> dict:
    """
    单档蒙特卡洛：给出 (年份 × 路径) 的碳价、缺口倍数、指数、承保吨数。

    ⚠ n_paths 必须传路径数（N_PATH）。v1 把「最长期限」传到了这个位置，
      实际只跑了 10 条路径，中位数与赔付率全部是抽样噪声 —— 这是 v1
      「保费在期限间非单调」的直接原因，随本版修正。

    模糊性不进 MC —— m₀ 由标度档给定为**确定值**；MC 只处理该档内部的
    真实风险（价格波动、路径扰动、吨数波动）。
    """
    rng = np.random.default_rng(seed)
    yrs = list(range(START, START + max(TERMS) + 2))
    nyr = len(yrs)
    n = int(n_paths)
    sp = SIG_P if sig_p is None else float(sig_p)

    P_c = _price_central(scen_key, yrs)
    m0 = m0_of(tier, caliber)
    m_c = _m_central(m0, scen_key, yrs)
    m_c = m_c * np.exp(-m_drift * np.maximum(np.arange(nyr) - 1, 0))

    Z = rng.standard_normal((3, nyr, n))
    Z = np.einsum("ij,jtn->itn", CHOL, Z)
    eP, eM, eQ = Z[0], Z[1], Z[2]

    # 碳价：对数 OU 围绕情景中枢
    lnP = np.empty((nyr, n))
    lnP[0] = np.log(P_c[0]) + sp * eP[0]
    for t in range(1, nyr):
        lnP[t] = (lnP[t - 1] + KAPPA_P * (np.log(P_c[t]) - lnP[t - 1])
                  + sp * eP[t])
    P = np.exp(lnP)

    # 缺口倍数：对数 OU 围绕自身中枢；档内剩余水平不确定仅通过初值注入
    lnm = np.empty((nyr, n))
    lnm[0] = np.log(m_c[0]) + SIG_M_LEVEL * eM[0]
    for t in range(1, nyr):
        lnm[t] = (lnm[t - 1] + KAPPA_M * (np.log(m_c[t]) - lnm[t - 1])
                  + SIG_M_PATH * eM[t])
    M = np.exp(lnm)
    I = P * M

    # 承保吨数：缺口率上升 + 对数扰动
    g = L1["gap_rate_0"] + L1["gap_tighten"] * np.arange(nyr)
    Q = np.exp(np.log(g)[:, None] + SIG_Q * eQ)
    Q = Q / Q[0]

    return {"years": yrs, "P": P, "M": M, "I": I, "Q": Q,
            "P_c": P_c, "m_c": m_c, "m0": m0}


# ============================================================ 定价

def _pv(payoff: np.ndarray, n: int) -> np.ndarray:
    df = 1.0 / (1.0 + DF_R) ** np.arange(n + 1)
    return (payoff[:n + 1] * df[:, None]).sum(axis=0)


def price_level(scen_key: str, tag: str, base: str, n: int, kappa: float,
                sim: dict, mac_ref: float) -> dict:
    """
    设计A　水平式：起赔价按**绝对元/吨**固定，不随模拟重定基。
      L1b: K = κ × MAC_REF（2026 名义，论文口径公开基准）
      L1a: K = κ × P_2026
    每吨赔付上限 CAP_AMOUNT，限额 SI = CAP_AMOUNT × n —— 故赔付率上限 100%，
    费率上限 = (1+安全边际)/(1−费用率) ≈ 153%。
    """
    P, I, Q = sim["P"], sim["I"], sim["Q"]
    Y = I if base == "L1b_指数口径" else P
    K = kappa * (mac_ref if base == "L1b_指数口径" else P_2026)
    pay = np.minimum(np.maximum(Y - K, 0.0), CAP_AMOUNT) * Q
    pv = _pv(pay, n)
    SI = CAP_AMOUNT * n
    rate = float(pv.mean() / SI * (1 + L1["safety_margin"]) / (1 - L1["expense_ratio"]))
    return {
        "设计": "A_水平式", "情景": scen_key, "标度档": tag, "赔付口径": base,
        "期限": n, "起赔倍数κ": kappa, "起赔价K": round(K, 1),
        "起期指数I0中位数": round(float(np.median(I[0])), 1),
        "虚实值": ("实值" if float(np.median(I[0])) > K else "虚值"),
        "期望赔付现值": round(float(pv.mean()), 1),
        "赔付率_期望%": round(float(pv.mean() / SI * 100), 2),
        "出险概率%": round(float((pv > 0).mean() * 100), 1),
        "毛保费率_占限额%": round(rate * 100, 2),
    }


def price_rebased(scen_key: str, tag: str, n: int, period: int, kappa: float,
                  sim: dict, mac_ref: float) -> dict:
    """
    设计B　重定基式：每 period 年把起赔价重设为 κ × 当期指数。

    把标度模糊从「水平」转成「相对」：期初指数水平无论落在区间何处，
    合约的虚实值状态都由同一比例决定，故跨档保费离散度大幅收窄。
    代价：若指数水平阶跃上移，阈值随之上抬，**保障被动削弱**（须披露）。

    限额与设计A 同口径（SI = CAP_AMOUNT × n），保证两者费率可比。
    """
    I, Q = sim["I"], sim["Q"]
    SI = CAP_AMOUNT * n
    total = np.zeros(I.shape[1])
    yr = 0
    while yr < n:
        seg = min(period, n - yr)
        K = kappa * I[yr]                      # 每段期初重定基（逐路径）
        pay = np.minimum(np.maximum(I[yr:yr + seg + 1] - K[None, :], 0.0),
                         CAP_AMOUNT) * Q[yr:yr + seg + 1]
        df = 1.0 / (1.0 + DF_R) ** (np.arange(seg + 1) + yr)
        total += (pay * df[:, None]).sum(axis=0)
        yr += seg
    rate = float(total.mean() / SI * (1 + L1["safety_margin"])
                 / (1 - L1["expense_ratio"]))
    return {
        "设计": f"B_重定基({period}y)", "情景": scen_key, "标度档": tag,
        "赔付口径": "L1b_指数口径", "期限": n, "起赔倍数κ": kappa,
        "起赔价K": f"κ×当期指数（每{period}年重定基）",
        "起期指数I0中位数": round(float(np.median(I[0])), 1),
        "虚实值": "逐期重定基",
        "期望赔付现值": round(float(total.mean()), 1),
        "赔付率_期望%": round(float(total.mean() / SI * 100), 2),
        "出险概率%": round(float((total > 0).mean() * 100), 1),
        "毛保费率_占限额%": round(rate * 100, 2),
    }


# ======================================================================== 主

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 78)
    print("s4　L1 长期险定价引擎（v2）")
    print("=" * 78)
    print(f"  参考水平 MAC_REF = {MAC_REF:.1f} 元/吨（2026 名义，论文口径公开基准）")
    print(f"  碳价起点 P_2026 = {P_2026:.2f} 元/吨")
    print(f"  随机过程: σ_P={SIG_P}  σ_m路径={SIG_M_PATH}  σ_m水平(档内)={SIG_M_LEVEL}"
          f"  σ_Q={SIG_Q}")
    print(f"            相关 ρ(P,m)={RHO_PM}  ρ(P,Q)={RHO_PQ}  ρ(m,Q)={RHO_MQ}")
    print(f"  路径数 {N_PATH:,}　贴现 {DF_R:.1%}　费用率 {L1['expense_ratio']:.0%}"
          f"　安全边际 {L1['safety_margin']:.0%}")

    print()
    print("--- 标度档标定（逐档重解 LP，含 β 形状自由度与 M1 结构外推口径）---")
    tiers = _lp_cache()
    rows = []
    for tag, t in tiers.items():
        m2v = m0_of(t, "M2_价格平移")
        m1v = m0_of(t, "M1_结构外推")
        rows.append({
            "标度档": tag, "γ1": round(t["γ1"], 5), "β": t["β"],
            "MAC2022_1995价": round(t["MAC2022_1995价"], 1),
            "MAC2026名义": round(t["MAC2022_1995价"] * deflator_to(2026), 0),
            "m0_M2": round(m2v, 3),
            "m0_M1": round(m1v, 3) if np.isfinite(m1v) else "—",
            "M1越界首年": t["越界首年"] if t["越界首年"] else "—",
        })
    d = pd.DataFrame(rows)
    print(d.to_string(index=False))
    m2_lo, m2_hi = float(d["m0_M2"].min()), float(d["m0_M2"].max())
    m1v = pd.to_numeric(d["m0_M1"], errors="coerce").dropna()
    print(f"\n  m₀（M2 口径）∈ [{m2_lo:.2f}, {m2_hi:.2f}]；"
          f"（M1 口径）∈ [{m1v.min():.2f}, {m1v.max():.2f}]")
    print(f"  ⇒ 合成可用区间 **[1.0, {m2_hi:.2f}]**（m<1 无经济含义，下界取 1）")
    print("  β 轴的作用：仅扫 γ₁（β=1）得上界 555.80 元/吨，放开 β 后为 "
          f"{d['MAC2022_1995价'].max():.2f} 元/吨，"
          f"上修 {(d['MAC2022_1995价'].max() / 555.801 - 1) * 100:.2f}%。")
    print("  M1 口径的降幅（论文档 2022→2026 −68.6%）已超过模型自身脆弱性的两倍"
          "\n  （资本 +5% → MAC ±20%，见 s3 第二节），无法与噪声区分，故不作主锚。")

    # ---------------------------------------------------------- 模拟与定价
    sims, base_seed = {}, RNG_SEED
    keys = [(scen, tag, cal) for scen in SCENARIOS for tag, t in tiers.items()
            for cal in CALIBERS if cal == "M2_价格平移" or t["do_M1"]]
    for i, (scen, tag, cal) in enumerate(keys):
        sims[(scen, tag, cal)] = simulate(scen, tiers[tag], cal,
                                          N_PATH, base_seed + i * 977)

    NA = []
    for (scen, tag, cal), sim in sims.items():
        for base in ("L1b_指数口径", "L1a_市价口径"):
            for n in TERMS:
                for kap in KAPPA:
                    r = price_level(scen, tag, base, n, kap, sim, MAC_REF)
                    r["成本曲线口径"] = cal
                    NA.append(r)
    A = pd.DataFrame(NA)

    NB = []
    for (scen, tag, cal), sim in sims.items():
        if cal != "M2_价格平移":
            continue                # 设计B 只在主锚口径上算，避免表过大
        for p in RENEW_PERIODS:
            for n in TERMS:
                if n < p:
                    continue
                r = price_rebased(scen, tag, n, p, KAPPA_BASE, sim, MAC_REF)
                r["成本曲线口径"] = cal
                NB.append(r)
    B = pd.DataFrame(NB)

    # ---------------------------------------------------------- 第一节
    print()
    print("=" * 78)
    print("第一节　主结论：可报价方案（设计B 重定基式，κ=1.30，M2 口径）")
    print("=" * 78)
    b = B[(B["起赔倍数κ"] == KAPPA_BASE) & (B["情景"] == "S1_基准")]
    pivb = b.pivot_table(index="标度档", columns=["设计", "期限"],
                         values="毛保费率_占限额%")
    print("  毛保费率（占名义限额 %）：")
    print(pivb.round(1).to_string())
    bb = (b.groupby(["设计", "期限"])["毛保费率_占限额%"]
          .agg(["min", "median", "max"]).round(1))
    bb = bb.rename(columns={"min": "区间下端", "median": "中位", "max": "审慎上端"})
    bb["区间宽度_倍"] = (bb["审慎上端"] / bb["区间下端"].replace(0, np.nan)).round(2)
    print("\n  跨标度档的保费区间（S1 基准）：")
    print(bb.drop(columns=["区间宽度_倍"]).to_string())
    print("\n  ⇒ 重定基后，同一规格的保费区间收窄至数倍以内 —— 可以报价了。")
    print("    报价规则：取**审慎上端**（最保守标度档）作为挂牌费率，")
    print("    并在条款中冻结标度约定，事后方可按该约定复核。")

    # ---------------------------------------------------------- 第二节
    print()
    print("=" * 78)
    print("第二节　为什么水平式（设计A）不可报价：水平模糊 = 虚实值状态模糊")
    print("=" * 78)
    s = A[(A["起赔倍数κ"] == KAPPA_BASE) & (A["赔付口径"] == "L1b_指数口径")
          & (A["成本曲线口径"] == "M2_价格平移")]
    band = (s.groupby(["情景", "期限"])["毛保费率_占限额%"]
            .agg(["min", "median", "max"]).round(1))
    band = band.rename(columns={"min": "区间下端", "median": "中位",
                                "max": "审慎上端"})
    band["上端/下端"] = (band["审慎上端"] / band["区间下端"].replace(0, np.nan)
                     ).round(2)
    print("  跨标度档的保费区间（同一合约规格，起赔价固定为 "
          f"{KAPPA_BASE * MAC_REF:.0f} 元/吨）：")
    print(band.to_string())
    mon = s[s["情景"] == "S1_基准"][["标度档", "期限", "起赔价K", "虚实值",
                                 "起期指数I0中位数", "出险概率%",
                                 "毛保费率_占限额%"]]
    print("\n  5 年期明细（S1 基准）：")
    print(mon[mon["期限"] == 5].to_string(index=False))
    r5 = s[(s["情景"] == "S1_基准") & (s["期限"] == 5)]
    mn, mx = float(r5["起期指数I0中位数"].min()), float(r5["起期指数I0中位数"].max())
    print(f"\n  起期指数水平跨档 {mn:.0f} – {mx:.0f} 元/吨，而 K 固定为 "
          f"{KAPPA_BASE * MAC_REF:.0f} 元/吨")
    print(f"  ⇒ 同一张保单，在不同标度约定下从**虚值**变成**实值**"
          f"（起期指数/K 最高 {mx / (KAPPA_BASE * MAC_REF):.1f} 倍），"
          f"\n     保费区间宽达 {float(band.loc[('S1_基准', 5), '上端/下端']):.0f} 倍。")
    print("     水平模糊无法用**调整费率**吸收 —— 它改变的是合约的虚实值状态，")
    print("     相当于把一张保单卖成了另一张保单。故水平式必须辅以标度冻结条款，")
    print("     否则事后可选标度约定操纵赔付。")

    # ---------------------------------------------------------- 第三节
    print()
    print("=" * 78)
    print("第三节　重定价条款的价值：模糊性收窄的量化")
    print("=" * 78)
    aa = (s[s["情景"] == "S1_基准"].groupby("期限")["毛保费率_占限额%"]
          .agg(["min", "max"]).round(1))
    aa["区间宽度_倍"] = (aa["max"] / aa["min"].clip(lower=1e-9)).round(2)
    print("  设计A 水平式：")
    print(aa.to_string())
    print("\n  设计B 重定基式：")
    print(bb.drop(columns=["区间宽度_倍"]).to_string())
    n_show = 5
    wa = float(aa.loc[n_show, "区间宽度_倍"])
    wb = float(bb.loc[("B_重定基(3y)", n_show), "区间宽度_倍"])
    print(f"\n  以 {n_show} 年期为例：设计A 区间宽 {wa:.1f} 倍 → 设计B(3y 续保) "
          f"{wb:.1f} 倍，收窄 {wa / wb:.1f} 倍。")
    print("  机制：重定基把「指数水平的模糊」转成「指数相对变动的风险」，")
    print("        水平档差在比例中被约掉，只剩各档**收敛速度**的差异。")
    print("  代价（必须随条款披露）：若指数水平阶跃上移，阈值随之上抬，")
    print("        被保险人拿到的保障被动削弱 —— 用**保障强度**换**可报价性**。")

    # ---------------------------------------------------------- 第四节
    print()
    print("=" * 78)
    print("第四节　可承保期限：结构模型的有效期决定重定价频率")
    print("=" * 78)
    vt = d[["标度档", "γ1", "β", "M1越界首年"]].copy()
    vt["越界年_数值"] = pd.to_numeric(vt["M1越界首年"], errors="coerce")
    vt["结构模型有效期_年"] = vt["越界年_数值"] - 2026
    vt["固定价格可承保至"] = vt["越界年_数值"].apply(
        lambda y: f"{int(y) - 1}年末" if pd.notna(y) else "—")
    vt = vt.drop(columns=["越界年_数值"])
    print(vt.to_string(index=False))
    ys = pd.to_numeric(d["M1越界首年"], errors="coerce").dropna().astype(int)
    print(f"\n  u 越界首年跨档 ∈ [{ys.min()}, {ys.max()}]，即结构模型有效期 "
          f"{ys.min() - 2026}–{ys.max() - 2026} 年（仅 β=1 档可算，故区间是下界）。")
    print("  u 越界后结构模型失去识别（不再满足单调性/可行性），")
    print("  而**成本曲线的下移速率**恰恰只能由该模型给出 → 长端指数水平")
    print("  缺乏模型依据。故：")
    print(f"    · 固定价格承诺的最长期限 = 最保守档的有效期 = "
          f"{ys.min() - 2026} 年（{ys.min() - 1}年末前）")
    print("    · 更长期限必须走**定期重定价 + 标度约定冻结 + 越界触发重估**"
          "\n      的三段式条款，而不是一次性锁价。")
    print("    · 每期续保时需重新诊断：u 是否越界、标度约定是否仍合法。")

    # ---------------------------------------------------------- 第五节
    print()
    print("=" * 78)
    print("第五节　两个赔付口径的费率差（保险的比较优势量化）")
    print("=" * 78)
    cmp_ = A[(A["起赔倍数κ"] == KAPPA_BASE) & (A["期限"] == 5)
             & (A["成本曲线口径"] == "M2_价格平移")]
    cp = cmp_.pivot_table(index=["情景", "标度档"], columns="赔付口径",
                          values="毛保费率_占限额%")
    cp["差(指数−市价)"] = (cp["L1b_指数口径"] - cp["L1a_市价口径"]).round(1)
    print(cp.round(1).to_string())
    print("  L1a（市价口径）可由碳期货/期权近似复制，保险的比较优势有限；")
    print("  L1b（指数口径）挂钩 MAC，**无任何金融市场可复制** ——")
    print("  这是保险唯一不可替代的位置，也是本项目的主张。")

    # ---------------------------------------------------------- 第六节
    print()
    print("=" * 78)
    print("第六节　不可识别参数的定价影响（成本曲线年下移率）")
    print("=" * 78)
    srows = []
    for scen in SCENARIOS:
        for dr in M_DRIFT_SENS:
            sim = simulate(scen, tiers["论文口径"], "M2_价格平移",
                           N_PATH, base_seed, m_drift=dr)
            r = price_level(scen, "论文口径", "L1b_指数口径", 5, KAPPA_BASE,
                            sim, MAC_REF)
            srows.append({"情景": scen, "成本曲线年下移率": dr,
                          "缺口倍数2026": round(sim["m0"], 2),
                          "缺口倍数2031": round(float(np.mean(
                              sim["M"][5])), 2),
                          "期望赔付%": r["赔付率_期望%"],
                          "毛保费率%": r["毛保费率_占限额%"]})
    S = pd.DataFrame(srows)
    print(S.to_string(index=False))
    print("  该参数在 s3 中被判定为**不可识别**（u 越界后失去约束）。")
    print("  引擎默认取 0，并把该参数作为共保/再保谈判的显式变量，而非藏进点估计。")

    print()
    print("  碳价波动率的定价影响（σ_P 是可观测参数，但市场样本短，故给出带宽）：")
    vrows = []
    for sig in (0.20, 0.28, 0.36):
        sim = simulate("S1_基准", tiers["论文口径"], "M2_价格平移",
                       N_PATH, base_seed, sig_p=sig)
        r = price_level("S1_基准", "论文口径", "L1b_指数口径", 5, KAPPA_BASE,
                        sim, MAC_REF)
        vrows.append({"σ_P": sig, "赔付率_期望%": r["赔付率_期望%"],
                      "出险概率%": r["出险概率%"],
                      "毛保费率_占限额%": r["毛保费率_占限额%"]})
    V = pd.DataFrame(vrows)
    print(V.to_string(index=False))
    print("  全国碳市场 2022–2026 日频收益率年化约 25%–35%，σ_P=0.28 取中值。")

    # ---------------------------------------------------------- 第七节
    print()
    print("=" * 78)
    print("第七节　省际费率系数（可信度压缩 γ=0.5）")
    print("=" * 78)
    pf = pd.read_csv(OUT / "s3_省际费率系数.csv")
    pr = provincial_rates(pf)
    print(pr.head(6).to_string(index=False))
    print("  …")
    print(pr.tail(4).to_string(index=False))

    # ---------------------------------------------------------- 落盘
    A.to_csv(OUT / "s4_保费率表.csv", index=False, encoding="utf-8-sig")
    B.to_csv(OUT / "s4_重定基对比.csv", index=False, encoding="utf-8-sig")
    band.to_csv(OUT / "s4_水平式保费区间.csv", encoding="utf-8-sig")
    bb.to_csv(OUT / "s4_重定基保费区间.csv", encoding="utf-8-sig")
    d.to_csv(OUT / "s4_标度档标定.csv", index=False, encoding="utf-8-sig")
    S.to_csv(OUT / "s4_不可识别参数敏感性.csv", index=False, encoding="utf-8-sig")
    V.to_csv(OUT / "s4_波动率敏感性.csv", index=False, encoding="utf-8-sig")
    pr.to_csv(OUT / "s4_省际费率系数.csv", index=False, encoding="utf-8-sig")
    cp.to_csv(OUT / "s4_口径对比.csv", encoding="utf-8-sig")
    with open(OUT / "s4_定价配置.json", "w", encoding="utf-8") as f:
        json.dump({
            "参考水平_MAC_REF": round(MAC_REF, 2),
            "碳价起点": P_2026,
            "缺口倍数可用区间": [1.0, round(float(m2_hi), 3)],
            "标度档": {t["tag"]: {"γ1": t["γ1"], "b1": t["b1"],
                              "MAC2022_1995价": round(t["MAC2022_1995价"], 2),
                              "m0_M2": round(m0_of(t, "M2_价格平移"), 3),
                              "m0_M1": round(m0_of(t, "M1_结构外推"), 3),
                              "越界首年": t["越界首年"]}
                    for t in tiers.values()},
            "随机过程": {"σ_P": SIG_P, "κ_P": KAPPA_P, "σ_m路径": SIG_M_PATH,
                     "σ_m水平": SIG_M_LEVEL, "σ_Q": SIG_Q,
                     "ρ_Pm": RHO_PM, "ρ_PQ": RHO_PQ, "ρ_mQ": RHO_MQ},
            "费率构成": {"费用率": L1["expense_ratio"],
                     "安全边际": L1["safety_margin"], "贴现率": DF_R},
            "原则": "γ1/β 为标度自由度，模糊性而非风险，不做伪概率平均；"
                    "保费以区间呈现，审慎报价取上端；条款须冻结标度约定。",
        }, f, ensure_ascii=False, indent=2)
    print("\n[已写出] s4_保费率表.csv / s4_重定基对比.csv / s4_保费区间.csv / "
          "s4_标度档标定.csv / s4_不可识别参数敏感性.csv / "
          "s4_省际费率系数.csv / s4_口径对比.csv / s4_定价配置.json")
    return A, B, d


def provincial_rates(pf: pd.DataFrame, gamma: float = 0.5) -> pd.DataFrame:
    """省际费率系数：以还原口径系数为风险信号，Bühlmann 可信度压缩后施加。"""
    out = pf.copy()
    raw = out["还原口径系数"].clip(lower=1e-6)
    out["原始系数"] = raw.round(3)
    out["压缩后系数"] = (1 + gamma * (raw - 1)).clip(0.5, 2.0).round(3)
    out["档次"] = pd.cut(out["压缩后系数"], [0, 0.8, 1.2, 1.6, 10],
                         labels=["低费率区", "基准区", "上浮区", "高费率区"])
    return out[["地区", "MAC_orig", "原始系数", "压缩后系数", "档次"]]


if __name__ == "__main__":
    main()
