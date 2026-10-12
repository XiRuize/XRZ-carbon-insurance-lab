# -*- coding: utf-8 -*-
"""
s25_annual_rolling.py —— 第三轮审稿响应：资本成本时间口径修正（口径乙·年度滚动）

背景（2026-10-03 第三轮外部审稿，问题 1，已拍板）：
    审稿人指出 s22/s24 的报价原理 GP=(E+ρ·φ·RM)/(1−e−π) 对**整个合同期**只计提
    一期 ρ·RM，而 §5.4 声明资本「持有至保单终了释放」——若 ρ=6% 为年化率，
    多年期保费系统性低估（3y 复利口径 267.6 vs 现行 152.3；15y ROL 达 94%）。
    经用户拍板（四项全过）：
        ① 口径乙定稿为**年度滚动续保**：每保单年度一份独立年度层合同，
          资本持有 1 年 ⇒ ρ 年化口径**按构造自洽**，无跨期复利问题；
        ② 期限结构收缩至 ≤5 年（1/3/5），10/15 年档撤回（长端只可年度滚动，
          不得单一定价——与「长端必挂重定价」铁律一致）；
        ③ 全国版定位改「系统性宏观对冲工具」；
        ④ 摘要授权重写。

口径乙（年度滚动）的定价定义：
    第 t 保单年度（t=1..T）：K_t = 1.30·I_{t−1}（B1 年度重定基，同构），
    年度名义赔付 pay_t = min(max(I_t − K_t, 0), CAP)·Q_t（年度合同独立结算，不贴现）；
    年度层参数 D_t = q75(pay_t)、A_t = q99.5(pay_t)（按该年度赔付分布）；
    年度毛保费 GP_t = (E_t + ρ·φ·RM_t)/(1−e−π)，RM_t = TVaR99(pay_t^layer) − E_t；
    资本 C_t = φ·RM_t 持有 1 年。
    T 年名义总保费 = Σ_{t=1..T} GP_t（续保年度重定价；名义加总，不跨期贴现）。
    主报价 = **首年（第 1 保单年度）年度层**。

不改 s22/s23/s24（旧口径产物保留作对照）；本脚本登记口径偏离：累计现值层
（s22 主线）被年度滚动层取代，二者在「资本持有期」上的口径差即审稿问题 1。

用法：python s25_annual_rolling.py [--check]    # 退出码 0 = 断言全过
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import L1, OUT                                    # noqa: E402
import s4_pricing as s4                                        # noqa: E402
import s23_review_response as s23                              # noqa: E402
from s24_review2_response import (TIER_PAPER, TIER_OLDQ, TIER_SUP,  # noqa: E402
                                  SENS_SIGM, _basis_paths, _tier, ROOTS)
from s22_commercial_reprice import phi_div                     # noqa: E402

pd.set_option("display.width", 240)
pd.set_option("display.unicode.east_asian_width", True)

RHO = L1["capital_charge"]          # 6% 年化（口径乙下资本持有 1 年，自洽）
PI_PROFIT = 0.08
E_EXP_IDX = 0.12
E_EXP_OLD = 0.25
N_EFF = 100
RHO_BAR = 0.30
PHI = phi_div(N_EFF, RHO_BAR)
Q_LIMIT = 0.995
TERMS_OUT = [1, 3, 5]               # 期限收缩（拍板②）：10/15 年档撤回
SIG_B = 0.1069
SUP_FACTOR_H5 = float(np.sqrt(6.41 / 5.0))


# ============================================================ 年度层指标
def _tvar(x: np.ndarray, a: float = 0.99) -> float:
    q = np.quantile(x, a)
    t = x[x >= q]
    return float(t.mean()) if t.size else float(q)


def annual_metrics(pay: np.ndarray) -> dict:
    """单保单年度层指标（D=q75、A=q99.5、口径乙 CoC）。pay 为名义年度赔付。"""
    D = float(np.quantile(pay, 0.75))
    A = float(np.quantile(pay, Q_LIMIT))
    layer = np.clip(pay - D, 0.0, A - D)
    E = float(layer.mean())
    rm1 = _tvar(layer) - E
    rm = rm1 * PHI
    gp = (E + RHO * rm) / (1.0 - E_EXP_IDX - PI_PROFIT)
    return {"D": D, "A": A, "层宽": A - D,
            "触发频率%": float((pay > 0).mean() * 100),
            "层赔付频率%": float((layer > 0).mean() * 100),
            "E_layer": E, "RM_独立": rm1, "RM_组合": rm,
            "GP_CoC": gp, "ROL_CoC%": gp / max(A - D, 1e-9) * 100,
            "赔付率%": E / max(gp, 1e-9) * 100}


def year_pays(sim: dict, t: int, kappa: float = s4.KAPPA_BASE,
              cap: float = s4.CAP_AMOUNT, I: np.ndarray | None = None) -> np.ndarray:
    """第 t 保单年度名义赔付（t 从 1 计；K_t = κ·I[t−1]，与 B1 同构）。"""
    II = sim["I"] if I is None else I
    pay = np.minimum(np.maximum(II[t] - kappa * II[t - 1], 0.0), cap) * sim["Q"][t]
    return pay


# ============================================================ R2' 年度层主表
def r2(sims: dict, tiers: dict) -> pd.DataFrame:
    rows = []
    for tag in (TIER_PAPER, TIER_OLDQ):
        sim = sims[tag]
        base = float(tiers[tag]["MAC2022_1995价"]) * 2.169309
        # 逐年剖面（t=1..5）：续保年度重定价的读数
        for t in range(1, 6):
            m = annual_metrics(year_pays(sim, t))
            rows.append({"标度档": tag, "2026基准_元每吨": round(base, 1),
                         "保单年度": t, "口径": "年度层",
                         "自留额D": round(m["D"], 1),
                         "责任上限A": round(m["A"], 1),
                         "层宽": round(m["层宽"], 1),
                         "触发频率%": round(m["触发频率%"], 1),
                         "层赔付频率%": round(m["层赔付频率%"], 1),
                         "E_layer": round(m["E_layer"], 1),
                         "RM_组合": round(m["RM_组合"], 1),
                         "GP_CoC": round(m["GP_CoC"], 1),
                         "ROL_CoC%": round(m["ROL_CoC%"], 1),
                         "赔付率%": round(m["赔付率%"], 1)})
        # 期限汇总：名义总保费 = Σ GP_t；年保费 = 首年 GP
        for T in TERMS_OUT:
            gps = [annual_metrics(year_pays(sim, t))["GP_CoC"] for t in range(1, T + 1)]
            m1 = annual_metrics(year_pays(sim, 1))
            rows.append({"标度档": tag, "2026基准_元每吨": round(base, 1),
                         "保单年度": T, "口径": f"名义合计_{T}y",
                         "自留额D": round(m1["D"], 1),
                         "责任上限A": round(m1["A"], 1),
                         "层宽": round(m1["层宽"], 1),
                         "触发频率%": round(m1["触发频率%"], 1),
                         "层赔付频率%": round(m1["层赔付频率%"], 1),
                         "E_layer": round(m1["E_layer"], 1),
                         "RM_组合": round(m1["RM_组合"], 1),
                         "GP_CoC": round(sum(gps), 1),
                         "ROL_CoC%": round(m1["ROL_CoC%"], 1),
                         "赔付率%": round(m1["赔付率%"], 1)})
    return pd.DataFrame(rows)


# ============================================================ R3' 年度资本线
def r3(sims: dict, tiers: dict) -> pd.DataFrame:
    rows = []
    for tag in (TIER_PAPER, TIER_SUP, TIER_OLDQ):
        sim = sims[tag]
        base = float(tiers[tag]["MAC2022_1995价"]) * 2.169309
        m1 = annual_metrics(year_pays(sim, 1))
        rows.append({"标度档": tag, "2026基准_元每吨": round(base, 1),
                     "保单年度": 1,
                     "自留额D": round(m1["D"], 1),
                     "责任上限A": round(m1["A"], 1),
                     "层宽": round(m1["层宽"], 1),
                     "E_不计入资本": round(m1["E_layer"], 1),
                     "RM_单保单": round(m1["RM_独立"], 1),
                     "φ": round(PHI, 4),
                     "最低资本_组合口径": round(m1["RM_组合"], 1),
                     "资本比层宽%": round(m1["RM_组合"] / max(m1["层宽"], 1e-9) * 100, 1),
                     "参考_年保费GP乙": round(m1["GP_CoC"], 1),
                     "资本比年保费": round(m1["RM_组合"] / max(m1["GP_CoC"], 1e-9), 2),
                     "持有期_年": 1})
    return pd.DataFrame(rows)


# ============================================================ R4' 省级锚（年度）
def r4(sims: dict, tiers: dict) -> pd.DataFrame:
    rows = []
    sim0 = sims[TIER_PAPER]
    base = float(tiers[TIER_PAPER]["MAC2022_1995价"]) * 2.169309
    for scen, dI in (("N_全国系统性版", 0.0),
                     ("P1_省级锚·随机游走", SIG_B),
                     ("P2_省级锚·超扩散等效", SIG_B * SUP_FACTOR_H5)):
        I = sim0["I"] if dI == 0.0 else _basis_paths(sim0["I"], dI,
                                                     s4.RNG_SEED + 7919)
        sim = {"I": I, "Q": sim0["Q"]}
        m1 = annual_metrics(year_pays(sim, 1))
        gps5 = [annual_metrics(year_pays(sim, t))["GP_CoC"] for t in range(1, 6)]
        rows.append({"情景": scen, "标度档": TIER_PAPER,
                     "基差年扰动σ_ξ": round(dI, 4),
                     "2026基准_元每吨": round(base, 1),
                     "首年D": round(m1["D"], 1), "首年A": round(m1["A"], 1),
                     "首年层宽": round(m1["层宽"], 1),
                     "首年触发频率%": round(m1["触发频率%"], 1),
                     "首年E": round(m1["E_layer"], 1),
                     "首年RM_组合": round(m1["RM_组合"], 1),
                     "首年GP_乙": round(m1["GP_CoC"], 1),
                     "首年ROL%": round(m1["ROL_CoC%"], 1),
                     "5y名义合计": round(sum(gps5), 1)})
    D = pd.DataFrame(rows)
    nb = float(D.loc[D["情景"] == "N_全国系统性版", "首年GP_乙"].iloc[0])
    D["首年GP相对全国版"] = [round(float(r["首年GP_乙"]) / nb, 3) for _, r in D.iterrows()]
    return D


# ============================================================ R5' σ_m 年度报价带
def r5(tiers: dict, seeds: dict) -> pd.DataFrame:
    rows = []
    for sm in SENS_SIGM:
        sim = s23.simulate_p("S1_基准", tiers[TIER_PAPER], "M2_价格平移",
                             L1["mc_paths"], seeds[TIER_PAPER], sig_m=sm)
        m1 = annual_metrics(year_pays(sim, 1))
        gps3 = [annual_metrics(year_pays(sim, t))["GP_CoC"] for t in range(1, 4)]
        gps5 = [annual_metrics(year_pays(sim, t))["GP_CoC"] for t in range(1, 6)]
        rows.append({"σ_m": sm, "首年层宽": round(m1["层宽"], 1),
                     "首年E": round(m1["E_layer"], 1),
                     "首年RM_组合": round(m1["RM_组合"], 1),
                     "首年GP_乙": round(m1["GP_CoC"], 1),
                     "首年ROL%": round(m1["ROL_CoC%"], 1),
                     "3y名义合计": round(sum(gps3), 1),
                     "5y名义合计": round(sum(gps5), 1)})
    D = pd.DataFrame(rows)
    b = float(D.loc[D["σ_m"] == 0.15, "首年GP_乙"].iloc[0])
    D["首年GP相对0.15"] = [round(float(r["首年GP_乙"]) / b, 3) for _, r in D.iterrows()]
    return D


# ============================================================ R6' 年度口径瀑布
def r6(sim: dict) -> pd.DataFrame:
    """首年年度口径瀑布（契约档）：0 基准保单 → … → 5 口径乙。终值须与 R2' 一致。"""
    # 水平式基准：K = 1.30 × MAC_REF 固定（与 s22/s7 的 A_水平式同口径）
    pay_A = np.minimum(np.maximum(sim["I"][1] - s4.KAPPA_BASE * s4.MAC_REF,
                                  0.0), s4.CAP_AMOUNT) * sim["Q"][1]

    def step(pay, D, A, phi, e, coc):
        layer = np.clip(pay - D, 0.0, None if A is None else A - D)
        E = float(layer.mean())
        rm1 = _tvar(layer) - E
        rm = rm1 * phi
        gp = ((E + rm * (1.0 + RHO)) if not coc
              else (E + RHO * rm)) / (1.0 - e - PI_PROFIT)
        return gp

    D0, A1 = (lambda m: (m["D"], m["A"]))(annual_metrics(year_pays(sim, 1)))
    steps = [
        ("0", "改造前基准全值保单（水平式·无免赔·e25·单保单·叠层口径）",
         step(pay_A, 0.0, None, 1.0, E_EXP_OLD, False)),
        ("1", "+ 结构切换（水平式 → B1 年度重定基/年度滚动）",
         step(year_pays(sim, 1), 0.0, None, 1.0, E_EXP_OLD, False)),
        ("2", "+ 年度层结构（D=q75 自留，A=q99.5 截断）",
         step(year_pays(sim, 1), D0, A1, 1.0, E_EXP_OLD, False)),
        ("3", "+ 组合分散（n_eff=100，φ=0.5541）",
         step(year_pays(sim, 1), D0, A1, PHI, E_EXP_OLD, False)),
        ("4", "+ 指数产品费用率（25% → 12%）",
         step(year_pays(sim, 1), D0, A1, PHI, E_EXP_IDX, False)),
        ("5", "保费原理改资本成本口径（口径乙）",
         step(year_pays(sim, 1), D0, A1, PHI, E_EXP_IDX, True)),
    ]
    rows, prev = [], None
    for tag, desc, gp in steps:
        rows.append({"步骤": tag, "机制": desc,
                     "首年年度保费": round(gp, 1),
                     "相对上一步": None if prev is None
                     else f"{(gp / prev - 1) * 100:+.1f}%"})
        prev = gp
    return pd.DataFrame(rows)


# ============================================================ R7' 首年自助标准误
def r7(sim: dict) -> dict:
    pay = year_pays(sim, 1)
    rng = np.random.default_rng(20261003)
    idx = rng.integers(0, pay.size, size=(1000, pay.size))
    gps = np.empty(1000)
    for b in range(1000):
        gps[b] = annual_metrics(pay[idx[b]])["GP_CoC"]
    se = float(gps.std(ddof=1))
    return {"首年GP_自助标准误": round(se, 2),
            "相对标准误%": round(se / annual_metrics(pay)["GP_CoC"] * 100, 2),
            "B": 1000}


# ============================================================ R2b 各档首年年度层
def r2b(tiers: dict, seeds: dict) -> pd.DataFrame:
    """表 4 的年度口径版：全部标度档的首年年度层（D=q75、A=q99.5、口径乙）。"""
    rows = []
    for tag in tiers:
        if tag == TIER_SUP:
            continue
        sim = s4.simulate("S1_基准", tiers[tag], "M2_价格平移",
                          L1["mc_paths"], seeds[tag])
        m1 = annual_metrics(year_pays(sim, 1))
        base = float(tiers[tag]["MAC2022_1995价"]) * 2.169309
        rows.append({"标度档": tag,
                     "γ1": round(float(tiers[tag]["γ1"]), 4),
                     "β": float(tiers[tag]["β"]),
                     "2026基准": round(base, 0),
                     "首年D": round(m1["D"], 1), "首年A": round(m1["A"], 1),
                     "首年层宽": round(m1["层宽"], 1),
                     "首年GP": round(m1["GP_CoC"], 1),
                     "首年ROL%": round(m1["ROL_CoC%"], 1),
                     "封顶松紧_CAP÷基准": round(s4.CAP_AMOUNT / base, 2)})
    # 段 1 真端点（γ₁=+0.101853，不在 s4 缓存内；同 s23-C6 的 seed 约定 5×977）
    sup_end = _tier(float(ROOTS[1]), 1.0, "段1右端_真端点")
    sim_end = s4.simulate("S1_基准", sup_end, "M2_价格平移",
                          L1["mc_paths"], s4.RNG_SEED + 5 * 977)
    m1 = annual_metrics(year_pays(sim_end, 1))
    base = float(sup_end["MAC2022_1995价"]) * 2.169309
    rows.append({"标度档": "段1右端_真端点",
                 "γ1": round(float(ROOTS[1]), 4), "β": 1.0,
                 "2026基准": round(base, 0),
                 "首年D": round(m1["D"], 1), "首年A": round(m1["A"], 1),
                 "首年层宽": round(m1["层宽"], 1),
                 "首年GP": round(m1["GP_CoC"], 1),
                 "首年ROL%": round(m1["ROL_CoC%"], 1),
                 "封顶松紧_CAP÷基准": round(s4.CAP_AMOUNT / base, 2)})
    return pd.DataFrame(rows)


# ============================================================ R8' 年度口径敏感性
def r8(tiers: dict, seeds: dict) -> pd.DataFrame:
    """年度口径参数敏感性：σ_m（重跑）/ κ（重跑）/ CAP（确定性）/ 组合结构（解析）。"""
    rows = []
    sim0 = None

    def fy(sim, kappa=s4.KAPPA_BASE, cap=s4.CAP_AMOUNT):
        return annual_metrics(year_pays(sim, 1, kappa=kappa, cap=cap))

    # (a) σ_m（重跑，逐档种子）
    for sm in SENS_SIGM:
        sim = s23.simulate_p("S1_基准", tiers[TIER_PAPER], "M2_价格平移",
                             L1["mc_paths"], seeds[TIER_PAPER], sig_m=sm)
        m = fy(sim)
        rows.append({"块": "a_σm", "参数": f"σ_m={sm}", "GP": round(m["GP_CoC"], 1),
                     "ROL%": round(m["ROL_CoC%"], 1)})
    sim0 = s4.simulate("S1_基准", tiers[TIER_PAPER], "M2_价格平移",
                       L1["mc_paths"], seeds[TIER_PAPER])
    # (b) κ
    for k in (1.15, 1.30, 1.50, 1.80):
        m = fy(sim0, kappa=k)
        rows.append({"块": "b_κ", "参数": f"κ={k}", "GP": round(m["GP_CoC"], 1),
                     "ROL%": round(m["ROL_CoC%"], 1)})
    # (c) CAP 倍数（确定性重算）
    for mult in (1.0, 2.0, 4.0):
        m = fy(sim0, cap=mult * s4.MAC_REF)
        rows.append({"块": "c_CAP", "参数": f"CAP={mult:g}×基准",
                     "GP": round(m["GP_CoC"], 1), "ROL%": round(m["ROL_CoC%"], 1)})
    # (d) 组合结构与费用（解析：给定首年 E、RM_独立）
    m1 = annual_metrics(year_pays(sim0, 1))
    E0, rm1_0 = m1["E_layer"], m1["RM_独立"]
    for e in (E_EXP_IDX, E_EXP_OLD):
        for neff in (1, 30, 100, 500):
            for rb in (0.10, 0.30, 0.50):
                phi = phi_div(neff, rb)
                gp = (E0 + RHO * rm1_0 * phi) / (1.0 - e - PI_PROFIT)
                rows.append({"块": "d_组合费用", "参数": f"e={e},n_eff={neff},ρ̄={rb}",
                             "GP": round(gp, 1),
                             "ROL%": round(gp / max(m1["层宽"], 1e-9) * 100, 1)})
    return pd.DataFrame(rows)


# ============================================================ R7b 路径数稳定带
def r7b(tiers: dict, seeds: dict) -> pd.DataFrame:
    rows = []
    for np_ in (20000, 40000, 80000):
        sim = s4.simulate("S1_基准", tiers[TIER_PAPER], "M2_价格平移",
                          np_, seeds[TIER_PAPER])
        m1 = annual_metrics(year_pays(sim, 1))
        rows.append({"路径数": np_, "首年D": round(m1["D"], 1),
                     "首年A": round(m1["A"], 1), "首年层宽": round(m1["层宽"], 1),
                     "首年GP": round(m1["GP_CoC"], 1),
                     "首年ROL%": round(m1["ROL_CoC%"], 1)})
    return pd.DataFrame(rows)


# ============================================================ 验收断言
def checks(r0out: dict, d2: pd.DataFrame, d4: pd.DataFrame, d5: pd.DataFrame,
           d6: pd.DataFrame, r7out: dict) -> list[dict]:
    res = []

    def add(tag, desc, ok, got, exp):
        res.append({"验收项": tag, "判据": desc, "结果": "PASS" if ok else "FAIL",
                    "实测": got, "期望": exp})

    add("A0", "s22 主表 3y 对照（累计口径复现，dev<0.5）",
        r0out["最大偏差"] is not None and r0out["最大偏差"] < 0.5,
        r0out["最大偏差"], "<0.5")
    fy = d2[(d2["标度档"] == TIER_PAPER) & (d2["保单年度"] == 1)
            & (d2["口径"] == "年度层")].iloc[0]
    add("A1", "首年年度 ROL ∈ [5%, 30%]",
        5.0 <= float(fy["ROL_CoC%"]) <= 30.0, float(fy["ROL_CoC%"]), "[5,30]")
    add("A2", "首年赔付率 ∈ [20%, 70%]",
        20.0 <= float(fy["赔付率%"]) <= 70.0, float(fy["赔付率%"]), "[20,70]")
    n_gp = float(d4.loc[d4["情景"] == "N_全国系统性版", "首年GP_乙"].iloc[0])
    p1_gp = float(d4.loc[d4["情景"].str.startswith("P1"), "首年GP_乙"].iloc[0])
    p2_gp = float(d4.loc[d4["情景"].str.startswith("P2"), "首年GP_乙"].iloc[0])
    add("A3", "省级锚首年 GP ≥ 全国版", p1_gp >= n_gp and p2_gp >= n_gp,
        f"P1={p1_gp}, P2={p2_gp}, N={n_gp}", "P1,P2 ≥ N")
    g = d5.set_index("σ_m")["首年GP_乙"]
    add("A4", "σ_m 单调：GP(0.30)>GP(0.15)>GP(0.05)",
        g.loc[0.30] > g.loc[0.15] > g.loc[0.05],
        f"{g.loc[0.05]}/{g.loc[0.15]}/{g.loc[0.30]}", "严格递增")
    fin = float(d6.iloc[-1]["首年年度保费"])
    add("A5", "瀑布终值 == R2' 首年 GP（dev<0.5）",
        abs(fin - float(fy["GP_CoC"])) < 0.5,
        f"瀑布={fin}, R2'={float(fy['GP_CoC'])}", "|dev|<0.5")
    add("A6", "期限收缩登记：输出期限 = [1,3,5]，10/15 撤回",
        TERMS_OUT == [1, 3, 5], TERMS_OUT, "[1,3,5]")
    add("A7", "ρ 年化口径自洽（年度滚动 ⇒ 资本持有 1 年，无跨期复利）",
        True, "构造性成立（口径乙定义）", "True")
    add("A8", "首年 GP 自助相对标准误 < 5%",
        r7out["相对标准误%"] < 5.0, r7out["相对标准误%"], "<5%")
    return res


# ============================================================ R0 复现锚
def r0(tiers: dict, seeds: dict) -> dict:
    """累计口径复现锚：本脚本的年度赔付线性加总须逐位复现 s23.pv_rebased_p，
    且 3y 指标与 s22 主表一致（同 seed 同公式）——保证年度口径与旧口径同源。"""
    sim = s4.simulate("S1_基准", tiers[TIER_PAPER], "M2_价格平移",
                      L1["mc_paths"], seeds[TIER_PAPER])
    L = s23.pv_rebased_p(sim, 3, s4.KAPPA_BASE, s4.CAP_AMOUNT)
    df = 1.0 / (1.0 + L1["discount"]) ** np.arange(1, 4)
    lin = sum(df[t - 1] * year_pays(sim, t) for t in (1, 2, 3))
    lin_dev = float(np.abs(lin - L).max() / max(float(L.std()), 1e-9))
    m = s23.layer_metrics(L)
    ref = {}
    try:
        R = pd.read_csv(OUT / "s22_商业版定价_主表.csv")
        row = R[(R["结构"] == "B1_年度重定基") & (R["标度档"] == TIER_PAPER)
                & (R["期限"] == 3) & (R["免赔档"] == "D75_q75")
                & (R["费用口径"] == "e12_指数") & (R["n_eff"] == 100)].iloc[0]
        ref = {"D": float(row["D_元每吨"]), "A": float(row["A_q995"]),
               "层宽": float(row["层宽"]), "E": float(row["E_layer"]),
               "RM": float(row["RM_组合"]), "GP": float(row["GP_CoC_元每吨"])}
        dev = {k: abs(m[k2] - v) for k, (k2, v) in
               {"D": ("D", ref["D"]), "A": ("A", ref["A"]),
                "层宽": ("层宽", ref["层宽"]), "E": ("E_layer", ref["E"]),
                "RM": ("RM_组合", ref["RM"]), "GP": ("GP_CoC", ref["GP"])}.items()}
        max_dev = round(max(dev.values()), 4)
    except Exception as exc:                                   # noqa: BLE001
        ref, max_dev = {"error": str(exc)}, None
    return {"线性加总_dev(相对SD)": lin_dev, "s22对照_3y": ref, "最大偏差": max_dev}


def main() -> None:
    only = "--check" in sys.argv
    t0 = time.time()
    tiers = s4._lp_cache()
    seeds = {tag: s4.RNG_SEED + i * 977 for i, tag in enumerate(tiers)}
    sims = {tag: s4.simulate("S1_基准", tiers[tag], "M2_价格平移",
                             L1["mc_paths"], seeds[tag]) for tag in tiers}
    sup_raw = _tier(-0.125, 1.8, TIER_SUP)
    tiers[TIER_SUP] = sup_raw
    seeds[TIER_SUP] = s4.RNG_SEED + 20 * 977
    sims[TIER_SUP] = s4.simulate("S1_基准", sup_raw, "M2_价格平移",
                                 L1["mc_paths"], seeds[TIER_SUP])

    print("=" * 104)
    print("s25　口径乙·年度滚动（第三轮审稿响应：ρ 时间口径修正；期限收缩 ≤5 年）")
    print("=" * 104)

    r0out = r0(tiers, seeds)
    print("\n【R0'】复现锚（累计口径同源性）")
    print("-" * 104)
    print(json.dumps(r0out, ensure_ascii=False, indent=2))

    d2 = r2(sims, tiers)
    print("\n【R2'】年度层主表（逐年剖面 + 名义合计；单位 元/吨）")
    print("-" * 104)
    print(d2.to_string(index=False))

    d3 = r3(sims, tiers)
    print("\n【R3'】年度资本线（C=φ·RM，持有 1 年 ⇒ ρ 年化自洽）")
    print("-" * 104)
    print(d3.to_string(index=False))

    d4 = r4(sims, tiers)
    print("\n【R4'】省级锚年度版 vs 全国系统性版（契约档）")
    print("-" * 104)
    print(d4.to_string(index=False))

    d5 = r5(tiers, seeds)
    print("\n【R5'】σ_m 年度报价带（契约档首年）")
    print("-" * 104)
    print(d5.to_string(index=False))

    d6 = r6(sims[TIER_PAPER])
    print("\n【R6'】年度口径塌缩瀑布（契约档首年）")
    print("-" * 104)
    print(d6.to_string(index=False))

    r7out = r7(sims[TIER_PAPER])
    print("\n【R7'】首年 GP 自助标准误（B=1,000）")
    print("-" * 104)
    print(json.dumps(r7out, ensure_ascii=False, indent=2))

    d2b = r2b(tiers, seeds)
    print("\n【R2b'】各标度档首年年度层（表 4 的年度口径版）")
    print("-" * 104)
    print(d2b.to_string(index=False))

    d8 = r8(tiers, seeds)
    print("\n【R8'】年度口径参数敏感性（σ_m / κ / CAP / 组合费用）")
    print("-" * 104)
    print(d8.to_string(index=False))

    d7b = r7b(tiers, seeds)
    print("\n【R7b'】路径数稳定带（首年年度层）")
    print("-" * 104)
    print(d7b.to_string(index=False))

    res = checks(r0out, d2, d4, d5, d6, r7out)
    print("\n【验收】")
    print("-" * 104)
    for r in res:
        print(f"  [{r['结果']:4s}] {r['验收项']:4s} {r['判据']}　→ {r['实测']}")
    nfail = sum(1 for r in res if r["结果"] == "FAIL")
    print(f"\n  合计：{len(res) - nfail} PASS / {nfail} FAIL　（耗时 {time.time() - t0:.0f}s）")

    if not only:
        d2.to_csv(OUT / "s25_R2_年度层主表.csv", index=False, encoding="utf-8-sig")
        d3.to_csv(OUT / "s25_R3_年度资本线.csv", index=False, encoding="utf-8-sig")
        d4.to_csv(OUT / "s25_R4_省级锚年度.csv", index=False, encoding="utf-8-sig")
        d5.to_csv(OUT / "s25_R5_sigma_m年度带.csv", index=False, encoding="utf-8-sig")
        d6.to_csv(OUT / "s25_R6_瀑布年度.csv", index=False, encoding="utf-8-sig")
        d2b.to_csv(OUT / "s25_R2b_各档年度层.csv", index=False, encoding="utf-8-sig")
        d8.to_csv(OUT / "s25_R8_敏感性年度.csv", index=False, encoding="utf-8-sig")
        d7b.to_csv(OUT / "s25_R7b_路径数稳定带.csv", index=False, encoding="utf-8-sig")
        with open(OUT / "s25_验收与复现.json", "w", encoding="utf-8") as f:
            json.dump({"R0_复现锚": r0out, "R7_自助标准误": r7out,
                       "验收": res, "口径乙定义": {
                           "结构": "年度滚动续保（每保单年度独立年度层）",
                           "D_t": "q75(pay_t)", "A_t": "q99.5(pay_t)",
                           "GP_t": "(E+ρ·φ·RM)/(1−e−π)",
                           "资本持有期": "1 年 ⇒ ρ=6% 年化口径按构造自洽",
                           "期限": TERMS_OUT,
                           "撤回": "10/15 年档单一定价撤回（长端只可年度滚动）"}},
                      f, ensure_ascii=False, indent=2)
        print("\n[已写出] s25_R2_年度层主表.csv / s25_R3_年度资本线.csv / "
              "s25_R4_省级锚年度.csv / s25_R5_sigma_m年度带.csv / "
              "s25_R6_瀑布年度.csv / s25_验收与复现.json")

    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
