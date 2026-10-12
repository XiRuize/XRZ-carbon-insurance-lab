# -*- coding: utf-8 -*-
"""
s26_review4_response.py —— 第四轮审稿响应（2026-10-03 晚，针对论文 v7）

背景：第四轮外部审稿（建议大修）10 项重大问题 + 8 项次要问题。经逐条核验，
多数已在 v7 处理；本脚本只补**引擎可算而 v7 尚未产出**的五块，并为改稿 v8 取证：

    Q1  候选参数敏感性：σ_P（碳价波动率）扫描 —— 回应"σ_P=0.28 与实测 14.8% 口径关系"
    Q2  1–5 保单年度最低资本线（资本逐年轮廓）—— 回应表 12 只给首年
    Q3  逆选择情景：组合结构 (ρ̄, n_eff) 恶化下 φ 与保费的螺旋 —— 回应"逆选择未建模"
    Q4  系统性分量占比情景：全国版对冲有效性上界的解析情景 —— 回应"需求端价值未量化"
    Q5  指数发布滞后：结算延迟 1–2 年的现值折减与资本占用延展 —— 回应"发布滞后未解决"

设计约束（与既有脚本一致）：
  * 不改 s22/s23/s24/s25 任何文件；本脚本只读其函数、写 s26_* 新产物。
  * 模糊性不进 MC：σ_P 扫描属**风险过程参数**，不是方向向量自由度，故可扫描。
  * 断言以负向守卫为主：σ_P 单调、资本线非退化、φ 单调、对冲上界 ∈ (0,1]。

用法：python s26_review4_response.py [--check]    # 退出码 0 = 断言全过
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
import s25_annual_rolling as s25                               # noqa: E402
from s24_review2_response import (TIER_PAPER, TIER_SUP,        # noqa: E402
                                  _tier as _tier_of)
from s22_commercial_reprice import phi_div                     # noqa: E402

RHO = L1["capital_charge"]
PI_PROFIT = 0.08
E_EXP_IDX = 0.12
N_EFF = 100
RHO_BAR = 0.30
PHI = phi_div(N_EFF, RHO_BAR)
SIG_B = 0.1069
VAR_PERSIST = 0.8586            # 省间固定水平差占基差方差（§3.5）
VAR_TIMEVARY = 0.1219           # 省内时变占基差方差（§3.5）

# σ_P 扫描档：基准 0.28（s4.SIG_P），含"实测折算 14.8%"作为低端点
SENS_SIGP = [0.148, 0.20, 0.28, 0.35, 0.45]
# 无风险贴现率（L1["discount"]），用于滞后现值折减
DF_R = L1["discount"]


def _gp(E: float, rm: float, e: float = E_EXP_IDX) -> float:
    return (E + RHO * rm) / (1.0 - e - PI_PROFIT)


# ============================================================ Q1 σ_P 敏感性
def q1(tier: dict, seed: int) -> pd.DataFrame:
    """碳价波动率 σ_P 扫描（首年年度层，契约档）。σ_P 是风险过程参数，可扫描。"""
    rows = []
    for sp in SENS_SIGP:
        sim = s4.simulate("S1_基准", tier, "M2_价格平移",
                          L1["mc_paths"], seed, sig_p=sp)
        m1 = s25.annual_metrics(s25.year_pays(sim, 1))
        gps5 = [s25.annual_metrics(s25.year_pays(sim, t))["GP_CoC"]
                for t in range(1, 6)]
        rows.append({"σ_P": sp, "首年层宽": round(m1["层宽"], 1),
                     "首年E": round(m1["E_layer"], 1),
                     "首年RM_组合": round(m1["RM_组合"], 1),
                     "首年GP": round(m1["GP_CoC"], 1),
                     "首年ROL%": round(m1["ROL_CoC%"], 1),
                     "5y名义合计": round(sum(gps5), 1)})
    D = pd.DataFrame(rows)
    b = float(D.loc[D["σ_P"] == 0.28, "首年GP"].iloc[0])
    D["首年GP相对0.28"] = [round(float(r["首年GP"]) / b, 3) for _, r in D.iterrows()]
    return D


# ============================================================ Q2 1–5 年资本线
def q2(sim: dict, tier: dict) -> pd.DataFrame:
    """逐保单年度最低资本 C_t=φ·RM_t（t=1..5），资本持有一年、逐年释放。"""
    rows = []
    base = float(tier["MAC2022_1995价"]) * 2.169309
    for t in range(1, 6):
        m = s25.annual_metrics(s25.year_pays(sim, t))
        rows.append({"标度档": TIER_PAPER, "2026基准_元每吨": round(base, 1),
                     "保单年度": t,
                     "自留额D": round(m["D"], 1),
                     "责任上限A": round(m["A"], 1),
                     "层宽": round(m["层宽"], 1),
                     "E_不计入资本": round(m["E_layer"], 1),
                     "RM_单保单": round(m["RM_独立"], 1),
                     "φ": round(PHI, 4),
                     "最低资本C_组合": round(m["RM_组合"], 1),
                     "C比层宽%": round(m["RM_组合"] / max(m["层宽"], 1e-9) * 100, 1),
                     "C比首年": round(m["RM_组合"] / s25.annual_metrics(
                         s25.year_pays(sim, 1))["RM_组合"], 3),
                     "年保费GP": round(m["GP_CoC"], 1),
                     "资本占用年数": 1})
    return pd.DataFrame(rows)


# ============================================================ Q3 逆选择情景
def q3(sim: dict) -> pd.DataFrame:
    """
    逆选择情景：高缺口企业进入 / 低缺口企业退出 ⇒ 组合同质化 ⇒ ρ̄ 上升、n_eff 下降。
    解析口径：φ=sqrt(ρ̄+(1−ρ̄)/n_eff)，保费 GP=(E+ρ·φ·RM1)/(1−e−π)，E/RM1 首年固定。
    """
    m1 = s25.annual_metrics(s25.year_pays(sim, 1))
    E0, rm1 = m1["E_layer"], m1["RM_独立"]
    base_phi = phi_div(N_EFF, RHO_BAR)
    base_gp = _gp(E0, rm1 * base_phi)
    scen = [
        ("基准组合（无逆选择）", 100, 0.30),
        ("轻度：n_eff 100→60，ρ̄ 0.30→0.40", 60, 0.40),
        ("中度：n_eff 100→30，ρ̄ 0.30→0.55", 30, 0.55),
        ("重度：n_eff 100→15，ρ̄ 0.30→0.70", 15, 0.70),
        ("极端：n_eff 100→8，ρ̄ 0.30→0.85", 8, 0.85),
    ]
    rows = []
    for desc, neff, rb in scen:
        phi = phi_div(neff, rb)
        gp = _gp(E0, rm1 * phi)
        rows.append({"情景": desc, "n_eff": neff, "ρ̄": rb,
                     "φ": round(phi, 4), "φ/基准φ": round(phi / base_phi, 3),
                     "首年GP": round(gp, 1),
                     "GP/基准GP": round(gp / base_gp, 3),
                     "ROL%": round(gp / max(m1["层宽"], 1e-9) * 100, 1)})
    return pd.DataFrame(rows)


# ============================================================ Q4 系统性分量情景
def q4(sim: dict) -> pd.DataFrame:
    """
    全国版对冲有效性：企业实际合规成本 = 系统性分量 + 基差。
    解析上界：年度同比口径下省间持久水平差被约掉，故可对冲方差占比上界 = 系统性份额；
    全国版因不覆盖省内时变（12.19%）与行业/个体基差，实际上界低于 1。
    """
    m1 = s25.annual_metrics(s25.year_pays(sim, 1))
    rows = []
    # 三种企业成本结构假设（系统性分量占比 w_sys），及对应的可对冲方差上界
    for w_sys, desc in [(0.30, "低系统性敞口（多数成本为企业个体）"),
                        (0.50, "中等"),
                        (0.70, "高系统性敞口（成本随全国指数同向）"),
                        (0.90, "极高（近乎全国口径）")]:
        # 可对冲方差上界：w_sys² （因 var(S+B)=w²σS²+(1-w)²σB²+...，取相关为 0 的保守情形）
        # 简化解析口径：对冲后方差残余 = (1−w_sys)²·var(B) + w_sys²·var(S)·(1−ρ_SI)
        # 取 ρ_SI=1（指数完美捕捉系统性分量），则残余 = (1−w_sys)²·var(B)
        resid = (1 - w_sys) ** 2
        rows.append({"企业成本结构": desc, "系统性分量占比w_sys": w_sys,
                     "可对冲方差占比上界": round(1 - resid, 3),
                     "残余（省内时变+行业+个体）": round(resid, 3),
                     "年保费GP": round(m1["GP_CoC"], 1),
                     "对冲成本/被保基准": round(m1["GP_CoC"] / 413.32, 3)})
    return pd.DataFrame(rows)


# ============================================================ Q5 指数发布滞后
def q5(sim: dict) -> pd.DataFrame:
    """
    指数发布滞后 L 年 ⇒ 赔付结算延迟 ⇒ 现值折减 + 资本占用延展。
    现值折减：(1+r)^{-L}；资本延展：ρ·RM·L（额外占用 L 年的资本成本）。
    """
    m1 = s25.annual_metrics(s25.year_pays(sim, 1))
    E0, rm = m1["E_layer"], m1["RM_组合"]
    rows = []
    for L in (0, 1, 2):
        df = (1 + DF_R) ** (-L)
        gp_adj = _gp(E0, rm) * 1.0 + RHO * rm * L   # 现值折减作用于 E；资本延展另计
        # 严格口径：E 折现、资本成本按 (1+L·ρ) 计，再除分母
        gp_strict = (E0 * df + RHO * rm * (1 + L)) / (1.0 - E_EXP_IDX - PI_PROFIT)
        rows.append({"发布滞后年数L": L, "赔付现值折减系数": round(df, 4),
                     "资本占用年数": 1 + L,
                     "年保费基准口径": round(m1["GP_CoC"], 1),
                     "年保费含滞后": round(gp_strict, 1),
                     "相对基准": round(gp_strict / m1["GP_CoC"], 3),
                     "ROL%": round(gp_strict / max(m1["层宽"], 1e-9) * 100, 1)})
    return pd.DataFrame(rows)


# ============================================================ 验收断言
def checks(d1, d2, d3, d4, d5) -> list[dict]:
    res = []

    def add(tag, desc, ok, got, exp):
        res.append({"验收项": tag, "判据": desc, "结果": "PASS" if ok else "FAIL",
                    "实测": got, "期望": exp})

    # B1 σ_P 单调：GP 随 σ_P 递增（尾部更重）
    g = d1.set_index("σ_P")["首年GP"]
    add("B1", "σ_P 单调：GP(0.45)>GP(0.28)>GP(0.148)",
        g.loc[0.45] > g.loc[0.28] > g.loc[0.148],
        f"{g.loc[0.148]}/{g.loc[0.28]}/{g.loc[0.45]}", "严格递增")
    add("B2", "σ_P=0.28 基准值与 s25 首年 GP 一致（dev<0.5）",
        abs(g.loc[0.28] - 89.9) < 0.5, g.loc[0.28], "≈89.9")
    # B3 资本线非退化且逐年递增（赔付分布逐年变重）
    c = d2["最低资本C_组合"].values
    add("B3", "资本线逐年非降（C_5 ≥ C_1）", c[-1] >= c[0],
        f"C1={c[0]}, C5={c[-1]}", "C5≥C1")
    add("B4", "资本/层宽稳定于 [40%,60%]", 
        bool(np.all((d2["C比层宽%"] >= 40) & (d2["C比层宽%"] <= 60))),
        list(d2["C比层宽%"]), "[40,60]")
    # B5 φ 单调随 n_eff 下降而上升
    phi = d3["φ"].values
    add("B5", "φ 随组合恶化（n_eff↓）单调上升",
        bool(np.all(np.diff(phi) > 0)), list(d3["φ"].round(3)), "递增")
    add("B6", "极端逆选择 GP 上浮 < 3 倍",
        float(d3["GP/基准GP"].iloc[-1]) < 3.0,
        float(d3["GP/基准GP"].iloc[-1]), "<3")
    # B7 对冲上界 ∈ (0,1]
    ub = d4["可对冲方差占比上界"].values
    add("B7", "对冲有效性上界 ∈ (0,1]", bool(np.all((ub > 0) & (ub <= 1))),
        list(ub), "(0,1]")
    # B8 滞后使保费上升
    gp = d5["年保费含滞后"].values
    add("B8", "发布滞后使保费上升（含滞后 ≥ 基准）", gp[-1] >= d5["年保费基准口径"].iloc[0],
        f"L=2:{gp[-1]} vs 基准:{d5['年保费基准口径'].iloc[0]}", "递增")
    return res


def main() -> None:
    only = "--check" in sys.argv
    t0 = time.time()
    tiers = s4._lp_cache()
    seeds = {tag: s4.RNG_SEED + i * 977 for i, tag in enumerate(tiers)}
    tier = tiers[TIER_PAPER]
    seed = seeds[TIER_PAPER]
    sim = s4.simulate("S1_基准", tier, "M2_价格平移", L1["mc_paths"], seed)

    print("=" * 100)
    print("s26　第四轮审稿响应：σ_P 敏感性 / 资本逐年线 / 逆选择 / 系统性分量 / 发布滞后")
    print("=" * 100)

    d1 = q1(tier, seed)
    print("\n【Q1】σ_P 敏感性（首年年度层，契约档）")
    print("-" * 100)
    print(d1.to_string(index=False))

    d2 = q2(sim, tier)
    print("\n【Q2】1–5 保单年度最低资本线（C_t=φ·RM_t，持有 1 年）")
    print("-" * 100)
    print(d2.to_string(index=False))

    d3 = q3(sim)
    print("\n【Q3】逆选择情景（组合结构恶化 ⇒ φ 与保费螺旋）")
    print("-" * 100)
    print(d3.to_string(index=False))

    d4 = q4(sim)
    print("\n【Q4】全国版对冲有效性：系统性分量占比情景")
    print("-" * 100)
    print(d4.to_string(index=False))

    d5 = q5(sim)
    print("\n【Q5】指数发布滞后：现值折减与资本延展")
    print("-" * 100)
    print(d5.to_string(index=False))

    res = checks(d1, d2, d3, d4, d5)
    print("\n【验收】")
    print("-" * 100)
    for r in res:
        print(f"  [{r['结果']:4s}] {r['验收项']:4s} {r['判据']}　→ {r['实测']}")
    nfail = sum(1 for r in res if r["结果"] == "FAIL")
    print(f"\n  合计：{len(res) - nfail} PASS / {nfail} FAIL　（耗时 {time.time() - t0:.0f}s）")

    if not only:
        d1.to_csv(OUT / "s26_Q1_sigma_P敏感性.csv", index=False, encoding="utf-8-sig")
        d2.to_csv(OUT / "s26_Q2_资本逐年线.csv", index=False, encoding="utf-8-sig")
        d3.to_csv(OUT / "s26_Q3_逆选择情景.csv", index=False, encoding="utf-8-sig")
        d4.to_csv(OUT / "s26_Q4_系统性分量情景.csv", index=False, encoding="utf-8-sig")
        d5.to_csv(OUT / "s26_Q5_发布滞后.csv", index=False, encoding="utf-8-sig")
        with open(OUT / "s26_验收与复现.json", "w", encoding="utf-8") as f:
            json.dump({"验收": res, "口径": {
                "Q1": "σ_P 扫描（风险过程参数，非方向向量自由度）",
                "Q2": "资本逐年轮廓 C_t=φ·RM_t，持有 1 年",
                "Q3": "逆选择解析情景：φ=sqrt(ρ̄+(1−ρ̄)/n_eff)",
                "Q4": "对冲上界 1−(1−w_sys)^2，取 ρ_SI=1 保守",
                "Q5": "GP=(E·(1+r)^−L + ρ·RM·(1+L))/(1−e−π)"}},
                f, ensure_ascii=False, indent=2)
        print("\n[已写出] s26_Q1..Q5_*.csv / s26_验收与复现.json")

    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
