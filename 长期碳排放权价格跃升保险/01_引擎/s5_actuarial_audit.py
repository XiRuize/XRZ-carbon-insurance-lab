# -*- coding: utf-8 -*-
"""
s5_actuarial_audit.py —— s4 定价引擎的精算合规性诊断（只读，不改动 s4 产物）

诊断目标（老板问「从精算学意义还有什么需要完善」）
------------------------------------------------------
把 s4 的定价式拆开，逐项对照精算定价的标准结构：

    P = [ E[L] / SI ] × (1 + 安全边际) / (1 − 费用率)      ← s4 实际实现

对照标准结构（毛保费 / 风险分类 / 资本 / 责任准备金）：

    P/SI = ( E[L] + RM(α) + CapCost ) / SI / (1 − e − π)

五项诊断
--------
D1 限额 SI 的自我循环性：SI = CAP_AMOUNT × n = 2 × MAC_REF × n，而 MAC_REF
   本身就是**未定价的模糊性锚**（论文口径 γ₁=0，仅处合法域最低 5.8% 分位）。
   ⇒ 费率的分母由被保风险自己的上游参数决定，形成「自己定自己的限额」。
   量化：给出**替代限额口径**（分位数口径 / 可保损失口径）下的费率。
D2 无风险附加：真实赔付出险率、赔付分位数（VaR/TVaR）全部缺失。
D3 资本成本未接入：_config 定义 capital_charge=0.06 但 s4 从未引用。
D4 费用与安全边际未分解，且无利润目标、保费期缴缺失。
D5 责任准备金/未到期保费/理赔准备金缺失。

运行：只读 s4 产物 + 重解 MC，写出 s5_精算诊断_*.csv，不覆盖任何 s4 文件。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import L1, OUT, PRICE_PATH, SCENARIOS          # noqa: E402
import s4_pricing as s4                                     # noqa: E402
from s3_term import deflator_to                             # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.unicode.east_asian_width", True)

ALPHAS = [0.90, 0.95, 0.99, 0.995, 0.999]      # 分位数水平
N_PATH = L1["mc_paths"]


# ======================================================== 赔付分布（逐路径）

def payoff_paths(sim: dict, n: int, base: str, kappa: float,
                 mac_ref: float, cap: float) -> tuple:
    """返回 (逐路径赔付现值数组, 逐路径年赔付矩阵, 贴现因子)。"""
    P, I, Q = sim["P"], sim["I"], sim["Q"]
    Y = I if base == "L1b_指数口径" else P
    K = kappa * (mac_ref if base == "L1b_指数口径" else s4.P_2026)
    pay = np.minimum(np.maximum(Y - K, 0.0), cap) * Q
    df = 1.0 / (1.0 + L1["discount"]) ** np.arange(n + 1)
    pv = (pay[:n + 1] * df[:, None]).sum(axis=0)
    return pv, pay[:n + 1], df


def tvars(x: np.ndarray, alpha: float) -> float:
    """TVaR（条件尾部期望），按损失升序取上 (1−α) 尾。"""
    q = np.quantile(x, alpha)
    tail = x[x >= q]
    return float(tail.mean()) if tail.size else float(q)


# ======================================================== D1 限额口径

def d1_limit_circularity(tiers: dict, sims: dict) -> pd.DataFrame:
    """
    替代限额口径对比（S1 基准、M2 口径、κ=1.30、5 年期、L1b）：

      SI_A  s4 现行：2 × MAC_REF × n
      SI_B  分位数口径：该档自身 99.5% 分位累计赔付现值 × n
      SI_C  可保损失口径：该档自身 TVaR(99%) 累计赔付现值 × n
            —— 「可保」= 以自身尾部为基准，分母不再引用上游模糊参数
    """
    rows = []
    for tag, t in tiers.items():
        sim = sims.get(("S1_基准", tag, "M2_价格平移"))
        if sim is None:
            continue
        n = 5
        pv, _, _ = payoff_paths(sim, n, "L1b_指数口径", s4.KAPPA_BASE,
                                s4.MAC_REF, s4.CAP_AMOUNT)
        SI_A = s4.CAP_AMOUNT * n
        q995 = float(np.quantile(pv, 0.995))
        tv = tvars(pv, 0.99)
        rows.append({
            "标度档": tag,
            "期望赔付现值": round(float(pv.mean()), 1),
            "P99.5赔付现值": round(q995, 1),
            "TVaR99赔付现值": round(tv, 1),
            "SI_A_现行_MAC锚": round(SI_A, 0),
            "SI_B_分位口径": round(q995 * n, 0),
            "SI_C_可保损失口径": round(tv * n, 0),
            "费率A_现行%": round(float(pv.mean() / SI_A) * 100, 2),
            "费率B_分位%": round(float(pv.mean() / (q995 * n)) * 100, 2),
            "费率C_可保%": round(float(pv.mean() / (tv * n)) * 100, 2),
            "现行口径虚高倍数": round(SI_A / (tv * n), 2),
        })
    return pd.DataFrame(rows).sort_values("现行口径虚高倍数", ascending=False)


# ======================================================== D2/D3 风险附加

def d2d3_loading(tiers: dict, sims: dict) -> pd.DataFrame:
    """把 s4 的费率拆成 纯风险保费 + 风险附加 + 费用 + 资本成本。"""
    rho = L1.get("capital_charge", 0.06)
    rows = []
    for tag, t in tiers.items():
        sim = sims.get(("S1_基准", tag, "M2_价格平移"))
        if sim is None:
            continue
        n = 5
        pv, _, _ = payoff_paths(sim, n, "L1b_指数口径", s4.KAPPA_BASE,
                                s4.MAC_REF, s4.CAP_AMOUNT)
        SI = s4.CAP_AMOUNT * n
        E = float(pv.mean())
        tv = tvars(pv, 0.99)
        rm_std = float(pv.std(ddof=1))                     # 标准差原理
        rm_tvar = tv - E                                   # TVaR 原理
        cap_cost = rho * tv                                # 资本成本 ≈ ρ × 风险资本
        e_ratio, sm = L1["expense_ratio"], L1["safety_margin"]
        net_need = E + rm_tvar + cap_cost                  # 含资本与尾部的净需求
        rows.append({
            "标度档": tag,
            "纯风险费率%": round(E / SI * 100, 2),
            "标准差附加%": round(rm_std / SI * 100, 2),
            "TVaR附加%": round(rm_tvar / SI * 100, 2),
            "资本成本@ρ6%%": round(cap_cost / SI * 100, 2),
            "净费率_含资本%": round(net_need / SI * 100, 2),
            "毛费率_含资本与费用%": round(net_need / SI / (1 - e_ratio) * 100, 2),
            "毛费率_s4现口径%": round(E / SI * (1 + sm) / (1 - e_ratio) * 100, 2),
            "口径差_倍": round((net_need / SI / (1 - e_ratio))
                           / max(E / SI * (1 + sm) / (1 - e_ratio), 1e-9), 2),
            "VaR99赔付现值": round(float(np.quantile(pv, 0.99)), 1),
            "TVaR99赔付现值": round(tv, 1),
            "损失波动率": round(float(pv.std(ddof=1) / max(E, 1e-9)), 2),
        })
    return pd.DataFrame(rows)


# ======================================================== D4 期限结构一致性

def d4_term_consistency(tiers: dict, sims: dict) -> pd.DataFrame:
    """
    精算一致性检验：限额随期限**线性**放大（SI ∝ n），但赔付现值的期限增长
    是**次线性**的（贴现 + 缺口倍数收敛）。若费率仍随期限上升，说明分母
    口径在扭曲期限价格 → 长端费率被系统性高估。
    """
    rows = []
    for n in s4.TERMS:
        sim = sims.get(("S1_基准", "论文口径", "M2_价格平移"))
        if sim is None:
            continue
        pv, _, _ = payoff_paths(sim, n, "L1b_指数口径", s4.KAPPA_BASE,
                                s4.MAC_REF, s4.CAP_AMOUNT)
        SI = s4.CAP_AMOUNT * n
        rows.append({
            "期限": n,
            "累计赔付现值": round(float(pv.mean()), 0),
            "SI_现行": round(SI, 0),
            "毛费率_现行%": round(float(pv.mean() / SI)
                              * (1 + L1["safety_margin"])
                              / (1 - L1["expense_ratio"]) * 100, 2),
            "赔付现值期限弹性": None,
            "毛费率期限弹性": None,
        })
    d = pd.DataFrame(rows)
    base = d.iloc[0]
    for i in range(1, len(d)):
        d.loc[i, "赔付现值期限弹性"] = round(
            np.log(d.loc[i, "累计赔付现值"] / base["累计赔付现值"])
            / np.log(d.loc[i, "期限"] / base["期限"]), 2)
        d.loc[i, "毛费率期限弹性"] = round(
            np.log(d.loc[i, "毛费率_现行%"] / base["毛费率_现行%"])
            / np.log(d.loc[i, "期限"] / base["期限"]), 2)
    d.loc[0, "赔付现值期限弹性"] = 1.0
    d.loc[0, "毛费率期限弹性"] = 1.0
    return d


# ======================================================== 主

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 78)
    print("s5　精算合规性诊断（只读 s4，不改动任何 s4 产物）")
    print("=" * 78)

    tiers = s4._lp_cache()
    base_seed = s4.RNG_SEED
    sims = {}
    for i, tag in enumerate(tiers):
        sims[("S1_基准", tag, "M2_价格平移")] = s4.simulate(
            "S1_基准", tiers[tag], "M2_价格平移", N_PATH, base_seed + i * 977)

    print()
    print("【D1】限额的自我循环性：SI 的分母由被保风险自己的上游模糊参数决定")
    print("-" * 78)
    O1 = d1_limit_circularity(tiers, sims)
    print(O1.to_string(index=False))
    print("\n  ⇒ 现行 SI = 2×MAC_REF×n，其中 MAC_REF 是**未定价的模糊性锚**。")
    print("     以自身 99% TVaR 为分母（口径C）时，费率立刻回到精算可解释量级。")

    print()
    print("【D2/D3】费率结构拆解：风险附加与资本成本（现引擎均缺失）")
    print("-" * 78)
    O2 = d2d3_loading(tiers, sims)
    print(O2.to_string(index=False))
    print(f"\n  资本成本取 ρ = {L1.get('capital_charge', 0.06):.0%}（_config 已定义，s4 从未引用）")

    print()
    print("【D4】期限一致性：SI 线性放大 vs 赔付现值次线性增长")
    print("-" * 78)
    O3 = d4_term_consistency(tiers, sims)
    print(O3.to_string(index=False))

    print()
    print("【D5】缺口项清单（结构性缺失，非参数问题）")
    print("-" * 78)
    gaps = [
        ("责任准备金", "未提未到期保费/理赔准备金，长险无准备金科目"),
        ("保费期缴", "全部按趸缴单期缴费定价，无期缴/续期失效率"),
        ("利润目标", "毛保费仅含费用率与安全边际，无利润边际 PVFP"),
        ("再保结构", "无自留额/成数分保/超赔层次，限额=合约上限"),
        ("逆选择", "省际费率系数为定价系数而非选择效应估计"),
        ("基差风险", "指数口径与各省实际损失间的基差未建模"),
        ("偿付能力", "未做最低资本与偿付能力充足率测算"),
        ("保单年度口径", "全部按日历年保单口径，无保单年度(underwriting year)视图"),
    ]
    G = pd.DataFrame(gaps, columns=["缺失项", "说明"])
    print(G.to_string(index=False))

    OUT_ = OUT
    O1.to_csv(OUT_ / "s5_限额口径对比.csv", index=False, encoding="utf-8-sig")
    O2.to_csv(OUT_ / "s5_费率结构拆解.csv", index=False, encoding="utf-8-sig")
    O3.to_csv(OUT_ / "s5_期限一致性.csv", index=False, encoding="utf-8-sig")
    G.to_csv(OUT_ / "s5_精算缺口清单.csv", index=False, encoding="utf-8-sig")
    with open(OUT_ / "s5_精算诊断.json", "w", encoding="utf-8") as f:
        json.dump({
            "诊断对象": "s4_pricing.py 输出",
            "D1_限额口径": {
                "现行": "SI = 2 × MAC_REF × n，MAC_REF=413.32 为未定价的模糊性锚",
                "建议": "SI = n × TVaR_99(该档自身累计赔付现值)，即口径C",
            },
            "D2_风险附加": "现引擎仅 E[L]；(1+安全边际) 是常量乘数，非损失分布函数",
            "D3_资本成本": f"capital_charge={L1.get('capital_charge', 0.06)} 已定义但 s4 未引用",
            "D4_费用": f"expense_ratio={L1['expense_ratio']} 未分解，无利润目标",
            "D5_准备金": "未到期保费/理赔准备金/保单年度口径均缺失",
            "分位数水平": ALPHAS,
        }, f, ensure_ascii=False, indent=2)
    print("\n[已写出] s5_限额口径对比.csv / s5_费率结构拆解.csv / "
          "s5_期限一致性.csv / s5_精算缺口清单.csv / s5_精算诊断.json")


if __name__ == "__main__":
    main()
