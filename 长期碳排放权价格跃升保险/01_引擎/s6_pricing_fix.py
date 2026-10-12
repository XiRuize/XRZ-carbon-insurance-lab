# -*- coding: utf-8 -*-
"""
s6_pricing_fix.py —— 按精算口径重建保费结构（对 s4 的补丁版，不覆盖 s4 产物）

修复内容（对应 s5 诊断 D1–D5）
------------------------------
F1  限额口径：SI = n × TVaR_α(该档自身累计赔付现值) × (1+缓冲)
    —— 分母不再引用上游模糊参数 MAC_REF，消除自我循环。
    **限额成为档位的函数**（而非恒定常量 2×MAC_REF×n），这正是「限额=可保
    损失上界」的精算含义。
F2  风险附加：以 TVaR 原理计风险边际 RM = TVaR_α − E[L]（可加口径成本），
    同时输出标准差原理 RM = z_α·σ[L] 作为交叉验证。
F3  资本成本：接入 _config 的 capital_charge ρ，资本成本 = ρ × TVaR_α(L)
    （风险资本以尾部为基准，非以期望为基准）。
F4  费用与利润：毛保费 = (E[L] + RM + CapCost) / (1 − e − π)，
    把 s4 的「乘以(1+安全边际)」改为显式的费用率 e 与利润边际 π。
F5  期限一致性：SI ∝ 累计赔付现值（而非 ∝ n），使费率不随期限系统漂移。
F6  保单年度视图：按保单年度(underwriting year)而非日历年统计赔付。

⚠ 本脚本不修改 s4。产出 s6_* 文件供对照与替换。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import L1, OUT                                  # noqa: E402
import s4_pricing as s4                                       # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.unicode.east_asian_width", True)

N_PATH = L1["mc_paths"]
ALPHA = 0.99            # 风险资本与限额的分位水平
ALPHA_LIMIT = 0.99      # 限额口径分位
BUFFER = 0.10           # 限额相对 TVaR 的缓冲
PI_PROFIT = 0.08        # 利润边际（占毛保费）
RHO = L1.get("capital_charge", 0.06)
E_EXP = L1["expense_ratio"]
DF_R = L1["discount"]


def _paths(sim: dict, n: int, kappa: float, mac_ref: float, cap: float):
    I, Q = sim["I"], sim["Q"]
    K = kappa * mac_ref
    pay = np.minimum(np.maximum(I - K, 0.0), cap) * Q
    df = 1.0 / (1.0 + DF_R) ** np.arange(n + 1)
    pv = (pay[:n + 1] * df[:, None]).sum(axis=0)
    und = (pay[:n + 1] * df[:, None])       # (n+1, npath) 保单年度分解
    return pv, und


def _tvar(x, a):
    q = np.quantile(x, a)
    t = x[x >= q]
    return float(t.mean()) if t.size else float(q)


def _z(a):
    """标准正态分位数（无 scipy 依赖的近似，精度 1e-6 内）。"""
    from math import erf, sqrt
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1 + erf(mid / sqrt(2))) < a:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build(tiers: dict, sims: dict) -> pd.DataFrame:
    rows = []
    for tag, t in tiers.items():
        for n in s4.TERMS:
            sim = sims.get(("S1_基准", tag, "M2_价格平移"))
            if sim is None:
                continue
            pv, und = _paths(sim, n, s4.KAPPA_BASE, s4.MAC_REF, s4.CAP_AMOUNT)
            E = float(pv.mean())
            TV = _tvar(pv, ALPHA)
            sd = float(pv.std(ddof=1))
            rm_tvar, rm_std = TV - E, _z(ALPHA) * sd
            cap_cost = RHO * TV
            SI_old = s4.CAP_AMOUNT * n
            SI_new = TV * (1 + BUFFER)                    # F1
            net = E + rm_tvar + cap_cost                  # F2+F3
            gp_new = net / (1 - E_EXP - PI_PROFIT)        # F4
            gp_old = E * (1 + L1["safety_margin"]) / (1 - E_EXP)
            rows.append({
                "标度档": tag, "期限": n,
                "E_期望赔付现值": round(E, 1),
                "TVaR99": round(TV, 1),
                "SD赔付": round(sd, 1),
                "SI_旧_MAC锚": round(SI_old, 0),
                "SI_新_TVaR口径": round(SI_new, 0),
                "纯风险费率%": round(E / SI_new * 100, 2),
                "风险附加%_TVaR": round(rm_tvar / SI_new * 100, 2),
                "风险附加%_标准差": round(rm_std / SI_new * 100, 2),
                "资本成本%": round(cap_cost / SI_new * 100, 2),
                "毛费率_新%": round(gp_new / SI_new * 100, 2),
                "毛费率_旧s4%": round(gp_old / SI_old * 100, 2),
                "新旧比": round((gp_new / SI_new) / max(gp_old / SI_old, 1e-9), 2),
                "保单年度赔付_末年占比%": round(
                    float(und[-1].mean() / max(und.sum(axis=0).mean(), 1e-9)) * 100, 1),
            })
    return pd.DataFrame(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 78)
    print("s6　精算口径重建（对 s4 的补丁；不覆盖 s4 产物）")
    print("=" * 78)
    print(f"  α(风险资本/限额)={ALPHA}  限额缓冲={BUFFER:.0%}  "
          f"ρ={RHO:.0%}  e={E_EXP:.0%}  π={PI_PROFIT:.0%}")
    tiers = s4._lp_cache()
    sims = {("S1_基准", tag, "M2_价格平移"):
            s4.simulate("S1_基准", tiers[tag], "M2_价格平移",
                        N_PATH, s4.RNG_SEED + i * 977)
            for i, tag in enumerate(tiers)}
    T = build(tiers, sims)

    print("\n【F1】限额改为 TVaR 口径 —— 限额成为档位的函数")
    print("-" * 78)
    p5 = T[T["期限"] == 5].sort_values("纯风险费率%")
    print(p5[["标度档", "SI_旧_MAC锚", "SI_新_TVaR口径", "纯风险费率%",
              "风险附加%_TVaR", "资本成本%", "毛费率_新%", "毛费率_旧s4%",
              "新旧比"]].to_string(index=False))

    print("\n【F4】期限一致性检验（新口径）")
    print("-" * 78)
    x = T[T["标度档"] == "论文口径"]
    print(x[["期限", "SI_新_TVaR口径", "纯风险费率%", "毛费率_新%"]].to_string(index=False))

    print("\n【F2】两套风险附加原理对照（TVaR vs 标准差）")
    print("-" * 78)
    print(p5[["标度档", "风险附加%_TVaR", "风险附加%_标准差",
              "E_期望赔付现值", "TVaR99", "SD赔付"]].to_string(index=False))
    print("\n  ⇒ TVaR 附加远大于标准差附加：赔付分布是**重尾右偏**（指数=价格×缺口"
          "\n     倍数，两个对数正态相乘），正态近似会严重低估所需资本。")

    T.to_csv(OUT / "s6_精算口径保费.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "s6_精算口径配置.json", "w", encoding="utf-8") as f:
        json.dump({
            "补丁对象": "s4_pricing.py",
            "F1_限额": f"SI = n × TVaR_{ALPHA}(累计赔付现值) × {1+BUFFER}",
            "F2_风险附加": f"RM = TVaR_{ALPHA} − E[L]（主）；z_{ALPHA}·σ[L]（交叉验证）",
            "F3_资本成本": f"{RHO} × TVaR_{ALPHA}",
            "F4_毛保费": f"(E[L]+RM+CapCost)/(1−{E_EXP}−{PI_PROFIT})",
            "F5_期限": "SI ∝ 累计赔付现值，非 ∝ 期限年数",
            "说明": "本文件为对照产物，s4 未被修改；替换与否由业务侧决定。",
        }, f, ensure_ascii=False, indent=2)
    print("\n[已写出] s6_精算口径保费.csv / s6_精算口径配置.json")


if __name__ == "__main__":
    main()
