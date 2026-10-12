# -*- coding: utf-8 -*-
"""
s7_rebuild.py —— 精算口径重建（修正版）

s6 的教训：把「限额」直接定义为 TVaR 会把风险附加**压成常数**
（因 RM/SI 与 RISK/SI 恒为定值），丧失定价信息。正确的精算分层是：

    限额 SI        = 可保损失上界 = P99.5(累计赔付现值) × n   ← 分母，责任上限
    纯风险保费      = E[L]
    风险附加 RM     = TVaR99(L) − E[L]        （可加原理，COC 口径）
    资本成本        = ρ × (TVaR99(L) − E[L])   ← 风险资本=尾部超出期望的部分
    毛保费          = (E[L] + RM + CapCost)/(1 − e − π)

    ⇒ 费率 = 毛保费/SI。让 RM/SI 与 CapCost/SI **随档位变化** —— 这正是
      「风险分类」应当体现的信息。s6 把 SI 设为 TVaR 恰好消掉了它。

期限处理（F5）：SI 随**累计赔付现值上界**增长（近线性于期限），
故费率随期限的漂移被消除，而不是被反号。
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

pd.set_option("display.width", 240)
pd.set_option("display.unicode.east_asian_width", True)

N_PATH = L1["mc_paths"]
Q_RISK = 0.99        # 风险资本与风险附加的分位
Q_LIMIT = 0.995      # 限额口径分位（责任上限）
RHO = L1.get("capital_charge", 0.06)
E_EXP = L1["expense_ratio"]
PI_PROFIT = 0.08
DF_R = L1["discount"]


def _tvar(x, a):
    q = np.quantile(x, a)
    t = x[x >= q]
    return float(t.mean()) if t.size else float(q)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 78)
    print("s7　精算口径重建（SI=责任上限；RM 随档位变化）")
    print("=" * 78)
    print(f"  限额分位={Q_LIMIT}  风险资本分位={Q_RISK}  ρ={RHO:.0%}  "
          f"e={E_EXP:.0%}  π={PI_PROFIT:.0%}")
    tiers = s4._lp_cache()
    sims = {("S1_基准", tag): s4.simulate("S1_基准", tiers[tag], "M2_价格平移",
                                          N_PATH, s4.RNG_SEED + i * 977)
            for i, tag in enumerate(tiers)}

    rows = []
    for tag in tiers:
        sim = sims[("S1_基准", tag)]
        for n in s4.TERMS:
            I, Q = sim["I"], sim["Q"]
            K = s4.KAPPA_BASE * s4.MAC_REF
            pay = np.minimum(np.maximum(I - K, 0.0), s4.CAP_AMOUNT) * Q
            df = 1.0 / (1.0 + DF_R) ** np.arange(n + 1)
            pv = (pay[:n + 1] * df[:, None]).sum(axis=0)
            E = float(pv.mean())
            si = float(np.quantile(pv, Q_LIMIT)) * n        # F1 责任上限
            tv = _tvar(pv, Q_RISK)
            rm = tv - E                                      # F2 风险附加
            cap = RHO * rm                                   # F3 资本成本
            net = E + rm + cap
            gp = net / (1 - E_EXP - PI_PROFIT)               # F4
            gp_old = E * (1 + L1["safety_margin"]) / (1 - E_EXP)
            si_old = s4.CAP_AMOUNT * n
            rows.append({
                "标度档": tag, "期限": n,
                "E": round(E, 1), "P99.5": round(float(np.quantile(pv, Q_LIMIT)), 1),
                "TVaR99": round(tv, 1), "SD": round(float(pv.std(ddof=1)), 1),
                "SI_旧": round(si_old, 0), "SI_新": round(si, 0),
                "纯风险%": round(E / si * 100, 2),
                "风险附加%": round(rm / si * 100, 2),
                "资本成本%": round(cap / si * 100, 2),
                "毛费率_新%": round(gp / si * 100, 2),
                "毛费率_旧s4%": round(gp_old / si_old * 100, 2),
                "新旧比": round((gp / si) / max(gp_old / si_old, 1e-9), 2),
            })
    T = pd.DataFrame(rows)

    print("\n【主表】S1 基准、κ=1.30、L1b 指数口径")
    print("-" * 78)
    for n in s4.TERMS:
        x = T[T["期限"] == n].sort_values("毛费率_新%")
        print(f"\n  ── {n} 年期 ──")
        print(x[["标度档", "SI_旧", "SI_新", "纯风险%", "风险附加%", "资本成本%",
                 "毛费率_新%", "毛费率_旧s4%", "新旧比"]].to_string(index=False))

    print("\n【F5】期限一致性（论文口径档）")
    print("-" * 78)
    y = T[T["标度档"] == "论文口径"]
    print(y[["期限", "SI_新", "E", "TVaR99", "纯风险%", "风险附加%",
             "毛费率_新%"]].to_string(index=False))
    base = y.iloc[0]
    for _, r in y.iloc[1:].iterrows():
        el = np.log(r["毛费率_新%"] / base["毛费率_新%"]) / np.log(r["期限"] / base["期限"])
        print(f"    期限 {int(r['期限']):2d} 年：毛费率期限弹性 = {el:.3f}")

    print("\n【跨档离散度】同一规格的保费区间（新口径 vs 旧口径）")
    print("-" * 78)
    for n in s4.TERMS:
        x = T[T["期限"] == n]
        lo_n, hi_n = x["毛费率_新%"].min(), x["毛费率_新%"].max()
        lo_o, hi_o = x["毛费率_旧s4%"].min(), x["毛费率_旧s4%"].max()
        print(f"    {n:2d} 年：新 [{lo_n:6.2f}%, {hi_n:6.2f}%] 宽 {hi_n/lo_n:5.2f}×"
              f"   |  旧 [{lo_o:6.2f}%, {hi_o:6.2f}%] 宽 {hi_o/lo_o:6.2f}×"
              f"   → 收窄 {(hi_o/lo_o)/(hi_n/lo_n):5.1f}×")

    T.to_csv(OUT / "s7_精算口径保费.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "s7_精算口径配置.json", "w", encoding="utf-8") as f:
        json.dump({
            "补丁对象": "s4_pricing.py（未修改 s4）",
            "SI_责任上限": f"quantile_{Q_LIMIT}(累计赔付现值) × 期限年数",
            "RM_风险附加": f"TVaR_{Q_RISK}(L) − E[L]（可加原理）",
            "资本成本": f"ρ({RHO}) × RM",
            "毛保费": f"(E[L]+RM+CapCost)/(1−{E_EXP}−{PI_PROFIT})",
            "旧口径缺陷": "SI = 2×MAC_REF×n 与风险无关 → 费率随档位漂移 22.6 倍",
        }, f, ensure_ascii=False, indent=2)
    print("\n[已写出] s7_精算口径保费.csv / s7_精算口径配置.json")


if __name__ == "__main__":
    main()
