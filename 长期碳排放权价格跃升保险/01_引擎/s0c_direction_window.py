# -*- coding: utf-8 -*-
"""
s0c_direction_window.py —— 方向向量 g 的**二维**模糊域：不只 γ₁，还有形状 β

为什么必须扩到二维
------------------
s0 只量化了 γ₁ ≡ b₁+1（标度自由度，β 固定为 1）。但二次型 DDF 的方向向量
本身也是一个自由度：

    g = (g_y, g_b) ，参数化中体现为 β ，平移联动式 b₁ = −1 + γ₁·β
    β=1 即「按观测值同比例平移」；β≠1 即改变产出扩张与减排的方向权重。

用户提示：deliver2 的属性是「参数 g 的取值影响 DDF 测算」。deliver2 的
scout_B 也记录文献中方向向量选择可让 MAC 差 15 倍（Wang K. 2018：
407 / 1226 / 6058 元/吨）。故仅沿 γ₁ 扫描会**低估**模糊域宽度。

本脚本的职责
------------
在 (γ₁, β) 二维格点上逐点重解 LP，回答三个问题：
  Q1 合法域（dev_max ≤ 1）在 (γ₁, β) 平面上是什么形状？是否连成一片？
  Q2 2022 年全国 MAC 的合法区间是多少？上确界 555.8 是否被 β≠1 突破？
     —— 这是审慎定价的上端点，若被突破则上端保费须上修。
  Q3 加上 β 之后，2026 年缺口倍数区间变成多少？

判据与 s0 一致：dev ≤ 1（碳成本不超过 GDP），由论文 LP 实现独立复算。
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from _config import OUT
from s0_gamma_window import ROOTS, run_at

# 方向形状档位：覆盖常见选择（等权 / 产出偏重 / 减排偏重）
BETAS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
# γ₁ 细网格：步长 0.02 覆盖 [段1左端, 段2右端]
G1_STEP = 0.02
CARBON_PRICE_95P = 55.30 / 2.2
DEFL_2026 = 2.169309
P_2026 = 97.90


def grid() -> list[tuple[float, float]]:
    g1s = list(np.arange(ROOTS[0], ROOTS[3] + 1e-9, G1_STEP))
    # 把 4 个精确根位也纳入（合法/非法分界处最需要采样）
    for r in ROOTS:
        for d in (-1e-4, +1e-4):
            if ROOTS[0] - 1e-9 <= r + d <= ROOTS[3] + 1e-9:
                g1s.append(r + d)
    g1s = sorted({round(g, 6) for g in g1s})
    return [(g, b) for b in BETAS for g in g1s]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pts = grid()
    print("=" * 78)
    print("s0c　方向向量 g 的二维模糊域（γ₁ × β）")
    print("=" * 78)
    print(f"  格点：γ₁ 步长 {G1_STEP}（含 4 个精确根位） × β ∈ {BETAS}")
    print(f"  合计 {len(pts)} 个 LP 求解，预计耗时较长，请稍候…")

    rows, t0 = [], time.time()
    for i, (g1, beta) in enumerate(pts):
        r = run_at(g1, beta=beta)
        rows.append({
            "γ1": round(g1, 6), "β": beta, "b1": r.get("b1"),
            "ok": bool(r.get("ok")),
            "MAC2022": r.get("MAC_2022均值", np.nan),
            "缺口倍数": r.get("政策缺口倍数", np.nan),
            "dev_max": r.get("dev_max", np.nan),
            "dev_mean": r.get("dev_mean", np.nan),
        })
        if (i + 1) % 40 == 0:
            print(f"    … {i + 1}/{len(pts)}  ({time.time() - t0:.0f}s)")

    d = pd.DataFrame(rows)
    d["合法"] = d["ok"] & (d["dev_max"] <= 1.0)
    print(f"  完成，耗时 {time.time() - t0:.0f}s")

    print()
    print("=" * 78)
    print("第一节　合法域在 (γ₁, β) 平面上的形状")
    print("=" * 78)
    piv = d.pivot_table(index="β", columns=d["γ1"].round(2), values="合法",
                        aggfunc="first")
    print("  合法=1 / 非法=0：")
    print(piv.fillna(-1).astype(int).to_string())
    print("\n  读取方式：行 = 方向形状 β，列 = 标度 γ₁。")
    print("  −1 表示该点 LP 不可行（约束相互冲突，比『非法』更强）。")

    leg = d[d["合法"]].copy()
    ill = d[d["ok"] & ~d["合法"]]
    inf = d[~d["ok"]]
    print(f"\n  合法 {len(leg)} 点 / 非法 {len(ill)} 点 / 不可行 {len(inf)} 点")

    print()
    print("=" * 78)
    print("第二节　Q2：MAC 的合法区间与上确界是否被 β 突破")
    print("=" * 78)
    if not len(leg):
        raise SystemExit("无合法点，脚本终止")
    lo, hi = float(leg["MAC2022"].min()), float(leg["MAC2022"].max())
    lo_r = leg.loc[leg["MAC2022"].idxmin()]
    hi_r = leg.loc[leg["MAC2022"].idxmax()]
    print(f"  2022 年全国 MAC 合法域 ∈ [{lo:.2f}, {hi:.2f}] 元/吨（约1995年不变价）")
    print(f"    下确界出现在 γ₁={lo_r['γ1']:+.4f}, β={lo_r['β']:.2f}")
    print(f"    上确界出现在 γ₁={hi_r['γ1']:+.4f}, β={hi_r['β']:.2f}")
    print(f"  跨度 {hi / lo:.2f} 倍")
    print(f"\n  对照：仅沿 γ₁（β=1）的区间 [168.21, 555.80]，跨度 3.30 倍")
    print(f"  ⇒ 加入形状 β 后，下确界 {lo:.1f} vs {168.21:.1f}"
          f"（{'下移' if lo < 168.21 else '未下移'} "
          f"{abs(lo / 168.21 - 1) * 100:.1f}%），"
          f"上确界 {hi:.1f} vs {555.80:.1f}"
          f"（{'上移' if hi > 555.80 else '未上移'} "
          f"{abs(hi / 555.80 - 1) * 100:.2f}%）")
    b1 = hi_r["β"]
    print(f"\n  **审慎定价的关键**：上确界出现在 β={b1:.2f}"
          + ("（即 β=1 的论文方向形状本身已是上端）" if abs(b1 - 1.0) < 1e-9
             else "（非 β=1）") + "。")

    print()
    print("=" * 78)
    print("第三节　Q3：折算到 2026 的缺口倍数区间")
    print("=" * 78)
    m_lo, m_hi = lo * DEFL_2026 / P_2026, hi * DEFL_2026 / P_2026
    print(f"  ×平减 {DEFL_2026:.4f} → 2026 名义 MAC ∈ [{lo * DEFL_2026:.0f}, "
          f"{hi * DEFL_2026:.0f}] 元/吨")
    print(f"  ÷碳价 {P_2026:.2f} → 缺口倍数 m ∈ [{m_lo:.2f}, {m_hi:.2f}]")
    print(f"  叠加 s2b 的结构外推（成本曲线实际下移）后，m 的下端可低至 1.0 以下；"
          f"\n  因 m<1 无经济含义（企业会全部自减排、市场不成立），实际下界取 1.0。")
    print(f"  ⇒ **2026 年缺口倍数的可用区间 = [1.0, {m_hi:.2f}]**，"
          f"上端由 (γ₁,β) 合法域的上确界决定。")

    print()
    print("=" * 78)
    print("第四节　与文献量级的对照（回答『15 倍』问题）")
    print("=" * 78)
    print(f"  本引擎全合法域跨度        ：{hi / lo:.2f} 倍（{lo:.0f}–{hi:.0f} 元/吨）")
    print( "  deliver2 scout_B 文献记录 ：Wang K.(2018) 方向向量三选 → 407/1226/6058，")
    print( "                             跨度约 14.9 倍")
    print(f"  ⇒ 本引擎的模糊域比文献的跨研究离散**窄**"
          f"（{hi / lo:.1f}× vs 14.9×），原因有三，须随表披露：")
    print("     ① 本引擎固定了同一样本、同一函数形式、同一约束集，只放开 (γ₁, β)；")
    print("     ② 文献的 15 倍含**跨样本跨模型**差异（不同国家/年份/参数化）；")
    print("     ③ 故本区间只覆盖**方向向量自由度**，不含数据修订、模型误设、")
    print("        测算口径等其他不确定性 —— 真实承保不确定性的下界，不是上界。")

    d.to_csv(OUT / "s0c_g二维模糊域.csv", index=False, encoding="utf-8-sig")
    out = {
        "格点": {"γ1步长": G1_STEP, "β档": BETAS, "求解次数": len(pts),
                 "合法": int(len(leg)), "非法": int(len(ill)),
                 "不可行": int(len(inf))},
        "MAC_2022_1995价_合法域": [round(lo, 3), round(hi, 3)],
        "跨度倍数": round(hi / lo, 3),
        "下确界点": {"γ1": float(lo_r["γ1"]), "β": float(lo_r["β"]),
                  "MAC": round(float(lo_r["MAC2022"]), 3)},
        "上确界点": {"γ1": float(hi_r["γ1"]), "β": float(hi_r["β"]),
                  "MAC": round(float(hi_r["MAC2022"]), 3)},
        "上确界是否被β突破": bool(hi > 555.80 + 1e-6),
        "对照_仅γ1区间": [168.207, 555.801],
        "折算2026": {
            "平减": DEFL_2026, "碳价": P_2026,
            "m区间": [round(m_lo, 3), round(m_hi, 3)],
            "可用区间_下界取1": [1.0, round(m_hi, 3)],
        },
        "文献对照": {"Wang_K_2018": [407, 1226, 6058], "跨度": 14.89,
                  "结论": "本引擎区间为方向向量自由度的下界，不含跨样本/模型不确定性"},
        "性质": "γ1 与 β 均为方向向量的约定性自由度，数据不可识别；"
                "区间属模糊性（ambiguity）而非风险（risk），无概率分布含义。",
    }
    with open(OUT / "s0c_g二维模糊域.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[已写出] s0c_g二维模糊域.csv / s0c_g二维模糊域.json")
    return d, out


if __name__ == "__main__":
    main()
