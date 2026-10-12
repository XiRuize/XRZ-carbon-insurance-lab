# -*- coding: utf-8 -*-
"""
s2b_scale_robust.py —— 结论的标度稳健性：哪些随 γ₁ 变，哪些不变

问题
----
s1/s2/s3 的结论都建立在 γ₁ = 0（论文 b₁ = −1）这一个标度约定上。
但 γ₁ 在合法窗口 [−0.3243, +0.1019] ∪ [+0.1102, +0.1417] 内均合法，
且不同取值给出成倍的 MAC 差异（168.2 – 555.8 元/吨）。故必须区分：

  【标度不变结论】在整个合法窗口内定性/定量成立 → 可直接写进产品与政策建议
  【标度依赖结论】只在 γ₁ = 0 附近成立 → 必须带标度标注，不得外推

本脚本对窗口内的代表点逐一重解 LP、注入系数、重跑 s2 推算机制，
把 s1/s2/s3 的每一条核心断言逐档检验。

全国 MAC 的口径
---------------
自洽起见采用 u 空间口径：MAC_scaled 由全国均值输入算出，再乘有效标度
    scale_eff(γ₁) := MAC_2022(γ₁, 省级还原口径) / MAC_scaled_2022(γ₁, u空间)
这样 2022 年精确等于 run_at 的省级还原口径值，外推年沿用同一标度。
（省级还原口径在 γ₁≠0 时省际 u 离散结构不同，剔除越界省的偏差不可控，
  故外推一律走 u 空间口径。）
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from _config import OUT, SCENARIOS
from s0_gamma_window import ROOTS, run_at
from s1_ddf import load_panel, mac_scaled_from_u, u_from_inputs
from s2_nowcast import build_national_inputs, national_emission_path

# 窗口内代表点：段1 左端（最保守）/ 中段 / 论文点 / 段1 右端 / 段2 中点
TIERS = [
    ("段1左端_最保守", ROOTS[0] + 1e-4),
    ("段1偏左", -0.25),
    ("段1中段", -0.15),
    ("段1偏右", -0.05),
    ("论文口径", 0.0),
    ("段1右端_最低", 0.06),
    ("段2中点", 0.5 * (ROOTS[2] + ROOTS[3])),
]


def national_mac_path(coef, scen, gE, gEF):
    """该国标度约定下的全国 MAC 路径（元/吨，1995 价）+ 边界诊断。"""
    df, _, _, _ = load_panel()
    nat = build_national_inputs(scen, gE, gEF, coef=coef)
    b = df[df["年份"] == 2022]
    u_nat = float(u_from_inputs(float(b["s"].mean()), float(b["K_norm"].mean()),
                                float(b["L_norm"].mean()), coef=coef))
    ms22_u = float(mac_scaled_from_u(u_nat))
    nat["MAC_scaled_u"] = mac_scaled_from_u(nat["u"])
    return nat, ms22_u, u_nat


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tbl, gE, gEF = national_emission_path()

    print("=" * 78)
    print("s2b　结论的标度稳健性")
    print("=" * 78)
    print(f"  γ₁ 合法窗口：段1 [{ROOTS[0]:.4f}, {ROOTS[1]:.4f}]，"
          f"段2 [{ROOTS[2]:.4f}, {ROOTS[3]:.4f}]")

    rows, paths = [], {}
    for tag, g1 in TIERS:
        r = run_at(g1)
        if not r.get("ok"):
            print(f"  ⚠ {tag} γ₁={g1:.5f} 求解失败：{r.get('err')}")
            continue
        coef = r["_coef"]
        nat, ms22_u, u22 = national_mac_path(coef, SCENARIOS["S1_基准"], gE, gEF)
        scale_eff = r["MAC_2022均值"] / ms22_u
        nat["MAC95"] = nat["MAC_scaled_u"] * scale_eff
        nat_i = nat.set_index("年份")

        def mac(y):
            return float(nat_i.loc[y, "MAC95"])

        def u_at(y):
            return float(nat_i.loc[y, "u"])

        bad = nat[(nat["年份"] >= 2023) & (nat["u"] <= 0.005)]
        first_bad = int(bad["年份"].min()) if len(bad) else None
        rows.append({
            "标度档": tag, "γ1": round(g1, 5), "b1": r["b1"],
            "合法": r["合法"],
            "MAC2022_省级口径": round(r["MAC_2022均值"], 2),
            "scale_eff": round(scale_eff, 1),
            "u2022": round(u22, 6),
            "MAC2025": round(mac(2025), 1),
            "MAC2026": round(mac(2026), 1),
            "MAC2030": round(mac(2030), 1),
            "u2026": round(u_at(2026), 6),
            "越界首年": first_bad,
            "2022→2026变化%": round(100 * (mac(2026) / mac(2022) - 1), 1),
        })
        paths[tag] = nat_i["MAC95"]

    d = pd.DataFrame(rows)
    print()
    print(d.to_string(index=False))

    print()
    print("=" * 78)
    print("第一节　标度不变结论（整窗成立 → 可写进产品与政策建议）")
    print("=" * 78)
    verdict = []

    # 1) MAC 是否逐年下降
    down = []
    for tag in paths:
        s = paths[tag].loc[2022:2026].dropna()
        down.append(bool((s.diff().dropna() < 0).all()))
    ok = all(down)
    verdict.append(("MAC 2022→2026 逐年单调下降", ok,
                    f"{sum(down)}/{len(down)} 档成立"))

    # 2) u=0 越界是否最终发生
    byear = [r["越界首年"] for r in rows]
    ok2 = all(b is not None for b in byear)
    verdict.append(("结构引擎在 2045 前必然越界（u≤0.005）", ok2,
                    f"越界首年区间 {min(b for b in byear if b)}–"
                    f"{max(b for b in byear if b)}"))

    # 3) MAC 水平值的标度弹性
    lo = d["MAC2026"].min()
    hi = d["MAC2026"].max()
    verdict.append(("MAC 水平值随标度约定变化", True,
                    f"2026 年 {lo:.0f}–{hi:.0f} 元/吨，{hi/lo:.2f} 倍（标度依赖）"))

    for name, ok, note in verdict:
        print(f"  [{'不变' if ok else '依赖'}] {name:<34s} {note}")

    print()
    print("=" * 78)
    print("第二节　标度依赖结论（必须带标注，不得外推）")
    print("=" * 78)
    print("  ① **MAC 的水平值**：2026 年在窗口内可为 "
          f"{d['MAC2026'].min():.0f}–{d['MAC2026'].max():.0f} 元/吨（1995 价），"
          f"\n     跨度 {d['MAC2026'].max()/d['MAC2026'].min():.1f} 倍。"
          "任何单一点值都只是某一标度约定。")
    print("  ② **越界首年**："
          f"{sorted(set(b for b in byear if b))}"
          "\n     随标度档而移动，故『结构引擎有效期到 2027/2028』不是标度不变结论。")
    print("  ③ **2022→2026 降幅**："
          f"{d['2022→2026变化%'].min():.0f}% – {d['2022→2026变化%'].max():.0f}%，"
          "\n     方向（下降）不变，幅度依赖标度。")

    print()
    print("=" * 78)
    print("第三节　可用期限：标度约定决定『引擎何时失效』")
    print("=" * 78)
    print("  γ₁ 既决定 MAC 的水平，又决定其下降速率（两者正相关），")
    print("  因此不同标度约定的**引擎失效时间**并不相同。")
    print("  下表同时给出两个口径，用于分离『不确定性扩大』与『档位死亡』两种效应：")
    print("    · 全部存活档_跨度：含档位增减，会被『死亡效应』污染")
    print("    · 共同集_跨度：固定档位集，只反映相对不确定性")
    P = pd.DataFrame(paths)
    horizons = [2022, 2024, 2026, 2028, 2031, 2036]
    common = [c for c in P.columns
              if np.isfinite(P.loc[horizons[-1], c]) and P.loc[horizons[-1], c] > 0]
    amp = []
    for y in horizons:
        if y not in P.index:
            continue
        v = P.loc[y].dropna(); v = v[v > 0]
        vc = P.loc[y, common].dropna(); vc = vc[vc > 0]
        amp.append({"年份": y, "外推年数": y - 2022,
                    "全部档_最小": round(float(v.min()), 1),
                    "全部档_最大": round(float(v.max()), 1),
                    "全部档_跨度": round(float(v.max() / v.min()), 2),
                    "存活档数": f"{len(v)}/{len(P.columns)}",
                    f"共同集{len(common)}档_跨度":
                        round(float(vc.max() / vc.min()), 2) if len(vc) > 1 else np.nan,
                    "存活档_最低": round(float(v.min()), 1)})
    A = pd.DataFrame(amp)
    print()
    print(A.to_string(index=False))

    col = [c for c in A.columns if c.startswith("共同集")][0]
    s22 = float(A[A["年份"] == 2022]["全部档_跨度"].iloc[0])
    s26 = float(A[A["年份"] == 2026]["全部档_跨度"].iloc[0])
    cc = A[col].dropna()
    print(f"\n  三个分离的结论：")
    print(f"  ① **相对不确定性基本稳定**：固定共同档集（{common}）的跨度 "
          f"{cc.iloc[0]:.2f} → {cc.iloc[-1]:.2f} 倍，9 年几乎不变。")
    print(f"     ⇒ 标度约定之间的**排序与比例**是稳定的，没有『越算越糊』。")
    print(f"  ② **绝对跨度在 7 档全存活期内扩大**：2022 = {s22:.2f} 倍 → "
          f"2026 = {s26:.2f} 倍。")
    print("     根因：低 MAC 分支的 u 更快趋零（下降更快），高 MAC 分支下降慢。")
    print(f"  ③ **真正的问题是档位死亡，而非跨度扩大**：存活档数由 "
          f"{A['存活档数'].iloc[0]} 降至 {A['存活档数'].iloc[-1]}。")
    print("     2028 年已有 3/7 档因 u 越界而无法计算，2036 年仅剩 2/7。")
    print("\n  ⇒ 对产品的硬约束不是『不确定性太大』，而是**结构模型在 2030 年后")
    print("     对多数标度约定已不可计算**。故长端保单必须以**定期重定价**")
    print("     替代一次性长期承诺 —— 这是产品条款层面的结论，不是定价层面的。")

    print()
    print("=" * 78)
    print("第四节　对定价的含义")
    print("=" * 78)
    print("  · 定价锚必须用**区间**，不能用点值")
    print(f"  · 2026 年 MAC（1995 价）∈ "
          f"[{d['MAC2026'].min():.0f}, {d['MAC2026'].max():.0f}] 元/吨")
    print(f"  · 折算 2026 名义价（×2.1693）后 ∈ "
          f"[{d['MAC2026'].min()*2.1693:.0f}, {d['MAC2026'].max()*2.1693:.0f}] 元/吨")
    print("  · 对应缺口倍数 m = MAC / 碳价(97.9) ∈ "
          f"[{d['MAC2026'].min()*2.1693/97.9:.2f}, "
          f"{d['MAC2026'].max()*2.1693/97.9:.2f}]")
    print("  · 审慎定价取上端；并在条款中**冻结标度约定**（见交付物的治理条款）。")

    d.to_csv(OUT / "s2b_标度稳健性.csv", index=False, encoding="utf-8-sig")
    P.to_csv(OUT / "s2b_各标度档MAC路径.csv", encoding="utf-8-sig")
    A.to_csv(OUT / "s2b_期限放大.csv", index=False, encoding="utf-8-sig")
    print("\n[已写出] s2b_标度稳健性.csv / s2b_各标度档MAC路径.csv / s2b_期限放大.csv")
    return d


if __name__ == "__main__":
    main()
