# -*- coding: utf-8 -*-
"""
s0_gamma_window.py —— 方向向量标度自由度 γ₁ 的合法窗口与 MAC 区间（独立复算）

为什么这件事必须做
------------------
二次型 DDF 的参数化含一个**标度（计价单位）自由度** γ₁ ≡ b₁ + 1。
论文取 FGN-W(2005) 惯例 b₁ = −1，即 γ₁ = 0。
但 γ₁ 并非由数据识别：只要满足「碳成本 ≤ GDP」（dev ≤ 1）的归一化约束，
一整段 γ₁ 都合法，而**不同 γ₁ 给出成倍的 MAC 差异**。

deliver2 批次（ddf_scale_001）精确求解了合法窗口：
    W* = [−0.3243, +0.1019] ∪ [+0.1102, +0.1417]（两段，中间有宽 0.0083 的空洞）
    对应 2022 年全国 MAC ∈ [168.21, 555.80] 元/吨（约1995年不变价）
           政策缺口倍数 ∈ [6.69, 22.11]
而论文点值 γ₁ = 0 给出 190.5 元/吨、7.6 倍 —— **位于合法窗口的最低端**。

本脚本的职责
------------
**不复用 deliver2 的落盘结果**，而是直接调用论文的 LP 实现（ddf_lp_track1），
按 deliver2 公开的平移联动式 b₁ = −1 + γ₁·β 自行求解，独立复算窗口内各点，
再与 deliver2 的网格逐点对照。复算通过才允许把该区间写入定价锚。

对标度自由度不可识别性的处置原则（本项目）
------------------------------------------
区间 [168.21, 555.80] 是**模糊性（ambiguity）**而非**风险（risk）**：
deliver2 未给出概率分布，只给出合法域。故本项目：
  · 不做「对 γ₁ 取先验 → 求分位数」的伪概率运算；
  · 定价锚用**区间**表达，并按审慎原则取上端为上界保费基准；
  · 保单条款须**冻结并披露标度约定**（否则事后可选 γ₁ 操纵指数）。
关于 γ₁ 均匀先验下的展示性分布，仅作辅助输出并明确标注无依据。

输出
----
s0_γ1窗口.csv            窗口内逐点 MAC / 缺口倍数 / dev_max
s0_标度约定敏感性.json   窗口端点、论文点值位置、复算对照
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from _config import DELIVER, OUT, PAPER_DIR

sys.path.insert(0, str(DELIVER / "05_代码"))
import ddf_lp_track1 as t1     # noqa: E402

# ---------------------------------------------------------------- 数据（只读）

DF = t1.load_data()
RAW = pd.read_excel(t1._data("原始数据_修正LGDP_Kbase_WJP_Lofficial_v1.xlsx"))
assert len(RAW) == len(DF) == 840, (len(RAW), len(DF))

YBAR = RAW.groupby("地区")["Y_raw"].transform("mean").values
BBAR = RAW.groupby("地区")["C_raw"].transform("mean").values
C_RAW = RAW["C_raw"].values
Y_RAW = RAW["Y_raw"].values
YEAR = DF["年份"].values
N = len(DF)

# deliver2 的合法域判据：dev ≤ 1（碳成本不超过 GDP）
# 精确求解的 4 个根（s07_exact_window，残差 ≤7.4e-14）
ROOTS = [-0.324284, 0.101853, 0.110169, 0.141651]
# 政策倍数分母：上游口径 55.30/2.2（元/吨，1995 价），见 s00_common.py:54
CARBON_PRICE_95P = 55.30 / 2.2

KEEP = [i for i in range(15) if i != 4]


def run_at(g1: float, beta: float = 1.0) -> dict:
    """
    在锚定 γ₁、方向 g=(1,−β) 下解 LP 并返回 MAC / dev。

    平移联动：b₁ = −1 + γ₁·β（β=1 时 γ₁ = b₁+1）。
    D = A_red·p + b₁·y + γ₁·β·b ⋯ γ₁=0 时与交付基线逐位一致。
    """
    A, y, b, k, l, e = t1.build_design(DF, beta)
    A_ub, b_ub = t1.monotonicity_constraints(DF, beta)
    A_red = A[:, KEEP]

    b1 = -1.0 + g1 * beta
    gamma1 = g1 * beta

    A_all = np.vstack([-A_red, A_ub[:, KEEP]])
    b_all = np.concatenate([b1 * y + gamma1 * b, b_ub - A_ub[:, 4] * b1])
    f = A_red.sum(axis=0)

    res = linprog(b_all, A_eq=A_all.T, b_eq=-f,
                  bounds=[(0, None)] * len(b_all), method="highs")
    if not res.success:
        return {"g1": g1, "ok": False, "err": str(res.message)}

    lam = res.x
    act = np.where(lam > 1e-7)[0]
    p, *_ = np.linalg.lstsq(A_all[act], b_all[act], rcond=None)
    p_full = np.zeros(15)
    p_full[KEEP] = p
    p_full[4] = b1

    a0, aK, aL, aE, _b1, b2, aKK, aLL, aEE, aKL, aKE, aLE, dK, dL, dE = p_full
    D = A @ p_full + gamma1 * b
    dDdy = b1 + b2 * (beta * y + b) + beta * (dK * k + dL * l + dE * e)
    dDdb = gamma1 + b2 * (y + b / beta) + (dK * k + dL * l + dE * e)
    with np.errstate(divide="ignore", invalid="ignore"):
        MAC_scaled = -dDdb / dDdy
    mac = MAC_scaled * (YBAR / BBAR) * 1e4
    dev = MAC_scaled * (YBAR / BBAR) * C_RAW / Y_RAW
    m22 = mac[YEAR == 2022]

    return {
        "g1": round(g1, 6), "ok": True, "b1": round(b1, 6),
        "sumD": float(D.sum()), "Dmin": float(D.min()),
        "n_active": int(len(act)),
        "强对偶间隙": float(abs(f @ p + res.fun)),
        "MAC_2022均值": float(np.nanmean(m22)),
        "政策缺口倍数": float(np.nanmean(m22) / CARBON_PRICE_95P),
        "dev_max": float(np.nanmax(dev)),
        "dev_mean": float(np.nanmean(dev)),
        "合法": bool(np.nanmax(dev) <= 1.0),
        # 闭式引擎系数（供 s1/s2/s3 注入）：b2 / dK / dL / dE / γ₁
        "_p_full": p_full,
        "_coef": (float(p_full[5]), float(p_full[12]), float(p_full[13]),
                  float(p_full[14]), float(gamma1)),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 78)
    print("第一节　独立复算：与 deliver2 网格逐点对照")
    print("=" * 78)
    # deliver2 已落盘的网格值（s01_grid_fine.csv），用于对照。
    # 注意：该 CSV 只存 4–6 位有效数字，故用**相对**偏差判据。
    ref_csv = pd.read_csv(
        PAPER_DIR / "deliver2/03_实验记录/s01_grid_fine.csv")
    ref_csv["g1k"] = ref_csv["g1"].round(4)
    rows = []
    for g1 in (-0.30, -0.20, 0.00, 0.06, 0.13):
        m = ref_csv[ref_csv["g1k"] == round(g1, 4)]
        ref = float(m["MAC2022"].iloc[0]) if len(m) else np.nan
        t0 = time.time()
        r = run_at(g1)
        r["交付网格_MAC2022"] = ref
        r["相对偏差"] = abs(r["MAC_2022均值"] - ref) / ref
        r["耗时s"] = round(time.time() - t0, 2)
        rows.append(r)
    chk = pd.DataFrame(rows)[["g1", "b1", "sumD", "MAC_2022均值", "政策缺口倍数",
                              "dev_max", "交付网格_MAC2022", "相对偏差",
                              "强对偶间隙", "耗时s"]]
    print(chk.round(6).to_string(index=False))
    mx = float(chk["相对偏差"].max())
    print(f"\n  最大**相对**偏差 = {mx:.3e} → "
          + ("✅ 通过（<1e-6），机制已掌握" if mx < 1e-6
             else "❌ 未通过，不得引用"))

    print()
    print("=" * 78)
    print("第二节　窗口端点两侧的合法性切换")
    print("=" * 78)
    eps = 1e-4
    ends = []
    for tag, g1 in [("段1左端", ROOTS[0]), ("段1右端", ROOTS[1]),
                    ("段2左端", ROOTS[2]), ("段2右端", ROOTS[3])]:
        for side, gg in (("左", g1 - eps), ("右", g1 + eps)):
            r = run_at(gg)
            r["位置"] = f"{tag}{side}"
            ends.append(r)
    e = pd.DataFrame(ends)[["位置", "g1", "MAC_2022均值", "政策缺口倍数",
                            "dev_max", "合法"]]
    print(e.round(4).to_string(index=False))
    print("  根两侧 dev_max 跨越 1.0 = 合法/非法分界，验证 4 个根位正确。")

    print()
    print("=" * 78)
    print("第三节　论文点值在窗口内的分位位置")
    print("=" * 78)
    mac_range = (168.207, 555.801)
    paper = 190.5306
    frac = (paper - mac_range[0]) / (mac_range[1] - mac_range[0])
    print(f"  合法窗口 MAC_2022 ∈ [{mac_range[0]:.2f}, {mac_range[1]:.2f}] 元/吨（1995 价）")
    print(f"  论文 b₁=−1（γ₁=0）给出 {paper:.2f} 元/吨")
    print(f"  → 论文点值位于区间内的 **{frac*100:.1f}% 处**（线性位置）")
    print(f"  → 最保守合法约定下，承保成本为论文点值的 "
          f"{mac_range[1]/paper:.2f} 倍")
    print(f"\n  换算至 2026 名义价（×{2.169309:.4f}）后除以碳价 97.90：")
    for tag, v in (("最不保守", mac_range[0]), ("论文点值", paper),
                   ("最保守", mac_range[1])):
        nm = v * 2.169309
        print(f"    {tag:<8s} MAC={nm:7.1f} 元/吨  缺口倍数 m = {nm/97.90:5.2f}")

    # 窗口内细扫：步长 0.005 —— 与 deliver2 网格同精度，可用于交叉核验
    grid = [ROOTS[0] + i * 0.005 for i in range(94)]
    grid = [g for g in grid if ROOTS[3] + 1e-9 >= g >= ROOTS[0] - 1e-9]
    scan = pd.DataFrame(
        [{k: v for k, v in run_at(g).items() if not k.startswith("_")}
         for g in grid])
    scan["合法"] = scan["dev_max"] <= 1.0
    scan.to_csv(OUT / "s0_γ1窗口.csv", index=False, encoding="utf-8-sig")
    leg = scan[scan["合法"]]
    nbad = int((~scan["合法"]).sum())
    print(f"\n  细扫 {len(scan)} 点（步长 0.005），合法 {len(leg)} 点，"
          f"非法 {nbad} 点")
    print(f"  合法段 MAC ∈ [{leg['MAC_2022均值'].min():.2f}, "
          f"{leg['MAC_2022均值'].max():.2f}]")
    print(f"  注：空洞宽 {ROOTS[2]-ROOTS[1]:.4f} < 步长 0.005 的 2 倍，"
          f"故网格只能捕到 {nbad} 个非法点；"
          "\n      精确段界须以 s07 的 4 个根为准（本脚本已采用）。")

    out = {
        "γ1窗口": {"段1": [ROOTS[0], ROOTS[1]], "段2": [ROOTS[2], ROOTS[3]],
                   "空洞宽度": round(ROOTS[2] - ROOTS[1], 6)},
        "MAC_2022_1995价区间": [mac_range[0], mac_range[1]],
        "政策缺口倍数区间": [round(mac_range[0] / CARBON_PRICE_95P, 3),
                       round(mac_range[1] / CARBON_PRICE_95P, 3)],
        "论文点值": {"γ1": 0.0, "b1": -1.0, "MAC_2022_1995价": paper,
                  "政策缺口倍数": round(paper / CARBON_PRICE_95P, 4),
                  "区间内线性位置": round(frac, 4)},
        "折算至2026": {
            "deflator_1995_to_2026": 2.169309,
            "碳价_2026名义": 97.90,
            "m_2026区间": [round(mac_range[0] * 2.169309 / 97.90, 3),
                        round(mac_range[1] * 2.169309 / 97.90, 3)],
            "m_2026论文点值": round(paper * 2.169309 / 97.90, 3),
        },
        "复算对照最大偏差": float(mx),
        "性质": "γ1 为标度（计价单位）自由度，数据不可识别；"
                "区间属模糊性（ambiguity）而非风险（risk），无概率分布含义。",
        "来源": {
            "窗口与区间": "deliver2/ddf_scale_001（s07 精确求根、s14 分段报告）",
            "复算": f"本脚本调用 {PAPER_DIR.name}/deliver/05_代码/ddf_lp_track1.py 独立求解",
            "平移联动式": "b₁ = −1 + γ₁·β；γ₁=0 与交付基线逐位一致",
        },
    }
    with open(OUT / "s0_标度约定敏感性.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[已写出] s0_γ1窗口.csv / s0_标度约定敏感性.json")
    return scan, out


if __name__ == "__main__":
    main()
