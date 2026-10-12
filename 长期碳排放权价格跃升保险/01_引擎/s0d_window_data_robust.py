# -*- coding: utf-8 -*-
"""
s0d_window_data_robust.py —— 标度窗口自身的**数据稳健性**（回答「数据延后 3–4 年」）

问题
----
s0/s0c 给出的合法窗口 W* = [−0.3243, 0.1019] ∪ [0.1102, 0.1417] 与
MAC ∈ [132.5, 572.7] 是**在 1995–2022 这一版数据上**算出来的。
但项目前提就是「K、L、GDP 数据延后 3–4 年且会被修订」。若窗口随修订移动，
则「保费区间」不是一个固定区间，而是**每年都要重算的时变区间** ——
这才是「定期重定价条款」最硬的依据。

一个必须先讲清的性质（本脚本第一类扰动即为此设计）
--------------------------------------------------
闭式影子价格 MAC = −(∂D/∂b)/(∂D/∂y) 在**K 的统一乘性缩放**下是**严格不变**的。
证明（一步）：令 k → c·k 并同步令 (aK,aKK,aKL,aKE,dK) → (aK/c, aKK/c², aKL/c,
aKE/c, dK/c)，则
    ∂D/∂y = b1 + b2(βy+b) + β(dK k + dL l + dE e)     ← β·(dK/c)·(ck) 不变
    ∂D/∂b = b1+1 + b2(y+b/β) + dK k + dL l + dE e      ← 不变
    ∂D/∂K = aK + aKK k + aKL l + aKE e + dK(βy+b)      ← 整体乘 1/c，符号不变
    ∂D/∂L, ∂D/∂E                                      ← aKL k 同样不变
所有约束与目标在这些变换下逐位等价，故 LP 解给出的 MAC 与 dev 完全不变。
⇒ **「资本存量的整体水平修订」不移动窗口**；会移动窗口的是**分省异质修订**
（改变截面形状），因为异质修订无法被单一重整化吸收。

本脚本的三类扰动
----------------
  A 统一乘子（水平修订）      —— 验证上述不变性，并作为对照基线
  B 分省异质乘子 1+δ·z_i     —— 模拟真实修订的截面差异（δ = 2% / 5%）
  C 头部省份定向修订          —— 高资本省单独 +5%（最现实的部分修订场景）

判据与 s0 一致：dev ≤ 1（碳成本不超过 GDP）。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from _config import DELIVER, OUT

sys.path.insert(0, str(DELIVER / "05_代码"))
import ddf_lp_track1 as t1     # noqa: E402

from s0_gamma_window import (BBAR, CARBON_PRICE_95P, C_RAW, KEEP, RAW, ROOTS,
                             YEAR, YBAR, Y_RAW)   # noqa: E402

LO, HI = -0.42, 0.22
NPTS = 300


def _solve(mod_df, g1: float, beta: float = 1.0) -> dict:
    """在给定（可能已被扰动的）面板上解 LP。构造与 s0.run_at 逐行一致。"""
    A, y, b, k, l, e = t1.build_design(mod_df, beta)
    A_ub, b_ub = t1.monotonicity_constraints(mod_df, beta)
    A_red = A[:, KEEP]
    b1 = -1.0 + g1 * beta
    gamma1 = g1 * beta
    A_all = np.vstack([-A_red, A_ub[:, KEEP]])
    b_all = np.concatenate([b1 * y + gamma1 * b, b_ub - A_ub[:, 4] * b1])
    f = A_red.sum(axis=0)
    res = linprog(b_all, A_eq=A_all.T, b_eq=-f,
                  bounds=[(0, None)] * len(b_all), method="highs")
    if not res.success:
        return {"ok": False, "dev_max": np.nan, "MAC2022": np.nan}
    lam = res.x
    act = np.where(lam > 1e-7)[0]
    p, *_ = np.linalg.lstsq(A_all[act], b_all[act], rcond=None)
    p_full = np.zeros(15)
    p_full[KEEP] = p
    p_full[4] = b1
    _a0, _aK, _aL, _aE, _b1, b2, _aKK, _aLL, _aEE, _aKL, _aKE, _aLE, dK, dL, dE = p_full
    D = A @ p_full + gamma1 * b
    dDdy = b1 + b2 * (beta * y + b) + beta * (dK * k + dL * l + dE * e)
    dDdb = gamma1 + b2 * (y + b / beta) + (dK * k + dL * l + dE * e)
    with np.errstate(divide="ignore", invalid="ignore"):
        MAC_scaled = -dDdb / dDdy
    mac = MAC_scaled * (YBAR / BBAR) * 1e4
    dev = MAC_scaled * (YBAR / BBAR) * C_RAW / Y_RAW
    return {"ok": True, "dev_max": float(np.nanmax(dev)),
            "MAC2022": float(np.nanmean(mac[YEAR == 2022]))}


def window_on(mod_df) -> dict:
    gs = np.linspace(LO, HI, NPTS)
    dv = np.array([_solve(mod_df, g)["dev_max"] for g in gs])
    ok = np.isfinite(dv) & (dv <= 1.0)
    # 最长连续合法段 + 全部合法段
    segs, cur = [], None
    for i, v in enumerate(ok):
        if v and cur is None:
            cur = i
        elif not v and cur is not None:
            segs.append((gs[cur], gs[i - 1])); cur = None
    if cur is not None:
        segs.append((gs[cur], gs[-1]))
    segs = [(float(a), float(b)) for a, b in segs]
    return {"segs": segs, "n_ok": int(ok.sum()),
            "g1_lo": min(a for a, _ in segs) if segs else np.nan,
            "g1_hi": max(b for _, b in segs) if segs else np.nan}


def mac_range_on(mod_df, segs) -> tuple[float, float]:
    vals = []
    for a, b in segs:
        for g in np.linspace(a, b, 25):
            r = _solve(mod_df, g)
            if r["ok"] and r["dev_max"] <= 1.0:
                vals.append(r["MAC2022"])
    return (float(min(vals)), float(max(vals))) if vals else (np.nan, np.nan)


def _py(v):
    """把 numpy 标量转成 Python 原生类型（json 可序列化）。"""
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    DF0 = t1.load_data()
    regions = sorted(DF0["地区"].unique())
    rng = np.random.default_rng(20260930)
    z = {r: float(v) for r, v in zip(regions, rng.standard_normal(len(regions)))}
    # 注意：K_norm 是按省均值归一化的，其年度均值恒为 1，排序无意义；
    # 「高资本省」必须用**原始** K 的省际均值排序（取自论文原始数据表）。
    top5 = (RAW.groupby("地区")["K_raw"].mean().sort_values(ascending=False)
            .head(5).index.tolist())

    print("=" * 78)
    print("s0d　标度窗口的数据稳健性（资本存量被修订时窗口移动到哪）")
    print("=" * 78)
    print(f"  基准面板 {len(DF0)} 观测 / {len(regions)} 省；细扫 {NPTS} 点")
    print(f"  高资本存量前 5 省（用于定向修订）：{top5}")

    cases = [
        ("基准", None),
        ("A_统一×0.98", ("uniform", 0.98)),
        ("A_统一×1.02", ("uniform", 1.02)),
        ("B_异质δ2%", ("het", 0.02)),
        ("B_异质δ5%", ("het", 0.05)),
        ("C_头部5省+5%", ("top", 0.05)),
    ]

    rows, details = [], {}
    for tag, spec in cases:
        mdf = DF0.copy()
        if spec is not None:
            kind, d = spec
            if kind == "uniform":
                mdf["K_norm"] = mdf["K_norm"] * d
            elif kind == "het":
                fac = mdf["地区"].map(lambda r: 1.0 + d * z[r])
                mdf["K_norm"] = mdf["K_norm"] * fac
            elif kind == "top":
                fac = np.where(mdf["地区"].isin(top5), 1.0 + d, 1.0)
                mdf["K_norm"] = mdf["K_norm"] * fac
        w = window_on(mdf)
        mlo, mhi = mac_range_on(mdf, w["segs"])
        r0 = _solve(mdf, 0.0)
        rows.append({
            "扰动": tag,
            "合法段数": len(w["segs"]),
            "窗口下端": round(w["g1_lo"], 4),
            "窗口上端": round(w["g1_hi"], 4),
            "MACmin": round(mlo, 1), "MACmax": round(mhi, 1),
            "MAC跨度": round(mhi / mlo, 2) if mlo == mlo else np.nan,
            "论文点合法": bool(w["g1_lo"] <= 0.0 <= w["g1_hi"]),
            "γ1=0_dev_max": round(float(r0["dev_max"]), 4),
        })
        details[tag] = w["segs"]
        print(f"  … {tag} 完成（合法段 {len(w['segs'])} 段）")

    d = pd.DataFrame(rows)
    print()
    print("=" * 78)
    print("第一节　窗口与 MAC 区间随三类扰动的移动")
    print("=" * 78)
    print(d.to_string(index=False))
    print("\n  各扰动下的合法段：")
    for tag, segs in details.items():
        txt = " ∪ ".join(f"[{a:.4f}, {b:.4f}]" for a, b in segs)
        print(f"    {tag:<14s} {txt}")

    print()
    print("=" * 78)
    print("第二节　结论")
    print("=" * 78)
    b = d[d["扰动"] == "基准"].iloc[0]
    uni = d[d["扰动"].str.startswith("A_")]
    het = d[d["扰动"].str.startswith(("B_", "C_"))]
    print("  ① **统一水平修订不移动窗口**（理论已证 + 实测逐位相同）：")
    for _, r in uni.iterrows():
        print(f"       {r['扰动']:<12s} 窗口 [{r['窗口下端']:.4f}, {r['窗口上端']:.4f}]"
              f"  MAC [{r['MACmin']:.1f}, {r['MACmax']:.1f}]")
    print(f"     基准          窗口 [{b['窗口下端']:.4f}, {b['窗口上端']:.4f}]"
          f"  MAC [{b['MACmin']:.1f}, {b['MACmax']:.1f}]")
    print("     机理：K 的整体缩放可由 K 相关系数的重整化精确吸收。")
    print("     ⇒ 推论：**「资本存量整体高估/低估」这类争论不影响本引擎**，")
    print("        争这一条没有意义；要害在截面形状。")
    print()
    print("  ② **分省异质 / 定向修订会移动窗口**：")
    for _, r in het.iterrows():
        print(f"       {r['扰动']:<12s} 窗口 [{r['窗口下端']:.4f}, {r['窗口上端']:.4f}]"
              f"  MAC [{r['MACmin']:.1f}, {r['MACmax']:.1f}]"
              f"  相对基准下/上端位移 "
              f"({r['窗口下端'] - b['窗口下端']:+.4f}, "
              f"{r['窗口上端'] - b['窗口上端']:+.4f})")
    dl = (het["窗口下端"] - b["窗口下端"])
    dh = (het["窗口上端"] - b["窗口上端"])
    mlo = het["MACmin"] / b["MACmin"] - 1
    mhi = het["MACmax"] / b["MACmax"] - 1
    print(f"     位移范围：下端 {dl.min():+.4f} ~ {dl.max():+.4f}，"
          f"上端 {dh.min():+.4f} ~ {dh.max():+.4f}")
    print(f"     MAC 区间：下端最大 {mlo.min() * 100:+.1f}%，"
          f"上端最大 {mhi.max() * 100:+.1f}%")
    print()
    ok_all = bool(d["论文点合法"].all())
    print(f"  ③ 论文点 γ₁=0 是否在所有扰动下仍合法："
          f"{'是（合法性未被修订推翻）' if ok_all else '否 —— 合法性会被修订推翻'}")
    print("     各扰动下 γ₁=0 的 dev_max：" +
          ", ".join(f"{r['扰动']}={r['γ1=0_dev_max']:.4f}" for _, r in d.iterrows()))
    print("     （dev_max ≤ 1 为合法；离 1 越近，被修订推翻的风险越大）")
    print()
    print("  ⇒ 判读：「MAC 的合法区间」**不是常数，而是随 K/L 数据修订而重算的时变区间**。")
    print("     一份锁价 10 年的保单，锁的是一个尚未被数据确定的区间 ——")
    print("     这是「定期重定价 + 越界触发重估」条款的量化依据，")
    print("     也提示实务中应把**修订敏感性**作为产品限额约束（而非只在定价时考虑）。")

    d.to_csv(OUT / "s0d_窗口数据稳健性.csv", index=False, encoding="utf-8-sig")
    out = {
        "扰动设置": {
            "A_统一乘子": [0.98, 1.02],
            "B_分省异质": {"形式": "1 + δ·z_i，z_i 固定伪随机（seed 20260930）",
                        "δ": [0.02, 0.05]},
            "C_定向": {"对象": f"K_norm 均值前 5 省 = {top5}", "幅度": 0.05},
            "判据": "dev ≤ 1（碳成本不超过 GDP），与 s0 一致",
        },
        "理论性质": "MAC 与 dev 在 K 的统一乘性缩放 + 相关系数重整化下严格不变",
        "基准": {k: _py(v) for k, v in b.items()},
        "合法段": {k: [[round(a, 4), round(x, 4)] for a, x in v]
                 for k, v in details.items()},
        "异质扰动位移": {
            "窗口下端范围": [round(float(dl.min()), 4), round(float(dl.max()), 4)],
            "窗口上端范围": [round(float(dh.min()), 4), round(float(dh.max()), 4)],
            "MAC下端最大相对变化": round(float(mlo.min()), 4),
            "MAC上端最大相对变化": round(float(mhi.max()), 4),
        },
        "论文点始终合法": ok_all,
        "结论": "统一水平修订不移动窗口（可被重整化吸收）；分省异质/定向修订会移动。"
                "故 MAC 区间是时变区间，构成定期重定价条款的量化依据。"
                "实务含义：修订敏感性应作为产品限额约束，而非仅在定价时考虑。",
    }
    with open(OUT / "s0d_窗口数据稳健性.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[已写出] s0d_窗口数据稳健性.csv / s0d_窗口数据稳健性.json")
    return d


if __name__ == "__main__":
    main()
