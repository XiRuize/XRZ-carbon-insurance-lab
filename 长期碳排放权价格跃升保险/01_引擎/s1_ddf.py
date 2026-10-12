# -*- coding: utf-8 -*-
"""
s1_ddf.py —— DDF 影子价格引擎（闭式解）+ 840 观测逐位校验 + 弹性与驱动分解

推导（本脚本第一节给出，并逐点验证）
--------------------------------------
由论文式(6)的平移性质参数限制，二次型 DDF 的对数导数结构为：

    ∂D/∂y = b1 + b2·s + dK·k + dL·l + dE·e
    ∂D/∂b = (b1+1) + b2·s + dK·k + dL·l + dE·e
    其中 s = y + b，b1 = -1（锚定）

    令 u := b2·s + dK·k + dL·l + dE·e
    则 ∂D/∂y = u - 1，∂D/∂b = u

    影子价格（论文式13、15）：
        MAC_scaled = -(∂D/∂b)/(∂D/∂y) = u / (1 - u)
        MAC_real   = MAC_scaled · (Ȳ_i / B̄_i) · 1e4      [元/吨，1995 年不变价]

结论：MAC 只由单一标量 u 决定，u 只依赖 (s, k, l, e) 四个量纲化输入。
      本估计中 dE = aE = aEE = aKE = aLE = 0（LP 顶点边界），故 e 不入 u。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from _config import (B2, DE, DDF, DELIVER, DK, DL, OUT, PATH_NORM,
                     PATH_PARAMS, PATH_RAW)


def mac_scaled_from_u(u):
    """由 u 计算量纲化影子价格。u=1 处发散（∂D/∂y=0），实际数据远离该点。"""
    u = np.asarray(u, dtype=float)
    return u / (1.0 - u)


# 标度约定注入（γ₁ 自由度）
# -------------------------
# 闭式解 u = b2·s + dK·k + dL·l 是 **γ₁ = 0（b₁ = −1）** 这一标度约定下的产物。
# 一般情形 u 还含一个常数项 γ₁：
#     u = γ₁ + b2·s + dK·k + dL·l + dE·e
# 故本模块所有函数支持 coef 注入，以便在 γ₁ 的合法窗口内逐档重算。
# coef = (b2, dK, dL, dE, gamma1)；默认取 _config 中论文口径（γ₁ = 0）。
COEF_PAPER = (B2, DK, DL, DE, 0.0)


def u_from_inputs(s, k, l, e=None, coef=None):
    """由量纲化输入计算 u。coef 缺省时用论文口径（γ₁ = 0）。"""
    b2, dk, dl, de, g1 = coef if coef is not None else COEF_PAPER
    u = g1 + b2 * np.asarray(s, float) + dk * np.asarray(k, float) \
        + dl * np.asarray(l, float)
    if e is not None:
        u = u + de * np.asarray(e, float)
    return u


def dD_from_inputs(s, b, k, l, e=None, coef=None, beta: float = 1.0):
    """
    β **一般情形**的两个偏导（闭式）。返回 (∂D/∂b, ∂D/∂y)。

    推导
    ----
    方向向量 g = (1, −β)，平移性质给出 b₁ = −1 + γ₁·β，记 G := γ₁·β。
    由二次型参数化（s0 的 run_at 构造）：

        ∂D/∂b = G + b₂·(y + b/β) + dK·k + dL·l + dE·e
        ∂D/∂y = b₁ + b₂·(βy + b) + β·(dK·k + dL·l + dE·e)

    以面板可直接算的量改写（s := y + b，故 y = s − b）：

        y + b/β = s − b·(1 − 1/β)
        βy + b  = β·s − b·(β − 1)

    β = 1 时两式同时退化为 u 与 u − 1，与 mac_scaled_from_u 一致。
    数值验证：与 s0 的 run_at 逐档对照，相对偏差 ≤ 1e-15（见 s0c）。
    """
    b2, dk, dl, de, G = coef if coef is not None else COEF_PAPER
    s = np.asarray(s, float)
    b = np.asarray(b, float)
    k = np.asarray(k, float)
    l = np.asarray(l, float)
    e_term = 0.0 if e is None else de * np.asarray(e, float)
    dDdb = G + b2 * (s - b * (1.0 - 1.0 / beta)) + dk * k + dl * l + e_term
    dDdy = (G - 1.0) + b2 * (beta * s - b * (beta - 1.0)) \
        + beta * (dk * k + dl * l + e_term)
    return dDdb, dDdy


def mac_scaled_general(s, b, k, l, e=None, coef=None, beta: float = 1.0):
    """β 一般情形的量纲化影子价格 MAC_scaled = −(∂D/∂b)/(∂D/∂y)。"""
    dDdb, dDdy = dD_from_inputs(s, b, k, l, e, coef, beta)
    return -dDdb / dDdy


def coef_from_gamma(g1: float, beta: float = 1.0):
    """在标度约定 γ₁ 下独立求解 LP，提取闭式引擎所需系数。

    返回 (b2, dK, dL, dE, γ₁)。其余参数不影响影子价格。
    该函数把 s0_gamma_window 的 LP 求解结果接到 s1 的闭式引擎上。
    """
    from s0_gamma_window import run_at
    r = run_at(g1, beta)
    if not r.get("ok"):
        raise RuntimeError(f"γ₁={g1} 处 LP 求解失败：{r.get('err')}")
    p = r["_p_full"]
    return (p[5], p[12], p[13], p[14], p[4] + 1.0)


# ------------------------------------------------------------------ 数据装载

def load_panel():
    raw = pd.read_excel(PATH_RAW)
    nrm = pd.read_excel(PATH_NORM)
    with open(PATH_PARAMS, encoding="utf-8") as f:
        arch = json.load(f)
    obs = pd.DataFrame(arch["derived_by_obs"]).rename(
        columns={"prov": "地区", "year": "年份"})
    df = nrm.merge(obs, on=["地区", "年份"], how="inner")
    df["s"] = df["Y_norm"] + df["C_norm"]
    # 省际均值（1995-2022）—— 样本外推算必须沿用同一标度
    means = raw.groupby("地区")[["Y_raw", "C_raw", "K_raw", "L_raw"]].mean()
    means.columns = ["Y_bar", "C_bar", "K_bar", "L_bar"]
    return df, raw, means, arch


# ------------------------------------------------------------------ 第一节：校验

def verify(df, arch):
    """闭式解 vs 归档派生量逐点比对。"""
    u = u_from_inputs(df["s"], df["K_norm"], df["L_norm"], df["E_norm"])
    mac = mac_scaled_from_u(u)
    rec = {
        "n_obs": int(len(df)),
        "max_abs_err_mac_scaled": float(np.abs(mac - df["MAC_scaled"]).max()),
        "max_rel_err_mac_scaled": float(
            (np.abs(mac - df["MAC_scaled"]) / df["MAC_scaled"].abs()).max()),
        "corr": float(np.corrcoef(mac, df["MAC_scaled"])[0, 1]),
        "sum_D_reported": arch["objective_sumD"],
        "certificate": {k: arch["certificate"][k] for k in
                        ("active_rank", "n_params", "nnls_resid", "strong_duality_gap")},
    }
    # 量纲化 = 原始 / 省际均值，逐格核对
    raw = pd.read_excel(PATH_RAW)
    for c, cn, m in [("Y_raw", "Y_norm", "Y_bar"), ("C_raw", "C_norm", "C_bar"),
                     ("K_raw", "K_norm", "K_bar"), ("L_raw", "L_norm", "L_bar")]:
        mu = raw.groupby("地区")[c].mean()
        chk = raw[c] / raw["地区"].map(mu)
        rec[f"rescale_err_{cn}"] = float(np.abs(chk - df[cn]).max())
    # u 的边界余量
    m22 = df["年份"] == 2022
    u22 = float(u[m22].mean())
    k22 = float(df.loc[m22, "K_norm"].mean())
    rec["u_2022"] = u22
    rec["k_2022"] = k22
    rec["mac_scaled_2022_pointwise_mean"] = float(mac[m22].mean())
    rec["mac_scaled_2022_of_mean_u"] = float(mac_scaled_from_u(u22))
    rec["mac_orig_2022_mean"] = float(df.loc[m22, "MAC_orig"].mean())
    rec["u_headroom_pct"] = float(100 * u22 / abs(DK * k22))
    return rec, u, mac


# --------------------------------------------------- 第二节：弹性与驱动分解

def elasticities(df, year=2022):
    """在给定年份的全国均值点上计算 ∂ln(MAC_scaled)/∂ln(x)。"""
    g = df[df["年份"] == year]
    s, k, l = g["s"].mean(), g["K_norm"].mean(), g["L_norm"].mean()
    u = float(B2 * s + DK * k + DL * l)
    dln_mac_du = 1.0 / (u * (1.0 - u))
    rows = []
    for name, val, coef in [("产出+排放 s=y+b", s, B2),
                            ("资本 k", k, DK),
                            ("劳动 l", l, DL)]:
        rows.append({
            "输入": name, "2022年全国均值": round(val, 6), "u的系数": round(coef, 6),
            "点弹性 ∂lnMAC/∂lnX": round(dln_mac_du * coef * val, 4),
            "1%误差引致MAC误差%": round(abs(dln_mac_du * coef * val), 2),
        })
    return u, dln_mac_du, pd.DataFrame(rows)


def decomposition(df, y0, y1):
    """Δu 的三源分解：产出+排放 / 资本 / 劳动。"""
    g = df.groupby("年份").agg(s=("s", "mean"), k=("K_norm", "mean"),
                              l=("L_norm", "mean"))
    g["u"] = u_from_inputs(g.s, g.k, g.l)
    a, z = g.loc[y0], g.loc[y1]
    parts = {
        f"Δs（{y0}→{y1}）": (z.s - a.s, B2 * (z.s - a.s)),
        f"Δk（{y0}→{y1}）": (z.k - a.k, DK * (z.k - a.k)),
        f"Δl（{y0}→{y1}）": (z.l - a.l, DL * (z.l - a.l)),
    }
    rows = [{"驱动源": k, "输入变化": round(v[0], 6), "对Δu的贡献": round(v[1], 6),
             "占总变化%": None} for k, v in parts.items()]
    tot = z.u - a.u
    for r in rows:
        r["占总变化%"] = round(100 * r["对Δu的贡献"] / tot, 1) if tot else None
    rows.append({"驱动源": "合计", "输入变化": None, "对Δu的贡献": round(sum(v[1] for v in parts.values()), 6),
                 "占总变化%": 100.0})
    rows.append({"驱动源": "实际Δu（校验）", "输入变化": None, "对Δu的贡献": round(tot, 6), "占总变化%": 100.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 边界有效期

def headroom(df):
    """逐年计算 u=0 边界的剩余空间，给出结构引擎的有效期诊断。"""
    g = df.groupby("年份").agg(s=("s", "mean"), k=("K_norm", "mean"),
                              l=("L_norm", "mean"))
    g["u"] = u_from_inputs(g.s, g.k, g.l)
    g["k_crit"] = (B2 * g.s + DL * g.l) / abs(DK)   # u=0 的临界 k
    g["余量"] = g.k_crit - g.k
    g["余量%"] = 100 * g["余量"] / g.k
    return g[["s", "k", "l", "u", "k_crit", "余量", "余量%"]]


# ------------------------------------------------------------------------ 主

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df, raw, means, arch = load_panel()
    rec, u, mac = verify(df, arch)
    df["u_calc"] = u
    df["MAC_scaled_calc"] = mac

    print("=" * 78)
    print("第一节　闭式引擎校验（对照论文归档派生量）")
    print("=" * 78)
    print(f"  观测数                      : {rec['n_obs']}")
    print(f"  MAC_scaled 最大绝对偏差     : {rec['max_abs_err_mac_scaled']:.3e}")
    print(f"  MAC_scaled 最大相对偏差     : {rec['max_rel_err_mac_scaled']:.3e}")
    print(f"  相关系数                    : {rec['corr']:.12f}")
    print(f"  论文 ΣD                     : {rec['sum_D_reported']:.6f}")
    print(f"  对偶证书 活跃行秩/自由参数  : {rec['certificate']['active_rank']}/{rec['certificate']['n_params']}")
    for k in ("Y_norm", "C_norm", "K_norm", "L_norm"):
        print(f"  量纲化还原误差 {k:<8s}    : {rec['rescale_err_' + k]:.3e}")
    print(f"  2022 全国 u                 : {rec['u_2022']:.6f}")
    print(f"  2022 全国 MAC_scaled（逐省均值）: {rec['mac_scaled_2022_pointwise_mean']:.6f}")
    print(f"  2022 全国 MAC_real（元/吨） : {rec['mac_orig_2022_mean']:.2f}")
    print(f"  u 距边界的余量              : {rec['u_headroom_pct']:.1f}%  ← 结构引擎有效期指标")

    u22, dln, el = elasticities(df, 2022)
    print()
    print("=" * 78)
    print("第二节　输入弹性（2022 全国均值点）")
    print("=" * 78)
    print(f"  ∂ln(MAC_scaled)/∂u = 1/(u(1-u)) = {dln:.3f}   ← u 越接近 0，水平值越不确定")
    print(el.to_string(index=False))

    dec = decomposition(df, 2009, 2022)
    print()
    print("=" * 78)
    print("第三节　2009→2022 的 MAC 下降：驱动分解")
    print("=" * 78)
    print(dec.to_string(index=False))

    hr = headroom(df)
    print()
    print("=" * 78)
    print("第四节　u=0 边界余量诊断（结构引擎有效期）")
    print("=" * 78)
    print(hr.loc[[1995, 2005, 2009, 2015, 2019, 2020, 2021, 2022]].round(4).to_string())

    # 落盘
    df[["地区", "年份", "Y_norm", "C_norm", "K_norm", "L_norm", "s",
        "u_calc", "MAC_scaled_calc", "MAC_scaled", "MAC_orig", "deviation"]].to_csv(
        OUT / "s1_观测级MAC复核.csv", index=False, encoding="utf-8-sig")
    el.to_csv(OUT / "s1_输入弹性表.csv", index=False, encoding="utf-8-sig")
    dec.to_csv(OUT / "s1_MAC下降驱动分解.csv", index=False, encoding="utf-8-sig")
    hr.round(6).to_csv(OUT / "s1_边界余量诊断.csv", encoding="utf-8-sig")
    with open(OUT / "s1_校验记录.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print("\n[已写出] s1_观测级MAC复核.csv / s1_输入弹性表.csv / "
          "s1_MAC下降驱动分解.csv / s1_边界余量诊断.csv / s1_校验记录.json")
    return df, means, rec


if __name__ == "__main__":
    main()
