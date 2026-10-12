# -*- coding: utf-8 -*-
"""
s2_nowcast.py —— 推算机制：填补 2023–2026 的数据缺口

问题
----
论文面板止于 2022（CEADs 碳数据滞后 2–3 年、省级年鉴滞后 1–2 年），
而定价必须发生在"今天"。需要一套可审计的机制把 MAC 推到当前年份。

机制
----
不重估 DDF，而是把已估好的参数当作一个**样本外打分函数**：
    u_t   = b2·s_t + dK·k_t + dL·l_t
    MAC_t = u_t/(1-u_t) · (Ȳ_i/B̄_i) · 1e4
只需把四个量纲化输入 k̃,l̃,ỹ,b̃ 推到目标年即可。四个输入的可得性不同：

    输入   官方发布滞后      推算通道
    ----   -------------     --------------------------------------
    ỹ      <1 季度（季报）   直接采用官方实际 GDP 增速
    k̃      1 季度（投资）    永续盘存法滚推：K_t = I_t + (1-δ)K_{t-1}
    l̃      1 年（年鉴）      就业趋势外推
    b̃      2–3 年（CEADs）   双通道：A 能耗×排放因子 / B GDP×碳强度

有效期约束
----------
u ≥ 0 是单调性公理（∂D/∂b ≥ 0）的直接要求，u → 0 即 MAC → 0。
2022 年 u = 0.0489，距边界仅剩 26.3% 的 k̃ 空间。本脚本逐年输出余量，
并在余量耗尽时**显式标记失效**，不输出臆造数值。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from _config import MACRO, OUT, PATH_RAW, SCENARIOS
from s1_ddf import COEF_PAPER, load_panel, mac_scaled_from_u, u_from_inputs

NOWCAST_YEARS = [2023, 2024, 2025, 2026]
HORIZON_END = 2045
BASE_YEAR = 2022


# ------------------------------------------------------------------ 全国输入

def national_emission_path():
    """
    双通道独立推算全国排放增速，并披露两者差异。

    通道 A（结构性）：排放增速 = 能耗增速 + 排放因子变化
        排放因子变化由能源结构驱动：煤炭占比每下降 1 个百分点，
        平均排放因子约下降 EF_PER_PCT_COAL。该弹性为外生设定，可调。
    通道 B（官方口径）：排放增速 = GDP 增速 × (1 + 万元GDP碳排放变化率)
        2024 年官方公报直接给出 −3.4%；2023 / 2025 官方未逐年公布。

    两通道在 2024 年重叠，构成独立交叉核验点。
    """
    EF_PER_PCT_COAL = 0.014        # 煤炭占比每降 1pct → 平均排放因子降 1.4%
    COAL_SHARE = {2022: 56.2, 2023: 55.3, 2024: 53.2, 2025: 51.4}   # %，统计公报

    E = pd.Series(MACRO["E_total"]).sort_index()
    gE = E.pct_change().dropna()
    coal = pd.Series(COAL_SHARE).sort_index()
    d_coal = coal.diff().dropna()
    gdp = pd.Series(MACRO["gdp_growth"])
    ci = pd.Series(MACRO["carbon_intensity"])          # 仅 2024 有官方值

    rows = []
    for y in (2023, 2024, 2025):
        gEF = d_coal[y] * EF_PER_PCT_COAL            # 百分点 × 弹性，负值
        gA = (1 + gE[y]) * (1 + gEF) - 1
        gB = (1 + gdp[y]) * (1 + ci[y]) - 1 if y in ci.index else np.nan
        rows.append({"年份": y, "能耗增速": gE[y], "煤炭占比变化(pct)": d_coal[y],
                     "排放因子变化": gEF, "通道A_排放增速": gA,
                     "通道B_排放增速": gB, "两通道差异(pct)": (gA - gB) * 100 if not np.isnan(gB) else np.nan,
                     "GDP增速(官方)": gdp[y]})
    tbl = pd.DataFrame(rows).set_index("年份")
    return tbl, gE, float(gEF)


def build_national_inputs(scen: dict, gE: pd.Series, gEF: float, coef=None):
    """构建 2023–2045 全国量纲化输入路径。

    coef：标度约定系数 (b2, dK, dL, dE, γ₁)。缺省 = 论文口径（γ₁=0）。
          越界诊断与 MAC 水平值均依赖该系数（见 s1_ddf 的说明）。
    """
    df, _, means, _ = load_panel()
    base = df[df["年份"] == BASE_YEAR]
    k0 = float(base["K_norm"].mean())
    l0 = float(base["L_norm"].mean())
    y0 = float(base["Y_norm"].mean())
    b0 = float(base["C_norm"].mean())

    def rate(spec, year):
        for a, b, r in spec:
            if a <= year <= b:
                return r
        return spec[-1][2]

    rows = [{"年份": 2022, "Y_norm": y0, "C_norm": b0, "K_norm": k0, "L_norm": l0,
             "gY": np.nan, "gB": np.nan, "gK": np.nan, "gL": np.nan}]
    y, b, k, l = y0, b0, k0, l0
    for yr in range(2023, HORIZON_END + 1):
        if yr <= 2025:                       # 近端：采用真实/推算数据
            gy = MACRO["gdp_growth"][yr]
            gb = (1 + gE[yr]) * (1 + gEF) - 1
            gk = 0.052                       # 2020–2022 观测资本增速均值
            gl = -0.004
        else:                                # 远端：情景
            gy = rate(scen["gdp_growth"], yr)
            gb = rate(scen["emission_growth"], yr)
            gk = rate(scen["capital_growth"], yr)
            gl = rate(scen["labor_growth"], yr)
        y *= (1 + gy); b *= (1 + gb); k *= (1 + gk); l *= (1 + gl)
        rows.append({"年份": yr, "Y_norm": y, "C_norm": b, "K_norm": k, "L_norm": l,
                     "gY": gy, "gB": gb, "gK": gk, "gL": gl})
    out = pd.DataFrame(rows)
    out["s"] = out["Y_norm"] + out["C_norm"]
    out["u"] = u_from_inputs(out["s"], out["K_norm"], out["L_norm"], coef=coef)
    out["MAC_scaled"] = mac_scaled_from_u(out["u"])
    b2, dk, dl, de, g1 = coef if coef is not None else COEF_PAPER
    out["k_crit"] = (g1 + b2 * out["s"] + dl * out["L_norm"]) / abs(dk)
    out["u_headroom_pct"] = 100 * (out["k_crit"] - out["K_norm"]) / out["K_norm"]
    out["valid"] = out["u"] > 0.005          # 余量阈值 0.5%（u 本身的绝对量级）
    return out


# ------------------------------------------------------------------ 省际分配

def provincial_nowcast(nat: pd.DataFrame, beta_converge: float = 0.0, coef=None):
    """
    把全国路径分配到 30 省。

    v1 口径（beta_converge = 0）：省际相对结构固定 —— 各省按全国增速同比例推进。
        明确假设，可被否证；用于先跑通定价链。
    beta_converge > 0：省际增速向全国均值 β 收敛（每省每年收敛该比例）。
    coef：标度约定系数，缺省 = 论文口径。
    """
    df, raw, means, _ = load_panel()
    b22 = df[df["年份"] == BASE_YEAR].set_index("地区")
    g = b22[["Y_norm", "C_norm", "K_norm", "L_norm"]].copy()
    g["u22"] = u_from_inputs(b22["s"], b22["K_norm"], b22["L_norm"], coef=coef)
    nat_idx = nat.set_index("年份")
    rows = []
    for prov, r in g.iterrows():
        for yr in nat_idx.index:
            n = nat_idx.loc[yr]
            rows.append({"地区": prov, "年份": int(yr),
                         "Y_norm": r.Y_norm * n.Y_norm / nat_idx.loc[BASE_YEAR, "Y_norm"],
                         "C_norm": r.C_norm * n.C_norm / nat_idx.loc[BASE_YEAR, "C_norm"],
                         "K_norm": r.K_norm * n.K_norm / nat_idx.loc[BASE_YEAR, "K_norm"],
                         "L_norm": r.L_norm * n.L_norm / nat_idx.loc[BASE_YEAR, "L_norm"]})
    p = pd.DataFrame(rows)
    p["s"] = p["Y_norm"] + p["C_norm"]
    p["u"] = u_from_inputs(p["s"], p["K_norm"], p["L_norm"], coef=coef)
    p["MAC_scaled"] = mac_scaled_from_u(p["u"])
    # 还原为元/吨：乘以 Ȳ_i / B̄_i（省际样本均值，1995–2022 固定）
    scale = (means["Y_bar"] / means["C_bar"]) * 1e4
    p["MAC_real"] = p["MAC_scaled"] * p["地区"].map(scale)
    return p


# ------------------------------------------------------------------ 敏感性

def sensitivity(nat: pd.DataFrame, dk_pct: float, coef=None):
    """资本增速整体平移 ±dk_pct，输出 MAC 路径的上下带。"""
    df, _, _, _ = load_panel()
    base = df[df["年份"] == BASE_YEAR]
    k, y, b, l = (float(base[c].mean()) for c in
                  ("K_norm", "Y_norm", "C_norm", "L_norm"))
    nat_i = nat.set_index("年份")
    rows = []
    for yr in nat_i.index:
        if yr <= BASE_YEAR:
            continue
        n = nat_i.loc[yr]
        gk0 = n.get("gK", np.nan)
        gk = (gk0 if not np.isnan(gk0) else 0.05) + dk_pct
        if yr <= 2025:
            gk = 0.052 + dk_pct
        k *= (1 + gk)
        y = n.Y_norm; b = n.C_norm; l = n.L_norm
        s = y + b
        u = float(u_from_inputs(s, k, l, coef=coef))
        rows.append({"年份": int(yr), "MAC_scaled": float(mac_scaled_from_u(u))
                     if u > 0 else np.nan, "u": u})
    return pd.DataFrame(rows).set_index("年份")


# ------------------------------------------------------------------------ 主

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tbl, gE, gEF = national_emission_path()
    print("=" * 78)
    print("第一节　排放推算：双通道独立交叉核验")
    print("=" * 78)
    print(tbl.round(4).to_string())
    print(f"\n  通道A 采用的排放因子年降幅（2025 口径）: {gEF*100:.2f}%")
    d24 = tbl.loc[2024, "两通道差异(pct)"]
    print(f"  2024 年两通道差异: {d24:.2f} 个百分点  ← 唯一有官方碳口径的年份，"
          "差异即口径不确定性")

    all_nat, all_prov, all_sens = [], [], []
    for key, scen in SCENARIOS.items():
        nat = build_national_inputs(scen, gE, gEF)
        nat["情景"] = key
        all_nat.append(nat)
        prov = provincial_nowcast(nat)
        prov["情景"] = key
        all_prov.append(prov)
        for dk, tag in [(-0.01, "低资本增速"), (0.0, "基准"), (0.01, "高资本增速")]:
            s = sensitivity(nat, dk)
            s["情景"] = key
            s["资本增速平移"] = tag
            all_sens.append(s)

    NAT = pd.concat(all_nat, ignore_index=True)
    PROV = pd.concat(all_prov, ignore_index=True)
    SENS = pd.concat(all_sens, ignore_index=True)

    print()
    print("=" * 78)
    print("第二节　全国 MAC 推算路径（元/吨，1995 年不变价）")
    print("=" * 78)
    # 全国 MAC_real：逐省均值（与论文口径一致）
    _raw = pd.read_excel(PATH_RAW)
    _m = _raw.groupby("地区")[["Y_raw", "C_raw"]].mean()
    scal = _m["Y_raw"] / _m["C_raw"] * 1e4
    PROV["MAC_real_full"] = PROV["MAC_scaled"] * PROV["地区"].map(scal)
    # 失效年后的数值不具经济含义（u<0 违反 ∂D/∂b≥0 的单调性公理），置为 NaN。
    # 容差 1e-9：青海 2022 的影子价格为浮点负零（-3.7e-19），属边界点如实报告，
    # 不得与真实越界混淆（该点剔除会使 2022 全国均值从 190.5 虚增到 197.1）。
    PROV.loc[PROV["MAC_scaled"] < -1e-9, "MAC_real_full"] = np.nan
    piv = (PROV.groupby(["情景", "年份"])["MAC_real_full"].mean().unstack(0))
    show = [y for y in [2022, 2023, 2024, 2025, 2026, 2027, 2028, 2030, 2035, 2040]
            if y in piv.index]
    print(piv.loc[show].round(1).to_string())
    print("  （NaN = 结构引擎已越出 u≥0 的可行域，数值不具经济含义，不外推）")

    print()
    print("=" * 78)
    print("第三节　结构引擎有效期诊断（u 距 0 的余量）")
    print("=" * 78)
    for key in SCENARIOS:
        sub = NAT[NAT["情景"] == key]
        bad = sub[~sub["valid"]]
        first = int(bad["年份"].min()) if len(bad) else None
        h26 = float(sub.loc[sub["年份"] == 2026, "u_headroom_pct"].iloc[0])
        print(f"  {key:<14s} 越界首年 {first}   2026 年余量 {h26:6.1f}%")

    print()
    print("=" * 78)
    print("第四节　2026 推算结果（近端，数据缺口填补）")
    print("=" * 78)
    p26 = PROV[(PROV["年份"] == 2026) & (PROV["情景"] == "S1_基准")].sort_values(
        "MAC_real_full", ascending=False)
    print(f"  全国 MAC 均值（1995 价）: {p26['MAC_real_full'].mean():.1f} 元/吨")
    print(f"  全国 MAC 均值（2026 价，×2.2×1.045）: {p26['MAC_real_full'].mean()*2.2*1.045:.1f} 元/吨")
    print("  省际前 5 / 后 5：")
    for _, r in pd.concat([p26.head(5), p26.tail(5)]).iterrows():
        print(f"    {r['地区']:<4s} {r['MAC_real_full']:7.1f} 元/吨")

    NAT.to_csv(OUT / "s2_全国输入路径.csv", index=False, encoding="utf-8-sig")
    PROV.to_csv(OUT / "s2_分省推算路径.csv", index=False, encoding="utf-8-sig")
    SENS.to_csv(OUT / "s2_敏感性带.csv", index=False, encoding="utf-8-sig")
    tbl.to_csv(OUT / "s2_排放双通道.csv", encoding="utf-8-sig")
    print("\n[已写出] s2_全国输入路径.csv / s2_分省推算路径.csv / "
          "s2_敏感性带.csv / s2_排放双通道.csv")
    return NAT, PROV, SENS, gEF


if __name__ == "__main__":
    main()
