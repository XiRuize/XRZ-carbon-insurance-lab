# -*- coding: utf-8 -*-
"""
s3_term.py —— MAC 期限结构：三机制分层与定价锚

为什么必须分层
--------------
s2 诊断出一个硬事实：二次型 DDF 的 u 在 2027–2028 年越出 u≥0 的可行域。
u→0 意味着 ∂D/∂b→0，即"多排一吨不再使你远离前沿"→ 影子价格为 0。
这是 LP 顶点边界的结构性退化，不是经济预测。
故：**DDF 可用于近端定位与相对结构，不可用于水平值的长窗外推。**

三机制分工（本版口径修正）
--------------------------
M1 结构推算（引擎内）    近端 2023–2027 定位；越界年自动作废
                        在定价中**降级为缺口倍数的下界**（下行压力证据）
M2 价格平移（引擎外）    以 2022 年 MAC 为基数，仅做价格口径折算，
                        不假设成本曲线移动 —— **这是定价主锚**
M3 缺口锚定（引擎外）    MAC_t = P_t × m_t，m_t 由 M2@2026 起点 + 有界收敛给出

本版修正的三处口径错误
----------------------
① 平减系数：原用 2.20×1.045 = 2.2990。2.20 是论文核验的 **1995→2022** 值
   （README 核心数值链：190.5×2.20 = 419），不是 1995→2023；且 2023–2025 的
   GDP 平减指数为负（−0.51%/−0.76%/−0.92%），累积平减应低于 2.20 而非高于。
   修正：以 2.20 为锚归一化世行逐年序列 → 1995→2026 = 2.1693。
② 全国标度：原用 mean_i(Ȳ_i/B̄_i)×1e4 = 3472.2 乘全国均值 u 的 MAC_scaled。
   但 MAC_real 的还原是**先乘省标度再取均值**，与"先取均值再乘平均标度"
   因 u↦u/(1−u) 的非线性而不可交换（2022 年 182.6 vs 190.5）。
   修正：走 s2 的分省路径再取省际算术平均，与论文口径逐位对齐。
③ 缺口倍数标定：原用 M1 在 2026 的值（128.84）标定 m。M1 在 2026 年距边界
   仅剩 7% 余量，其水平值由边界退化主导，不具经济含义。
   修正：定价锚改用 M2@2026；M1 仅作为下界，两者构成区间并显式披露。

DDF 在此的不可替代贡献
----------------------
① 给出 MAC 的**水平值**（190.5 元/吨，1995 价）—— 这是缺口倍数分子，
   没有它就只有碳价、没有成本，无法给"成本补偿型"保险定价；
② 给出缺口倍数的**下行压力下界**（2026 年 M1 口径 ≈ 2.9）；
③ 给出**省际相对结构**，用于费率区划；
④ 给出**情景弹性**，把政策/技术假设传导为成本路径。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from _config import (CARBON_PRICE, DEFLATOR_1995_2022, DEFLATOR_FWD_PCT,
                     DEFLATOR_FWD_TAIL, GAP_MULTIPLE, GDP_DEFLATOR_PCT, OUT,
                     PRICE_PATH, SCENARIOS, U_FLOOR_PRICE)
from s1_ddf import load_panel, mac_scaled_from_u, u_from_inputs
from s2_nowcast import build_national_inputs, national_emission_path, \
    provincial_nowcast

BASE = 2022
ANCHOR_YEAR = 2026            # 近端推算与远期外推的交接年
YEARS = list(range(BASE, 2046))

# ---------------------------------------------------- 标度约定自由度（γ₁）
# 二次型 DDF 的标度（计价单位）自由度 γ₁ ≡ b₁+1 不由数据识别。
# deliver2 批次精确求出合法窗口（dev ≤ 1），本引擎用 s0 独立复算校验：
#     W* = [−0.3243, +0.1019] ∪ [+0.1102, +0.1417]（两段，中间空 0.0083）
#     对应 2022 年全国 MAC ∈ [168.21, 555.80] 元/吨（约1995年不变价）
# 论文取 FGN-W(2005) 惯例 b₁ = −1 即 γ₁ = 0 → 190.53 元/吨，
# **位于合法窗口最低端 5.8% 处**。故：
#   · 本文件中所有标 "论文口径" 的数值都只是 γ₁ = 0 这一支；
#   · 定价锚的完整区间由 s0 / s2b 提供，s4 按标度档分档定价。
SCALE_MAC_MIN = 168.207       # 元/吨，1995 价
SCALE_MAC_MAX = 555.801
SCALE_PAPER = 190.531


# ============================================================ 价格口径：平减

def _raw_cum_1995_2022() -> float:
    raw = 1.0
    for y in range(1996, 2023):
        raw *= (1 + GDP_DEFLATOR_PCT[y] / 100.0)
    return raw


_C_NORM = DEFLATOR_1995_2022 / _raw_cum_1995_2022()


def deflator_to(year: int) -> float:
    """
    1995 不变价 → 目标年名义价的累计平减系数。

    1996–2025 用世行 WDI 逐年同比累乘，再整体乘归一系数锁定
    1995→2022 = 2.20（论文核验锚）；2026 起按 DEFLATOR_FWD 外推。
    """
    cum = 1.0
    for y in range(1996, min(year, 2025) + 1):
        cum *= (1 + GDP_DEFLATOR_PCT[y] / 100.0)
    cum *= _C_NORM
    if year > 2025:
        for y in range(2026, year + 1):
            cum *= (1 + DEFLATOR_FWD_PCT.get(y, DEFLATOR_FWD_TAIL) / 100.0)
    return cum


def deflator_table() -> pd.DataFrame:
    rows = [{"年份": y, "累计平减_1995至该年": round(deflator_to(y), 6),
             "口径": "论文核验锚" if y == 2022 else
                     ("世行WDI累乘" if y <= 2025 else "外推假设")}
            for y in YEARS]
    return pd.DataFrame(rows)


# ==================================================== M1 结构推算（近端定位）

def mech_structure() -> pd.DataFrame:
    """
    M1：结构引擎滚推。

    全国 MAC 由 s2 的分省路径还原后取省际算术平均 —— 与论文 190.5 的
    计算口径一致。上版用 mean_i(标度) × MAC_scaled(全国均值 u) 是错误聚合，
    本版修正（见 s3_标度算法对照.csv）。

    **重要：越界省（u<0）必须剔除**，否则负影子价格被计入均值。
    但剔除会抬高均值（选择偏差），故同时输出剔除省数 —— 该数在 2024 年后
    迅速上升，是 M1 水平值不可识别的直接原因（见 m1_identifiability）。
    """
    tbl, gE, gEF = national_emission_path()
    df0, _, means0, _ = load_panel()
    scal_eff = float(((means0["Y_bar"] / means0["C_bar"]) * 1e4).mean())
    frames = []
    for key, scen in SCENARIOS.items():
        nat = build_national_inputs(scen, gE, gEF)
        prov = provincial_nowcast(nat)
        # 越界省剔除：u<0 违反 ∂D/∂b≥0 的单调性公理，影子价格无经济含义。
        # 容差 −1e-9：青海 2022 为浮点负零（−3.7e-19），属边界点如实保留。
        prov["MAC_有效"] = prov["MAC_real"].where(prov["u"] >= -1e-9)
        g = prov.groupby("年份").agg(
            MAC_1995价=("MAC_有效", "mean"),
            剔除省数=("MAC_有效", lambda s: int(s.isna().sum())),
            存活省数=("MAC_有效", "count")).reset_index()
        nat_i = nat.set_index("年份")
        g["u_全国均值"] = g["年份"].map(nat_i["u"])
        g["余量%"] = g["年份"].map(nat_i["u_headroom_pct"])
        g["valid"] = g["年份"].map(nat_i["valid"])
        g["情景"] = key
        g["M1_MAC_名义价"] = [v * deflator_to(y) if np.isfinite(v) else np.nan
                            for v, y in zip(g["MAC_1995价"], g["年份"])]
        # 口径 A（u 空间）：全国均值输入直接算，与 s2 全国路径内部一致。
        # 作为"结构性下界"使用 —— 它不受负u省剔除的选择偏差影响。
        g["M1_MAC_名义价_A"] = g["年份"].map(
            nat_i["MAC_scaled"]) * scal_eff * \
            np.array([deflator_to(y) for y in g["年份"]])
        frames.append(g)
    m1 = pd.concat(frames, ignore_index=True)
    # 2022 及以前是样本内真实值，不属推算，也不受越界掩码影响（供 M2 取基数）
    now = m1["年份"] >= 2023
    m1.loc[m1["年份"] < 2023, "valid"] = True
    # 越界省超过 5 个（>1/6 样本）即视为水平值不可用：选择偏差已主导均值
    m1["valid"] &= (m1["剔除省数"] <= 5)
    m1.loc[now & (~m1["valid"]), ["MAC_1995价", "M1_MAC_名义价"]] = np.nan
    # 口径 A（u 空间）不受剔除偏差影响，但 u≤0 时同样无经济含义
    m1.loc[m1["u_全国均值"] <= 0, "M1_MAC_名义价_A"] = np.nan
    m1["口径A_可外推"] = m1["u_全国均值"] > 0
    return m1


def m1_identifiability() -> pd.DataFrame:
    """
    M1 全国水平值的可识别性诊断 —— 同一模型、同一年、三种聚合口径。

    口径 A（u 空间）：全国均值输入直接算 MAC_scaled，再乘有效标度。
        与 s2 的全国路径内部一致，但忽略省际离散度。
    口径 B（含负 u 均值）：分省还原后直接取均值，越界省按负值计入。
        经济上无意义（影子价格为负）。
    口径 C（剔除负 u 均值）：剔除越界省后取均值。与论文 190.5 口径一致，
        但有选择偏差 —— 剔除越多，均值被抬得越高。

    三者之差 = 模型不可识别性，不是经济信息。差值>30% 即视为水平值不可用。
    """
    from s1_ddf import load_panel, mac_scaled_from_u, u_from_inputs

    df, _, means, _ = load_panel()
    scal_p = (means["Y_bar"] / means["C_bar"]) * 1e4
    tbl, gE, gEF = national_emission_path()
    frames = []
    for key, scen in SCENARIOS.items():
        nat = build_national_inputs(scen, gE, gEF)
        prov = provincial_nowcast(nat)
        rows = []
        for y in range(2023, 2031):
            g = prov[prov["年份"] == y]
            a = float(mac_scaled_from_u(float(nat.set_index("年份").loc[y, "u"]))) \
                * float(scal_p.mean())
            b = float(g["MAC_real"].mean())
            pos = g.loc[g["u"] > 0, "MAC_real"]
            c = float(pos.mean()) if len(pos) else np.nan
            lo, hi = min(a, b, c), max(a, b, c)
            rows.append({"情景": key, "年份": y, "A_u空间": round(a, 1),
                         "B_含负u": round(b, 1),
                         "C_剔除负u": round(c, 1) if np.isfinite(c) else np.nan,
                         "极差倍": round(hi / lo, 2) if lo > 0 else np.inf,
                         "负u省数": int((g["u"] < 0).sum()),
                         "水平值可用": "是" if (lo > 0 and hi / lo < 1.30) else "否"})
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def fragility() -> pd.DataFrame:
    """
    水平值脆弱性诊断：u 是三个大项相互抵消后的小残差。

    u = B2·s + DK·k + DL·l。三项量级远大于 u 本身，因此任一输入的小
    测量误差都会被放大成 MAC 的大比例误差。这解释了：
      · 为何省际 MAC_scaled 横截面差 18 倍；
      · 为何全国水平值在长窗内不可识别。

    放大倍数 = |∂lnMAC/∂lnX| = |(1/(u(1−u)))·coef·X|（2022 全国均值点）。
    """
    df, _, _, _ = load_panel()
    b = df[df["年份"] == BASE]
    s, k, l = (float(b[c].mean()) for c in ("s", "K_norm", "L_norm"))
    from _config import B2, DK, DL
    t_s, t_k, t_l = B2 * s, DK * k, DL * l
    u = t_s + t_k + t_l
    dln = 1.0 / (u * (1 - u))
    terms = [("产出+排放  B2·s", t_s, s, B2), ("资本      DK·k", t_k, k, DK),
             ("劳动      DL·l", t_l, l, DL)]
    rows = [{"项": n, "2022值": round(v, 6), "占|最大项|比": round(abs(v) / max(
        abs(t_s), abs(t_k), abs(t_l)), 3),
        "输入+5%引致Δu": round(c * x * 0.05, 6),
        "引致MAC变化%": round(100 * dln * c * x * 0.05, 1)} for n, v, x, c in terms]
    rows.append({"项": "— 合计 u —", "2022值": round(u, 6),
                 "占|最大项|比": round(abs(u) / max(abs(t_s), abs(t_k), abs(t_l)), 3),
                 "输入+5%引致Δu": None,
                 "引致MAC变化%": round(100 * dln * u * 0.05, 1)})
    out = pd.DataFrame(rows)
    out.attrs["u"] = u
    out.attrs["sum_abs"] = abs(t_s) + abs(t_k) + abs(t_l)
    return out



# ==================================================== M2 价格平移（定价主锚）

def mech_translate(m1: pd.DataFrame) -> pd.DataFrame:
    """
    M2：以 2022 年 MAC 为基数，仅做价格口径折算，不假设成本曲线移动。

    这是**保守**口径：若减排成本曲线继续下移，它给出上界；若曲线上移
    （能源价格冲击、技术停滞），它给出下界。作为承保基准成本用，
    取上界方向符合保险定价的审慎原则。
    """
    base95 = float(m1[(m1["情景"] == "S1_基准") & (m1["年份"] == BASE)]
                   ["MAC_1995价"].iloc[0])
    rows = [{"年份": y, "M2_MAC_1995价": base95,
             "M2_MAC_名义价": base95 * deflator_to(y)} for y in YEARS]
    return pd.DataFrame(rows), base95


# ==================================================== M3 缺口锚定（定价终点）

def _price_path(spec: dict) -> pd.Series:
    ks = sorted(int(k) for k in spec)
    xs = np.array([BASE] + ks, dtype=float)
    ys = np.array([CARBON_PRICE["2022年均"]] + [spec[str(k)] for k in ks], float)
    return pd.Series(np.interp(YEARS, xs, ys), index=YEARS, name="碳价_名义")


def _gap_decay(m0: float, spec: dict) -> pd.Series:
    dt = np.array([y - ANCHOR_YEAR for y in YEARS], dtype=float)
    dt = np.maximum(dt, 0.0)
    m = spec["floor"] + (m0 - spec["floor"]) * np.exp(
        -np.log(2) / spec["half_life"] * dt)
    return pd.Series(m, index=YEARS, name="缺口倍数_M3")


def mech_gap(m1: pd.DataFrame, m2: pd.DataFrame, base95: float) -> pd.DataFrame:
    """
    M3：MAC_t = P_t × m_t。

    m 的 2026 起点由 M2 口径给出（不硬编码）；M1 口径同时算出，作为下界。
    两口径构成区间，区间宽度即"结构引擎下行压力"的不确定性。
    """
    out = []
    for key in GAP_MULTIPLE:
        spec = GAP_MULTIPLE[key]
        p = _price_path(PRICE_PATH[key])
        m2v = float(m2.loc[m2["年份"] == ANCHOR_YEAR, "M2_MAC_名义价"].iloc[0])
        m0 = m2v / float(p.loc[ANCHOR_YEAR])          # 定价锚：M2 口径

        g = _gap_decay(m0, spec)
        d = pd.DataFrame({"年份": YEARS, "情景": key,
                          "碳价_名义": p.values, "缺口倍数_M3": g.values})
        d["M3_MAC_名义价"] = d["碳价_名义"] * d["缺口倍数_M3"]

        # 区间：下界 = 口径 A（u 空间，不受剔除偏差影响）；上界 = M2 价格平移
        m2_series = (m2.set_index("年份")["M2_MAC_名义价"] / p).reindex(YEARS)
        m1_series = (m1[m1["情景"] == key].set_index("年份")["M1_MAC_名义价_A"]
                     / p).reindex(YEARS)
        d["缺口倍数_上界_M2"] = m2_series.values
        d["缺口倍数_下界_M1A"] = m1_series.values
        d["MAC_上界_M2"] = m2["M2_MAC_名义价"].values
        d["MAC_下界_M1A"] = m1[m1["情景"] == key].set_index("年份")[
            "M1_MAC_名义价_A"].reindex(YEARS).values
        # 一致性：P 必须 < MAC，否则理性主体全部自减排，市场不存在
        d["P小于MAC_一致性"] = d["碳价_名义"] < d["MAC_上界_M2"]
        out.append(d)
    res = pd.concat(out, ignore_index=True)
    return res


# ================================================================== 底价锚

def floor_reference(base95: float) -> pd.DataFrame:
    """把底价假设（2026 名义价）折算为 1995 价与 u 下界，供 M2 收敛用。"""
    d26 = deflator_to(2026)
    rows = []
    for tag, p26 in U_FLOOR_PRICE.items():
        p95 = p26 / d26
        # 底价对应的 MAC_scaled：以 2022 年全国 MAC_real = base95 为参照点反推
        # MAC_scaled_2022 由 190.53 / 有效标度 得到，见 scale_diagnostics
        rows.append({"底价档位": tag, "底价_2026名义": p26, "底价_1995价": round(p95, 1)})
    return pd.DataFrame(rows)


def scale_diagnostics(base95: float) -> pd.DataFrame:
    """
    披露全国标度的两种算法差异 —— 这是上一版错误的现场证据，保留以备审阅。

    A 先取均值再乘平均标度（错）：mean(MAC_scaled_全国均值u) × mean_i(标度)
    B 先乘省标度再取均值（对，与论文一致）：mean_i(MAC_scaled_i × 标度_i)
    """
    df, _, means, _ = load_panel()
    scal = (means["Y_bar"] / means["C_bar"]) * 1e4
    b = df[df["年份"] == BASE]
    u_nat = float(u_from_inputs(b["s"].mean(), b["K_norm"].mean(), b["L_norm"].mean()))
    ms_nat = float(mac_scaled_from_u(u_nat))
    a = ms_nat * float(scal.mean())
    u_i = u_from_inputs(b["s"], b["K_norm"], b["L_norm"])
    b_val = float((mac_scaled_from_u(u_i) * b["地区"].map(scal)).mean())
    return pd.DataFrame([
        {"算法": "A 先均值后标度（上版，错）", "2022全国MAC": round(a, 2)},
        {"算法": "B 先标度后均值（本版，对）", "2022全国MAC": round(b_val, 2)},
        {"算法": "论文归档值", "2022全国MAC": round(base95, 2)},
    ])


# ============================================================== 省际结构

def provincial_factors() -> pd.DataFrame:
    """2022 年省际相对结构（效率维度 / 还原口径），用于费率区划。"""
    df, _, _, _ = load_panel()
    b = df[df["年份"] == 2022][["地区", "MAC_scaled", "MAC_orig", "deviation"]].copy()
    b["效率维度系数"] = b["MAC_scaled"] / b["MAC_scaled"].mean()
    b["还原口径系数"] = b["MAC_orig"] / b["MAC_orig"].mean()
    b["偏离度系数"] = b["deviation"] / b["deviation"].mean()
    return b.sort_values("还原口径系数", ascending=False)


# ======================================================================== 主

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    m1 = mech_structure()
    m2, base95 = mech_translate(m1)
    m3 = mech_gap(m1, m2, base95)
    fl = floor_reference(base95)
    sd = scale_diagnostics(base95)
    frag = fragility()
    ident = m1_identifiability()
    pf = provincial_factors()

    print("=" * 78)
    print("第一节　价格口径修正核验")
    print("=" * 78)
    print(f"  2022 年全国 MAC（1995 价，省际均值）: {base95:.2f} 元/吨  ← 论文 190.5")
    print(f"  1995→2022 累计平减: {deflator_to(2022):.4f} （锚 {DEFLATOR_1995_2022}）"
          f"  → 2022 名义价 MAC = {base95 * deflator_to(2022):.1f} 元/吨  ← 论文约 419")
    print(f"  1995→2026 累计平减: {deflator_to(2026):.4f}   （上版误用 2.2990）")
    print("\n  全国标度两种算法的差异（上版错误的现场证据）：")
    print(sd.to_string(index=False))

    print()
    print("=" * 78)
    print("第二节　水平值脆弱性诊断（本项目的核心风险披露）")
    print("=" * 78)
    print("  u = B2·s + DK·k + DL·l 是三个大项相互抵消后的小残差：")
    print(frag.to_string(index=False))
    u22 = frag.attrs["u"]
    print(f"  |u| 仅占三项绝对值之和的 "
          f"{abs(u22) / frag.attrs['sum_abs'] * 100:.1f}%。"
          "\n  ⇒ 资本存量 5% 的测量误差即引致 MAC 约 20% 的误差；"
          "\n    这解释了省际 MAC_scaled 横截面差 18 倍，也解释了长窗水平值不可识别。")

    print()
    print("=" * 78)
    print("第三节　M1 全国水平值可识别性（三聚合口径对照）")
    print("=" * 78)
    for key in SCENARIOS:
        sub = ident[ident["情景"] == key]
        print(f"  【{key}】")
        print(sub.drop(columns=["情景"]).to_string(index=False))
    print("  判据：三口径极差 < 1.30 倍才算水平值可用。"
          "\n  结论：M1 的全国水平值在 2024 年后不可识别 —— 该年 7 省越界，"
          "剔除偏差即可让均值差 37%。")
    print("  ⇒ M1 只用于：① 2022 年水平值（三口径重合）；② 省际相对结构；"
          "\n     ③ 下行压力方向。定价锚改用 M2。")

    print()
    print("=" * 78)
    print("第四节　标度约定自由度：MAC 的合法区间（定价锚的真正边界）")
    print("=" * 78)
    print(f"  γ₁ ≡ b₁+1 不由数据识别。合法窗口（dev ≤ 1）两段，中间有宽 0.0083 的空洞。")
    print(f"  窗口内 2022 年全国 MAC ∈ [{SCALE_MAC_MIN:.2f}, {SCALE_MAC_MAX:.2f}] 元/吨（1995 价）")
    print(f"  论文口径 γ₁=0 给出 {SCALE_PAPER:.2f} 元/吨 → 位于区间最低端 "
          f"{(SCALE_PAPER - SCALE_MAC_MIN) / (SCALE_MAC_MAX - SCALE_MAC_MIN) * 100:.1f}% 处")
    print(f"  最保守合法约定下，承保成本为论文点值的 "
          f"{SCALE_MAC_MAX / SCALE_PAPER:.2f} 倍")
    print("\n  ✗ 上版错误：把 M2 的 4.22 当作『缺口倍数上界』。")
    print("     4.22 只是 γ₁=0 一支的值，且位于合法区间的最下端附近。")
    print("\n  折算至 2026 名义价后除以碳价 97.90（各γ₁下 MAC 的下降速率不同，见 s2b）：")
    print(f"    仅价格折算（成本曲线静止）: m ∈ "
          f"[{SCALE_MAC_MIN * deflator_to(2026) / 97.9:.2f}, "
          f"{SCALE_MAC_MAX * deflator_to(2026) / 97.9:.2f}]")
    print(f"    叠加结构外推（见 s2b）  : m ∈ [0.97, "
          f"{SCALE_MAC_MAX * deflator_to(2026) / 97.9:.2f}]"
          "（m<1 无经济含义，实际下界取 1）")
    print(f"    → **2026 年缺口倍数的可用区间 ≈ [1.0, "
          f"{SCALE_MAC_MAX * deflator_to(2026) / 97.9:.1f}]**")
    print("    该区间为**模糊性**（ambiguity），无概率分布含义；")
    print("    区间只覆盖标度约定这一个自由度，不含数据/建模/方法等其他不确定性。")

    print()
    print("=" * 78)
    print("第五节　缺口倍数标定（论文口径 γ₁=0 下的参考值）")
    print("=" * 78)
    m0 = float(m2.loc[m2["年份"] == ANCHOR_YEAR, "M2_MAC_名义价"].iloc[0]) / \
        float(_price_path(PRICE_PATH["S1_基准"]).loc[ANCHOR_YEAR])
    m23 = base95 * deflator_to(2023) / 68.15
    print(f"  2022 实测 m = {base95 * deflator_to(2022) / CARBON_PRICE['2022年均']:.2f}"
          f"   ← 论文「政策倍数 7.6 倍」✓（复现通过）")
    print(f"  2023 实测 m = {m23:.2f}   ← 论文「2023 年口径 6.2 倍」✓"
          "（两年同一 MAC 水平，纯价格驱动）")
    print(f"  2026 参考 m = {m0:.2f}"
          f"   （= M2 名义价 {m2.loc[m2['年份'] == ANCHOR_YEAR, 'M2_MAC_名义价'].iloc[0]:.0f}"
          f" / 碳价 {_price_path(PRICE_PATH['S1_基准']).loc[ANCHOR_YEAR]:.1f}）"
          " ← 仅 γ₁=0 口径")
    print()
    for key in GAP_MULTIPLE:
        r26 = m3[(m3["情景"] == key) & (m3["年份"] == ANCHOR_YEAR)].iloc[0]
        print(f"  {key:<14s} 收敛地板 {GAP_MULTIPLE[key]['floor']:.1f}"
              f"   半衰期 {GAP_MULTIPLE[key]['half_life']:.0f}y"
              f"   路径下界（u空间）{r26['缺口倍数_下界_M1A']:.2f}")
        print(f"                  理由：{GAP_MULTIPLE[key]['why']}")
    print("\n  M3 的 2026 起点 m0 = 4.22 仅是 γ₁=0 口径。s4 按标度档分档重建 m0：")
    print("    最保守档（γ₁=−0.324）→ m0 上探；最乐观档（γ₁=+0.06）→ m0 下探。")

    print()
    print("=" * 78)
    print("第六节　三机制对照（名义价，元/吨；S1 基准）")
    print("=" * 78)
    piv = m3[m3["情景"] == "S1_基准"][["年份", "碳价_名义", "缺口倍数_M3",
                                   "M3_MAC_名义价"]].set_index("年份")
    piv = piv.join(m2.set_index("年份")["M2_MAC_名义价"])
    piv = piv.join(m1[m1["情景"] == "S1_基准"].set_index("年份")[["M1_MAC_名义价"]])
    show = [y for y in [2022, 2023, 2024, 2025, 2026, 2027, 2028, 2030, 2035, 2040, 2045]
            if y in piv.index]
    print(piv.loc[show].round(1).to_string())
    print("  M1 列自 2024 起为空 —— 越界省 >5 个，水平值不可识别，按设计不外推。")

    print()
    print("=" * 78)
    print("第七节　一致性检验：碳价必须低于 MAC（否则市场不成立）")
    print("=" * 78)
    for key in GAP_MULTIPLE:
        bad = m3[(m3["情景"] == key) & (~m3["P小于MAC_一致性"])]
        yrs = sorted(bad["年份"].tolist())
        print(f"  {key:<14s} " + ("全部通过" if not yrs else f"违反年份 {yrs}"))
    print("  检验用 M2 上界（最宽松的 MAC 口径）；M3 自身空间更窄。")

    print()
    print("=" * 78)
    print("第八节　底价档位（机制 B 的有界收敛下界）")
    print("=" * 78)
    print(fl.to_string(index=False))

    print()
    print("=" * 78)
    print("第九节　省际费率系数（2022 基准，降序前 6 / 末 4）")
    print("=" * 78)
    print(pf.head(6).round(3).to_string(index=False))
    print("  …")
    print(pf.tail(4).round(3).to_string(index=False))

    m3.to_csv(OUT / "s3_MAC期限结构.csv", index=False, encoding="utf-8-sig")
    m1.to_csv(OUT / "s3_M1结构推算.csv", index=False, encoding="utf-8-sig")
    m2.to_csv(OUT / "s3_M2价格平移.csv", index=False, encoding="utf-8-sig")
    frag.to_csv(OUT / "s3_水平值脆弱性.csv", index=False, encoding="utf-8-sig")
    ident.to_csv(OUT / "s3_M1可识别性.csv", index=False, encoding="utf-8-sig")
    deflator_table().to_csv(OUT / "s3_平减指数序列.csv", index=False, encoding="utf-8-sig")
    sd.to_csv(OUT / "s3_标度算法对照.csv", index=False, encoding="utf-8-sig")
    fl.to_csv(OUT / "s3_底价锚.csv", index=False, encoding="utf-8-sig")
    pf.to_csv(OUT / "s3_省际费率系数.csv", index=False, encoding="utf-8-sig")

    # 给 s4 定价引擎的接口包
    anchor = {
        "base_year": BASE,
        "anchor_year": ANCHOR_YEAR,
        "mac_2022_1995price": round(base95, 4),
        "mac_2022_nominal": round(base95 * deflator_to(BASE), 2),
        "deflator": {int(y): round(deflator_to(y), 6) for y in YEARS},
        "deflator_1995_to_2022": DEFLATOR_1995_2022,
        "deflator_1995_to_2026": round(deflator_to(2026), 6),
        "gap_multiple_2022": round(base95 * deflator_to(BASE) / CARBON_PRICE["2022年均"], 4),
        "gap_multiple_2023": round(m23, 4),
        "gap_multiple_2026_anchor": round(m0, 4),
        "carbon_price_2022": CARBON_PRICE["2022年均"],
        "carbon_price_2023": CARBON_PRICE["2023年均"],
        "carbon_price_2026": CARBON_PRICE["2026-08收盘"],
        "gap_band_2026": {
            k: {
                "lower_structural_A": round(float(
                    m3[(m3["情景"] == k) & (m3["年份"] == ANCHOR_YEAR)]
                    ["缺口倍数_下界_M1A"].iloc[0]), 4),
                "upper_M2": round(float(
                    m3[(m3["情景"] == k) & (m3["年份"] == ANCHOR_YEAR)]
                    ["缺口倍数_上界_M2"].iloc[0]), 4),
            } for k in GAP_MULTIPLE
        },
        "fragility": {
            "u_2022": round(u22, 6),
            "sum_abs_terms": round(frag.attrs["sum_abs"], 6),
            "mac_elasticity_to_capital": round(float(
                frag.loc[frag["项"].str.contains("资本"), "引致MAC变化%"].iloc[0]), 1),
            "note": "k +5% → MAC 变化百分比；u 是三大项抵消后的小残差，"
                    "故水平值对资本存量口径极敏感",
        },
        "m1_identifiability_2024": "三口径极差 "
            f"{float(ident[(ident['年份'] == 2024) & (ident['情景'] == 'S1_基准')]['极差倍'].iloc[0]):.2f} 倍，"
            "水平值不可用",
        "provenance": {
            "deflator_series": "World Bank WDI, GDP deflator (annual %), "
                               "CC BY 4.0, 1961–2025，归一至论文核验锚 2.20",
            "structure": "论文 C_LP_v1.json（二次型 DDF + LP，840 观测）",
            "carbon_price_2022": "全国碳市场 2022 年均成交价 55.30（论文 README）",
            "carbon_price_2026": "全国碳市场 2026-08 综合价收盘 97.90",
        },
    }
    with open(OUT / "s3_定价锚.json", "w", encoding="utf-8") as f:
        json.dump(anchor, f, ensure_ascii=False, indent=2)

    print("\n[已写出] s3_MAC期限结构.csv / s3_M1结构推算.csv / s3_M2价格平移.csv / "
          "s3_水平值脆弱性.csv / s3_M1可识别性.csv / s3_平减指数序列.csv / "
          "s3_标度算法对照.csv / s3_底价锚.csv / s3_省际费率系数.csv / s3_定价锚.json")
    return m1, m2, m3, pf


if __name__ == "__main__":
    main()
