# -*- coding: utf-8 -*-
"""
s30_anchor_update_2025.py —— 「把定价锚从 2022 更新到 2025」的后果核算

背景（2026-10-04，用户提问）：
    用户注意到 deliver5 的 2023—2025 段不是纯外推，问「把价格更新到 2025 年会怎么样」。
    本脚本把该动作拆成可核验的三块：

    A 口径审计：2023—2025 在引擎里到底是不是「实测」
       —— 名义碳价路径 s3_term._price_path 是 np.interp(2022年均实测 → 2026锚)，
          中间年份全是插出来的，与 _config.CARBON_PRICE 里已有的实测值不符。
    B 锚切换：把 MAC 锚从 2022 观测值换成 2025 观测值后，
          MAC_REF / K / CAP 的水平变化，以及「尺度不变性」是否成立。
    C 可承保期：观测终点从 2022 移到 2025 后，距定义域边界还剩几年。

只读既有产物（s2b_各标度档MAC路径.csv、s29_*），不改任何既有脚本或归档。
用法：python s30_anchor_update_2025.py [--check]    # 退出码 0 = 断言全过
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _config import CARBON_PRICE, L1, OUT, PRICE_PATH  # noqa: E402
import s4_pricing as s4                                 # noqa: E402
from s3_term import deflator_to                          # noqa: E402

RHO = L1.get("capital_charge", 0.06)
PI_PROFIT = 0.08
DF_R = L1["discount"]
E_EXP_IDX = 0.12
Q_LIMIT = 0.995
N_EFF_BASE = 100
RHO_BAR_BASE = 0.30
N_PATH = L1["mc_paths"]
TERMS_S = [1, 3, 5]

# 档名映射：s2b 产物列名 → s4.TIERS_RAW 的档名（后者用于展示 γ₁）
S2B_COLS = ["段1左端_最保守", "段1偏左", "段1中段", "段1偏右",
            "论文口径", "段1右端_最低", "段2中点"]


def _tvar(x: np.ndarray, a: float = 0.99) -> float:
    q = np.quantile(x, a)
    t = x[x >= q]
    return float(t.mean()) if t.size else float(q)


def pv_rebased(sim: dict, n: int, period: int = 1, cap: float | None = None
               ) -> np.ndarray:
    """B1 年度重定基（复刻 s22.pv_rebased，仅把 CAP 参数化以便做锚敏感性）。"""
    I, Q = sim["I"], sim["Q"]
    cap = s4.CAP_AMOUNT if cap is None else float(cap)
    total = np.zeros(I.shape[1])
    yr = 0
    while yr < n:
        seg = min(period, n - yr)
        idx = np.arange(yr + 1, yr + seg + 1)
        K = s4.KAPPA_BASE * I[yr]
        pay = np.minimum(np.maximum(I[idx] - K[None, :], 0.0), cap) * Q[idx]
        total += (pay * (1.0 / (1.0 + DF_R) ** idx[:, None])).sum(axis=0)
        yr += seg
    return total


def price_one(sim: dict, n: int, cap: float | None) -> dict:
    """单档单期限：D=q75 免赔 + A=q99.5 截断 + e=12% + φ(100,0.30)。"""
    L = pv_rebased(sim, n, period=1, cap=cap)
    qs = {p: float(np.quantile(L, p)) for p in (0.75, Q_LIMIT)}
    D, A = qs[0.75], qs[Q_LIMIT]
    pay = np.minimum(np.maximum(L - D, 0.0), A - D)
    E = float(pay.mean())
    rm1 = _tvar(pay) - E
    phi = float(np.sqrt(RHO_BAR_BASE + (1.0 - RHO_BAR_BASE) / N_EFF_BASE))
    rm = rm1 * phi
    gp = (E + RHO * rm) / (1.0 - E_EXP_IDX - PI_PROFIT)      # 口径乙
    return {"E": E, "D": D, "A_q995": A, "层宽": A - D,
            "TVaR99": _tvar(pay), "RM": rm,
            "GP": gp, "ROL%": gp / max(A - D, 1e-9) * 100,
            "触发概率%": float((pay > 0).mean() * 100),
            "E_full_mean": float(L.mean())}


def main() -> int:
    fails: list[str] = []
    OUT.mkdir(parents=True, exist_ok=True)

    mac = pd.read_csv(OUT / "s2b_各标度档MAC路径.csv")
    mac = mac.set_index("年份")

    # ============================================ A 口径审计：名义碳价
    pp = PRICE_PATH["S1_基准"]
    ks = sorted(int(k) for k in pp)
    years_a = list(range(2022, 2027))
    interp = np.interp(years_a, np.array([2022] + ks, float),
                       np.array([CARBON_PRICE["2022年均"]] + [float(pp[str(k)]) for k in ks],
                                float))
    # _config 里已登记的实测值（键名非纯年份，逐个对齐）
    obs = {"2022年均": 55.30, "2023年均": 68.15,
           "2025年均": 62.36, "2025年末收盘": 74.63}
    obs_by_year = {2023: ("年均", obs["2023年均"]),
                   2025: ("年均", obs["2025年均"]),
                   2025.5: ("年末收盘", obs["2025年末收盘"])}
    rows_a = []
    for i, y in enumerate(years_a):
        rec = {"年份": y, "定价用_插值": round(float(interp[i]), 2)}
        if y in (2023, 2025):
            nm, v = obs_by_year[y]
            rec["实测_" + nm] = v
            rec["偏差%"] = round((interp[i] - v) / v * 100, 2)
        if y == 2025:
            rec["实测_年末收盘"] = obs["2025年末收盘"]
            rec["偏差_年末%"] = round((interp[i] - obs["2025年末收盘"])
                                  / obs["2025年末收盘"] * 100, 2)
        rows_a.append(rec)
    A_tab = pd.DataFrame(rows_a)

    print("=" * 78)
    print("第一节　口径审计：2023—2025 在引擎里是「实测」还是插值？")
    print("=" * 78)
    print(A_tab.to_string(index=False))
    print("\n  ⇒ 定价链 (s3_term._price_path) 用 np.interp 从 2022年均 55.30 直插 2026 的 97.90，")
    print("     中间年份（含 2025=87.25）全是插出来的；_config.CARBON_PRICE 里已登记的")
    print("     实测值（2023=68.15、2025年均=62.36、2025年末=74.63）**未进路径**。")
    print("     而 s3_term.py:438 的校验段恰恰用的是实测 68.15 —— 同一文件两个口径。")

    # 断言：插值与实测确实不符（若将来修正路径，此断言会失败并提示更新本脚本）
    if abs(float(interp[3]) - obs["2025年均"]) / obs["2025年均"] < 0.05:
        fails.append("2025 插值已与实测一致——路径可能已被修正，请复核本脚本前提")

    # ============================================ B 锚切换
    d26 = deflator_to(2026)
    d25 = deflator_to(2025)
    print("\n" + "=" * 78)
    print("第二节　锚切换：MAC_REF / K / CAP")
    print("=" * 78)
    print(f"  平减系数：1995→2025 = {d25:.4f}，1995→2026 = {d26:.4f}")

    rows_b = []
    anchors = {}
    for col in S2B_COLS:
        m22 = float(mac.loc[2022, col])
        m25 = float(mac.loc[2025, col])
        anchors[col] = {"MAC2022_1995价": m22, "MAC2025_1995价": m25}
        rows_b.append({
            "标度档": col, "MAC2022": round(m22, 2), "MAC2025": round(m25, 2),
            "跌幅%": round((m25 / m22 - 1) * 100, 2),
            "名义2026_锚2022": round(m22 * d26, 1),
            "名义2026_锚2025": round(m25 * d26, 1),
        })
    B_tab = pd.DataFrame(rows_b)
    print(B_tab.to_string(index=False))

    lo22, hi22 = B_tab["MAC2022"].min(), B_tab["MAC2022"].max()
    lo25, hi25 = B_tab["MAC2025"].min(), B_tab["MAC2025"].max()
    w22, w25 = hi22 / lo22, hi25 / lo25
    print(f"\n  标度区间（七档）：2022 [{lo22:.1f}, {hi22:.1f}] 宽比 {w22:.3f}")
    print(f"                    2025 [{lo25:.1f}, {hi25:.1f}] 宽比 {w25:.3f}"
          f"   → 放大 {(w25 / w22 - 1) * 100:+.1f}%")

    ref22 = float(mac.loc[2022, "论文口径"])
    ref25 = float(mac.loc[2025, "论文口径"])
    ref26 = float(mac.loc[2026, "论文口径"])          # 引擎 s2b 自身的 DDF 外推值
    MAC_REF_22 = ref22 * d26
    MAC_REF_25 = ref25 * d26
    MAC_REF_26X = ref26 * d26
    CAP_22 = s4.CAP_MULT * MAC_REF_22
    CAP_25 = s4.CAP_MULT * MAC_REF_25
    print(f"\n  论文口径锚：")
    print(f"    MAC_REF（2026 名义）  {MAC_REF_22:8.2f}  →  {MAC_REF_25:8.2f}"
          f"   ({MAC_REF_25 / MAC_REF_22 - 1:+.1%})")
    print(f"    K = 1.30 × MAC_REF   {s4.KAPPA_BASE * MAC_REF_22:8.2f}  →"
          f"  {s4.KAPPA_BASE * MAC_REF_25:8.2f}")
    print(f"    CAP = 2.0 × MAC_REF  {CAP_22:8.2f}  →  {CAP_25:8.2f}")
    print(f"    引擎当前硬编码 CAP_AMOUNT = {s4.CAP_AMOUNT:.2f}（=2×{MAC_REF_22:.2f}）")

    # —— 同口径三锚对照（全部折到 2026 名义元/吨，禁止跨口径比）——
    print(f"\n  三锚同口径对照（2026 名义，元/吨）：")
    for nm, v in (("锚2022（现行定价基准）", MAC_REF_22),
                  ("锚2025（最新观测）", MAC_REF_25),
                  ("引擎 s2b 自身外推 2026", MAC_REF_26X)):
        print(f"    {nm:24s} {v:8.2f}   相对锚2022 = {MAC_REF_22 / v:5.3f}×")
    print(f"    ⇒ 现行定价锚比引擎自身外推高 **{MAC_REF_22 / MAC_REF_26X:.2f} 倍**，")
    print(f"      等价于定价链隐含假设「MAC 的 1995 不变价值 2022→2026 完全不变」。")
    if 0.9 < MAC_REF_22 / MAC_REF_26X < 1.1:
        fails.append("定价锚已与引擎外推一致——内部矛盾可能已被修复，请复核本脚本前提")

    # 越界诊断（取 s29 已算好的结果，不重算）
    diag = pd.read_csv(OUT / "s29_定义域诊断_论文口径.csv")
    dsel = diag[diag["年份"].isin([2022, 2023, 2024, 2025, 2026, 2027, 2028])]
    print("\n  观测终点 vs 定义域边界（s29 论文口径，引擎 S1，g_K=5%）：")
    print(dsel[["年份", "u_=dDdb", "K除以样本上界", "MAC95", "符号"]]
          .to_string(index=False))
    print(f"    ⇒ 以 2022 为观测终点，距越界（K 超样本上界）尚有 4.25 年；")
    print(f"      以 2025 为观测终点，只剩 1.25 年（越界年 2026.25）。")

    # ============================================ C 定价对照
    print("\n" + "=" * 78)
    print("第三节　定价对照：锚 × CAP 口径 × 期限（B1 年度重定基，D=q75，A=q99.5）")
    print("=" * 78)
    crows = []
    for col in S2B_COLS:
        for n in TERMS_S:
            for an, mac95 in (("锚2022", anchors[col]["MAC2022_1995价"]),
                              ("锚2025", anchors[col]["MAC2025_1995价"])):
                tier = {"tag": col, "MAC2022_1995价": mac95,
                        "MAC2026_1995价_M1": None}
                seed = (s4.RNG_SEED + S2B_COLS.index(col) * 977
                        + TERMS_S.index(n) * 131 + (0 if an == "锚2022" else 7))
                sim = s4.simulate("S1_基准", tier, "M2_价格平移", N_PATH, seed)
                for cap_rule, capv in (("CAP固定", CAP_22),
                                       ("CAP随锚", CAP_22 if an == "锚2022" else CAP_25)):
                    r = price_one(sim, n, capv)
                    crows.append({"标度档": col, "期限": n, "锚": an,
                                  "CAP口径": cap_rule, **r})
    C = pd.DataFrame(crows)

    piv = C[(C["CAP口径"] == "CAP固定")].pivot_table(
        index=["标度档", "期限"], columns="锚", values="GP")
    piv["变化%"] = (piv["锚2025"] / piv["锚2022"] - 1) * 100
    print("\n  【CAP 固定 826.64】毛保费 GP（元/吨）")
    print(piv.round(1).to_string())

    piv2 = C.pivot_table(index=["标度档", "期限"], columns=["锚", "CAP口径"],
                         values="GP")
    print("\n  【CAP 固定 vs 随锚缩放】GP 对照（n=5，元/吨）")
    print(f"    {'标度档':14s} {'锚22/CAP固定':>14s} {'锚25/CAP固定':>14s}"
          f" {'锚25/CAP随锚':>14s} {'固定→随锚%':>12s}")
    for col in S2B_COLS:
        a = float(piv2.loc[(col, 5), ("锚2022", "CAP固定")])
        b = float(piv2.loc[(col, 5), ("锚2025", "CAP固定")])
        c = float(piv2.loc[(col, 5), ("锚2025", "CAP随锚")])
        print(f"    {col:14s} {a:14.1f} {b:14.1f} {c:14.1f}"
              f" {(c / b - 1) * 100:+11.1f}%")

    # 尺度不变性检验：锚2025 + CAP随锚  vs 锚2022 + CAP固定，比 ROL
    rol = C.pivot_table(index=["标度档", "期限"], columns=["锚", "CAP口径"],
                        values="ROL%")
    dev = []
    for col in S2B_COLS:
        for n in TERMS_S:
            a = rol.loc[(col, n), ("锚2022", "CAP固定")]
            b = rol.loc[(col, n), ("锚2025", "CAP随锚")]
            dev.append(abs(b - a))
    print(f"\n  ROL%（元/吨÷层宽）在两锚间的最大绝对偏差：{max(dev):.3f} pp")
    print("  ⇒ B1 重定基是比值结构，整条路径等比缩放 ⇒ 费率大体不变；")
    print("     路径**不是**等比缩放（缺口倍数向共同 floor=1.5 收敛）⇒ 仍有高阶偏差。")

    # ============================================ 尺度不变性：纯缩放实验
    print("\n" + "=" * 78)
    print("第四节　尺度不变性的解析检验（缺口倍数的 floor 是加法常数）")
    print("=" * 78)
    floor = s4.GAP_MULTIPLE["S1_基准"]["floor"]
    hl = s4.GAP_MULTIPLE["S1_基准"]["half_life"]
    P26 = float(PRICE_PATH["S1_基准"]["2026"])
    m0_22 = MAC_REF_22 / P26
    m0_25 = MAC_REF_25 / P26
    print(f"  m₀：锚2022 = {m0_22:.4f}   锚2025 = {m0_25:.4f}   floor = {floor}，半衰期 {hl}y")
    rows_d = []
    for dy in (0, 5, 10, 15):
        dec = 2 ** (-dy / hl)
        a = floor + (m0_22 - floor) * dec
        b = floor + (m0_25 - floor) * dec
        rows_d.append({"距2026": f"+{dy}y", "m_锚2022": round(a, 4),
                       "m_锚2025": round(b, 4), "比值": round(b / a, 4)})
    D_tab = pd.DataFrame(rows_d)
    print(D_tab.to_string(index=False))
    print("  ⇒ 比值从 0.494 单调升到 0.626 ⇒ **不是等比缩放**：锚下移后远期相对更高。")
    print("     因此『费率完全不变』只在 P 路径主导、缺口倍数恒定时成立；这里不成立。")

    # ============================================ 第五节 年度层口径（对外报价）
    print("\n" + "=" * 78)
    print("第五节　年度层口径对照（= 论文 v9 对外主报价口径，s25）")
    print("=" * 78)
    import s25_annual_rolling as s25                          # noqa: E402
    # seed 约定：s25 用 seeds[tag] = RNG_SEED + i*977，i 为 _lp_cache() 内序列序；
    # 「论文口径」在 s4.TIERS_RAW 中排第 5（i=4）⇒ 归档可复现 seed。
    SEED_PAPER = s4.RNG_SEED + 4 * 977
    rows_e = []
    for an, mac95, capv in (("锚2022（现行）", ref22, CAP_22),
                            ("锚2025/CAP固定", ref25, CAP_22),
                            ("锚2025/CAP随锚", ref25, CAP_25)):
        tier = {"tag": "论文口径", "MAC2022_1995价": mac95,
                "MAC2026_1995价_M1": None}
        sim = s4.simulate("S1_基准", tier, "M2_价格平移", N_PATH, SEED_PAPER)
        gps, first = [], None
        for t in range(1, 6):
            m = s25.annual_metrics(s25.year_pays(sim, t, cap=capv))
            if t == 1:
                first = m
            gps.append(m["GP_CoC"])
        rows_e.append({"口径": an, "首年D": first["D"], "首年A": first["A"],
                       "首年层宽": first["层宽"], "首年E": first["E_layer"],
                       "首年RM": first["RM_组合"], "首年GP": first["GP_CoC"],
                       "ROL%": first["ROL_CoC%"],
                       "3y名义合计": sum(gps[:3]), "5y名义合计": sum(gps)})
    E_tab = pd.DataFrame(rows_e)
    print(E_tab.round(1).to_string(index=False))
    gp_arch = 89.9
    dev_arch = abs(rows_e[0]["首年GP"] - gp_arch)
    print(f"\n  复现锚：本脚本锚2022 首年 GP = {rows_e[0]['首年GP']:.1f}"
          f"  vs 归档 s25_R2_年度层主表 = {gp_arch}  偏差 {dev_arch:.2f}")
    if dev_arch > 1.0:
        fails.append(f"年度层首年 GP 未复现归档 {gp_arch}（得 "
                     f"{rows_e[0]['首年GP']:.1f}，偏差 {dev_arch:.2f}）")
    print("  ⇒ 换锚到 2025 后，对外主报价（首年年度层 GP）同向下降，")
    print("     ROL 仍在 ~11% 量级 ⇒ 变动的是**保障水平**，不是费率。")

    # ============================================ 结论与写盘
    g5 = C[(C["期限"] == 5) & (C["标度档"] == "论文口径") & (C["CAP口径"] == "CAP固定")]
    gp22 = float(g5[g5["锚"] == "锚2022"]["GP"].iloc[0])
    gp25 = float(g5[g5["锚"] == "锚2025"]["GP"].iloc[0])
    print("\n" + "=" * 78)
    print("结论（论文口径档，n=5，CAP 固定口径）")
    print("=" * 78)
    print(f"  锚2022：GP = {gp22:7.1f} 元/吨     锚2025：GP = {gp25:7.1f} 元/吨"
          f"   ({gp25 / gp22 - 1:+.1%})")
    print("  ⇒ 换锚不会抬高保费（绝对水平大体随限额下移）；它改变的是**保障水平**。")

    out = {
        "口径审计_名义碳价": A_tab.to_dict("records"),
        "锚对照_各档MAC": B_tab.to_dict("records"),
        "标度区间宽比": {"2022": w22, "2025": w25,
                    "放大%": (w25 / w22 - 1) * 100},
        "锚水平": {"MAC_REF_2022": MAC_REF_22, "MAC_REF_2025": MAC_REF_25,
                 "MAC_REF_引擎外推2026": MAC_REF_26X,
                 "锚2022比引擎外推_倍": MAC_REF_22 / MAC_REF_26X,
                 "K_2022": s4.KAPPA_BASE * MAC_REF_22,
                 "K_2025": s4.KAPPA_BASE * MAC_REF_25,
                 "CAP_2022": CAP_22, "CAP_2025": CAP_25,
                 "引擎硬编码CAP_AMOUNT": s4.CAP_AMOUNT},
        "越界年": {"引擎S1口径": 2026.25, "deliver5层丙mid口径": 2025.11,
                 "锚2022可承保余地_年": 2026.25 - 2022,
                 "锚2025可承保余地_年": 2026.25 - 2025},
        "尺度不变性": {"floor": floor, "half_life": hl,
                   "m0_2022": m0_22, "m0_2025": m0_25,
                   "比值衰减表": D_tab.to_dict("records"),
                   "ROL最大偏差pp": float(max(dev))},
        "定价对照_论文口径n5_CAP固定": {"GP_锚2022": gp22, "GP_锚2025": gp25,
                                  "变化%": (gp25 / gp22 - 1) * 100},
        "年度层口径_对外报价": E_tab.to_dict("records"),
        "归档复现偏差_首年GP": dev_arch,
        "断言失败": fails,
    }
    with open(OUT / "s30_锚更新2025_核算.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    C.to_csv(OUT / "s30_锚更新2025_定价对照.csv", index=False, encoding="utf-8-sig")
    B_tab.to_csv(OUT / "s30_锚更新2025_档位对照.csv", index=False, encoding="utf-8-sig")
    E_tab.to_csv(OUT / "s30_锚更新2025_年度层对照.csv", index=False, encoding="utf-8-sig")

    if fails:
        print("\n✗ 断言失败：")
        for f_ in fails:
            print("   -", f_)
        return 1
    print("\n✓ 全部断言通过。产物：s30_锚更新2025_核算.json / _定价对照.csv / _档位对照.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
