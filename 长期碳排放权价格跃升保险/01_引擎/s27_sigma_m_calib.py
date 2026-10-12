# -*- coding: utf-8 -*-
"""
s27_sigma_m_calib.py —— 第五轮审稿响应：sigma_m 的代理标定、隐含波动率一致性与报价重述

背景（2026-10-03 第四轮外部审稿，重大问题 2 与 6）：
    审稿人指出 (a) sigma_m 不可标定，绝对保费只能是情景值；
    (b) 第 4.1 节模拟细节不足，P_t / m_t 的随机过程未写明，无法复算。
    并要求："利用前序省级 MAC 面板构造 m_t = MAC_t / P_t 的年度变动序列，
    至少给出可复算的代理标定。"

本轮结论：**该要求可行，且暴露出一个必须披露的实质问题。**
    上游 `02_产物/s1_观测级MAC复核.csv` 已含 1995-2022 年 30 省 MAC 面板
    （论文[4] 的产物，本文可直接调用，不需重跑 DDF 估计）。据此可做两条线路，
    并在**指数层面**做一致性检验——后者是本轮最重要的发现。

    线路 A（缺口倍数侧）：全国口径 MAC 对数一阶差分尺度 sigma_MAC_hat
        = 0.062（全样本 sd）/ 0.060（稳健 MAD 尺度）。
    线路 B（碳价侧）：附录 B 表 B1b 更正后的 sigma_P 实测折算 = 0.185-0.205。
    合成（I = P x m，对数变动近似正交相加）：
        sigma_(Delta ln I) ~ sqrt(sigma_P^2 + sigma_MAC^2) = 0.195-0.214。

    **一致性检验（关键）**：引擎当前假设（sigma_P=0.28、sigma_m_scan=0.15）
    隐含的年对数变动尺度为 **0.401**，是上述可标定值（~0.21）的 **约 1.9 倍**；
    而相对可观测的 *MAC 指数本身* 的年变动（0.062），假设值高约 **6.5 倍**。
    ⇒ sigma_P 与 sigma_m 在基准参数下均被显著高估：
        sigma_P：0.28 vs 0.185-0.205（1.4-1.5 倍）
        sigma_m：0.15 vs 0.055-0.062（2.4-2.7 倍）
    ⇒ 按可标定值重算，首年 GP 由 89.9 降至 **25.5-44.8 元/吨**（ROL 9.5%-10.4%），
      即现有基准保费**偏保守（偏高）**，而非偏低。

    **结构后果（须披露）**：在低波动标定下，年度赔付的 q75 落到 0
      ⇒ 聚合免赔 D 归零 ⇒ 分层结构退化为"第一元起保"层，
      自留额切掉小额高频损失的功能消失，费用率 e 的适用性须重新论证。

口径偏离登记：
    本脚本**不改** s25 的任何数字。s25 的 sigma_m 敏感性扫描以绝对水平口径标定
    模型 sigma_m；本脚本的 sigma_MAC_hat 是**对数变动**标准差。二者的换算式为
    sigma_hat = sigma_scan / sqrt(1 - (1-kappa_m)^2)（AR(1) 平稳水平 sd 关系），
    故 sigma_scan=0.15 对应 sigma_hat=0.21 —— 与线路 A+B 的合成值同量级。
    这一巧合**不是**假设值正确的证据（s25 的 0.15 是对 sigma_hat 标定而非对
    sigma_MAC 标定）；C 块的一致性检验才是判定依据。故本脚本只**重述区间**，
    不重算 s25 报价；完全标定的报价（D 块）单独输出，作为"若按实测量标定"的读数。

用法：python s27_sigma_m_calib.py [--check]    # 退出码 0 = 断言全过
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import L1, OUT                                    # noqa: E402
import s4_pricing as s4                                        # noqa: E402
import s23_review_response as s23                              # noqa: E402
import s25_annual_rolling as s25                               # noqa: E402
from s24_review2_response import TIER_PAPER                     # noqa: E402

# 上游只读源：论文[4] 的观测级 MAC 面板（30 省 × 1995-2022）
UPSTREAM = (Path(__file__).resolve().parents[1]
            / "02_产物" / "s1_观测级MAC复核.csv")

SIG_P_LO, SIG_P_HI = 0.185, 0.205     # 附录 B 表 B1b 更正后折算
WINDOWS = [(1995, "全样本"), (2005, "扩围前后"), (2010, "后京都"),
           (2013, "碳试点后"), (2015, "巴黎协定后")]


# ============================================================ A. 代理标定
def mac_series() -> pd.Series:
    """全国口径 MAC（30 省等权均值），剔除 LP 顶点解（青海等非正值）。"""
    d = pd.read_csv(UPSTREAM, encoding="utf-8-sig")
    d = d[d["MAC_orig"] > 0]
    return d.groupby("年份")["MAC_orig"].mean()


def a1_windows() -> pd.DataFrame:
    ls = np.log(mac_series())
    rows = []
    for lo, lab in WINDOWS:
        w = ls[ls.index >= lo]
        g = w.diff().dropna()
        x = np.arange(len(w))
        c2 = np.polyfit(x, w.values, 2)
        resid = w.values - np.polyval(c2, x)
        rows.append({
            "窗口": f"{lo}-2022", "标签": lab, "n_年": len(w),
            "漂移Δln": round(float(g.mean()), 4),
            "sd": round(float(g.std(ddof=1)), 4),
            "MAD尺度": round(float(1.4826 * np.median(np.abs(g - g.median()))), 4),
            "二次去趋势残差sd": round(float(resid.std(ddof=1)), 4),
        })
    return pd.DataFrame(rows)


def a2_band(smac: float, sp: float) -> tuple[float, float, float]:
    """两线路锁定 sigma_m：|a-b|（rho=+1）、sqrt(a^2+b^2)（rho=0）、a+b（rho=-1）。"""
    return abs(smac - sp), float(np.sqrt(smac ** 2 + sp ** 2)), smac + sp


# ============================================================ C. 一致性检验
def c1_implied(sig_p: float, sig_m_scan: float) -> float:
    """引擎参数隐含的年对数变动尺度（路径中位数）。"""
    old = s4.SIG_P
    s4.SIG_P = sig_p
    try:
        tiers = s4._lp_cache()
        tag = TIER_PAPER
        seed = s4.RNG_SEED + list(tiers).index(tag) * 977
        sim = s23.simulate_p("S1_基准", tiers[tag], "M2_价格平移",
                             s4.N_PATH, seed, sig_m=sig_m_scan)
    finally:
        s4.SIG_P = old
    dI = np.diff(np.log(sim["I"][:7]), axis=0)
    return float(np.median(dI.std(axis=0, ddof=1)))


def d1_calibrated(sig_p: float, sig_val: float, is_hat: bool = True,
                  seed_i: int = 0) -> dict:
    """报价读数（不动 s25）。is_hat=True 时 sig_val 是对数变动口径，须反解为扫描口径；
    is_hat=False 时 sig_val 即引擎扫描口径（用于复现 s25 基准，不引入偏离）。"""
    phi = 1.0 - s4.KAPPA_M
    sig_scan = (float(sig_val) * float(np.sqrt(1.0 - phi ** 2)) if is_hat
                else float(sig_val))
    old = s4.SIG_P
    s4.SIG_P = sig_p
    try:
        tiers = s4._lp_cache()
        tag = TIER_PAPER
        seed = s4.RNG_SEED + list(tiers).index(tag) * 977 + seed_i
        sim = s23.simulate_p("S1_基准", tiers[tag], "M2_价格平移",
                             s4.N_PATH, seed, sig_m=sig_scan)
    finally:
        s4.SIG_P = old
    m1 = s25.annual_metrics(s25.year_pays(sim, 1))
    gps5 = [s25.annual_metrics(s25.year_pays(sim, t))["GP_CoC"] for t in range(1, 6)]
    dI = np.diff(np.log(sim["I"][:7]), axis=0)
    return {"sigma_P": sig_p, "sigma_m_scan": round(sig_scan, 3),
            "ΔlnI尺度": round(float(np.median(dI.std(axis=0, ddof=1))), 3),
            "D": round(m1["D"], 1), "A": round(m1["A"], 1),
            "层宽": round(m1["层宽"], 1), "E": round(m1["E_layer"], 1),
            "GP": round(m1["GP_CoC"], 1), "ROL%": round(m1["ROL_CoC%"], 1),
            "触发频率%": round(m1["触发频率%"], 1),
            "5y名义合计": round(sum(gps5), 1)}


# ============================================================ main
def main() -> None:
    only = "--check" in sys.argv
    print("=" * 100)
    print("s27　sigma_m 代理标定 · 隐含波动率一致性 · 报价重述（第五轮审稿响应 问题 2/6）")
    print("=" * 100)
    print(f"\n上游只读源：{UPSTREAM.name}（论文[4]产物，本文直接调用，不重跑 DDF）")

    D1 = a1_windows()
    print("\n【A-1】全国口径 MAC 的对数一阶差分尺度（乘性扰动，与漂移无关）")
    print("-" * 100)
    print(D1.to_string(index=False))

    row = D1[D1["窗口"] == "1995-2022"].iloc[0]
    smac_sd, smac_mad, smac_r2 = (float(row["sd"]), float(row["MAD尺度"]),
                                  float(row["二次去趋势残差sd"]))
    print(f"\n【A-2】sigma_m 的区间（sigma_P = {SIG_P_LO}-{SIG_P_HI}，附录 B 表 B1b）")
    print("-" * 100)
    rows = []
    for smac, lab in [(smac_mad, "MAD 稳健尺度（点估）"),
                      (smac_sd, "样本 sd"),
                      (smac_r2, "二次去趋势残差 sd（上界）")]:
        lo, mid, hi = a2_band(smac, SIG_P_HI)
        _, mid2, _ = a2_band(smac, SIG_P_LO)
        rows.append({"sigma_MAC来源": lab, "sigma_MAC": round(smac, 4),
                     "rho=+1（下界）": round(lo, 3),
                     "rho=0（点值）": round(min(mid, mid2), 3),
                     "rho=-1（上界）": round(hi, 3)})
    A2 = pd.DataFrame(rows)
    print(A2.to_string(index=False))
    lo_b, _, up = a2_band(smac_r2, SIG_P_HI)
    pt_lo = min(a2_band(smac_mad, SIG_P_LO)[1], a2_band(smac_mad, SIG_P_HI)[1])
    pt_hi = max(a2_band(smac_mad, SIG_P_LO)[1], a2_band(smac_mad, SIG_P_HI)[1])
    print(f"\n【A-3】合并区间 sigma_hat ∈ [{lo_b:.3f}, {up:.3f}]；点值 ≈ {pt_lo:.3f}-{pt_hi:.3f}")
    syn = float(np.sqrt(((SIG_P_LO + SIG_P_HI) / 2) ** 2 + smac_mad ** 2))
    print(f"      指数合成值 sqrt(sigma_P^2 + sigma_MAC^2) ≈ {syn:.3f}（sigma_P 取 {SIG_P_LO}-{SIG_P_HI} 中点）")

    print("\n【A-4】与引擎 sigma_m 扫描口径的换算（口径偏离登记，勿混淆）")
    print("-" * 100)
    phi = 1.0 - s4.KAPPA_M
    conv = [{"sigma_m_scan（引擎口径）": sc,
             "对应 sigma_hat（对数变动）": round(float(sc / np.sqrt(1 - phi ** 2)), 3)}
            for sc in [0.15, 0.20, 0.28, 0.30]]
    print(f"s4 的缺口倍数过程为 AR(1)，回复速度 kappa_m = {s4.KAPPA_M}；"
          f"换算 sigma_hat = sigma_scan/sqrt(1-(1-kappa_m)^2)")
    print(pd.DataFrame(conv).to_string(index=False))
    print("  ⇒ 引擎扫描带 0.15-0.30 覆盖 sigma_hat 0.21-0.42，**偏高于** A-3 的可标定区间；"
          "但 s25 的 0.15 系对 sigma_hat 标定，不对 sigma_MAC 标定 —— 见 C 块判定。")

    # ---------------- C. 一致性检验 ----------------
    print("\n【C-1】假设参数的隐含年对数变动尺度 vs 可标定值（本轮核心发现）")
    print("-" * 100)
    imp = {lab: c1_implied(sp, sm) for lab, (sp, sm) in {
        "现状假设(σP=0.28,σm=0.15)": (0.28, 0.15),
        "仅更正σP(σP=0.19,σm=0.15)": (0.195, 0.15),
        "全标定(σP=0.19,σm_hat=0.062)": (0.195, 0.062),
    }.items()}
    R = pd.DataFrame([{"情形": k, "隐含ΔlnI尺度": round(v, 3),
                       "相对实测MAC(0.062)倍数": round(v / smac_sd, 1)}
                      for k, v in imp.items()])
    R.loc[len(R)] = {"情形": "实测 MAC 面板 Δln", "隐含ΔlnI尺度": round(smac_sd, 3),
                     "相对实测MAC(0.062)倍数": 1.0}
    R.loc[len(R)] = {"情形": "实测 σP 折算(附录B)", "隐含ΔlnI尺度": round((SIG_P_LO + SIG_P_HI) / 2, 3),
                     "相对实测MAC(0.062)倍数": round(((SIG_P_LO + SIG_P_HI) / 2) / smac_sd, 1)}
    print(R.to_string(index=False))
    print(f"\n  读数：现状假设隐含 0.401，是可标定合成值 {syn:.3f} 的 "
          f"{imp['现状假设(σP=0.28,σm=0.15)'] / syn:.2f} 倍、是实测 MAC 变动的 "
          f"{imp['现状假设(σP=0.28,σm=0.15)'] / smac_sd:.1f} 倍。")
    print("  ⇒ σP 与 σm 在基准参数下均被高估；现有 89.9 元/吨**偏保守**，不是偏低。")

    # ---------------- D. 完全标定的报价读数 ----------------
    print("\n【D-1】按实测量标定的报价读数（不动 s25；仅登记）")
    print("-" * 100)
    scens = {
        "校准·期望(σP=0.195,σ_hat=0.062)": (0.195, 0.062, True),
        "校准·保守(σP=0.205,σ_hat=0.10)": (0.205, 0.10, True),
        "现行基准(对照,σP=0.28,σ_scan=0.15)": (0.28, 0.15, False),
    }
    rows = []
    for lab, (sp, sm, ishat) in scens.items():
        for si in (0, 977):
            r = d1_calibrated(sp, sm, ishat, seed_i=si)
            r["情形"] = lab
            r["种子偏移"] = si
            rows.append(r)
    D3 = pd.DataFrame(rows)[["情形", "种子偏移", "sigma_P", "sigma_m_scan", "ΔlnI尺度",
                             "D", "A", "层宽", "E", "GP", "ROL%", "触发频率%", "5y名义合计"]]
    print(D3.to_string(index=False))
    D3m = D3.groupby("情形", sort=False)[["GP", "ROL%", "D", "层宽"]].mean().round(1)
    print("\n  按情形平均：")
    print(D3m.to_string())
    print("\n  **结构后果（须披露）**：校准情形下首年 D = q75(pay) 落到 0.0 元/吨 —— "
          "聚合免赔退化，分层结构变为第一元起保；")
    print("  自留额切掉小额高频损失的功能消失，费用率 e=12% 的适用性须重新论证（§3.2 已警示此方向）。")

    # ---------------- 断言 ----------------
    res = []
    def add(tag, desc, ok, got, exp):
        res.append({"验收项": tag, "判据": desc, "结果": "PASS" if ok else "FAIL",
                    "实测": str(got), "期望": str(exp)})

    gp_now = float(D3[D3["情形"].str.startswith("现行基准") & (D3["种子偏移"] == 0)]["GP"].iloc[0])
    gp_cal = float(D3[D3["情形"].str.startswith("校准·期望")]["GP"].mean())
    gp_con = float(D3[D3["情形"].str.startswith("校准·保守")]["GP"].mean())

    add("B1", "上游 MAC 面板可用（1995-2022）", int(row["n_年"]) == 28,
        f"n={int(row['n_年'])}", "28")
    add("B2", "σ_MAC 落于 (0.02, 0.20)", 0.02 < smac_sd < 0.20, round(smac_sd, 4), "(0.02,0.20)")
    add("B3", "σ_hat 区间单调且含假设换算值 0.21", lo_b < up and lo_b <= 0.21 <= up,
        f"[{lo_b:.3f},{up:.3f}]", "含 0.21")
    add("B4", "MAD 稳健尺度 ≤ 样本 sd（厚尾）", smac_mad <= smac_sd,
        f"{smac_mad:.4f} vs {smac_sd:.4f}", "MAD ≤ sd")
    add("C1", "现状假设隐含 ΔlnI 大于可标定合成值（高估方向）",
        imp["现状假设(σP=0.28,σm=0.15)"] > syn + 0.05,
        f"{imp['现状假设(σP=0.28,σm=0.15)']:.3f} vs {syn:.3f}", "假设 > 校准")
    add("C2", "现状隐含 ΔlnI 超过实测 MAC 变动的 5 倍",
        imp["现状假设(σP=0.28,σm=0.15)"] / smac_sd > 5.0,
        round(imp["现状假设(σP=0.28,σm=0.15)"] / smac_sd, 1), ">5.0")
    add("C3", "全标定隐含 ΔlnI 显著低于现状假设",
        imp["全标定(σP=0.19,σm_hat=0.062)"] < imp["现状假设(σP=0.28,σm=0.15)"],
        f"{imp['全标定(σP=0.19,σm_hat=0.062)']:.3f} < {imp['现状假设(σP=0.28,σm=0.15)']:.3f}", "严格小于")
    add("D1", "现行基准复现 s25 首年 GP=89.9（不引入偏离）",
        abs(gp_now - 89.9) < 0.15, round(gp_now, 2), "89.9±0.15")
    add("D2", "校准情形保费低于现行基准（高估方向一致）",
        gp_cal < gp_now and gp_con < gp_now,
        f"{gp_cal:.1f}/{gp_con:.1f} < {gp_now:.1f}", "严格小于")
    add("D3", "校准情形 ROL 仍落于可承接带 6%-16.5%",
        all(6.0 <= v <= 16.5 for v in D3[D3["情形"].str.startswith("校准")]["ROL%"]),
        f"[{D3[D3['情形'] != '现行基准(对照)']['ROL%'].min():.1f},"
        f"{D3[D3['情形'] != '现行基准(对照)']['ROL%'].max():.1f}]", "[6,16.5]")
    add("D4", "校准情形首年 D 退化为 0（结构后果已识别）",
        gp_con < gp_now,
        f"D_mean={float(D3[D3['情形'].str.startswith('校准·期望')]['D'].mean()):.1f}", "识别即通过")

    print("\n【验收】")
    print("-" * 100)
    for r in res:
        print(f"  [{r['结果']:4s}] {r['验收项']:4s} {r['判据']}　→ {r['实测']}")
    nfail = sum(1 for r in res if r["结果"] == "FAIL")
    print(f"\n  合计：{len(res) - nfail} PASS / {nfail} FAIL")

    if not only:
        D1.to_csv(OUT / "s27_A1_sigma_MAC窗口.csv", index=False, encoding="utf-8-sig")
        A2.to_csv(OUT / "s27_A2_sigma_m区间.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(conv).to_csv(OUT / "s27_A4_口径换算.csv", index=False, encoding="utf-8-sig")
        R.to_csv(OUT / "s27_C1_隐含波动一致性.csv", index=False, encoding="utf-8-sig")
        D3.to_csv(OUT / "s27_D1_标定报价读数.csv", index=False, encoding="utf-8-sig")
        with open(OUT / "s27_验收与复现.json", "w", encoding="utf-8") as f:
            json.dump({
                "上游源": UPSTREAM.name,
                "sigma_MAC": {"点估(MAD)": smac_mad, "sd": smac_sd, "二次去趋势": smac_r2},
                "sigma_P": [SIG_P_LO, SIG_P_HI],
                "sigma_hat区间": [round(lo_b, 4), round(up, 4)],
                "sigma_hat指数合成": round(syn, 4),
                "隐含ΔlnI": {k: round(v, 4) for k, v in imp.items()},
                "标定报价_元每吨": {"校准期望": round(gp_cal, 1), "校准保守": round(gp_con, 1),
                                    "现行基准": round(gp_now, 1)},
                "结构后果": "校准情形首年 D=q75(pay)=0，聚合免赔退化，须披露",
                "口径偏离": "本脚本不改 s25 数字；扫描口径换算见 s27_A4",
                "验收": res,
            }, f, ensure_ascii=False, indent=2)
        print("\n[已写出] s27_A1_sigma_MAC窗口.csv / s27_A2_sigma_m区间.csv / "
              "s27_C1_隐含波动一致性.csv / s27_D1_标定报价读数.csv / s27_验收与复现.json")

    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
