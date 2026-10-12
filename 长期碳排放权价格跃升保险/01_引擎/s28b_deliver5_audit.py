# -*- coding: utf-8 -*-
"""s28b —— deliver5 的三项实质审计

一、资本路径接缝（🔴 核心）
    层丙（"现实情形"）的 2023–2025 段与 2026–2030 段**不是同一条资本路径**：
      · 2023–2025：g_K,t = r·(1+g_I,t) − δ，g_I 取固定资产投资实测（+3.0% / +3.2% / −3.8%）
        ⇒ K_2025(mid) = 1.21180 × K_2022
      · 2026 起点：取自 T-09 的 2025 归一化 K，而 T-09 用的是**层乙（压力测试）口径**
        g_K = 12.81%/年（`run_T14_scenario_extrapolation.py:151`
        `kbase25 = k22*(1+T09_TIERS[t])**3`）⇒ K_2025(mid) = 1.43563 × K_2022
      ⇒ **接缝比 1.18471（+18.47%）**
    本脚本在**其余输入完全不变**的前提下，把 K 换成"层丙自身路径连续推进"的反事实，
    逐年重算 I_t / band / U_t / 半宽 / R_widen / 符号，量化接缝的完整影响。

二、外推出支撑程度：2026–2030 归一化 K 相对 1995–2022 样本上界。

三、与保险引擎自有 MAC 路径对照（读 02_产物/s2b、s3）。

只读 deliver5 与 02_产物；输出 02_产物/s28b_*.csv / .json。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
ROOT = PROJ.parent
PAPER = ROOT / "背景材料与论文" / "中国省际碳排放影子价格测算与碳修正绿色GDP核算——基于二次型方向距离函数的实证研究（1995–2022）"
D5 = PAPER / "deliver5"
OUT = PROJ / "02_产物"
MIRROR_D3 = HERE / "_s28_deliver5_mirror" / "deliver3" / "03_实验记录"
sys.path.insert(0, str(MIRROR_D3))

import run_T09_extrapolation as t09  # noqa: E402

R_PRIMARY = 0.12890728324110395
DELTA = {"low": 0.074, "mid": 0.0638, "high": 0.057}
G_I_NAT = {2023: 0.030, 2024: 0.032, 2025: -0.038}
G_I_P6_MID = 0.045
T09_TIERS = {"low": 0.0442, "mid": 0.1281, "high": 0.2067}
G_S_STAR = 0.83 ** 0.2 - 1.0
G_L_P7_MID = -0.005
G_Y, G_E = 0.045, 0.9717526154721383 * 1.045 - 1.0
CVAR = ["P1", "P3b"]
YEARS = (2026, 2027, 2028, 2029, 2030)
C_LAYER_TIERS = {t: R_PRIMARY * (1.0 + G_I_P6_MID) - DELTA[t] for t in DELTA}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    panel = t09.load_raw()
    df = t09.build_norm(panel, end_year=2022)
    b = df[df["年份"] == 2022]
    provs = b["地区"].tolist()
    y22, l22, e22 = (b[c].to_numpy(float) for c in ("Y_norm", "L_norm", "E_norm"))
    c22, k22 = b["C_norm"].to_numpy(float), b["K_norm"].to_numpy(float)
    rr = (b["ybar"] / b["bbar"]).to_numpy(float)

    cgrid = t09.grid_of(t09.WSTAR)
    FAM = {c: t09.solve_params(df, c)["p_full"] for c in cgrid
           if t09.solve_params(df, c) is not None}
    cgrid = list(FAM)
    c0 = min(cgrid, key=abs)

    def mean_mac(y, cc, k, l, e):
        return float(np.nanmean(t09.eval_obs(FAM[c0], y, cc, k, l, e)["MAC_scaled"] * rr * 1e4))

    # 2022 校验
    f22 = {c: float(np.nanmean(t09.eval_obs(FAM[c], y22, c22, k22, l22, e22)["MAC_scaled"] * rr * 1e4))
           for c in cgrid}
    I22 = (min(f22.values()), max(f22.values()))
    ref_hw = (I22[1] - I22[0]) / 2.0

    # 起点因子
    agg = panel.groupby("年份")[["L_raw"]].sum().sort_index()
    gL09 = float(agg["L_raw"].pct_change().loc[2020:2022].mean())
    ym25 = np.array([t09.Y_INDEX_2023[p] / 100.0 * t09.Y_INDEX_2024[p] / 100.0
                     * (1 + t09.Y_RATE_2025[p] / 100.0) for p in provs])
    em25 = float(np.prod([1.0 + t09.G_E_NAT[y] for y in (2023, 2024, 2025)]))
    lm25 = (1.0 + gL09) ** 3
    _p, _pm = t09.build_c_paths()
    cfac25 = {cv: float(_p[cv][2025]) for cv in CVAR}
    gC = {"P1": G_E, "P3b": (1.0 + G_Y) * (1.0 + G_S_STAR) - 1.0}

    # K 起点两版（逐档，T14 §局限 3「按档秩匹配」）
    k_own = {tk: k22 * np.prod([1.0 + (R_PRIMARY * (1.0 + G_I_NAT[t]) - DELTA[tk])
                                for t in (2023, 2024, 2025)]) for tk in DELTA}
    k_t09 = {tk: k22 * (1.0 + T09_TIERS[tk]) ** 3 for tk in DELTA}

    def layer_c(k_start, label):
        """完整复刻 T-14 层丙构造（K 三档 × C 两变体），返回逐年结果。"""
        out = {}
        for t in YEARS:
            n = t - 2025
            y = y22 * ym25 * (1.0 + G_Y) ** n
            l = l22 * lm25 * (1.0 + G_L_P7_MID) ** n
            e = e22 * em25 * (1.0 + G_E) ** n
            cc = {cv: c22 * cfac25[cv] * (1.0 + gC[cv]) ** n for cv in CVAR}
            kk = {tk: k_start[tk] * (1.0 + C_LAYER_TIERS[tk]) ** n for tk in ("low", "mid", "high")}
            # 轴1：中央档（K mid） × 中央 C（P3b），枚举 c
            f = {c: float(np.nanmean(t09.eval_obs(FAM[c], y, cc["P3b"], kk["mid"], l, e)["MAC_scaled"] * rr * 1e4))
                 for c in cgrid}
            I = (min(f.values()), max(f.values()))
            # 轴2：c0 处 K 三档 × C 两变体
            band = [mean_mac(y, cc[cv], kk[tk], l, e) for tk in ("low", "mid", "high") for cv in CVAR]
            band = (min(band), max(band))
            U = (min(I[0], band[0]), max(I[1], band[1]))
            hw = (U[1] - U[0]) / 2.0
            out[t] = dict(I_t=I, band_c0=band, U_t_c0=U, halfwidth=hw,
                          R_widen=hw / ref_hw,
                          central_c0=f[c0],
                          sign=("全负" if U[1] < 0 else "全正" if U[0] > 0 else "跨零"),
                          k_mid_rel2022=float(kk["mid"][0] / k22[0]))
        return out

    act = layer_c(k_t09, "T14 实际")
    fix = layer_c(k_own, "层丙连续口径")

    # 归档比对（校验管线可信）
    t14j = json.loads((MIRROR_D3 / "T14_output.json").read_text(encoding="utf-8"))
    Lc = t14j["results"]["S_gy4.5"]["layers"]["layer_C_bing"]
    chk = []
    for t in YEARS:
        a_lo, a_hi = Lc["I_t"][str(t)]
        chk.append(dict(year=t, I_lo_arch=a_lo, I_lo_mine=act[t]["I_t"][0],
                        I_hi_arch=a_hi, I_hi_mine=act[t]["I_t"][1],
                        R_arch=Lc["R_widen"]["c0"][str(t)], R_mine=act[t]["R_widen"]))

    # 出支撑（K_norm / 1995—2022 全样本上界）
    # 【2026-10-04 更正】原写法分母用 kin.iloc[0]（单个省的样本峰值），口径错误，
    # 导致比值被系统性放大（曾报 ×1.576@2026）。正确分母为全样本 K_norm 上界。
    k_ub = float(df[df["年份"] <= 2022]["K_norm"].max())
    k22_nat = float(df[df["年份"] == 2022]["K_norm"].mean())
    # 层丙 mid 全国 K 路径 = 2022 全国均值 × Π(1+g_K)，g_K = r(1+g_I)−δ
    gk_mid = R_PRIMARY * (1.0 + 0.045) - DELTA["mid"]
    oos = {t: float(k22_nat * (1.0 + gk_mid) ** (t - 2022) / k_ub) for t in YEARS}
    oos_note = {
        "分母": "1995—2022 全样本 K_norm 上界",
        "k_ub": k_ub, "k22_全国均值": k22_nat, "g_K_层丙mid": float(gk_mid),
        "2022比值": float(k22_nat / k_ub),
        "更正记录": "原 kin.iloc[0] 分母口径错误，×1.576@2026 作废",
    }

    # 引擎对照
    eng = {}
    p2b = OUT / "s2b_各标度档MAC路径.csv"
    if p2b.exists():
        with p2b.open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row.get("年份") in {str(y) for y in range(2022, 2031)}:
                    eng[int(row["年份"])] = float(row["论文口径"])

    res = dict(ref_halfwidth=ref_hw, I_2022=list(I22),
               seam_ratio=float((1.0 + T09_TIERS["mid"]) ** 3
                                / np.prod([1.0 + (R_PRIMARY * (1.0 + G_I_NAT[t]) - DELTA["mid"])
                                           for t in (2023, 2024, 2025)])),
               k_2025_mid=dict(own=float(k_own["mid"][0] / k22[0]), t09=float(k_t09["mid"][0] / k22[0])),
               archive_check=chk,
               actual={str(t): {k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in act[t].items()} for t in YEARS},
               counterfactual={str(t): {k: (list(v) if isinstance(v, tuple) else v)
                                        for k, v in fix[t].items()} for t in YEARS},
               out_of_support_ratio=oos,
               out_of_support_note=oos_note,
               engine_paper_caliber=eng)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "s28b_deliver5_审计.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    rows = [["年", "口径", "I_t下", "I_t上", "U_t下", "U_t上", "半宽", "R_widen",
             "c0中央值", "符号", "K/2022"]]
    for t in YEARS:
        for nm, d in (("T14实际", act[t]), ("层丙连续", fix[t])):
            rows.append([t, nm, f"{d['I_t'][0]:.2f}", f"{d['I_t'][1]:.2f}",
                         f"{d['U_t_c0'][0]:.2f}", f"{d['U_t_c0'][1]:.2f}",
                         f"{d['halfwidth']:.2f}", f"{d['R_widen']:.4f}",
                         f"{d['central_c0']:.2f}", d["sign"], f"{d['k_mid_rel2022']:.4f}"])
    with (OUT / "s28b_接缝前后对照.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows(rows)

    eng_rows = [["年", "deliver5层丙I_t下", "deliver5层丙I_t上", "引擎论文口径MAC95", "引擎落于区间内"]]
    for t in YEARS:
        lo, hi = act[t]["I_t"]
        e = eng.get(t)
        eng_rows.append([t, f"{lo:.2f}", f"{hi:.2f}", "" if e is None else f"{e:.2f}",
                         "" if e is None else str(lo <= e <= hi)])
    with (OUT / "s28b_与引擎MAC路径对照.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows(eng_rows)

    print("== 管线校验（我 vs 归档）==")
    for c in chk:
        print(f"  {c['year']}: I_lo {c['I_lo_mine']:.4f}/{c['I_lo_arch']:.4f}  "
              f"I_hi {c['I_hi_mine']:.4f}/{c['I_hi_arch']:.4f}  "
              f"R {c['R_mine']:.6f}/{c['R_arch']:.6f}")
    print("\n== 接缝前后（层丙，S_gy4.5）==")
    print(f"  K_2025(mid)/K_2022：层丙自身 {k_own["mid"][0]/k22[0]:.5f}；T-09 口径 {k_t09["mid"][0]/k22[0]:.5f}"
          f"；接缝比 {res['seam_ratio']:.5f}")
    for t in YEARS:
        a, f = act[t], fix[t]
        print(f"  {t}: I_t {a['I_t'][0]:8.2f}~{a['I_t'][1]:8.2f} → {f['I_t'][0]:8.2f}~{f['I_t'][1]:8.2f}"
              f" | 位移 {f['I_t'][0]-a['I_t'][0]:+7.2f}/{f['I_t'][1]-a['I_t'][1]:+7.2f}"
              f" | R_widen {a['R_widen']:.4f} → {f['R_widen']:.4f}"
              f" | 符号 {a['sign']} → {f['sign']}")
    print("\n== 出支撑（K_norm / 样本上界）==")
    for t, v in oos.items():
        print(f"  {t}: ×{v:.4f}")
    print("\n== 引擎对照 ==")
    for r in eng_rows:
        print("  ", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
