# -*- coding: utf-8 -*-
"""
s24_review2_response.py —— 第二轮审稿响应工作包（R0–R6）

背景（2026-10-03 第二轮外部审稿）：审稿意见 M1–M8。本脚本落实其中的**引擎级**部分。
不改任何既有铁律与口径（结构仍 B1 年度重定基 + 聚合免赔 D=q75 + 责任上限 A=q99.5(L)；
报价原理仍为口径乙 CoC），也不修改 s22 / s23 / s0c / s4 —— 旧脚本的缺陷在本脚本内**登记并量化**。

R0  复现锚：s23.simulate_p（默认参数）与 s4.simulate 逐位相同；论文口径档 3y 指标
    与 s22 主表逐项一致（同 seed）。
R1  二维合法域诊断（回应 M1，本轮**独立查出**两项新事实）
    R1a **扫描盒截断**：s0c/s23-C4 的 γ₁ 扫描范围取 [ROOTS[0], ROOTS[3]]，而该范围是
        **在 β=1 上解出来的**；换到 β≠1 后合法域在 γ₁ 方向可以越出该盒。实测 β∈[0.40,0.62]
        区间内存在 MAC 明显低于 131.445 的合法点 ⇒ **131.445 不是域下确界**。
    R1b **判据不足**：β≲0.08 时 LP 返回「零参数平凡解」（dev_max≈0、MAC≈0），而它满足
        dev_max ≤ 1。⇒ 现行判据下「合法域下端」趋向 0，**不存在**。
    R1c **上界侧稳健**：γ₁ < ROOTS[0] 且 β ≥ 1.2 时合法域收窄，β ≥ 2.5 无合法点，
        故 585.871 基本稳健（资本侧上端不受 R1a/R1b 影响）。
    ⇒ 结论：表「报价取合法域下端」这条规则**不可执行**（下端不存在且不可收敛）。
R2  主报价档重建算：主报价改挂**冻结标度约定档**（γ₁=0、β=1，2026 名义基准 413.32）。
    3/5/10/15 年 D / A / 层宽 / E / RM / GP / ROL / 年化保费。
R3  最低资本线（回应 M3）：最低资本 = φ·RM，**不含 E、不除 (1−e−π)**；
    并列单一保单口径 RM_独立。含**上确界端点档**（β=1.8、γ₁=−0.125，MAC 585.871）。
R4  省级锚情景（回应 M2-C「全国版 + 省级版并列」）：在指数对数变动上叠加
    **年化基差扰动 σ_B = 0.1069**（s18 实测），重算省级版报价；并列全国系统性版。
R5  σ_m 报价带（回应 M4）：σ_m ∈ [0.05, 0.30] 六档 → 主报价写成区间。
R6  敞口变量 Q_t（回应 M5）：σ_Q ∈ {0.10, 0.25, 0.40} 对保费的影响；以及
    **核定敞口被系统性高报 x% 的解析后果**（层参数与保费等比放大、ROL 不变）。
R7  尾部估计收敛诊断（**在新主报价档 3y 上重做**，回应 M6 的"图 1 更新"与
    避免沿用旧报价档的收敛数字）：自助标准误（B=1,000）与路径数倍增稳定带。

用法：python s24_review2_response.py [--check]    # 退出码 0 = 断言全过
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import L1, OUT                                    # noqa: E402
import s4_pricing as s4                                         # noqa: E402
import s23_review_response as s23                               # noqa: E402
from s0_gamma_window import ROOTS, run_at                       # noqa: E402
from s22_commercial_reprice import phi_div                      # noqa: E402

pd.set_option("display.width", 240)
pd.set_option("display.unicode.east_asian_width", True)

RHO = L1["capital_charge"]          # 6%
PI_PROFIT = 0.08
E_EXP_IDX = 0.12
N_EFF = 100
RHO_BAR = 0.30
Q_LIMIT = 0.995
TERMS = s4.TERMS
DEFL_2026 = 2.169309

TIER_PAPER = "论文口径"                 # γ₁=0, β=1 —— 本轮改为主报价所用的冻结约定档
TIER_SUP = "β1.8_上确界端点（γ1=−0.125）"
TIER_OLDQ = "β0.6_下确界"                 # s22 原主报价档（本轮降为对照档）

# R1 扫描盒（s0c / s23-C4 实际使用的范围）
BOX_BETA = (0.40, 2.00)
BOX_G1 = (float(ROOTS[0]), float(ROOTS[3]))
REF_LO_FINER = 131.445                 # s23-C4 加密网格"下确界"
REF_HI = 585.871                       # s23-C4 加密网格"上确界"

SIG_B = 0.1069                         # s18 实测：单年省际基差波动
SUP_FACTOR_H5 = float(np.sqrt(6.41 / 5.0))   # h=5 超扩散等效放大（Var比 6.41/h=5）
SENS_SIGM = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SENS_KAPPAM = [0.15, 0.30, 0.60]
SENS_RHOPM = [0.00, 0.45, 0.75]
SENS_SIGQ = [0.10, 0.25, 0.40]

R1A_BETAS = [round(b, 2) for b in np.arange(0.40, 0.621, 0.02)]
R1A_G1S = [round(g, 2) for g in np.arange(0.14, 0.361, 0.01)]
R1B_BETAS = [0.02, 0.04, 0.06, 0.08]
R1B_G1S = [-0.05, 0.0, 0.05]
R1C_BETAS = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0]
R1C_G1S = [round(g, 2) for g in np.arange(-0.80, -0.059, 0.02)]


# ============================================================ R0 复现锚
def r0(tiers: dict) -> dict:
    s_ref = s4.simulate("S1_基准", tiers[TIER_PAPER], "M2_价格平移",
                        L1["mc_paths"], s4.RNG_SEED + 4 * 977)
    s_new = s23.simulate_p("S1_基准", tiers[TIER_PAPER], "M2_价格平移",
                           L1["mc_paths"], s4.RNG_SEED + 4 * 977)
    same_I = bool(np.array_equal(s_ref["I"], s_new["I"]))
    same_Q = bool(np.array_equal(s_ref["Q"], s_new["Q"]))
    # 与 s22 主表交叉校验（论文口径档 · B1 · D75 · e12 · n100）
    L = s23.pv_rebased_p(s_new, 3, s4.KAPPA_BASE, s4.CAP_AMOUNT)
    m = s23.layer_metrics(L)
    ref = {}
    try:
        R = pd.read_csv(OUT / "s22_商业版定价_主表.csv")
        row = R[(R["结构"] == "B1_年度重定基") & (R["标度档"] == TIER_PAPER)
                & (R["期限"] == 3) & (R["免赔档"] == "D75_q75")
                & (R["费用口径"] == "e12_指数") & (R["n_eff"] == 100)].iloc[0]
        ref = {"D": float(row["D_元每吨"]), "A": float(row["A_q995"]),
               "层宽": float(row["层宽"]), "E": float(row["E_layer"]),
               "RM": float(row["RM_组合"]), "GP": float(row["GP_CoC_元每吨"])}
    except Exception as exc:                                   # noqa: BLE001
        ref = {"error": str(exc)}
    dev = {}
    for k, key in (("D", "D"), ("A", "A"), ("层宽", "层宽"), ("E", "E_layer"),
                   ("RM", "RM_组合"), ("GP", "GP_CoC")):
        if k in ref:
            dev[k] = abs(m[key] - ref[k])
    return {"same_I": same_I, "same_Q": same_Q,
            "max_dI": float(np.abs(s_ref["I"] - s_new["I"]).max()),
            "复算_3y": {k: round(v, 2) for k, v in m.items()},
            "s22对照_3y": ref, "绝对偏差": {k: round(v, 4) for k, v in dev.items()},
            "最大偏差": round(max(dev.values()), 4) if dev else None}


# ============================================================ R1 二维域诊断
def _scan(betas, g1s) -> pd.DataFrame:
    rows = []
    for beta in betas:
        for g1 in g1s:
            try:
                r = run_at(float(g1), beta=float(beta))
            except Exception:                                  # noqa: BLE001
                continue
            if not r.get("ok"):
                continue
            rows.append({"γ1": float(g1), "β": float(beta),
                         "MAC2022": float(r["MAC_2022均值"]),
                         "dev_max": float(r["dev_max"]),
                         "MAC2026名义": float(r["MAC_2022均值"]) * DEFL_2026,
                         "合法": bool(r["dev_max"] <= 1.0)})
    return pd.DataFrame(rows)


def r1() -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    a = _scan(R1A_BETAS, R1A_G1S)
    b = _scan(R1B_BETAS, R1B_G1S)
    c = _scan(R1C_BETAS, R1C_G1S)
    a["诊断块"], b["诊断块"], c["诊断块"] = "R1a_扫描盒截断", "R1b_退化解", "R1c_上界侧"
    for d in (a, b, c):
        d["扫描盒内"] = (d["β"].between(*BOX_BETA) & d["γ1"].between(*BOX_G1))
    allpts = pd.concat([a, b, c], ignore_index=True)

    la = a[a["合法"]]
    lb = b[b["合法"]]
    lc = c[c["合法"]]
    lo_a = la.loc[la["MAC2022"].idxmin()] if len(la) else None
    hi_c = lc.loc[lc["MAC2022"].idxmax()] if len(lc) else None
    deg = lb[lb["MAC2022"] < 1.0] if len(lb) else lb

    meta = {
        "扫描盒（s0c / s23-C4 实际使用）": {"β": list(BOX_BETA),
                                        "γ1": [round(v, 6) for v in BOX_G1]},
        "R1a_扫描盒截断": {
            "扫描范围": {"β": [R1A_BETAS[0], R1A_BETAS[-1]], "γ1": [R1A_G1S[0], R1A_G1S[-1]]},
            "合法点数": int(len(la)),
            "最低合法点": None if lo_a is None else {
                "γ1": float(lo_a["γ1"]), "β": float(lo_a["β"]),
                "MAC2022": round(float(lo_a["MAC2022"]), 3),
                "dev_max": round(float(lo_a["dev_max"]), 4),
                "在扫描盒内": bool(lo_a["扫描盒内"])},
            "对比_s23加密下确界": REF_LO_FINER,
            "更低的幅度%": None if lo_a is None else round(
                (float(lo_a["MAC2022"]) - REF_LO_FINER) / REF_LO_FINER * 100, 3),
            "结论": "β 在扫描盒内、但 γ1 超出扫描盒 ⇒ 131.445 是扫描盒边界值，不是域下确界",
        },
        "R1b_退化解": {
            "扫描范围": {"β": R1B_BETAS, "γ1": R1B_G1S},
            "合法点数": int(len(lb)),
            "退化点数（MAC<1）": int(len(deg)),
            "退化实例": [
                {"γ1": float(r["γ1"]), "β": float(r["β"]),
                 "MAC2022": round(float(r["MAC2022"]), 6),
                 "dev_max": round(float(r["dev_max"]), 8)}
                for _, r in lb.sort_values("MAC2022").head(4).iterrows()],
            "结论": "dev_max ≤ 1 不足以排除 ω 方向的零参数平凡解 ⇒ 合法域在 MAC 上无正下界",
        },
        "R1c_上界侧": {
            "扫描范围": {"β": R1C_BETAS, "γ1": [R1C_G1S[0], R1C_G1S[-1]]},
            "合法点数": int(len(lc)),
            "最高合法点": None if hi_c is None else {
                "γ1": float(hi_c["γ1"]), "β": float(hi_c["β"]),
                "MAC2022": round(float(hi_c["MAC2022"]), 3)},
            "无合法点的β档": sorted({float(b) for b in R1C_BETAS
                                 if not len(lc[abs(lc["β"] - b) < 1e-9])}),
            "对比_s23加密上确界": REF_HI,
            "结论": "γ1 < ROOTS[0] 且 β ≥ 1.2 时合法域收窄 ⇒ 上确界稳健，资本侧上端不受影响",
        },
        "耗时s": round(time.time() - t0, 1),
    }
    return allpts, meta


# ============================================================ 档位构造
def _tier(g1: float, beta: float, tag: str) -> dict:
    return s23._tier_raw(tag, g1, beta)


def _quote_row(sim: dict, n: int, kappa: float = s4.KAPPA_BASE,
               cap: float = s4.CAP_AMOUNT, struct: str = "B1") -> dict:
    L = (s23.pv_rebased_p(sim, n, kappa, cap) if struct == "B1"
         else s23.pv_level_p(sim, n, kappa, cap))
    m = s23.layer_metrics(L)
    return m


# ============================================================ R2 主报价档重建算
def r2(sims: dict, tiers: dict) -> pd.DataFrame:
    rows = []
    for tag in (TIER_PAPER, TIER_OLDQ):
        sim = sims[tag]
        base = float(tiers[tag]["MAC2022_1995价"]) * DEFL_2026
        for n in TERMS:
            m = _quote_row(sim, n)
            rows.append({
                "标度档": tag, "γ1": round(float(tiers[tag]["γ1"]), 6),
                "β": float(tiers[tag]["β"]),
                "2026基准_元每吨": round(base, 1),
                "期限": n,
                "自留额D": round(m["D"], 1),
                "责任上限A": round(m["A"], 1),
                "层宽A−D": round(m["层宽"], 1),
                "触发事件频率%": round(m["触发概率_full%"], 2),
                "层赔付频率%": round(m["层赔付频率%"], 2),
                "层期望损失E": round(m["E_layer"], 1),
                "RM_组合": round(m["RM_组合"], 1),
                "毛保费GP_乙": round(m["GP_CoC"], 0),
                "ROL_乙%": round(m["ROL_CoC%"], 1),
                "年化保费": round(m["GP_CoC"] / n, 1),
                "期望赔付率%": round(m["E_layer"] / max(m["GP_CoC"], 1e-9) * 100, 1),
                "保费比基准": round(m["GP_CoC"] / (base * n), 4),
            })
    return pd.DataFrame(rows)


# ============================================================ R3 最低资本线
def r3(sims: dict, tiers: dict) -> pd.DataFrame:
    """最低资本 = 风险附加 RM（组合调整）；不含 E、不除 (1−e−π)。"""
    rows = []
    for tag in (TIER_PAPER, TIER_OLDQ, TIER_SUP):
        sim = sims[tag]
        base = float(tiers[tag]["MAC2022_1995价"]) * DEFL_2026
        phi = phi_div(N_EFF, RHO_BAR)
        for n in TERMS:
            m = _quote_row(sim, n)
            # 单一保单口径 RM：把组合调整除回去
            rm_ind = m["RM_组合"] / phi
            rows.append({
                "标度档": tag, "期限": n,
                "2026基准_元每吨": round(base, 1),
                "责任上限A": round(m["A"], 1),
                "层宽A−D": round(m["层宽"], 1),
                "E_不计入资本": round(m["E_layer"], 1),
                "RM_单保单": round(rm_ind, 1),
                "φ": round(phi, 4),
                "RM_组合": round(m["RM_组合"], 1),
                "最低资本_组合口径": round(m["RM_组合"], 1),
                "最低资本_单保单口径": round(rm_ind, 1),
                "资本比层宽%": round(m["RM_组合"] / max(m["层宽"], 1e-9) * 100, 1),
                "参考_毛保费GP乙": round(m["GP_CoC"], 0),
                "参考_资本比保费": round(m["RM_组合"] / max(m["GP_CoC"], 1e-9), 2),
                # 被否决的叠层**毛保费**口径（仅供量级演示，不作为资本要求）
                "叠层口径GP甲": round((m["E_layer"] + m["RM_组合"] * (1 + RHO))
                                  / (1 - E_EXP_IDX - PI_PROFIT), 0),
                "叠层口径ROL%": round((m["E_layer"] + m["RM_组合"] * (1 + RHO))
                                   / (1 - E_EXP_IDX - PI_PROFIT)
                                   / max(m["层宽"], 1e-9) * 100, 1),
            })
    return pd.DataFrame(rows)


# ============================================================ R4 省级锚情景
def _basis_paths(I: np.ndarray, sig_xi: float, seed: int) -> np.ndarray:
    """基差随机游走（年增量 iid，标准差 sig_xi）→ 省级指数 I^s = I·exp(Ση)。"""
    rng = np.random.default_rng(seed)
    nyr, n = I.shape
    eta = rng.standard_normal((nyr, n)) * sig_xi
    lnB = np.cumsum(eta, axis=0)
    lnB -= lnB[0][None, :]          # 起点归零：t₀ 处省级=全国
    return I * np.exp(lnB)


def r4(sims: dict, tiers: dict) -> pd.DataFrame:
    rows = []
    for tag in (TIER_PAPER, TIER_OLDQ):
        base_sim = sims[tag]
        base = float(tiers[tag]["MAC2022_1995价"]) * DEFL_2026
        for scen, dI in (("N_全国系统性版", 0.0),
                         ("P1_省级锚·随机游走", SIG_B),
                         ("P2_省级锚·超扩散等效", SIG_B * SUP_FACTOR_H5)):
            if dI == 0.0:
                sim = {"I": base_sim["I"], "Q": base_sim["Q"]}
            else:
                sim = {"I": _basis_paths(base_sim["I"], dI,
                                         s4.RNG_SEED + 7919),
                       "Q": base_sim["Q"]}
            for n in TERMS:
                m = _quote_row(sim, n)
                rows.append({
                    "情景": scen, "标度档": tag, "期限": n,
                    "基差年扰动σ_ξ": round(dI, 4),
                    "2026基准_元每吨": round(base, 1),
                    "自留额D": round(m["D"], 1),
                    "责任上限A": round(m["A"], 1),
                    "层宽A−D": round(m["层宽"], 1),
                    "触发事件频率%": round(m["触发概率_full%"], 2),
                    "层赔付频率%": round(m["层赔付频率%"], 2),
                    "E_layer": round(m["E_layer"], 1),
                    "RM_组合": round(m["RM_组合"], 1),
                    "最低资本": round(m["RM_组合"], 1),
                    "毛保费GP_乙": round(m["GP_CoC"], 0),
                    "ROL_乙%": round(m["ROL_CoC%"], 1),
                    "年化保费": round(m["GP_CoC"] / n, 1),
                })
    D = pd.DataFrame(rows)
    # 便捷列：省级版相对全国版的倍数
    base = D[D["情景"] == "N_全国系统性版"].set_index(["标度档", "期限"])
    D["GP相对全国版"] = [
        round(float(r["毛保费GP_乙"])
              / max(float(base.loc[(r["标度档"], r["期限"]), "毛保费GP_乙"]), 1e-9), 3)
        for _, r in D.iterrows()]
    return D


# ============================================================ R5 σ_m 报价带
def r5(tiers: dict, seeds: dict) -> pd.DataFrame:
    """逐档使用 s22 的档位种子（seeds[tag]）。若误用单一种子，非契约档的
    基准值将不等于 s22 主表——该不一致已由断言 R5b 拦截。"""
    rows = []
    for tag in (TIER_PAPER, TIER_OLDQ):
        for n in (3, 5):
            for sm in SENS_SIGM:
                sim = s23.simulate_p("S1_基准", tiers[tag], "M2_价格平移",
                                     L1["mc_paths"], seeds[tag],
                                     sig_m=sm)
                m = _quote_row(sim, n)
                rows.append({
                    "标度档": tag, "期限": n, "σ_m": sm,
                    "层宽": round(m["层宽"], 1),
                    "E_layer": round(m["E_layer"], 1),
                    "RM_组合": round(m["RM_组合"], 1),
                    "毛保费GP_乙": round(m["GP_CoC"], 0),
                    "ROL_乙%": round(m["ROL_CoC%"], 1),
                })
    D = pd.DataFrame(rows)
    D["GP相对σ_m0.15"] = [
        round(float(r["毛保费GP_乙"])
              / float(D[(D["标度档"] == r["标度档"]) & (D["期限"] == r["期限"])
                        & (D["σ_m"] == 0.15)]["毛保费GP_乙"].iloc[0]), 3)
        for _, r in D.iterrows()]
    return D


# ============================================================ R6 敞口变量
def r6(tiers: dict, seeds: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    tag = TIER_PAPER
    for n in (3, 5):
        for sq in SENS_SIGQ:
            old = s4.SIG_Q
            try:
                s4.SIG_Q = sq
                sim = s23.simulate_p("S1_基准", tiers[tag], "M2_价格平移",
                                     L1["mc_paths"], seeds[tag])
            finally:
                s4.SIG_Q = old
            m = _quote_row(sim, n)
            rows.append({"标度档": tag, "期限": n, "σ_Q": sq,
                         "层宽": round(m["层宽"], 1),
                         "E_layer": round(m["E_layer"], 1),
                         "RM_组合": round(m["RM_组合"], 1),
                         "毛保费GP_乙": round(m["GP_CoC"], 0),
                         "ROL_乙%": round(m["ROL_CoC%"], 1)})
    D = pd.DataFrame(rows)
    # 解析后果：核定敞口被系统性高报 x% ⇒ 层参数与保费等比放大、ROL 不变
    over = [{"高报幅度x%": x,
             "D倍数": round(1 + x / 100, 4), "A倍数": round(1 + x / 100, 4),
             "层宽倍数": round(1 + x / 100, 4),
             "GP倍数": round(1 + x / 100, 4), "ROL": "不变"}
            for x in (0, 5, 10, 20)]
    return D, {"解析结论": "pay_t ∝ Q_t 且 D、A 均为 L 的分位 ⇒ 敞口等比高报使 D/A/层宽/GP "
                        "等比放大，ROL 不变；故道德风险的后果是**限额虚增**而非定价失灵。",
               "数值对照": over}


# ============================================================ R7 收敛诊断（新主报价档）
def r7(tiers: dict, seeds: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """自助标准误（同一样本内推断）+ 路径数倍增稳定带（非嵌套，仅作稳定带）。"""
    sim = s4.simulate("S1_基准", tiers[TIER_PAPER], "M2_价格平移",
                      L1["mc_paths"], seeds[TIER_PAPER])
    L = s23.pv_rebased_p(sim, 3, s4.KAPPA_BASE, s4.CAP_AMOUNT)
    rng = np.random.default_rng(20261003)
    n = L.size
    rows = np.empty((1000, 4))
    for b in range(1000):
        idx = rng.integers(0, n, n)
        Lb = L[idx]
        D = float(np.quantile(Lb, 0.75))
        A = float(np.quantile(Lb, Q_LIMIT))
        pay = np.clip(Lb - D, 0.0, A - D)
        E = float(pay.mean())
        rm = (s23._tvar(pay) - E) * phi_div(N_EFF, RHO_BAR)
        rows[b] = [A, s23._tvar(pay), A - D,
                   (E + RHO * rm) / (1.0 - E_EXP_IDX - PI_PROFIT)]
    base = {"q99.5": float(np.quantile(L, Q_LIMIT)),
            "A−D": float(np.quantile(L, Q_LIMIT) - np.quantile(L, 0.75))}
    pay0 = np.clip(L - np.quantile(L, 0.75), 0.0,
                   np.quantile(L, Q_LIMIT) - np.quantile(L, 0.75))
    E0 = float(pay0.mean())
    rm0 = (s23._tvar(pay0) - E0) * phi_div(N_EFF, RHO_BAR)
    base["TVaR99"] = s23._tvar(pay0)
    base["GP"] = (E0 + RHO * rm0) / (1.0 - E_EXP_IDX - PI_PROFIT)
    names = ["q99.5", "TVaR99", "A−D", "GP"]
    boot = pd.DataFrame([{
        "统计量": nm,
        "点估计": round(base[nm], 2),
        "自助SE": round(float(rows[:, i].std(ddof=1)), 3),
        "相对SE%": round(float(rows[:, i].std(ddof=1)) / base[nm] * 100, 2),
    } for i, nm in enumerate(names)])

    srows = []
    for npth in (20_000, 40_000, 80_000):
        sm = s4.simulate("S1_基准", tiers[TIER_PAPER], "M2_价格平移",
                         npth, s4.RNG_SEED + 4 * 977)
        M = _quote_row(sm, 3)
        srows.append({"路径数": npth, "q99.5": round(M["A"], 1),
                      "层宽": round(M["层宽"], 1), "TVaR99": round(
                          M["RM_组合"] / phi_div(N_EFF, RHO_BAR) + M["E_layer"], 1),
                      "毛保费GP乙": round(M["GP_CoC"], 1),
                      "ROL%": round(M["ROL_CoC%"], 2)})
    band = pd.DataFrame(srows)
    meta = {"说明": "路径数倍增检验**不是嵌套子样本**（标准正态填充顺序依赖路径数），"
                  "仅作稳定带；同一样本内推断以自助标准误为准。",
            "GP变动%": round((band["毛保费GP乙"].iloc[-1]
                          / band["毛保费GP乙"].iloc[0] - 1) * 100, 2),
            "层宽变动%": round((band["层宽"].iloc[-1]
                           / band["层宽"].iloc[0] - 1) * 100, 2)}
    return boot, band, meta


# ============================================================ R8 CAP 倍数（确定性）
def r8(sims: dict, tiers: dict) -> pd.DataFrame:
    """CAP 换倍数只改赔付函数、不改路径 ⇒ 一次模拟、多次重算（无新增抽样噪声）。"""
    rows = []
    for tag in (TIER_PAPER, TIER_OLDQ):
        sim = sims[tag]
        for n in (3, 5):
            for mult in (1.0, 2.0, 4.0):
                m = _quote_row(sim, n, kappa=s4.KAPPA_BASE,
                               cap=mult * s4.MAC_REF)
                rows.append({"标度档": tag, "期限": n, "CAP倍数": mult,
                             "CAP_元每吨": round(mult * s4.MAC_REF, 2),
                             "层宽": round(m["层宽"], 1),
                             "E_layer": round(m["E_layer"], 1),
                             "毛保费GP_乙": round(m["GP_CoC"], 0),
                             "ROL_乙%": round(m["ROL_CoC%"], 1)})
    return pd.DataFrame(rows)


# ============================================================ R9 参数标定杠杆（M4）
def r9(tiers: dict, seeds: dict) -> pd.DataFrame:
    """κ_m 与 ρ(P,m) 单变量敏感性。**逐档使用 s22 档位种子**——s23-C3 曾对
    所有档位使用同一个档位种子，使论文口径档的基准值（237）偏离 s22 主表
    （227）；本函数取代该结果，并以断言 R9b 拦截同类错误。"""
    rows = []
    for tag in (TIER_PAPER, TIER_OLDQ):
        for scope, vals in (("κ_m", SENS_KAPPAM), ("ρ(P,m)", SENS_RHOPM)):
            for v in vals:
                kw = {"kappa_m": v} if scope == "κ_m" else {"rho_pm": v}
                sim = s23.simulate_p("S1_基准", tiers[tag], "M2_价格平移",
                                     L1["mc_paths"], seeds[tag], **kw)
                m = _quote_row(sim, 5)
                rows.append({
                    "标度档": tag, "期限": 5, "参数": scope, "取值": v,
                    "层宽": round(m["层宽"], 1),
                    "E_layer": round(m["E_layer"], 1),
                    "毛保费GP_乙": round(m["GP_CoC"], 0),
                    "ROL_乙%": round(m["ROL_CoC%"], 1)})
    D = pd.DataFrame(rows)
    base = {"κ_m": s4.KAPPA_M, "ρ(P,m)": s4.RHO_PM}
    D["是否基准"] = [abs(r["取值"] - base[r["参数"]]) < 1e-9
                     for _, r in D.iterrows()]
    D["GP相对基准"] = np.nan
    for (tag, scope), g in D.groupby(["标度档", "参数"]):
        g0 = g[g["是否基准"]]["毛保费GP_乙"]
        if len(g0):
            D.loc[g.index, "GP相对基准"] = (
                g["毛保费GP_乙"] / float(g0.iloc[0])).round(3)
    return D


# ============================================================ 验收
def checks(r0out, r1meta, r2d, r3d, r4d, r5d, r6d, r7boot, r7meta, r8d,
           r9d) -> list[dict]:
    res = []

    def add(tag, desc, ok, got, exp):
        res.append({"验收项": tag, "判据": desc, "结果": "PASS" if ok else "FAIL",
                    "实测": got, "期望": exp})

    add("R0", "simulate_p 与 s4.simulate 逐位相同", r0out["same_I"] and r0out["same_Q"],
        f"I={r0out['same_I']} Q={r0out['same_Q']} max|ΔI|={r0out['max_dI']:.3e}", "逐位相同")
    mv = r0out["最大偏差"]
    add("R0b", "论文口径档 3y 指标与 s22 主表一致（CSV 舍入容差 0.5）",
        mv is not None and mv <= 0.5,
        f"最大绝对偏差 {mv}", "≤ 0.5 元/吨（CSV 已舍入）")

    a = r1meta["R1a_扫描盒截断"]
    add("R1a", "存在合法点 MAC < 131.445（β 在扫描盒内）",
        a["最低合法点"] is not None and a["最低合法点"]["MAC2022"] < REF_LO_FINER,
        f"最低 {a['最低合法点']['MAC2022']:.2f}（γ1={a['最低合法点']['γ1']}, "
        f"β={a['最低合法点']['β']}）；较 131.445 低 {a['更低的幅度%']}%",
        "< 131.445 ⇒ 131.445 非下确界")
    b = r1meta["R1b_退化解"]
    add("R1b", "存在 dev_max≤1 且 MAC≈0 的退化合法点", b["退化点数（MAC<1）"] > 0,
        f"{b['退化点数（MAC<1）']} 个退化点", "> 0 ⇒ 合法域无正下界")
    c = r1meta["R1c_上界侧"]
    add("R1c", "上界侧扩展不突破 585.871",
        c["最高合法点"] is not None and c["最高合法点"]["MAC2022"] <= REF_HI * 1.001,
        f"扩展扫描最高 {c['最高合法点']['MAC2022']:.2f} ≤ 585.871",
        "≤ 585.871×1.001")

    q = r2d[r2d["标度档"] == TIER_PAPER]
    add("R2", "主报价档 GP 逐期限为正且 ROL ∈ [8%,15%]",
        bool((q["毛保费GP_乙"] > 0).all() and q["ROL_乙%"].between(8, 15).all()),
        f"3/5/10/15y GP = {list(q['毛保费GP_乙'])}, ROL = {list(q['ROL_乙%'])}",
        "GP>0 且 ROL ∈ [8,15]%")
    add("R2b", "毛保费随期限单调递增",
        bool(q.sort_values("期限")["毛保费GP_乙"].is_monotonic_increasing),
        f"{list(q.sort_values('期限')['毛保费GP_乙'])}", "单调递增")

    add("R3", "最低资本 = φ·RM（不含 E、不除 1−e−π），且必小于所承接层宽",
        bool((r3d["最低资本_组合口径"] < r3d["层宽A−D"] + 1e-9).all()
             and (r3d["最低资本_组合口径"] > 0).all()
             and (r3d["最低资本_组合口径"] <= r3d["RM_单保单"] + 1e-9).all()),
        f"资本/层宽 中位 {r3d['资本比层宽%'].median():.1f}%；"
        f"资本/保费 ∈ [{r3d['参考_资本比保费'].min():.2f}, {r3d['参考_资本比保费'].max():.2f}]"
        "（资本 > 保费是层产品的固有特征：保费只收资本成本 ρ·RM）",
        "资本 < 层宽 且 组合口径 ≤ 单保单口径")
    sup = r3d[r3d["标度档"] == TIER_SUP]
    add("R3b", "上确界端点档进入最低资本测算（资本侧取上端）",
        bool(len(sup) > 0 and sup["最低资本_组合口径"].max() > 0),
        f"上确界端点档最低资本（3y）= {sup[sup['期限'] == 3]['最低资本_组合口径'].iloc[0]:.1f}",
        "> 0 且随表披露")

    p1 = r4d[(r4d["情景"] == "P1_省级锚·随机游走") & (r4d["标度档"] == TIER_PAPER)]
    nn = r4d[(r4d["情景"] == "N_全国系统性版") & (r4d["标度档"] == TIER_PAPER)]
    mg = p1.merge(nn[["期限", "毛保费GP_乙"]].rename(columns={"毛保费GP_乙": "GP_nat"}),
                  on="期限")
    add("R4", "省级锚情景保费 ≥ 全国版（基差扰动抬高触发与赔付）",
        bool((mg["毛保费GP_乙"] >= mg["GP_nat"] - 1e-9).all()),
        "；".join(f"{int(r['期限'])}y {r['毛保费GP_乙']:.0f} vs {r['GP_nat']:.0f}"
                 for _, r in mg.iterrows()), "逐期限 ≥ 全国版")

    q5 = r5d[(r5d["标度档"] == TIER_PAPER) & (r5d["期限"] == 3)].sort_values("σ_m")
    add("R5", "σ_m 对毛保费单调递增", bool(q5["毛保费GP_乙"].is_monotonic_increasing),
        f"3y GP = {list(q5['毛保费GP_乙'])}（σ_m {list(q5['σ_m'])}）", "单调递增")
    # R5b：基准档一致性 —— σ_m 取基准值时各档 GP 必须等于 s22 主表值
    for tag, n in ((TIER_PAPER, 3), (TIER_PAPER, 5), (TIER_OLDQ, 5)):
        row5 = r5d[(r5d["标度档"] == tag) & (r5d["期限"] == n)
                   & (np.isclose(r5d["σ_m"], s4.SIG_M_PATH))]
        row2 = r2d[(r2d["标度档"] == tag) & (r2d["期限"] == n)]
        if len(row5) and len(row2):
            got5 = float(row5["毛保费GP_乙"].iloc[0])
            got2 = float(row2["毛保费GP_乙"].iloc[0])
            add("R5b" if tag == TIER_PAPER else "R5c",
                f"{tag}·{n}y：σ_m 基准取值下的 GP 与 s22 主表一致"
                "（逐档种子一致性的负向守卫）",
                abs(got5 - got2) <= 0.5,
                f"σ_m 扫描基准 {got5:.0f} vs s22 主表 {got2:.0f}",
                "|Δ| ≤ 0.5 元/吨（否则说明用了错的档位种子）")

    add("R6", "σ_Q 影响方向：敞口波动↑ ⇒ RM↑ ⇒ GP↑",
        bool(r6d[r6d["期限"] == 3].sort_values("σ_Q")["毛保费GP_乙"]
             .is_monotonic_increasing),
        f"3y GP = {list(r6d[r6d['期限'] == 3].sort_values('σ_Q')['毛保费GP_乙'])}",
        "单调递增")

    g7 = r7boot[r7boot["统计量"] == "GP"].iloc[0]
    w7 = r7boot[r7boot["统计量"] == "A−D"].iloc[0]
    add("R7", "新主报价档 3y 尾部收敛：GP 相对SE ≤ 3%、层宽相对SE ≤ 5%",
        bool(float(g7["相对SE%"]) <= 3.0 and float(w7["相对SE%"]) <= 5.0),
        f"GP 相对SE {g7['相对SE%']}%、层宽 {w7['相对SE%']}%；"
        f"路径数 2→8 万 GP 变动 {r7meta['GP变动%']}%", "≤ 3% / ≤ 5%")

    x8 = r8d[r8d["期限"] == 5].pivot_table(index="标度档", columns="CAP倍数",
                                           values="毛保费GP_乙")
    rx = r8d.pivot_table(index=["标度档", "期限"], columns="CAP倍数",
                         values="ROL_乙%")
    rx["极差_pp"] = rx.max(axis=1) - rx.min(axis=1)
    add("R8", "CAP 倍数只改变保障额度、不改变承保成本比率（ROL 极差 ≤ 1.5pp）",
        bool((rx["极差_pp"] <= 1.5).all()),
        "；".join(
            f"{t} 5y 1×={x8.loc[t, 1.0]:.0f}→2×={x8.loc[t, 2.0]:.0f}"
            f"→4×={x8.loc[t, 4.0]:.0f} 元/吨（ROL 极差 "
            f"{rx.loc[(t, 5), '极差_pp']:.1f}pp）" for t in x8.index),
        "ROL 极差 ≤ 1.5pp")

    g9 = r9d[r9d["标度档"] == TIER_PAPER]
    km = g9[g9["参数"] == "κ_m"].sort_values("取值")
    rp = g9[g9["参数"] == "ρ(P,m)"].sort_values("取值")
    add("R9", "κ_m ↑ ⇒ GP ↓；ρ(P,m) ↑ ⇒ GP ↑（单调）",
        bool(km["毛保费GP_乙"].is_monotonic_decreasing
             and rp["毛保费GP_乙"].is_monotonic_increasing),
        f"κ_m {list(km['取值'])} → GP {list(km['毛保费GP_乙'])}；"
        f"ρ(P,m) {list(rp['取值'])} → GP {list(rp['毛保费GP_乙'])}",
        "κ_m 单调不增、ρ(P,m) 单调不减")
    for tag in (TIER_PAPER, TIER_OLDQ):
        g0 = r9d[(r9d["标度档"] == tag) & (r9d["参数"] == "ρ(P,m)")
                 & (np.isclose(r9d["取值"], s4.RHO_PM))]
        r0d = r2d[(r2d["标度档"] == tag) & (r2d["期限"] == 5)]
        if len(g0) and len(r0d):
            add("R9b" if tag == TIER_PAPER else "R9c",
                f"{tag}·5y：ρ(P,m) 基准取值下的 GP 与 s22 主表一致"
                "（逐档种子一致性的负向守卫）",
                abs(float(g0["毛保费GP_乙"].iloc[0])
                    - float(r0d["毛保费GP_乙"].iloc[0])) <= 0.5,
                f"扫描基准 {float(g0['毛保费GP_乙'].iloc[0]):.0f} vs "
                f"s22 主表 {float(r0d['毛保费GP_乙'].iloc[0]):.0f}",
                "|Δ| ≤ 0.5 元/吨")
    return res


# ============================================================ 主
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    only = "--check" in sys.argv
    t0 = time.time()

    tiers = s4._lp_cache()
    # 档位种子必须与 s22 一致：s22 按 dict 顺序取 RNG_SEED + i*977
    seeds = {tag: s4.RNG_SEED + i * 977 for i, tag in enumerate(tiers)}
    sims = {tag: s4.simulate("S1_基准", tiers[tag], "M2_价格平移",
                             L1["mc_paths"], seeds[tag])
            for tag in tiers}
    # 追加：上确界端点档（资本侧上端）；γ1 因含 Unicode 下标故标签用 ASCII
    sup_raw = _tier(-0.125, 1.8, TIER_SUP)
    tiers[TIER_SUP] = sup_raw
    seeds[TIER_SUP] = s4.RNG_SEED + 20 * 977
    sims[TIER_SUP] = s4.simulate("S1_基准", sup_raw, "M2_价格平移",
                                 L1["mc_paths"], seeds[TIER_SUP])

    print("=" * 104)
    print("s24　第二轮审稿响应工作包（R0–R9）")
    print("=" * 104)

    r0out = r0(tiers)
    print("\n【R0】复现锚")
    print("-" * 104)
    print(f"  逐位相同：I={r0out['same_I']}  Q={r0out['same_Q']}  最大|ΔI|={r0out['max_dI']:.3e}")
    print(f"  论文口径档 3y 复算：{r0out['复算_3y']}")
    print(f"  s22 主表对照：      {r0out['s22对照_3y']}")
    print(f"  最大绝对偏差：{r0out['最大偏差']}")

    d1, r1meta = r1()
    print("\n【R1】二维合法域诊断")
    print("-" * 104)
    print(json.dumps(r1meta, ensure_ascii=False, indent=2))

    d2 = r2(sims, tiers)
    print("\n【R2】主报价档重建算（冻结标度约定档 γ1=0 · β=1；单位 元/吨）")
    print("-" * 104)
    print(d2.to_string(index=False))

    d3 = r3(sims, tiers)
    print("\n【R3】最低资本线（= φ·RM；不含 E、不除 (1−e−π)）")
    print("-" * 104)
    print(d3.to_string(index=False))

    d4 = r4(sims, tiers)
    print("\n【R4】省级锚情景 vs 全国系统性版（冻结约定档）")
    print("-" * 104)
    print(d4.to_string(index=False))

    d5 = r5(tiers, seeds)
    print("\n【R5】σ_m 报价带")
    print("-" * 104)
    print(d5.to_string(index=False))

    d6, r6meta = r6(tiers, seeds)
    print("\n【R6】敞口变量 Q_t")
    print("-" * 104)
    print(d6.to_string(index=False))
    print(json.dumps(r6meta, ensure_ascii=False, indent=2))

    d7a, d7b, r7meta = r7(tiers, seeds)
    print("\n【R7】尾部估计收敛诊断（新主报价档 · B1 · 3y · D75 · e12 · n100）")
    print("-" * 104)
    print(d7a.to_string(index=False))
    print(d7b.to_string(index=False))
    print(json.dumps(r7meta, ensure_ascii=False, indent=2))

    d8 = r8(sims, tiers)
    print("\n【R8】CAP 倍数敏感性（确定性重算，无新增抽样噪声）")
    print("-" * 104)
    print(d8.to_string(index=False))

    d9 = r9(tiers, seeds)
    print("\n【R9】κ_m 与 ρ(P,m) 敏感性（M4 标定杠杆；逐档种子）")
    print("-" * 104)
    print(d9.to_string(index=False))

    res = checks(r0out, r1meta, d2, d3, d4, d5, d6, d7a, r7meta, d8, d9)
    print("\n【验收】")
    print("-" * 104)
    for r in res:
        print(f"  [{r['结果']:4s}] {r['验收项']:5s} {r['判据']}　→ {r['实测']}")
    nfail = sum(1 for r in res if r["结果"] == "FAIL")
    print(f"\n  合计：{len(res) - nfail} PASS / {nfail} FAIL　（耗时 {time.time() - t0:.0f}s）")

    if not only:
        d1.to_csv(OUT / "s24_R1_域诊断.csv", index=False, encoding="utf-8-sig")
        d2.to_csv(OUT / "s24_R2_主报价单.csv", index=False, encoding="utf-8-sig")
        d3.to_csv(OUT / "s24_R3_最低资本.csv", index=False, encoding="utf-8-sig")
        d4.to_csv(OUT / "s24_R4_省级锚情景.csv", index=False, encoding="utf-8-sig")
        d5.to_csv(OUT / "s24_R5_sigma_m报价带.csv", index=False, encoding="utf-8-sig")
        d6.to_csv(OUT / "s24_R6_敞口敏感性.csv", index=False, encoding="utf-8-sig")
        d7a.to_csv(OUT / "s24_R7a_自助标准误.csv", index=False, encoding="utf-8-sig")
        d7b.to_csv(OUT / "s24_R7b_路径数稳定带.csv", index=False, encoding="utf-8-sig")
        d8.to_csv(OUT / "s24_R8_CAP倍数.csv", index=False, encoding="utf-8-sig")
        d9.to_csv(OUT / "s24_R9_标定杠杆.csv", index=False, encoding="utf-8-sig")
        with open(OUT / "s24_R0R1_复算与域诊断.json", "w", encoding="utf-8") as f:
            json.dump({"R0_复现锚": r0out, "R1_域诊断": r1meta,
                       "R6_敞口解析": r6meta, "R7_收敛诊断": r7meta,
                       "验收": res}, f, ensure_ascii=False, indent=2)
        print("\n[已写出] s24_R1_域诊断.csv / s24_R2_主报价单.csv / s24_R3_最低资本.csv / "
              "s24_R4_省级锚情景.csv / s24_R5_sigma_m报价带.csv / "
              "s24_R6_敞口敏感性.csv / s24_R7a_自助标准误.csv / "
              "s24_R7b_路径数稳定带.csv / s24_R8_CAP倍数.csv / "
              "s24_R9_标定杠杆.csv / s24_R0R1_复算与域诊断.json")

    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
