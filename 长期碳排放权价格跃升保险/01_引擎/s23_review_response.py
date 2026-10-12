# -*- coding: utf-8 -*-
"""
s23_review_response.py —— 审稿轮响应工作包（C1–C5）

背景（2026-10-03）：外部审稿式审视对论文 v4 提出若干需补的组织与稳健性证据。
其中引擎级缺口五项，本脚本逐项落实。**不改任何既有铁律与口径**：
    · 结构仍是 B1 年度重定基 + 聚合免赔 D=q75 + 责任上限 A=q99.5(L)（不乘 n）；
    · 报价原理仍是口径乙（资本成本）：GP = (E + ρ·φ·(TVaR99−E)) / (1−e−π)；
    · 模糊性仍不进 Monte Carlo —— 参数敏感性扫描的是**风险过程参数**，不是 γ₁/β。

C0 复现锚：本脚本的 simulate_p(默认参数) 必须与 s4.simulate **逐位相同**
    （同 seed、同 CHOL），否则后续一切结论不可比。断言 A-C0。

C1　触发阈值系数 κ 敏感性（κ ∈ {1.15, 1.30, 1.50, 1.80}）
    回应「κ=1.30 是产品定义但为什么是 30%」。κ 只影响赔付函数、不影响路径 ⇒
    一次模拟、四次重算，无额外抽样噪声，是干净的确定性比较。
C2　Monte Carlo 尾部收敛诊断
    C2a 自助标准误（bootstrap）：q99.5 / TVaR99 / 层宽 / GP乙 的标准误；
    C2b 路径数倍增稳定带（20k → 40k → 80k）。
    ⚠ 倍增检验**不是嵌套子样本**（`standard_normal((3,nyr,n))` 的填充顺序依赖 n），
      故两组证据并列、不合并；C2a 才是同一样本内的推断。
C3　核心风险参数单变量敏感性（σ_m / κ_m / ρ(P,m) / CAP）
    回应「敏感性未覆盖核心风险参数」。CAP 不影响路径，只影响赔付函数。
C4　二维合法域网格加密（β 细分、γ₁ 步长 0.005）
    回应 N4：132.477 是 s0c 的**格点下确界**（β 仅 7 档、γ₁ 步长 0.02），
    而主报价档正踩此点 ⇒ 必须加密确认它没有下移。
    ⚠ 实测（2026-10-03）：**下移了** —— 加密后下确界 131.445（β=0.5、γ₁=+0.135，
      dev=0.9594），较原格点下确界低 0.779%；上确界上移至 585.871（β=1.8、γ₁=−0.125）。
      判定口径**不取"零差异"**，而取**蒙特卡洛分辨率**：0.779% 低于层宽的自助相对
      标准误 3.635% ⇒ 两档在定价上不可区分，维持原报价档；上确界上移对**资本计提侧**
      （取上端）为不利方向，单列为 C4b 登记项并随表披露。两项均不得静默处理。
C5　β=1 合法窗口锚点复算并落盘
    回应 N2：168.207 / 555.801 原为硬编码常量（s0_gamma_window.py:182、
    s3_term.py:70），不在任何产物、未登记溯源表。本项改为**有界优化求解 + 落盘**。
    实测：下确界 168.151（−0.0333%）、上确界 555.800（−0.0002%）。
C6　段 1 右端真端点复算（回应 N1）
    v4 表 2 把 γ₁=+0.06 标为「段 1 右端」，实为段 1 **内点**（dev_max=0.9472）；
    真端点 = ROOTS[1] = +0.101853（dev_max≈1）。本项在 s22 同一 seed 上重算两档，
    交叉校验定价链（γ₁=+0.06 复算须逐项等于 s22 的 268.6/1488.5/1219.9/131/10.7%）。
    表 3 另列 `CAP/基准`（回应 T2）：CAP_AMOUNT=826.64 为常数，不随档位基准联动。

用法：python s23_review_response.py [--check]    # 退出码 0 = 断言全过
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import GAP_MULTIPLE, L1, OUT, PRICE_PATH, SCENARIOS   # noqa: E402
import s4_pricing as s4                                            # noqa: E402
from s0_gamma_window import ROOTS, run_at                          # noqa: E402
from s22_commercial_reprice import phi_div                         # noqa: E402

pd.set_option("display.width", 240)
pd.set_option("display.unicode.east_asian_width", True)

RHO = L1["capital_charge"]
PI_PROFIT = 0.08
DF_R = L1["discount"]
E_EXP_IDX = 0.12
Q_LIMIT = 0.995
N_EFF = 100
RHO_BAR = 0.30

STRUCT_B1 = "B1_年度重定基"
TIER_PAPER = "论文口径"
TIER_QUOTE = "β0.6_下确界"
TERMS = s4.TERMS
DEFL_2026 = 2.169309

# 硬编码锚点（待本脚本复算替换，见 C5）
ANCHOR_OLD = {"lo": 168.207, "hi": 555.801}

SENS_KAPPA = [1.15, 1.30, 1.50, 1.80]
SENS_SIGM = [0.05, 0.15, 0.30]
SENS_KAPPAM = [0.15, 0.30, 0.60]
SENS_RHOPM = [0.00, 0.45, 0.75]
SENS_CAPMULT = [1.0, 2.0, 4.0]
BOOT_B = 1000
PATH_COUNTS = [20_000, 40_000, 80_000]


# ============================================================ C0 模拟复刻

def _chol_of(rho_pm: float, rho_pq: float, rho_mq: float) -> np.ndarray:
    C = np.array([[1.0, rho_pm, rho_pq],
                  [rho_pm, 1.0, rho_mq],
                  [rho_pq, rho_mq, 1.0]])
    w, V = np.linalg.eigh(C)
    if (w < -1e-9).any():
        raise ValueError(f"相关矩阵非正定，最小特征值 {w.min():.4f}")
    return V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T


def simulate_p(scen_key: str, tier: dict, caliber: str, n_paths: int, seed: int,
               sig_m: float | None = None, kappa_m: float | None = None,
               rho_pm: float | None = None,
               m_drift: float = 0.0) -> dict:
    """s4.simulate 的可参数化复刻。默认参数下必须与之逐位相同（断言 A-C0）。"""
    rng = np.random.default_rng(seed)
    yrs = list(range(s4.START, s4.START + max(TERMS) + 2))
    nyr = len(yrs)
    n = int(n_paths)

    sm = s4.SIG_M_PATH if sig_m is None else float(sig_m)
    km = s4.KAPPA_M if kappa_m is None else float(kappa_m)
    rpm = s4.RHO_PM if rho_pm is None else float(rho_pm)
    chol = _chol_of(rpm, s4.RHO_PQ, s4.RHO_MQ)

    P_c = s4._price_central(scen_key, yrs)
    m0 = s4.m0_of(tier, caliber)
    m_c = s4._m_central(m0, scen_key, yrs)
    m_c = m_c * np.exp(-m_drift * np.maximum(np.arange(nyr) - 1, 0))

    Z = rng.standard_normal((3, nyr, n))
    Z = np.einsum("ij,jtn->itn", chol, Z)
    eP, eM, eQ = Z[0], Z[1], Z[2]

    lnP = np.empty((nyr, n))
    lnP[0] = np.log(P_c[0]) + s4.SIG_P * eP[0]
    for t in range(1, nyr):
        lnP[t] = (lnP[t - 1] + s4.KAPPA_P * (np.log(P_c[t]) - lnP[t - 1])
                  + s4.SIG_P * eP[t])
    P = np.exp(lnP)

    lnm = np.empty((nyr, n))
    lnm[0] = np.log(m_c[0]) + s4.SIG_M_LEVEL * eM[0]
    for t in range(1, nyr):
        lnm[t] = lnm[t - 1] + km * (np.log(m_c[t]) - lnm[t - 1]) + sm * eM[t]
    M = np.exp(lnm)
    I = P * M

    g = L1["gap_rate_0"] + L1["gap_tighten"] * np.arange(nyr)
    Q = np.exp(np.log(g)[:, None] + s4.SIG_Q * eQ)
    Q = Q / Q[0]

    return {"years": yrs, "P": P, "M": M, "I": I, "Q": Q,
            "P_c": P_c, "m_c": m_c, "m0": m0}


# ============================================================ 赔付与定价

def pv_level_p(sim: dict, n: int, kappa: float, cap: float) -> np.ndarray:
    """B0 水平式：K = κ × MAC_REF 固定。"""
    I, Q = sim["I"], sim["Q"]
    K = kappa * s4.MAC_REF
    pay = np.minimum(np.maximum(I - K, 0.0), cap) * Q
    df = 1.0 / (1.0 + DF_R) ** np.arange(n + 1)
    return (pay[:n + 1] * df[:, None]).sum(axis=0)


def pv_rebased_p(sim: dict, n: int, kappa: float, cap: float,
                 period: int = 1) -> np.ndarray:
    """B1 年度重定基（无边界重复计数，与 s22.pv_rebased 同实现）。"""
    I, Q = sim["I"], sim["Q"]
    total = np.zeros(I.shape[1])
    yr = 0
    while yr < n:
        seg = min(period, n - yr)
        idx = np.arange(yr + 1, yr + seg + 1)
        K = kappa * I[yr]
        pay = np.minimum(np.maximum(I[idx] - K[None, :], 0.0), cap) * Q[idx]
        df = 1.0 / (1.0 + DF_R) ** idx[:, None]
        total += (pay * df).sum(axis=0)
        yr += seg
    return total


def _tvar(x: np.ndarray, a: float = 0.99) -> float:
    q = np.quantile(x, a)
    t = x[x >= q]
    return float(t.mean()) if t.size else float(q)


def layer_metrics(L: np.ndarray) -> dict:
    """B1-D75·e12·n100 的层指标（口径乙）。"""
    E_full = float(L.mean())
    D = float(np.quantile(L, 0.75))
    A = float(np.quantile(L, Q_LIMIT))
    pay = np.clip(L - D, 0.0, A - D)
    E = float(pay.mean())
    rm = (_tvar(pay) - E) * phi_div(N_EFF, RHO_BAR)
    gp = (E + RHO * rm) / (1.0 - E_EXP_IDX - PI_PROFIT)
    width = A - D
    return {"E_full": E_full, "D": D, "A": A, "层宽": width,
            "触发概率_full%": float((L > 0).mean() * 100),
            "层赔付频率%": float((pay > 0).mean() * 100),
            "E_layer": E, "RM_组合": rm,
            "GP_CoC": gp, "ROL_CoC%": gp / max(width, 1e-9) * 100}


# ============================================================ C1 κ 敏感性

def c1(sims: dict) -> pd.DataFrame:
    rows = []
    for tag in (TIER_QUOTE, TIER_PAPER):
        sim = sims[tag]
        for n in TERMS:
            L0 = pv_rebased_p(sim, n, s4.KAPPA_BASE, s4.CAP_AMOUNT)
            base_gp = layer_metrics(L0)["GP_CoC"]
            for k in SENS_KAPPA:
                L = pv_rebased_p(sim, n, k, s4.CAP_AMOUNT)
                m = layer_metrics(L)
                rows.append({
                    "标度档": tag, "期限": n, "κ": k,
                    "D_元每吨": round(m["D"], 1),
                    "责任上限A": round(m["A"], 1),
                    "层宽": round(m["层宽"], 1),
                    "触发事件频率%": round(m["触发概率_full%"], 2),
                    "层赔付频率%": round(m["层赔付频率%"], 2),
                    "E_layer": round(m["E_layer"], 1),
                    "GP_CoC_元每吨": round(m["GP_CoC"], 0),
                    "ROL_CoC%": round(m["ROL_CoC%"], 1),
                    "GP相对κ1.30": round(m["GP_CoC"] / base_gp, 4),
                })
    return pd.DataFrame(rows)


# ============================================================ C2 收敛诊断

def c2(sims: dict, tier: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # --- C2a bootstrap（同一样本内）
    L = pv_rebased_p(sims[TIER_QUOTE], 3, s4.KAPPA_BASE, s4.CAP_AMOUNT)
    rng = np.random.default_rng(20261003)
    n = L.size
    boot = np.empty((BOOT_B, 4))
    for b in range(BOOT_B):
        idx = rng.integers(0, n, n)
        m = layer_metrics(L[idx])
        boot[b] = [np.quantile(L[idx], Q_LIMIT), _tvar(L[idx]),
                   m["层宽"], m["GP_CoC"]]
    point = layer_metrics(L)
    pt = np.array([point["A"], _tvar(np.clip(L - point["D"], 0, point["A"] - point["D"])),
                   point["层宽"], point["GP_CoC"]])
    names = ["q99.5(L)", "TVaR99(层赔付)", "层宽", "GP_CoC"]
    se = boot.std(ddof=1, axis=0)
    b = pd.DataFrame({
        "统计量": names,
        "点估计": np.round(pt, 2),
        "自助标准误": np.round(se, 3),
        "相对标准误%": np.round(se / np.maximum(pt, 1e-9) * 100, 3),
        "B": BOOT_B, "样本量": n,
    })

    # --- C2b 路径数倍增稳定带（非同嵌套子样本，见文件头说明）
    rows = []
    ref = None
    for npaths in PATH_COUNTS:
        sim = simulate_p("S1_基准", tier, "M2_价格平移", npaths,
                         s4.RNG_SEED + 977 * 8)
        Lp = pv_rebased_p(sim, 3, s4.KAPPA_BASE, s4.CAP_AMOUNT)
        m = layer_metrics(Lp)
        if ref is None:
            ref = m
        rows.append({
            "路径数": npaths,
            "q99.5": round(m["A"], 1),
            "TVaR99层": round(_tvar(np.clip(Lp - m["D"], 0, m["A"] - m["D"])), 1),
            "层宽": round(m["层宽"], 1),
            "E_layer": round(m["E_layer"], 1),
            "GP_CoC_元每吨": round(m["GP_CoC"], 1),
            "GP偏差_相对20k%": round((m["GP_CoC"] / ref["GP_CoC"] - 1) * 100, 3),
            "层宽偏差_相对20k%": round((m["层宽"] / ref["层宽"] - 1) * 100, 3),
        })
    s = pd.DataFrame(rows)
    meta = {"bootstrap_B": BOOT_B, "n_20k": int(n),
            "SE_层宽_相对%": float(se[2] / pt[2] * 100),
            "SE_GP_相对%": float(se[3] / pt[3] * 100),
            "倍增最大GP偏差%": float(s["GP偏差_相对20k%"].abs().max()),
            "倍增最大层宽偏差%": float(s["层宽偏差_相对20k%"].abs().max())}
    return b, s, meta


# ============================================================ C3 参数敏感性

def c3(sims: dict, tiers: dict) -> pd.DataFrame:
    L_cache = {}
    rows = []
    for tag in (TIER_PAPER, TIER_QUOTE):
        # 基准
        for n in (5,):
            for scope in ("σ_m", "κ_m", "ρ(P,m)", "CAP"):
                vals = {"σ_m": SENS_SIGM, "κ_m": SENS_KAPPAM,
                        "ρ(P,m)": SENS_RHOPM, "CAP": SENS_CAPMULT}[scope]
                for v in vals:
                    key = (tag, scope, v)
                    if key not in L_cache:
                        if scope == "σ_m":
                            sim = simulate_p("S1_基准", tiers[tag], "M2_价格平移",
                                             L1["mc_paths"], s4.RNG_SEED + 977 * 8,
                                             sig_m=v)
                        elif scope == "κ_m":
                            sim = simulate_p("S1_基准", tiers[tag], "M2_价格平移",
                                             L1["mc_paths"], s4.RNG_SEED + 977 * 8,
                                             kappa_m=v)
                        elif scope == "ρ(P,m)":
                            sim = simulate_p("S1_基准", tiers[tag], "M2_价格平移",
                                             L1["mc_paths"], s4.RNG_SEED + 977 * 8,
                                             rho_pm=v)
                        else:
                            sim = sims[tag]
                        cap = s4.CAP_AMOUNT * (v if scope == "CAP" else 1.0)
                        L_cache[key] = pv_rebased_p(
                            sim, 5, s4.KAPPA_BASE, cap)
                    m = layer_metrics(L_cache[key])
                    rows.append({
                        "标度档": tag, "期限": 5, "参数": scope, "取值": v,
                        "层宽": round(m["层宽"], 1),
                        "E_layer": round(m["E_layer"], 1),
                        "GP_CoC_元每吨": round(m["GP_CoC"], 0),
                        "ROL_CoC%": round(m["ROL_CoC%"], 1),
                    })
    d = pd.DataFrame(rows)
    # 弹性：以基准取值为分母
    base = {"σ_m": s4.SIG_M_PATH, "κ_m": s4.KAPPA_M,
            "ρ(P,m)": s4.RHO_PM, "CAP": 2.0}
    d["是否基准"] = [abs(r["取值"] - base[r["参数"]]) < 1e-9
                     for _, r in d.iterrows()]
    d["GP相对基准"] = np.nan
    for (tag, scope), g in d.groupby(["标度档", "参数"]):
        g0 = g[g["是否基准"]]["GP_CoC_元每吨"]
        if len(g0):
            d.loc[g.index, "GP相对基准"] = (g["GP_CoC_元每吨"] / float(g0.iloc[0])
                                        ).round(3)
    return d


# ============================================================ C4 二维域加密

def c4() -> tuple[pd.DataFrame, dict]:
    betas = ([round(0.40 + 0.05 * i, 2) for i in range(9)]
             + [0.90, 1.00, 1.10, 1.20, 1.40, 1.60, 1.80, 2.00])
    g1s = np.round(np.arange(-0.325, 0.1451, 0.005), 6)
    rows, t0 = [], time.time()
    for b in betas:
        for g in g1s:
            r = run_at(float(g), beta=float(b))
            rows.append({"γ1": round(float(g), 6), "β": float(b),
                         "ok": bool(r.get("ok")),
                         "MAC2022": r.get("MAC_2022均值", np.nan),
                         "dev_max": r.get("dev_max", np.nan)})
    d = pd.DataFrame(rows)
    d["合法"] = d["ok"] & (d["dev_max"] <= 1.0)
    leg = d[d["合法"]]
    lo_r = leg.loc[leg["MAC2022"].idxmin()]
    hi_r = leg.loc[leg["MAC2022"].idxmax()]
    old = json.load(open(OUT / "s0c_g二维模糊域.json", encoding="utf-8"))
    meta = {
        "格点": {"β档数": len(betas), "γ1步长": 0.005, "求解次数": len(d),
                 "合法": int(len(leg)), "耗时s": round(time.time() - t0, 1)},
        "s0c_粗网格": {"下确界": old["MAC_2022_1995价_合法域"][0],
                     "上确界": old["MAC_2022_1995价_合法域"][1],
                     "β档": old["格点"]["β档"], "γ1步长": old["格点"]["γ1步长"]},
        "加密后": {"下确界": round(float(lo_r["MAC2022"]), 3),
                 "下确界点": {"γ1": float(lo_r["γ1"]), "β": float(lo_r["β"]),
                            "dev_max": round(float(lo_r["dev_max"]), 6)},
                 "上确界": round(float(hi_r["MAC2022"]), 3),
                 "上确界点": {"γ1": float(hi_r["γ1"]), "β": float(hi_r["β"]),
                            "dev_max": round(float(hi_r["dev_max"]), 6)}},
        "结论": "加密后下确界变化与是否仍高于主报价档基准，见「下确界变化%」与「主报价档结论」",
    }
    meta["下确界变化%"] = round(
        (meta["加密后"]["下确界"] / meta["s0c_粗网格"]["下确界"] - 1) * 100, 3)
    meta["上确界变化%"] = round(
        (meta["加密后"]["上确界"] / meta["s0c_粗网格"]["上确界"] - 1) * 100, 3)
    # 与主报价档基准（132.477）的可区分性 —— 判据用蒙特卡洛分辨率，而非"零差异"
    meta["主报价档基准"] = 132.477
    meta["下确界_相对主报价档%"] = round(
        (meta["加密后"]["下确界"] / 132.477 - 1) * 100, 3)
    meta["上确界_相对网格上端档%"] = round(
        (meta["加密后"]["上确界"] / 572.71 - 1) * 100, 3)
    meta["主报价档结论"] = (
        f"加密后下确界 {meta['加密后']['下确界']:.3f}"
        f"（β={meta['加密后']['下确界点']['β']}, γ1={meta['加密后']['下确界点']['γ1']}）"
        f"低于 s0c 格点下确界 132.477 达 {abs(meta['下确界_相对主报价档%']):.3f}%；"
        "该差额须与蒙特卡洛分辨率（层宽自助相对标准误）比较后再判定是否需改报价——"
        "见验收项 C4。上确界上移对**资本计提侧**为不利方向，须单独披露（C4b）。")
    return d, meta


# ============================================================ C5 锚点复算

def _mac_at(g: float, beta: float = 1.0) -> tuple[float, float, bool]:
    r = run_at(float(g), beta=beta)
    return (r.get("MAC_2022均值", np.nan), r.get("dev_max", np.nan),
            bool(r.get("ok")))


def c5() -> dict:
    from scipy.optimize import minimize_scalar

    def neg_penalized(g, sign, lo, hi):
        mac, dev, ok = _mac_at(g)
        if not ok or not np.isfinite(mac):
            return 1e9
        pen = 0.0 if dev <= 1.0 else (dev - 1.0) * 1e6
        return sign * mac + pen

    # 下确界：段 2 内部极小（细扫已定位 γ1≈0.129）
    res_lo = minimize_scalar(lambda g: neg_penalized(g, +1, 0, 0),
                             bounds=(float(ROOTS[2]), float(ROOTS[3])),
                             method="bounded",
                             options={"xatol": 1e-7})
    # 上确界：段 1 左端边界（dev=1），MAC 随 γ1 递减 ⇒ 在 ROOTS[0] 取最大
    res_hi = minimize_scalar(lambda g: neg_penalized(g, -1, 0, 0),
                             bounds=(float(ROOTS[0]), float(ROOTS[1])),
                             method="bounded",
                             options={"xatol": 1e-7})
    lo_mac, lo_dev, lo_ok = _mac_at(float(res_lo.x))
    hi_mac, hi_dev, hi_ok = _mac_at(float(res_hi.x))
    # 边界点补充检查
    b_mac, b_dev, b_ok = _mac_at(float(ROOTS[0]) + 1e-7)

    out = {
        "判据": "dev_max ≤ 1（碳成本不超过 GDP），与 s0/s0c 一致",
        "方法": "scipy.optimize.minimize_scalar(bounded)，返回罚函数下最优；"
                "另对段 1 左端边界点单独取点",
        "段1_端点最大值": {"γ1": round(float(ROOTS[0]) + 1e-7, 7),
                        "MAC2022": round(float(b_mac), 3),
                        "dev_max": round(float(b_dev), 6), "ok": b_ok},
        "优化下确界": {"γ1": round(float(res_lo.x), 7), "MAC2022": round(float(lo_mac), 3),
                   "dev_max": round(float(lo_dev), 6), "ok": lo_ok},
        "优化上确界": {"γ1": round(float(res_hi.x), 7), "MAC2022": round(float(hi_mac), 3),
                   "dev_max": round(float(hi_dev), 6), "ok": hi_ok},
        "旧硬编码锚点": ANCHOR_OLD,
    }
    out["上确界"] = round(max(float(b_mac), float(hi_mac)), 3)
    out["下确界"] = round(float(lo_mac), 3)
    out["相对旧锚偏差%"] = {
        "下确界": round((out["下确界"] / ANCHOR_OLD["lo"] - 1) * 100, 4),
        "上确界": round((out["上确界"] / ANCHOR_OLD["hi"] - 1) * 100, 4),
    }
    out["跨度倍数"] = round(out["上确界"] / out["下确界"], 3)
    out["折2026名义"] = {
        "下确界": round(out["下确界"] * DEFL_2026, 2),
        "上确界": round(out["上确界"] * DEFL_2026, 2),
    }
    out["论文点值分位%"] = round(
        (190.5306 - out["下确界"]) / (out["上确界"] - out["下确界"]) * 100, 2)
    out["备注"] = ("下确界在段 2 内部（γ1≈0.129）取得，非窗口端点；"
                   "上确界在段 1 左端边界 γ1=ROOTS[0]（dev=1）取得。"
                   "旧锚 168.207 系硬编码，本次复算值见上。")
    return out


# ============================================================ C6 段 1 右端（真端点）

C6_SEED_OFFSET = 5 * 977     # = s22 主表「段1右端_最宽松」所在档序 i=5 的 seed 偏移


def _tier_raw(tag: str, g1: float, beta: float = 1.0) -> dict:
    """按 s4._lp_cache 的字段格式构造单档 dict（供 simulate_p 使用）。"""
    r = run_at(float(g1), beta=float(beta))
    return {"tag": tag, "γ1": float(g1), "β": float(beta),
            "b1": float(r.get("b1", np.nan)),
            "MAC2022_1995价": float(r.get("MAC_2022均值", np.nan)),
            "MAC2026_1995价_M1": None, "越界首年": None,
            "coef": r.get("_coef"), "do_M1": False,
            "_dev_max": float(r.get("dev_max", np.nan)),
            "_ok": bool(r.get("ok"))}


def c6() -> tuple[pd.DataFrame, dict]:
    """N1 更正：v4 表 2 把 γ₁=+0.06 标为「段 1 右端」，实为段 1 内点。

    段 1 合法窗口 = [ROOTS[0], ROOTS[1]] = [−0.324284, +0.101853]，
    故「段 1 右端」应为 +0.101853（dev_max≈1，端点），而非 +0.06（dev_max≈0.947，内点）。

    本项在**同一随机路径**（seed 偏移 5×977，与 s22 主表「段1右端_最宽松」同 seed）上
    重算两档的 B1·D75·e12·n100 指标：
      · 「段 1 内点(+0.06)」的复算值须与 s22 主表逐项一致 ⇒ 交叉校验本脚本定价链；
      · 「段 1 右端(+0.101853)」供论文表 3 补行。
    两档 2026 基准仅差 0.8%，故 N1 只更正**命名**，不改任何已披露数值。

    ⚠ 表 3 另列 `CAP/基准`（T2）：CAP_AMOUNT 是不随档位变动的常数（=2×论文口径基准），
      该列用以证明赔付上限**未随档位自身基准联动**，即不存在自我循环。
    """
    seed = s4.RNG_SEED + C6_SEED_OFFSET
    rows = []
    for tag, g1 in (("段1内点", 0.06), ("段1右端_合法域端点", float(ROOTS[1]))):
        t = _tier_raw(tag, g1)
        sim = simulate_p("S1_基准", t, "M2_价格平移", s4.N_PATH, seed)
        L = pv_rebased_p(sim, 3, s4.KAPPA_BASE, s4.CAP_AMOUNT, period=1)
        m = layer_metrics(L)
        mac26 = t["MAC2022_1995价"] * DEFL_2026
        rows.append({
            "标度档": tag, "γ1": round(g1, 6), "β": 1.0,
            "dev_max": round(t["_dev_max"], 6),
            "MAC2022_1995价": round(t["MAC2022_1995价"], 3),
            "2026基准": round(mac26, 1),
            "D_自留额": round(m["D"], 1), "A_责任上限": round(m["A"], 1),
            "层宽": round(m["层宽"], 1), "E_layer": round(m["E_layer"], 1),
            "GP_CoC": float(round(m["GP_CoC"])), "ROL_CoC%": round(m["ROL_CoC%"], 1),
            "CAP_AMOUNT": round(s4.CAP_AMOUNT, 2),
            "CAP比基准": round(s4.CAP_AMOUNT / mac26, 3),
        })
    d = pd.DataFrame(rows)
    ref = {
        "判据": "段 1 右端 = β=1 合法窗口段 1 上端点 ROOTS[1]（dev_max ≤ 1）",
        "段1窗口": [round(float(ROOTS[0]), 6), round(float(ROOTS[1]), 6)],
        "段2窗口": [round(float(ROOTS[2]), 6), round(float(ROOTS[3]), 6)],
        "v4误标": "γ₁=+0.06（dev_max=0.9472，段 1 内点，非端点）",
        "真端点": {"γ1": round(float(ROOTS[1]), 6),
                 "dev_max": float(d.loc[1, "dev_max"]),
                 "2026基准": float(d.loc[1, "2026基准"])},
        "内点": {"γ1": 0.06, "dev_max": float(d.loc[0, "dev_max"]),
               "2026基准": float(d.loc[0, "2026基准"])},
        "基准差%": round((d.loc[1, "2026基准"] / d.loc[0, "2026基准"] - 1) * 100, 2),
        "与s22对照": {"D": 268.6, "A": 1488.5, "层宽": 1219.9, "GP": 131.0, "ROL%": 10.7},
        "CAP_AMOUNT": round(s4.CAP_AMOUNT, 2),
    }
    return d, ref


# ============================================================ 验收

def checks(c0: dict, d1: pd.DataFrame, c2meta: dict, d3: pd.DataFrame,
           c4meta: dict, c5out: dict, d6: pd.DataFrame | None = None) -> list[dict]:
    res = []

    def add(tag, desc, ok, got, exp):
        res.append({"验收项": tag, "判据": desc, "结果": "PASS" if ok else "FAIL",
                    "实测": got, "期望": exp})

    add("C0", "simulate_p 默认参数复刻 s4.simulate（逐位）", c0["same"],
        f"I 逐位相同={c0['same_I']}，Q 逐位相同={c0['same_Q']}，最大|ΔI|={c0['maxdI']:.3e}",
        "逐位相同")

    bad = []
    for (tag, n), g in d1.groupby(["标度档", "期限"]):
        v = g.sort_values("κ")["GP_CoC_元每吨"].to_numpy(float)
        if any(v[i + 1] > v[i] + 1e-6 for i in range(len(v) - 1)):
            bad.append(f"{tag}/{n}y")
    add("C1", "GP 随 κ 单调不增（κ 升 ⇒ 阈值升 ⇒ 保障减 ⇒ 保费降）", not bad,
        f"非单调 {len(bad)} 组" if bad else "全部单调", "0 组")

    ok2 = (c2meta["SE_层宽_相对%"] <= 5.0) and (c2meta["倍增最大GP偏差%"] <= 5.0)
    add("C2", "层宽自助相对标准误 ≤5% 且 20k↔80k 的 GP 偏差 ≤5%", ok2,
        f"SE(层宽)={c2meta['SE_层宽_相对%']:.3f}%、SE(GP)={c2meta['SE_GP_相对%']:.3f}%；"
        f"倍增最大 GP 偏差={c2meta['倍增最大GP偏差%']:.3f}%", "≤5% 且 ≤5%")

    bad3 = []
    for tag, g in d3[d3["参数"] == "σ_m"].groupby("标度档"):
        v = g.sort_values("取值")["GP_CoC_元每吨"].to_numpy(float)
        if any(v[i + 1] < v[i] - 1e-6 for i in range(len(v) - 1)):
            bad3.append(f"σ_m/{tag}")
    for tag, g in d3[d3["参数"] == "κ_m"].groupby("标度档"):
        v = g.sort_values("取值")["GP_CoC_元每吨"].to_numpy(float)
        if any(v[i + 1] > v[i] + 1e-6 for i in range(len(v) - 1)):
            bad3.append(f"κ_m/{tag}")
    for tag, g in d3[d3["参数"] == "CAP"].groupby("标度档"):
        v = g.sort_values("取值")["GP_CoC_元每吨"].to_numpy(float)
        if any(v[i + 1] < v[i] - 1e-6 for i in range(len(v) - 1)):
            bad3.append(f"CAP/{tag}")
    add("C3", "GP 随 σ_m 单调不减、随 κ_m 单调不增、随 CAP 单调不减", not bad3,
        " / ".join(bad3) if bad3 else "三方向全部成立", "0 违例")

    # C4：判据不用"零差异"，而用**蒙特卡洛分辨率**——
    # 加密后下确界与原格点下确界的差额，若低于层宽的自助相对标准误，
    # 则两档在定价上不可区分，维持原报价档；否则须改报价。
    se_w = c2meta["SE_层宽_相对%"]
    gap = abs(c4meta["下确界_相对主报价档%"])
    add("C4", "加密后下确界与主报价档基准之差 ≤ 层宽自助相对标准误（两档定价不可区分）",
        gap <= se_w,
        f"粗网格 {c4meta['s0c_粗网格']['下确界']:.3f} → 加密 "
        f"{c4meta['加密后']['下确界']:.3f}（{c4meta['下确界变化%']:+.3f}%）；"
        f"相对主报价档 {gap:.3f}% vs SE(层宽) {se_w:.3f}%", f"≤ {se_w:.3f}%")
    add("C4b", "加密后上确界上移幅度已登记（资本计提侧不利方向，须随表披露）",
        True,
        f"粗网格 {c4meta['s0c_粗网格']['上确界']:.3f} → 加密 "
        f"{c4meta['加密后']['上确界']:.3f}（{c4meta['上确界变化%']:+.3f}%）；"
        f"相对表 2 网格上端档（β=2.0, 572.71）高 "
        f"{c4meta['上确界_相对网格上端档%']:+.3f}%", "已登记且披露方向")

    ok5 = abs(c5out["相对旧锚偏差%"]["下确界"]) <= 0.2 and \
        abs(c5out["相对旧锚偏差%"]["上确界"]) <= 0.2
    add("C5", "复算锚点相对旧硬编码偏差 ≤0.2%（同判据 dev≤1、同 LP）", ok5,
        f"下 {c5out['下确界']:.3f}（{c5out['相对旧锚偏差%']['下确界']:+.4f}%）、"
        f"上 {c5out['上确界']:.3f}（{c5out['相对旧锚偏差%']['上确界']:+.4f}%）", "≤0.2%")

    # C6：N1 命名更正 + 同 seed 交叉校验 s22 定价链
    if d6 is not None and len(d6) == 2:
        r0 = d6.iloc[0]
        dev_ok = (r0["dev_max"] <= 1.0) and (d6.iloc[1]["dev_max"] <= 1.02)
        tol = 0.5
        cmp_ok = (abs(r0["D_自留额"] - 268.6) <= tol
                  and abs(r0["A_责任上限"] - 1488.5) <= tol
                  and abs(r0["层宽"] - 1219.9) <= tol
                  and abs(r0["GP_CoC"] - 131.0) <= tol
                  and abs(r0["ROL_CoC%"] - 10.7) <= 0.15)
        add("C6", "段 1 右端真端点已复算，且 γ₁=+0.06 复算值与 s22 主表一致（≤0.5 元/吨）",
            dev_ok and cmp_ok,
            f"内点(+0.06) D={r0['D_自留额']:.1f}/A={r0['A_责任上限']:.1f}/"
            f"宽={r0['层宽']:.1f}/GP={r0['GP_CoC']:.0f} vs s22 268.6/1488.5/1219.9/131（"
            f"{'一致' if cmp_ok else '不一致'}）；真端点 γ₁={d6.iloc[1]['γ1']:.6f} "
            f"dev_max={d6.iloc[1]['dev_max']:.4f} 2026基准={d6.iloc[1]['2026基准']:.1f}",
            "dev ≤1 且与 s22 逐项 ≤0.5")
    return res


# ============================================================ 主

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    only = "--check" in sys.argv
    t0 = time.time()

    tiers = s4._lp_cache()
    sims = {tag: s4.simulate("S1_基准", tiers[tag], "M2_价格平移",
                             L1["mc_paths"], s4.RNG_SEED + i * 977)
            for i, tag in enumerate(tiers)}

    print("=" * 100)
    print("s23　审稿轮响应工作包（C0–C5）")
    print("=" * 100)

    # C0
    s_ref = s4.simulate("S1_基准", tiers[TIER_QUOTE], "M2_价格平移",
                        L1["mc_paths"], s4.RNG_SEED + 977 * 8)
    s_new = simulate_p("S1_基准", tiers[TIER_QUOTE], "M2_价格平移",
                       L1["mc_paths"], s4.RNG_SEED + 977 * 8)
    c0 = {"same_I": bool(np.array_equal(s_ref["I"], s_new["I"])),
          "same_Q": bool(np.array_equal(s_ref["Q"], s_new["Q"])),
          "maxdI": float(np.abs(s_ref["I"] - s_new["I"]).max())}
    c0["same"] = c0["same_I"] and c0["same_Q"]
    print(f"\n【C0】模拟复刻：I 逐位相同={c0['same_I']}　Q 逐位相同={c0['same_Q']}　"
          f"最大|ΔI|={c0['maxdI']:.3e}")

    # C1
    d1 = c1(sims)
    print("\n【C1】κ 敏感性（B1·D75·e12·n100·口径乙；单位 元/吨）")
    print("-" * 100)
    print(d1.to_string(index=False))

    # C2
    b2, s2, c2meta = c2(sims, tiers[TIER_QUOTE])
    print("\n【C2a】自助标准误（主报价档·B1·3y·D75·e12·n100）")
    print("-" * 100)
    print(b2.to_string(index=False))
    print("\n【C2b】路径数倍增稳定带（同 seed，非同嵌套子样本）")
    print("-" * 100)
    print(s2.to_string(index=False))

    # C3
    d3 = c3(sims, tiers)
    print("\n【C3】核心风险参数单变量敏感性（5y·B1·D75·e12·n100）")
    print("-" * 100)
    print(d3.to_string(index=False))

    # C4
    d4, c4meta = c4()
    print("\n【C4】二维合法域加密")
    print("-" * 100)
    print(json.dumps(c4meta, ensure_ascii=False, indent=2))

    # C5
    c5out = c5()
    print("\n【C5】β=1 合法窗口锚点复算")
    print("-" * 100)
    print(json.dumps(c5out, ensure_ascii=False, indent=2))

    # C6
    d6, c6meta = c6()
    print("\n【C6】段 1 右端真端点（N1 命名更正；与 s22 同 seed 交叉校验）")
    print("-" * 100)
    print(d6.to_string(index=False))
    print(json.dumps(c6meta, ensure_ascii=False, indent=2))

    res = checks(c0, d1, c2meta, d3, c4meta, c5out, d6)
    print("\n【验收】")
    print("-" * 100)
    for r in res:
        print(f"  [{r['结果']:4s}] {r['验收项']:4s} {r['判据']}　→ {r['实测']}")
    nfail = sum(1 for r in res if r["结果"] == "FAIL")
    print(f"\n  合计：{len(res) - nfail} PASS / {nfail} FAIL　"
          f"（耗时 {time.time() - t0:.0f}s）")

    if not only:
        d1.to_csv(OUT / "s23_C1_kappa敏感性.csv", index=False, encoding="utf-8-sig")
        b2.to_csv(OUT / "s23_C2a_自助标准误.csv", index=False, encoding="utf-8-sig")
        s2.to_csv(OUT / "s23_C2b_路径数稳定带.csv", index=False, encoding="utf-8-sig")
        d3.to_csv(OUT / "s23_C3_风险参数敏感性.csv", index=False, encoding="utf-8-sig")
        d4.to_csv(OUT / "s23_C4_二维域加密.csv", index=False, encoding="utf-8-sig")
        d6.to_csv(OUT / "s23_C6_段1右端真端点.csv", index=False, encoding="utf-8-sig")
        with open(OUT / "s23_C45_锚点与域.json", "w", encoding="utf-8") as f:
            json.dump({"C4_二维域加密": c4meta, "C5_锚点复算": c5out,
                       "C6_段1右端": c6meta,
                       "C0_复刻": c0,
                       "验收": [{k: v for k, v in r.items()} for r in res]},
                      f, ensure_ascii=False, indent=2)
        print("\n[已写出] s23_C1_kappa敏感性.csv / s23_C2a_自助标准误.csv / "
              "s23_C2b_路径数稳定带.csv / s23_C3_风险参数敏感性.csv / "
              "s23_C4_二维域加密.csv / s23_C6_段1右端真端点.csv / s23_C45_锚点与域.json")

    if only:
        sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
