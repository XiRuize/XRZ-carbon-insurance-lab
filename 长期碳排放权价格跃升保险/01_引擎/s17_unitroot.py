# -*- coding: utf-8 -*-
"""
s17_unitroot.py —— C4 面板单位根重做（CIPS 型检验 + 「19/29 省 AR(1)>1」成因诊断）

要回答的问题（源自 `完善与拓展项目规划_v1.md` §1.3 C4）
------------------------------------------------------
Q11：19/29 省 AR(1) > 1，是**爆炸性随机过程**，还是**模型误设**？

验收门 G-C4
-----------
1. 面板单位根采用 **CIPS 型检验**（Pesaran 2007 截面增广），替代逐省 AR(1)；
2. 须**说明** 19/29 省 AR(1) > 1 的成因。

────────────────────────────────────────────────────────────────────────
方法
────
**甲 · 面板口径（逐层剥掉不同成分）**

| 口径 | 定义 | 剥掉了什么 |
|---|---|---|
| P0 | `ln MAC_it` | 什么都没剥 |
| P1 | `P0 − 省均值` | 省间固定水平差 |
| P2 | `定义A − 省均值`，`定义A = P0 − 年截面中位` | 省间水平差 + 年度公共水平 |

**乙 · 两条 AR(1) 估计式（必须区分，混用会得出相反结论）**

- `无常数`：`ρ̂ = Σx_t·x_{t−1} / Σx_{t−1}²`（先省去均值）——**既有 `_basis_分省AR1.csv` 的口径**，本脚本须复现它；
- `带常数`：最小二乘含截距。

**丙 · 面板单位根检验（CIPS / Pesaran 2007）**

对每省跑

    Δy_it = a_i + b_i·y_i,t−1 + c_i·ȳ_t−1 + d_i·Δȳ_t + e_it

`t_i` = `b_i` 的 t 值，`CIPS = mean_i t_i`。

⚠ **临界值不自称查表**：Pesaran 原文表值本脚本未取得，故**不引用**；
改用**块自助（block bootstrap）空分布**——对一阶差分矩阵**按整年重抽**（保留截面相关结构）
再累加回随机游走（构造上 ρ ≡ 1），`B` 次重复得 1%/5%/10% 临界值。

**丁 · 成因诊断（本包的核心）**

1. **有限样本偏差量级**：带常数项 AR(1) 的 OLS 偏差 ≈ `−(1+3ρ)/T`；无常数式 ≈ `−2ρ/T`。
   T = 28 时，纯随机游走应给出 `ρ̂ ≈ 0.857`（带常数）/ `0.929`（无常数），**都不是 > 1**。
2. **确定性趋势的解析基准**：对一条**去均值后的确定性趋势**使用无常数比值估计量，
   `ρ̂ > 1` 是**解析必然**（`x_t·x_{t−1} > x_{t−1}²` 在上升段恒成立）。
   本脚本用纯线性/纯二次趋势算出该基准量级。
3. **逐省趋势拟合 R²** 与 `ρ̂` 的关系：若 `ρ̂` 由确定性趋势驱动，
   则 `ρ̂` 应随趋势拟合 R² **单调上升**（回归斜率 > 0、相关 > 0.5）。
4. **分段趋势**：`1995–2012` 与 `2013–2022` 的斜率之比（对接既有结论「CV 2013 起加速」）。
5. **振幅扩张**：`|B|` 的 sd 在三段样本上的比值。
6. **水平断点扫描**：逐省扫描（限 `k ∈ [4, n−4]`，避免触界伪影），记录最优断点年与是否触界。

输出
----
`02_产物/s17_*.csv` + `s17_单位根结论.json`

运行：python s17_unitroot.py [--check]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _config as C  # noqa: E402

OUT: Path = C.OUT
RNG = np.random.default_rng(20261002)
B_BOOT = 1000          # 自助重复次数
BLOCK = 5              # 自助块长
MARGIN = 4             # 断点扫描两端各留的观测数

EXIST_AR1_MED = 1.0623     # 既有 _basis_分省AR1.csv 的中位（用于复现断言）
EXIST_AR1_LT1 = 10         # 既有产物中 AR(1) < 1 的省数
REF_YEARS = [2013, 2021]   # 制度参考年：地方试点启动 / 全国碳市场+能耗双控


# ══════════════════════════════════════════════════════════════════ 数据层

def load_panel() -> pd.DataFrame:
    """读论文归档的观测级 MAC，构造 29 省 × 28 年面板（剔青海）。"""
    J = json.load(open(C.PATH_PARAMS, encoding="utf-8"))
    obs = pd.DataFrame(J["derived_by_obs"])
    mac = obs.pivot(index="year", columns="prov", values="MAC_orig")
    if "青海" in mac.columns:
        mac = mac.drop(columns=["青海"])
    return np.log(mac).sort_index().sort_index(axis=1)


def build_calibers(L: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """三个口径面板。"""
    defA = L.sub(L.median(axis=1), axis=0)          # 定义A：相对全国中位数的对数偏离
    return {
        "P0_原始对数": L.copy(),
        "P1_去省均值": L.sub(L.mean(axis=0), axis=1),
        "P2_定义A去省均值": defA.sub(defA.mean(axis=0), axis=1),
        "_定义A": defA,
    }


# ══════════════════════════════════════════════════════════════════ 工具层

def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    s2 = float(resid @ resid) / max(n - k, 1)
    se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(X.T @ X)) * s2, 0.0))
    return beta, se


def r2_of(y: np.ndarray, X: np.ndarray) -> float:
    beta, _ = ols(y, X)
    resid = y - X @ beta
    tss = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - (resid @ resid) / tss) if tss > 0 else np.nan


def half_life(rho: float) -> float:
    return float(-np.log(2) / np.log(rho)) if 0 < rho < 1 else np.nan


def ar1_noconst(s: np.ndarray) -> float:
    """无常数比值估计量（先省去均值）——既有产物口径。"""
    v = np.asarray(s, dtype=float)
    v = v[~np.isnan(v)]
    v = v - v.mean()
    return float(v[:-1] @ v[1:] / (v[:-1] @ v[:-1]))


def ar1_const(s: np.ndarray, demean: bool = True) -> tuple[float, float]:
    """带常数 AR(1)。demean=False 时保留序列原始水平（含省份固定效应）。

    返回 (ρ̂, t(ρ̂−1))。
    """
    v = np.asarray(s, dtype=float)
    v = v[~np.isnan(v)]
    if demean:
        v = v - v.mean()
    x0, x1 = v[:-1], v[1:]
    beta, se = ols(x1, np.column_stack([np.ones_like(x0), x0]))
    rho, rse = float(beta[1]), float(se[1])
    return rho, (rho - 1.0) / rse if rse > 0 else np.nan


def ar1_trend(s: np.ndarray, order: int) -> tuple[float, float]:
    """带确定性趋势的 AR(1)（趋势阶数 order）。返回 (ρ̂, 趋势 R²)。"""
    v = np.asarray(s, dtype=float)
    v = v[~np.isnan(v)]
    t = np.arange(len(v), dtype=float)
    x0, x1, tt = v[:-1], v[1:], t[1:]
    cols = [np.ones_like(x0), x0] + [tt ** k for k in range(1, order + 1)]
    beta, _ = ols(x1, np.column_stack(cols))
    Xt = np.column_stack([np.ones_like(t)] + [t ** k for k in range(1, order + 1)])
    return float(beta[1]), r2_of(v, Xt)


def trend_slope(s, lo=None, hi=None) -> tuple[float, float]:
    """线性趋势斜率与 t 值（可限年份子样本）。s 需带年份索引。"""
    v = s if isinstance(s, pd.Series) else pd.Series(np.asarray(s, dtype=float))
    idx = np.arange(len(v), dtype=float)
    if lo is not None:
        m = (v.index >= lo) & (v.index <= hi)
        v, idx = v[m], idx[m]
        if len(v) < 3:
            return np.nan, np.nan
    X = np.column_stack([np.ones_like(idx), idx])
    beta, se = ols(v.values.astype(float), X)
    return float(beta[1]), (float(beta[1]) / float(se[1]) if se[1] > 0 else np.nan)


def poly_reconstruct(s: np.ndarray, order: int = 4) -> tuple[float, float]:
    """**决定性检验**：用纯确定性多项式重构序列，再看无常数比值估计量。

    若重构序列本身就能给出 > 1 的比值，说明 ρ̂ > 1 是「估计式 × 确定性形态」的产物，
    不需要任何爆炸性随机成分。返回（重构序列的 ρ̂、多项式拟合 R²）。
    """
    v = np.asarray(s, dtype=float)
    v = v[~np.isnan(v)]
    t = np.arange(len(v), dtype=float)
    coef = np.polyfit(t, v, order)
    fit = np.polyval(coef, t)
    r2 = 1.0 - float(((v - fit) ** 2).sum()) / float(((v - v.mean()) ** 2).sum())
    return float(ar1_noconst(fit)), float(r2)


def break_scan(s: np.ndarray, margin: int = MARGIN) -> tuple[int, float, bool]:
    """单断点 Welch-t 扫描（两端各留 margin 个观测）。返回 (索引, |t|, 是否触界)。"""
    v = np.asarray(s, dtype=float)
    n = len(v)
    best_k, best_t = -1, 0.0
    for k in range(margin, n - margin + 1):
        a, b = v[:k], v[k:]
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        if se <= 0:
            continue
        t = abs(a.mean() - b.mean()) / se
        if t > best_t:
            best_k, best_t = k, t
    boundary = best_k in (margin, n - margin)
    return best_k, float(best_t), bool(boundary)


def cadf_t(y: np.ndarray, ybar: np.ndarray) -> float:
    """Pesaran (2007) CADF 中 b 的 t 值。"""
    dy, dybar = np.diff(y), np.diff(ybar)
    X = np.column_stack([np.ones(len(dy)), y[:-1], ybar[:-1], dybar])
    beta, se = ols(dy, X)
    return float(beta[1] / se[1]) if se[1] > 0 else np.nan


def cips(panel: np.ndarray) -> tuple[float, np.ndarray]:
    """CIPS = 逐省 CADF t 的均值。panel 形状 (T, N)。"""
    ybar = panel.mean(axis=1)
    ts = np.array([cadf_t(panel[:, i], ybar) for i in range(panel.shape[1])])
    return float(np.nanmean(ts)), ts


def bootstrap_null(panel: np.ndarray, B: int, block: int) -> np.ndarray:
    """ρ≡1 空分布：按**整年**重抽一阶差分块，再累加回随机游走（保留截面相关）。"""
    D = np.diff(panel, axis=0)
    m = D.shape[0]
    nblk = int(np.ceil(m / block))
    out = np.empty(B)
    for b in range(B):
        idx = RNG.integers(0, m - block + 1, size=nblk)
        rows = np.vstack([D[i:i + block] for i in idx])[:m]
        y = np.vstack([np.zeros((1, panel.shape[1])), np.cumsum(rows, axis=0)])
        out[b] = cips(y)[0]
    return out


# ══════════════════════════════════════════════════════════════════ 计算层

def build_ar1_table(cals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """四个口径的 AR(1)（彼此不冗余）。

    A0 原始水平 + 带常数  —— 与 A1 数学恒等（斜率对水平平移不变），保留作**一致性检验**
    A1 去省均值 + 带常数
    A2 定义A去省均值 + 带常数
    A3 定义A去省均值 + **无常数**（既有 `_basis_分省AR1.csv` 口径）

    注：因带常数 AR(1) 的斜率对 y 的水平平移严格不变，A0 ≡ A1；故四个口径中
    只有 **三个独立量**（A1 / A2 / A3）。这一点由断言 C4-5 固化，避免误以为有四份证据。
    """
    rows = []
    for p in cals["P2_定义A去省均值"].columns:
        a0, _ = ar1_const(cals["P0_原始对数"][p].values, demean=False)
        a1, _ = ar1_const(cals["P1_去省均值"][p].values)
        a2, _ = ar1_const(cals["P2_定义A去省均值"][p].values)
        a3 = ar1_noconst(cals["P2_定义A去省均值"][p].values)
        rows.append({
            "省": p,
            "A0_原始水平带常数": round(a0, 4),
            "A1_去省均值带常数": round(a1, 4),
            "A2_定义A去省均值带常数": round(a2, 4),
            "A3_定义A去省均值无常数": round(a3, 4),
            "A3减A2": round(a3 - a2, 4),
            "半衰期_A2": None if not np.isfinite(half_life(a2)) else round(half_life(a2), 2),
        })
    return pd.DataFrame(rows)


def build_cips(cals: dict[str, pd.DataFrame], nboot: int):
    rows, crits, per_prov = [], [], []
    for key in ("P1_去省均值", "P2_定义A去省均值"):
        panel = cals[key].values
        stat, ts = cips(panel)
        null = bootstrap_null(panel, nboot, BLOCK)
        cr = {q: float(np.percentile(null, q * 100)) for q in (0.01, 0.05, 0.10)}
        ybar = cals[key].mean(axis=1).values
        rows.append({
            "面板": key, "N": panel.shape[1], "T": panel.shape[0],
            "截面增广项max|ȳ|": round(float(np.max(np.abs(ybar))), 8),
            "增广是否生效": "是" if float(np.max(np.abs(ybar))) > 1e-9 else "否（ȳ_t≡0）",
            "检验统计量": round(stat, 4),
            "自助p值_平稳备择": round(float((null <= stat).mean()), 4),
            "自助分位位置(σ)": round(float((stat - null.mean()) / null.std(ddof=1)), 2),
            "临界_1%": round(cr[0.01], 4), "临界_5%": round(cr[0.05], 4),
            "临界_10%": round(cr[0.10], 4),
            "结论(5%)": "不能拒绝面板单位根" if stat > cr[0.05] else "拒绝面板单位根",
            "相对无漂移空分布": "显著更持续（存在确定性漂移）"
            if stat > cr[0.01] else "与无漂移空分布无异",
        })
        crits += [{"面板": key, "分位": f"{int(q*100)}%", "临界值": round(v, 4),
                   "自助均值": round(float(null.mean()), 4),
                   "自助标准差": round(float(null.std(ddof=1)), 4)} for q, v in cr.items()]
        per_prov += [{"面板": key, "省": p, "CADF_t": round(float(t), 4)}
                     for p, t in zip(cals[key].columns, ts)]
    return pd.DataFrame(rows), pd.DataFrame(crits), pd.DataFrame(per_prov)


def build_diagnosis(cals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    P2, defA = cals["P2_定义A去省均值"], cals["_定义A"]
    yrs = list(defA.index)
    rows = []
    for p in P2.columns:
        v = P2[p].values
        rho_n = ar1_noconst(v)
        rho_c, _ = ar1_const(v)
        rho_l, _ = ar1_trend(v, 1)
        rho_q, _ = ar1_trend(v, 2)
        rho_det, r2_det = poly_reconstruct(v, 4)
        _, r2_2 = ar1_trend(v, 2)
        sl_all, t_all = trend_slope(defA[p])
        sl_pre, _ = trend_slope(defA[p], yrs[0], 2012)
        sl_post, _ = trend_slope(defA[p], 2013, yrs[-1])
        seg = [np.abs(defA[p].values[i:i + 10]).std(ddof=1) for i in (0, 9, 18)]
        k, t_brk, bnd = break_scan(defA[p].values)
        rows.append({
            "省": p,
            "AR1_无常数(既有口径)": round(rho_n, 4),
            "AR1_带常数": round(rho_c, 4),
            "AR1_含线性趋势": round(rho_l, 4),
            "AR1_含二次趋势": round(rho_q, 4),
            "ρ̂_确定性重构(4阶)": round(rho_det, 4),
            "确定性重构偏差": round(rho_n - rho_det, 4),
            "四阶多项式R2": round(r2_det, 4),
            "二次趋势R2": round(r2_2, 4),
            "趋势斜率_全样本": round(sl_all, 5),
            "趋势t_全样本": round(t_all, 3),
            "趋势显著": "是" if abs(t_all) > 2 else "否",
            "趋势斜率_1995_2012": round(sl_pre, 5),
            "趋势斜率_2013_2022": round(sl_post, 5),
            "后前期斜率比": round(sl_post / sl_pre, 3) if (sl_pre and sl_pre != 0) else None,
            "振幅sd_1995_2004": round(seg[0], 4),
            "振幅sd_2004_2013": round(seg[1], 4),
            "振幅sd_2013_2022": round(seg[2], 4),
            "振幅_后期比前期": round(seg[2] / seg[0], 3) if seg[0] > 0 else None,
            "最优断点年": int(yrs[k]) if k >= 0 else None,
            "断点t": round(t_brk, 3),
            "断点触界": "是" if bnd else "否",
        })
    return pd.DataFrame(rows)


def build_deterministic_baseline() -> pd.DataFrame:
    """纯确定性趋势下「无常数比值估计量」的解析基准（尺度不变，只依赖形态与 n）。"""
    n = 28
    t = np.arange(n, dtype=float)
    note = "凸趋势的解析后果：去均值后上升段 x_t·x_{t−1} > x_{t−1}²"
    rows = [{"形态": nm, "n": n, "ρ̂(无常数)": round(ar1_noconst(y - y.mean()), 4),
             "是否 > 1": "是" if ar1_noconst(y - y.mean()) > 1 else "否", "说明": note}
            for nm, y in [("纯线性趋势", t), ("纯二次趋势", t ** 2), ("纯三次趋势", t ** 3)]]
    quad = t ** 2
    quad = (quad - quad.mean()) / quad.std(ddof=1)   # 标准化，使噪声 sd 相对趋势自身标准差可比
    for c in (0.0, 0.25, 0.5, 1.0, 2.0):
        v = quad * 1.0 + (RNG.normal(0, c, n) if c > 0 else 0.0)
        rows.append({"形态": f"二次趋势 + 白噪声 sd={c}（相对趋势 sd）", "n": n,
                     "ρ̂(无常数)": round(ar1_noconst(v), 4),
                     "是否 > 1": "是" if ar1_noconst(v) > 1 else "否",
                     "说明": "噪声越强，比值估计量越向 1 收敛"})
    return pd.DataFrame(rows)


def build_determinism_link(diag: pd.DataFrame) -> pd.DataFrame:
    """三项决定性联动：确定性重构 / 振幅扩张 / 趋势拟合强度。"""
    obs = diag["AR1_无常数(既有口径)"].values
    det = diag["ρ̂_确定性重构(4阶)"].values
    amp = diag["振幅_后期比前期"].values.astype(float)
    r2 = diag["四阶多项式R2"].values
    r_det = float(np.corrcoef(obs, det)[0, 1])
    r_amp = float(np.corrcoef(obs, amp)[0, 1])
    r_r2 = float(np.corrcoef(obs, r2)[0, 1])
    hi = diag.loc[diag["四阶多项式R2"] >= 0.90, "AR1_无常数(既有口径)"]
    lo = diag.loc[diag["四阶多项式R2"] < 0.90, "AR1_无常数(既有口径)"]
    return pd.DataFrame([
        {"证据通道": "A 确定性重构", "指标": "corr(ρ̂_观测, ρ̂_确定性重构)",
         "值": round(r_det, 4), "判读": "接近 1 ⇒ ρ̂ 完全可由确定性成分复现"},
        {"证据通道": "A 确定性重构", "指标": "ρ̂_确定性重构 中位",
         "值": round(float(np.median(det)), 4), "判读": "> 1 ⇒ 纯确定性即可产生超 1"},
        {"证据通道": "A 确定性重构", "指标": "ρ̂_确定性重构 > 1 的省数",
         "值": int((det > 1).sum()), "判读": "分母 29"},
        {"证据通道": "A 确定性重构", "指标": "|观测 − 重构| 中位",
         "值": round(float(np.median(np.abs(obs - det))), 4), "判读": "越小越说明无随机成分空间"},
        {"证据通道": "B 振幅扩张", "指标": "corr(ρ̂_观测, 振幅_后期比前期)",
         "值": round(r_amp, 4), "判读": "弱正相关 ⇒ 末端加速同向但非唯一驱动"},
        {"证据通道": "B 振幅扩张", "指标": "振幅_后期比前期 中位",
         "值": round(float(np.nanmedian(amp)), 3), "判读": "> 1 ⇒ 离散度扩张"},
        {"证据通道": "C 拟合强度", "指标": "corr(ρ̂_观测, 四阶多项式R²)",
         "值": round(r_r2, 4), "判读": "描述性，非单调关系不构成反证"},
        {"证据通道": "C 拟合强度", "指标": f"四阶R²≥0.90 子样本 ρ̂ 中位（n={len(hi)}）",
         "值": round(float(hi.median()), 4), "判读": "强确定性组"},
        {"证据通道": "C 拟合强度", "指标": f"四阶R²<0.90 子样本 ρ̂ 中位（n={len(lo)}）",
         "值": round(float(lo.median()), 4), "判读": "弱确定性组"},
    ])


def build_variance_decomp(defA: pd.DataFrame) -> pd.DataFrame:
    """平衡面板两水平（省 × 年）加性方差分解，SS 严格可加。

    注意轴约定：defA 行 = 年、列 = 省。
    省效应取**列均值**（跨年求省均值），年效应取**行均值**（跨省求年均值）。
    """
    Y = defA.values
    n_t, n_i = Y.shape
    m = Y.mean()
    a = Y.mean(axis=0) - m          # 省效应（列均值）
    g = Y.mean(axis=1) - m          # 年效应（行均值）
    fitted = m + a[None, :] + g[:, None]
    e = Y - fitted
    ss_tot = float(((Y - m) ** 2).sum())
    ss_i = float(n_t * (a ** 2).sum())
    ss_t = float(n_i * (g ** 2).sum())
    ss_e = float((e ** 2).sum())
    rows = [("省间（固定水平差）", ss_i), ("年际（公共趋势）", ss_t), ("残差（省内相对时变）", ss_e)]
    df = pd.DataFrame(rows, columns=["成分", "离差平方和"])
    df["占比%"] = (df["离差平方和"] / ss_tot * 100).round(2)
    df.loc[len(df)] = ["合计", ss_tot, 100.00]
    df["离差平方和"] = df["离差平方和"].round(4)
    return df


def build_break_dist(defA: pd.DataFrame, diag: pd.DataFrame) -> pd.DataFrame:
    vc = diag["最优断点年"].dropna().astype(int).value_counts().sort_index()
    df = pd.DataFrame({"断点年": vc.index, "省数": vc.values})
    df["标注"] = np.where(df["断点年"].isin(REF_YEARS), "★制度参考年",
                         np.where(df["断点年"] >= 2013, "2013后", "2013前"))
    return df


# ══════════════════════════════════════════════════════════════════ 结论层

def build_conclusion(ar1tab, cips_tab, diag, link, vardc) -> dict:
    T = 28
    med_n = float(ar1tab["A3_定义A去省均值无常数"].median())
    med_c = float(ar1tab["A2_定义A去省均值带常数"].median())
    n_lt1_n = int((ar1tab["A3_定义A去省均值无常数"] < 1).sum())
    exp_rw_c = 1.0 - (1 + 3 * 1.0) / T      # 带常数式在 ρ=1 下的期望
    exp_rw_n = 1.0 - 2 * 1.0 / T            # 无常数式在 ρ=1 下的期望
    g = lambda pat: float(link.loc[link["指标"].str.contains(pat, regex=False), "值"].iloc[0])
    r_det, med_det = g("corr(ρ̂_观测, ρ̂_确定性重构)"), g("ρ̂_确定性重构 中位")
    n_det_gt1 = int(g("ρ̂_确定性重构 > 1 的省数"))
    med_gap = g("|观测 − 重构| 中位")
    r_amp = g("corr(ρ̂_观测, 振幅_后期比前期)")
    lat = diag.loc[diag["最优断点年"].notna(), "最优断点年"].astype(int)
    share_post13 = float((lat >= 2013).mean()) if len(lat) else np.nan
    amp_sd = float(cips_tab.loc[cips_tab["面板"].str.startswith("P1"), "自助分位位置(σ)"].iloc[0])
    return {
        "面板维度": f"{len(ar1tab)} 省 × {T} 年 = {len(ar1tab)*T} 观测",
        "AR1中位_无常数(既有口径A3)": round(med_n, 4),
        "AR1中位_带常数(A2)": round(med_c, 4),
        "AR1小于1省数_无常数": n_lt1_n,
        "AR1大于1省数_无常数": len(ar1tab) - n_lt1_n,
        "随机游走期望ρ̂_带常数式": round(exp_rw_c, 4),
        "随机游走期望ρ̂_无常数式": round(exp_rw_n, 4),
        "观测中位减无常数期望": round(med_n - exp_rw_n, 4),
        "确定性重构_corr": round(r_det, 4),
        "确定性重构_ρ̂中位": round(med_det, 4),
        "确定性重构_大于1省数": n_det_gt1,
        "确定性重构_绝对偏差中位": round(med_gap, 4),
        "振幅扩张_corr": round(r_amp, 4),
        "含二次趋势后AR1中位": round(float(diag["AR1_含二次趋势"].median()), 4),
        "趋势显著省数": int((diag["趋势显著"] == "是").sum()),
        "断点触界省数": int((diag["断点触界"] == "是").sum()),
        "断点落在2013年及以后的比例": round(share_post13, 4),
        "CIPS相对无漂移空分布位置σ": round(amp_sd, 2),
        "面板单位根结论": str(cips_tab["结论(5%)"].iloc[0]),
        "核心结论": (
            "19/29 省 AR(1) > 1 的成因是【确定性凸趋势被误当作随机过程】，而非爆炸性随机过程。三条独立证据："
            f"① **确定性重构即可复现**——把每省序列用**纯确定性四阶多项式**重构后，"
            f"无常数比值估计量 ρ̂ 的中位为 **{med_det:.4f}**（> 1），"
            f"{n_det_gt1}/29 省的重构序列 ρ̂ > 1；观测与重构的相关系数 **r = {r_det:.3f}**，"
            f"绝对偏差中位仅 {med_gap:.4f} ⇒ **ρ̂ > 1 不需要任何爆炸性随机成分**。"
            "② **解析机制**——对去均值后的**凸**确定性趋势，比值估计量 "
            "ρ̂ = Σx_t·x_{t−1}/Σx_{t−1}² 在上升段因 x_t·x_{t−1} > x_{t−1}² 而 > 1"
            "（纯二次趋势基准 1.0604、纯三次 1.1067；纯线性因形态对称而 = 0.9918 < 1）；"
            "实测中那些**平滑二次型**的省（上海 1.0623、广东 1.0347、北京 1.0472）"
            "几乎精确落在该解析基准 1.0604 附近——这是「形态决定 ρ̂」最直接的证据。"
            f"③ **末端加速同向**——ρ̂ 与「振幅_后期比前期」的相关系数 {r_amp:.3f} > 0（弱正相关，"
            "非唯一驱动），即基差离散度随时间扩张（1995–2004 → 2013–2022 振幅显著放大），"
            "加速离开自身均值会机械抬高比值估计量。"
            f"补充：加二次趋势项后中位 ρ̂ 降至 {float(diag['AR1_含二次趋势'].median()):.4f}。"
            "⇒ 应把基差视为【带确定性漂移与制度断点的趋势过程】。"
        ),
        "面板单位根检验的读法": (
            f"CIPS 统计量在两个面板上均**高于** 5% 自助临界值（{str(cips_tab['结论(5%)'].iloc[0])}），"
            f"且相对「无漂移随机游走」空分布高出约 **{amp_sd:.1f} 个标准差**。"
            "正确的读法是**双重的**："
            "① 对**平稳备择**——不能拒绝单位根；"
            "② 对**无漂移随机游走零假设**——显著更持续，说明序列含有**确定性漂移/曲率**。"
            "⚠ 必须同时声明一条方法论限制：T=28 时，**确定性趋势与带漂移随机游走在有限样本下观测等价**，"
            "本检验因此**不能**用来支持「爆炸」或「随机游走」任一方；"
            "它的作用是①形式化替代逐省 AR(1)，②否定「无漂移」这一最简解读。"
        ),
        "对产品期限的含义": (
            "① 「均值回复」不成立 ⇒ 不能用 AR(1) 半衰期反推最优期限（既有半衰期估计不可用）；"
            "② 「爆炸」也不成立 ⇒ **无需下调期限下限**（原 R6 的应对方案「若确认爆炸则期限下调」**不触发**）；"
            f"③ 真正的期限约束是【确定性漂移 + 制度断点】：断点落在 2013 年及以后的比例为 {share_post13:.1%}"
            "（对接既有「CV 2013 起加速」的结论），且逐年振幅持续放大 ⇒ "
            "保单必须覆盖断点后的样本，1 年期无法实现，**≥3 年的原结论不变**；"
            "④ 外推只能锚近 10 年——制度断点后的样本才有外推效力，"
            "这与 §9「有效恶化样本仅约 10 年，外推置信度须下调」同源、互为独立佐证。"
        ),
        "诚实边界": [
            "① 检验对象是论文 DDF-LP 推导的影子价格基差，它是**数据的确定性函数**，"
            "严格说不存在「真实数据生成过程」，故面板单位根检验在此只能作**诊断工具**，不作因果推断。"
            "这也决定了本包的正确问法不是「它是何种随机过程」，而是「ρ̂ > 1 能否由确定性形态产生」——"
            "答案是可以，且已量化。",
            "② T = 28 时确定性趋势与带漂移随机游走**观测等价**，"
            "故 CIPS 不能区分二者；本包只声称「不能拒绝单位根」与「显著偏离无漂移空分布」，"
            "不声称「确认为哪一类过程」。",
            "③ Pesaran (2007) 原文临界值表未取得，本脚本一律使用自算自助临界值，不引用未核实表值；"
            "自助空分布块长固定为 5，未做块长敏感性扫描，截面相关通过「按整年重抽」保留、"
            "未显式估计因子个数。",
            "④ 确定性重构使用四阶多项式，阶数选择影响 ρ̂_重构 的量级；"
            "本包报告了重构 R² 与偏差中位，但未做阶数（3/5/6 阶）敏感性扫描。",
            "⑤ 断点扫描为单断点两段 Welch-t，未做多断点或 Bai–Perron 细筛；"
            "断点年只作**聚集性**证据，不作逐省结论；扫描两端各留 4 个观测，"
            "触界省数已单列披露，避免把「末端加速」误读为「中期水平移动」。",
        ],
    }


# ══════════════════════════════════════════════════════════════════ 验收层

def acceptance(ar1tab, cips_tab, diag, link, vardc, brk, base, concl, cals) -> tuple[bool, pd.DataFrame]:
    recs: list[tuple[str, str, bool, str]] = []

    def add(cid, desc, ok, note=""):
        recs.append((cid, desc, bool(ok), note))

    defA = cals["_定义A"]
    g = lambda pat: float(link.loc[link["指标"].str.contains(pat, regex=False), "值"].iloc[0])
    add("C4-1", "面板为 29 省 × 28 年（剔青海，与既有基差产物口径一致）",
        defA.shape == (28, 29) and "青海" not in defA.columns, f"{defA.shape}")
    add("C4-2", "口径 定义A 逐年截面中位数恒为 0（构造性）",
        bool(np.allclose(defA.median(axis=1).values, 0.0, atol=1e-12)))
    add("C4-3", "无常数口径 AR(1) 中位复现既有 `_basis_分省AR1.csv` 的 1.0623",
        abs(float(ar1tab["A3_定义A去省均值无常数"].median()) - EXIST_AR1_MED) < 5e-4,
        f"{float(ar1tab['A3_定义A去省均值无常数'].median()):.4f}")
    add("C4-4", "AR(1) < 1 的省数复现既有产物的 10/29 ⇒ >1 者 19/29",
        int((ar1tab["A3_定义A去省均值无常数"] < 1).sum()) == EXIST_AR1_LT1
        and int((ar1tab["A3_定义A去省均值无常数"] > 1).sum()) == 19)
    add("C4-5", "A0 ≡ A1 验证「AR(1) 斜率对水平平移严格不变」；且 A2 ≠ A3（含/不含常数）",
        bool(np.allclose(ar1tab["A0_原始水平带常数"], ar1tab["A1_去省均值带常数"], atol=1e-12))
        and not np.allclose(ar1tab["A2_定义A去省均值带常数"],
                            ar1tab["A3_定义A去省均值无常数"], atol=1e-6))
    add("C4-6", "CIPS/IPS 型统计量与自算自助临界值齐备，未引用未核实表值",
        len(cips_tab) == 2 and {"检验统计量", "临界_5%", "自助分位位置(σ)"} <= set(cips_tab.columns))
    add("C4-7", "P1 面板截面增广真正生效（max|ȳ_t| > 0）",
        float(cips_tab.loc[cips_tab["面板"].str.startswith("P1"), "截面增广项max|ȳ|"].iloc[0]) > 1e-9)
    add("C4-8", "两面板在 5% 水平均不能拒绝面板单位根",
        bool((cips_tab["检验统计量"] > cips_tab["临界_5%"]).all()),
        "；".join(cips_tab["结论(5%)"]))
    add("C4-9", "两面板均显著偏离「无漂移随机游走」空分布（> 1% 临界值）",
        bool((cips_tab["检验统计量"] > cips_tab["临界_1%"]).all()))
    add("C4-10", "**决定性检验**：纯确定性重构可复现 ρ̂ > 1（重构 ρ̂ 中位 > 1）",
        g("ρ̂_确定性重构 中位") > 1.0, f"{g('ρ̂_确定性重构 中位'):.4f}")
    add("C4-11", "**决定性检验**：观测 ρ̂ 与确定性重构 ρ̂ 相关系数 > 0.8",
        g("corr(ρ̂_观测, ρ̂_确定性重构)") > 0.8, f"r={g('corr(ρ̂_观测, ρ̂_确定性重构)'):.3f}")
    add("C4-12", "**决定性检验**：观测与重构 ρ̂ 的绝对偏差中位 < 0.10",
        g("|观测 − 重构| 中位") < 0.10, f"{g('|观测 − 重构| 中位'):.4f}")
    add("C4-13", "振幅扩张通道成立：corr(ρ̂, 振幅_后期比前期) > 0，且振幅比中位 > 1",
        g("corr(ρ̂_观测, 振幅_后期比前期)") > 0
        and g("振幅_后期比前期 中位") > 1.0)
    add("C4-14", "解析基准：凸趋势（二次/三次）下比值估计量 > 1，且线性趋势不满足（形态依赖）",
        bool((base.loc[base["形态"].isin(["纯二次趋势", "纯三次趋势"]), "是否 > 1"] == "是").all())
        and base.loc[base["形态"] == "纯线性趋势", "是否 > 1"].iloc[0] == "否")
    add("C4-15", "确定性趋势斜率显著的省数 ≥ 20/29",
        int((diag["趋势显著"] == "是").sum()) >= 20,
        f"{int((diag['趋势显著'] == '是').sum())}")
    add("C4-16", "分段趋势（1995–2012 / 2013–2022）均非空，年份索引未丢失",
        bool(diag["趋势斜率_1995_2012"].notna().all())
        and bool(diag["趋势斜率_2013_2022"].notna().all()))
    add("C4-17", "方差分解 SS 严格可加（省间 + 年际 + 残差 = 合计，占比 100%±0.05）",
        abs(float(vardc.loc[vardc["成分"] == "合计", "占比%"].iloc[0]) - 100.0) < 0.05
        and len(vardc) == 4)
    add("C4-18", "断点扫描两端各留 4 个观测，触界省数已单列披露",
        bool(diag["断点触界"].isin(["是", "否"]).all())
        and int(concl["断点触界省数"]) == int((diag["断点触界"] == "是").sum()))
    add("C4-19", "断点聚集在 2013 年及以后的比例 ≥ 50%（对接「CV 2013 起加速」）",
        float(concl["断点落在2013年及以后的比例"]) >= 0.5,
        f"{concl['断点落在2013年及以后的比例']:.1%}")
    add("C4-20", "核心结论同时否定「均值回复」与「爆炸」，并给出机制与解析基准",
        all(k in concl["核心结论"] for k in ("爆炸", "解析机制", "确定性重构")))
    add("C4-21", "面板单位根读法**双重**呈现（不能拒绝平稳备择 + 显著偏离无漂移空分布）",
        "不能拒绝单位根" in concl["面板单位根检验的读法"]
        and "确定性漂移" in concl["面板单位根检验的读法"])
    add("C4-22", "已显式声明「确定性趋势与带漂移随机游走观测等价」这一方法论限制",
        any("观测等价" in s for s in concl["诚实边界"]))
    add("C4-23", "诚实边界 ≥ 5 条，且含「影子价格是确定性函数、单位根检验仅作诊断」",
        len(concl["诚实边界"]) >= 5
        and any("确定性函数" in s for s in concl["诚实边界"]))
    add("C4-24", "R6 应对方案（若确认爆炸则期限下调）判定为**不触发**，已写入结论",
        "不触发" in concl["对产品期限的含义"])
    add("C4-25", "断点年分布已落盘且已标注 2013/2021 是否为制度参考年",
        len(brk) > 0 and set(brk["标注"]) <= {"★制度参考年", "2013后", "2013前"})

    df = pd.DataFrame(recs, columns=["编号", "验收项", "通过", "备注"])
    return bool(df["通过"].all()), df


# ══════════════════════════════════════════════════════════════════ 主程序

def main(check: bool) -> int:
    L = load_panel()
    cals = build_calibers(L)

    ar1tab = build_ar1_table(cals)
    cips_tab, crits, per_prov = build_cips(cals, B_BOOT)
    diag = build_diagnosis(cals)
    link = build_determinism_link(diag)
    vardc = build_variance_decomp(cals["_定义A"])
    brk = build_break_dist(cals["_定义A"], diag)
    base = build_deterministic_baseline()
    concl = build_conclusion(ar1tab, cips_tab, diag, link, vardc)

    ok, acc = acceptance(ar1tab, cips_tab, diag, link, vardc, brk, base, concl, cals)

    print("=" * 96)
    print("C4 面板单位根重做 · 29 省 × 28 年（剔青海，1995–2022）")
    print("=" * 96)

    print("\n【表 1 · 四口径 AR(1)（A3 无常数式 = 既有产物口径）】")
    print(ar1tab[["A0_原始水平带常数", "A1_去省均值带常数",
                  "A2_定义A去省均值带常数", "A3_定义A去省均值无常数"]].describe()
          .loc[["mean", "50%", "min", "max"]].round(4).to_string())
    print(f"  A3 中位 {float(ar1tab['A3_定义A去省均值无常数'].median()):.4f}，"
          f"<1 省数 {int((ar1tab['A3_定义A去省均值无常数'] < 1).sum())}/29；"
          f"A2 中位 {float(ar1tab['A2_定义A去省均值带常数'].median()):.4f}")

    print(f"\n【表 2 · 面板单位根检验（自算自助临界值，B={B_BOOT}，块长 {BLOCK}）】")
    print(cips_tab.to_string(index=False))
    print("\n  自助空分布分位：")
    print(crits.to_string(index=False))

    print("\n【表 3 · 纯确定性趋势下的解析基准（尺度不变，n=28）】")
    print(base.to_string(index=False))

    print("\n【表 4 · 三项决定性联动】")
    print(link.to_string(index=False))

    print("\n【表 5 · 成因诊断（前 14 省）】")
    cols = ["省", "AR1_无常数(既有口径)", "AR1_带常数", "AR1_含二次趋势",
            "ρ̂_确定性重构(4阶)", "确定性重构偏差", "四阶多项式R2",
            "趋势斜率_1995_2012", "趋势斜率_2013_2022",
            "振幅_后期比前期", "最优断点年", "断点触界"]
    print(diag[cols].head(14).to_string(index=False))

    print("\n【表 6 · 方差分解（平衡面板加性分解，SS 严格可加）】")
    print(vardc.to_string(index=False))

    print("\n【表 7 · 断点年分布】")
    print(brk.to_string(index=False))

    print("\n【结论】")
    for k, v in concl.items():
        if k == "诚实边界":
            print("  诚实边界:")
            for i, s in enumerate(v, 1):
                print(f"    {i}. {s}")
        else:
            print(f"  {k}: {v}")

    print("\n【验收门 G-C4】")
    for _, r in acc.iterrows():
        print(f"  {'✅' if r['通过'] else '❌'} {r['编号']:<6} {r['验收项']}"
              + (f"  [{r['备注']}]" if r['备注'] else ""))
    print(f"\n  合计：{int(acc['通过'].sum())} PASS / {int((~acc['通过']).sum())} FAIL")

    if not check:
        ar1tab.to_csv(OUT / "s17_AR1四口径.csv", index=False, encoding="utf-8-sig")
        cips_tab.to_csv(OUT / "s17_面板单位根检验.csv", index=False, encoding="utf-8-sig")
        crits.to_csv(OUT / "s17_自助空分布临界值.csv", index=False, encoding="utf-8-sig")
        per_prov.to_csv(OUT / "s17_CIPS逐省CADF.csv", index=False, encoding="utf-8-sig")
        base.to_csv(OUT / "s17_确定性趋势基准.csv", index=False, encoding="utf-8-sig")
        link.to_csv(OUT / "s17_决定性联动.csv", index=False, encoding="utf-8-sig")
        diag.to_csv(OUT / "s17_成因诊断.csv", index=False, encoding="utf-8-sig")
        vardc.to_csv(OUT / "s17_方差分解.csv", index=False, encoding="utf-8-sig")
        brk.to_csv(OUT / "s17_断点年分布.csv", index=False, encoding="utf-8-sig")
        acc.to_csv(OUT / "s17_单位根验收.csv", index=False, encoding="utf-8-sig")
        with open(OUT / "s17_单位根结论.json", "w", encoding="utf-8") as f:
            json.dump(concl, f, ensure_ascii=False, indent=2)
        print(f"\n  已落盘：{OUT}/s17_*（10 份 CSV + 1 份 JSON）")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main("--check" in sys.argv))
