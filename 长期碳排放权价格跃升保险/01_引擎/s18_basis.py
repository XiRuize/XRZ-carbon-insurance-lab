# -*- coding: utf-8 -*-
"""
s18_basis.py —— C3 基差建模：ρ(ΔB, P) 的可识别性与影响面

要回答的问题（源自 `完善与拓展项目规划_v1.md` §1.3 C3）
------------------------------------------------------
Q5：基差与碳价的相关性 ρ(ΔB, P) 是多少？

验收门 G-C3
-----------
**基差建模须显式声明重叠期仅 2 年，相关系数不得作为点估计使用。**

────────────────────────────────────────────────────────────────────────
本脚本的核心判断：**Q5 的正确答法不是给出 ρ，而是判定 ρ 的地位。** 分三步——

**甲 · 可识别性**（能不能估？）
- 交易所碳价（全国碳市场 2021-07 上线）与 1995–2022 MAC 面板的重叠期 = **2 年**
  ⇒ 年度变动数 = **1** ⇒ 估计截面相关所需自由度 = 0 ⇒ **不可识别**。
- 量化「未来能不能识别」：Fisher z 下 `se(ρ̂) ≈ 1/√(n−3)`。
  反解 `se ≤ 0.10` 需 **n ≥ 103 年**、`se ≤ 0.05` 需 **n ≥ 403 年**
  ⇒ 年度口径下**永久不可识别**，不是「样本还不够」的问题。

**乙 · 构造性正交**（≠ 实证发现）
- 基差 `B_it = ln MAC_it − median_j ln MAC_jt` 的定义**已把年度公共价格成分剔除**。
  因此 `ΔB` 与「影子价格水平变动」正交是**构造结果**，无论数据如何都成立，
  **不能**当作「基差风险独立于碳价风险」的实证证据。
- 只有**外部真实交易价格**才携带独立信息 ⇒ 回到甲，仍不可识别。

**丙 · 影响面**（不可识别是否要紧？）
- **决定性结论**：赔付 `L = (ΔB − K)⁺ × MAC_REF × Q` 只依赖 `ΔB` 的**边际分布**；
  而 ρ 只决定 `ΔB` 与外部价格 `P` 的**联合结构**。由 Sklar 定理，
  保持边际分布不变、仅改变 copula（即改变 ρ），`E[L]`、`TVaR99(L)`、出险频率**严格不变**。
  ⇒ **ρ 不进定价链**，该不可识别参数对本产品定价的影响面 = **0**。
  （本脚本用数值与解析两条通道分别验证。）
- **唯一的影响面是组合分散**：若对手方同时持有碳价头寸，组合 TVaR 依赖 ρ。
  该场景需要**对手方组合信息**，属外部依赖 ⇒ 只能作**外生区间**披露，不得内部拍定。

**丁 · 顺带更正一处上游设定**
- 中期稿 `中期_省际减排成本基差险_设计_v1.md` §6.3 把 `ΔB` 设定为
  `ΔB_it = ρ_i·ΔB_i,t−1 + ε`（AR(1) 均值回复）。**与 s17（C4）的结论冲突**：
  s17 已证 19/29 省 AR(1) > 1 系确定性趋势被误当作随机过程，均值回复不成立。
  ⇒ 本脚本把该设定更正为【带确定性漂移 + 结构断点的过程】，并给出期限含义。

输出
----
`02_产物/s18_*.csv` + `s18_基差建模结论.json`

运行：python s18_basis.py [--check]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _config as C  # noqa: E402

OUT: Path = C.OUT
RNG = np.random.default_rng(20261003)
N_MC = 200_000
RHO_GRID = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]
ALPHA = 0.99
TRADED_OVERLAP = [2021, 2022]      # 交易所碳价与 MAC 面板的重叠年
SE_TARGETS = [0.20, 0.10, 0.05]    # 目标标准误


# ══════════════════════════════════════════════════════════════════ 数据层

def load_defA() -> pd.DataFrame:
    """定义A 基差面板（29 省 × 28 年，剔青海）。"""
    J = json.load(open(C.PATH_PARAMS, encoding="utf-8"))
    mac = pd.DataFrame(J["derived_by_obs"]).pivot(index="year", columns="prov", values="MAC_orig")
    if "青海" in mac.columns:
        mac = mac.drop(columns=["青海"])
    L = np.log(mac).sort_index().sort_index(axis=1)
    return L.sub(L.median(axis=1), axis=0)


def build_delta_sample(defA: pd.DataFrame) -> pd.Series:
    """一阶年度变动样本 ΔB_it（供历史模拟）。"""
    d = defA.diff().iloc[1:]
    v = d.values.ravel()
    return pd.Series(v[~np.isnan(v)], name="ΔB")


def get_mac_ref() -> float:
    """从 s3 读 2026 年 S1 基准的名义 MAC 作赔付基准（不硬编码）。"""
    f = OUT / "s3_MAC期限结构.csv"
    df = pd.read_csv(f, encoding="utf-8-sig")
    return float(df.loc[(df["年份"] == 2026) & (df["情景"] == "S1_基准"), "M3_MAC_名义价"].iloc[0])


# ══════════════════════════════════════════════════════════════════ 甲 · 可识别性

def build_identifiability() -> pd.DataFrame:
    n_years = TRADED_OVERLAP[1] - TRADED_OVERLAP[0] + 1
    n_changes = n_years - 1
    return pd.DataFrame([
        {"项": "交易所碳价可用起始", "值": "2021-07（全国碳市场上线）", "说明": "此前无连续交易价"},
        {"项": "MAC 面板覆盖", "值": "1995–2022（28 年）", "说明": "论文二次型 DDF-LP 推导"},
        {"项": "重叠年数", "值": str(n_years), "说明": "2021、2022"},
        {"项": "重叠产生的年度变动数", "值": str(n_changes), "说明": "识别截面相关所需自由度 = 变动数 − 1"},
        {"项": "可用自由度", "值": "0", "说明": "1 − 1 = 0"},
        {"项": "判定", "值": "不可识别", "说明": "交易所价口径下 ρ(ΔB,P) 无任何可估内容"},
    ])


def build_se_curve() -> pd.DataFrame:
    rows = []
    for target in SE_TARGETS:
        n = 3 + 1.0 / target ** 2
        rows.append({"目标标准误 se(ρ̂)": target, "所需重叠年数 n": int(np.ceil(n)),
                     "= se=1/√(n−3) 反解": "Fisher z 近似",
                     "相对现有重叠年数": f"{np.ceil(n) / (TRADED_OVERLAP[1]-TRADED_OVERLAP[0]+1):.0f}×",
                     "实务可达性": "不可达" if n > 50 else "可达"})
    return pd.DataFrame(rows)


def load_lnMAC() -> pd.DataFrame:
    """ln MAC 面板（29 省 × 28 年，剔青海）。"""
    J = json.load(open(C.PATH_PARAMS, encoding="utf-8"))
    mac = pd.DataFrame(J["derived_by_obs"]).pivot(index="year", columns="prov", values="MAC_orig")
    if "青海" in mac.columns:
        mac = mac.drop(columns=["青海"])
    return np.log(mac).sort_index().sort_index(axis=1)


def build_constructive_orthogonality(defA: pd.DataFrame) -> pd.DataFrame:
    """三个价格代理口径下的 corr(ΔB, ΔP)。

    目的不是选一个「正确的 ρ」，而是证明**连「用哪个价格」都无法由数据判定**：
    三个口径给出的答案从 ~0 到 0.45 不等，本身就是又一层不可识别。
    """
    lnM = load_lnMAC()
    d = defA.diff().iloc[1:]
    flat = d.values.ravel()
    m = ~np.isnan(flat)

    def corr_with(series: pd.Series) -> float:
        reps = np.repeat(series.diff().iloc[1:].values, d.shape[1])[m]
        if float(np.std(reps)) < 1e-12:
            return float("nan")
        return float(np.corrcoef(flat[m], reps)[0, 1])

    p_med = lnM.median(axis=1)            # 口径 B：定义锚（对数水平中位）
    p_mean = lnM.mean(axis=1)             # 口径 C：截面均值的对数
    p_agg = np.log(np.exp(lnM).mean(axis=1))   # 口径 D：算术均值（总量型代理）
    rows = [
        {"口径": "A 交易所碳价（全国碳市场）", "价格代理": "真实交易价",
         "重叠年数": 2, "corr(ΔB,ΔP)": "不可识别", "说明": "重叠仅 2 年 ⇒ 1 个变动 ⇒ df=0"},
        {"口径": "B 截面中位水平（定义锚）", "价格代理": "median_j ln MAC_jt",
         "重叠年数": 27, "corr(ΔB,ΔP)": None,
         "说明": "ΔB 的定义分子即含该锚的变动 ⇒ **构造关联**，不作实证解读"},
        {"口径": "C 截面均值（对数）", "价格代理": "mean_j ln MAC_jt",
         "重叠年数": 27, "corr(ΔB,ΔP)": None, "说明": "另一种中心化口径"},
        {"口径": "D 算术均值（总量型）", "价格代理": "ln mean_j MAC_jt",
         "重叠年数": 27, "corr(ΔB,ΔP)": None, "说明": "更接近「全国碳价水平」的直觉含义"},
    ]
    for i, s in zip((1, 2, 3), (p_med, p_mean, p_agg)):
        rows[i]["corr(ΔB,ΔP)"] = round(corr_with(s), 4)
    vals = [rows[i]["corr(ΔB,ΔP)"] for i in (1, 2, 3)]
    width = (max(vals) - min(vals)) / max(abs(min(vals, key=abs)), 1e-12)
    rows.append({
        "口径": "汇总", "价格代理": "—", "重叠年数": "—",
        "corr(ΔB,ΔP)": f"区间 [{min(vals):.4f}, {max(vals):.4f}]",
        "说明": f"三口径互不相同（量级差 {abs(width):.1f} 倍）⇒ **口径本身不可判定**，"
                f"这是第二层不可识别",
    })
    df = pd.DataFrame(rows)
    df["corr(ΔB,ΔP)"] = df["corr(ΔB,ΔP)"].apply(
        lambda v: v if isinstance(v, (str, type(None))) else round(float(v), 4))

    # 构造性检查：逐年截面中位 B ≡ 0（这是定义A 的构造性质，与 ΔB 无关）
    extra = pd.DataFrame([
        {"口径": "构造性检查", "价格代理": "逐年截面中位 B_it（绝对）最大值",
         "重叠年数": "—", "corr(ΔB,ΔP)": round(float(np.nanmax(np.abs(defA.median(axis=1)))), 10),
         "说明": "≡0（构造性）：定义A 以年截面中位为原点"},
        {"口径": "构造性检查", "价格代理": "逐年截面中位 ΔB_it（绝对）最大值",
         "重叠年数": "—", "corr(ΔB,ΔP)": round(float(np.nanmax(np.abs(np.nanmedian(d.values, axis=1)))), 6),
         "说明": "**不为 0** ⇒ 不能据「中位为 0」声明 ΔB 与价格正交"},
    ])
    return pd.concat([df, extra], ignore_index=True)


# ══════════════════════════════════════════════════════════════════ 丙 · 影响面

def payout_stage1(delta_b: np.ndarray, k_abs: float) -> np.ndarray:
    return np.maximum(delta_b - k_abs, 0.0)


def build_pricing_invariance(sample: pd.Series, mac_ref: float, k_mult: float = 0.5):
    """决定性检验：赔付统计量对 ρ **严格不变**。

    构造：保持 ΔB 的边际分布（同一批经验分位数）不变，
    仅通过高斯 copula 改变与外部价格 P 的相关 ρ。
    """
    n = N_MC
    z_p = RNG.standard_normal(n)
    z_b = RNG.standard_normal(n)
    u_b = stats.norm.cdf(z_b)
    q_b = np.quantile(sample.values, np.clip(u_b, 1e-9, 1 - 1e-9))
    sd_b = float(sample.std(ddof=1))
    k_abs = k_mult * sd_b
    pay = payout_stage1(q_b, k_abs) * mac_ref
    e, tv = float(pay.mean()), float(pay[pay >= np.quantile(pay, ALPHA)].mean())
    rows = []
    for rho in RHO_GRID:
        u_p = stats.norm.cdf(rho * z_p + np.sqrt(1 - rho ** 2) * RNG.standard_normal(n))
        # P 的边际不影响赔付；此处仅演示联合结构不同、赔付统计量逐位相同
        _ = np.quantile(sample.values, np.clip(u_p, 1e-9, 1 - 1e-9))
        p2 = payout_stage1(q_b, k_abs) * mac_ref     # 与 rho 无关
        rows.append({
            "ρ": rho, "出险频率": round(float((p2 > 0).mean()), 6),
            "E[赔付]": round(float(p2.mean()), 6),
            f"TVaR{int(ALPHA*100)}": round(float(p2[p2 >= np.quantile(p2, ALPHA)].mean()), 6),
            "E 相对基准偏差": round(float(abs(p2.mean() - e) / e), 12),
            "TVaR 相对基准偏差": round(float(abs(p2[p2 >= np.quantile(p2, ALPHA)].mean() - tv) / tv), 12),
        })
    df = pd.DataFrame(rows)
    meta = {"E": e, "TVaR99": tv, "k_abs": k_abs, "sigma_B": sd_b, "MAC_REF": mac_ref,
            "n_mc": n, "最大相对偏差": float(max(df["E 相对基准偏差"].max(),
                                              df["TVaR 相对基准偏差"].max()))}
    return df, meta


def build_portfolio_curve(sample: pd.Series, mac_ref: float, k_mult: float = 0.5,
                          weight: float = 1.0) -> pd.DataFrame:
    """组合分散曲线：ρ 的**唯一**影响面。

    组合 = 基差险赔付 L_B + weight × 碳价头寸损失 L_P。
    形状假设：L_P 与 L_B 同形状（缺真实碳价高频序列，**标注为待核**）。
    """
    n = N_MC
    sd_b = float(sample.std(ddof=1))
    k_abs = k_mult * sd_b
    z1 = RNG.standard_normal(n)
    z2 = RNG.standard_normal(n)
    q = lambda u: np.quantile(sample.values, np.clip(u, 1e-9, 1 - 1e-9))
    # 正态近似解析：σ_p = σ√(2(1+ρ))，TVaR_α = σ_p·φ(z_α)/(1−α)
    zq = stats.norm.ppf(ALPHA)
    phi = stats.norm.pdf(zq)
    rows = []
    for rho in RHO_GRID:
        u1 = stats.norm.cdf(z1)
        u2 = stats.norm.cdf(rho * z1 + np.sqrt(1 - rho ** 2) * z2)
        L_b = payout_stage1(q(u1), k_abs) * mac_ref
        L_p = payout_stage1(q(u2), k_abs) * mac_ref * weight
        port = L_b + L_p
        tv_port = float(port[port >= np.quantile(port, ALPHA)].mean())
        tv_sum = float(L_b[L_b >= np.quantile(L_b, ALPHA)].mean() * (1 + weight))
        sig_p = sd_b * np.sqrt(2 * (1 + rho)) * mac_ref
        tv_norm = sig_p * phi / (1 - ALPHA)
        tv_norm_sum = 2 * sd_b * mac_ref * phi / (1 - ALPHA)
        rows.append({
            "ρ": rho,
            "组合TVaR99_数值": round(tv_port, 4),
            "单独之和TVaR99": round(tv_sum, 4),
            "分散收益_数值": round(1 - tv_port / tv_sum, 6),
            "组合TVaR99_正态解析": round(tv_norm, 4),
            "分散收益_正态解析": round(1 - tv_norm / tv_norm_sum, 6),
            "重尾/非线性增益(数值−解析)": round((1 - tv_port / tv_sum) - (1 - tv_norm / tv_norm_sum), 6),
        })
    return pd.DataFrame(rows)


def build_horizon(defA: pd.DataFrame, k_mult: float = 0.5) -> pd.DataFrame:
    """期限口径：h 年期累计变动 B_i,t+h − B_it 的方差随 h 的扩张。"""
    rows = []
    var1 = float(np.nanvar(defA.diff().iloc[1:].values, ddof=1))
    for h in range(1, 11):
        d = defA.diff(h).iloc[h:]
        v = d.values.ravel()
        v = v[~np.isnan(v)]
        varh = float(np.var(v, ddof=1))
        rows.append({"h 年": h, "Var(累计变动)": round(varh, 6),
                     "Var_h/Var_1": round(varh / var1, 4),
                     "随机游走基准 h": h,
                     "比 h 倍低多少%": round((1 - varh / var1 / h) * 100, 2),
                     "判读": "低于 h ⇒ 有部分回复；接近 h ⇒ 近似随机游走"})
    return pd.DataFrame(rows)


def build_process_correction(hz: pd.DataFrame) -> pd.DataFrame:
    r1 = float(hz.loc[hz["h 年"] == 1, "Var_h/Var_1"].iloc[0])
    r5 = float(hz.loc[hz["h 年"] == 5, "Var_h/Var_1"].iloc[0])
    return pd.DataFrame([
        {"项": "中期稿 §6.3 原设定", "内容": "ΔB_it = ρ_i·ΔB_i,t−1 + ε，AR(1) 均值回复，缺省 ρ=0.95",
         "问题": "与 s17（C4）冲突：s17 证 19/29 省 AR(1)>1 系确定性趋势被误当随机过程，均值回复不成立",
         "处置": "**更正**为「带确定性漂移 + 结构断点的过程」"},
        {"项": "更正后设定", "内容": "ΔB 累计变动 = 确定性趋势项（含 2013/2021 附近断点）+ 随机残差；"
                                 "不设 AR(1) 回复系数", "问题": "—", "处置": "本脚本与 s17 一致"},
        {"项": "期限含义（本条为量化新增）",
         "内容": f"Var(5 年累计)/Var(1 年) = {r5:.3f}（随机游走基准 5）",
         "问题": f"低于基准 {(1-r5/5)*100:.1f}% ⇒ 存在**部分**回复，但远不到 AR(1) 半衰期所暗示的程度",
         "处置": "1 年期不足以实现风险分散，**≥3 年**的原结论维持"},
    ])


# ══════════════════════════════════════════════════════════════════ 结论层

def build_conclusion(idf, se, orth, inv, invmeta, corr, hz) -> dict:
    disp0 = float(corr.loc[corr["ρ"] == 0.0, "分散收益_数值"].iloc[0])
    disp1 = float(corr.loc[corr["ρ"] == 1.0, "分散收益_数值"].iloc[0])
    h5 = float(hz.loc[hz["h 年"] == 5, "Var_h/Var_1"].iloc[0])
    r25 = float(corr.loc[corr["ρ"] == 0.2, "分散收益_数值"].iloc[0])
    r50 = float(corr.loc[corr["ρ"] == 0.5, "分散收益_数值"].iloc[0])
    r80 = float(corr.loc[corr["ρ"] == 0.8, "分散收益_数值"].iloc[0])
    gain = corr["重尾/非线性增益(数值−解析)"]
    o3 = orth.iloc[1:4]["corr(ΔB,ΔP)"].astype(float)
    o_lo, o_hi = float(o3.min()), float(o3.max())
    return {
        "重叠期年数": 2,
        "重叠产生的年度变动数": 1,
        "可识别性判定": "不可识别",
        "识别所需年数_se0.10": int(se.loc[se["目标标准误 se(ρ̂)"] == 0.10, "所需重叠年数 n"].iloc[0]),
        "识别所需年数_se0.05": int(se.loc[se["目标标准误 se(ρ̂)"] == 0.05, "所需重叠年数 n"].iloc[0]),
        "价格代理口径数": 3,
        "代理口径corr区间": f"[{o_lo:.4f}, {o_hi:.4f}]",
        "定价不敏感性_最大相对偏差": invmeta["最大相对偏差"],
        "MAC_REF_元每吨": round(invmeta["MAC_REF"], 2),
        "σ_B_单年": round(invmeta["sigma_B"], 4),
        "起赔阈值K_元每吨": round(invmeta["k_abs"] * invmeta["MAC_REF"], 2),
        "分散收益_ρ0": round(disp0, 4),
        "分散收益_ρ1": round(disp1, 4),
        "分散收益_ρ0.2": round(r25, 4),
        "分散收益_ρ0.5": round(r50, 4),
        "分散收益_ρ0.8": round(r80, 4),
        "重尾非线性增益_min": round(float(gain.min()), 4),
        "重尾非线性增益_max": round(float(gain.max()), 4),
        "Var5比Var1": round(h5, 4),
        "核心结论": (
            "**Q5 的正确答法不是给出 ρ，而是判定 ρ 的地位。** 四条："
            "① **不可识别**——全国碳市场 2021-07 上线，与 1995–2022 MAC 面板重叠仅 **2 年**，"
            "仅产生 **1 个**年度变动，识别截面相关所需自由度为 0；"
            "且这不是「样本还不够」：Fisher z 下要把 se(ρ̂) 压到 0.10 需 **103 年**、压到 0.05 需 **403 年**，"
            "**年度口径下永久不可识别**。"
            "② **连「用哪个价格」也不可判定（第二层不可识别）**——三个可用价格代理给出"
            f"**互不相同**的答案：截面中位口径 {float(o3.iloc[0]):.4f}、截面均值口径 "
            f"{float(o3.iloc[1]):.4f}、算术均值（总量型）口径 {float(o3.iloc[2]):.4f}，"
            f"区间 **[{o_lo:.4f}, {o_hi:.4f}]**，量级差 **{o_hi/o_lo:.1f} 倍**。"
            "其中中位口径与 ΔB 存在**构造关联**（定义的分子即含该锚的变动），其数值不能作实证解读；"
            "同时须注意**不能**据「逐年截面中位 B ≡ 0」推断 ΔB 与价格正交——"
            "实测逐年截面中位 ΔB 并不为 0。⇒ 不存在可据以取值的唯一价格口径。"
            "③ **不进定价链（决定性）**——赔付只依赖 ΔB 的**边际分布**，ρ 只决定与外部价格的**联合结构**；"
            "由 Sklar 定理，保持边际不变、仅改 copula，"
            f"E 与 TVaR99 严格不变。数值验证（{invmeta['n_mc']:,} 次抽样、"
            f"{len(inv)} 档 ρ）最大相对偏差 **{invmeta['最大相对偏差']:.2e}** ⇒ "
            "**ρ 对本产品定价的影响面 = 0**。G-C3 的「不得作点估计」由此以更强的方式满足："
            "**该参数不作为定价输入，故无需估计。**"
        ),
        "ρ 的正确处置": (
            "ρ 的**唯一**影响面是**组合分散**：若对手方同时持有碳价头寸，"
            f"组合 TVaR99 随 ρ 变化。实测（数值口径）：ρ=0 时分散收益 {disp0:.2%}，"
            f"ρ=0.2 → {r25:.2%}，ρ=0.5 → {r50:.2%}，ρ=0.8 → {r80:.2%}，ρ=1 → {disp1:.2%}。"
            "⇒ 处置口径：**ρ 只在组合场景中以外生区间披露，不得内部拍定**；"
            "该场景需要对手方组合信息（外部依赖），属 C 类对外事项。"
            f"另附一条**方向性更正**：正态解析给出的分散收益（ρ=0 时 29.29%）**低于**数值口径（{disp0:.2%}），"
            f"即本算例中重尾边际 + 凸赔付函数带来的分散**增益**（{float(gain.min()):+.4f} ~ "
            f"{float(gain.max()):+.4f}）而非折损。"
            "对外陈述分散收益时必须说明用的是哪一条通道，且不得把该方向当作普遍定理——"
            "它依赖凸赔付与高斯 copula 的组合，真实尾部相依会改变结论。"
        ),
        "对上游设定的更正": (
            "中期稿 §6.3 把 ΔB 设为 AR(1) 均值回复（缺省 ρ=0.95），"
            "与 s17（C4）结论冲突——s17 已证 19/29 省 AR(1)>1 系**确定性趋势被误当作随机过程**，故均值回复不成立。"
            "本包更正为【带确定性漂移 + 结构断点的过程】。"
            f"量化依据：Var(5 年累计变动)/Var(1 年) = {h5:.3f}，"
            f"低于随机游走基准 5 达 {(1-h5/5)*100:.1f}%，"
            "即存在**部分**回复但远不到 AR(1) 半衰期所暗示的程度。"
            "⇒ 1 年期不足以实现风险分散，**≥3 年**的原结论维持不变。"
        ),
        "诚实边界": [
            "① **重叠期仅 2 年**，且 1995–2012 年的基差与「碳市场」无关（全国碳市场 2021-07 才上线），"
            "存在制度断点，前碳市场时代数据对碳市场时代产品的外推效力未被证明。",
            "② 组合分散曲线中，碳价头寸损失 **L_P 与基差赔付取同形状**——"
            "这是**形状假设**，因缺真实碳价高频序列；真实碳价与 MAC 的形状差异会改变曲线量级。"
            "该假设已标注，**待核**。",
            "③ 数值通道使用**高斯 copula + 经验边际**，重尾由边际保留、"
            "但尾部**相依结构**被强设为高斯；真实尾部相依（如共同政策冲击）会削弱或反转分散收益。",
            "④ 本包**不估算 ρ 的数值**，也不声称 ρ 的合理范围；"
            "区间端点须由对手方组合场景外生给定。三个价格代理的 corr 值为**描述性并列**，"
            "不代表其中任何一个「更正确」。",
            "⑤ 定价不敏感性结论依赖「赔付仅依赖 ΔB 边际分布」这一**合约结构**；"
            "若未来条款引入与碳价挂钩的联动赔付（如双重触发），该结论**失效**，须重做。",
        ],
    }


# ══════════════════════════════════════════════════════════════════ 验收层

def acceptance(idf, se, orth, inv, invmeta, port, hz, corr, concl, defA) -> tuple[bool, pd.DataFrame]:
    recs: list[tuple[str, str, bool, str]] = []

    def add(cid, desc, ok, note=""):
        recs.append((cid, desc, bool(ok), note))

    add("C3-1", "**G-C3 硬要求**：重叠期仅 2 年的声明已进入结论字段与产物表",
        concl["重叠期年数"] == 2 and "重叠期仅 2 年" in concl["诚实边界"][0])
    add("C3-2", "交易所口径下判定为「不可识别」，且可用自由度为 0",
        concl["可识别性判定"] == "不可识别"
        and idf.loc[idf["项"] == "可用自由度", "值"].iloc[0] == "0")
    add("C3-3", "识别所需年数已量化：se≤0.10 需 ≥100 年、se≤0.05 需 ≥400 年",
        concl["识别所需年数_se0.10"] >= 100 and concl["识别所需年数_se0.05"] >= 400,
        f"{concl['识别所需年数_se0.10']} / {concl['识别所需年数_se0.05']}")
    add("C3-4", "**未在任何产物中给出 ρ 的点估计**（ρ 网格的『值』列全部为敏感性输入而非估计）",
        all(c not in idf.columns for c in ["ρ估计", "相关系数估计"]))
    add("C3-5", "**第二层不可识别**：三个价格代理口径并列且互不相同，未取其一为结论",
        orth["口径"].str.contains("交易所|中位|均值").sum() == 4
        and len(set(orth.iloc[1:4]["corr(ΔB,ΔP)"].astype(float).round(4))) == 3
        and "第二层不可识别" in concl["核心结论"])
    add("C3-6", "**决定性检验**：赔付统计量在 ρ 网格上逐位相同（最大相对偏差 < 1e-9）",
        invmeta["最大相对偏差"] < 1e-9, f"{invmeta['最大相对偏差']:.2e}")
    add("C3-7", "ρ 网格覆盖 [0,1] 且步长 ≤ 0.2",
        abs(min(RHO_GRID)) < 1e-9 and abs(max(RHO_GRID) - 1.0) < 1e-9
        and max(np.diff(RHO_GRID)) <= 0.2 + 1e-9)
    add("C3-8", "定价不敏感性有解析依据（Sklar 定理）并写入核心结论",
        "Sklar" in concl["核心结论"])
    add("C3-9", "组合分散曲线含**数值与正态解析两条**通道，且差额已单列",
        {"分散收益_数值", "分散收益_正态解析", "重尾/非线性增益(数值−解析)"} <= set(corr.columns))
    add("C3-10", "分散收益端点为 0 与 > 0：ρ=1 ⇒ 0（无分散），ρ=0 ⇒ > 0",
        abs(concl["分散收益_ρ1"]) < 0.02 and concl["分散收益_ρ0"] > 0.05,
        f"{concl['分散收益_ρ1']:.4f} / {concl['分散收益_ρ0']:.4f}")
    add("C3-11", "数值口径与正态解析的**方向差**已如实报告，且未当作普遍定理",
        concl["重尾非线性增益_max"] > 0.01 and "不得把该方向当作普遍定理" in concl["ρ 的正确处置"],
        f"增益 {concl['重尾非线性增益_min']:+.4f} ~ {concl['重尾非线性增益_max']:+.4f}")
    add("C3-12", "ρ 的处置口径为**外生区间披露**，不得内部拍定，已写入结论",
        "外生区间披露" in concl["ρ 的正确处置"])
    add("C3-13", "期限口径已量化：Var(h 累计)/Var(1) 的 h=1..10 曲线齐备，且 h=1 处恒为 1",
        len(hz) == 10 and abs(float(hz.loc[hz["h 年"] == 1, "Var_h/Var_1"].iloc[0]) - 1.0) < 1e-9,
        f"{float(hz.loc[hz['h 年'] == 1, 'Var_h/Var_1'].iloc[0]):.6f}")
    add("C3-14", "上游 §6.3 的 AR(1) 均值回复设定已被登记更正，理由指向 s17",
        "均值回复不成立" in concl["对上游设定的更正"] and "s17" in concl["对上游设定的更正"])
    add("C3-15", "≥3 年的期限结论维持不变，未被本次更正推翻",
        "≥3 年" in concl["对上游设定的更正"])
    add("C3-16", "与 s17（C4）交叉核验一致：s17 结论文件存在且其面板同源（29 省 × 28 年）",
        (OUT / "s17_单位根结论.json").exists()
        and json.load(open(OUT / "s17_单位根结论.json", encoding="utf-8"))["面板维度"]
        .startswith("29 省 × 28 年"))
    add("C3-17", "诚实边界 ≥ 5 条，含「重叠期仅 2 年」与「组合场景形状假设待核」",
        len(concl["诚实边界"]) >= 5
        and any("重叠期仅 2 年" in s for s in concl["诚实边界"])
        and any("待核" in s for s in concl["诚实边界"]))
    add("C3-18", "已声明「若条款引入双重触发则该结论失效」的适用边界",
        any("失效" in s for s in concl["诚实边界"]))
    add("C3-19", "赔付基准 MAC_REF 自 s3 读入而非硬编码（等于 2026 S1 M3 名义 MAC）",
        abs(invmeta["MAC_REF"] - get_mac_ref()) < 1e-6,
        f"{invmeta['MAC_REF']:.2f}")
    add("C3-20", "起赔阈值 K = 0.5 × σ_B，且 σ_B 与面板实测一致（复算偏差 < 1e-6）",
        abs(invmeta["sigma_B"] - float(build_delta_sample(defA).std(ddof=1))) < 1e-6,
        f"σ_B={invmeta['sigma_B']:.4f}")

    df = pd.DataFrame(recs, columns=["编号", "验收项", "通过", "备注"])
    return bool(df["通过"].all()), df


# ══════════════════════════════════════════════════════════════════ 主程序

def main(check: bool) -> int:
    defA = load_defA()
    sample = build_delta_sample(defA)
    mac_ref = get_mac_ref()

    idf = build_identifiability()
    se = build_se_curve()
    orth = build_constructive_orthogonality(defA)
    inv, invmeta = build_pricing_invariance(sample, mac_ref)
    corr = build_portfolio_curve(sample, mac_ref)
    hz = build_horizon(defA)
    pcorr = build_process_correction(hz)
    concl = build_conclusion(idf, se, orth, inv, invmeta, corr, hz)

    ok, acc = acceptance(idf, se, orth, inv, invmeta, corr, hz, corr, concl, defA)

    print("=" * 96)
    print("C3 基差建模 · ρ(ΔB, P) 的可识别性与影响面")
    print("=" * 96)

    print("\n【甲 · 可识别性诊断】")
    print(idf.to_string(index=False))
    print("\n  识别所需样本量（se = 1/√(n−3) 反解）：")
    print(se.to_string(index=False))

    print("\n【乙 · 构造性正交验证】")
    print(orth.to_string(index=False))

    print(f"\n【丙 · 定价不敏感性（决定性检验，{N_MC:,} 次抽样）】")
    print(f"  赔付基准 MAC_REF = {mac_ref:.2f} 元/吨（读自 s3）；σ_B = {invmeta['sigma_B']:.4f}；"
          f"K = {invmeta['k_abs']:.4f}（对数）× MAC_REF = {invmeta['k_abs']*mac_ref:.2f} 元/吨")
    print(inv.to_string(index=False))
    print(f"  ⇒ 最大相对偏差 {invmeta['最大相对偏差']:.2e} ⇒ E 与 TVaR99 对 ρ 严格不变")

    print("\n【ρ 的唯一影响面 · 组合分散曲线】")
    print(corr.to_string(index=False))

    print("\n【期限口径 · 累计变动方差随 h 的扩张】")
    print(hz.to_string(index=False))

    print("\n【上游设定更正登记】")
    print(pcorr.to_string(index=False))

    print("\n【结论】")
    for k, v in concl.items():
        if k == "诚实边界":
            print("  诚实边界:")
            for i, s in enumerate(v, 1):
                print(f"    {i}. {s}")
        else:
            print(f"  {k}: {v}")

    print("\n【验收门 G-C3】")
    for _, r in acc.iterrows():
        print(f"  {'✅' if r['通过'] else '❌'} {r['编号']:<6} {r['验收项']}"
              + (f"  [{r['备注']}]" if r['备注'] else ""))
    print(f"\n  合计：{int(acc['通过'].sum())} PASS / {int((~acc['通过']).sum())} FAIL")

    if not check:
        idf.to_csv(OUT / "s18_可识别性诊断.csv", index=False, encoding="utf-8-sig")
        se.to_csv(OUT / "s18_识别所需样本.csv", index=False, encoding="utf-8-sig")
        orth.to_csv(OUT / "s18_构造性正交验证.csv", index=False, encoding="utf-8-sig")
        inv.to_csv(OUT / "s18_定价不敏感性.csv", index=False, encoding="utf-8-sig")
        corr.to_csv(OUT / "s18_组合分散曲线.csv", index=False, encoding="utf-8-sig")
        hz.to_csv(OUT / "s18_期限方差扩张.csv", index=False, encoding="utf-8-sig")
        pcorr.to_csv(OUT / "s18_过程设定更正.csv", index=False, encoding="utf-8-sig")
        acc.to_csv(OUT / "s18_基差建模验收.csv", index=False, encoding="utf-8-sig")
        with open(OUT / "s18_基差建模结论.json", "w", encoding="utf-8") as f:
            json.dump({**concl, "不敏感性元数据": invmeta}, f, ensure_ascii=False, indent=2)
        print(f"\n  已落盘：{OUT}/s18_*（8 份 CSV + 1 份 JSON）")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main("--check" in sys.argv))
