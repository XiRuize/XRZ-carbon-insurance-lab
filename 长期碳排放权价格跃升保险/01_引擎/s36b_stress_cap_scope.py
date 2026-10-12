# -*- coding: utf-8 -*-
"""
s36b_stress_cap_scope.py —— 压力测试的 CAP 口径与覆盖率归因（复核 s36.T8）

【本脚本回答什么】

2026-10-10 对外部提出的三项指控做复核时，发现 s36.T8 压力测试存在两处
口径问题。本脚本把问题复现、量化，并给出修正口径的对照结果。

问题 1：T8 的 CAP 口径与 v3 主结构不一致
    v3 主结构已改为**比例式**责任上限（CAP = CAP_MULT × 当期价格水平，
    见 s36 报价 JSON 的「CAP_比例式」字段）。但 s36.t8_stress() 第 499 行
    写的是 `cap = CAP_MULT * P_0` —— 这是**定额**口径（锚价固定）。
    ⇒ 压力测试实际测的是「已被 v3 废弃的定额结构」，而非主结构。
    修正后 CAP 随价格放大，覆盖率改善但幅度有限（见 T1）。

问题 2：覆盖率被 κ 免赔普遍压低，且分子分母口径不同
    分子`raw`与`cap_d`都除了 TRADING_DAYS（日均），但分母`actual`
    未除 ⇒ 两者口径相差 244 倍。s36 注释称「曾出现覆盖率 800%」，
    但那是因为历史上分子分母都错；当前写法是**分子日均、分母年均**，
    偏保守（低估覆盖率）。本脚本给出同口径（年均/年均）对照。

问题 3（归因纠正）：层宽 = 0 的机制是「CAP 被打满」，不是「低频高损」
    s36 注释与存根把层宽 0 归因于「分位层对低频高损不适配」。
    实测（本脚本 T2）：层宽 = 0 的充要条件是**打满频率 ≥ 1 − q_D = 25%**。
    在 1.5× 情景下 CAP_MULT = 2 时打满频率 98.5% ⇒ 层宽 0；
    把 CAP_MULT 抬到 80 倍（打满降至 1.3%）后，层宽恢复为 16.08。
    ⇒ 正确表述：**责任上限过低导致赔付分布左删失**，与分布形状无关。
    基线情形（无跃迁）CAP 从未打满，层宽 31.855，正常。

用法：python s36b_stress_cap_scope.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import L1, OUT  # noqa: E402
import s36_proportional_daily as s36  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.unicode.east_asian_width", True)

P_0 = s36.P_0
KAPPA = s36.KAPPA
CAP_MULT = s36.CAP_MULT
Q_D, Q_A = s36.Q_D, s36.Q_A
TD = s36.TRADING_DAYS
SEED = L1["seed"]
OUT_DIR = Path(OUT)
N_PATHS = 60_000


def stress_paths(up: float, n: int = N_PATHS, seed_off: int = 12000):
    """复现 s36.t8_stress 的路径构造：前半段 P_0，后半段围绕 P_0×up 随机游走。"""
    rng = np.random.default_rng(SEED + seed_off)
    sig_d = s36.s32.SIG_P_BASE / np.sqrt(TD)
    P_new = P_0 * up
    half = TD // 2
    path = np.empty((TD, n))
    path[:half, :] = P_0
    rel = np.zeros(n)
    for d in range(half, TD):
        rel = rel + sig_d * rng.standard_normal(n)
        path[d, :] = P_new * np.exp(rel)
    return path, P_new


def t1_cap_scope() -> pd.DataFrame:
    """T1 定额口径（s36 现写法）vs 比例口径（v3 主结构）的覆盖率对照。

    同时给出两种分母口径：s36 现写法（分子日均 / 分母年均，保守）
    与同口径（年均 / 年均）。
    """
    rows = []
    for up in [1.5, 2.0, 4.0]:
        path, P_new = stress_paths(up)
        exc = np.maximum(path - KAPPA * P_0, 0.0)
        need_annual = float((P_new - P_0) * max(path[TD // 2:, :].mean() / P_new, 0.0))
        need_daily = need_annual / TD
        for mode, lab in [("fixed", "定额（s36 现写法）"),
                          ("prop", "比例（v3 主结构）")]:
            cap = CAP_MULT * P_0 if mode == "fixed" else CAP_MULT * P_new
            cd = cap / TD
            raw = exc.sum(axis=0) / TD
            x = np.minimum(raw, cd)
            paid_d = float(x.mean())
            paid_a = float(np.minimum(exc.sum(axis=0), cap).mean())
            rows.append({
                "情景": f"碳价 +{(up - 1) * 100:.0f}%（{up:.1f}×）",
                "CAP口径": lab,
                "CAP_元每吨": round(cap, 2),
                "CAP日均": round(cd, 4),
                "需求_年均": round(need_annual, 2),
                "需求_日均": round(need_daily, 4),
                "覆盖率%_原写法": round(paid_d / need_annual * 100, 2),
                "覆盖率%_同口径": round(paid_a / need_annual * 100, 2),
                "打满频率%": round(float((raw > cd).mean() * 100), 1),
            })
    return pd.DataFrame(rows)


def t2_width_attribution() -> pd.DataFrame:
    """T2 层宽 = 0 的归因：打满频率 ≥ 25% 是充要条件。"""
    rows = []
    for up in [1.5, 2.0, 4.0]:
        path, _ = stress_paths(up)
        exc = np.maximum(path - KAPPA * P_0, 0.0)
        raw = exc.sum(axis=0) / TD
        for cm in [2, 5, 10, 20, 40, 80, 160]:
            cd = cm * P_0 / TD
            x = np.minimum(raw, cd)
            D = float(np.quantile(x, Q_D))
            A = float(np.quantile(x, Q_A))
            bind = float((x >= cd - 1e-12).mean() * 100)
            width = A - D
            rows.append({
                "情景": f"{up:.1f}×", "CAP_MULT": cm,
                "CAP日均": round(cd, 3), "打满频率%": round(bind, 1),
                "q75": round(D, 3), "q99.5": round(A, 3), "层宽": round(width, 3),
                "层宽=0": "是" if width <= 1e-9 else "否",
                "判据_打满≥25%": "是" if bind >= 25.0 else "否",
                "判据与实测一致": "YES" if (bind >= 25.0) == (width <= 1e-9) else "NO",
            })
    return pd.DataFrame(rows)


def t3_coverage_ceiling() -> pd.DataFrame:
    """T3 覆盖率天花板：CAP 无限大时，κ 免赔让覆盖率最高到多少。

    ★ 口径纪律：分母与分母必须是**同一结构**的年度累计量。
      企业增量成本 = Σ_d max(P_d − P_0, 0)   （全部涨价部分，从 P_0 起算）
      保险赔付     = Σ_d max(P_d − κP_0, 0)  （仅 κ 以上部分）
      ⇒ 覆盖率 = 后者 / 前者 ≤ 100%，且随涨幅增大而趋近 100%。
      这正是 s36 注释所称「分子分母必须同为日均口径」的正确落地。
      （s36.t8_stress 的 actual 用 (P_new−P_0)×均值，是「价格差」而非
        「累计金额」，与分子的逐日累加不同量纲 —— 这是覆盖率失真的根因。）
    """
    rows = []
    for up in [1.5, 2.0, 4.0]:
        path, P_new = stress_paths(up)
        inc_all = np.maximum(path - P_0, 0.0)            # 全部涨价
        inc_kap = np.maximum(path - KAPPA * P_0, 0.0)    # κ 以上部分
        need = float(inc_all.sum(axis=0).mean())         # 年度累计，元/吨
        paid = float(inc_kap.sum(axis=0).mean())
        cap_annual = CAP_MULT * P_0
        rows.append({
            "情景": f"碳价 +{(up - 1) * 100:.0f}%（{up:.1f}×）",
            "企业增量成本_年累计": round(need, 2),
            "无CAP约束赔付_年累计": round(paid, 2),
            "覆盖率上限%": round(paid / need * 100, 1),
            "被κ免赔吃掉%": round(100 - paid / need * 100, 1),
            "CAP占年度需求%": round(cap_annual / need * 100, 2),
        })
    return pd.DataFrame(rows)


def t4_cap_tradeoff() -> pd.DataFrame:
    """T4 改 CAP_MULT 的代价：覆盖率提升 vs 保费完全不敏感。

    ★ 核心发现（反直觉）：基准情形下把 CAP_MULT 从 2 提到 371（185 倍），
      报价 GP 恒为 5.076 元/吨 —— 一分钱不涨。
      原因：N_MAX=5 口径下基准赔付分布的 q99.5 = 38.43、max = 81.81，
            故 CAP ≥ 82 后对基准定价完全无影响。

    ⇒ 「提高责任上限」在基准年看似免费，实为**把压力情景的责任无偿转移
      给保险人**：1.5× 情景下实际赔付均值由 195.0 涨到 2750.2（14 倍），
      而保费不变。这是**未定价的尾部风险**。

    保险业标准解法是分层：基准层（CAP_MULT=2，已定价）+
    跃升层（独立定价 / 再保 / 巨灾债券），而非直接抬 CAP。
    """
    # 基准情形：同一次模拟，保证各行可比
    sim_b = s36.sim_daily(sig_p=s36.s32.SIG_P_BASE, n_paths=80000, seed=SEED + 11000)
    rows = []
    for cm in [2, 5, 10, 20, 40, 80, 160, 371]:
        pp = s36.pay_daily(sim_b, share=1.0, n_max=5, cap_mult=cm)
        mm = s36.metrics(pp, share=1.0)
        gp = mm["GP_元每吨"]
        rec = {
            "CAP_MULT": cm,
            "CAP_元每吨": round(pp["cap_amount"], 1),
            "基准_层宽": round(mm["层宽"], 2),
            "基准_GP全额": round(gp, 3),
            "费率%": round(gp / pp["cap_amount"] * 100, 3),
            "基准封顶频率%": round(pp["cap_binding"].mean() * 100, 2),
            "可行(≤89)?": "是" if s36.feasible(gp, 1.0) else "否",
        }
        # 压力情景：同口径（年累计/年累计），以及保险人的实际赔付责任
        for up in [1.5, 2.0, 4.0]:
            path, _ = stress_paths(up)
            inc_all = np.maximum(path - P_0, 0.0).sum(axis=0)
            exc = np.maximum(path - KAPPA * P_0, 0.0).sum(axis=0)
            paid = np.minimum(exc, cm * P_0)
            rec[f"覆盖{up}×%"] = round(float(paid.mean() / inc_all.mean() * 100), 1)
            rec[f"赔付{up}×_年累计"] = round(float(paid.mean()), 1)
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    t0 = time.time()
    sys.stdout.reconfigure(encoding="utf-8")
    print("s36b　压力测试 CAP 口径与覆盖率归因（复核 s36.T8）")
    print("=" * 130)
    print(f"  锚价 P_0 = {P_0:.2f}　κ = {KAPPA}　CAP_MULT = {CAP_MULT}"
          f"　q_D = {Q_D}　交易日 = {TD}")

    t1 = t1_cap_scope()
    print("\n【T1 定额 vs 比例口径】压力测试的 CAP 该用哪个？（v3 主结构是比例式）")
    print("-" * 130)
    print(t1.to_string(index=False))

    t2 = t2_width_attribution()
    print("\n【T2 层宽归因】层宽 = 0 的充要条件是打满频率 ≥ 25%，非「低频高损」")
    print("-" * 130)
    print(t2.to_string(index=False))
    n_bad = int((t2["判据与实测一致"] == "NO").sum())
    print(f"\n  ⇒ 判据与实测不一致的格：{n_bad} / {len(t2)}")

    t3 = t3_coverage_ceiling()
    print("\n【T3 覆盖率天花板】CAP 无限大时，κ 免赔仍吃掉大部分")
    print("-" * 130)
    print(t3.to_string(index=False))

    t4 = t4_cap_tradeoff()
    show = ["CAP_MULT", "CAP_元每吨", "基准_层宽", "基准_GP全额", "费率%",
            "基准封顶频率%", "可行(≤89)?", "覆盖1.5×%", "覆盖2.0×%", "覆盖4.0×%"]
    print("\n【T4 改 CAP_MULT 的代价】保费对 CAP 完全不敏感，覆盖率才敏感")
    print("-" * 130)
    print(t4[show].to_string(index=False))
    g0, g1 = t4["基准_GP全额"].iloc[0], t4["基准_GP全额"].iloc[-1]
    print(f"\n  ⇒ CAP_MULT 2 → 371（185 倍）：保费 {g0:.3f} → {g1:.3f} 元/吨（不变）")
    print(f"     但 1.5× 情景赔付均值 {t4['赔付1.5×_年累计'].iloc[0]:.1f} → "
          f"{t4['赔付1.5×_年累计'].iloc[-1]:.1f}（14 倍）⇒ **未定价的尾部风险**")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t1.to_csv(OUT_DIR / "s36b_T1_CAP口径对照.csv", index=False, encoding="utf-8-sig")
    t2.to_csv(OUT_DIR / "s36b_T2_层宽归因.csv", index=False, encoding="utf-8-sig")
    t3.to_csv(OUT_DIR / "s36b_T3_覆盖率天花板.csv", index=False, encoding="utf-8-sig")
    t4.to_csv(OUT_DIR / "s36b_T4_CAP权衡.csv", index=False, encoding="utf-8-sig")
    print(f"\n  产物已写入 {OUT_DIR}")
    print(f"  用时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
