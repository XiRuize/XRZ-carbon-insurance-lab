# -*- coding: utf-8 -*-
"""
s13_policy_year.py —— B4 保单年度（underwriting year, UY）视图

设计依据：`02_产物/中期_双指标报价与重定价条款_设计_v1.md` §8.2 / §8.3、§9.3 A17。

为什么本产品尤其需要 UY 视图（§8.1）：指数水平逐年抬升、重定基使阈值逐年上抬、
重定价把保单切成 1–3 年子期间 ⇒ 同一张保单各年赔付占比极不均匀，
日历年（calendar year）口径会把不同承保年度的责任混在一起。

输出：
    02_产物/s13_保单年度视图.csv     逐档 × 期限 × 年序的赔付分解与子期间归属
    02_产物/s13_承保年度叠加.csv     各日历年内不同承保年度的责任叠加（2028 集中风险年）
    02_产物/s13_保单年度验收.csv     断言结果
    02_产物/s13_保单年度配置.json    口径留痕

用法：
    python s13_policy_year.py            # 生成产物 + 打印
    python s13_policy_year.py --check    # 只跑断言（退出码 0 = 全过）

⚠ 只读上游：s4_pricing（模拟过程）、s4_定价配置.json。不修改任何既有脚本或产物。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import L1, OUT, GAP_MULTIPLE                     # noqa: E402
import s4_pricing as s4                                        # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.unicode.east_asian_width", True)

SCEN = "S1_基准"
CALIBER = "M2_价格平移"
N_PATH = L1["mc_paths"]
SEED0 = s4.RNG_SEED
DF_R = L1["discount"]
Q_RISK, Q_LIMIT = 0.99, 0.995
KAPPA = s4.KAPPA_BASE

# §3.2(a) 结构有效性失效年（s2_全国输入路径.csv：u 首次 ≤0.005）
BREACH_YEAR = 2028
# §5.1 重定价周期：min(3, 距越界年数−1)，下限 1 年
PERIOD_CAP = 3


def _tvar(x: np.ndarray, a: float) -> float:
    q = np.quantile(x, a)
    t = x[x >= q]
    return float(t.mean()) if t.size else float(q)


def reprice_periods(start_year: int, term: int) -> list[tuple[int, int, int]]:
    """§5.1 重定价子期间切分。

    返回 [(子期间序号, 起年, 止年含)]，止年的含义是「该子期间覆盖到哪一年」。
    周期 = max(1, min(3, 距越界年数 − 1))；2026 起保 ⇒ 每年一期。

    ⚠ 覆盖 n+1 个时点（t=0..n），与 s7 的 `pay[:n+1]` / `arange(n+1)` 口径一致。
    """
    last = start_year + term
    out, y, k = [], start_year, 1
    while y <= last:
        to_breach = BREACH_YEAR - y
        plen = max(1, min(PERIOD_CAP, to_breach - 1))
        end = min(y + plen - 1, last)
        out.append((k, y, end))
        y = end + 1
        k += 1
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    only_check = "--check" in sys.argv

    tiers = s4._lp_cache()
    cfg = json.load(open(OUT / "s4_定价配置.json", encoding="utf-8"))["标度档"]

    print("=" * 118)
    print("s13　B4 保单年度（UY）视图　·　S1 基准 · L1b 指数口径 · M2 价格平移")
    print("=" * 118)
    print(f"  路径数={N_PATH}  贴现率={DF_R:.1%}  越界年={BREACH_YEAR}  "
          f"重定价周期上限={PERIOD_CAP} 年")

    rows, agg_rows = [], []
    share_sum: dict[tuple, float] = {}      # 未舍入的占比累计（供 A17-1 精确断言）
    E_sum: dict[tuple, float] = {}          # 未舍入的逐年现值之和（供 A17-2 精确断言）
    for i, tag in enumerate(tiers):
        sim = s4.simulate(SCEN, tiers[tag], CALIBER, N_PATH, SEED0 + i * 977)
        I, Q = sim["I"], sim["Q"]
        K = KAPPA * s4.MAC_REF
        pay = np.minimum(np.maximum(I - K, 0.0), s4.CAP_AMOUNT) * Q   # (年 × 路径)
        for n in s4.TERMS:
            start = s4.START
            per = reprice_periods(start, n)
            periods = np.zeros(n + 1, dtype=int)
            for k, y0, y1 in per:
                periods[y0 - start: y1 - start + 1] = k
            df = 1.0 / (1.0 + DF_R) ** np.arange(n + 1)
            pv_y = pay[:n + 1] * df[:, None]                 # 逐年折现赔付
            cum_pv = pv_y.sum(axis=0)
            E_tot = float(cum_pv.mean())
            for t in range(n + 1):
                col = pv_y[t]
                E_t = float(col.mean())
                share = E_t / E_tot if E_tot > 0 else 0.0
                share_sum[(tag, n)] = share_sum.get((tag, n), 0.0) + share
                E_sum[(tag, n)] = E_sum.get((tag, n), 0.0) + E_t
                rows.append({
                    "标度档": tag, "期限": n, "承保年度UY": start,
                    "年序t": t, "日历年": start + t,
                    "子期间序号": int(periods[t]),
                    "子期间起年": [p[1] for p in per if p[0] == periods[t]][0],
                    "子期间止年": [p[2] for p in per if p[0] == periods[t]][0],
                    "E_t_现值": round(E_t, 1),
                    "E_t_未折现": round(float(pay[t].mean()), 1),
                    "占全期比重": round(share, 6),
                    "线性占比_1除n": round(1.0 / (n + 1), 6),
                    "逐年P99.5": round(float(np.quantile(col, Q_LIMIT)), 1),
                    "逐年TVaR99": round(_tvar(col, Q_RISK), 1),
                    "越界后年份": "是" if start + t > BREACH_YEAR else "否",
                })
            cum = pv_y.cumsum(axis=0)
            for t in range(n + 1):
                agg_rows.append({
                    "标度档": tag, "期限": n, "承保年度UY": start, "年序t": t,
                    "累计E": round(float(cum[t].mean()), 1),
                    "累计P99.5": round(float(np.quantile(cum[t], Q_LIMIT)), 1),
                    "累计TVaR99": round(_tvar(cum[t], Q_RISK), 1),
                    "累计占比": round(float(cum[t].mean()) / E_tot, 6) if E_tot > 0 else 0.0,
                })

    D = pd.DataFrame(rows)
    A = pd.DataFrame(agg_rows)

    # ---------- 打印：论文口径档的 UY 曲线（UPR 赚取模式） ----------
    print("\n【UPR 赚取模式：论文口径档的赔付/风险占比曲线 vs 线性 1/(n+1)】")
    print("-" * 118)
    for n in s4.TERMS:
        x = D[(D["标度档"] == "论文口径") & (D["期限"] == n)]
        cum = x["占全期比重"].cumsum()
        dev = float(cum.iloc[-2] if len(cum) > 1 else 0) - (len(x) - 1) / len(x)
        print(f"\n  ── {n} 年期 ──  （末年占比 {x['占全期比重'].iloc[-1]:.3%}，"
              f"前 {n} 年累计 {cum.iloc[-2] if len(cum) > 1 else 0:.3%}）")
        print(x[["年序t", "日历年", "子期间序号", "E_t_现值", "占全期比重", "线性占比_1除n",
                 "逐年P99.5", "越界后年份"]].to_string(index=False))

    # ---------- 2028 集中风险年 ----------
    print("\n【2028 集中风险年：各承保年度在 2028 年的责任叠加（论文口径 5 年期）】")
    print("-" * 118)
    overlap = []
    for uy in [2026, 2027, 2028, 2029]:
        for n in [5]:
            p = reprice_periods(uy, n)
            if uy <= BREACH_YEAR <= uy + n:
                yr = BREACH_YEAR - uy
                rec = D[(D["标度档"] == "论文口径") & (D["期限"] == n)
                        & (D["承保年度UY"] == 2026) & (D["年序t"] == yr)]
                share = float(rec["占全期比重"].iloc[0]) if len(rec) else np.nan
                overlap.append({"承保年度UY": uy, "期限": n, "覆盖2028": "是",
                                "2028在其第几年": yr, "该年赔付占本保单全期比重": round(share, 4),
                                "子期间数": len(p)})
    O = pd.DataFrame(overlap)
    print(O.to_string(index=False) if len(O) else "  （无）")

    # ---------- 结构性观察：线性 vs 风险模式 ----------
    print("\n【UPR 计提模式差异：线性法低估早期负债的幅度】")
    print("-" * 118)
    mis = []
    for n in s4.TERMS:
        x = D[(D["标度档"] == "论文口径") & (D["期限"] == n)].reset_index(drop=True)
        # 第 1 年（t=0）后，若按线性已赚 1/(n+1)，风险模式实际已赚 w0
        w0 = float(x["占全期比重"].iloc[0])
        lin0 = 1.0 / (n + 1)
        mis.append({"期限": n, "第1年风险占比": round(w0, 4), "线性占比": round(lin0, 4),
                    "相对低估倍数": round(lin0 / w0, 3) if w0 > 0 else np.inf})
    M = pd.DataFrame(mis)
    print(M.to_string(index=False))
    print("  ⇒ 线性法在第 1 年后高估已赚保费（= 低估 UPR）⇒ 早期负债被压低的倍数见上表。")

    # ================= 断言 =================
    res = []

    def add(code, name, ok, got, exp):
        res.append({"验收项": code, "名称": name, "结果": "PASS" if ok else "FAIL",
                    "实测": str(got), "期望": str(exp)})

    # A17-1 累计占比终值 = 1（对**未舍入**值断言；CSV 列为 6 位小数，不用于断言）
    bad = [(k, v) for k, v in share_sum.items() if abs(v - 1.0) > 1e-12]
    add("A17-1", "各档 × 期限的赔付占比之和 = 1", not bad,
        f"违例 {len(bad)} 组" + (f"，例 {bad[0]}" if bad else ""), "0 组违例（容差 1e-12）")

    # A17-2 主表与 s7 的 E 对账（未舍入的逐年现值之和；s7 的 E 为 1 位小数）
    s7 = pd.read_csv(OUT / "s7_精算口径保费.csv")
    diff = []
    for (tg, n), got in E_sum.items():
        exp = float(s7[(s7["标度档"] == tg) & (s7["期限"] == n)]["E"].iloc[0])
        diff.append(abs(got - exp))
    add("A17-2", "逐年赔付现值之和 = s7 的 E（日历口径对账）", max(diff) <= 0.06,
        f"最大绝对偏差 {max(diff):.3e}", "≤ 0.06（s7 的 E 舍入到 1 位小数）")

    # A17-3 子期间切分：连续、无重叠、覆盖全期、周期 ≤3 且 ≤ 距越界年数−1
    perr = []
    for n in s4.TERMS:
        per = reprice_periods(s4.START, n)
        cover = []
        for k, y0, y1 in per:
            if y1 < y0:
                perr.append((n, k, "空期"))
            cover += list(range(y0, y1 + 1))
        if cover != list(range(s4.START, s4.START + n + 1)):
            perr.append((n, "覆盖不符", cover[:3]))
        for k, y0, y1 in per:
            ln = y1 - y0 + 1
            if ln < 1 or ln > PERIOD_CAP:
                perr.append((n, k, f"长度 {ln}"))
            if ln > max(1, BREACH_YEAR - y0 - 1):
                perr.append((n, k, f"超越界约束(len={ln},y0={y0})"))
    add("A17-3", "重定价子期间连续覆盖全期且周期合规", not perr,
        f"违例 {perr[:3]}" if perr else "全部合规", "0 违例")

    # A17-4 越界后年份标识正确
    bad4 = D[(D["日历年"] > BREACH_YEAR) != (D["越界后年份"] == "是")]
    add("A17-4", "越界后年份标识与日历年一致", len(bad4) == 0,
        f"{len(bad4)} 行不符", "0 行")

    # A17-5 累计占比单调不减
    bad5 = []
    for (tg, n), g in A.groupby(["标度档", "期限"]):
        d = g.sort_values("年序t")["累计占比"].diff().dropna()
        if (d < -1e-9).any():
            bad5.append((tg, n))
    add("A17-5", "累计赔付占比随年序单调不减", not bad5,
        f"违例 {bad5[:3]}" if bad5 else "全部单调", "0 违例")

    # A17-6 越界后年份的占比之和非零（说明确有 2029+ 责任需受限计量）
    late = D[(D["标度档"] == "论文口径") & (D["期限"] == 15) & (D["越界后年份"] == "是")]
    share_late = float(late["占全期比重"].sum())
    add("A17-6", "15 年期存在 2029+ 责任（A14 受限计量的对象）", share_late > 0,
        f"占比 {share_late:.4f}", "> 0")

    V = pd.DataFrame(res)
    print("\n【验收】B4 保单年度视图")
    print("-" * 118)
    print(V.to_string(index=False))
    nf = int((V["结果"] == "FAIL").sum())
    print(f"\n  合计：{len(V) - nf} PASS / {nf} FAIL")

    if only_check:
        sys.exit(1 if nf else 0)

    D.to_csv(OUT / "s13_保单年度视图.csv", index=False, encoding="utf-8-sig")
    A.to_csv(OUT / "s13_累计分布量.csv", index=False, encoding="utf-8-sig")
    O.to_csv(OUT / "s13_承保年度叠加.csv", index=False, encoding="utf-8-sig")
    M.to_csv(OUT / "s13_UPR模式差异.csv", index=False, encoding="utf-8-sig")
    V.to_csv(OUT / "s13_保单年度验收.csv", index=False, encoding="utf-8-sig")

    with open(OUT / "s13_保单年度配置.json", "w", encoding="utf-8") as f:
        json.dump({
            "版本": "v1（B4 产物，2026-10-02）",
            "口径": {"情景": SCEN, "成本曲线": CALIBER, "起赔倍数κ": KAPPA,
                     "贴现率": DF_R, "路径数": N_PATH, "随机种子": SEED0},
            "越界年": BREACH_YEAR,
            "重定价周期": f"max(1, min({PERIOD_CAP}, 距越界年数−1))（§5.1）",
            "UPR赚取模式": "按风险模式（累计赔付占比曲线）分摊，非按年限线性；"
                          "线性法会低估早期 UPR、高估已赚保费",
            "IBNR口径": "指数挂钩 ⇒ 出险时点可由指数直接判定，IBNR 接近零",
            "受限计量": "2029 年及以后的赔付现值不得计提责任增量（A14），"
                        "须在越界判定当期一次性补提",
            "上游只读": ["s4_pricing.py", "s4_定价配置.json"],
        }, f, ensure_ascii=False, indent=2)

    print("\n[已写出] s13_保单年度视图.csv / s13_累计分布量.csv / s13_承保年度叠加.csv /"
          " s13_UPR模式差异.csv / s13_保单年度验收.csv / s13_保单年度配置.json")


if __name__ == "__main__":
    main()
