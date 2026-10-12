# -*- coding: utf-8 -*-
"""
s8_quote_sheet.py —— A1「费率 × 限额」双指标报价单生成

上游：# 只读
    02_产物/s7_精算口径保费.csv         （精算口径重建，SI=责任上限）
    02_产物/s4_口径对比.csv             （L1a 市价口径对照）
    02_产物/s3_定价锚.json              （平减/缺口倍数口径留痕）
下游产物：
    02_产物/s8_报价单.csv               （期限 × 双指标区间 + 审慎取值 + 保费区间）
    02_产物/s8_报价单_分档明细.csv      （期限 × 9 标度档的逐档保费）
    02_产物/s8_审慎取值规则.json        （R1–R10 规则表 + L1a 对照 + 自检结果）

设计依据：`02_产物/中期_双指标报价与重定价条款_设计_v1.md` §1.2、§2.2、§9.3。
本地实现口径见文件末「与设计稿的三处偏离」。

⚠ 本脚本**不重算**任何精算量，只对 s7 的落盘结果做「同档成对」汇总与规则化。
   这是刻意的：报价单必须与可复现的精算产物逐位对齐，不得在报价层引入新参数。

用法：
    python s8_quote_sheet.py            # 生成产物
    python s8_quote_sheet.py --check    # 仅跑 A1–A5 验收断言，退出码 0 = 通过
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import OUT                                    # noqa: E402

pd.set_option("display.width", 240)
pd.set_option("display.unicode.east_asian_width", True)

TERMS = [3, 5, 10, 15]
# 9 个标度档（顺序 = s7 产物中的出现顺序，即 s4._lp_cache 的顺序）
TIER_ORDER = [
    "段1左端_最保守", "段1偏左", "段1中段", "段1偏右", "论文口径",
    "段1右端_最宽松", "段2中点", "β2.0_上确界", "β0.6_下确界",
]
# 定价参数集（条款 §5.3 七项；本脚本只做「冻结留痕」，不修改）
FROZEN_PARAMS = {
    "κ_起赔倍数": 1.30,
    "ρ_资本成本率": 0.06,
    "e_费用率": 0.25,
    "π_利润边际": 0.08,
    "贴现率": 0.025,
    "随机过程参数": {
        "σ_P": 0.28, "σ_m_路径": 0.15, "σ_m_水平": 0.12, "σ_Q": 0.25,
        "ρ_Pm": 0.45, "ρ_PQ": -0.40, "ρ_mQ": 0.30, "路径数": 20000,
    },
    "合法域求法": "dev(γ₁,β;K,L,Y,B) ≤ 1（归一化约束形式、约束水平、求解精度三项均冻结）",
    "网格取点规则": "9 点 = 两段端点 + 段内分位 + β 边界；重定价时不得优化取点",
}
SAFETY = {  # 挂牌值约定
    "分位_限额": 0.995, "分位_风险资本": 0.99,
    "费率区间取整": "四舍五入到 0.5%",
}


# ----------------------------------------------------------------------
# 载入
# ----------------------------------------------------------------------
def load() -> tuple[pd.DataFrame, list[str]]:
    T = pd.read_csv(OUT / "s7_精算口径保费.csv")
    need = {"标度档", "期限", "E", "P99.5", "TVaR99", "SI_新", "毛费率_新%"}
    miss = need - set(T.columns)
    if miss:
        raise SystemExit(f"[FATAL] s7_精算口径保费.csv 缺列：{sorted(miss)}")
    tiers = [t for t in TIER_ORDER if t in set(T["标度档"])]
    extra = sorted(set(T["标度档"]) - set(tiers))
    if extra:
        raise SystemExit(f"[FATAL] 出现未登记的标度档：{extra}（网格规则被改动？）")
    return T, tiers


# ----------------------------------------------------------------------
# 汇总（核心：同档成对，禁止混档）
# ----------------------------------------------------------------------
def build(T: pd.DataFrame, tiers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows, quote_rows = [], []
    for n in TERMS:
        x = T[T["期限"] == n].set_index("标度档").loc[tiers].copy()
        # 逐档保费 —— 唯一诚实的「保费」定义：同一档的 SI 与其自身费率相乘
        x["保费_元每吨"] = x["SI_新"] * x["毛费率_新%"] / 100.0

        j_lo = str(x["SI_新"].idxmin())     # 审慎档：限额下端所在档
        j_hi = str(x["SI_新"].idxmax())
        p_lo = str(x["保费_元每吨"].idxmin())
        p_hi = str(x["保费_元每吨"].idxmax())

        for t in tiers:
            r = x.loc[t]
            detail_rows.append({
                "期限": n, "标度档": t,
                "SI_新": r["SI_新"], "毛费率_新%": r["毛费率_新%"],
                "保费_元每吨": round(r["保费_元每吨"], 0),
                "E": r["E"], "P99.5": r["P99.5"], "TVaR99": r["TVaR99"],
                "纯风险%": r["纯风险%"], "风险附加%": r["风险附加%"], "资本成本%": r["资本成本%"],
                "是审慎档": t == j_lo, "是保费最小档": t == p_lo, "是保费最大档": t == p_hi,
            })

        si_lo, si_hi = x["SI_新"].min(), x["SI_新"].max()
        rt_lo, rt_hi = x["毛费率_新%"].min(), x["毛费率_新%"].max()
        pr_lo, pr_hi = x["保费_元每吨"].min(), x["保费_元每吨"].max()

        quote_rows.append({
            "期限": n,
            "限额区间_元每吨": f"{si_lo:,.0f} – {si_hi:,.0f}",
            "限额下端_SImin": si_lo, "限额上端_SImax": si_hi,
            "限额宽度": round(si_hi / si_lo, 3),
            "费率区间_%": f"{rt_lo:.2f} – {rt_hi:.2f}",
            "费率下端_rate_min": rt_lo, "费率上端_rate_max": rt_hi,
            "费率宽度": round(rt_hi / rt_lo, 3),
            "挂牌费率区间_取整": f"{round(rt_lo*2)/2*1:.1f}% – {round(rt_hi*2)/2*1:.1f}%",
            # —— 审慎取值：取 SImin 所在档的**全部指标**（A4：不得混档）
            "审慎档": j_lo,
            "挂牌限额": si_lo,
            "挂牌费率%": round(x.loc[j_lo, "毛费率_新%"], 2),
            "挂牌保费_元每吨": round(x.loc[j_lo, "保费_元每吨"], 0),
            # —— 保费上下界：对全部 9 档取 min/max（不是 SI端点 × 费率端点）
            "保费下端档": p_lo, "保费上端档": p_hi,
            "保费下界_元每吨": round(pr_lo, 0), "保费上界_元每吨": round(pr_hi, 0),
            "保费宽度": round(pr_hi / pr_lo, 3),
            "保费端点是否同档": (p_lo == j_lo) and (p_hi == j_hi),
            "角点A_最小责任": round(si_lo * x.loc[j_lo, "毛费率_新%"] / 100.0, 0),
            "角点B_最大责任": round(si_hi * x.loc[j_hi, "毛费率_新%"] / 100.0, 0),
        })
    return pd.DataFrame(quote_rows), pd.DataFrame(detail_rows)


# ----------------------------------------------------------------------
# 验收断言（设计稿 §9.3 A1–A5）
# ----------------------------------------------------------------------
def check(Q: pd.DataFrame, D: pd.DataFrame, T: pd.DataFrame) -> list[dict]:
    res = []

    def add(tag, desc, ok, got, exp):
        res.append({"验收项": tag, "判据": desc, "结果": "PASS" if ok else "FAIL",
                    "实测": got, "期望": exp})

    # A1 双指标齐备性：报价单每一行费率与限额成对出现，无单独出现的费率
    paired = all(
        isinstance(r["限额区间_元每吨"], str) and "–" in r["限额区间_元每吨"]
        and isinstance(r["费率区间_%"], str) and "–" in r["费率区间_%"]
        for _, r in Q.iterrows()
    )
    add("A1", "报价单每行「限额区间」与「费率区间」成对出现", paired,
        f"{len(Q)}/{len(Q)} 行成对", "100%")

    # A2 费率宽度：5 年期跨档 ≤ 1.20×
    w5 = float(Q.loc[Q["期限"] == 5, "费率宽度"].iloc[0])
    add("A2", "5 年期跨档费率宽度 ≤ 1.20×", w5 <= 1.20, f"{w5:.3f}×", "≤ 1.200×")

    # A3 限额宽度披露：报价单含限额宽度列且等于 max/min（列值已四舍五入到 3 位）
    ok3 = "限额宽度" in Q.columns and all(
        abs(r["限额宽度"] - r["限额上端_SImax"] / r["限额下端_SImin"]) < 5e-4
        for _, r in Q.iterrows()
    )
    add("A3", "限额宽度列存在且 = SImax/SImin", ok3,
        " / ".join(f"{int(r['期限'])}y {r['限额宽度']:.3f}×" for _, r in Q.iterrows()), "逐行自洽")

    # A4 审慎取值：挂牌限额 = SImin，且挂牌费率取自**同一档**
    bad4 = []
    for _, r in Q.iterrows():
        n, j = int(r["期限"]), r["审慎档"]
        row = D[(D["期限"] == n) & (D["标度档"] == j)].iloc[0]
        if not (abs(r["挂牌限额"] - r["限额下端_SImin"]) < 1e-6
                and abs(r["挂牌费率%"] - row["毛费率_新%"]) < 1e-9
                and abs(r["挂牌保费_元每吨"] - row["保费_元每吨"]) < 1.0):
            bad4.append(n)
    add("A4", "挂牌取 SImin 且费率/保费取自同一档（不得混档）", not bad4,
        "全部同档" if not bad4 else f"混档期限={bad4}", "无混档")

    # A5 保费上下界：= 全 9 档 min/max（并检验它是否等于 SI端点×费率端点）
    bad5, mixed = [], []
    for _, r in Q.iterrows():
        n = int(r["期限"])
        x = D[D["期限"] == n]
        if not (abs(r["保费下界_元每吨"] - x["保费_元每吨"].min()) < 1.0
                and abs(r["保费上界_元每吨"] - x["保费_元每吨"].max()) < 1.0):
            bad5.append(n)
        # 混档检验：SI端点 × 费率端点 是否与本档真实保费一致
        a = r["限额下端_SImin"] * r["费率下端_rate_min"] / 100.0
        b = r["限额上端_SImax"] * r["费率上端_rate_max"] / 100.0
        if abs(a - r["保费下界_元每吨"]) > 1.0 or abs(b - r["保费上界_元每吨"]) > 1.0:
            mixed.append(n)
    add("A5", "保费下/上界 = 全 9 档 min/max 逐档保费", not bad5,
        " + ".join(f"{int(r['期限'])}y [{r['保费下界_元每吨']:.0f}, {r['保费上界_元每吨']:.0f}]"
                   for _, r in Q.iterrows()), "逐档 min/max")
    add("A5b", "（反向）保费端点 ≠ SI端点×费率端点 ⇒ 端点跨档会造出不存在的价",
        True, f"混档期限 = {mixed if mixed else '无'}", "登记差异")

    # 补：全档保费对限额的**近似**单调性。
    # 注意：本产品费率与风险量方向相反（设计稿 §1.1 第 2 条），故保费**不严格**随
    # SI 单调；此处不用硬判据，改为登记秩相关与反转点，避免把真实性质误判为缺陷。
    for n in TERMS:
        x = D[D["期限"] == n].sort_values("SI_新").reset_index(drop=True)
        rho = float(pd.Series(x["SI_新"]).corr(pd.Series(x["保费_元每吨"]), method="spearman"))
        inv = []
        for i in range(len(x) - 1):
            if x.loc[i + 1, "保费_元每吨"] < x.loc[i, "保费_元每吨"]:
                inv.append(f"{x.loc[i,'标度档']}→{x.loc[i+1,'标度档']}"
                           f"（SI {x.loc[i,'SI_新']:.0f}→{x.loc[i+1,'SI_新']:.0f}，"
                           f"保费 {x.loc[i,'保费_元每吨']:.0f}→{x.loc[i+1,'保费_元每吨']:.0f}）")
        res.append({"验收项": f"M{n}", "判据": f"{n} 年期保费对限额的秩相关（登记，非硬判据）",
                    "结果": "PASS" if rho >= 0.95 else "FAIL",
                    "实测": f"Spearman ρ={rho:.4f}；反转 {len(inv)} 处" +
                            ("：" + "；".join(inv) if inv else ""),
                    "期望": "ρ ≥ 0.95（不要求严格单调）"})
    return res


# ----------------------------------------------------------------------
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    only_check = "--check" in sys.argv
    T, tiers = load()
    Q, D = build(T, tiers)
    res = check(Q, D, T)

    print("=" * 96)
    print("s8　A1「费率 × 限额」双指标报价单")
    print("=" * 96)

    print("\n【表 A】L1b 指数口径 · 挂牌报价单（S1 基准 · κ=1.30 · M2 价格平移）")
    print("-" * 96)
    print(Q[["期限", "限额区间_元每吨", "限额宽度", "费率区间_%", "费率宽度",
             "审慎档", "挂牌限额", "挂牌费率%", "挂牌保费_元每吨"]].to_string(index=False))

    print("\n【表 E′】审慎取值与保费上下界（同档成对，非端点相乘）")
    print("-" * 96)
    print(Q[["期限", "审慎档", "挂牌限额", "挂牌费率%", "挂牌保费_元每吨",
             "保费下端档", "保费上界_元每吨", "保费宽度"]].to_string(index=False))

    print("\n【表 E″】双端点角点（角点 A/B 均为**同档**乘积）")
    print("-" * 96)
    print(Q[["期限", "角点A_最小责任", "角点B_最大责任", "保费下界_元每吨",
             "保费上界_元每吨", "保费宽度"]].to_string(index=False))

    print("\n【验收】A1–A5 + 单调性")
    print("-" * 96)
    for r in res:
        print(f"  [{r['结果']:4s}] {r['验收项']:4s} {r['判据']}　→ {r['实测']}")

    nfail = sum(1 for r in res if r["结果"] == "FAIL")
    print(f"\n  合计：{len(res) - nfail} PASS / {nfail} FAIL")

    if only_check:
        sys.exit(1 if nfail else 0)

    Q.to_csv(OUT / "s8_报价单.csv", index=False, encoding="utf-8-sig")
    D.to_csv(OUT / "s8_报价单_分档明细.csv", index=False, encoding="utf-8-sig")

    # L1a 对照（表 B）
    l1a_note = None
    try:
        C = pd.read_csv(OUT / "s4_口径对比.csv")
        c = C[C["情景"] == "S1_基准"]
        l1a_note = {
            "口径": "L1a 市价口径（挂钩可观测碳价 P，非指数口径）",
            "期限": 5,
            "费率区间%": [round(float(c["L1a_市价口径"].min()), 2),
                          round(float(c["L1a_市价口径"].max()), 2)],
            "跨档宽度": round(float(c["L1a_市价口径"].max() / c["L1a_市价口径"].min()), 3),
            "说明": "跨 9 档近乎恒定（因挂钩可观测 P）；L1b 的模糊性在 L1a 下不出现。"
                    "L1a 作为「可复制性对照」列示，不作挂牌主推。",
        }
    except Exception as e:                                   # noqa: BLE001
        l1a_note = {"错误": f"未能读取 s4_口径对比.csv：{e}"}

    with open(OUT / "s8_审慎取值规则.json", "w", encoding="utf-8") as f:
        json.dump({
            "版本": "v1（A1 产物，2026-10-02）",
            "上游": "s7_精算口径保费.csv（SI=quantile_99.5(累计赔付现值)×n）",
            "口径声明": "本文所有费率为 S1 基准、κ=1.30、L1b 指数口径、M2 价格平移下的"
                        "毛费率（占限额%）；限额单位为元/吨（保单全期累计）。"
                        "γ₁ 与 β 为不可识别的标度约定自由度，属模糊性而非风险，"
                        "无概率分布，全篇不对档位做加权/期望/分位数运算。",
            "规则表_R1_R10": {
                "R1_限额审慎方向": "取下限 SI_min（= 9 档中最小的 quantile_99.5×n）",
                "R2_费率取值": "取 SI_min 所在档**该档自身**的费率（不得跨档取 rate_min）",
                "R3_挂牌口径": "挂牌为 (SI_min 档的 SI, rate) 单点 + 全档双指标区间披露",
                "R4_跨档比较禁令": "禁止跨标度档对费率做除法/排序；跨档比较须先统一 SI 口径",
                "R5_情景取值": "挂牌取 S1 基准；S2/S3 并列列示，不取 S2 作挂牌",
                "R6_口径取值": "挂牌主推 L1b；L1a 作可复制性对照",
                "R7_κ取值": "κ=1.30 固定，不随档位调整（调 κ 改变虚实值状态=换产品）",
                "R8_修订触发线": "任一省级 K 修订 > 2% ⇒ 合法域重算，挂牌暂停",
                "R9_u余量触发线": "u_headroom_pct < 0.5% ⇒ 触发重估",
                "R10_有效期": "当期挂牌有效期：2026 起保取 1 年（=min(3, 距越界年数−1)）；"
                              "一般不超过 3 年",
            },
            "定价参数集_冻结留痕": FROZEN_PARAMS,
            "挂牌值约定": SAFETY,
            "L1a对照": l1a_note,
            "验收结果": res,
        }, f, ensure_ascii=False, indent=2)

    print("\n[已写出] s8_报价单.csv / s8_报价单_分档明细.csv / s8_审慎取值规则.json")


# ----------------------------------------------------------------------
# 与设计稿的三处偏离（须在汇报中显式说明，不得静默）
# ----------------------------------------------------------------------
# 1) 审慎取值「R2」：设计稿 §2.2 表 D 的 R2 写「取 rate_min」，并在 §2.2 表 E 给出
#    3 年 (6,495, 52.70%)、5 年 (22,123, 31.65%) 等组合。但 3 年期的 rate_min=52.70%
#    属于「段1左端_最保守」档，而 SI_min=6,495 属于「β0.6_下确界」档 —— 二者**不同档**。
#    设计稿 §2.2 注文称该组合「对应 β0.6_下确界 档」，与表 E 的实际数字不符。
#    ⇒ 本脚本按 A4「不得混档」执行：R2 改为「取 SI_min 档自身费率」。
#    后果：5 年期挂牌保费由设计稿的 7,002 修正为 7,703（β0.6 档真实同档保费）。
# 2) 保费上下界：设计稿 §2.2 的双端点表用「SI 端点 × 费率端点」构造，得出
#    5 年 25,972 / 10 年 65,979 / 15 年 116,164。经复核，这三个上界**都不是任何
#    合法档位的真实保费**（混档产物），真实上界应取全 9 档保费的最大值：
#    β2.0_上确界档 → 5 年 26,112 / 10 年 65,978 / 15 年 116,714。
#    3 年期设计稿未给端点，本脚本补全为 [3,834, 14,186]（宽 3.700×）。
# 3) 保费区间 ≠ 限额区间 × 费率区间：因各档 (SI, rate) 不同向，端点档不同源。
#    本脚本对全部 9 档逐一算保费再取 min/max，并登记「端点是否同档」（列
#    `保费端点是否同档`）。设计稿未做此检验。
#    上述三处均属**设计稿的内部不一致**，不是引擎错误；引擎（s4/s7）未被修改。


if __name__ == "__main__":
    main()
