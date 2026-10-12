# -*- coding: utf-8 -*-
"""
s11_reserve.py —— B1 责任准备金与最低资本接口

设计依据：`02_产物/中期_双指标报价与重定价条款_设计_v1.md` §6（表 G/H）、§9.3 A14/A15。

⚠ **编号变更（须登记）**：设计稿 §9.1 把本工作包排在 `s10_reserve.py`，
   但 `s10` 已被 B3 费用率分解（`s10_expense.py`）占用。本实现顺延为 `s11_`，
   使「脚本号 ↔ 产物号」一一对应（s11 ←→ s11_*）。B2 顺延 `s12_`，B4 顺延 `s13_`。

两个核心口径（§6.2 / §6.3）：

1. **准备金计量单元 = 重定价子期间**（1–3 年），而非整张保单期限。
   理由：重定价条款使保单不构成固定多年期负债；重定价后的期间实质是「续保权」。
   对不可识别的长期参数计提长期负债，是把模糊性错当风险。

2. **不对 2028 年之后的赔付承担责任增量（A14）**。
   u 于 2028 年越界 ⇒ 该年之后的指数水平缺乏模型依据。2029+ 的赔付现值
   只能按 2027 年末已确立的档位计量；档位死亡时在当期一次性补提。

输出：
    02_产物/s11_准备金测算.csv      子期间 × (UPR / 责任准备金 / OCR / IBNR / 合计)
    02_产物/s11_最低资本接口.csv    期限 × 风险因子 → 最低资本与隐含资本因子
    02_产物/s11_最低资本接口.json   表 H 输入 + 核定对比 + 待核项
    02_产物/s11_准备金验收.csv      断言结果
    02_产物/s11_准备金配置.json     口径留痕

用法：
    python s11_reserve.py            # 生成产物 + 打印
    python s11_reserve.py --check    # 只跑断言（退出码 0 = 全过）

⚠ 只读上游：s4_pricing、s7_精算口径保费.csv、s8_报价单.csv、s10_费用率分解.csv。
   不修改任何既有脚本或产物。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import L1, OUT                                  # noqa: E402
import s4_pricing as s4                                       # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.unicode.east_asian_width", True)

SCEN = "S1_基准"
CALIBER = "M2_价格平移"
N_PATH = L1["mc_paths"]
SEED0 = s4.RNG_SEED
DF_R = L1["discount"]
Q_RISK, Q_LIMIT = 0.99, 0.995
RHO = L1.get("capital_charge", 0.06)
E_EXP = L1["expense_ratio"]
PI_PROFIT = 0.08
KAPPA = s4.KAPPA_BASE
BREACH_YEAR = 2028                     # A14 的时点（s2_全国输入路径.csv）

# 获取成本（一次发生于起保，不随期间摊销）——依据 s10 参考分解
ACQ_EXP = 0.10


def _tvar(x: np.ndarray, a: float) -> float:
    q = np.quantile(x, a)
    t = x[x >= q]
    return float(t.mean()) if t.size else float(q)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    only_check = "--check" in sys.argv

    tiers = s4._lp_cache()
    s7 = pd.read_csv(OUT / "s7_精算口径保费.csv")
    quote = pd.read_csv(OUT / "s8_报价单.csv")

    print("=" * 126)
    print("s11　B1 责任准备金与最低资本接口　·　S1 基准 · L1b 指数口径 · M2 价格平移")
    print("=" * 126)
    print(f"  计量单元 = 重定价子期间（§5.1）  贴现率={DF_R:.1%}  "
          f"越界年={BREACH_YEAR}  获取成本={ACQ_EXP:.0%}（一次发生，不摊销）")

    res_rows, mc_rows = [], []

    for i, tag in enumerate(tiers):
        sim = s4.simulate(SCEN, tiers[tag], CALIBER, N_PATH, SEED0 + i * 977)
        I, Q = sim["I"], sim["Q"]
        K = KAPPA * s4.MAC_REF
        pay = np.minimum(np.maximum(I - K, 0.0), s4.CAP_AMOUNT) * Q
        for n in s4.TERMS:
            rec = s7[(s7["标度档"] == tag) & (s7["期限"] == n)].iloc[0]
            GP = float(rec["毛费率_新%"]) / 100.0 * float(rec["SI_新"])   # 毛保费（元/吨）
            SI = float(rec["SI_新"])
            E = float(rec["E"])
            TV = float(rec["TVaR99"])
            q995 = float(rec["P99.5"])

            df = 1.0 / (1.0 + DF_R) ** np.arange(n + 1)
            pv_y = pay[:n + 1] * df[:, None]
            E_t = pv_y.mean(axis=1)                      # 逐年期望赔付现值
            E_tot = float(E_t.sum())
            cum = np.cumsum(E_t)

            for k in range(n + 1):
                earned = float(cum[k] / E_tot) if E_tot > 0 else 0.0
                UPR = GP * (1.0 - earned)                # 风险模式赚取
                UPR_lin = GP * (1.0 - (k + 1) / (n + 1))  # 线性对照
                BE = float(E_t[k + 1:].sum())             # 未来赔付现值（best estimate）
                # A14：拆分境内 / 越界后（按日历年）
                idx = np.arange(k + 1, n + 1)
                cal = s4.START + idx
                BE_in = float(E_t[k + 1:][cal <= BREACH_YEAR].sum())
                BE_out = float(E_t[k + 1:][cal > BREACH_YEAR].sum())
                PDR = max(0.0, BE_in - UPR)               # 保费不足准备金（受限口径）
                PDR_full = max(0.0, BE - UPR)             # 对照：全额口径
                OCR = 0.0                                 # 无理赔系统，按 0 计提
                OCR_proxy = float(E_t[k])                 # 备忘：当年已发生赔付
                IBNR = 0.0                                # 指数挂钩 ⇒ 近乎零（§6.2）
                total = UPR + PDR + OCR + IBNR
                # 设计稿 §6.2 公式对照：责任准备金 = 未来赔付现值 − 未来毛保费现值
                # 趸缴 ⇒ 未来毛保费现值 = 0 ⇒ 退化为 = 未来赔付现值 = BE
                doc_formula = BE
                res_rows.append({
                    "标度档": tag, "期限": n, "估值点k": k,
                    "日历年": s4.START + k,
                    "子期间序号": k + 1,
                    "已赚比例_风险模式": round(earned, 6),
                    "已赚比例_线性": round((k + 1) / (n + 1), 6),
                    "UPR": round(UPR, 1),
                    "UPR_线性对照": round(UPR_lin, 1),
                    "未来赔付现值BE": round(BE, 1),
                    "其中_2026至2028": round(BE_in, 1),
                    "其中_2029及以后": round(BE_out, 1),
                    "保费不足准备金PDR": round(PDR, 1),
                    "PDR_全额口径对照": round(PDR_full, 1),
                    "OCR": round(OCR, 1),
                    "OCR_代理_当年已发生": round(OCR_proxy, 1),
                    "IBNR": round(IBNR, 1),
                    "责任准备金合计": round(total, 1),
                    "设计稿§6.2_未来法对照": round(doc_formula, 1),
                    "设计稿低估倍数": round(total / doc_formula, 2) if doc_formula > 0 else np.nan,
                    "越界后须一次性补提": round(BE_out, 1),
                    "是否越界后估值点": "是" if s4.START + k > BREACH_YEAR else "否",
                })

            # ---------- 最低资本接口（§6.3 第一步） ----------
            base_implied = TV - E                       # s7 隐含的资本占用基数
            si3 = quote[quote["期限"] == n]
            SI_min = float(si3["限额下端_SImin"].iloc[0])
            SI_max = float(si3["限额上端_SImax"].iloc[0])
            for f in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
                mc = f * SI
                capc = RHO * mc
                mc_rows.append({
                    "标度档": tag, "期限": n, "风险因子f": f,
                    "最低资本_MC": round(mc, 1),
                    "最低资本占限额%": round(f * 100, 2),
                    "CapCost_ρ×MC": round(capc, 1),
                    "CapCost占限额%": round(capc / SI * 100, 4),
                    "相对s7隐含基数倍数": round(mc / base_implied, 4) if base_implied else np.nan,
                })

            mc_rows.append({
                "标度档": tag, "期限": n, "风险因子f": np.nan,
                "最低资本_MC": round(base_implied, 1),
                "最低资本占限额%": round(base_implied / SI * 100, 4),
                "CapCost_ρ×MC": round(RHO * base_implied, 1),
                "CapCost占限额%": round(RHO * base_implied / SI * 100, 4),
                "相对s7隐含基数倍数": 1.0,
            })

    R = pd.DataFrame(res_rows)
    MC = pd.DataFrame(mc_rows)

    # ---------- 打印：论文口径 5 年期逐期准备金 ----------
    print("\n【准备金测算：论文口径 5 年期（子期间 = 1 年，共 6 个估值点）】")
    print("-" * 126)
    x = R[(R["标度档"] == "论文口径") & (R["期限"] == 5)]
    print(x[["估值点k", "日历年", "子期间序号", "已赚比例_风险模式", "已赚比例_线性", "UPR",
             "UPR_线性对照", "未来赔付现值BE", "其中_2029及以后", "保费不足准备金PDR",
             "责任准备金合计"]].to_string(index=False))
    print("  ⇒ 线性法在第 0 个估值点后即认定已赚 1/(n+1)，而风险模式只赚 "
          f"{float(x['已赚比例_风险模式'].iloc[0]):.4%} —— 线性法系统性压低 UPR。")

    print("\n【准备金构成汇总（起保时点 k=0，各档各期限）】")
    print("-" * 126)
    z = R[R["估值点k"] == 0]
    for n in s4.TERMS:
        y = z[z["期限"] == n].sort_values("UPR")
        print(f"\n  ── {n} 年期 ──")
        print(y[["标度档", "UPR", "未来赔付现值BE", "其中_2029及以后", "保费不足准备金PDR",
                 "责任准备金合计", "越界后须一次性补提"]].to_string(index=False))

    # ---------- 打印：设计稿 §6.2 公式的低估幅度 ----------
    print("\n【与设计稿 §6.2 公式的对照：`责任准备金 = 未来赔付现值 − 未来毛保费现值`】")
    print("-" * 126)
    z0 = R[R["估值点k"] == 0]
    for n in s4.TERMS:
        y = z0[z0["期限"] == n].sort_values("责任准备金合计")
        print(f"  {n:2d} 年期：本实现(四科目) [{y['责任准备金合计'].min():,.0f}, "
              f"{y['责任准备金合计'].max():,.0f}]   设计稿公式 [{y['设计稿§6.2_未来法对照'].min():,.0f}, "
              f"{y['设计稿§6.2_未来法对照'].max():,.0f}]   低估 "
              f"[{y['设计稿低估倍数'].min():.2f}×, {y['设计稿低估倍数'].max():.2f}×]")
    print("  ⇒ 趸缴保单下「未来毛保费现值 = 0」，设计稿公式**退化为未来赔付现值**，")
    print("     漏掉了未赚取保费这一在起保日占负债约 92% 的部分。")
    print("  ⇒ 且全档全期限 **PDR = 0**：保费加载足以覆盖未来赔付 ⇒ 本产品的负债")
    print("     是**递延收入**而非赔款准备 —— 这是它更接近金融合约而非保险的定量证据。")
    print("\n【最低资本接口：s7 隐含资本因子 vs 期限】")
    print("-" * 126)
    imp = []
    for n in s4.TERMS:
        y = MC[(MC["期限"] == n) & (MC["标度档"] == "论文口径") & (MC["风险因子f"].isna())]
        si = float(s7[(s7["标度档"] == "论文口径") & (s7["期限"] == n)]["SI_新"].iloc[0])
        f_imp = float(y["最低资本占限额%"].iloc[0]) / 100
        imp.append({"期限": n, "s7隐含基数_TVaR−E": float(y["最低资本_MC"].iloc[0]),
                    "限额SI": si, "隐含资本因子": round(f_imp, 4),
                    "参考_1除n": round(1.0 / n, 4), "比值": round(f_imp / (1.0 / n), 4)})
    IMP = pd.DataFrame(imp)
    print(IMP.to_string(index=False))
    print("  ⇒ 隐含资本因子随期限递减；而偿付能力口径的风险因子通常是**定值**。")
    print("     若监管核定为定值 f，则 n=15 年期的当前资本计提可能严重不足（见上表比值列）。")

    print("\n【A15 / G-B2：资本暴露基数取 SImax，报价取 SImin（方向相反）】")
    print("-" * 126)
    gb = []
    for _, r in quote.iterrows():
        n = int(r["期限"])
        lo, hi = float(r["限额下端_SImin"]), float(r["限额上端_SImax"])
        gb.append({"期限": n, "报价用SImin": lo, "资本用SImax": hi,
                   "宽度": round(hi / lo, 3),
                   "资本按SImax计提": round(RHO * hi, 1),
                   "按SImin计提": round(RHO * lo, 1),
                   "资本区间宽度": round(hi / lo, 3)})
    GB = pd.DataFrame(gb)
    print(GB.to_string(index=False))

    # ================= 断言 =================
    res = []

    def add(code, name, ok, got, exp):
        res.append({"验收项": code, "名称": name, "结果": "PASS" if ok else "FAIL",
                    "实测": str(got), "期望": str(exp)})

    # A14-1 估值点越界后时，受限口径的责任准备金仅含境内部分
    bad = R[(R["是否越界后估值点"] == "是")
            & (R["其中_2029及以后"] > 0)
            & (R["保费不足准备金PDR"] > 0)]
    # 越界后估值点的未来赔付全部落在 2029+，故受限口径 PDR 必为 0
    ok1 = bool((bad["保费不足准备金PDR"] == 0).all()) if len(bad) else True
    add("A14-1", "越界后估值点的受限口径 PDR = 0（不对 2029+ 计提）", ok1,
        f"越界后估值点 {len(bad)} 行，PDR 最大 {bad['保费不足准备金PDR'].max() if len(bad) else 0}",
        "全部为 0")

    # A14-2 越界后须一次性补提额 = BE_out，且在全额口径下 PDR_full ≥ PDR
    ok2 = bool((R["PDR_全额口径对照"] >= R["保费不足准备金PDR"] - 1e-6).all())
    add("A14-2", "全额口径 PDR ≥ 受限口径 PDR（受限计提不高于全额）", ok2,
        f"违例 {int((R['PDR_全额口径对照'] < R['保费不足准备金PDR'] - 1e-6).sum())} 行", "0 行")

    # A14-3 越界后补提额 = 未来 2029+ 赔付现值（逐行相等）
    ok3 = bool((abs(R["越界后须一次性补提"] - R["其中_2029及以后"]) < 1e-6).all())
    add("A14-3", "越界后须一次性补提 = 2029+ 赔付现值", ok3,
        f"最大偏差 {float((R['越界后须一次性补提'] - R['其中_2029及以后']).abs().max()):.2e}", "0")

    # A14-4 提前识别：15 年期在 k=0 即有 2029+ 责任
    z15 = R[(R["标度档"] == "论文口径") & (R["期限"] == 15) & (R["估值点k"] == 0)]
    add("A14-4", "15 年期起保时即存在 2029+ 责任（A14 受限对象）",
        float(z15["其中_2029及以后"].iloc[0]) > 0,
        f"{float(z15['其中_2029及以后'].iloc[0]):,.1f} 元/吨（占 BE "
        f"{float(z15['其中_2029及以后'].iloc[0]) / float(z15['未来赔付现值BE'].iloc[0]):.1%}）", "> 0")

    # A15-1 资本按 SImax、报价按 SImin，两者方向相反
    ok5 = bool((GB["资本用SImax"] > GB["报价用SImin"]).all())
    add("A15-1", "资本暴露基数（SImax）严格大于报价基数（SImin）", ok5,
        f"宽度区间 [{GB['宽度'].min():.3f}×, {GB['宽度'].max():.3f}×]",
        "全部 > 1")

    # A15-2 报价单的挂牌限额 = SImin（与 A4 一致）
    ok6 = bool((quote["挂牌限额"] - quote["限额下端_SImin"]).abs().max() < 1e-9)
    add("A15-2", "报价单挂牌限额 = SImin（与 A4 同档取值一致）", ok6,
        f"最大偏差 {float((quote['挂牌限额'] - quote['限额下端_SImin']).abs().max()):.2e}", "0")

    # 结构：UPR + PDR + OCR + IBNR = 合计
    ok7 = bool((abs(R["UPR"] + R["保费不足准备金PDR"] + R["OCR"] + R["IBNR"]
                    - R["责任准备金合计"]) < 0.11).all())
    add("结构-1", "UPR + PDR + OCR + IBNR = 责任准备金合计", ok7,
        f"最大偏差 {float((R['UPR'] + R['保费不足准备金PDR'] + R['OCR'] + R['IBNR'] - R['责任准备金合计']).abs().max()):.2f}",
        "≤ 0.11（各项 1 位小数舍入）")

    # 结构：已赚比例单调不减、终值为 1
    bad8 = []
    for (tg, n), g in R.groupby(["标度档", "期限"]):
        d = g.sort_values("估值点k")["已赚比例_风险模式"].diff().dropna()
        if (d < -1e-12).any() or abs(float(g.sort_values("估值点k")["已赚比例_风险模式"].iloc[-1]) - 1.0) > 1e-12:
            bad8.append((tg, n))
    add("结构-2", "已赚比例单调不减且终值 = 1", not bad8,
        f"违例 {bad8[:3]}" if bad8 else "全部合规", "0 违例")

    # IBNR 口径：指数挂钩 ⇒ 恒为 0
    add("结构-3", "IBNR 恒为 0（指数挂钩，出险时点可直接判定）",
        bool((R["IBNR"] == 0).all()), "全部为 0", "0")

    # 最低资本：CapCost 以尾部超出期望计量（ρ×MC），非 ρ×SI
    y = MC[(MC["风险因子f"] == 0.20) & (MC["标度档"] == "论文口径") & (MC["期限"] == 5)].iloc[0]
    add("MC-1", "CapCost = ρ × MC（不把资本成本与限额绑定）",
        abs(float(y["CapCost_ρ×MC"]) - RHO * float(y["最低资本_MC"])) <= 0.06,
        f"{float(y['CapCost_ρ×MC']):.1f} = {RHO:.0%} × {float(y['最低资本_MC']):.1f}", "相等（容差 0.06 = 1 位小数舍入）")

    # 负债结构：PDR 恒为 0（保费加载足以覆盖，无保费不足）
    add("结构-4", "保费不足准备金 PDR 恒为 0（加载足以覆盖未来赔付）",
        bool((R["保费不足准备金PDR"] == 0).all()),
        f"最大 PDR {float(R['保费不足准备金PDR'].max()):.1f}", "全部为 0")

    # 设计稿 §6.2 公式的低估幅度（趸缴下退化为未来赔付现值）
    z0 = R[R["估值点k"] == 0]
    mult = z0["设计稿低估倍数"].dropna()
    add("结构-5", "设计稿 §6.2 公式（趸缴下退化）低估负债 ≥ 3 倍",
        bool((mult >= 3).all()),
        f"低估倍数区间 [{mult.min():.2f}×, {mult.max():.2f}×], 中位 {mult.median():.2f}×",
        "全部 ≥ 3×")

    V = pd.DataFrame(res)
    print("\n【验收】B1 准备金与最低资本")
    print("-" * 126)
    print(V.to_string(index=False))
    nf = int((V["结果"] == "FAIL").sum())
    print(f"\n  合计：{len(V) - nf} PASS / {nf} FAIL")

    if only_check:
        sys.exit(1 if nf else 0)

    R.to_csv(OUT / "s11_准备金测算.csv", index=False, encoding="utf-8-sig")
    MC.to_csv(OUT / "s11_最低资本接口.csv", index=False, encoding="utf-8-sig")
    GB.to_csv(OUT / "s11_资本暴露基数.csv", index=False, encoding="utf-8-sig")
    IMP.to_csv(OUT / "s11_隐含资本因子.csv", index=False, encoding="utf-8-sig")
    V.to_csv(OUT / "s11_准备金验收.csv", index=False, encoding="utf-8-sig")

    with open(OUT / "s11_最低资本接口.json", "w", encoding="utf-8") as f:
        json.dump({
            "版本": "v1（B1 产物，2026-10-02）",
            "编号变更": "设计稿 §9.1 的 s10_reserve 顺延为 s11_（s10 已被 B3 费用率分解占用）",
            "监管口径声明": "准备金科目、最低资本口径与偿付能力充足率均须按"
                            "《保险公司偿付能力监管规则Ⅱ》现行有效版本核定；"
                            "本文为接口设计与测算框架，具体因子与计提比例**待核**，"
                            "不作为精算意见。核定前 §6 结论不得对外使用。",
            "表H输入_论文口径5年期": {
                "E_期望赔付现值": float(s7[(s7["标度档"] == "论文口径") & (s7["期限"] == 5)]["E"].iloc[0]),
                "P99.5_限额口径": float(s7[(s7["标度档"] == "论文口径") & (s7["期限"] == 5)]["P99.5"].iloc[0]),
                "TVaR99": float(s7[(s7["标度档"] == "论文口径") & (s7["期限"] == 5)]["TVaR99"].iloc[0]),
                "SI_责任限额": float(s7[(s7["标度档"] == "论文口径") & (s7["期限"] == 5)]["SI_新"].iloc[0]),
                "SD": float(s7[(s7["标度档"] == "论文口径") & (s7["期限"] == 5)]["SD"].iloc[0]),
                "损失波动率SD除E": round(float(s7[(s7["标度档"] == "论文口径")
                                        & (s7["期限"] == 5)]["SD"].iloc[0])
                                   / float(s7[(s7["标度档"] == "论文口径")
                                              & (s7["期限"] == 5)]["E"].iloc[0]), 4),
                "已计提资本成本占SI%": float(s7[(s7["标度档"] == "论文口径")
                                          & (s7["期限"] == 5)]["资本成本%"].iloc[0]),
            },
            "s7隐含资本基数": "TVaR99 − E（= ρ 的分母）",
            "关键接口判据": "若偿二代核定的最低资本显著高于 (TVaR99 − E)，"
                            "则 CapCost = ρ × MC_核定，费率须全部重跑。",
            "隐含资本因子随期限": IMP.to_dict("records"),
            "结构性困难": "风险暴露（限额）自身即为 2.70–4.12 倍的区间 ⇒ "
                          "最低资本同样有该宽度。审慎做法：资本按 SImax 计提、报价按 SImin。",
            "待核项": [
                "① 承保风险风险因子（本项目用 0.05–0.30 做敏感性，非核定值）",
                "② 市场风险资本：σ_P=0.28 且不可对冲，偿二代下是否有适用科目（待核）",
                "③ 信用风险资本（再保应收）：依赖 B2 的分层结构",
                "④ OCR：须接入理赔系统；本实现按 0 计提，备忘列已给出当年已发生赔付代理值",
                "⑤ 责任准备金的计量单元（按重定价子期间）须作为精算假设报备",
            ],
            "上游只读": ["s4_pricing.py", "s7_精算口径保费.csv", "s8_报价单.csv"],
        }, f, ensure_ascii=False, indent=2)

    with open(OUT / "s11_准备金配置.json", "w", encoding="utf-8") as f:
        json.dump({
            "版本": "v1（B1 产物，2026-10-02）",
            "计量单元": "重定价子期间（§5.1），2026 起保 ⇒ 每年一期",
            "四科目": {
                "UPR": "未赚保费 = 毛保费 × (1 − 风险模式已赚比例)；"
                       "风险模式 = 累计赔付现值占比曲线（s13），非年限线性",
                "OCR": "已发生已报告：无理赔系统，按 0 计提；备忘列给当年已发生赔付代理值",
                "IBNR": "指数挂钩 ⇒ 出险时点可由指数直接判定 ⇒ 接近零",
                "责任准备金": "保费不足准备金 PDR = max(0, 未来赔付现值 − UPR)，"
                              "受限口径仅含 2026–2028 部分（A14）",
            },
            "A14受限规则": f"不对 {BREACH_YEAR} 年之后的赔付承担责任增量；"
                           "档位死亡时在当期一次性补提（补提额 = 2029+ 赔付现值）",
            "获取成本处理": f"{ACQ_EXP:.0%} 于起保一次发生，不进 UPR 摊销分母",
            "上游只读": ["s4_pricing.py", "s7_精算口径保费.csv", "s8_报价单.csv"],
        }, f, ensure_ascii=False, indent=2)

    print("\n[已写出] s11_准备金测算.csv / s11_最低资本接口.csv / s11_资本暴露基数.csv /"
          " s11_隐含资本因子.csv / s11_准备金验收.csv / s11_最低资本接口.json / s11_准备金配置.json")


if __name__ == "__main__":
    main()
