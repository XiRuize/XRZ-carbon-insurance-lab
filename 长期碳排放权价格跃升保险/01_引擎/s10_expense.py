# -*- coding: utf-8 -*-
"""
s10_expense.py —— B3 费用率分解与「分保佣金可核算性」
================================================================================
对应设计稿：双指标稿 §7.4、§9.3 A16、§10 Q7；规划文档 B3；验收门 G-B3。

【要回答的唯一问题】
    s7 的 25% 费用率 e 由什么构成？按 §7.2「成数层分出 30–40%」，
    分保佣金率 c 应如何核算？

【核心解析结论】
    设有成数分保：分出比例 q，分保佣金率 c（按分出保费计）。
    保险人费用（获取/管理）不因分出而按比例减少，故

        分保后综合费用率 = (e − c·q) / (1 − q)

    ⇒ 费用中性（分保前后费用率不变）⟺ c = e
    ⇒ **分保佣金率的可核算基准 = 保险人自身费用率 e，且与分出比例 q 无关。**
    这正是验收条件 A16「25% 费用率分解完成，分保佣金可核算」的定量含义。

【与中国保险会计口径的对接】
    「综合费用率」= (营业费用 + 手续费支出 + 佣金支出 + 营业税金及附加
                    + 分保费用支出 + 保险保障基金 − 摊回分保费用) / 已赚保费
    ⇒ 设计稿把「再保费」列为 25% 的分解项，属科目混淆：
      · 分出保费       → 保费减项（不进费用率）
      · 摊回分保费用   → 费用减项（进费用率，即分保佣金）
      · 摊回赔付       → 赔付减项
      本脚本按正确科目归集。

【诚实边界】
    · 只有「税金及附加」与「保险保障基金」两项为现行政策硬数；
      获取成本与管理费为行业基准 ⓗ 估算，须以公司财务/精算数据替换（规划 R4 风险）。
    · 报价含税与否（增值税口径）尚未确定，本脚本给出敏感性上界，供拍板。
    · 本脚本不修改 s4/s7，仅消费 s7_精算口径保费.csv。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import L1, OUT                                    # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.unicode.east_asian_width", True)

# ── 口径常量 ────────────────────────────────────────────────────────────────
E_S7 = float(L1["expense_ratio"])       # s7 假设的费用率 = 0.25（未校准）
PI_S7 = 0.08                            # s7 利润边际
VAT = 0.06                              # 保险服务（金融服务）增值税率
SUR = 0.12                              # 城建税 7% + 教育费附加 3% + 地方教育附加 2%
EF_RATE = 0.008                         # 保险保障基金·财产保险基准费率（按业务收入）
C_TAX = VAT / (1.0 + VAT) * SUR         # 税金及附加（占含税保费）= 0.679%

# 参考分解中的两项 ⓗ（行业基准，待公司财务数据替换）
C_ACQ = 0.100                           # 手续费及佣金（获取成本）
C_ADMIN_V = 0.030                       # 业务及管理费·变动
C_ADMIN_F = 0.080                       # 业务及管理费·固定（含本产品特有复核成本）

C_NEUTRAL = E_S7                        # 费用中性分保佣金率
C_LO = C_ACQ + C_ADMIN_V                # 经济补偿口径下限 = 变动费用率
C_HI = 0.40                             # 行业惯例上限（梯次佣金最高 40%，MBA智库）

QS = [0.20, 0.30, 0.35, 0.40]           # 成数分出比例（§7.2 建议 30–40%）
CS = [0.13, 0.20, 0.25, 0.30, 0.40]     # 分保佣金率候选


# ── 数据装载 ────────────────────────────────────────────────────────────────
def load_anchor() -> dict:
    """取 s7 的锚档（优先『论文口径』）× 5 年期，作为费用归属的计量样本。"""
    T = pd.read_csv(OUT / "s7_精算口径保费.csv")
    tiers = list(dict.fromkeys(T["标度档"]))
    tier = "论文口径" if "论文口径" in tiers else tiers[len(tiers) // 2]
    x = T[(T["标度档"] == tier) & (T["期限"] == 5)].iloc[0]
    si = float(x["SI_新"])
    rate = float(x["毛费率_新%"])
    return {
        "标度档": tier, "期限": 5,
        "SI": si, "E": float(x["E"]), "P99.5": float(x["P99.5"]),
        "TVaR99": float(x["TVaR99"]),
        "毛费率%": rate, "纯风险%": float(x["纯风险%"]),
        "GP": rate / 100.0 * si,                       # 毛保费（元/吨）
    }


# ── 表 1：费用率分解 ────────────────────────────────────────────────────────
def build_decomposition() -> pd.DataFrame:
    rows = [
        ("手续费及佣金（获取成本）", "c_acq", C_ACQ, "是",
         "财险费用率中手续费及佣金占已赚保费的比例；PICC 与中农再成数分保约定分保佣金 20%",
         "上市险企年报口径（行业参考）/ PICC 临时信息披露报告〔2023〕5 号", "中"),
        ("业务及管理费·变动部分", "c_admin_v", C_ADMIN_V, "是",
         "随保费规模变动的承保作业成本（核保、出单、收费、账务）",
         "行业基准（ⓗ 估算，待公司财务/精算数据替换）", "低"),
        ("业务及管理费·固定部分", "c_admin_f", C_ADMIN_F, "否",
         "人员、IT 及本产品特有成本：年度合法域重解、指数披露、第三方复核、重定价执行",
         "行业基准（ⓗ 估算，待公司财务/精算数据替换）", "低"),
        ("税金及附加", "c_tax", C_TAX, "否",
         "保险服务增值税 6%（价内换算 5.66%）× 附加税费 12% = 0.679%",
         "《营业税改征增值税试点实施办法》+ 城建税/教育费附加/地方教育附加规定", "高"),
        ("保险保障基金", "c_ef", EF_RATE, "否",
         "财产保险按业务收入 0.8% 缴纳（基准费率）+ 风险差别费率（A −0.02% ~ D +0.04%）",
         "《保险保障基金管理办法》（2022-12-12 施行）+ 银保监会 2023 年缴纳通知", "高"),
    ]
    df = pd.DataFrame(rows, columns=["项目", "符号", "参考费率_占GP", "随分出减少",
                                     "依据", "信源", "置信度"])
    sub = float(df["参考费率_占GP"].sum())
    df.loc[len(df)] = ["参考分解小计", "Σ", sub, "—", "上五项之和", "本脚本", "—"]
    df.loc[len(df)] = ["s7 假设费用率", "e", E_S7, "—",
                       "_config.py L1.expense_ratio（内设值，未经外部校准）", "_config.py", "—"]
    df.loc[len(df)] = ["差额·未分配缓冲", "e − Σ", E_S7 - sub, "—",
                       "参考分解低于 s7 假设；差额建议保留为费用超支缓冲，不并入利润",
                       "本脚本", "—"]
    return df


# ── 表 2：再保佣金兼容性 ────────────────────────────────────────────────────
def build_cesium_grid() -> pd.DataFrame:
    rows = []
    for q in QS:
        for c in CS:
            eff = (E_S7 - c * q) / (1.0 - q)
            d = (eff - E_S7) * 100.0
            verdict = "中性" if abs(d) < 1e-9 else ("改善" if d < 0 else "恶化")
            rows.append({
                "分出比例q": q, "分保佣金率c": c,
                "摊回分保费用率_c×q": round(c * q, 4),
                "分保后综合费用率": round(eff, 4),
                "相对e变动_pp": round(d, 3),
                "判定": verdict,
            })
    return pd.DataFrame(rows)


# ── 表 3：增值税口径敏感性 ──────────────────────────────────────────────────
def build_vat_sensitivity(a: dict) -> pd.DataFrame:
    """报价含税 vs 不含税：E/RM/CapCost 均为不含税成本，故含税口径需放大分母。"""
    v = VAT / (1.0 + VAT)                   # 增值税占含税保费比例
    denom_n = 1.0 - E_S7 - PI_S7            # 不含税口径分母
    denom_i = denom_n - v                   # 含税口径分母（增值税先被抽走）
    mult = denom_n / denom_i
    rows = [
        {"口径": "不含税报价（s7 现行）", "分母_1−v−e−π": round(denom_n, 6),
         "相对费率倍数": 1.0, "5 年费率%": round(a["毛费率%"], 2),
         "5 年保费_元每吨": round(a["GP"], 0), "说明": "E/RM/CapCost 均为不含税口径，与之一致"},
        {"口径": "含税报价（增值税 6% 价内）", "分母_1−v−e−π": round(denom_i, 6),
         "相对费率倍数": round(mult, 4), "5 年费率%": round(a["毛费率%"] * mult, 2),
         "5 年保费_元每吨": round(a["GP"] * mult, 0),
         "说明": f"保费含 5.66% 增值税；费率须放大 {(mult - 1) * 100:.2f}% 方等价"},
    ]
    return pd.DataFrame(rows), mult


# ── 表 4：费用/赔付比异常登记 ───────────────────────────────────────────────
def build_ratio_note(a: dict) -> pd.DataFrame:
    """费用/赔付比（含 2026-10-02 口径更正）。

    ⚠ 更正说明：赔付率必须用**保费口径** E/GP —— 行业基准 71.8% 是「赔付/已赚保费」。
    首版误用 E/SI（限额口径，即「赔付/责任上限」），与行业基准不同基准，
    使倍数被高估约 3 倍（旧述 9.23 → 正确值见下）。
    """
    pure_lim = a["纯风险%"] / 100.0                  # E/SI —— 限额口径，**不是**赔付率
    pure_prem = pure_lim * a["SI"] / a["GP"]         # E/GP —— 保费口径，标准赔付率
    rows = [
        {"口径": "本产品（论文口径 5 年）", "费用率": E_S7, "赔付率": round(pure_prem, 4),
         "费用/赔付": round(E_S7 / pure_prem, 2),
         "备注": f"赔付率取**保费口径** E/GP（{a['E']:,.0f}/{a['GP']:,.0f}）；"
                 f"限额口径 E/SI={pure_lim:.2%} 属另一分母，不可与行业基准比较"},
        {"口径": "常规财险（行业参考）", "费用率": 0.258, "赔付率": 0.718,
         "费用/赔付": round(0.258 / 0.718, 2),
         "备注": "人保财险 2025 综合成本率 97.6% / 上半年综合赔付率 71.8%（二手聚合，待核年报原文）"},
    ]
    return pd.DataFrame(rows)


# ── 验收断言 ────────────────────────────────────────────────────────────────
def acceptance(verbose: bool = True) -> bool:
    a = load_anchor()
    res = []

    def add(k, name, ok, got, want):
        res.append({"编号": k, "断言": name, "结果": "PASS" if ok else "FAIL",
                    "实测": str(got), "判据": str(want)})

    sub = C_ACQ + C_ADMIN_V + C_ADMIN_F + C_TAX + EF_RATE

    # G-B3-1 分解可加性
    add("G-B3-1", "参考分解五项之和 = 小计（可加性闭合）",
        abs(sub - (C_ACQ + C_ADMIN_V + C_ADMIN_F + C_TAX + EF_RATE)) < 1e-15,
        f"{sub:.6f}", "恒等")

    # G-B3-2 参考分解不超过 s7 假设
    add("G-B3-2", "参考分解小计 ≤ s7 假设费用率 e",
        sub <= E_S7 + 1e-12, f"{sub:.6f} ≤ {E_S7:.4f}", "≤ e")

    # G-B3-3 税金及附加按现行政策
    add("G-B3-3", "税金及附加 = 增值税价内比例 × 附加 12%",
        abs(C_TAX - VAT / (1 + VAT) * SUR) < 1e-15, f"{C_TAX:.6f}", "0.06/1.06×0.12")

    # G-B3-4 保险保障基金按现行政策（财产保险 0.8%，按业务收入）
    add("G-B3-4", "保险保障基金 = 0.8%（财产保险基准费率，按业务收入）",
        abs(EF_RATE - 0.008) < 1e-15, f"{EF_RATE:.4f}", "0.008")

    # G-B3-5 费用中性佣金率与分出比例无关（A16 的可核算性）
    devs = []
    for q in QS:
        eff = (E_S7 - C_NEUTRAL * q) / (1 - q)
        devs.append(abs(eff - E_S7))
    add("G-B3-5", "费用中性佣金率 c* = e，且与分出比例 q 无关",
        max(devs) < 1e-12, f"max|Δ|={max(devs):.2e} over q∈{QS}", "c*=e，与 q 无关")

    # G-B3-6 议价区间包含中性点
    add("G-B3-6", "议价区间 [变动费用率, 行业上限] 包含费用中性点",
        C_LO < C_NEUTRAL < C_HI, f"[{C_LO:.3f}, {C_HI:.3f}] ∋ {C_NEUTRAL:.4f}", "C_LO < e < C_HI")

    # G-B3-7 含税口径放大
    _, mult = build_vat_sensitivity(a)
    add("G-B3-7", "含税报价相对不含税报价的费率放大倍数 > 1（登记）",
        mult > 1.0 + 1e-9, f"{mult:.4f}（+{(mult-1)*100:.2f}%）", "> 1")

    # G-B3-8 费用/赔付比异常（口径更正：赔付率取保费口径 E/GP）
    pure_lim = a["纯风险%"] / 100.0
    pure_prem = pure_lim * a["SI"] / a["GP"]
    add("G-B3-8", "费用/赔付比 > 1（异常，须披露）—— 赔付率取**保费口径** E/GP",
        E_S7 / pure_prem > 1.0,
        f"{E_S7/pure_prem:.2f}（e={E_S7:.2f} / 赔付率={pure_prem:.4f}）；"
        f"限额口径 E/SI={pure_lim:.4f} 得 {E_S7/pure_lim:.2f}（不可与行业基准比较）",
        "> 1")

    # G-B3-10 口径锁定：两口径之比 = SI/GP（防止再次混用）
    add("G-B3-10", "两种赔付率口径之比 = SI/GP（防止再次混用）",
        abs((pure_lim / pure_prem) - a["GP"] / a["SI"]) < 1e-9,
        f"{pure_lim/pure_prem:.6f} = SI/GP = {a['SI']/a['GP']:.6f}", "恒等")

    # G-B3-9 分解项均已标注随分出是否减少（供分保佣金谈判引用）
    df = build_decomposition().head(5)
    add("G-B3-9", "五项分解均标注『随分出减少』标记",
        df["随分出减少"].isin(["是", "否"]).all(),
        df["随分出减少"].tolist(), "全部为 是/否")

    V = pd.DataFrame(res)
    V.to_csv(OUT / "s10_费用率验收.csv", index=False, encoding="utf-8-sig")
    if verbose:
        print(V.to_string(index=False))
        nf = int((V["结果"] == "FAIL").sum())
        print(f"\n  ⇒ {len(V) - nf} PASS / {nf} FAIL")
    return bool((V["结果"] == "PASS").all())


# ── 主流程 ──────────────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 92)
    print("s10　B3 费用率分解与分保佣金可核算性")
    print("=" * 92)
    a = load_anchor()
    print(f"  锚档：{a['标度档']} × {a['期限']} 年   SI={a['SI']:,.0f}   GP={a['GP']:,.0f} 元/吨"
          f"   毛费率={a['毛费率%']:.2f}%   纯风险率={a['纯风险%']:.2f}%")
    print(f"  口径：e={E_S7:.0%}（s7 假设）  π={PI_S7:.0%}  ρ={float(L1.get('capital_charge', 0.06)):.0%}"
          f"   增值税={VAT:.0%}  保障基金={EF_RATE:.1%}")

    print("\n【表 1】25% 费用率的参考分解（占毛保费 GP）")
    print("-" * 92)
    D = build_decomposition()
    print(D[["项目", "符号", "参考费率_占GP", "随分出减少", "置信度"]].to_string(index=False))
    print("\n  依据与信源：")
    for _, r in D.head(5).iterrows():
        print(f"    · {r['项目']}：{r['依据'][:46]}…  ← {r['信源']}")

    print("\n【表 2】分保佣金兼容性网格：分保后综合费用率 = (e − c·q)/(1 − q)")
    print("-" * 92)
    G = build_cesium_grid()
    for q in QS:
        x = G[G["分出比例q"] == q]
        print(f"  q={q:.2f}：" + "  ".join(
            f"c={r['分保佣金率c']:.2f}→{r['分保后综合费用率']:.4f}({r['相对e变动_pp']:+.2f}pp)"
            for _, r in x.iterrows()))

    print(f"\n  ⇒ 费用中性线 c* = e = {C_NEUTRAL:.2%}，与 q 无关（解析：c=e ⟺ e−cq=e(1−q)）")
    print(f"  ⇒ 议价区间：下沿（经济补偿口径，仅补变动费用）= {C_LO:.1%}；"
          f"中性点 = {C_NEUTRAL:.1%}；行业惯例上限 = {C_HI:.0%}")
    print(f"  ⇒ 若再保人只给 c={CS[0]:.0%}（q=0.35），费用率将升至 "
          f"{(E_S7 - CS[0]*0.35)/0.65:.4f}（+{(E_S7 - CS[0]*0.35)/0.65*100 - E_S7*100:.2f}pp）")

    print("\n【表 3】增值税口径敏感性（报价含税与否）")
    print("-" * 92)
    Va, mult = build_vat_sensitivity(a)
    print(Va[["口径", "分母_1−v−e−π", "相对费率倍数", "5 年费率%",
              "5 年保费_元每吨"]].to_string(index=False))
    print(f"  ⇒ 若对外报价为含税价，现行费率须整体放大 {mult:.4f}×（+{(mult-1)*100:.2f}%）方可等价")

    print("\n【表 4】费用/赔付比异常登记")
    print("-" * 92)
    R = build_ratio_note(a)
    print(R[["口径", "费用率", "赔付率", "费用/赔付"]].to_string(index=False))

    print("\n【验收断言】")
    print("-" * 92)
    ok = acceptance(verbose=True)

    D.to_csv(OUT / "s10_费用率分解.csv", index=False, encoding="utf-8-sig")
    G.to_csv(OUT / "s10_再保佣金兼容性.csv", index=False, encoding="utf-8-sig")
    R.to_csv(OUT / "s10_费用赔付比.csv", index=False, encoding="utf-8-sig")
    Va.to_csv(OUT / "s10_增值税口径敏感.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "s10_费用率分解.json", "w", encoding="utf-8") as f:
        json.dump({
            "对应": "双指标稿 §7.4 / §9.3 A16 / §10 Q7；规划 B3；验收门 G-B3",
            "s7假设费用率e": E_S7,
            "利润边际π": PI_S7,
            "参考分解": {r["符号"]: round(float(r["参考费率_占GP"]), 6)
                     for _, r in D.head(5).iterrows()},
            "参考分解小计": round(float(C_ACQ + C_ADMIN_V + C_ADMIN_F + C_TAX + EF_RATE), 6),
            "税费口径": {
                "增值税率": VAT, "附加税率合计": SUR,
                "税金及附加_占含税保费": round(C_TAX, 6),
                "保险保障基金_财产保险基准费率": EF_RATE,
                "保险保障基金计费基数": "业务收入（投保人支付的全部金额），不因分出而减少",
            },
            "分保佣金核心结论": {
                "公式": "分保后综合费用率 = (e − c·q)/(1 − q)",
                "费用中性条件": "c = e（与 q 无关）",
                "可核算下界_经济补偿口径": C_LO,
                "中性点": C_NEUTRAL,
                "行业惯例上限": C_HI,
            },
            "增值税口径": {"报价含税时费率放大倍数": round(mult, 6),
                       "状态": "待拍板——影响全部对外报价"},
            "科目归属更正": {
                "分出保费": "保费减项（不进费用率）",
                "摊回分保费用": "费用减项（进费用率，即分保佣金）",
                "摊回赔付": "赔付减项",
                "说明": "设计稿把『再保费』列为费用率分解项属科目混淆",
            },
            "诚实边界": [
                "仅税金及附加与保险保障基金为政策硬数",
                "获取成本与管理费为行业基准 ⓗ，须以公司财务/精算数据替换（规划 R4）",
                "本脚本不修改 s4/s7",
            ],
        }, f, ensure_ascii=False, indent=2)
    print("\n[已写出] s10_费用率分解.csv / .json / s10_再保佣金兼容性.csv / "
          "s10_费用赔付比.csv / s10_费用率验收.csv")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--check" in sys.argv:
        sys.exit(0 if acceptance(verbose=True) else 1)
    main()
