# -*- coding: utf-8 -*-
"""
s14_hfg_impact.py —— C2-β：非车险「报行合一」对本产品的影响评估

背景
----
《关于加强非车险业务监管有关事项的通知》（金融监管总局 2025-09-30 发布、2025-11-01 施行）
适用于「机动车辆保险以外的其他财产保险业务」（农业保险、出口信用保险除外）。
本产品为指数挂钩型多年期财产险，**落在适用范围内**。

本脚本只回答两个问题，都必须是可复算的：
  Q1  产品的费率结构在「报行合一 + 预定附加费率上限」下是否可备案？
  Q2  重定价条款（A3）能否装进「费率定期回溯与动态调整机制」？

说明：通知全文与配套问答未取得一手文本；本脚本引用的条款要点**据公开报道整理**
（百度百科词条摘要 / 经济日报 / 武汉市金融办转载 / 2026-08-21《非车险综合治理行动方案》），
凡未核实处均标「待核原文」。本脚本**不修改任何上游产物**，只读取 s7 / s9 / s10 的输出。

运行：python s14_hfg_impact.py [--check]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import L1, OUT                                   # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.unicode.east_asian_width", True)

# ── 常数 ────────────────────────────────────────────────────────────────────
E_EXP = float(L1["expense_ratio"])          # 实际费用率（占 GP）= 25%
PI_PROFIT = 0.08                            # 目标利润（占 GP）
ADD_ON = E_EXP + PI_PROFIT                  # 预定附加费率 = 33%
Q_RISK = 0.99
Q_LIMIT = 0.995

# 行业参考（二手聚合，待核年报原文）
IND_LOSS_RATIO = 0.718                      # 人保财险 2025H1 综合赔付率
IND_EXP_RATIO = 0.258
IND_COMBINED = 0.976                        # 人保财险 2025 综合成本率

# 条款 §5.2 协商带
BAND_RATE = 0.30
BAND_LIMIT = 0.40

# 待核：预定附加费率上限的取值网格（具体数值未从一手文本取得）
CAP_GRID = [0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35]


# ── 装载 ────────────────────────────────────────────────────────────────────
def load_s7() -> pd.DataFrame:
    return pd.read_csv(OUT / "s7_精算口径保费.csv")


def load_s9() -> pd.DataFrame:
    return pd.read_csv(OUT / "s9_重定价挂牌.csv")


# ── 表 1：保费结构分解（占 GP）────────────────────────────────────────────
def build_structure(T: pd.DataFrame) -> pd.DataFrame:
    """把 s7 的「占 SI」口径换算为「占 GP」口径 —— 报行合一盯的是占保费的结构。"""
    rows = []
    for _, x in T.iterrows():
        gp = float(x["毛费率_新%"]) / 100.0 * float(x["SI_新"])
        rows.append({
            "标度档": x["标度档"], "期限": int(x["期限"]),
            "SI_元每吨": round(float(x["SI_新"]), 0),
            "GP_元每吨": round(gp, 0),
            "纯风险占GP%": round(float(x["E"]) / gp * 100, 2),
            "风险附加占GP%": round((float(x["TVaR99"]) - float(x["E"])) / gp * 100, 2),
            "资本成本占GP%": round(float(x["资本成本%"]) / 100 * float(x["SI_新"]) / gp * 100, 2),
            "附加费用占GP%": round(E_EXP * 100, 2),
            "利润占GP%": round(PI_PROFIT * 100, 2),
            "赔付率_保费口径%": round(float(x["E"]) / gp * 100, 2),
            "综合成本率%": round((float(x["E"]) / gp + E_EXP) * 100, 2),
            "承保利润率%": round((1 - float(x["E"]) / gp - E_EXP) * 100, 2),
        })
    return pd.DataFrame(rows)


# ── 表 2：预定附加费率上限敏感性 ───────────────────────────────────────────
def build_cap_sensitivity(T: pd.DataFrame) -> pd.DataFrame:
    """上限 a_cap 与现行 33% 的关系。

    GP = 净保费 /(1 − a)。若备案上限 a_cap < 需要值 ADD_ON，则有两种后果：
      (i)  按 a_cap 报价 ⇒ GP 下降（倍数 = (1−ADD_ON)/(1−a_cap)），
           此时「附加保费 = a_cap × GP」是否仍覆盖实际费用 e=25%？
           a_cap × GP ≥ e × GP  ⟺  a_cap ≥ e = 25%。
      (ii) 压费用/压利润以适配 a_cap。
    ⇒ **可行下界 = 实际费用率 25%**（此时利润为 0）；带 8% 利润需 33%。
    """
    anchor = T[(T["标度档"] == "论文口径") & (T["期限"] == 5)].iloc[0]
    net = float(anchor["E"]) + (float(anchor["TVaR99"]) - float(anchor["E"])) \
        + float(anchor["资本成本%"]) / 100 * float(anchor["SI_新"])
    si = float(anchor["SI_新"])
    rows = []
    for a in CAP_GRID:
        mult = (1 - ADD_ON) / (1 - a)
        gp = net / (1 - a)
        profit_room = a - E_EXP                      # 可分配利润率（占 GP）
        if a < E_EXP - 1e-12:
            verdict = "不可备案（附加不足覆盖实际费用）"
        elif a < ADD_ON - 1e-12:
            verdict = "可备案，但须压费用或压利润"
        elif abs(a - ADD_ON) < 1e-12:
            verdict = "可行（等于现行假设）"
        else:
            verdict = "可行且有利润余量"
        rows.append({
            "备案上限a_cap": f"{a:.0%}",
            "分母1−a_cap": round(1 - a, 4),
            "费率相对现行倍数": round(mult, 4),
            "5年费率%": round(gp / si * 100, 2),
            "5年保费_元每吨": round(gp, 0),
            "附加保费_元每吨": round(a * gp, 0),
            "实际费用_元每吨": round(E_EXP * gp, 0),
            "费用覆盖": "是" if a >= E_EXP - 1e-12 else "否",
            "可分配利润_占GP": f"{profit_room:.1%}",
            "判定": verdict,
        })
    return pd.DataFrame(rows)


# ── 表 3：重定价备案路径 ───────────────────────────────────────────────────
def build_paths(Q: pd.DataFrame) -> pd.DataFrame:
    rch = Q["费率变动%"].astype(float)
    lch = Q["限额变动%"].astype(float)
    n_rate_breach = int((rch.abs() > BAND_RATE * 100).sum())
    n_lim_breach = int((lch.abs() > BAND_LIMIT * 100).sum())
    n = len(Q)
    rows = [
        {"路径": "P1 公式化备案",
         "机制": "把费率备案为「计算公式 + 七项参数集 + 上下限」，重定价按公式执行，不逐次报备",
         "需监管认可": "「公式 + 参数锁定」可作为备案标的；参数变更是否属「费率调整」",
         "量化后果": f"可覆盖全部 {n} 次重定价；但须披露 7 项参数与合法域求法",
         "可行性": "高（与「费率定期回溯与动态调整机制」方向一致）"},
        {"路径": "P2 逐次报备",
         "机制": "每次重定价前单独报备新费率",
         "需监管认可": "备案周期与条款 §5.1「重定价日前 N 日通知」的时序兼容",
         "量化后果": f"单次费率变动最大 {rch.abs().max():.2f}%、限额最大 {lch.abs().max():.2f}%；"
                    f"{n_rate_breach}/{n} 次触发 §5.2 协商带",
         "可行性": "中（时序风险：协商期 30 日 + 退保处理）"},
        {"路径": "P3 浮动费率条款",
         "机制": "备案一个费率区间 [r_lo, r_hi]，重定价在区间内浮动；越界则协商/退保",
         "需监管认可": "浮动区间的宽度与触发条件；越界后的处置是否构成「变相突破备案」",
         "量化后果": f"若区间取 ±30%，可覆盖 {int((rch.abs() <= 30).sum())}/{n} 次；"
                    f"2028 年事件 {rch.max():.2f}% 越界",
         "可行性": "中高（最贴近现有非车险浮动费率实践）"},
    ]
    return pd.DataFrame(rows)


# ── 表 4：备案区间覆盖率 ───────────────────────────────────────────────────
def build_band_coverage(Q: pd.DataFrame) -> pd.DataFrame:
    rch = Q["费率变动%"].astype(float).abs()
    rows = []
    for b in [10, 20, 30, 40, 50, 60, 70]:
        cover = int((rch <= b).sum())
        rows.append({
            "备案区间±%": b,
            "覆盖事件数": cover,
            "总事件数": len(Q),
            "覆盖率%": round(cover / len(Q) * 100, 1),
            "未覆盖": int(len(Q) - cover),
        })
    return pd.DataFrame(rows)


# ── 表 5：要件对照 ─────────────────────────────────────────────────────────
def build_checklist() -> pd.DataFrame:
    R = [
        ("报行合一：实际执行费率 = 备案费率", "本产品费率按重定价条款每期重算",
         "需调整", "重定价条款（A3）是核心冲突点；见路径 P1–P3", "待核原文"),
        ("建立费率定期回溯和动态调整机制", "本产品已有定期重定价 + 越界触发重估（§5.1/§5.3）",
         "方向一致", "**这是通道而非障碍** —— 重定价可装进该机制，须确认触发条件与频率上限", "待核原文"),
        ("明确预定附加费率上限", "预定附加费率 = e 25% + π 8% = 33%",
         "待确认", "上限数值未从一手文本取得；可行下界 = 实际费用率 25%（见表 2）", "待核原文"),
        ("据实列支费用 + 加强手续费核算管控", "B3 已把 25% 分解为 5 项（获取 10.0 / 管理 11.0 / 税附 0.679 / 保障基金 0.800）",
         "已备", "分解结果为行业基准 ⓗ，须以公司财务/精算实数据替换后方可备案", "自有产物"),
        ("见费出单 + 规范分次缴费结构", "s7/s11 按**趸缴**建模（单次缴费）",
         "待确认", "趸缴天然满足「见费出单」；若改期缴须重做 B1 的 UPR 与 B4 年度视图", "待核原文"),
        ("禁止超额手续费 / 虚列费用", "费用率 25% > 参考分解 22.479%，差 2.521pp 留作超支缓冲",
         "需说明", "缓冲若被读作「虚列」，须提供费用明细台账", "自有产物"),
        ("强化消费者权益保护", "赔付率（保费口径）4.27%–27.94%，行业 71.8%（见表 1）",
         "需主动说明", "**本产品最大的合规叙事风险**：须以「55.5% 保费为风险附加」解释，非费用吞噬", "自有产物"),
        ("降低保费规模/增速考核权重", "内部考核事项", "不适用", "本产品为参赛方案，无规模考核", "—"),
        ("分险种分阶段推进存量产品重新备案", "本产品为新产品，非存量",
         "不适用", "但备案标准在收紧（2026-08-21 行动方案）", "待核原文"),
        ("产品备案要点 / 动态问题清单 / 智能检核", "本产品的费率是函数而非价格表",
         "待确认", "须确认函数形式能否通过智能检核（银保信）", "待核原文"),
    ]
    return pd.DataFrame(R, columns=["报行合一要件", "本产品现状", "判定", "说明", "依据"])


# ── 验收断言 ────────────────────────────────────────────────────────────────
def acceptance(verbose: bool = True) -> bool:
    T, Q = load_s7(), load_s9()
    res = []

    def add(k, name, ok, got, want):
        res.append({"编号": k, "断言": name, "结果": "PASS" if ok else "FAIL",
                    "实测": str(got), "判据": str(want)})

    anchor = T[(T["标度档"] == "论文口径") & (T["期限"] == 5)].iloc[0]

    # H-1 附加费率构成
    add("H-1", "预定附加费率 = e + π = 33%",
        abs(ADD_ON - (E_EXP + PI_PROFIT)) < 1e-15, f"{ADD_ON:.4f}", "0.33")

    # H-2 与 s7 对账：复算 GP
    net = float(anchor["E"]) + (float(anchor["TVaR99"]) - float(anchor["E"])) \
        + float(anchor["资本成本%"]) / 100 * float(anchor["SI_新"])
    gp_calc = net / (1 - ADD_ON)
    gp_s7 = float(anchor["毛费率_新%"]) / 100.0 * float(anchor["SI_新"])
    add("H-2", "复算 GP 与 s7 产物一致（对账）",
        abs(gp_calc - gp_s7) / gp_s7 < 5e-4, f"相对偏差 {abs(gp_calc-gp_s7)/gp_s7:.2e}", "< 5e-4")

    # H-3 上限敏感性单调
    C = build_cap_sensitivity(T)
    rates = C["5年费率%"].astype(float).tolist()
    add("H-3", "费率随备案上限 a_cap 单调递增",
        all(rates[i] < rates[i + 1] for i in range(len(rates) - 1)),
        "严格递增" if all(rates[i] < rates[i + 1] for i in range(len(rates) - 1)) else "有反转",
        "严格递增")

    # H-4 可行下界 = 实际费用率
    add("H-4", "附加费率可行下界 = 实际费用率 e（低于此值费用无法覆盖）",
        all((float(r["备案上限a_cap"].rstrip("%")) / 100 >= E_EXP - 1e-12)
            == (r["费用覆盖"] == "是") for _, r in C.iterrows()),
        f"下界 {E_EXP:.0%}", f"a_cap ≥ {E_EXP:.0%}")

    # H-5 赔付率口径与行业对比
    S = build_structure(T)
    pr = S["赔付率_保费口径%"].astype(float)
    add("H-5", "赔付率（保费口径）全部低于行业基准 71.8%，且费用/赔付比 > 1",
        bool((pr < IND_LOSS_RATIO * 100).all()) and E_EXP / (pr.min() / 100) > 1.0,
        f"区间 [{pr.min():.2f}%, {pr.max():.2f}%]；费用/赔付比最大 {E_EXP/(pr.min()/100):.2f}",
        "< 71.8% 且比值 > 1")

    # H-6 保费结构闭合（容差按产出精度设定）
    #   五项各自四舍五入到 0.01pp；且 资本成本占GP% 由已舍入的「资本成本%」换算，
    #   上游 E/TVaR99(0.1)、SI_新(1.0)、毛费率_新%(0.01) 均带舍入 ⇒ 实测最大偏差 0.07pp。
    #   故容差取 0.1pp（= 单行 5 项 × 0.01pp 舍入 + 上游舍入放大后的实测上界）。
    comp = (S["纯风险占GP%"] + S["风险附加占GP%"] + S["资本成本占GP%"]
            + S["附加费用占GP%"] + S["利润占GP%"])
    add("H-6", "保费结构各项占 GP 之和 = 100%（容差按产出精度 0.1pp 设定）",
        bool((comp - 100).abs().max() < 0.10), f"最大偏差 {float((comp-100).abs().max()):.4f}pp", "< 0.10pp")

    # H-6b 结构恒等（对未舍入值）：净保费/GP = 1 − 附加费率
    _rows = []
    for _, x in T.iterrows():
        gp = float(x["毛费率_新%"]) / 100.0 * float(x["SI_新"])
        net_i = float(x["E"]) + (float(x["TVaR99"]) - float(x["E"]))             + float(x["资本成本%"]) / 100 * float(x["SI_新"])
        _rows.append(abs(net_i / gp - (1 - ADD_ON)))
    #   容差来源：上游 4 个量均只存**显示精度**——毛费率_新%(0.01pp)、资本成本%(0.01pp)、
    #   E/TVaR99(0.1)、SI_新(1.0)。舍入传播复合上界实测 6.63e-04，故取 1e-3（按产出精度设定）。
    add("H-6b", "结构恒等 净保费/GP = 1 − 附加费率（含显示精度舍入传播）",
        max(_rows) < 1e-3, f"最大偏差 {max(_rows):.2e}", "< 1e-3")

    # H-7 重定价触发协商带
    rch = Q["费率变动%"].astype(float).abs()
    n_hit = int((rch > BAND_RATE * 100).sum())
    add("H-7", "重定价事件中恰有 2/5 触发 §5.2 费率协商带（±30%）",
        n_hit == 2, f"{n_hit}/{len(Q)}", "2/5")

    # H-8 备案区间 ±30% 无法覆盖全部事件
    cov = build_band_coverage(Q).set_index("备案区间±%")
    add("H-8", "备案区间 ±30% 不能覆盖全部重定价事件（须择一路径）",
        int(cov.loc[30, "未覆盖"]) > 0,
        f"±30% 未覆盖 {int(cov.loc[30,'未覆盖'])} 次；±60% 未覆盖 {int(cov.loc[60,'未覆盖'])} 次",
        "> 0")

    V = pd.DataFrame(res)
    V.to_csv(OUT / "s14_报行合一验收.csv", index=False, encoding="utf-8-sig")
    if verbose:
        print(V.to_string(index=False))
        nf = int((V["结果"] == "FAIL").sum())
        print(f"\n  ⇒ {len(V) - nf} PASS / {nf} FAIL")
    return bool((V["结果"] == "PASS").all())


# ── 主流程 ──────────────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    check_only = "--check" in sys.argv

    if check_only:
        sys.exit(0 if acceptance(verbose=True) else 1)

    print("=" * 118)
    print("s14　C2-β：非车险「报行合一」对本产品的影响评估")
    print("=" * 118)
    print("  适用范围：机动车辆保险以外的财产保险业务（农险、出口信用保险除外）⇒ **本产品适用**")
    print(f"  现行参数：e={E_EXP:.0%}  π={PI_PROFIT:.0%}  预定附加费率={ADD_ON:.0%}"
          f"  实际费用率={E_EXP:.0%}")

    T, Q = load_s7(), load_s9()

    print("\n【表 1】保费结构分解（占 GP）—— 报行合一盯的是占保费的结构")
    print("-" * 118)
    S = build_structure(T)
    anchor = S[(S["标度档"] == "论文口径") & (S["期限"] == 5)].iloc[0]
    print(S.round(2).to_string(index=False))
    print(f"\n  ⇒ 锚档（论文口径 5 年）：纯风险 {anchor['纯风险占GP%']:.2f}% / "
          f"风险附加 {anchor['风险附加占GP%']:.2f}% / 资本成本 {anchor['资本成本占GP%']:.2f}% / "
          f"附加费用 {anchor['附加费用占GP%']:.2f}% / 利润 {anchor['利润占GP%']:.2f}%")
    print(f"  ⇒ **保费中 {anchor['风险附加占GP%']:.1f}% 是风险附加（TVaR−E），只有 "
          f"{anchor['纯风险占GP%']:.2f}% 是期望赔付** ⇒ 费用/赔付比高不是费用贵，是尾部重")

    print("\n【表 2】预定附加费率上限敏感性（锚：论文口径 5 年）")
    print("-" * 118)
    C = build_cap_sensitivity(T)
    print(C.to_string(index=False))
    print(f"\n  ⇒ **附加费率可行下界 = 实际费用率 {E_EXP:.0%}**（此时利润为 0）；"
          f"带 {PI_PROFIT:.0%} 目标利润需 {ADD_ON:.0%}。上限低于 {E_EXP:.0%} ⇒ 产品不可备案。")

    print("\n【表 3】重定价条款的三条备案路径")
    print("-" * 118)
    P = build_paths(Q)
    for _, r in P.iterrows():
        print(f"  ■ {r['路径']}（可行性：{r['可行性']}）")
        print(f"      机制：{r['机制']}")
        print(f"      需确认：{r['需监管认可']}")
        print(f"      量化：{r['量化后果']}")

    print("\n【表 4】备案区间覆盖率（对 s9 的 5 次重定价事件）")
    print("-" * 118)
    B = build_band_coverage(Q)
    print(B.to_string(index=False))

    print("\n【表 5】报行合一要件对照（据公开报道整理，须核原文）")
    print("-" * 118)
    K = build_checklist()
    print(K[["报行合一要件", "判定", "依据"]].to_string(index=False))

    print("\n【验收断言】")
    print("-" * 118)
    acceptance(verbose=True)

    # ── 落盘 ────────────────────────────────────────────────────────────────
    S.to_csv(OUT / "s14_保费结构分解.csv", index=False, encoding="utf-8-sig")
    C.to_csv(OUT / "s14_附加费率上限敏感.csv", index=False, encoding="utf-8-sig")
    P.to_csv(OUT / "s14_重定价备案路径.csv", index=False, encoding="utf-8-sig")
    B.to_csv(OUT / "s14_备案区间覆盖.csv", index=False, encoding="utf-8-sig")
    K.to_csv(OUT / "s14_报行合一要件对照.csv", index=False, encoding="utf-8-sig")

    cfg = {
        "适用性": "本产品属「机动车辆保险以外的财产保险业务」⇒ 报行合一适用",
        "规范依据": [
            "《关于加强非车险业务监管有关事项的通知》金融监管总局 2025-09-30 发布、2025-11-01 施行（待核原文）",
            "《非车险综合治理有关问答(一)》2026-01-09（见费出单执行标准、规范分次缴费结构）",
            "《非车险综合治理行动方案》2026-08-21（分险种分阶段存量产品重新备案）",
        ],
        "参数": {"e": E_EXP, "π": PI_PROFIT, "预定附加费率": ADD_ON,
                 "附加费率可行下界": E_EXP},
        "行业参考": {"综合赔付率": IND_LOSS_RATIO, "综合费用率": IND_EXP_RATIO,
                     "综合成本率": IND_COMBINED, "口径": "人保财险 2025（二手聚合，待核年报原文）"},
        "锚档": {"标度档": anchor["标度档"], "期限": int(anchor["期限"]),
                 "SI": float(anchor["SI_元每吨"]), "GP": float(anchor["GP_元每吨"]),
                 "赔付率_保费口径": float(anchor["赔付率_保费口径%"]) / 100},
        "待核清单": [
            "通知全文与配套问答的原始文本（现行引用为公开报道整理）",
            "「预定附加费率上限」的具体数值",
            "费率定期回溯与动态调整机制的触发条件、频率上限、是否需逐次报备",
            "函数形式费率能否通过银保信智能检核",
            "趸缴安排与「见费出单」「分次缴费结构」的合规关系",
        ],
    }
    (OUT / "s14_报行合一配置.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n[已写出] s14_保费结构分解.csv / s14_附加费率上限敏感.csv / s14_重定价备案路径.csv")
    print("         s14_备案区间覆盖.csv / s14_报行合一要件对照.csv / s14_报行合一验收.csv")
    print("         s14_报行合一配置.json")


if __name__ == "__main__":
    main()
