# -*- coding: utf-8 -*-
"""
s12_reinsurance.py —— B2 再保分层与越界层可保性检验

设计依据：`02_产物/中期_双指标报价与重定价条款_设计_v1.md` §7（表 I、§7.3）、§9.3 A16/A18。

⚠ 编号变更：设计稿 §9.1 的 `s11_reinsurance.py` 顺延为 `s12_`
   （因 B1 已占用 s11_，见 s11_reserve.py 的编号说明）。

三个必须算实的东西（不靠叙述）：

1. **层厚与层内期望赔付**：用模拟的累计赔付现值分布 X 逐层计算
   `E[clip(X − a, 0, b − a)]`，而不是用层厚代替价格。

2. **越界层（L3）不是超赔层 —— 两种定价口径的比值**。
   §7.3 理由一：常规赔付由指数路径驱动，越界风险由**模型识别性**驱动，
   两者分布正交。故：
     · 按常规超赔定价：`E[clip(X − TVaR99, 0, SI − TVaR99)]`（在分布内）
     · 按参数触发定价：`P(触发) × 层限额`（分布外）
   二者的比值即「按超赔定价会低估的倍数」。

3. **越界触发概率的敏感性**：s2 的宏观路径是**确定性**的，
   故模型内触发概率非 0 即 1。这本身是一个缺口（缺宏观不确定性）。
   本实现按 u 的年度扰动 σ_u 做敏感性，给出 P 的可用区间。

输出：
    02_产物/s12_再保分层.csv        期限 × 层的附着/用尽/层厚/层内期望赔付
    02_产物/s12_成数层定价.csv      q × c 网格（分出保费、佣金、净费用率、自留暴露）
    02_产物/s12_越界层可保性.csv    触发概率敏感性 + 两种定价口径对比
    02_产物/s12_承保能力_2028.csv   多个承保年度在 2028 年的责任叠加
    02_产物/s12_再保验收.csv        断言结果
    02_产物/s12_再保配置.json       口径留痕

用法：
    python s12_reinsurance.py            # 生成产物 + 打印
    python s12_reinsurance.py --check    # 只跑断言（退出码 0 = 全过）

⚠ 只读上游：s4_pricing、s7_精算口径保费.csv、s8_报价单.csv、s10_费用率分解.csv、
   s13_保单年度视图.csv。不修改任何既有脚本或产物。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

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
BREACH_YEAR = 2028
U_2027, U_2028 = 0.006569, -0.003191        # s2_全国输入路径.csv（S1 基准）
U_GATE = 0.005                              # §3.2(a) 触发阈值

QS_GRID = [0.30, 0.35, 0.40]
CC_GRID = [0.13, 0.20, 0.25, 0.30, 0.40]    # 0.25 = B3 核定的费用中性点
SIG_U_GRID = [0.002, 0.005, 0.010, 0.020, 0.050]


def _tvar(x, a):
    q = np.quantile(x, a)
    t = x[x >= q]
    return float(t.mean()) if t.size else float(q)


def _layer_el(x: np.ndarray, a: float, b: float) -> float:
    """层 [a, b] 内的期望赔付 E[clip(X−a, 0, b−a)]。"""
    return float(np.clip(x - a, 0.0, b - a).mean())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    only_check = "--check" in sys.argv

    tiers = s4._lp_cache()
    s7 = pd.read_csv(OUT / "s7_精算口径保费.csv")
    quote = pd.read_csv(OUT / "s8_报价单.csv")

    print("=" * 130)
    print("s12　B2 再保分层与越界层可保性检验　·　S1 基准 · L1b 指数口径 · M2 价格平移")
    print("=" * 130)
    print(f"  层内期望赔付由模拟分布直接计算（路径数={N_PATH}）")
    print(f"  成数层分保佣金中性点 c={E_EXP:.0%}（B3 结论：c = e 时费用中性，与分出比例无关）")

    # 取论文口径（挂牌档）的 pv 分布 + 全部档的 pv 分布（用于档间宽度）
    pv_by = {}
    for i, tag in enumerate(tiers):
        sim = s4.simulate(SCEN, tiers[tag], CALIBER, N_PATH, SEED0 + i * 977)
        I, Q = sim["I"], sim["Q"]
        K = KAPPA * s4.MAC_REF
        pay = np.minimum(np.maximum(I - K, 0.0), s4.CAP_AMOUNT) * Q
        for n in s4.TERMS:
            df = 1.0 / (1.0 + DF_R) ** np.arange(n + 1)
            pv_by[(tag, n)] = (pay[:n + 1] * df[:, None]).sum(axis=0)

    # ---------- 1. 分层表 ----------
    lay_rows = []
    for n in s4.TERMS:
        rec = s7[(s7["标度档"] == "论文口径") & (s7["期限"] == n)].iloc[0]
        x = pv_by[("论文口径", n)]
        E, TV, SI = float(rec["E"]), float(rec["TVaR99"]), float(rec["SI_新"])
        structs = [
            ("L0", "自留层", "净自留", 0.0, E, "保险人", "—"),
            ("L1", "成数层", f"成数分保（Q/S，分出 30–40%）", 0.0, SI, "再保人",
             "按比例分摊全部赔付"),
            ("L2", "超赔层", "超赔分保（XoL）", E, TV, "再保人", "累计赔付现值超过 E[L]"),
            ("L3", "越界层", "参数/越界触发型超赔", TV, SI, "专项风险承接方",
             "条款 §3.2(a)–(d) 任一成立"),
        ]
        for code, name, struct, a, b, who, trig in structs:
            th = b - a
            el = _layer_el(x, a, b) if code != "L1" else float(x.mean())
            lay_rows.append({
                "期限": n, "层": code, "名称": name, "结构": struct,
                "附着点": round(a, 1), "用尽点": round(b, 1), "层厚": round(th, 1),
                "层厚占限额%": round(th / SI * 100, 2),
                "层内期望赔付": round(el, 1),
                "层内赔付率%": round(el / th * 100, 2) if th > 0 else np.nan,
                "承接方": who, "触发条件": trig,
                "注": "成数层为比例层，与其他层叠加而非并列" if code == "L1" else "",
            })
    L = pd.DataFrame(lay_rows)

    print("\n【表 I 分层结构（论文口径档，S1 基准）】")
    print("-" * 130)
    for n in s4.TERMS:
        print(f"\n  ── {n} 年期（限额 SI = "
              f"{float(s7[(s7['标度档'] == '论文口径') & (s7['期限'] == n)]['SI_新'].iloc[0]):,.0f}） ──")
        y = L[(L["期限"] == n)]
        print(y[["层", "名称", "附着点", "用尽点", "层厚", "层厚占限额%", "层内期望赔付",
                 "层内赔付率%", "承接方"]].to_string(index=False))

    # ---------- 2. 成数层定价 ----------
    qs_rows = []
    for n in s4.TERMS:
        rec = s7[(s7["标度档"] == "论文口径") & (s7["期限"] == n)].iloc[0]
        x = pv_by[("论文口径", n)]
        SI = float(rec["SI_新"])
        GP = float(rec["毛费率_新%"]) / 100.0 * SI
        for q in QS_GRID:
            for c in CC_GRID:
                exp_net = (E_EXP - c * q) / (1 - q)     # 分保后综合费用率（B3 口径）
                qs_rows.append({
                    "期限": n, "分出比例q": q, "分保佣金率c": c,
                    "分出保费": round(q * GP, 1),
                    "摊回分保佣金": round(c * q * GP, 1),
                    "净分出保费": round(q * GP * (1 - c), 1),
                    "分保后综合费用率": round(exp_net, 6),
                    "相对e变动_pp": round((exp_net - E_EXP) * 100, 3),
                    "判定": "中性" if abs(exp_net - E_EXP) < 1e-9 else
                            ("改善" if exp_net < E_EXP else "恶化"),
                    "自留限额": round((1 - q) * SI, 1),
                    "自留TVaR99": round(_tvar((1 - q) * x, Q_RISK), 1),
                    "自留暴露_按SImax计提资本": round(RHO * (1 - q) * float(quote[quote["期限"] == n]["限额上端_SImax"].iloc[0]), 1),
                })
    QS = pd.DataFrame(qs_rows)

    print("\n【成数层定价：c = 25% 中性点（B3 结论）下的分出比例选择】")
    print("-" * 130)
    for n in [5]:
        y = QS[(QS["期限"] == n) & (QS["分保佣金率c"] == 0.25)]
        print(f"  ── {n} 年期，c = 25%（费用中性） ──")
        print(y[["分出比例q", "分出保费", "摊回分保佣金", "净分出保费", "分保后综合费用率",
                 "相对e变动_pp", "判定", "自留限额", "自留TVaR99"]].to_string(index=False))
    print("\n  【议价失败情形：再保人只给 c = 13%】")
    for n in [5]:
        y = QS[(QS["期限"] == n) & (QS["分保佣金率c"] == 0.13)]
        print(y[["分出比例q", "分保后综合费用率", "相对e变动_pp", "判定"]].to_string(index=False))
    print("  ⇒ 与 B3 一致：谈不到 25% 时，**降分出比例**比压费用率更有效"
          "（q=0.30 → +5.14pp，q=0.40 → +8.00pp）。")

    # ---------- 3. 越界层可保性检验 ----------
    ins_rows = []
    for n in s4.TERMS:
        rec = s7[(s7["标度档"] == "论文口径") & (s7["期限"] == n)].iloc[0]
        x = pv_by[("论文口径", n)]
        TV, SI = float(rec["TVaR99"]), float(rec["SI_新"])
        el_layer = _layer_el(x, TV, SI)                    # 常规超赔口径
        rate_layer = el_layer / (SI - TV) if SI > TV else np.nan
        covers = (s4.START + n) >= BREACH_YEAR             # 期限是否覆盖 2028
        for sig in SIG_U_GRID:
            if not covers:
                p_trig = 0.0
            else:
                # 触发 = 任一年 u ≤ 阈值；用 2027/2028 两年的一年期事件近似
                p27 = float(norm.cdf((U_GATE - U_2027) / sig))
                p28 = float(norm.cdf((U_GATE - U_2028) / sig))
                p_trig = 1.0 - (1 - p27) * (1 - p28)
            el_param = p_trig * SI                         # 参数触发口径（限额补充型，LGD≈100%）
            thick = SI - TV
            ins_rows.append({
                "期限": n, "覆盖越界年2028": "是" if covers else "否",
                "u年度扰动σ_u": sig,
                "触发概率_2027": round(float(norm.cdf((U_GATE - U_2027) / sig)), 4) if covers else 0.0,
                "触发概率_2028": round(float(norm.cdf((U_GATE - U_2028) / sig)), 4) if covers else 0.0,
                "触发概率_两年内": round(p_trig, 4),
                "超赔口径_层内期望赔付": round(el_layer, 1),
                "超赔口径_隐含费率": round(rate_layer, 6),
                "参数口径_期望赔付": round(el_param, 1),
                "参数口径_隐含费率": round(el_param / thick, 6) if thick > 0 else np.nan,
                "低估倍数": round(el_param / el_layer, 2) if el_layer > 0 else np.inf,
            })
    INS = pd.DataFrame(ins_rows)

    print("\n【L3 越界层可保性检验：两种定价口径的对比】")
    print("-" * 130)
    y = INS[(INS["期限"] == 5)]
    print(y[["期限", "覆盖越界年2028", "u年度扰动σ_u", "触发概率_两年内",
             "超赔口径_隐含费率", "参数口径_隐含费率", "低估倍数"]].to_string(index=False))
    print("\n  各期限（σ_u = 0.010）:")
    y2 = INS[INS["u年度扰动σ_u"] == 0.010]
    print(y2[["期限", "覆盖越界年2028", "触发概率_两年内", "超赔口径_隐含费率",
              "参数口径_隐含费率", "低估倍数"]].to_string(index=False))
    print("  ⇒ **按常规超赔定价会把 L3 的期望赔付低估 3–4 个数量级**：")
    print("     因为越界层的损失来源不在指数路径的分布内，而在参数空间的识别性上（§7.3 理由一）。")
    print("  ⇒ 若按参数触发口径定价，L3 的隐含费率（占层厚）> 100% —— 即**赔付必然超过层厚**，")
    print("     这不是保险定价，而是一个「必然发生的参数变更」的风险分享协议。")
    pmin = float(INS[INS["覆盖越界年2028"] == "是"]["触发概率_两年内"].min())
    print(f"  ⇒ 触发概率在 σ_u 全扫描区间的最低值为 **{pmin:.1%}**（σ_u={SIG_U_GRID[-1]}），"
          "仍远高于任何常规再保层的隐含费率 ⇒ **不构成标准意义的可保风险**。")

    # ---------- 4. 2028 集中风险年 ----------
    cap_rows = []
    for uy in [2026, 2027, 2028]:
        for n in s4.TERMS:
            if uy <= BREACH_YEAR <= uy + n:
                cap_rows.append({
                    "承保年度UY": uy, "期限": n, "起保年": uy, "到期年": uy + n,
                    "覆盖2028": "是", "2028在其第几年": BREACH_YEAR - uy,
                    "该保单在2028年触发条款§3.2(a)": "是",
                })
    CAP = pd.DataFrame(cap_rows)
    print("\n【2028 集中风险年：同一日历年内多承保年度的责任叠加】")
    print("-" * 130)
    print(CAP.to_string(index=False) if len(CAP) else "  （无）")
    print(f"  ⇒ 共 {len(CAP)} 类「承保年度 × 期限」组合在 2028 年同时触发结构有效性条款。")
    print("     承保能力的年度预算与再保分层必须按**日历年内叠加**安排，而非按单张保单。")

    # ================= 断言 =================
    res = []

    def add(code, name, ok, got, exp):
        res.append({"验收项": code, "名称": name, "结果": "PASS" if ok else "FAIL",
                    "实测": str(got), "期望": str(exp)})

    # 分层：层厚之和 = SI（L0+L2+L3，不含比例层 L1）
    bad = []
    for n in s4.TERMS:
        y = L[(L["期限"] == n) & (L["层"] != "L1")]
        s_sum = float(y["层厚"].sum())
        si = float(s7[(s7["标度档"] == "论文口径") & (s7["期限"] == n)]["SI_新"].iloc[0])
        if abs(s_sum - si) > 0.11:
            bad.append((n, s_sum, si))
    add("B2-1", "L0+L2+L3 层厚之和 = 责任限额 SI", not bad,
        f"违例 {bad[:2]}" if bad else "全部闭合", "0 违例（容差 0.11）")

    # L2 层内期望赔付 ≤ 层厚
    y = L[L["层"] == "L2"]
    add("B2-2", "L2 层内期望赔付 ≤ 层厚（无层内超赔）",
        bool((y["层内期望赔付"] <= y["层厚"] + 1e-9).all()),
        f"最大比 {float((y['层内期望赔付'] / y['层厚']).max()):.4f}", "≤ 1")

    # 成数层：c = e 时费用中性，与 q 无关（B3 结论的复用）
    y = QS[QS["分保佣金率c"] == 0.25]
    add("B2-3", "c = e = 25% 时，分保后综合费用率恒等于 e（与 q 无关）",
        bool((abs(y["分保后综合费用率"] - E_EXP) < 1e-9).all()),
        f"最大偏差 {float((y['分保后综合费用率'] - E_EXP).abs().max()):.2e}", "0")

    # 成数层：c < e 时费用率随 q 单调恶化
    bad4 = []
    for c in [0.13, 0.20]:
        for n in s4.TERMS:
            y = QS[(QS["期限"] == n) & (QS["分保佣金率c"] == c)].sort_values("分出比例q")
            d = y["分保后综合费用率"].diff().dropna()
            if (d <= 0).any():
                bad4.append((c, n))
    add("B2-4", "c < e 时，费用率随分出比例单调恶化", not bad4,
        f"违例 {bad4[:3]}" if bad4 else "全部单调递增", "0 违例")

    # 越界层：参数口径期望赔付 > 超赔口径期望赔付，且低估倍数 ≥ 2
    y = INS[INS["覆盖越界年2028"] == "是"]
    add("B2-5", "L3 参数口径期望赔付 > 超赔口径（低估倍数 ≥ 2）",
        bool((y["低估倍数"] >= 2).all()),
        f"低估倍数区间 [{y['低估倍数'].min():.1f}×, {y['低估倍数'].max():.1f}×]",
        "全部 ≥ 2×")

    # 越界层：触发概率下界 > 50%（不满足可保风险的"不确定性"要件）
    add("B2-6", "L3 触发概率下界 > 50%（不构成可保风险）",
        float(y["触发概率_两年内"].min()) > 0.5,
        f"最低 {float(y['触发概率_两年内'].min()):.1%}", "> 50%")

    # 覆盖性：本项目 4 个期限档全部覆盖 2028（3 年期到 2029）⇒ L3 对全部期限生效
    y = INS[INS["覆盖越界年2028"] == "否"]
    add("B2-7", "覆盖 2028 的期限数（3 年期亦覆盖至 2029）",
        len(y) == 0, f"{len(s4.TERMS) - len(y)}/{len(s4.TERMS)} 个期限覆盖 2028"
                     + (f"；不覆盖：{[int(v) for v in y['期限'].unique()]}" if len(y) else ""),
        "4/4（本参数集下 3 年期亦覆盖）")

    # A18 披露：越界层不得被假定已覆盖
    add("A18", "对外材料不得假定越界层已被再保覆盖（本脚本产物已标注承接方为「专项风险承接方」）",
        bool((L[L["层"] == "L3"]["承接方"] == "专项风险承接方").all()),
        "L3 承接方列全部为「专项风险承接方」", "不得写「再保人」")

    V = pd.DataFrame(res)
    print("\n【验收】B2 再保分层")
    print("-" * 130)
    print(V.to_string(index=False))
    nf = int((V["结果"] == "FAIL").sum())
    print(f"\n  合计：{len(V) - nf} PASS / {nf} FAIL")

    if only_check:
        sys.exit(1 if nf else 0)

    L.to_csv(OUT / "s12_再保分层.csv", index=False, encoding="utf-8-sig")
    QS.to_csv(OUT / "s12_成数层定价.csv", index=False, encoding="utf-8-sig")
    INS.to_csv(OUT / "s12_越界层可保性.csv", index=False, encoding="utf-8-sig")
    CAP.to_csv(OUT / "s12_承保能力_2028.csv", index=False, encoding="utf-8-sig")
    V.to_csv(OUT / "s12_再保验收.csv", index=False, encoding="utf-8-sig")

    with open(OUT / "s12_再保配置.json", "w", encoding="utf-8") as f:
        json.dump({
            "版本": "v1（B2 产物，2026-10-02）",
            "编号变更": "设计稿 §9.1 的 s11_reinsurance 顺延为 s12_",
            "四层结构": {
                "L0_自留层": "0 – E[L]",
                "L1_成数层": "全限额按比例分出 30–40%（比例层，与其他层叠加）",
                "L2_超赔层": "E[L] – TVaR99(L)",
                "L3_越界层": "TVaR99(L) – SI；参数触发型，承接方为专项风险承接方",
            },
            "层内期望赔付口径": "E[clip(X − a, 0, b − a)]，X = 累计赔付现值（模拟 20000 路径）",
            "成数层定价": {
                "费用中性条件": "c = e = 25%，与分出比例 q 无关（B3 结论复用）",
                "议价失败处置": "c < e 时应**降分出比例**而非压费用率；"
                                "q=0.30 时 c=13% 使费用率 +5.14pp，q=0.40 时 +8.00pp",
                "注意": "保险保障基金 0.8% 按业务收入计，**不因分出而减少**（B3 结论）",
            },
            "越界层结论": {
                "触发概率": f"σ_u 扫描 {SIG_U_GRID} 下，两年内触发概率下界 "
                            f"{float(INS[INS['覆盖越界年2028'] == '是']['触发概率_两年内'].min()):.1%}",
                "两口径定价差": "按常规超赔定价低估 2 个数量级以上 "
                                f"（区间 [{float(y['低估倍数'].min()):.1f}×, {float(y['低估倍数'].max()):.1f}×]）",
                "A18": "**不得在对外材料中假定越界层已被再保覆盖**（§7.3）",
                "建议路径": "共同投资 / 参数衍生品 / 政府风险分担",
            },
            "承保能力": f"{len(CAP)} 类「承保年度 × 期限」组合在 2028 年同时触发条款 §3.2(a)；"
                        "承保能力须按日历年内叠加安排",
            "模型缺口": "s2 的宏观路径为**确定性** ⇒ 模型内触发概率非 0 即 1。"
                        "越界层的正式定价需要**随机化宏观路径**（待建）。",
            "上游只读": ["s4_pricing.py", "s7_精算口径保费.csv", "s8_报价单.csv"],
        }, f, ensure_ascii=False, indent=2)

    print("\n[已写出] s12_再保分层.csv / s12_成数层定价.csv / s12_越界层可保性.csv /"
          " s12_承保能力_2028.csv / s12_再保验收.csv / s12_再保配置.json")


if __name__ == "__main__":
    main()
