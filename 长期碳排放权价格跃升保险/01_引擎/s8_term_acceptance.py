# -*- coding: utf-8 -*-
"""
s8_term_acceptance.py —— A2 期限一致性验收（固化断言，防静默回归）

上游：# 只读
    02_产物/s7_精算口径保费.csv         （新精算口径）
    02_产物/s5_期限一致性.csv           （旧口径对照，用于登记修补前后差异）
下游产物：
    02_产物/s8_期限一致性验收.csv       （标度档 × 期限 的弹性矩阵与判据）

设计依据：`02_产物/中期_双指标报价与重定价条款_设计_v1.md` §4.4（表 F、V1–V4）。
验收条件：同文 §9.3 的 A6（费率弹性区间）、A7（弹性单调性）、A8（费率随期限递减）。

⚠ 为什么必须写成脚本：设计稿 §4.3 的三条结论（费率宽度 1.100×、
   弹性 [−1.062, −0.996]、衰减 0.014）在实现前只是**叙述**。若不固化为断言，
   后续任何参数改动（如 ρ、π、σ 或网格取点规则）都可能使它们静默失效而无人察觉。

🔴 本脚本对设计稿的一处**实质性更正**（关系 V4 的定义，见文件末「与设计稿的偏离」）：
   设计稿表 F 标注为「赔付现值弹性」的一列，数值 1.271 / 1.326 / 1.364，
   经复核**不是 E（期望赔付现值）的弹性**（实为 1.687 / 1.867 / 1.908），
   而是 **P99.5（累计赔付现值 99.5% 分位）的期限弹性**。
   因此 V4 的判据须由「赔付现值弹性 ≤ SI 弹性」改写为
   **「毛保费弹性 ≤ SI 弹性」**（后者才等价于费率随期限递减，且非恒真）。

用法：
    python s8_term_acceptance.py            # 生成产物 + 打印验收
    python s8_term_acceptance.py --check    # 仅跑断言，退出码 0 = 通过
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import OUT                                    # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.unicode.east_asian_width", True)

TERMS = [3, 5, 10, 15]
BASE = 3

# 验收阈值（设计稿 §4.4 V1/V2 + §4.5 降级 1 的例外条款）
V1_LO, V1_HI = -1.10, -0.95          # 费率期限弹性允许区间
V2_SLACK = 0.03                      # 15y 弹性不得低于 5y 弹性 − 0.03
V2_MAX_EXCEPT = 2                    # 降级 1：越界档位数 ≤ 2/9 时改为主承保档验收
MAIN_TIERS = ["论文口径", "段1左端_最保守", "段1偏左", "段1中段", "段1偏右"]
TOL_ID = 5e-4                        # 恒等式容差（SI 在 s7 中四舍五入为整数）


def _elas(v: pd.Series, n: int) -> float:
    """幂律弹性 ln(v_n/v_base) / ln(n/base)。"""
    a, b = float(v.loc[n]), float(v.loc[BASE])
    if a <= 0 or b <= 0:
        raise ValueError("弹性输入必须为正")
    return float(np.log(a / b) / np.log(n / BASE))


def build(T: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    tiers = list(dict.fromkeys(T["标度档"]))
    for tag in tiers:
        x = T[T["标度档"] == tag].set_index("期限").sort_index()
        gp = (x["SI_新"] * x["毛费率_新%"] / 100.0).rename("GP")
        for n in TERMS:
            r = x.loc[n]
            rows.append({
                "标度档": tag, "期限": n,
                "E_期望累计赔付现值": r["E"],
                "P99.5": r["P99.5"], "SI_新": r["SI_新"],
                "毛保费GP_元每吨": round(float(gp.loc[n]), 0),
                "毛费率_新%": r["毛费率_新%"],
                # 四个弹性：E / P99.5 / SI / GP。费率弹性 = GP弹性 − SI弹性（恒等式）
                "E弹性": round(_elas(x["E"], n), 4),
                "P99.5弹性": round(_elas(x["P99.5"], n), 4),
                "SI弹性": round(_elas(x["SI_新"], n), 4),
                "GP弹性": round(_elas(gp, n), 4),
                "费率弹性": round(_elas(x["毛费率_新%"], n), 4),
                "恒等式_SI减P99.5": round(_elas(x["SI_新"], n) - _elas(x["P99.5"], n), 6),
                "恒等式_费率对GP减SI": round(
                    _elas(x["毛费率_新%"], n) - (_elas(gp, n) - _elas(x["SI_新"], n)), 9),
                "弹性判据": "基期" if n == BASE else (
                    "∈[−1.10,−0.95]" if V1_LO <= _elas(x["毛费率_新%"], n) <= V1_HI else "越界"),
            })
    D = pd.DataFrame(rows)

    per = {}
    for tag in tiers:
        y = D[D["标度档"] == tag].set_index("期限")
        per[tag] = {
            "弹性_5y": float(y.loc[5, "费率弹性"]),
            "弹性_10y": float(y.loc[10, "费率弹性"]),
            "弹性_15y": float(y.loc[15, "费率弹性"]),
            "衰减_15y减5y": float(y.loc[15, "费率弹性"] - y.loc[5, "费率弹性"]),
            "V1_通过": bool(y.loc[[5, 10, 15], "费率弹性"].between(V1_LO, V1_HI).all()),
            "V2_通过": bool(float(y.loc[15, "费率弹性"]) >= float(y.loc[5, "费率弹性"]) - V2_SLACK),
            "V3_通过": bool((y["毛费率_新%"].diff().dropna() < 0).all()),
            # V4（更正版）：毛保费弹性 ≤ SI 弹性 ⇔ 费率弹性 ≤ 0
            "V4_通过": bool((y.loc[[5, 10, 15], "GP弹性"]
                             <= y.loc[[5, 10, 15], "SI弹性"] + 1e-9).all()),
        }
    return D, per


def check(D: pd.DataFrame, per: dict) -> list[dict]:
    res = []

    # V1：全部 9 档 × 3 期限对
    all_el = D[D["期限"] != BASE]["费率弹性"]
    ok1 = bool(all_el.between(V1_LO, V1_HI).all())
    bad1 = D[(D["期限"] != BASE) & (~D["费率弹性"].between(V1_LO, V1_HI))]
    res.append({"验收项": "A6/V1", "判据": f"全部 9 档 × 3 期限对，费率弹性 ∈ [{V1_LO}, {V1_HI}]",
                "结果": "PASS" if ok1 else "FAIL",
                "实测": f"[{all_el.min():.3f}, {all_el.max():.3f}]（{len(all_el)} 个）",
                "期望": f"全部 ∈ [{V1_LO}, {V1_HI}]"})
    if not ok1:
        nb = len(bad1["标度档"].unique())
        res.append({"验收项": "A6/降级1", "判据": f"越界档位数 ≤ {V2_MAX_EXCEPT}/9 时改为主承保档验收",
                    "结果": "PASS" if nb <= V2_MAX_EXCEPT else "FAIL",
                    "实测": f"越界档 = {sorted(bad1['标度档'].unique())}",
                    "期望": f"≤ {V2_MAX_EXCEPT} 档"})

    # V2：弹性不随期限衰减
    n2 = sum(1 for v in per.values() if v["V2_通过"])
    res.append({"验收项": "A7/V2", "判据": f"每档 15y 弹性 ≥ 5y 弹性 − {V2_SLACK}",
                "结果": "PASS" if n2 == len(per) else "FAIL",
                "实测": f"{n2}/{len(per)} 档通过；论文口径衰减 {per['论文口径']['衰减_15y减5y']:+.3f}",
                "期望": "全部档位通过"})

    # V3：毛费率随期限严格递减
    n3 = sum(1 for v in per.values() if v["V3_通过"])
    res.append({"验收项": "A8/V3", "判据": "全部 9 档毛费率随期限严格递减",
                "结果": "PASS" if n3 == len(per) else "FAIL",
                "实测": f"{n3}/{len(per)} 档通过", "期望": "全部档位通过"})

    # V4（更正版）：毛保费弹性 ≤ SI 弹性
    n4 = sum(1 for v in per.values() if v["V4_通过"])
    res.append({"验收项": "V4′", "判据": "毛保费弹性 ≤ SI 弹性（⇔ 费率弹性 ≤ 0），全部档位",
                "结果": "PASS" if n4 == len(per) else "FAIL",
                "实测": f"{n4}/{len(per)} 档通过", "期望": "全部档位通过"})

    # 恒等式 1：SI = P99.5 × n ⇒ SI弹性 = P99.5弹性 + 1
    d1 = D[D["期限"] != BASE]["恒等式_SI减P99.5"]
    okid1 = bool((d1 - 1.0).abs().max() <= TOL_ID)
    res.append({"验收项": "ID1", "判据": "SI = P99.5 × n ⇒ SI弹性 − P99.5弹性 = 1（结构恒等式）",
                "结果": "PASS" if okid1 else "FAIL",
                "实测": f"最大偏差 {float((d1 - 1.0).abs().max()):.2e}", "期望": f"≤ {TOL_ID:.0e}"})

    # 恒等式 2：费率 = GP/SI ⇒ 费率弹性 = GP弹性 − SI弹性
    d2 = D["恒等式_费率对GP减SI"].abs()
    okid2 = bool(d2.max() <= 1e-9)
    res.append({"验收项": "ID2", "判据": "费率 = GP/SI ⇒ 费率弹性 = GP弹性 − SI弹性（分解恒等式）",
                "结果": "PASS" if okid2 else "FAIL",
                "实测": f"最大偏差 {float(d2.max()):.2e}", "期望": "≤ 1e-9"})

    # 回归哨兵：论文口径档的锚值（设计稿 §4.3/§4.4 的引用值）
    y = D[D["标度档"] == "论文口径"].set_index("期限")
    sent = {
        "SI_3": (float(y.loc[3, "SI_新"]), 12396, 1.0),
        "SI_5": (float(y.loc[5, "SI_新"]), 39536, 1.0),
        "SI_10": (float(y.loc[10, "SI_新"]), 203863, 1.0),
        "SI_15": (float(y.loc[15, "SI_新"]), 556371, 1.0),
        "费率_3": (float(y.loc[3, "毛费率_新%"]), 55.91, 0.005),
        "费率_5": (float(y.loc[5, "毛费率_新%"]), 33.04, 0.005),
        "费率_10": (float(y.loc[10, "毛费率_新%"]), 16.43, 0.005),
        "费率_15": (float(y.loc[15, "毛费率_新%"]), 10.89, 0.005),
        "弹性_5": (float(y.loc[5, "费率弹性"]), -1.0298, 0.0005),
        "弹性_10": (float(y.loc[10, "费率弹性"]), -1.0172, 0.0005),
        "弹性_15": (float(y.loc[15, "费率弹性"]), -1.0164, 0.0005),
    }
    bad = [f"{k}({got:.4f}≠{exp})" for k, (got, exp, tol) in sent.items() if abs(got - exp) > tol]
    res.append({"验收项": "SENTRY", "判据": "论文口径档 11 个锚值不得漂移（回归哨兵）",
                "结果": "PASS" if not bad else "FAIL",
                "实测": "全部锚值一致" if not bad else f"漂移：{bad}",
                "期望": "11/11 一致（容差见脚本）"})
    return res


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    only_check = "--check" in sys.argv
    T = pd.read_csv(OUT / "s7_精算口径保费.csv")
    D, per = build(T)
    res = check(D, per)

    print("=" * 118)
    print("s8　A2 期限一致性验收（S1 基准 · L1b · κ=1.30 · M2 价格平移）")
    print("=" * 118)

    print("\n【表 F′】论文口径档：五个量的期限弹性（以 3 年为基）")
    print("-" * 118)
    y = D[D["标度档"] == "论文口径"].set_index("期限")
    print(y[["E_期望累计赔付现值", "P99.5", "SI_新", "毛保费GP_元每吨", "毛费率_新%",
             "E弹性", "P99.5弹性", "SI弹性", "GP弹性", "费率弹性"]].to_string())

    print("\n【逐档汇总】")
    print("-" * 118)
    S = pd.DataFrame(per).T
    print(S[["弹性_5y", "弹性_10y", "弹性_15y", "衰减_15y减5y",
             "V1_通过", "V2_通过", "V3_通过", "V4_通过"]].round(4).to_string())

    print("\n【旧口径对照（s5_期限一致性.csv，登记用）】")
    print("-" * 118)
    try:
        O = pd.read_csv(OUT / "s5_期限一致性.csv")
        cols = [c for c in ["期限", "赔付现值期限弹性", "毛费率期限弹性", "SI_现行"]
                if c in O.columns]
        print(O[cols].to_string(index=False))
        print("  ⇒ 旧口径毛费率弹性 0.69–0.91（长端被系统性**低估**）；新口径 −0.996 ~ −1.062。")
        print("  ⇒ 旧口径的「赔付现值期限弹性」1.69–1.91 是 E 的弹性；设计稿表 F 新口径下")
        print("     标注「赔付现值弹性」的 1.271–1.364 实为 P99.5 的弹性 —— 同名不同义，见文末偏离说明。")
    except Exception as e:                                   # noqa: BLE001
        print(f"  [WARN] 未能读取 s5_期限一致性.csv：{e}")

    print("\n【验收】A6/A7/A8 + V4′ + ID1/ID2 + 回归哨兵")
    print("-" * 118)
    for r in res:
        print(f"  [{r['结果']:4s}] {r['验收项']:9s} {r['判据']}　→ {r['实测']}")
    nfail = sum(1 for r in res if r["结果"] == "FAIL")
    print(f"\n  合计：{len(res) - nfail} PASS / {nfail} FAIL")

    if only_check:
        sys.exit(1 if nfail else 0)

    D.to_csv(OUT / "s8_期限一致性验收.csv", index=False, encoding="utf-8-sig")
    print("\n[已写出] s8_期限一致性验收.csv")


# ----------------------------------------------------------------------
# 与设计稿的偏离（须在汇报中显式说明，不得静默）
# ----------------------------------------------------------------------
# 1) V4 的判据对象：设计稿 §4.4 表 F 标注「赔付现值弹性」的一列，其值 1.271 / 1.326 /
#    1.364（5/10/15 年）与 E（期望累计赔付现值）的弹性 1.687 / 1.867 / 1.908 **不符**，
#    而与 P99.5（99.5% 分位）的弹性 1.271 / 1.326 / 1.364 **逐位吻合**。
#    ⇒ 该列名指向有误。V4「赔付现值弹性 ≤ SI 弹性」若按字面（E 弹性）解释，则
#      1.687 ≤ 2.271 成立但**无判别力**（因 SI = P99.5×n 使 SI弹性 = P99.5弹性 + 1，
#      而 E 弹性与 SI 弹性的大小关系并非恒真）；若按实际数值（P99.5 弹性）解释，
#      则它**恒真**（相差恰好 1），同样无判别力。
#    ⇒ 本脚本把 V4 更正为**有意义的那条结构条件**：毛保费弹性 ≤ SI 弹性。
#      理由：费率弹性 ≡ GP弹性 − SI弹性（ID2 已逐位验证），故「费率随期限递减」
#      等价于「GP弹性 ≤ SI弹性」。这条**非恒真**——若有人把限额口径改回常量×n，
#      SI弹性 会降到 1.0 而 GP弹性 仍在 1.24–1.35，费率弹性会转正，V4′ 立即报错。
# 2) 新增 ID1/ID2 两条**结构恒等式**断言。它们把「限额口径 = P99.5 × n」与
#    「费率 = GP/SI」这两个定义钉死，使 D1 类缺陷（限额定义被改动）无法静默复发。
# 3) 设计稿 §4.4 表 F 的「SI 弹性 2.271/2.326/2.364」经复核**正确**（2.2705/2.3257/2.3636）；
#    「毛费率 55.91/33.04/16.43/10.89」**正确**；「弹性 5y −1.030、10y −1.017、15y −1.016」
#    **正确**（−1.0298/−1.0172/−1.0164）；§4.3 的「全 27 个弹性值 ∈ [−1.062, −0.996]」**正确**。
#    ⇒ 设计稿的**结论全部成立**，仅「赔付现值弹性」一列的名实不符需更正。
# 4) 全部改动只发生在验收脚本层；s4_pricing.py / s7_rebuild.py **未被修改**。


if __name__ == "__main__":
    main()
