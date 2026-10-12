# -*- coding: utf-8 -*-
"""
s22_commercial_reprice.py —— L2「商业版」重构定价：可保性修复工作包

背景（2026-10-03，用户判断「没有企业愿意买」，经核成立）：
    现行产品（D=0、无截断、独立档定价、e=25%）绝对保费 ≈ 12× 期望损失、
    ≈ 3× 标的自身价值（GP/P99.5 = 1.63–1.68、RM/E = 2.98–8.73）。
    10-02 裁定（限额口径勘误 §5.9）已否决「换风险度量压费率」——
    本脚本**不改度量，改结构**：把风险真实地移出直接保单，再按原度量定价。

四个结构杠杆（全部与铁律相容，风险度量不动）：
    L1 聚合免赔 D：累计赔付现值 L ≤ D 由被保险人自留（D 取 L 的 q50/q75/q90）。
        免赔档是显式条款，客户自留额在报价单同列披露。
    L2 责任上限 A：A = q99.5(L)（铁律口径；按 10-02 裁定不乘 n）。
    L3 组合分散 φ：RM_保单 = (TVaR99(L)−E[L]) × φ，φ = sqrt(ρ̄ + (1−ρ̄)/n_eff)。
        现行费率的根：把**单一保单**当作保险人的全部组合来计提尾部资本。
        分散是保险存在的前提；RM 仍是损失分布的泛函 + 显式组合结构参数，
        与「不可常量乘数」铁律不冲突（φ 随组合结构披露，非隐藏常数）。
        ⚠ 正态幂近似，待组合层联合模拟（后续工作包）复核；ρ̄、n_eff 全表披露。
    L4 指数产品费用率 e=12%：无查勘定损/无现场理赔（指数自动触发），
        「报行合一」下界 = 实际费用率 ⇒ 须以实际成本结构支撑后方可备案（待核）。
        e=25% 现行口径并列输出，不静默替换。
触发结构（并列，不混用）：
    A_水平式：K = 1.30 × MAC_REF 固定（现行 s7 骨架，保证可比与复现）。
    B1_年度重定基：pay_t = min(max(I_t − 1.30·I_{t−1}, 0), CAP)·Q_t，t = 1..n。
        模糊性进条款（水平档差在比例中约掉）；与「2026 起保重定价周期=1 年」一致。

登记缺陷（不改 s4，主线保持可复现）：
    s4.price_rebased 的分段循环把重定基边界年计入两段（period=3, n=5 时第 3 年
    在 seg1 与 seg2 各计一次），设计B 费率高估；方向性结论（区间收窄）不受影响。
    本脚本 B1 为无重复计数实现，口径偏离在此显式登记。

验证锚（V1）：D=0 无截断、e=25%、φ=1、A_水平式 ⇒ 必须逐档复现
    s7_精算口径保费.csv 的毛费率（同 seed 同公式，容差 0.05pp）。

用法：python s22_commercial_reprice.py [--check]    # 退出码 0 = 断言全过
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import L1, OUT                                  # noqa: E402
import s4_pricing as s4                                      # noqa: E402

pd.set_option("display.width", 240)
pd.set_option("display.unicode.east_asian_width", True)

RHO = L1.get("capital_charge", 0.06)      # 资本成本率 6%（不动）
PI_PROFIT = 0.08                          # 利润边际 8%（不动）
DF_R = L1["discount"]                     # 贴现 2.5%（不动）
E_EXP_OLD = L1["expense_ratio"]           # 0.25 现行费用率
E_EXP_IDX = 0.12                          # 指数产品费用率（待核备案）
Q_LIMIT = 0.995
D_ORDER = ["D0_无免赔", "D50_q50", "D75_q75", "D90_q90"]
D_PCT = {"D0_无免赔": None, "D50_q50": 0.50, "D75_q75": 0.75, "D90_q90": 0.90}
N_EFF_BASE = 100
RHO_BAR_BASE = 0.30
TERMS = s4.TERMS

STRUCT_A = "A_水平式"
STRUCT_B1 = "B1_年度重定基"


# ---------------------------------------------------------------- 赔付现值
def pv_level(sim: dict, n: int) -> np.ndarray:
    """A_水平式：与 s7_rebuild 逐位一致（K = κ×MAC_REF 固定，年 0..n）。"""
    I, Q = sim["I"], sim["Q"]
    K = s4.KAPPA_BASE * s4.MAC_REF
    pay = np.minimum(np.maximum(I - K, 0.0), s4.CAP_AMOUNT) * Q
    df = 1.0 / (1.0 + DF_R) ** np.arange(n + 1)
    return (pay[:n + 1] * df[:, None]).sum(axis=0)


def pv_rebased(sim: dict, n: int, period: int = 1) -> np.ndarray:
    """B 年度/周期重定基：段起点 yr 重定基 K=κ·I[yr]，覆盖 yr+1..yr+seg。
    无边界重复计数、无浪费年（与 s4.price_rebased 的两处口径差，见文件头登记）。"""
    I, Q = sim["I"], sim["Q"]
    total = np.zeros(I.shape[1])
    yr = 0
    while yr < n:
        seg = min(period, n - yr)
        idx = np.arange(yr + 1, yr + seg + 1)          # 覆盖 yr+1..yr+seg
        K = s4.KAPPA_BASE * I[yr]
        pay = np.minimum(np.maximum(I[idx] - K[None, :], 0.0), s4.CAP_AMOUNT) * Q[idx]
        df = 1.0 / (1.0 + DF_R) ** idx[:, None]
        total += (pay * df).sum(axis=0)
        yr += seg
    return total


def _tvar(x: np.ndarray, a: float = 0.99) -> float:
    q = np.quantile(x, a)
    t = x[x >= q]
    return float(t.mean()) if t.size else float(q)


def layer_pay(L: np.ndarray, D: float, A: float | None) -> np.ndarray:
    """层赔付：min(max(L−D,0), A−D)；A=None 表示不设 PV 级截断（仅复现 s7 用）。"""
    pay = np.maximum(L - D, 0.0)
    if A is not None:
        pay = np.minimum(pay, A - D)
    return pay


def phi_div(n_eff: int, rho_bar: float) -> float:
    """组合分散因子：sqrt(ρ̄ + (1−ρ̄)/n_eff)。n_eff=1 ⇒ 1（独立档）。"""
    return float(np.sqrt(rho_bar + (1.0 - rho_bar) / n_eff))


def gp_of(E: float, rm: float, e: float) -> float:
    """毛保费 = (E + RM(1+ρ)) / (1 − e − π)。与 s7 F2–F4 同构。"""
    return (E + rm * (1.0 + RHO)) / (1.0 - e - PI_PROFIT)


# ---------------------------------------------------------------- 主表
def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    tiers = s4._lp_cache()
    sims = {tag: s4.simulate("S1_基准", tiers[tag], "M2_价格平移",
                             L1["mc_paths"], s4.RNG_SEED + i * 977)
            for i, tag in enumerate(tiers)}

    rows, base_rows = [], []
    for tag, sim in sims.items():
        for n in TERMS:
            for struct in (STRUCT_A, STRUCT_B1):
                L = (pv_level(sim, n) if struct == STRUCT_A
                     else pv_rebased(sim, n, period=1))
                E_full = float(L.mean())
                qs = {p: float(np.quantile(L, p))
                      for p in (0.50, 0.75, 0.90, Q_LIMIT)}
                base_rows.append({
                    "结构": struct, "标度档": tag, "期限": n,
                    "E_full": E_full, "SD_full": float(L.std(ddof=1)),
                    "TVaR99_full": _tvar(L),
                    "q50": qs[0.50], "q75": qs[0.75], "q90": qs[0.90],
                    "A_q995": qs[Q_LIMIT],
                    "触发概率_full%": float((L > 0).mean() * 100),
                })
                for d_tag in D_ORDER:
                    p = D_PCT[d_tag]
                    D = 0.0 if p is None else qs[p]
                    A = qs[Q_LIMIT]
                    # D0 × A_水平式 = 现行产品：无 PV 级截断（复现 s7 用）
                    uncapped = (p is None and struct == STRUCT_A)
                    pay = layer_pay(L, D, None if uncapped else A)
                    E = float(pay.mean())
                    sd = float(pay.std(ddof=1))
                    tv = _tvar(pay)
                    rm1 = tv - E
                    trig = float((pay > 0).mean() * 100)
                    for e_tag, e in (("e25_现行", E_EXP_OLD), ("e12_指数", E_EXP_IDX)):
                        for neff in (1, N_EFF_BASE):
                            phi = phi_div(neff, RHO_BAR_BASE)
                            rm = rm1 * phi
                            gp = gp_of(E, rm, e)
                            # 口径乙（资本成本口径，并列披露不替换）：
                            # GP = (E + ρ·RM·φ)/(1−e−π)——资本持有成本只计一次，
                            # 消除「RM 附加 + ρ×RM」对同一笔资本的双重计提。
                            # 属新产品的保费原理选择 ⇒ 待拍板，甲口径并行输出。
                            gp_coc = (E + RHO * rm) / (1.0 - e - PI_PROFIT)
                            rows.append({
                                "结构": struct, "标度档": tag, "期限": n,
                                "免赔档": d_tag, "D_元每吨": round(D, 1),
                                "A_q995": round(A, 1),
                                "层宽": round(A - D, 1),
                                "无截断": bool(uncapped),
                                "触发概率%": round(trig, 2),
                                "E_layer": round(E, 1), "SD_layer": round(sd, 1),
                                "TVaR99_layer": round(tv, 1),
                                "RM_独立": round(rm1, 1), "φ": round(phi, 4),
                                "RM_组合": round(rm, 1),
                                "费用口径": e_tag, "e": e, "n_eff": neff,
                                "GP_元每吨": round(gp, 0),
                                "GP_CoC_元每吨": round(gp_coc, 0),
                                "GP_CoC_ROL%": round(
                                    gp_coc / max(A - D, 1e-9) * 100, 1),
                                "GP比E": round(gp / max(E, 1e-9), 2),
                                "GP比标的价值": round(gp / (s4.MAC_REF * n), 3),
                            })
    R = pd.DataFrame(rows)
    B = pd.DataFrame(base_rows)

    # ---- 报价单：每期限×结构，D75 + e12 + n100，审慎档=层宽下端所在档
    cur = R[(R["结构"] == STRUCT_A) & (R["免赔档"] == "D0_无免赔")
            & (R["费用口径"] == "e25_现行") & (R["n_eff"] == 1)
            ].set_index(["标度档", "期限"])["GP_元每吨"]
    q_rows = []
    for struct in (STRUCT_A, STRUCT_B1):
        for n in TERMS:
            x = R[(R["结构"] == struct) & (R["期限"] == n)
                  & (R["免赔档"] == "D75_q75") & (R["费用口径"] == "e12_指数")
                  & (R["n_eff"] == N_EFF_BASE)].set_index("标度档")
            j = str(x["层宽"].idxmin())
            gp_min, gp_max = x["GP_元每吨"].min(), x["GP_元每吨"].max()
            # 现行挂牌保费：A_水平式审慎档（A=q99.5 最小档）的现行 GP
            xa = R[(R["结构"] == STRUCT_A) & (R["期限"] == n)
                   & (R["免赔档"] == "D75_q75")].set_index("标度档")
            ja = str(xa["A_q995"].idxmin())
            q_rows.append({
                "结构": struct, "期限": n, "审慎档": j,
                "D_自留额": float(x.loc[j, "D_元每吨"]),
                "责任上限A": float(x.loc[j, "A_q995"]),
                "层宽": float(x.loc[j, "层宽"]),
                "挂牌保费_元每吨": float(x.loc[j, "GP_元每吨"]),
                "挂牌保费_CoC_元每吨": float(x.loc[j, "GP_CoC_元每吨"]),
                "ROL_CoC%": float(x.loc[j, "GP_CoC_ROL%"]),
                "保费区间": f"{gp_min:,.0f} – {gp_max:,.0f}",
                "保费宽度": round(gp_max / max(gp_min, 1e-9), 3),
                "现行挂牌保费_元每吨": float(cur.loc[(ja, n)]),
                "塌缩倍数": round(float(cur.loc[(ja, n)]) / float(x.loc[j, "GP_元每吨"]), 1),
                "GP比标的价值": float(x.loc[j, "GP比标的价值"]),
            })
    Q = pd.DataFrame(q_rows)

    # ---- 敏感性：论文口径 · 5y · A_水平式 · D75
    x1 = R[(R["结构"] == STRUCT_A) & (R["标度档"] == "论文口径")
           & (R["期限"] == 5) & (R["免赔档"] == "D75_q75")].iloc[0]
    rm1, E = float(x1["RM_独立"]), float(x1["E_layer"])
    s_rows = [{"配置": "E(层期望)", "e": "—", "n_eff": "—", "ρ̄": "—", "φ": "—",
               "GP_元每吨": round(E, 0)}]
    for e in (E_EXP_IDX, E_EXP_OLD):
        for neff in (1, 30, 100, 500):
            for rb in (0.10, 0.30, 0.50):
                s_rows.append({
                    "配置": "GP", "e": e, "n_eff": neff, "ρ̄": rb,
                    "φ": round(phi_div(neff, rb), 4),
                    "GP_元每吨": round(gp_of(E, rm1 * phi_div(neff, rb), e), 0),
                    "GP_CoC_元每吨": round(
                        (E + RHO * rm1 * phi_div(neff, rb))
                        / (1.0 - e - PI_PROFIT), 0),
                })
    S = pd.DataFrame(s_rows)
    return R, B, Q, S, _checks(R, B, Q)


# ---------------------------------------------------------------- 验收断言
def _checks(R: pd.DataFrame, B: pd.DataFrame, Q: pd.DataFrame) -> list[dict]:
    res = []

    def add(tag, desc, ok, got, exp):
        res.append({"验收项": tag, "判据": desc, "结果": "PASS" if ok else "FAIL",
                    "实测": got, "期望": exp})

    # V1 复现 s7：D=0 无截断、e25、φ=1（A_水平式）
    try:
        s7 = pd.read_csv(OUT / "s7_精算口径保费.csv")
        m = R[(R["结构"] == STRUCT_A) & (R["免赔档"] == "D0_无免赔") & (R["无截断"])
              & (R["费用口径"] == "e25_现行") & (R["n_eff"] == 1)
              & (R["标度档"].isin(set(s7["标度档"])))]
        mg = m.merge(s7[["标度档", "期限", "毛费率_新%"]], on=["标度档", "期限"])
        # 毛费率 = GP / (q99.5×n)；s7 的 SI_新 = q99.5×n
        mg["rate_calc"] = mg["GP_元每吨"] / (mg["A_q995"] * mg["期限"]) * 100
        dev = float((mg["rate_calc"] - mg["毛费率_新%"]).abs().max())
        add("V1", "复现 s7（D=0 无截断 · e25 · φ=1）逐档毛费率", dev <= 0.05,
            f"最大偏差 {dev:.4f}pp（{len(mg)} 组）", "≤ 0.05pp")
    except Exception as exc:                                 # noqa: BLE001
        add("V1", "复现 s7", False, f"异常：{exc}", "逐档 ≤ 0.05pp")

    # V2 免赔单调性：GP 对免赔档单调不增（逐结构×档×期限×费用×n_eff）
    bad2 = []
    for k, g in R.groupby(["结构", "标度档", "期限", "费用口径", "n_eff"]):
        v = [float(g[g["免赔档"] == d]["GP_元每吨"].iloc[0]) for d in D_ORDER]
        if any(v[i + 1] > v[i] + 1e-6 for i in range(len(v) - 1)):
            bad2.append(str(k[:3]))
    add("V2", "GP 随免赔档单调不增", not bad2,
        f"反转 {len(bad2)} 组" if bad2 else "全部单调", "0 组")

    # V3 触发概率与免赔分位自洽
    bad3 = []
    for p, d_tag in ((0.50, "D50_q50"), (0.75, "D75_q75"), (0.90, "D90_q90")):
        x = R[R["免赔档"] == d_tag]
        if len(x) and (x["触发概率%"] > (1 - p) * 100 + 2).any():
            bad3.append(d_tag)
    add("V3", "P(L>D) ≤ 名义超越概率+2pp（MC 误差内）", not bad3,
        " / ".join(bad3) if bad3 else "全部自洽", "≤ +2pp")

    # V4 层统计有界：E_layer ≤ E_full（0.1 位舍入容差）；
    #     TVaR99_layer ≤ 层宽仅对**截断层**成立（无截断的现行产品行不适用）
    mb = B.set_index(["结构", "标度档", "期限"])
    bad4 = []
    for _, r in R.iterrows():
        key = (r["结构"], r["标度档"], r["期限"])
        if r["E_layer"] > float(mb.loc[key, "E_full"]) + 0.11:
            bad4.append(f"E:{key}")
        if (not r["无截断"]) and r["TVaR99_layer"] > r["层宽"] + 0.11:
            bad4.append(f"TVaR:{key}")
    add("V4", "E_layer ≤ E_full 且（截断层）TVaR99_layer ≤ 层宽", not bad4,
        f"违例 {len(bad4)}" if bad4 else "全部有界", "0 违例")

    # V5 商业版 < 现行：D75+e12+n100 的 GP 低于现行（D=0 无截断 e25 φ=1）
    com = R[(R["免赔档"] == "D75_q75") & (R["费用口径"] == "e12_指数")
            & (R["n_eff"] == N_EFF_BASE)]
    cmp_df = com[["结构", "标度档", "期限", "GP_元每吨"]].merge(
        R[(R["结构"] == STRUCT_A) & (R["免赔档"] == "D0_无免赔") & (R["无截断"])
          & (R["费用口径"] == "e25_现行") & (R["n_eff"] == 1)]
        [["标度档", "期限", "GP_元每吨"]].rename(columns={"GP_元每吨": "GP_cur"}),
        on=["标度档", "期限"], how="left")
    bad_mask = cmp_df["GP_元每吨"] >= cmp_df["GP_cur"] - 1e-9
    # ⚠ 不可用 .apply(axis=1)：掩码全 False 时 pandas 对空 DataFrame 返回列名列表，
    #   会把 5 个列名误报成 5 组违例（2026-10-03 实测踩坑）。
    bad5 = [f"{r['标度档']}/{int(r['期限'])}y ({r['结构']})"
            for _, r in cmp_df[bad_mask].iterrows()]
    med = float((cmp_df["GP_元每吨"] / cmp_df["GP_cur"]).median())
    add("V5", "商业版（D75·e12·n100）GP < 现行 GP", not bad5,
        f"未下降 {len(bad5)} 组；中位塌缩至现行的 {med:.1%}", "全部下降")

    # M2 登记：资本成本口径 ≤ 双重计提口径（逐行）
    bad_m2 = R[R["GP_CoC_元每吨"] > R["GP_元每吨"] + 1e-6]
    add("M2", "CoC 口径 ≤ 双重计提口径（登记，逐行）", bad_m2.empty,
        f"违例 {len(bad_m2)} 行" if len(bad_m2) else "全部成立", "0 违例")

    # 登记项：原始产品跨档带宽（D0 · e25 · φ=1）—— B1 年度重定基应窄于 A 水平式
    #（与 s4 第四节结论同向；分层后带宽另行披露，不作为判据）
    band = {}
    for struct in (STRUCT_A, STRUCT_B1):
        w = []
        for n in TERMS:
            x = R[(R["结构"] == struct) & (R["期限"] == n)
                  & (R["免赔档"] == "D0_无免赔") & (R["费用口径"] == "e25_现行")
                  & (R["n_eff"] == 1) & (~R["无截断"] if struct == STRUCT_B1 else True)]
            w.append(float(x["GP_元每吨"].max() / x["GP_元每吨"].min()))
        band[struct] = [round(v, 2) for v in w]
    add("M1", "原始产品跨档带宽（登记）：B1 年度重定基应窄于 A 水平式",
        all(b <= a for b, a in zip(band[STRUCT_B1], band[STRUCT_A])),
        f"A {band[STRUCT_A]} vs B1 {band[STRUCT_B1]}（逐期限 B1 ≤ A）",
        "B1 ≤ A")
    return res


# ---------------------------------------------------------------- 主
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    only_check = "--check" in sys.argv
    R, B, Q, S, res = build()

    print("=" * 96)
    print("s22　L2 商业版重构定价（可保性修复：D 免赔 + A=q99.5 截断 + 组合分散 φ + e=12%）")
    print("=" * 96)

    print("\n【表 1】现行 vs 商业版（D75 · e12 · n_eff=100 · ρ̄=0.30）—— 审慎档（层宽下端）")
    print("-" * 96)
    print(Q[["结构", "期限", "审慎档", "D_自留额", "责任上限A", "层宽",
             "现行挂牌保费_元每吨", "挂牌保费_元每吨", "挂牌保费_CoC_元每吨",
             "ROL_CoC%", "塌缩倍数", "GP比标的价值"]].to_string(index=False))

    print("\n【表 2】敏感性（论文口径 · 5y · A_水平式 · D75）：e × n_eff × ρ̄")
    print("-" * 96)
    print(S.to_string(index=False))

    print("\n【表 3】跨档带宽（D75 · e12 · n100）")
    print("-" * 96)
    for struct in (STRUCT_A, STRUCT_B1):
        for n in TERMS:
            x = R[(R["结构"] == struct) & (R["期限"] == n)
                  & (R["免赔档"] == "D75_q75") & (R["费用口径"] == "e12_指数")
                  & (R["n_eff"] == N_EFF_BASE)]
            print(f"  {struct}  {n:2d}y  GP [{x['GP_元每吨'].min():>8,.0f}, "
                  f"{x['GP_元每吨'].max():>8,.0f}]  宽 "
                  f"{x['GP_元每吨'].max() / x['GP_元每吨'].min():.2f}×")

    print("\n【验收】V1–V5 + 登记项")
    print("-" * 96)
    for r in res:
        print(f"  [{r['结果']:4s}] {r['验收项']:4s} {r['判据']}　→ {r['实测']}")
    nfail = sum(1 for r in res if r["结果"] == "FAIL")
    print(f"\n  合计：{len(res) - nfail} PASS / {nfail} FAIL")
    if only_check:
        sys.exit(1 if nfail else 0)

    R.to_csv(OUT / "s22_商业版定价_主表.csv", index=False, encoding="utf-8-sig")
    B.to_csv(OUT / "s22_损失分布基准统计.csv", index=False, encoding="utf-8-sig")
    Q.to_csv(OUT / "s22_商业版报价单.csv", index=False, encoding="utf-8-sig")
    S.to_csv(OUT / "s22_敏感性_论文口径5y.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "s22_商业版配置.json", "w", encoding="utf-8") as f:
        json.dump({
            "版本": "v1（2026-10-03，可保性修复工作包）",
            "结构杠杆": {
                "L1_聚合免赔D": "L 的 q50/q75/q90；客户自留额显式披露",
                "L2_责任上限A": "q99.5(L)（不乘 n，10-02 裁定口径）",
                "L3_组合分散φ": f"sqrt(ρ̄+(1−ρ̄)/n_eff)；基准 n_eff={N_EFF_BASE}, "
                            f"ρ̄={RHO_BAR_BASE}；正态幂近似，待组合层联合模拟复核",
                "L4_费用率": f"指数产品 e={E_EXP_IDX}（无查勘定损；报行合一下界=实际费用率，"
                          f"待核备案）；现行 e={E_EXP_OLD} 并列输出",
            },
            "触发结构": {
                STRUCT_A: "K=1.30×MAC_REF 固定（现行 s7 骨架）",
                STRUCT_B1: "pay_t=min(max(I_t−1.30·I_{t−1},0),CAP)·Q_t，t=1..n（年度重定基）",
            },
            "登记缺陷": "s4.price_rebased 分段循环把重定基边界年计入两段 ⇒ 设计B 费率高估；"
                     "方向性结论（区间收窄）不受影响；本脚本 B1 为无重复计数实现",
            "不变量": f"ρ={RHO}、π={PI_PROFIT}、贴现={DF_R}、κ=1.30、CAP=2×MAC_REF、"
                   "两层结构（模糊性不进 MC）逐条保留",
        }, f, ensure_ascii=False, indent=2)
    print("\n[已写出] s22_商业版定价_主表.csv / s22_损失分布基准统计.csv / "
          "s22_商业版报价单.csv / s22_敏感性_论文口径5y.csv / s22_商业版配置.json")


if __name__ == "__main__":
    main()
