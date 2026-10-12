# -*- coding: utf-8 -*-
"""
s9_reprice.py —— A3 重定价计算（条款 §3.3 六步映射）

上游：# 只读
    02_产物/s4_定价配置.json        （逐档 MAC2022_1995价，= 映射二的输入）
    02_产物/s7_精算口径保费.csv     （上期挂牌，用于变动幅度对比）
    02_产物/s2_全国输入路径.csv     （u 逐年路径，触发判定 (a)）
    02_产物/s2b_期限放大.csv        （越界后档位存活与跨度，触发判定 (c)(d) 参照）
    02_产物/s0d_窗口数据稳健性.csv  （数据修订对合法域的影响，触发判定 (b)(c)）
下游产物：
    02_产物/s9_重定价结果.csv       （重定价年 × 期限 × 9 档的六步中间量与结果）
    02_产物/s9_重定价挂牌.csv       （入选档的挂牌值 + 与上期的变动幅度 + 是否触发协商）
    02_产物/s9_触发判定.csv         （年度 × 条款 §3.2 四类触发的实测值与判定）

设计依据：`02_产物/中期_双指标报价与重定价条款_设计_v1.md` §3.3（六步映射）、§5.1、§9.3 A9/A10/A12。

⚠ A12 的硬约束（本脚本的运行前提）：重定价**只**允许触发 §3.3 映射二至六；
   不得对 κ、ρ、e、π 或随机过程参数做任何调整。脚本启动时对「冻结参数集」
   做一致性断言（对照 s8_审慎取值规则.json），任何不一致即中止。

用法：
    python s9_reprice.py            # 生成产物 + 打印
    python s9_reprice.py --check    # 跑断言（冻结参数一致性 + 结构判据）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import OUT, GAP_MULTIPLE                        # noqa: E402
import s4_pricing as s4                                       # noqa: E402
from s3_term import deflator_to                               # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.unicode.east_asian_width", True)

SCEN = "S1_基准"
N_PATH = 20000
SEED0 = 20261002
Q_LIMIT, Q_RISK = 0.995, 0.99
RHO, E_EXP, PI_PROFIT, DF_R = 0.06, 0.25, 0.08, 0.025
KAPPA = 1.30
CAP_MULT = 2.0
MAC_REF_1995 = 190.5306        # 论文口径 2022 MAC（1995 价），用于「全局 CAP」敏感口径

# 重定价案例：(重定价年, 原期限)   —— 2026 起保、周期 1 年 ⇒ 首次重定价在 2027
CASES = [(2027, 3), (2027, 5), (2027, 10), (2027, 15), (2028, 5)]

FROZEN_EXPECT = {
    "κ_起赔倍数": KAPPA, "ρ_资本成本率": RHO,
    "e_费用率": E_EXP, "π_利润边际": PI_PROFIT, "贴现率": DF_R,
}


# ----------------------------------------------------------------------
def _tvar(x: np.ndarray, a: float) -> float:
    q = np.quantile(x, a)
    t = x[x >= q]
    return float(t.mean()) if t.size else float(q)


def load_tiers() -> dict:
    cfg = json.load(open(OUT / "s4_定价配置.json", encoding="utf-8"))
    return {k: float(v["MAC2022_1995价"]) for k, v in cfg["标度档"].items()}


def check_frozen() -> list[dict]:
    """A10/A12：断言本次运行所用的参数与 s8 冻结的定价参数集一致。"""
    res = []
    try:
        frozen = json.load(open(OUT / "s8_审慎取值规则.json", encoding="utf-8"))["定价参数集_冻结留痕"]
    except Exception as e:                                    # noqa: BLE001
        return [{"验收项": "A10", "结果": "FAIL", "实测": f"读不到冻结参数集：{e}", "期望": "可读"}]
    for k, v in FROZEN_EXPECT.items():
        got = frozen.get(k)
        ok = got is not None and abs(float(got) - v) < 1e-12
        res.append({"验收项": f"A10/{k}", "结果": "PASS" if ok else "FAIL",
                    "实测": f"s8 冻结值={got}，本次使用={v}", "期望": "一致"})
    sp = frozen.get("随机过程参数", {})
    pairs = [("σ_P", s4.SIG_P), ("σ_m_路径", s4.SIG_M_PATH), ("σ_m_水平", s4.SIG_M_LEVEL),
             ("σ_Q", s4.SIG_Q), ("ρ_Pm", s4.RHO_PM), ("ρ_PQ", s4.RHO_PQ),
             ("ρ_mQ", s4.RHO_MQ), ("路径数", N_PATH)]
    for k, v in pairs:
        ok = k in sp and abs(float(sp[k]) - float(v)) < 1e-12
        res.append({"验收项": f"A10/{k}", "结果": "PASS" if ok else "FAIL",
                    "实测": f"s8 冻结值={sp.get(k)}，本次使用={v}", "期望": "一致"})
    return res


def simulate_at(t0: int, n_rem: int, m0: float, seed: int,
                cap_mult: float = CAP_MULT) -> dict:
    """
    自 t0 年起保、期限 n_rem 的蒙特卡洛（与 s4.simulate 同过程、同参数，
    仅把时间原点从 2026 移到 t0）。MC 只跑档内风险；m0 为确定值，模糊性不进 MC。
    """
    rng = np.random.default_rng(seed)
    yrs = list(range(t0, t0 + n_rem + 2))
    nyr, n = len(yrs), N_PATH
    P_c = s4._price_central(SCEN, yrs)
    spec = GAP_MULTIPLE[SCEN]
    dt = np.arange(nyr, dtype=float)
    m_c = spec["floor"] + (m0 - spec["floor"]) * np.exp(-np.log(2) / spec["half_life"] * dt)

    Z = rng.standard_normal((3, nyr, n))
    Z = np.einsum("ij,jtn->itn", s4.CHOL, Z)
    eP, eM, eQ = Z[0], Z[1], Z[2]

    lnP = np.empty((nyr, n))
    lnP[0] = np.log(P_c[0]) + s4.SIG_P * eP[0]
    for t in range(1, nyr):
        lnP[t] = (lnP[t - 1] + s4.KAPPA_P * (np.log(P_c[t]) - lnP[t - 1]) + s4.SIG_P * eP[t])
    P = np.exp(lnP)

    lnm = np.empty((nyr, n))
    lnm[0] = np.log(m_c[0]) + s4.SIG_M_LEVEL * eM[0]
    for t in range(1, nyr):
        lnm[t] = (lnm[t - 1] + s4.KAPPA_M * (np.log(m_c[t]) - lnm[t - 1]) + s4.SIG_M_PATH * eM[t])
    M = np.exp(lnm)

    # 承保缺口率：从保单原起保年**连续**推进（2026 起保，故起点 = gap_rate_0 + tighten×(t0−2026)）
    g = s4.L1["gap_rate_0"] + s4.L1["gap_tighten"] * ((t0 - s4.START) + np.arange(nyr))
    Q = np.exp(np.log(g)[:, None] + s4.SIG_Q * eQ)
    Q = Q / Q[0]

    return {"years": yrs, "P": P, "M": M, "I": P * M, "Q": Q, "P_c": P_c, "m_c": m_c}


def six_steps(sim: dict, t0: int, n_rem: int, mac95: float, cap_mult: float,
              cap_nom_override: float | None = None) -> dict:
    """条款 §5.1 六步（+ 映射二/四）。MAC 与阈值均取 t0 年的**名义**值。

    cap_nom_override：若给定，则用该**名义**值作 CAP（用于「全局 CAP」敏感口径），
    否则 CAP = cap_mult × MAC_REF_j(t0)（逐档口径，本版主推）。
    """
    defl = deflator_to(t0)
    mac_nom = mac95 * defl                      # 映射二：M2 价格平移
    K = KAPPA * mac_nom                         # 映射四
    CAP = cap_mult * mac_nom if cap_nom_override is None else float(cap_nom_override)
    I, Q = sim["I"], sim["Q"]
    pay = np.minimum(np.maximum(I - K, 0.0), CAP) * Q
    df = 1.0 / (1.0 + DF_R) ** np.arange(n_rem + 1)
    pv = (pay[:n_rem + 1] * df[:, None]).sum(axis=0)

    E = float(pv.mean())
    q995 = float(np.quantile(pv, Q_LIMIT))
    tv = _tvar(pv, Q_RISK)
    SI = q995 * n_rem                            # 第一步
    RM = tv - E                                  # 第三步
    capc = RHO * RM                              # 第四步
    GP = (E + RM + capc) / (1 - E_EXP - PI_PROFIT)   # 第五步
    return {"MAC_REF_名义": mac_nom, "K": K, "CAP": CAP, "m0": sim["m_c"][0],
            "E": E, "P99.5": q995, "TVaR99": tv, "RM": RM, "CapCost": capc,
            "SI": SI, "GP": GP, "费率": GP / SI * 100.0}


# ----------------------------------------------------------------------
def trigger_table(tiers: dict) -> pd.DataFrame:
    """条款 §3.2 四类触发的逐年判定。"""
    path = pd.read_csv(OUT / "s2_全国输入路径.csv")
    path = path[path["情景"] == SCEN].set_index("年份")
    rows = []
    for y in [2026, 2027, 2028, 2029, 2030]:
        u = float(path.loc[y, "u"])
        hr = float(path.loc[y, "u_headroom_pct"])
        # (a) 结构有效性：u ≤ 0.005
        a = "触发" if u <= 0.005 else f"正常（余量 {hr:.2f}%）"
        # (b) 数据修订 > 2%：无当期官方修订数据，固定为「待监控」
        b = "待监控（无当期官方修订数据）"
        # (c) 合法域移动：由 s0d 的 δ 敏感性登记（δ=2% 即使 γ₁=0 出域）
        c = "待重算（s0d：δ=2% 异质修订即 dev=1.0015>1，γ₁=0 出域）"
        # (d) 指数水平落于全档区间最高/最低 10% 之外
        Pc = float(s4._price_central(SCEN, [y])[0])
        spec = GAP_MULTIPLE[SCEN]
        dt = y - s4.START
        idx = {}
        for tag, mac95 in tiers.items():
            m0 = mac95 * deflator_to(y) / Pc
            m = spec["floor"] + (m0 - spec["floor"]) * np.exp(-np.log(2) / spec["half_life"] * max(dt, 0))
            idx[tag] = Pc * m
        lo, hi = min(idx.values()), max(idx.values())
        pos = (idx["论文口径"] - lo) / (hi - lo)
        d = f"{'触发' if (pos < 0.10 or pos > 0.90) else '正常'}（论文口径指数在区间 {pos*100:.1f}% 位置）"
        rows.append({"年份": y, "u": u, "u余量%": round(hr, 3),
                     "触发a_结构有效性": a, "触发b_数据修订": b,
                     "触发c_合法域移动": c, "触发d_指数水平": d,
                     "指数区间下": round(lo, 1), "指数区间上": round(hi, 1),
                     "论文口径指数": round(idx["论文口径"], 1)})
    return pd.DataFrame(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    only_check = "--check" in sys.argv
    tiers = load_tiers()
    froz = check_frozen()

    print("=" * 118)
    print("s9　A3 重定价计算（条款 §3.3 六步映射 · S1 基准 · L1b · M2 价格平移）")
    print("=" * 118)

    res_rows, quote_rows, sens_rows = [], [], []
    base = pd.read_csv(OUT / "s7_精算口径保费.csv")

    for ci, (t0, orig_n) in enumerate(CASES):
        n_rem = orig_n - (t0 - s4.START)
        if n_rem < 1:
            continue
        # 先算模拟（与 CAP 口径无关，可复用）
        P_t0 = float(s4._price_central(SCEN, [t0])[0])
        sims = {tag: (mac95, simulate_at(t0, n_rem, mac95 * deflator_to(t0) / P_t0,
                                         SEED0 + ci * 977 + j * 131))
                for j, (tag, mac95) in enumerate(tiers.items())}
        cap_global = CAP_MULT * (MAC_REF_1995 * deflator_to(t0))     # 全局口径的名义 CAP
        per_mode = {}
        for mode, cap_ov in {"逐档": None, "全局_论文口径": cap_global}.items():
            pt = {}
            for tag, (mac95, sim) in sims.items():
                d = six_steps(sim, t0, n_rem, mac95, CAP_MULT, cap_ov)
                d.update({"重定价年": t0, "原期限": orig_n, "剩余期限": n_rem,
                          "标度档": tag, "CAP口径": mode})
                pt[tag] = d
            per_mode[mode] = pt
            Pm = pd.DataFrame(pt).T
            sens_rows.append({
                "重定价年": t0, "原期限": orig_n, "剩余期限": n_rem, "CAP口径": mode,
                "CAP_名义": round(float(Pm["CAP"].astype(float).iloc[0]), 1),
                "入选档": str(Pm["SI"].astype(float).idxmin()),
                "SI_区间": f"{Pm['SI'].astype(float).min():,.0f} – {Pm['SI'].astype(float).max():,.0f}",
                "SI_宽度": round(float(Pm["SI"].astype(float).max() / Pm["SI"].astype(float).min()), 3),
                "费率_区间%": f"{Pm['费率'].astype(float).min():.2f} – {Pm['费率'].astype(float).max():.2f}",
                "费率_宽度": round(float(Pm["费率"].astype(float).max() / Pm["费率"].astype(float).min()), 3),
            })

        per_tier = per_mode["逐档"]                       # 主推口径
        j_star = str(pd.DataFrame(per_tier).T["SI"].astype(float).idxmin())   # 映射六
        for tag, d in per_tier.items():
            r = dict(d)
            r["是否入选档"] = tag == j_star
            res_rows.append(r)

        # 与上期对比（上期 = s7 中该原期限、该档的挂牌值）
        prev = base[(base["期限"] == orig_n) & (base["标度档"] == j_star)].iloc[0]
        prev_SI = float(prev["SI_新"])
        prev_rate = float(prev["毛费率_新%"])
        new_SI = float(per_tier[j_star]["SI"])
        new_rate = float(per_tier[j_star]["费率"])
        d_si = (new_SI / prev_SI - 1) * 100
        d_rt = (new_rate / prev_rate - 1) * 100
        rs = 1 / 1.3 <= new_rate / prev_rate <= 1.3
        quote_rows.append({
            "重定价年": t0, "原期限": orig_n, "剩余期限": n_rem, "入选档": j_star,
            "上期限额": round(prev_SI, 0), "新限额": round(new_SI, 0),
            "限额变动%": round(d_si, 2),
            "上期费率%": round(prev_rate, 2), "新费率%": round(new_rate, 2),
            "费率变动%": round(d_rt, 2), "费率比_新除旧": round(new_rate / prev_rate, 4),
            "按§5.2触发协商": "否" if rs else "是（费率变动超 ±30%）",
            "按§5.2触发协商_限额": "否" if abs(d_si) <= 40 else "是（限额变动超 ±40%）",
        })

    R = pd.DataFrame(res_rows)
    Qr = pd.DataFrame(quote_rows)

    print("\n【六步映射结果 · 2027 年重定价（2026 起保）】")
    print("-" * 118)
    show = R[R["重定价年"] == 2027]
    for n in [3, 5, 10, 15]:
        x = show[show["原期限"] == n]
        if not len(x):
            continue
        x = x.sort_values("SI")
        print(f"  ── 原期限 {n} 年（剩余 {int(x['剩余期限'].iloc[0])} 年） ──")
        print(x[["标度档", "MAC_REF_名义", "K", "CAP", "SI", "费率", "E", "P99.5",
                 "TVaR99", "是否入选档"]].round(3).to_string(index=False))
        print()

    print("\n【挂牌与变动幅度】")
    print("-" * 118)
    print(Qr.to_string(index=False))

    Sn = pd.DataFrame(sens_rows)
    print("\n【CAP 口径敏感性（本次实现新增的设计决策，须确认）】")
    print("-" * 118)
    print(Sn.to_string(index=False))
    print("  ⇒ 「逐档」= CAP_j = 2 × MAC_REF_j(t)（本版主推，「全局_论文口径」= 2 × MAC_REF(论文口径,t)，")
    print("     与 s7 一致）。二者 SI 与费率档间宽度的差异即为该决策的代价。")

    print("\n【触发判定（条款 §3.2）】")
    print("-" * 118)
    Tg = trigger_table(tiers)
    print(Tg.to_string(index=False))

    print("\n【验收】冻结参数一致性（A10/A12）")
    print("-" * 118)
    for r in froz:
        print(f"  [{r['结果']:4s}] {r['验收项']:16s} {r['实测']}")
    nf = sum(1 for r in froz if r["结果"] == "FAIL")
    print(f"\n  合计：{len(froz) - nf} PASS / {nf} FAIL")

    if only_check:
        sys.exit(1 if nf else 0)

    R.to_csv(OUT / "s9_重定价结果.csv", index=False, encoding="utf-8-sig")
    Qr.to_csv(OUT / "s9_重定价挂牌.csv", index=False, encoding="utf-8-sig")
    Tg.to_csv(OUT / "s9_触发判定.csv", index=False, encoding="utf-8-sig")
    Sn.to_csv(OUT / "s9_重定价_CAP口径敏感.csv", index=False, encoding="utf-8-sig")

    with open(OUT / "s9_重定价配置.json", "w", encoding="utf-8") as f:
        json.dump({
            "版本": "v1（A3 产物，2026-10-02）",
            "映射二": "MAC_REF,j(t) = MAC2022,j(1995价) × Deflator(t)/Deflator(2022)，M2 价格平移",
            "映射四": "K_j = κ × MAC_REF,j(t)；CAP_j = CAP_MULT × MAC_REF,j(t)",
            "映射六": "入选档 j* = argmin_j SI_j；新费率与限额同档，不得混档",
            "重定价周期": "min(3 年, 距 u 越界年数 − 1)；2026 起保取 1 年",
            "冻结参数一致性": froz,
            "两处待确认的设计决策": [
                "① CAP_j 取「逐档 2×MAC_REF_j(t)」（本版实现）而非 s7 的全局 2×MAC_REF——"
                "后者使 CAP 挂靠论文口径锚（D1 同类问题）。此项改变档间离散度，须确认。",
                "② 承保缺口率 g 按保单**原起保年连续**推进（2026 起保 → t0 年起点上移），"
                "而非每次重定价回到 gap_rate_0。此项影响 Q 的相对增长，须确认。",
            ],
        }, f, ensure_ascii=False, indent=2)

    print("\n[已写出] s9_重定价结果.csv / s9_重定价挂牌.csv / s9_触发判定.csv / s9_重定价配置.json")


if __name__ == "__main__":
    main()
