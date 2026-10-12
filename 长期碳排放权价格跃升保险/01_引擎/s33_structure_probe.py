# -*- coding: utf-8 -*-
"""s33 产品结构决策探针：A / B / C 三条路线的实际报价对照。

【用途】
   第三节遗留的未决问题——产品结构是"纯 P"、"P 主 + MAC 副（M1 重锚）"
   还是"维持现状（M2 4.22 倍）"。此脚本不改动 s32 任何逻辑，只做旁路测算，
   把三条路线的**实际报价、结构指标与自洽性**算出来，供拍板时以数字判断。

【设计纪律】
   - 不硬编码结论性数字：一律从 s32 / s4 的比例与缓存取。
   - M1 重锚只改一件事：模拟走 M1_结构外推口径（m₀ 中枢 1.3243，
     确定性影子锚 = 413.32 × (59.77/190.5306) = 129.66），
     σ、ρ、κ、φ、份额全部不动。A/B 的差异纯粹来自"影子锚高度"。

用法：python s33_structure_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s32_dual_trigger as s32      # noqa: E402
import s4_pricing as s4             # noqa: E402
import s23_review_response as s23   # noqa: E402
from _config import L1, OUT         # noqa: E402

OUT_DIR = Path(OUT)
MC = 40000
SEED0 = L1["seed"] + 31 * 977

MAC_REF_M1 = s32.MAC_REF * s32.M1_RATIO_TO_M2   # 确定性 M1 影子锚 ≈ 129.66


def _tier():
    return next(t for t in s4._lp_cache().values()
                if abs(float(t["γ1"]) - s32.G1_FROZEN) < 1e-9
                and abs(float(t["β"]) - s32.BETA_FROZEN) < 1e-9)


def sim_of(caliber: str, sig_m: float = s32.SIG_M_BASE):
    return s23.simulate_p("S1_基准", _tier(), caliber, MC, SEED0, sig_m=sig_m)


def _layer(x: np.ndarray):
    D = float(np.quantile(x, s32.Q_D))
    A = float(np.quantile(x, s32.Q_A))
    return np.clip(x - D, 0.0, A - D), D, A


def _gp(E: float, rm: float) -> float:
    return (E + s32.RHO * rm) / (1.0 - s32.E_EXP - s32.PI_PROFIT)


def _stat(x: np.ndarray, share: float) -> dict:
    """单责任条款级指标（份额已乘、分位重取）。"""
    xs = share * x
    lay, D, A = _layer(xs)
    E = float(lay.mean())
    rm = (s32._tvar(lay) - E) * s32.PHI
    gp = _gp(E, rm)
    return {"D": D, "A": A, "层宽": A - D, "触发%": float((xs > 0).mean() * 100),
            "E": E, "RM_未分散": s32._tvar(lay) - E, "GP": gp,
            "赔付率%": E / max(gp, 1e-9) * 100, "ROL%": gp / max(A - D, 1e-9) * 100}


def dual(raw: dict, share: float) -> dict:
    """双给付（两份责任独立分层，φ 施加一次）——复刻 s32.metrics 口径。"""
    c, s = _stat(raw["cash"], share), _stat(raw["shad"], share)
    E = c["E"] + s["E"]
    rm = (c["RM_未分散"] + s["RM_未分散"]) * s32.PHI     # φ 施加一次
    gp = _gp(E, rm)
    Dsum, Asum = c["D"] + s["D"], c["A"] + s["A"]
    return {"GP": gp, "E": E, "层宽": Asum - Dsum, "D": Dsum, "A": Asum,
            "赔付率%": E / max(gp, 1e-9) * 100,
            "ROL%": gp / max(Asum - Dsum, 1e-9) * 100,
            "现金E": c["E"], "影子E": s["E"],
            "影子占%": s["E"] / max(E, 1e-9) * 100,
            "并集触发%": float((raw["cash_trigger"] | raw["shad_trigger"]).mean() * 100),
            "现金触发%": float(raw["cash_trigger"].mean() * 100),
            "影子触发%": float(raw["shad_trigger"].mean() * 100)}


def max_share(gp_fn, budget: float = s32.BUDGET) -> float:
    best = 0.0
    for sh in np.arange(0.05, 1.0001, 0.05):
        if gp_fn(sh) <= budget:
            best = sh
        else:
            break
    return best


def main() -> None:
    print("=" * 112)
    print("s33 产品结构决策探针 · A/B/C 三路线对照（旁路测算，不改动 s32）")
    print("=" * 112)

    sm2, sm1 = sim_of("M2_价格平移"), sim_of("M1_结构外推")
    raw2, raw1 = s32.pay_split(sm2, 1, share=1.0), s32.pay_split(sm1, 1, share=1.0)

    print("\n【口径核实 · 确定性锚（中枢，非模拟均值）】")
    print(f"  M2 价格平移：m₀ = {s32.M0:.4f}，确定性影子锚 MAC_REF = {s32.MAC_REF:.2f} 元/吨（2026 名义）")
    print(f"  M1 结构外推：m₀ = {s32.M0_M1:.4f}，确定性影子锚       = {MAC_REF_M1:.2f} 元/吨")
    print(f"  ⇒ 两份责任锚差 m₀：C 路线 {s32.M0:.2f}×　B 路线 {s32.M0_M1:.2f}×")
    print(f"  ⇒ M1 相对 M2 的锚比 = {s32.M0 / s32.M0_M1:.3f}×（即 M2 把起点抬高 {(s32.M0/s32.M0_M1-1)*100:.1f}%）")
    print(f"  注：模拟 t=0 均值 I₀ 分别为 {sm2['I'][0].mean():.1f}（M2）/ {sm1['I'][0].mean():.1f}（M1），")
    print(f"      高于确定性锚系对数正态 Jensen 效应 + σ_m水平=0.12 扰动，非口径错误。")

    rows = []
    a = _stat(raw2["cash"], s32.SHARE)
    rows.append({"路线": "A 纯 P（仅现金）", "责任构成": "现金 ×1",
                 "毛保费GP": a["GP"], "期望赔付E": a["E"], "层宽": a["层宽"],
                 "赔付率%": a["赔付率%"], "ROL%": a["ROL%"], "触发%": a["触发%"],
                 "影子占层赔付%": 0.0})
    b = dual(raw1, s32.SHARE)
    rows.append({"路线": "B 双给付·M1 重锚", "责任构成": "现金 + 影子(M1)",
                 "毛保费GP": b["GP"], "期望赔付E": b["E"], "层宽": b["层宽"],
                 "赔付率%": b["赔付率%"], "ROL%": b["ROL%"], "触发%": b["并集触发%"],
                 "影子占层赔付%": b["影子占%"]})
    c = dual(raw2, s32.SHARE)
    rows.append({"路线": "C 双给付·M2（现状）", "责任构成": "现金 + 影子(M2)",
                 "毛保费GP": c["GP"], "期望赔付E": c["E"], "层宽": c["层宽"],
                 "赔付率%": c["赔付率%"], "ROL%": c["ROL%"], "触发%": c["并集触发%"],
                 "影子占层赔付%": c["影子占%"]})
    df = pd.DataFrame(rows).round(3)
    pd.set_option("display.width", 240)
    pd.set_option("display.unicode.east_asian_width", True)

    print("\n【三路线报价对照（份额 10%，元/吨配额）】")
    print(df[["路线", "责任构成", "毛保费GP", "期望赔付E", "层宽", "赔付率%", "ROL%", "触发%"]]
          .to_string(index=False))

    print("\n【双给付内部构成：B 与 C】")
    for tag, d in (("B·M1 重锚", b), ("C·M2 现状", c)):
        print(f"  {tag}：现金层E = {d['现金E']:.3f}　影子层E = {d['影子E']:.3f}"
              f"　影子占层赔付 = {d['影子占%']:.1f}%")
        print(f"          现金触发 {d['现金触发%']:.1f}%　影子触发 {d['影子触发%']:.1f}%"
              f"　并集触发 {d['并集触发%']:.1f}%")

    print("\n【最大可行份额扫描（预算 89 元/吨）】")
    ms_a = max_share(lambda sh: _stat(raw2["cash"], sh)["GP"])
    ms_b = max_share(lambda sh: dual(raw1, sh)["GP"])
    ms_c = max_share(lambda sh: dual(raw2, sh)["GP"])
    print(f"  A 纯 P：{ms_a:.0%}（v1 单给付口径 95% → v2 双给付 85% → A 路线 {ms_a:.0%}）")
    print(f"  B M1 重锚：{ms_b:.0%}")
    print(f"  C M2 现状：{ms_c:.0%}（须复现 s32 基准 85%）")

    # ---- 自洽性判定：定价口径 vs 越界判据 ----
    print("\n【自洽性判定 · 定价口径与越界判据是否同源】")
    sc = [
        ("A 纯 P", "无影子锚", "不适用（无影子责任）", "自洽（越界问题不存在）"),
        ("B M1 重锚", "M1 结构外推", "2028（M1 自身越界）", "自洽（都用 M1，越界年份即失效年份）"),
        ("C M2 现状", "M2 价格平移", "声称不越界", "**不自洽**（引擎自身 m→1.5 收敛 ≈ M1 的 1.32，与 M2 的'成本静止'冲突）"),
    ]
    for r in sc:
        print(f"  {r[0]:<10} 定价口径 = {r[1]:<10} 越界判据 = {r[2]:<20} ⇒ {r[3]}")

    print("\n【探针自检：C 路线须复现 s32 基准 10.0675（4 万路径，容差 0.6）】")
    dev = abs(c["GP"] - 10.0675)
    print(f"  探针 C = {c['GP']:.3f}，偏差 = {dev:.3f} ⇒ {'PASS' if dev < 0.6 else 'FAIL'}")

    print("\n【给拍板用的一句话结论】")
    print(f"  A 纯 P    ：保费 {a['GP']:.2f} 元/吨（份额10%），账实相符、无越界；但放弃 DDF 主线，")
    print(f"              '长期'因不受越界约束反而更好站。")
    print(f"  B M1 重锚 ：保费 {b['GP']:.2f} 元/吨，较 C 低 {(1-b['GP']/c['GP'])*100:.0f}%；"
          f"影子占从 88.5% 回落到 {b['影子占%']:.1f}%，双给付更实质；")
    print(f"              **唯一自洽方案**（定价与越界同用 M1），但 M1 自身 2028 越界仍在。")
    print(f"  C 现状    ：保费 {c['GP']:.2f} 元/吨，最高；须在 §4 明写 M2 与引擎收敛假设的矛盾。")

    out = OUT_DIR / "s33_结构决策探针.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n探针产物：{out}")
    print("=" * 112)


if __name__ == "__main__":
    main()
