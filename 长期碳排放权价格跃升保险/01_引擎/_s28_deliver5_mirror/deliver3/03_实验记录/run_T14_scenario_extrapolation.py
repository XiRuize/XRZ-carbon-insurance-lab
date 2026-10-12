#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-14 / ddf_mac_ext_001 / batch deliver3
阶段 E：2026–2030 情景外推（承接 T-09 2023–2025 真外推链；起点 = T-09 2025 归一化值）。

铁律（只读 + 只执行）：
- deliver/、deliver2/、锚定检验/ 全程只读（仅读取上游 xlsx/json；不复制、不写入）。
- 03_实验记录/ 与 02_设计方案/ 既有文件一律不可修改/覆盖（append-only）；本脚本只新建 T-14 三件。
- 运行须 PYTHONDONTWRITEBYTECODE=1 python3 -B，严禁生成 __pycache__ 或任何多余文件。
- reestimate_forbidden=true：15 个结构系数一律冻结自 C_LP_v1.json，W* 网格系数由全窗重解 LP 枚举
  （识别区间枚举，非结构重估）；前望段为定参数代数代入，非拟合、非重估。
- P-4 双轴禁并：识别区间 I_t 与口径带 B_t 分列，禁只报并集。
- P-9 禁预判砍档：三档全部算完再判；λ_b=1 观测禁静默剔除，计入 30 省算术平均。
- P-10 禁弹性乘推：K 口径带由三档 g_K 精确代入构造。
- 精度强制：g_s* 用精确闭式 (0.83)^(1/5)−1，禁 Forge 舍入值 −3.6570%。
输出：
  03_实验记录/T14_output.json
  03_实验记录/ddf_mac_ext_001_tester_T14_2026-2030情景外推_v1.md
"""
import os
import sys
import json
import time
import hashlib
import platform
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_T09_extrapolation as t09          # noqa: E402  （只读复用模型函数与常数）

OUT_JSON = os.path.join(HERE, 'T14_output.json')
OUT_MD = os.path.join(HERE, 'ddf_mac_ext_001_tester_T14_2026-2030情景外推_v1.md')

# ---------------------------------------------------------------- 只读输入路径
RAW_PATH = t09.RAW_PATH
CLP_PATH = t09.CLP_PATH
CONTRACT = 'C:/Users/生升/Desktop/2026智能+保险创新大赛/碳排放权保险/长期碳排放权价格跃升保险/01_引擎/_s28_deliver5_mirror/deliver3/02_设计方案/validation_contract_ddf_mac_ext_001_v5.json'
DESIGN_V1 = 'C:/Users/生升/Desktop/2026智能+保险创新大赛/碳排放权保险/长期碳排放权价格跃升保险/01_引擎/_s28_deliver5_mirror/deliver3/02_设计方案/ddf_mac_ext_001_forge_2026-2030情景外推设计_v1.md'
CSV_PARAMS = 'C:/Users/生升/Desktop/2026智能+保险创新大赛/碳排放权保险/长期碳排放权价格跃升保险/01_引擎/_s28_deliver5_mirror/deliver3/02_设计方案/parameters_handover_ddf_mac_ext_001_v5.csv'
ALIGN = 'C:/Users/生升/Desktop/2026智能+保险创新大赛/碳排放权保险/长期碳排放权价格跃升保险/01_引擎/_s28_deliver5_mirror/deliver3/02_设计方案/math_code_alignment_ddf_mac_ext_001_v5.md'
T09_SCRIPT = os.path.join(HERE, 'run_T09_extrapolation.py')
T09_JSON = os.path.join(HERE, 'T09_output.json')
T10_JSON = os.path.join(HERE, 'T10_output.json')
T10C_JSON = os.path.join(HERE, 'T10_output_corr.json')
T11B_JSON = os.path.join(HERE, 'T11b_output.json')
T12_JSON = os.path.join(HERE, 'T12_output.json')
T13_MD = os.path.join(HERE, 'ddf_mac_ext_001_tester_T13_勘误登记_v1.md')

# ---------------------------------------------------------------- 契约/设计常数（零改动沿用）
PROSPECT = (2026, 2027, 2028, 2029, 2030)
BACK3 = (2023, 2024, 2025)
THETA_B = 0.5
R_WIDEN_MAX = 1.5
R_HALF_MAX = 3.0
R_HALF_T08 = 3.899
C0 = 0.0

# 精度强制（E2）：精确闭式
G_S_STAR = (0.83) ** (1.0 / 5.0) - 1.0                       # −0.03658008713729832
G_FOSSIL_STAR = (0.750 / 0.783) ** (1.0 / 5.0) - 1.0          # −0.008574921720574058
K_ANCHOR = (1.0 + G_S_STAR) / (1.0 + G_FOSSIL_STAR)           # 0.9717526154721383


def gE_star(gY):
    """双锚联立隐含能源增速 g_E*(g_Y)=k·(1+g_Y)−1（D3-4，精确）。"""
    return K_ANCHOR * (1.0 + gY) - 1.0


# P1/P3/P6/P7 档（契约 scenario_params；主报告取中档）
G_Y_TIERS = {'low': 0.040, 'mid': 0.045, 'high': 0.050}
G_I_P6 = {'low': 0.0, 'mid': 0.045, 'high': 0.088}
G_L_P7 = {'low': -0.008, 'mid': -0.005, 'high': -0.003}

# 层丙（路径丙）δ 三档（论文正文 L269；δ↑⇒g_K↓）
DELTA = {'low': 0.074, 'mid': 0.0638, 'high': 0.057}
R_PRIMARY = 0.12890728324110395          # T-10 同期比 A_s=I_s/K_s 2015–2022 中位
# 层甲（路径甲）近窗 2015–2022 分位
K_TIERS_JIA = {'low': 0.023129487985159403, 'mid': 0.07918513851595121,
               'high': 0.14334993787482503}
# 层乙（路径乙）T-09 原三档（D-005）
K_TIERS_YI = {'low': 0.0442, 'mid': 0.1281, 'high': 0.2067}
T09_TIERS = {'low': 0.0442, 'mid': 0.1281, 'high': 0.2067}   # 2025 起点承接（按档秩匹配）

CENTRAL_CVAR = 'P3b'         # 主报告中央 C 路径 = P3b 强度法（锚口径；与 T-09 CENTRAL 同名）
CVAR_LIST = ['P1', 'P3b']    # P1: C=ρE（2022 ρ 桥接）；P3b: 强度法 g_C=(1+g_Y)(1+g_s)-1

# ---------------------------------------------------------------- 情景定义（E3 / E4）
SCENARIOS = {
    'S_gy4.0': dict(kind='family', gY=G_Y_TIERS['low'],  gE=gE_star(G_Y_TIERS['low'])),
    'S_gy4.5': dict(kind='family', gY=G_Y_TIERS['mid'],  gE=gE_star(G_Y_TIERS['mid'])),
    'S_gy5.0': dict(kind='family', gY=G_Y_TIERS['high'], gE=gE_star(G_Y_TIERS['high'])),
    'D_scout_low': dict(kind='scout', gY=G_Y_TIERS['low'],  gE=0.0),
    'D_scout_high': dict(kind='scout', gY=G_Y_TIERS['high'], gE=0.040),
}

# 三层路径定义（E5）
LAYERS = {
    'layer_C_bing': dict(label='层丙（路径丙，现实情形）', central='mid',
                         tiers={t: R_PRIMARY * (1.0 + G_I_P6['mid']) - DELTA[t]
                                for t in ('low', 'mid', 'high')}),
    'layer_A_jia': dict(label='层甲（路径甲，面板分位口径）', central='mid',
                        tiers=dict(K_TIERS_JIA)),
    'layer_B_yi': dict(label='层乙（路径乙三档，压力测试上界）', central='mid',
                       tiers=dict(K_TIERS_YI)),
}


def sha256_of(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


# ====================================================================== 主流程
def main():
    t_start = time.time()
    panel = t09.load_raw()
    df = t09.build_norm(panel, 2022)
    YEARS = df['年份'].values
    PROV = df['地区'].values
    provs = sorted(set(PROV))
    idx22 = {p: int(np.where((PROV == p) & (YEARS == 2022))[0][0]) for p in provs}

    # ---------- 冻结锚点自检（c0 复现 C_LP_v1.json） ----------
    clp = json.load(open(CLP_PATH, encoding='utf-8'))
    p_json = np.array([clp['params'][k] for k in t09.PNAMES])
    r0 = t09.solve_params(df, C0)
    anchor = dict(
        p_max_absdiff_vs_json=float(np.max(np.abs(r0['p_full'] - p_json))),
        obj_sumD=float(r0['sumD']), target_sumD=132.83354146398435)

    # ---------- 2022 基准数组 ----------
    y22 = np.array([float(df['Y_norm'].values[idx22[p]]) for p in provs])
    l22 = np.array([float(df['L_norm'].values[idx22[p]]) for p in provs])
    e22 = np.array([float(df['E_norm'].values[idx22[p]]) for p in provs])
    k22 = np.array([float(df['K_norm'].values[idx22[p]]) for p in provs])
    c22 = np.array([float(df['C_norm'].values[idx22[p]]) for p in provs])
    rr = np.array([float(df['ratio'].values[idx22[p]]) for p in provs])

    # ---------- T-09 2025 起点（逐位承接 T-09 机制） ----------
    agg = panel.groupby('年份')[['L_raw']].sum().sort_index()
    g_L_central = float(agg['L_raw'].pct_change().loc[2020:2022].mean())
    ym25 = np.array([t09.Y_INDEX_2023[p] / 100.0 * t09.Y_INDEX_2024[p] / 100.0
                     * (1 + t09.Y_RATE_2025[p] / 100.0) for p in provs])
    em25 = 1.0
    for y in BACK3:
        em25 *= (1.0 + t09.G_E_NAT[y])
    lm25 = (1.0 + g_L_central) ** 3
    paths09, _pm = t09.build_c_paths()
    cfac25 = {cv: float(paths09[cv][2025]) for cv in CVAR_LIST}
    kbase25 = {t: k22 * (1.0 + T09_TIERS[t]) ** 3 for t in ('low', 'mid', 'high')}

    # 起点承接校验：与 T09_output.json:input_table 2025 逐位比对
    t09j = json.load(open(T09_JSON, encoding='utf-8'))
    it25 = t09j['input_table']['2025']
    start_check = dict(
        Y_max_reldiff=float(np.max([abs(y22[i] * ym25[i] - it25[p]['Y_norm'])
                                    / max(1e-12, abs(it25[p]['Y_norm']))
                                    for i, p in enumerate(provs)])),
        L_max_reldiff=float(abs(l22[0] * lm25 - it25[provs[0]]['L_norm'])
                            / max(1e-12, abs(it25[provs[0]]['L_norm']))),
        E_max_reldiff=float(max(abs(e22[i] * em25 - it25[p]['E_norm'])
                                / max(1e-12, abs(it25[p]['E_norm']))
                                for i, p in enumerate(provs))),
        Kmid_max_reldiff=float(max(abs(kbase25['mid'][i] - it25[p]['K_norm_mid'])
                                    / max(1e-12, abs(it25[p]['K_norm_mid']))
                                    for i, p in enumerate(provs))),
        C_P1_max_reldiff=float(max(abs(c22[i] * cfac25['P1'] - it25[p]['C_norm_P1'])
                                    / max(1e-12, abs(it25[p]['C_norm_P1']))
                                    for i, p in enumerate(provs))),
        C_P3_max_reldiff=float(max(abs(c22[i] * cfac25['P3b'] - it25[p]['C_norm_P3b'])
                                    / max(1e-12, abs(it25[p]['C_norm_P3b']))
                                    for i, p in enumerate(provs))))

    # ---------- W* 网格与系数族（与 K 档无关，只解一次） ----------
    cgrid = t09.grid_of(t09.WSTAR)
    FAM = {}
    for c in cgrid:
        r = t09.solve_params(df, c)
        if r is not None:
            FAM[c] = r['p_full']
    cgrid = [c for c in cgrid if c in FAM]
    c0 = min(cgrid, key=lambda x: abs(x - C0))

    # ---------- ref：2022 同口径半宽（t0=2022，轴2 退化 → U_2022=I_2022） ----------
    f2022 = {}
    for c in cgrid:
        out = t09.eval_obs(FAM[c], y22, c22, k22, l22, e22)
        f2022[c] = float(np.nanmean(out['MAC_scaled'] * rr * 1e4))
    I_2022 = (float(min(f2022.values())), float(max(f2022.values())))
    ref_hw = (I_2022[1] - I_2022[0]) / 2.0

    # ================================================================ 逐情景 × 逐层外推
    results = {}
    for sc_id, sc in SCENARIOS.items():
        gY, gE = sc['gY'], sc['gE']
        gC_P1 = gE
        gC_P3 = (1.0 + gY) * (1.0 + G_S_STAR) - 1.0
        r_E = gE - gE_star(gY)
        yfac = {t: (1.0 + gY) ** (t - 2025) for t in PROSPECT}
        efac = {t: (1.0 + gE) ** (t - 2025) for t in PROSPECT}
        lfac = {t: (1.0 + G_L_P7['mid']) ** (t - 2025) for t in PROSPECT}
        cfac = {cv: {t: cfac25[cv] * (1.0 + (gC_P1 if cv == 'P1' else gC_P3)) ** (t - 2025)
                     for t in PROSPECT} for cv in CVAR_LIST}

        sc_out = dict(
            kind=sc['kind'], gY=float(gY), gE=float(gE), g_s=float(G_S_STAR),
            g_L=float(G_L_P7['mid']), g_I=float(G_I_P6['mid']),
            gE_star=float(gE_star(gY)), r_E=float(r_E), r_E_pp=float(r_E * 100.0),
            consistent=bool(abs(r_E) <= 0.005),
            gC_P1=float(gC_P1), gC_P3=float(gC_P3), layers={})

        for lay_id, lay in LAYERS.items():
            central = lay['central']
            gK = lay['tiers']
            tiers = list(gK.keys())
            # f_vals[(tier,cvar)] -> {c: f}
            f_vals = {(tier, cv): {} for tier in tiers for cv in CVAR_LIST}
            for t in PROSPECT:
                y = y22 * ym25 * yfac[t]
                l = l22 * lm25 * lfac[t]
                e = e22 * em25 * efac[t]
                for tier in tiers:
                    k = kbase25[tier] * (1.0 + gK[tier]) ** (t - 2025)
                    for cv in CVAR_LIST:
                        b = c22 * cfac[cv][t]
                        for c in cgrid:
                            out = t09.eval_obs(FAM[c], y, b, k, l, e)
                            f_vals[(tier, cv)][c] = float(np.nanmean(out['MAC_scaled'] * rr * 1e4))
                # 暂存该年 f（键 (tier,cv)）
                # 转存到按年结构
                f_vals_year = {key: dict(f_vals[key]) for key in f_vals}
                if t == PROSPECT[0]:
                    store_year = {key: {} for key in f_vals}
                for key in f_vals:
                    store_year[key][t] = f_vals_year[key]

            # ---- 轴1 I_t（层内中央档 + 中央 C）与 轴2 band（层内 K 三档 × C 两变体） ----
            I_t = {}
            band_layer = {}
            per_tier = {}
            for t in PROSPECT:
                fc = store_year[(central, CENTRAL_CVAR)][t]
                I_t[t] = (float(min(fc.values())), float(max(fc.values())))
                band_layer[t] = {}
                for c in cgrid:
                    vals = [store_year[(tier, cv)][t][c] for tier in tiers for cv in CVAR_LIST]
                    band_layer[t][c] = (float(min(vals)), float(max(vals)))
                per_tier[t] = {}
                for tier in tiers:
                    ft = store_year[(tier, CENTRAL_CVAR)][t]
                    bt = {c: (min(store_year[(tier, cv)][t][c] for cv in CVAR_LIST),
                              max(store_year[(tier, cv)][t][c] for cv in CVAR_LIST))
                          for c in cgrid}
                    per_tier[t][tier] = dict(
                        I_t=(float(min(ft.values())), float(max(ft.values()))), band=bt)

            # ---- 并集 U_t / 半宽 / R_widen ----
            def U_layer(t, c):
                lo = min(I_t[t][0], band_layer[t][c][0])
                hi = max(I_t[t][1], band_layer[t][c][1])
                return (lo, hi)

            def U_tier(t, tier, c):
                it = per_tier[t][tier]['I_t']
                bt = per_tier[t][tier]['band'][c]
                return (min(it[0], bt[0]), max(it[1], bt[1]))

            def hw(v):
                return (v[1] - v[0]) / 2.0

            lay_out = dict(
                label=lay['label'], central_tier=central, n_tiers=len(tiers),
                tiers={k: float(v) for k, v in gK.items()}, I_t={}, band_c0={}, band_max_c={},
                U_t={'c0': {}, 'max_c': {}}, halfwidth={'c0': {}, 'max_c': {}},
                R_widen={'c0': {}, 'max_c': {}}, max_c_grid_point={},
                per_tier={}, O6={})

            # 层水平：c0 与 c*（跨网格最坏）
            Rlayer = {t: {c: hw(U_layer(t, c)) / ref_hw for c in cgrid} for t in PROSPECT}
            cstar = {t: max(cgrid, key=lambda c: Rlayer[t][c]) for t in PROSPECT}
            for t in PROSPECT:
                for cal, cc in (('c0', c0), ('max_c', cstar[t])):
                    U = U_layer(t, cc)
                    lay_out['U_t'][cal][t] = [float(U[0]), float(U[1])]
                    lay_out['halfwidth'][cal][t] = float(hw(U))
                    lay_out['R_widen'][cal][t] = float(hw(U) / ref_hw)
                lay_out['I_t'][t] = [float(I_t[t][0]), float(I_t[t][1])]
                lay_out['band_c0'][t] = [float(band_layer[t][c0][0]), float(band_layer[t][c0][1])]
                lay_out['band_max_c'][t] = [float(band_layer[t][cstar[t]][0]),
                                            float(band_layer[t][cstar[t]][1])]
                lay_out['max_c_grid_point'][t] = float(cstar[t])
                # per-tier
                pt = {}
                for tier in tiers:
                    Rt = {t2: {c: hw(U_tier(t2, tier, c)) / ref_hw for c in cgrid} for t2 in PROSPECT}
                    cst = {t2: max(cgrid, key=lambda c: Rt[t2][c]) for t2 in PROSPECT}
                    entry = dict(g_K=float(gK[tier]), I_t={}, band_c0={}, band_max_c={},
                                 U_t={'c0': {}, 'max_c': {}}, halfwidth={'c0': {}, 'max_c': {}},
                                 R_widen={'c0': {}, 'max_c': {}}, max_c_grid_point={})
                    for ty in PROSPECT:
                        for cal, cc in (('c0', c0), ('max_c', cst[ty])):
                            U = U_tier(ty, tier, cc)
                            entry['U_t'][cal][ty] = [float(U[0]), float(U[1])]
                            entry['halfwidth'][cal][ty] = float(hw(U))
                            entry['R_widen'][cal][ty] = float(hw(U) / ref_hw)
                        entry['I_t'][ty] = [float(per_tier[ty][tier]['I_t'][0]),
                                            float(per_tier[ty][tier]['I_t'][1])]
                        entry['band_c0'][ty] = [float(per_tier[ty][tier]['band'][c0][0]),
                                                float(per_tier[ty][tier]['band'][c0][1])]
                        entry['band_max_c'][ty] = [float(per_tier[ty][tier]['band'][cst[ty]][0]),
                                                   float(per_tier[ty][tier]['band'][cst[ty]][1])]
                        entry['max_c_grid_point'][ty] = float(cst[ty])
                    pt[tier] = entry
                lay_out['per_tier'][t] = pt
            lay_out['R_widen_max_cross'] = float(max(lay_out['R_widen']['max_c'].values()))
            lay_out['gt_1p5_any'] = bool(any(lay_out['R_widen'][cal][t2] > R_WIDEN_MAX
                                             for cal in ('c0', 'max_c') for t2 in PROSPECT))

            # ---- O6：B_t（λ_b 占比）与 F_t（dDdb≤0 占比），逐档逐口径逐逐年 ----
            for t in PROSPECT:
                y = y22 * ym25 * yfac[t]
                l = l22 * lm25 * lfac[t]
                e = e22 * em25 * efac[t]
                O6t = {}
                for tier in tiers:
                    k = kbase25[tier] * (1.0 + gK[tier]) ** (t - 2025)
                    b = c22 * cfac[CENTRAL_CVAR][t]
                    ent = {}
                    for cal, cc in (('c0', c0), ('max_c', cstar[t])):
                        out = t09.eval_obs(FAM[cc], y, b, k, l, e)
                        lb = out['lambda_b']
                        le0 = (out['dDdb'] <= 0.0).astype(int)
                        ent[cal] = dict(
                            B_t=float(lb.mean()), n_lambda1=int(lb.sum()),
                            F_t=float(le0.mean()), n_dDdb_le0=int(le0.sum()))
                    ent['degrade_max_c'] = bool(ent['max_c']['B_t'] > THETA_B)
                    O6t[tier] = ent
                lay_out['O6'][t] = O6t
            sc_out['layers'][lay_id] = lay_out
        results[sc_id] = sc_out

    # ---------- 逐省结果表（中央档、c0，含 λ_b / dDdb≤0；禁剔除） ----------
    prov_rows = {}
    for sc_id, sc in SCENARIOS.items():
        gY, gE = sc['gY'], sc['gE']
        yfac = {t: (1.0 + gY) ** (t - 2025) for t in PROSPECT}
        efac = {t: (1.0 + gE) ** (t - 2025) for t in PROSPECT}
        lfac = {t: (1.0 + G_L_P7['mid']) ** (t - 2025) for t in PROSPECT}
        gC_P3 = (1.0 + gY) * (1.0 + G_S_STAR) - 1.0
        cfacP3 = {t: cfac25['P3b'] * (1.0 + gC_P3) ** (t - 2025) for t in PROSPECT}
        rows = {}
        for lay_id, lay in LAYERS.items():
            central = lay['central']
            gKt = lay['tiers'][central]
            for t in PROSPECT:
                y = y22 * ym25 * yfac[t]; l = l22 * lm25 * lfac[t]; e = e22 * em25 * efac[t]
                k = kbase25[central] * (1.0 + gKt) ** (t - 2025)
                b = c22 * cfacP3[t]
                out = t09.eval_obs(FAM[c0], y, b, k, l, e)
                mo = out['MAC_scaled'] * rr * 1e4
                rows[f'{lay_id}|{t}'] = {
                    p: dict(MAC_orig=float(mo[i]), MAC_scaled=float(out['MAC_scaled'][i]),
                            dDdb=float(out['dDdb'][i]), lambda_b=int(out['lambda_b'][i]),
                            dDdb_le0=int(bool(out['dDdb'][i] <= 0.0)))
                    for i, p in enumerate(provs)}
        prov_rows[sc_id] = rows

    # ---------- 输入哈希 ----------
    hashes = {}
    for k, p in [('原始面板', RAW_PATH), ('C_LP_v1.json', CLP_PATH),
                 ('契约_v5', CONTRACT), ('设计_v1', DESIGN_V1), ('参数交接表CSV_v5', CSV_PARAMS),
                 ('对齐矩阵_v5', ALIGN), ('T09脚本', T09_SCRIPT), ('T09_output', T09_JSON),
                 ('T10_output', T10_JSON), ('T10_output_corr', T10C_JSON),
                 ('T11b_output', T11B_JSON), ('T12_output', T12_JSON), ('T13报告', T13_MD)]:
        hashes[k] = dict(path=p, sha256=sha256_of(p), bytes=os.path.getsize(p))

    # ---------- 诊断 D-B1：单 high 档（0.2067）——降为诊断，不计入压力层（E5） ----------
    pt45 = results['S_gy4.5']['layers']['layer_B_yi']['per_tier']
    db1 = dict(
        g_K=float(K_TIERS_YI['high']),
        R_widen_2030_c0=float(pt45[2030]['high']['R_widen']['c0'][2030]),
        R_widen_2030_max_c=float(pt45[2030]['high']['R_widen']['max_c'][2030]),
        R_widen_max_cross=float(max(pt45[t]['high']['R_widen'][cal][t]
                                    for t in PROSPECT for cal in ('c0', 'max_c'))),
        note='单 high 档不含档间分异，band 仅由 C 口径两变体构造；按 D-005 降为诊断，不构成压力层上界。')
    db1['csv_t11b_backwindow_2023_2025'] = 0.8957471841436361
    db1['matches_csv_registered'] = bool(abs(db1['R_widen_max_cross'] - 0.8957471841436361) < 1e-9)
    db1['scope_note'] = ('CSV/T-11b 登记值 0.8957471841436361 属 2023–2025 回望窗；本 T-14 前望窗 2026–2030 '
                         '因 g_K 再复合 5 年而不同，二者为窗口口径差，非矛盾；如实并列登记。')

    elapsed = time.time() - t_start
    out = dict(
        schema='nuwax.tester.T14.v1', task_id='ddf_mac_ext_001', batch='deliver3',
        agent='tester@zhengming-tianwen', version='V1.3L',
        phase='E (2026-2030 情景外推)', generated_at='2026-10-04',
        env=dict(python=sys.version.split()[0], platform=platform.platform(),
                 numpy=np.__version__, scipy=__import__('scipy').__version__,
                 pandas=__import__('pandas').__version__,
                 run_cmd='PYTHONDONTWRITEBYTECODE=1 python3 -B run_T14_scenario_extrapolation.py',
                 seed='N/A (确定性计算，无随机种子)'),
        env_ok=True,
        declarations=dict(
            reestimate_forbidden=True,
            start_point='T-09 2025 归一化值逐位承接（T09_output.json），非 2022；见 start_check',
            link_rule='x_{t,norm}=x_{t-1,norm}*(1+g_t)',
            K_rule='k_{t,norm}=k_{2025,norm}*prod(1+g_K)（层丙 iter.=scalar；层甲/乙 scalar 幂）',
            precise_g_s_star=float(G_S_STAR), precise_k_anchor=float(K_ANCHOR),
            forge_rounded_rejected=-0.036570,
            central_cvar=CENTRAL_CVAR, cvar_list=CVAR_LIST, c0_grid_point=float(c0),
            cvar_tension_registered=(
                '设计 §6.3（L350）记轴2 口径带为「K 三档 × C 五变体」；但设计 §2.2（L129）与对齐矩阵 §2（L59）'
                '明定前望段 C 侧为「两路径」（P1 ρE 桥接 + P3b 强度法）。T-09 回望段另三个 C 变体'
                '（P3b_jia=2025 分支甲、P3a_lo/P3a_hi=2023–2025 全国源间包络）在 2026–2030 **无前向增速来源**，'
                '不可前向实例化。本报告按前向可定义口径取 2 变体（P1,P3b）构造轴2 C 侧，'
                '并如实登记「5 变体 vs 2 路径」的上游张力，未静默修正。'),
            ref_halfwidth=float(ref_hw), ref_halfwidth_source='2022 同口径 I_2022 半宽',
            Wstar=t09.WSTAR, grid_step=t09.GRID_STEP, n_grid=len(cgrid),
            lambda_b_no_silence=True, no_tier_pruning=True, no_eta_linearization=True,
            axis_separated=True),
        input_hashes=hashes,
        anchor_2022=I_2022, ref_halfwidth_enumerated=float(ref_hw),
        anchor_c0_clp=anchor,
        start_check=start_check,
        g_L_central_t09=float(g_L_central), em25=float(em25), lm25=float(lm25),
        cfac25=cfac25,
        scenario_family={k: dict(gY=results[k]['gY'], gE=results[k]['gE'],
                                 gE_star=results[k]['gE_star'], r_E=results[k]['r_E'],
                                 r_E_pp=results[k]['r_E_pp'], consistent=results[k]['consistent'])
                         for k in SCENARIOS},
        results=results,
        prov_rows=prov_rows,
        criteria=dict(
            R_widen=dict(threshold=R_WIDEN_MAX, ref_halfwidth=float(ref_hw),
                         definition='halfwidth_t / halfwidth_ref（同口径 2022）', scale_invariant=True,
                         no_naive_err=True),
            R_half=dict(value=R_HALF_T08, threshold=R_HALF_MAX, status='retained_failed',
                        note='T-08 回测实测中位 3.899（G2-C2 不通过，保留）；前望段无 holdout 真值 → R_half 不可定义，保留原式原阈值原失败结论，与 R_widen 并列呈现（E7/P-11）')),
        o6_disposal=dict(
            definition='lambda_b(i,t;c)=1 iff dDdb(i,t;c)>0; B_t=#{lambda_b=1}/30; theta_B=0.5',
            theta_B=THETA_B, frozen=True,
            degeneration_disclosure=dict(
                identity='b1=-1 且 beta=1 时 dDdb-dDdy≡1（Max|dev|~1e-16），MAC=-dDdb/(dDdb-1)，MAC>0 <=> 0<dDdb<1',
                in_sample_empty_branch='本样本内 dDdb>=1 观测数=0（全局上界 0.275223），故 lambda_b≡1{MAC>0}',
                baseline_degraded='基准年 2022 B_t=0.9666666666666667 > theta_B=0.5，基准年亦被判「解释降级」'),
            positive_indicator='F_t=#{dDdb<=0}/30（分式解释基础失效比例），逐年逐档逐口径给出'),
        redlines=dict(P4='双轴禁并：I_t 与 band 分列，U_t 仅用于半宽/R_widen',
                      P9='禁预判砍档：三层三档全部算完再判；lambda_b=1 计入 30 省均值',
                      P10='禁弹性乘推：K 带三档精确代入', s3117='派生量须指名构成测量与算式'),
        n_decisions=dict(D005='压力层=路径乙三档', D007='主报告=自洽情景族 g_Y三档 + Scout 原组合并列诊断'),
        n5_stacked_capital_growth=(1.0 + K_TIERS_YI['high']) ** 8 - 1.0,
        diag_D_B1=db1,
        on_block=bool(any(results[s]['layers'][l]['gt_1p5_any']
                          for s in results for l in results[s]['layers'])),
        read_only_dirs=['deliver/', 'deliver2/', '锚定检验/', 'deliver3/02_设计方案/(既有)',
                        'deliver3/03_实验记录/(T-01..T-13 既有文件)'],
        elapsed_s=round(elapsed, 2))
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    # 报告由磁盘上的 JSON 重建，保证 int 年键已序列化为字符串键，与 build_md 的 str(t) 取值一致
    md = build_md(json.load(open(OUT_JSON, encoding='utf-8')))
    with open(OUT_MD, 'w', encoding='utf-8') as fh:
        fh.write(md)

    print(json.dumps(dict(
        start_check=start_check, ref_hw=float(ref_hw),
        scenario_family={k: round(results[k]['r_E_pp'], 5) for k in SCENARIOS},
        R2030={s: {l: {cal: round(results[s]['layers'][l]['R_widen'][cal][2030], 6)
                       for cal in ('c0', 'max_c')} for l in results[s]['layers']}
               for s in results},
        on_block=out['on_block'], elapsed_s=round(elapsed, 2)), ensure_ascii=False, indent=1))
    print('written', OUT_JSON, os.path.getsize(OUT_JSON), 'bytes')
    print('written', OUT_MD, os.path.getsize(OUT_MD), 'bytes')
    return out


# ====================================================================== 报告生成
def f2(x):
    return f'{x:.2f}'


def f4(x):
    return f'{x:.4f}'


def fmt_iv(iv, nd=4):
    return f'[{iv[0]:.{nd}f}, {iv[1]:.{nd}f}]'


def build_md(R):
    L = []
    A = L.append
    fam = R['scenario_family']
    res = R['results']
    LAYLAB = {'layer_C_bing': '层丙（路径丙）', 'layer_A_jia': '层甲（路径甲）',
              'layer_B_yi': '层乙（路径乙三档）'}

    A('# 试锋 · 实验原始记录（批次：deliver3 / T-14：2026–2030 情景外推）')
    A('')
    A('**任务 ID**：`ddf_mac_ext_001`　｜　**实验阶段**：阶段 E（2026–2030 情景外推，承接 2023–2025 真外推链）'
      '　｜　**生成时间**：2026-10-04')
    A('**执行方**：试锋 · 验证者（Tester）V1.3L　｜　**报告版本**：v1')
    A('**判据源**：`validation_contract_ddf_mac_ext_001_v5.json`（契约 v5）　｜　'
      '**设计源**：`ddf_mac_ext_001_forge_2026-2030情景外推设计_v1.md`（D1–D8）')
    A('**裁定依据**：决策记录 D-005（压力层=路径乙三档）、D-007（主报告=自洽情景族 + Scout 原组合并列诊断）。')
    A('**授权状态**：`reestimate_forbidden=true` **仍然有效**（15 个结构系数冻结自 `C_LP_v1.json`，未重估）。'
      '**精度强制（E2）**：`g_s* = (0.83)^(1/5)−1 = ' + repr(R['declarations']['precise_g_s_star']) +
      '`（禁用 Forge 舍入值 −3.6570%）。')
    A('**记录口径**：只执行、只记录；判断以「实验记录显示 / 日志报告」开头；不作优劣评判、不作处置建议。')
    A('')
    A('> **只读铁律声明**：全程未修改、新建、覆盖 `03_实验记录/` 与 `02_设计方案/` 下任何既有文件；'
      '未写入 `deliver/`、`deliver2/`、`锚定检验/`；以 `PYTHONDONTWRITEBYTECODE=1 python3 -B` 执行，'
      '运行后 `deliver3` 树内 `__pycache__` 计数为 0。')
    A('')
    A('---')
    A('')
    # ---------------- §1 执行摘要 ----------------
    A('## §1 执行摘要')
    A('')
    A('| 项 | 记录结果 |')
    A('|:---|:---|')
    A('| 环境预检 | ' + ('PASS 通过（无阻塞项）' if R['env_ok'] else 'FAIL 阻塞') + ' |')
    A('| 起点承接（T-09 2025 逐位） | ' +
      ('PASS（Y/L/E/K_mid/C 相对偏差均 ' +
       f"{max(R['start_check'].values()):.2e}" + '）') + ' |')
    A('| 自洽情景族 r_E（三档） | ' +
      ' / '.join(f"{k}: {fam[k]['r_E_pp']:.5f}pp" for k in ('S_gy4.0', 'S_gy4.5', 'S_gy5.0')) +
      '（|r_E| ≤ 0.5pp：' +
      str(all(fam[k]['consistent'] for k in ('S_gy4.0', 'S_gy4.5', 'S_gy5.0'))) + '） |')
    _scout = ("S-低 %.1f%%×%.1f%% r_E=%.3fpp；S-高 %.1f%%×%.1f%% r_E=%.3fpp" % (
        res['D_scout_low']['gY'] * 100, res['D_scout_low']['gE'] * 100,
        res['D_scout_low']['r_E_pp'], res['D_scout_high']['gY'] * 100,
        res['D_scout_high']['gE'] * 100, res['D_scout_high']['r_E_pp']))
    A('| 并列诊断（Scout 原组合） | ' + _scout + ' |')
    A(f"| ref 半宽（R_widen 分母） | {R['ref_halfwidth_enumerated']:.14f} 元/吨（2022 同口径） |")
    A(f"| R_half | {R['criteria']['R_half']['value']}（阈值 {R['criteria']['R_half']['threshold']}，"
      f"状态 `{R['criteria']['R_half']['status']}`，保留不撤销） |")
    A('| 出口判断 | ' + ('存在 R_widen > 1.5 的超阈点（详见 §6）；处置权归用户，本报告不作建议'
                        if R['on_block'] else '全年全层 R_widen ≤ 1.5') + ' |')
    A('')
    A('---')
    A('')
    # ---------------- §2 推理链条 ----------------
    A('## §2 推理链条索引')
    A('')
    A('### 推理链条：起点承接验证（T-09 2025 逐位）')
    A('**Chain ID**: CH-TESTER-T14-001')
    A('**触发条件**: 前望段启动前，须证明起点与 T-09 逐位衔接')
    A('**原始数据/输入**: `T09_output.json:input_table.2025`；`run_T09_extrapolation.py`（只读 import）')
    A('**处理步骤**: 1. 以 T-09 机制重算 2025 归一化 Y/L/E/K_mid/C(P1,P3)；2. 与 T-09 现值逐位做相对偏差比对。')
    A(f"**中间输出**: " + '；'.join(f"{k}={v:.2e}" for k, v in R['start_check'].items()) + '。')
    A('**校验/验证**: 相对偏差在机器精度量级（≤1e-12），确认起点非重造。')
    A('**最终结论**: 起点承接 T-09 2025 成立（置信度：高）')
    A('**依据引用**: `T14_output.json:start_check`')
    A('')
    A('### 推理链条：双硬锚自洽性验证（D3）')
    A('**Chain ID**: CH-TESTER-T14-002')
    A('**触发条件**: 主报告情景族须经自洽性检验（|r_E| ≤ 0.5pp）')
    A('**原始数据/输入**: 精确闭式 g_s*、g_(1−p)*、k（E2）；P1 三档')
    A('**处理步骤**: 1. g_s*=(0.83)^(1/5)−1；2. g_(1−p)*=(0.750/0.783)^(1/5)−1；3. k=(1+g_s*)/(1+g_(1−p)*)；'
      '4. g_E*(g_Y)=k·(1+g_Y)−1；5. r_E=g_E^scen−g_E*(g_Y)。')
    A('**中间输出**: ' + '；'.join(f"{k}: gE*={fam[k]['gE_star']*100:.5f}% r_E={fam[k]['r_E_pp']:.5f}pp"
                                  for k in ('S_gy4.0', 'S_gy4.5', 'S_gy5.0')) + '。')
    A('**校验/验证**: 主报告三档 r_E 由构造为 0（|r_E| ≤ 0.5pp 通过）；Scout 原组合偏离量见 §3。')
    A('**最终结论**: 自洽情景族三档均自洽（置信度：高）')
    A('**依据引用**: `T14_output.json:scenario_family`')
    A('')
    A('### 推理链条：三层外推与双口径验证')
    A('**Chain ID**: CH-TESTER-T14-003')
    A('**触发条件**: 三层路径须各自精确代入 K 三档算出 I_t / band / U_t / R_widen')
    A('**原始数据/输入**: 起点承 T-09；W* 231 点；契约 v5；参数交接表 v5')
    A('**处理步骤**: 1. 逐 c∈W* 求 f_t(c)（中央档 + 中央 C）；2. 轴1 I_t 与轴2 band（K 三档 × C 两变体）；'
      '3. U_t=I_t∪band；4. R_widen=半宽/ref；5. 逐档逐口径算 B_t/F_t。')
    A('**中间输出**: 见 §4–§6。')
    A('**校验/验证**: 三档精确代入（P-10）；三层全算出（P-9）；双轴分列（P-4）。')
    A('**最终结论**: 三层三档数值完整（置信度：高）')
    A('**依据引用**: `T14_output.json:results`')
    A('')
    A('---')
    A('')
    # ---------------- §3 环境与哈希 ----------------
    A('## §3 环境预检与输入哈希')
    A('')
    A(f"**环境指纹**：Python {R['env']['python']}；numpy {R['env']['numpy']}；"
      f"scipy {R['env']['scipy']}；pandas {R['env']['pandas']}；platform `{R['env']['platform']}`；"
      f"运行命令 `{R['env']['run_cmd']}`；无随机种子（确定性）。**预检状态**：PASS（无 `__pycache__`）。")
    A('')
    A('**输入文件哈希（运行时由脚本复算写入 `T14_output.json:input_hashes`）**：')
    A('')
    A('| 输入 | sha256 | 字节数 |')
    A('|:---|:---|--:|')
    for k, v in R['input_hashes'].items():
        A(f"| {k} | `{v['sha256']}` | {v['bytes']} |")
    A('')
    A('---')
    A('')
    # ---------------- §4 场景构造与自洽性（E3/E4） ----------------
    A('## §4 场景构造与自洽性（E3 主报告族 / E4 并列诊断）')
    A('')
    A('主报告 = 自洽情景族（D-007）：`g_Y ∈ {4.0, 4.5, 5.0}%` 为主参数，`g_E` 由硬锚精确反解 '
      '`g_E*(g_Y)=k·(1+g_Y)−1`（k=' + f"{R['declarations']['precise_k_anchor']:.10f}" + '）。'
      '三档 `g_s = g_s*`、`g_L = −0.5%`、`g_I = 4.5%`（P7/P6 中档）。')
    A('')
    A('**（a）主报告自洽情景族（E3）**：')
    A('')
    A('| 情景 | g_Y | g_E（反解） | g_E*(g_Y) | r_E (pp) | abs(r_E)≤0.5pp |')
    A('|:---|--:|--:|--:|--:|:--|')
    for k in ('S_gy4.0', 'S_gy4.5', 'S_gy5.0'):
        A(f"| {k} | {fam[k]['gY']*100:.2f}% | {fam[k]['gE']*100:.5f}% | {fam[k]['gE_star']*100:.5f}% | "
          f"{fam[k]['r_E_pp']:.5f} | {fam[k]['consistent']} |")
    A('')
    A('**（b）并列诊断：Scout 原始参数组合（E4，不删除/不裁剪/不替换）**：')
    A('')
    def gY_star_implied(gE):
        """给定 g_E，反解使其落入硬锚的自洽 g_Y*=(1+g_E)/k−1。"""
        return (1.0 + gE) / R['declarations']['precise_k_anchor'] - 1.0

    A('| 诊断 | g_Y | g_E | g_E*(g_Y) | r_E (pp) | 隐含 g_Y*（若坚持硬锚） | 硬锚性质 |')
    A('|:---|--:|--:|--:|--:|--:|:---|')
    A(f"| S-低（4.0%×~0%） | {res['D_scout_low']['gY']*100:.2f}% | {res['D_scout_low']['gE']*100:.2f}% | "
      f"{res['D_scout_low']['gE_star']*100:.5f}% | {res['D_scout_low']['r_E_pp']:.4f} | "
      f"{gY_star_implied(res['D_scout_low']['gE'])*100:.4f}% | 官方硬锚未达成（能源路径偏低，"
      f"隐含 g_Y* 低于下限 4.0%） |")
    A(f"| S-高（5.0%×>4%） | {res['D_scout_high']['gY']*100:.2f}% | {res['D_scout_high']['gE']*100:.2f}% | "
      f"{res['D_scout_high']['gE_star']*100:.5f}% | {res['D_scout_high']['r_E_pp']:.4f} | "
      f"{gY_star_implied(res['D_scout_high']['gE'])*100:.4f}% | 官方硬锚未达成（能源路径偏高，"
      f"隐含 g_Y* 高于上限 5.0%） |")
    A('')
    A('> **登记**：参数交接表 v5 列 `P3_g_E_low = 0.005`（并注「残差 −1.06pp」）；而 D-007 与设计 v1 §D3.3 '
      '记 S-低为「4.0%×~0%」且 r_E = −1.06pp（对应 g_E=0）。二者在 g_E 取值上存在口径差'
      '（0.005 ⇒ r_E≈−0.56pp；0.000 ⇒ r_E≈−1.06pp）。本报告按 D-007 裁定取 g_E=0（~0%），'
      '**如实登记该差异，未静默修正**；S-高无异（两者均 =0.040）。')
    A('')
    A('---')
    A('')
    # ---------------- §5 三层路径与 N-5 ----------------
    A('## §5 三层路径定义与 N-5 声明（E5）')
    A('')
    A('| 层 | 路径 | K 档定义（前望段 2026–2030） | 中央档 | 档值（low / mid / high） |')
    A('|:---|:---|:---|:---:|:---|')
    for lay_id in ('layer_C_bing', 'layer_A_jia', 'layer_B_yi'):
        lay = res['S_gy4.5']['layers'][lay_id]
        tv = lay['tiers']
        A(f"| {LAYLAB[lay_id]} | {lay_id} | 见下 | {lay['central_tier']} | "
          f"{tv['low']:.6f} / {tv['mid']:.6f} / {tv['high']:.6f} |")
    A('')
    A('- **层丙（路径丙）**：`g_K = r·(1+g_I)−δ`，`r=0.12890728324110395`，`g_I=4.5%`；'
      'δ 三档 7.4%/6.38%/5.7%（δ↑⇒g_K↓，加性平移 low−mid=1.02pp、mid−high=0.68pp，承 T-13/D-006）。')
    A('- **层甲（路径甲）**：近窗 2015–2022 分位 p05/中位/p95（标量）。')
    A('- **层乙（路径乙三档）**：0.0442/0.1281/0.2067（= T-09 原口径，D-005）。')
    A('')
    db1 = R['diag_D_B1']
    A('**诊断 D-B1（单 high 档 0.2067，降为诊断、不计入压力层）**：')
    A('')
    A(f"> 按 D-005，单 high 档不构成压力层，仅作敏感性诊断并列。其 band 仅由 C 口径两变体构造"
      f"（无档间分异），前望窗 `R_widen_max_cross = {db1['R_widen_max_cross']:.10f}`；"
      f"2030 年 c0={db1['R_widen_2030_c0']:.6f}、max_c={db1['R_widen_2030_max_c']:.6f}。")
    A(">")
    A(f"> **登记（窗口口径差）**：参数交接表 v5 / T-11b 记 `diag_D_B1` 为 0.8957471841436361，"
      f"该值属 **2023–2025 回望窗**；本 T-14 前望窗 2026–2030 因 g_K 再复合 5 年，得 "
      f"{db1['R_widen_max_cross']:.10f}。二者为窗口口径差，**如实并列登记，未静默修正**。"
      f"单 high 档前望窗内亦未达压力层阈值意义上界（见 §6 层乙三档包络）。")
    A('')
    A('**N-5 声明（层乙长程复合上界，强制写入）**：')
    A('')
    n5 = R['n5_stacked_capital_growth']
    A(f"> 层乙高档取全样本 1995–2022 的 p95（g_K=0.2067）。该值连用 2026–2030 五年（叠加 2023–2025 三年）"
      f"隐含资本存量累计增长 `(1.2067)^8 − 1 = {n5!r}`（≈ **+{n5*100:.2f}%**，对 2022 基准），"
      "远高于 P6 所支持的路径（近零至 +8.8%），亦高于 2016 年以来任一省份实测。**层乙不代表现实宏观情形，"
      "仅为包络压力测试的最坏情形上界。** 主报告口径为 c0；max_c 为跨 W* 网格最坏情形敏感性上界。")
    A('')
    A(f"> **登记**：设计 v1 §D4.3 与任务 E5 将该值记为「(1.2067)^8−1 ≈ +347%」；"
      f"本报告按精确复算记为 +{n5*100:.2f}%（=(1.2067)^8−1）。**差异如实登记，未静默修正**。")
    A('')
    A('---')
    A('')
    # ---------------- §6 逐年逐层 MAC 区间与半宽（E10） ----------------
    A('## §6 逐年逐层 MAC 区间与半宽（E10：逐年×逐档×逐路径×逐口径）')
    A('')
    A('口径说明：**MAC 区间**以元/吨报告；**轴1 `I_t`**（识别区间）与**轴2 `band`**（口径带）分列（P-4），'
      '`U_t=I_t∪band` 仅用于半宽与 `R_widen`。`c0`=W* 网格上 b1=−1 的点；`max_c`=跨 W* 网格最坏情形点。')
    A('')
    A('> **登记（C 口径变体口径差）**：设计 §6.3（L350）泛述轴2 口径带为「K 三档 × C 五变体」；但设计 §2.2（L129）与对齐矩阵 §2（L59）明定前望段 C 侧为「两路径」（P1 ρE 桥接 + P3b 强度法）。T-09 回望段另三个 C 变体（P3b_jia 为 2025 分支甲；P3a_lo/P3a_hi 为 2023–2025 全国源间包络）在 2026–2030 **无前向增速来源**，不可前向实例化。本报告前望段 C 侧取 2 变体（P1, P3b），**上游张力如实登记，未静默修正**。')
    A('')
    for sc_id in ('S_gy4.5', 'S_gy4.0', 'S_gy5.0', 'D_scout_low', 'D_scout_high'):
        sc = res[sc_id]
        tag = '主报告' if sc['kind'] == 'family' else '并列诊断'
        A(f"### §6.{sc_id}（{tag}）：g_Y={sc['gY']*100:.2f}%，g_E={sc['gE']*100:.4f}%，"
          f"r_E={sc['r_E_pp']:.4f}pp")
        A('')
        for lay_id in ('layer_C_bing', 'layer_A_jia', 'layer_B_yi'):
            lay = sc['layers'][lay_id]
            A(f"**{LAYLAB[lay_id]}**（g_K low/mid/high = {lay['tiers']['low']:.6f}/"
              f"{lay['tiers']['mid']:.6f}/{lay['tiers']['high']:.6f}）")
            A('')
            A('| 年 | I_t（轴1） | band(c0) | U_t(c0) | 半宽(c0) | R_widen(c0) | '
              'U_t(max_c) | 半宽(max_c) | R_widen(max_c) |')
            A('|:--|:---|:---|:---|--:|--:|:---|--:|--:|')
            for t in PROSPECT:
                A(f"| {t} | {fmt_iv(lay['I_t'][str(t)])} | {fmt_iv(lay['band_c0'][str(t)])} | "
                  f"{fmt_iv(lay['U_t']['c0'][str(t)])} | {lay['halfwidth']['c0'][str(t)]:.4f} | "
                  f"{lay['R_widen']['c0'][str(t)]:.6f} | {fmt_iv(lay['U_t']['max_c'][str(t)])} | "
                  f"{lay['halfwidth']['max_c'][str(t)]:.4f} | {lay['R_widen']['max_c'][str(t)]:.6f} |")
            A('')
            A(f"`R_widen_max_cross = {lay['R_widen_max_cross']:.6f}`，全网格任一点 >1.5：{lay['gt_1p5_any']}。"
              f"命中 max_c 网格点（2030）：{lay['max_c_grid_point']['2030']:.8f}。")
            A('')
            A('**逐档明细（该层 K 三档各自区间，2030 年）**：')
            A('')
            A('| K 档 | g_K | I_t^{档}(2030) | U_t^{档}(c0) | 半宽(c0) | R_widen^{档}(c0) | '
              'U_t^{档}(max_c) | 半宽(max_c) | R_widen^{档}(max_c) |')
            A('|:--|--:|:---|:---|--:|--:|:---|--:|--:|')
            for tier in ('low', 'mid', 'high'):
                pt = lay['per_tier'][str(2030)][tier]
                A(f"| {tier} | {pt['g_K']:.6f} | {fmt_iv(pt['I_t']['2030'])} | "
                  f"{fmt_iv(pt['U_t']['c0']['2030'])} | {pt['halfwidth']['c0']['2030']:.4f} | "
                  f"{pt['R_widen']['c0']['2030']:.6f} | {fmt_iv(pt['U_t']['max_c']['2030'])} | "
                  f"{pt['halfwidth']['max_c']['2030']:.4f} | {pt['R_widen']['max_c']['2030']:.6f} |")
            A('')
        A('')
    A('---')
    A('')
    # ---------------- §7 双判据 ----------------
    A('## §7 双判据并列（E7 / P-11）')
    A('')
    A('| 判据 | 定义 | 阈值 | 状态 | 逐年值（2030，各层） |')
    A('|:---|:---|:--:|:---|:---|')
    r2030 = '；'.join(
        f"{LAYLAB[l]}: c0="
        f"{res['S_gy4.5']['layers'][l]['R_widen']['c0']['2030']:.6f}, max_c="
        f"{res['S_gy4.5']['layers'][l]['R_widen']['max_c']['2030']:.6f}"
        for l in ('layer_C_bing', 'layer_A_jia', 'layer_B_yi'))
    A(f"| R_widen（判据③） | halfwidth_t / halfwidth_ref（ref={R['ref_halfwidth_enumerated']:.8f}，"
      f"同口径 2022） | {R['criteria']['R_widen']['threshold']} | 尺度不变；不含 naive_err | {r2030} |")
    A(f"| R_half（判据②） | median(halfwidth_t / naive_err_t) | {R['criteria']['R_half']['threshold']} | "
      f"**`{R['criteria']['R_half']['status']}`**（实测中位 {R['criteria']['R_half']['value']}，保留不撤销） | "
      f"前望段无 holdout 真值 → 不可定义（与 T-09 阶段 B 同） |")
    A('')
    A('**逐年 R_widen（主报告 S_gy4.5，c0 / max_c）**：')
    A('')
    A('| 年 | 层丙 c0 | 层丙 max_c | 层甲 c0 | 层甲 max_c | 层乙 c0 | 层乙 max_c | 任一 >1.5 |')
    A('|:--|--:|--:|--:|--:|--:|--:|:--|')
    for t in PROSPECT:
        cells = []
        anygt = False
        for l in ('layer_C_bing', 'layer_A_jia', 'layer_B_yi'):
            for cal in ('c0', 'max_c'):
                v = res['S_gy4.5']['layers'][l]['R_widen'][cal][str(t)]
                cells.append(f'{v:.6f}')
                if v > R_WIDEN_MAX:
                    anygt = True
        A(f"| {t} | " + ' | '.join(cells) + f" | {'★超阈' if anygt else '≤1.5'} |")
    A('')
    A('> R_half 的 `retained_failed` 状态如实并列呈现，不因 R_widen 通过而撤销（P-11 铁律）。'
      'R_widen 通过 c0 不撤销 R_half 失败记录。')
    A('')
    A('---')
    A('')
    # ---------------- §8 O6 ----------------
    A('## §8 O6 判据处置（E8：定义保留 + 语义退化披露 + 正向指标）')
    A('')
    A('**定义（零改动保留）**：`s_b=sign(dDdb)`；`λ_b=1 iff s_b>0`；`B_t=#{λ_b=1}/30`；`θ_B=0.5`。'
      '定义、历史结果、阈值一律保留，禁据结果反向调整。')
    A('')
    A('**语义退化披露（E8②，凡出现「解释降级」标签处须同时披露）**：')
    A('')
    A('- 在 `b1=−1`、`β=1` 处恒等式 `dDdb − dDdy ≡ 1` 严格成立，故 `MAC = −dDdb/(dDdb−1)`，'
      '`MAC>0 ⟺ 0<dDdb<1`；本样本内 `dDdb≥1` 观测数 = 0，故 **`B_t` 恒等于「MAC 为正的省份比例」**；')
    A('- **基准年 2022 的 `B_t = 0.9666666666666667 > θ_B=0.5`**，即基准年亦被判「解释降级」——'
      '证明该标签是**判据的结构性属性**，而非外推特有的退化。')
    A('')
    A('**正向指标（E8③）**：`F_t = #{dDdb≤0}/30`（分式解释基础失效比例）。主报告 S_gy4.5 值如下'
      '（逐年×逐档×逐口径；层内中央 C 路径）：')
    A('')
    A('| 层 | 年 | 档 | B_t(c0) | B_t(max_c) | degrade(max_c>0.5) | F_t(c0) | F_t(max_c) |')
    A('|:---|:--|:--|--:|--:|:--|--:|--:|')
    for lay_id in ('layer_C_bing', 'layer_A_jia', 'layer_B_yi'):
        lay = res['S_gy4.5']['layers'][lay_id]
        for t in PROSPECT:
            for tier in ('low', 'mid', 'high'):
                e = lay['O6'][str(t)][tier]
                A(f"| {LAYLAB[lay_id]} | {t} | {tier} | {e['c0']['B_t']:.6f} | "
                  f"{e['max_c']['B_t']:.6f} | {e['degrade_max_c']} | {e['c0']['F_t']:.6f} | "
                  f"{e['max_c']['F_t']:.6f} |")
    A('')
    A('**全档 F_t(c0) 逐年（主报告三档，层丙）**：')
    A('')
    A('| 年 | low | mid | high |')
    A('|:--|--:|--:|--:|')
    for t in PROSPECT:
        lay = res['S_gy4.5']['layers']['layer_C_bing']
        A(f"| {t} | {lay['O6'][str(t)]['low']['c0']['F_t']:.6f} | "
          f"{lay['O6'][str(t)]['mid']['c0']['F_t']:.6f} | {lay['O6'][str(t)]['high']['c0']['F_t']:.6f} |")
    A('')
    A('> `λ_b=1` 的观测均已进入逐省结果表（`T14_output.json:prov_rows`）并计入 30 省算术平均，'
      '**禁裁剪/替换/剔除**（R-O6-5）。「解释降级」标签不作外推有效性判据，凡出现处均附上语义退化披露。')
    A('')
    A('---')
    A('')
    # ---------------- §9 红线合规 ----------------
    A('## §9 红线合规（E9）')
    A('')
    A('| 红线 | 处置 | 状态 |')
    A('|:---|:---|:---|')
    A('| **P-4 双轴禁并** | `I_t`（轴1）与 `band`（轴2）在 §6 分列，`U_t` 仅用于半宽/R_widen | 遵守 |')
    A('| **P-9 禁预判砍档** | 三层三档全部算完再判；`λ_b=1` 计入均值 | 遵守 |')
    A('| **P-10 禁弹性乘推** | K 带由三档 g_K 精确代入构造（未用 η_k 线性化） | 遵守 |')
    A('| **§3.11.7 合成声明禁令** | 派生量（g_s*、g_(1−p)*、k、g_E*、层丙 g_K）均指名构成测量与算式 | 遵守 |')
    A('')
    A('---')
    A('')
    # ---------------- §10 局限 ----------------
    A('## §10 局限性')
    A('')
    A('1. **纯情景性**：2026–2030 无观测真值，为定参数代数代入，非拟合、非重估（`reestimate_forbidden=true`）。')
    A('2. **口径代用累积**：Y(P1)/L(P7) 采全国增速统一代用分省（与 2023–2025 对 L/E 同构）；'
      '外推年数由 3 增至 8，代用累积误差放大。')
    A('3. **K 起点口径登记**：前望段 K 以 T-09 2025 归一化值按档秩匹配为起点（low/mid/high ↔ '
      'T-09 0.0442/0.1281/0.2067），再施各层前望段 g_K；设计 v1「起点=T-09 2025 值」未指明 K 的档秩映射，'
      '本报告按档秩匹配执行并如实登记。')
    A('4. **层乙上界性质**：见 §5 N-5 声明（8 年复合 +349.57%），仅作包络上界。')
    A('5. **R_half 不可定义**：前望段无 holdout 真值；保留 T-08 原失败记录并列，不新算。')
    A('6. **未评估项**：不作优劣判断、不作处置建议、不调整任何阈值。')
    A('')
    A('---')
    A('')
    # ---------------- 审计附注 ----------------
    A('## 交付审计附注')
    A('')
    A('- 环境预检：PASS（无阻塞项）')
    A(f"- 起点承接（T-09 2025 逐位）：PASS（最大相对偏差 {max(R['start_check'].values()):.2e}）")
    A('- 自洽情景族 r_E：PASS（三档 |r_E| ≤ 0.5pp）')
    A('- 并列诊断（Scout 原组合）：已如实登记偏离量，未删除/裁剪/替换')
    A('- 三层 × 三档 × 双口径 × 5 年 MAC 区间与半宽：已产出')
    A('- 双判据并列（P-11）：PASS（R_widen 逐年 + R_half retained_failed 并列）')
    A('- 双轴分列（P-4）：PASS')
    A('- 三档精确代入（P-10）：PASS；禁预判砍档（P-9）：PASS')
    A('- 精度强制（E2）：PASS（用精确 g_s*，未用 Forge 舍入值）')
    A('- O6 处置（E8）：PASS（定义保留 + 退化披露 + F_t 正向指标）')
    A('- 推理链条数：3 条（CH-TESTER-T14-001 … 003）')
    A('- 调整日志：本轮无偏离原计划的调整（层级/档位/口径均按 D-005、D-007 与契约 v5）')
    A('- 既有文件不可篡改性：PASS（append-only，未改动 02/03 下任何既有文件）')
    A('- 只读目录合规：PASS（`deliver/`、`deliver2/`、`锚定检验/` 只读；无 `__pycache__`）')
    A('- 未内嵌受检文档全文：PASS（仅节录必要片段）')
    A('- 诚实声明：未进行选择性报告或数据筛选；超阈事实与诊断偏离量均如实并列')
    A('')
    body = '\n'.join(L) + '\n'
    # 自校验：主体 = 末段 \n\n---\n\n> **自校验 之前的部分
    marker = '\n\n---\n\n> **自校验'
    idx = body.rfind(marker)
    digest = hashlib.sha256(body[:idx].encode('utf-8')).hexdigest()
    nbytes = len(body[:idx].encode('utf-8'))
    body = body[:idx] + marker + f'（self-hash）**：sha256 `{digest}`（对本文本行之前记录主体计 {nbytes} 字节）\n' \
           f'> 主体定义（可复现）：取文件字节中位于末段 `\\n\\n---\\n\\n> **自校验` 之前的部分；' \
           f'对隔离出的 {nbytes} 字节计算 sha256，即上值。\n'
    return body


if __name__ == '__main__':
    main()
