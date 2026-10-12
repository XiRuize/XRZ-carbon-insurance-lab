#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-09 / ddf_mac_ext_001 / batch deliver3
阶段 B：2023–2025 真外推（基准 1995–2022，t0=2022）。

铁律（只读 + 只执行）：
- deliver/、deliver2/、锚定检验/ 全程只读（仅读取上游 xlsx/json；不复制、不写入）。
- 03_实验记录/ 既有 T-01..T-08 文件不可篡改；本脚本只新建 T-09 文件。
- 运行须 PYTHONDONTWRITEBYTECODE=1 python3 -B，严禁生成 __pycache__ 或任何多余文件。
- 不重估任何结构参数（reestimate_forbidden=true）：c0 处 15 系数取自 C_LP_v1.json；
  W* 网格上系数由「全窗重解 LP」给出（识别区间枚举，M-10/M-12，非结构重估）。
- 不静默丢弃出支撑观测（P-9）；不用 eta 弹性线性化（P-10）；双判据并列（P-11/P-12）。
输出：03_实验记录/T09_output.json
"""
import os
import sys
import json
import time
import hashlib
import platform
import numpy as np
import pandas as pd
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'T09_output.json')

# ---------------------------------------------------------------- 只读输入
RAW_PATH = 'C:/Users/生升/Desktop/2026智能+保险创新大赛/碳排放权保险/长期碳排放权价格跃升保险/01_引擎/_s28_deliver5_mirror/deliver/04_数据/原始数据_修正LGDP_Kbase_WJP_Lofficial_v1.xlsx'
CLP_PATH = 'C:/Users/生升/Desktop/2026智能+保险创新大赛/碳排放权保险/长期碳排放权价格跃升保险/01_引擎/_s28_deliver5_mirror/deliver/06_复现归档/outputs/C_LP_v1.json'
CONTRACT = 'C:/Users/生升/Desktop/2026智能+保险创新大赛/碳排放权保险/长期碳排放权价格跃升保险/01_引擎/_s28_deliver5_mirror/deliver3/02_设计方案/validation_contract_ddf_mac_ext_001.json'
DESIGN_V5 = 'C:/Users/生升/Desktop/2026智能+保险创新大赛/碳排放权保险/长期碳排放权价格跃升保险/01_引擎/_s28_deliver5_mirror/deliver3/02_设计方案/ddf_mac_ext_001_forge_外推方案设计_v5.md'
CSV_PARAMS = 'C:/Users/生升/Desktop/2026智能+保险创新大赛/碳排放权保险/长期碳排放权价格跃升保险/01_引擎/_s28_deliver5_mirror/deliver3/02_设计方案/parameters_handover_ddf_mac_ext_001.csv'
T08 = 'C:/Users/生升/Desktop/2026智能+保险创新大赛/碳排放权保险/长期碳排放权价格跃升保险/01_引擎/_s28_deliver5_mirror/deliver3/03_实验记录/ddf_mac_ext_001_tester_T08_外推与回测_v1.md'

PNAMES = ['a0', 'aK', 'aL', 'aE', 'b1', 'b2', 'aKK', 'aLL', 'aEE',
          'aKL', 'aKE', 'aLE', 'dK', 'dL', 'dE']
KEEP = [i for i in range(15) if i != 4]

# ---------------------------------------------------------------- 契约/设计常数
WSTAR = [[-0.3242844303508603, 0.10185339066179176],
         [0.11016869827632653, 0.1416513547484503]]          # 契约 A-5（冻结 W*）
C0 = 0.0                                                      # 交付锚点 b1=-1，gamma1=0
G_K = {'low': 0.0442, 'mid': 0.1281, 'high': 0.2067}          # 契约 K_band.g_K_tiers
K_TIERS = ['low', 'mid', 'high']
REF_HALFWIDTH_CONTRACT = 193.7968                             # 契约 A-4（真外推口径，唯一允许直接引用）
R_WIDEN_MAX = 1.5                                             # 契约 G2-C3
R_HALF_MAX = 3.0                                             # 契约 G2-C2（retained_failed）
R_HALF_T08 = 3.899                                           # T-08 实测中位（回测）
GRID_STEP = 0.002

# 全国实际 GDP 增速（国家统计公报；用于 P3b 强度法）
G_GDP_NAT = {2023: 0.052, 2024: 0.050, 2025: 0.050}
# 全国 CO2 强度变化率（官方公报；2023 官方逐字 / 2024 官方逐字 / 2025 二手，Scout-I）
G_INTENSITY = {2023: 0.000, 2024: -0.034, 2025: -0.050}
# 全国能源消费增速（Scout-D/G，统计公报）
G_E_NAT = {2023: 0.057, 2024: 0.043, 2025: 0.035}
# P3a：全国 CO2 源间增速包络（Scout-G 主表：2023 IEA+4.7/GCP+4.9/CM+2.5；2024 CM-0.3~气象+0.6；2025 CREA-1~GCP+0.4）
G_C_P3A = {2023: [-0.0, 0.025, 0.049], 2024: [-0.003, 0.006], 2025: [-0.010, 0.004]}

# Y 增速（Scout-A2 表 3-9 / 2025 公报）——2023 指数(上年=100)、2024 指数、2025 实际增长率(%)
Y_INDEX_2023 = {'北京': 105.2, '天津': 104.3, '河北': 105.5, '山西': 105.0, '内蒙古': 107.3,
                '辽宁': 105.3, '吉林': 106.3, '黑龙江': 102.6, '上海': 105.0, '江苏': 105.8,
                '浙江': 106.0, '安徽': 105.8, '福建': 104.5, '江西': 104.1, '山东': 106.0,
                '河南': 104.1, '湖北': 106.0, '湖南': 104.6, '广东': 104.8, '广西': 104.1,
                '海南': 109.2, '重庆': 106.1, '四川': 106.0, '贵州': 104.9, '云南': 104.4,
                '陕西': 104.3, '甘肃': 106.4, '青海': 105.3, '宁夏': 106.6, '新疆': 106.8}
Y_INDEX_2024 = {'北京': 105.2, '天津': 105.1, '河北': 105.4, '山西': 102.3, '内蒙古': 105.8,
                '辽宁': 105.1, '吉林': 104.3, '黑龙江': 103.2, '上海': 105.0, '江苏': 105.8,
                '浙江': 105.5, '安徽': 105.8, '福建': 105.5, '江西': 105.1, '山东': 105.7,
                '河南': 105.1, '湖北': 105.8, '湖南': 104.8, '广东': 103.5, '广西': 104.2,
                '海南': 103.7, '重庆': 105.7, '四川': 105.7, '贵州': 105.3, '云南': 103.3,
                '陕西': 105.3, '甘肃': 105.8, '青海': 102.7, '宁夏': 105.4, '新疆': 106.1}
Y_RATE_2025 = {'北京': 5.4, '天津': 4.8, '河北': 5.6, '山西': 4.0, '内蒙古': 4.7,
               '辽宁': 3.7, '吉林': 5.0, '黑龙江': 4.2, '上海': 5.4, '江苏': 5.3,
               '浙江': 5.5, '安徽': 5.5, '福建': 5.0, '江西': 5.2, '山东': 5.5,
               '河南': 5.6, '湖北': 5.5, '湖南': 4.8, '广东': 3.9, '广西': 5.1,
               '海南': 4.0, '重庆': 5.3, '四川': 5.5, '贵州': 4.9, '云南': 4.1,
               '陕西': 5.1, '甘肃': 5.8, '青海': 4.1, '宁夏': 5.3, '新疆': 5.5}

# L 口径断点登记（Scout-B）：2023/2024 30/30 可得但输入未转录；2025 不可得 → 全国代理趋势外延
L_GRID_NOTE = ('L 分省 2023/2024 值未在本轮只读输入中转录（Scout-B 仅实测 6 省），2025 30/30 不可得；'
               '依 v1 §4.3 候选网格 {近3年, 近5年, 保持2022} 以全国面板加总增速作代理，'
               '中央取「近3年均值」（2020–2022）。L 不在轴2 口径带（设计 §二映射表未列 L）。')
# 6 省实测 L 佐证（Scout-B，仅登记不用于构造）
L_6PROV_2024 = {'黑龙江': (1319.0, 1305.0), '山东': (5370.0, 5315.0), '湖南': (3238.0, 3198.0),
                '广西': (2529.0, 2483.0), '四川': (4722.0, 4663.0), '陕西': (2085.0, 2067.0)}


# ====================================================================== 模型（内联，只读上游数据）
def load_raw():
    return pd.read_excel(RAW_PATH)


def build_norm(panel, end_year=2022):
    """HC-2：逐省 1995..end_year 均值作冻结基准；返回含 *_norm/ratio/cy 的面板（全窗）。"""
    m = panel[panel['年份'] <= end_year].groupby('地区')[
        ['Y_raw', 'L_raw', 'E_raw', 'C_raw', 'K_raw']].mean()
    df = panel.copy()
    for c, cc in [('Y_raw', 'Y_norm'), ('L_raw', 'L_norm'), ('E_raw', 'E_norm'),
                  ('C_raw', 'C_norm'), ('K_raw', 'K_norm')]:
        df[cc] = (df[c] / df['地区'].map(m[c])).values
    df['ratio'] = (df['地区'].map(m['Y_raw']) / df['地区'].map(m['C_raw'])).values
    df['cy'] = (df['C_raw'] / df['Y_raw']).values
    df['ybar'] = df['地区'].map(m['Y_raw']).values
    df['bbar'] = df['地区'].map(m['C_raw']).values
    return df


def _design(y, b, k, l, e, beta=1.0):
    n = len(y)
    A = np.column_stack([
        np.ones(n), k, l, e, y,
        0.5 * beta * y ** 2 + 0.5 * b ** 2 / beta + y * b,
        0.5 * k ** 2, 0.5 * l ** 2, 0.5 * e ** 2,
        k * l, k * e, l * e,
        k * (beta * y + b), l * (beta * y + b), e * (beta * y + b)])
    s_y = beta * y + b
    s_b = y + b / beta
    sb = beta * y + b
    rows = []
    rhs = []
    r = np.zeros((n, 15)); r[:, 4] = 1; r[:, 5] = s_y; r[:, 12] = beta * k; r[:, 13] = beta * l; r[:, 14] = beta * e
    rows.append(r); rhs.append(np.zeros(n))
    r = np.zeros((n, 15)); r[:, 4] = -1; r[:, 5] = -s_b; r[:, 12] = -k; r[:, 13] = -l; r[:, 14] = -e
    rows.append(r); rhs.append(np.ones(n))
    r = np.zeros((n, 15)); r[:, 1] = -1; r[:, 6] = -k; r[:, 9] = -l; r[:, 10] = -e; r[:, 12] = -sb
    rows.append(r); rhs.append(np.zeros(n))
    r = np.zeros((n, 15)); r[:, 2] = -1; r[:, 7] = -l; r[:, 9] = -k; r[:, 11] = -e; r[:, 13] = -sb
    rows.append(r); rhs.append(np.zeros(n))
    r = np.zeros((n, 15)); r[:, 3] = -1; r[:, 8] = -e; r[:, 10] = -k; r[:, 11] = -l; r[:, 14] = -sb
    rows.append(r); rhs.append(np.zeros(n))
    return A, np.vstack(rows), np.concatenate(rhs)


def solve_params(df, g1, beta=1.0):
    y = df['Y_norm'].values; b = df['C_norm'].values
    k = df['K_norm'].values; l = df['L_norm'].values; e = df['E_norm'].values
    A, A_ub, b_ub = _design(y, b, k, l, e, beta)
    A_red = A[:, KEEP]
    b1 = -1.0 + g1 * beta
    gamma1 = g1 * beta
    A_all = np.vstack([-A_red, A_ub[:, KEEP]])
    b_all = np.concatenate([b1 * y + gamma1 * b, b_ub - A_ub[:, 4] * b1])
    f = A_red.sum(axis=0)
    res = linprog(b_all, A_eq=A_all.T, b_eq=-f, bounds=[(0, None)] * len(b_all), method='highs')
    if not res.success:
        return None
    lam = res.x
    act = np.where(lam > 1e-7)[0]
    p, *_ = np.linalg.lstsq(A_all[act], b_all[act], rcond=None)
    p_full = np.zeros(15); p_full[KEEP] = p; p_full[4] = b1
    D = A @ p_full + gamma1 * b
    return dict(g1=float(g1), b1=float(b1), p_full=p_full, sumD=float(D.sum()), D=D, ok=True)


def eval_obs(p_full, y, b, k, l, e, beta=1.0):
    """返回逐观测 dDdy/dDdb/MAC_scaled/dDdb符号。MAC_orig 由调用方按比例换算。"""
    a0, aK, aL, aE, b1, b2, aKK, aLL, aEE, aKL, aKE, aLE, dK, dL, dE = p_full
    gamma1 = b1 + 1.0
    dDdy = b1 + b2 * (beta * y + b) + beta * (dK * k + dL * l + dE * e)
    dDdb = gamma1 + b2 * (y + b / beta) + (dK * k + dL * l + dE * e)
    with np.errstate(divide='ignore', invalid='ignore'):
        MAC_scaled = -dDdb / dDdy
    return dict(dDdy=dDdy, dDdb=dDdb, MAC_scaled=MAC_scaled,
                s_b=np.sign(dDdb), lambda_b=(dDdb > 0).astype(int))


def ddf_value(p_full, y, b, k, l, e, beta=1.0):
    a0, aK, aL, aE, b1, b2, aKK, aLL, aEE, aKL, aKE, aLE, dK, dL, dE = p_full
    gamma1 = b1 + 1.0
    A = np.column_stack([
        np.ones(len(y)), k, l, e, y,
        0.5 * y ** 2 + 0.5 * b ** 2 + y * b,
        0.5 * k ** 2, 0.5 * l ** 2, 0.5 * e ** 2,
        k * l, k * e, l * e,
        k * (y + b), l * (y + b), e * (y + b)])
    return A @ p_full + gamma1 * b


# ====================================================================== 增长率输入构造
def grid_of(segs, step=GRID_STEP):
    g = []
    for a, b in segs:
        g += list(np.round(np.linspace(a, b, max(2, int(round((b - a) / step)) + 1)), 8))
    return sorted(set(g))


def c_factors(gC):
    """给定逐年 g_C（dict 年->率），返回该路径在 2023/2024/2025 的累积因子（t 年相对 2022）。"""
    f23 = 1 + gC[2023]
    f24 = f23 * (1 + gC[2024])
    f25 = f24 * (1 + gC[2025])
    return {2023: f23, 2024: f24, 2025: f25}


def build_c_paths():
    """返回 (central_var, band_vars)：central 用于轴1；band_vars 用于轴2 C 侧包络。"""
    gP1 = dict(G_E_NAT)                                        # P1: C∝E, g_C=g_E_nat
    gP3b = {t: (1 + G_GDP_NAT[t]) * (1 + G_INTENSITY[t]) - 1 for t in (2023, 2024, 2025)}
    # DG-2025: P3 连乘仅 2023–2024；2025 双分支（甲=连乘 / 乙=截断+趋势外延，主选）
    gP3b_yi = dict(gP3b); gP3b_yi[2025] = 0.5 * (gP3b[2023] + gP3b[2024])   # 乙：2023–2024 均值
    # P3a 源间包络（低/高）
    a_lo = {2023: min(G_C_P3A[2023]), 2024: min(G_C_P3A[2024]), 2025: min(G_C_P3A[2025])}
    a_hi = {2023: max(G_C_P3A[2023]), 2024: max(G_C_P3A[2024]), 2025: max(G_C_P3A[2025])}
    a_lo_yi = dict(a_lo); a_lo_yi[2025] = 0.5 * (a_lo[2023] + a_lo[2024])
    a_hi_yi = dict(a_hi); a_hi_yi[2025] = 0.5 * (a_hi[2023] + a_hi[2024])
    paths = {
        'P1':      c_factors(gP1),
        'P3b':     c_factors(gP3b_yi),      # 中央，2025 分支乙（主选）
        'P3b_jia': c_factors(gP3b),          # 2025 分支甲
        'P3a_lo':  c_factors(a_lo_yi),
        'P3a_hi':  c_factors(a_hi_yi),
    }
    return paths, dict(gP1=gP1, gP3b=gP3b, gP3b_yi=gP3b_yi, P3a=P3a_meta())


def P3a_meta():
    return {t: [float(G_C_P3A[t][0]), float(G_C_P3A[t][1])] for t in G_C_P3A}


def main():
    t_start = time.time()
    panel = load_raw()
    df = build_norm(panel, 2022)
    YEARS = df['年份'].values
    PROV = df['地区'].values
    provs = sorted(set(PROV))
    idx22 = {p: int(np.where((PROV == p) & (YEARS == 2022))[0][0]) for p in provs}
    N = len(df)

    # ---------- 冻结锚点自检（c0 复现 C_LP_v1.json） ----------
    clp = json.load(open(CLP_PATH, encoding='utf-8'))
    p_json = np.array([clp['params'][k] for k in PNAMES])
    r0 = solve_params(df, C0)
    anchor = dict(
        MAC2022_lp=float(np.nanmean(
            (eval_obs(r0['p_full'], df['Y_norm'].values, df['C_norm'].values, df['K_norm'].values,
                      df['L_norm'].values, df['E_norm'].values)['MAC_scaled']
             * (df['ratio'].values) * 1e4)[YEARS == 2022])),
        target_MAC2022=190.53058207373442,
        obj_sumD=float(r0['sumD']), target_sumD=132.83354146398435,
        p_max_absdiff_vs_json=float(np.max(np.abs(r0['p_full'] - p_json))))

    # ---------- 全国面板 L 代理增速（候选网格） ----------
    agg = panel.groupby('年份')[['L_raw']].sum().sort_index()
    gL = agg['L_raw'].pct_change()
    g_L_grid = dict(near3=float(gL.loc[2020:2022].mean()),
                    near5=float(gL.loc[2018:2022].mean()),
                    hold2022=float(gL.loc[2022]))
    g_L_central = g_L_grid['near3']

    # ---------- W* 网格与系数族 ----------
    cgrid = grid_of(WSTAR)
    FAM = {}
    for c in cgrid:
        r = solve_params(df, c)
        if r is not None:
            FAM[c] = r['p_full']
    cgrid = [c for c in cgrid if c in FAM]

    # ---------- 2023–2025 归一化输入构造（逐省） ----------
    Y22 = {p: float(df['Y_norm'].values[idx22[p]]) for p in provs}
    L22 = {p: float(df['L_norm'].values[idx22[p]]) for p in provs}
    E22 = {p: float(df['E_norm'].values[idx22[p]]) for p in provs}
    K22 = {p: float(df['K_norm'].values[idx22[p]]) for p in provs}
    C22n = {p: float(df['C_norm'].values[idx22[p]]) for p in provs}
    Craw22 = {p: float(df['C_raw'].values[idx22[p]]) for p in provs}
    Yraw22 = {p: float(df['Y_raw'].values[idx22[p]]) for p in provs}
    ratio = {p: float(df['ratio'].values[idx22[p]]) for p in provs}   # ybar/bbar（冻结）
    ybar = {p: float(df['ybar'].values[idx22[p]]) for p in provs}
    bbar = {p: float(df['bbar'].values[idx22[p]]) for p in provs}

    def ymult(p, t):
        m = 1.0
        if t >= 2023:
            m *= Y_INDEX_2023[p] / 100.0
        if t >= 2024:
            m *= Y_INDEX_2024[p] / 100.0
        if t >= 2025:
            m *= 1 + Y_RATE_2025[p] / 100.0
        return m

    def emult(t):
        m = 1.0
        for y in range(2023, t + 1):
            m *= 1 + G_E_NAT[y]
        return m

    def lmult(t):
        return (1 + g_L_central) ** (t - 2022)

    def kmult(t, tier):
        return (1 + G_K[tier]) ** (t - 2022)

    paths, pm = build_c_paths()

    # ---------- 逐 (c,t,tier,Cvar) 求 f_t = nanmean_i MAC_orig ----------
    y22 = np.array([Y22[p] for p in provs]); l22 = np.array([L22[p] for p in provs])
    e22 = np.array([E22[p] for p in provs]); k22 = np.array([K22[p] for p in provs])
    c22 = np.array([C22n[p] for p in provs]); cr22 = np.array([Craw22[p] for p in provs])
    yr22 = np.array([Yraw22[p] for p in provs]); rr = np.array([ratio[p] for p in provs])
    yb = np.array([ybar[p] for p in provs]); bb = np.array([bbar[p] for p in provs])
    ymg = np.array([ymult(p, 2025) for p in provs])
    ymg23 = np.array([ymult(p, 2023) for p in provs])
    ymg24 = np.array([ymult(p, 2024) for p in provs])
    ymult_by_t = {2023: ymg23, 2024: ymg24, 2025: ymg}
    emult_by_t = {t: emult(t) for t in (2023, 2024, 2025)}
    lmult_by_t = {t: lmult(t) for t in (2023, 2024, 2025)}

    CENTRAL = 'P3b'
    Cvar_list = ['P1', 'P3b', 'P3b_jia', 'P3a_lo', 'P3a_hi']

    # 每年每 c 的记录
    f_central = {t: {} for t in (2023, 2024, 2025)}          # 轴1：中央口径 f_t(c)
    band = {t: {} for t in (2023, 2024, 2025)}               # 轴2：B_t(c)=[min,max]
    # 附加：逐省 MAC/ dev / D / lambda_b（中央口径）
    per_obs_store = {}

    def inputs(t, tier, cvar):
        y = y22 * ymult_by_t[t]; l = l22 * lmult_by_t[t]; e = e22 * emult_by_t[t]
        k = k22 * kmult(t, tier); b = c22 * paths[cvar][t]
        return y, b, k, l, e

    for c in cgrid:
        p = FAM[c]
        for t in (2023, 2024, 2025):
            vals = []
            for tier in K_TIERS:
                for cvar in Cvar_list:
                    y, b, k, l, e = inputs(t, tier, cvar)
                    out = eval_obs(p, y, b, k, l, e)
                    mo = out['MAC_scaled'] * rr * 1e4
                    f = float(np.nanmean(mo))
                    vals.append(f)
                    if tier == 'mid' and cvar == CENTRAL:
                        f_central[t][c] = f
                        D = ddf_value(p, y, b, k, l, e)
                        cy = (cr22 * paths[cvar][t]) / (yr22 * ymult_by_t[t])
                        dev = out['MAC_scaled'] * rr * cy
                        per_obs_store[(t, c)] = dict(
                            MAC=mo, dev=dev, D=D, s_b=out['s_b'], lambda_b=out['lambda_b'])
            band[t][c] = (float(min(vals)), float(max(vals)))

    # ---------- 轴1 I_t 与 2022 基准（同口径枚举） ----------
    def I_t_at(t, fam_map):
        vs = [fam_map[c] for c in cgrid]
        return (float(min(vs)), float(max(vs)))

    I_t = {t: I_t_at(t, f_central[t]) for t in (2023, 2024, 2025)}
    # 2022 基准：同一中央口径枚举（t0=2022，轴2 退化 → U_2022 = I_2022）
    f2022 = {}
    for c in cgrid:
        y, b, k, l, e = y22, c22, k22, l22, e22
        out = eval_obs(FAM[c], y, b, k, l, e)
        f2022[c] = float(np.nanmean(out['MAC_scaled'] * rr * 1e4))
    I_2022 = (float(min(f2022.values())), float(max(f2022.values())))
    ref_halfwidth_enumerated = (I_2022[1] - I_2022[0]) / 2.0

    # ---------- 轴2 并集 U_t(c) 与 R_widen ----------
    def U_tc(t, c):
        lo = min(I_t[t][0], band[t][c][0])
        hi = max(I_t[t][1], band[t][c][1])
        return (lo, hi)

    def halfwidth(t, c):
        lo, hi = U_tc(t, c)
        return (hi - lo) / 2.0

    # ref：同口径 2022 半宽（真外推退化 → I_2022 半宽）；同时给契约值对照
    ref_hw = ref_halfwidth_enumerated
    R_widen = {}
    for t in (2023, 2024, 2025):
        R_widen[t] = {c: halfwidth(t, c) / ref_hw for c in cgrid}
    R_widen_c0 = {t: R_widen[t][min(cgrid, key=lambda x: abs(x - C0))] for t in (2023, 2024, 2025)}
    R_widen_max = {t: max(R_widen[t].values()) for t in (2023, 2024, 2025)}
    R_widen_max_cross = max(R_widen_max.values())

    # ---------- O4：Λ_t(c) 合法占比 D_t ----------
    O4 = {}
    for t in (2023, 2024, 2025):
        ok = [c for c in cgrid if float(np.nanmax(per_obs_store[(t, c)]['dev'])) <= 1.0]
        O4[t] = dict(D_t=len(ok) / len(cgrid), n_ok=len(ok), n_total=len(cgrid),
                     I_t_leg=None if not ok else (float(min(f_central[t][c] for c in ok)),
                                                  float(max(f_central[t][c] for c in ok))))
    # ---------- O5：φ_t 与 out-of-support ----------
    c0 = min(cgrid, key=lambda x: abs(x - C0))
    O5 = {}
    for t in (2023, 2024, 2025):
        D0 = per_obs_store[(t, c0)]['D']
        phi = int((D0 < -1e-8).sum()) / 30.0
        oos = [provs[i] for i in range(30)
               if all(per_obs_store[(t, c)]['D'][i] < -1e-8 for c in cgrid)]
        O5[t] = dict(phi_t=phi, n_neg=int((D0 < -1e-8).sum()), out_of_support=oos)
    # ---------- O6：B_t（按档）与 λ_b 记录 ----------
    O6 = {}
    for t in (2023, 2024, 2025):
        Bt = {}
        for tier in K_TIERS:
            at_c0 = None; mx = 0.0
            for c in cgrid:
                p = FAM[c]
                y, b, k, l, e = inputs(t, tier, CENTRAL)
                lb = eval_obs(p, y, b, k, l, e)['lambda_b']
                frac = float(lb.mean())
                if c == c0:
                    at_c0 = frac
                mx = max(mx, frac)
            Bt[tier] = dict(at_c0=at_c0, max_c=mx)
        O6[t] = dict(B_t=Bt, theta_B=0.5,
                     degrade={tier: (Bt[tier]['max_c'] > 0.5) for tier in K_TIERS})

    # ---------- 逐省逐 (t,tier,path) 结果表（中央 c0） ----------
    prov_table = {}
    for t in (2023, 2024, 2025):
        rows = {}
        for tier in K_TIERS:
            for cvar in Cvar_list:
                y, b, k, l, e = inputs(t, tier, cvar)
                mo = eval_obs(FAM[c0], y, b, k, l, e)['MAC_scaled'] * rr * 1e4
                rows[f'{tier}|{cvar}'] = {provs[i]: float(mo[i]) for i in range(30)}
        prov_table[t] = rows

    # ---------- 输入构造表（逐省 2023–2025 归一化输入，中央口径） ----------
    input_table = {}
    for t in (2023, 2024, 2025):
        input_table[t] = {}
        for i, p in enumerate(provs):
            input_table[t][p] = dict(
                Y_norm=float(y22[i] * ymult_by_t[t][i]),
                L_norm=float(l22[i] * lmult_by_t[t]),
                E_norm=float(e22[i] * emult_by_t[t]),
                K_norm_mid=float(k22[i] * kmult(t, 'mid')),
                C_norm_P1=float(c22[i] * paths['P1'][t]),
                C_norm_P3b=float(c22[i] * paths['P3b'][t]))

    # ---------- 输入哈希 ----------
    hashes = {}
    for k, p in [('原始面板', RAW_PATH), ('C_LP_v1.json', CLP_PATH), ('契约_v5', CONTRACT),
                 ('设计_v5', DESIGN_V5), ('参数交接表CSV', CSV_PARAMS), ('T08报告', T08)]:
        hashes[k] = dict(path=p, sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest(),
                         bytes=os.path.getsize(p))

    elapsed = time.time() - t_start
    out = dict(
        schema='nuwax.tester.T09.v1', task_id='ddf_mac_ext_001', batch='deliver3',
        agent='tester@zhengming-tianwen', version='V1.3L', phase='B (2023-2025 真外推)',
        env=dict(python=sys.version.split()[0], platform=platform.platform(),
                 numpy=np.__version__, scipy=__import__('scipy').__version__,
                 pandas=__import__('pandas').__version__,
                 cwd='03_实验记录', run_cmd='PYTHONDONTWRITEBYTECODE=1 python3 -B run_T09_extrapolation.py',
                 seed='N/A (确定性计算，无随机种子)'),
        declarations=dict(
            D003_closed=True, reestimate_forbidden=True,
            coef_source='C_LP_v1.json @ c0；W* 网格系数由全窗重解 LP 枚举（M-10/M-12）',
            ref_halfwidth_contract=REF_HALFWIDTH_CONTRACT,
            ref_halfwidth_enumerated=ref_halfwidth_enumerated,
            Wstar=WSTAR, grid_step=GRID_STEP, n_grid=len(cgrid)),
        input_hashes=hashes, anchor_check=anchor,
        g_L_grid=g_L_grid, g_L_central=g_L_central, L_note=L_GRID_NOTE,
        L_6prov={k: list(v) for k, v in L_6PROV_2024.items()},
        g_GDP_nat=G_GDP_NAT, g_intensity=G_INTENSITY, g_E_nat=G_E_NAT,
        g_K_tiers=G_K, P3a_meta=pm['P3a'],
        gC_paths={k: {str(t): v for t, v in d.items()} for k, d in paths.items()},
        gP3b={str(t): v for t, v in pm['gP3b'].items()},
        gP3b_yi={str(t): v for t, v in pm['gP3b_yi'].items()},
        I_2022=I_2022, I_t=I_t, ref_halfwidth_enumerated=ref_halfwidth_enumerated,
        band=band, f_central=f_central,
        R_widen_c0=R_widen_c0, R_widen_max=R_widen_max, R_widen_max_cross=R_widen_max_cross,
        R_half=dict(value=R_HALF_T08, threshold=R_HALF_MAX, status='retained_failed',
                    note='T-08 回测实测中位 3.899（G2-C2 不通过，保留）；阶段 B 无 holdout 真值 → R_half 不可定义（见报告 §6）'),
        O4=O4, O5=O5, O6=O6,
        prov_table=prov_table, input_table=input_table,
        on_block=bool(R_widen_max_cross > R_WIDEN_MAX),
        read_only_dirs=['deliver/', 'deliver2/', '锚定检验/', 'deliver3/03_实验记录/(T-01..T-08 既有文件)'])
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    # ---- 控制台摘要 ----
    print(json.dumps(dict(
        anchor=anchor, n_grid=len(cgrid),
        ref_hw_enumerated=ref_halfwidth_enumerated,
        I_2022=I_2022, I_t=I_t,
        R_widen_c0=R_widen_c0, R_widen_max=R_widen_max, on_block=out['on_block'],
        O4={t: O4[t]['D_t'] for t in O4}, O5_phi={t: O5[t]['phi_t'] for t in O5},
        O6={t: {k: round(v['max_c'], 3) for k, v in O6[t]['B_t'].items()} for t in O6},
        elapsed_s=round(elapsed, 2)), ensure_ascii=False, indent=1))
    print('\nwritten', OUT, os.path.getsize(OUT), 'bytes')


if __name__ == '__main__':
    main()
