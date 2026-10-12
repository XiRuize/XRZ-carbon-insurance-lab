"""CEA 20-trading-day research forecast, using a verified 2026 calendar.

Run with the bundled Python runtime. Requires numpy. Raw inputs stay intact.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'domestic_cea_data_20261005' / 'cea_daily_research_combined.csv'
OUT = ROOT / 'cea_20d_analysis_20261005'
ASOF = '2026-10-05'
CALENDAR_URL = 'https://overview.cneeex.com/c/2025-12-24/496961.shtml'
HOLIDAYS = [
    ('元旦', '2026-01-01', '2026-01-03'),
    ('春节', '2026-02-15', '2026-02-23'),
    ('清明节', '2026-04-04', '2026-04-06'),
    ('劳动节', '2026-05-01', '2026-05-05'),
    ('端午节', '2026-06-19', '2026-06-21'),
    ('中秋节', '2026-09-25', '2026-09-27'),
    ('国庆节', '2026-10-01', '2026-10-07'),
]
MODEL_NAMES = {
    'persistence': '价格不变基准',
    'mean20': '最近20日均价',
    'drift60': '最近60日收益率漂移',
    'ar1_60': '最近60日收益率AR(1)',
}
HORIZON = 20
WINDOW = 60


def write_csv(filename, rows):
    assert rows
    with (OUT / filename).open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def is_open(day):
    return day.weekday() < 5 and not any(date.fromisoformat(a) <= day <= date.fromisoformat(b)
                                        for _, a, b in HOLIDAYS)


def prediction(prices, origin, model):
    history = prices[:origin + 1]
    if model == 'persistence':
        return float(history[-1])
    if model == 'mean20':
        return float(history[-20:].mean())
    returns = np.diff(np.log(history[-61:]))
    if model == 'drift60':
        return float(history[-1] * np.exp(HORIZON * returns.mean()))
    design = np.column_stack([np.ones(len(returns) - 1), returns[:-1]])
    intercept, slope = np.linalg.lstsq(design, returns[1:], rcond=None)[0]
    last_return, total = returns[-1], 0.0
    for _ in range(HORIZON):
        last_return = intercept + slope * last_return
        total += last_return
    return float(history[-1] * np.exp(total))


def simulated_terminal(prices, origin, count, seed):
    # Circular moving blocks preserve some short-term return dependence.
    # Centering daily log returns enforces the zero-drift baseline assumption.
    returns = np.diff(np.log(prices[origin - WINDOW:origin + 1]))
    centered = returns - returns.mean()
    block_sums = np.array([centered[(start + np.arange(5)) % len(centered)].sum()
                           for start in range(len(centered))])
    rng = np.random.default_rng(seed)
    terminal_log_returns = rng.choice(block_sums, size=(count, HORIZON // 5)).sum(axis=1)
    return prices[origin] * np.exp(terminal_log_returns)


def main():
    OUT.mkdir(exist_ok=True)
    with INPUT.open(encoding='utf-8-sig') as file:
        all_rows = list(csv.DictReader(file))
    rows = [row for row in all_rows if row['date'].startswith('2026')]
    dates = [date.fromisoformat(row['date']) for row in rows]
    prices = np.array([float(row['close_cny_per_t']) for row in rows])
    assert np.all(np.isfinite(prices)) and np.all(prices > 0)
    assert len(set(dates)) == len(dates) and dates == sorted(dates)
    assert dates[-1] == date(2026, 9, 30)

    calendar = []
    day = date(2026, 1, 1)
    while day.year == 2026:
        holiday = next((label for label, a, b in HOLIDAYS
                        if date.fromisoformat(a) <= day <= date.fromisoformat(b)), '')
        calendar.append({'date': day.isoformat(), 'is_trading_day': int(is_open(day)),
                         'holiday': holiday, 'is_weekend': int(day.weekday() >= 5),
                         'source_url': CALENDAR_URL, 'available_date': '2025-12-24'})
        day += timedelta(days=1)
    expected = [date.fromisoformat(row['date']) for row in calendar
                if row['is_trading_day'] and row['date'] <= dates[-1].isoformat()]
    assert dates == expected, 'The observed 2026 dates do not match the official trading calendar'
    write_csv('trading_calendar_2026.csv', calendar)
    write_csv('observations_2026.csv', rows)

    # Select before the test period. Remove validation origins whose 20-day
    # labels are unavailable on the first test origin (19 purged origins).
    test_start = len(prices) - HORIZON - 40
    validation = list(range(WINDOW, test_start - HORIZON + 1))
    test = list(range(test_start, len(prices) - HORIZON))
    assert len(validation) == 42 and len(test) == 40
    assert dates[validation[-1] + HORIZON] <= dates[test[0]]

    samples = []
    for origin in range(WINDOW, len(prices) - HORIZON):
        partition = 'validation' if origin in validation else ('test' if origin in test else 'purged')
        sample = {
            'origin_date': dates[origin].isoformat(),
            'history_start_date': dates[origin - WINDOW].isoformat(),
            'history_end_date': dates[origin].isoformat(),
            'target_date': dates[origin + HORIZON].isoformat(),
            'partition': partition,
            'origin_close_cny_per_t': float(prices[origin]),
            'target_close_cny_per_t': float(prices[origin + HORIZON]),
            'target_return_pct': float((prices[origin + HORIZON] / prices[origin] - 1) * 100),
            'origin_source_type': rows[origin]['source_type'],
            'target_source_type': rows[origin + HORIZON]['source_type'],
        }
        for model in MODEL_NAMES:
            sample[f'{model}_forecast_cny_per_t'] = prediction(prices, origin, model)
        samples.append(sample)
    write_csv('samples_20d.csv', samples)

    metrics = []
    for model, name in MODEL_NAMES.items():
        for partition, origins in [('validation', validation), ('test', test)]:
            residuals = np.array([prediction(prices, origin, model) - prices[origin + HORIZON]
                                  for origin in origins])
            metrics.append({'model': model, 'model_name': name, 'partition': partition,
                            'n_origins': len(origins), 'mae_cny_per_t': float(np.abs(residuals).mean()),
                            'rmse_cny_per_t': float(np.sqrt(np.mean(residuals ** 2))),
                            'bias_prediction_minus_actual_cny_per_t': float(residuals.mean())})
    write_csv('model_backtest_metrics.csv', metrics)
    chosen = min((row for row in metrics if row['partition'] == 'validation'),
                 key=lambda row: row['mae_cny_per_t'])['model']
    baseline_test = next(row for row in metrics if row['model'] == 'persistence' and row['partition'] == 'test')
    candidate_test = next(row for row in metrics if row['model'] == chosen and row['partition'] == 'test')
    adopted = chosen if candidate_test['mae_cny_per_t'] < baseline_test['mae_cny_per_t'] else 'persistence'
    # Fallback is descriptive: its measured errors remain the benchmark. There
    # is no claim of an extra unseen evaluation set after this decision.

    interval_backtest = []
    for origin in test:
        distribution = simulated_terminal(prices, origin, 20000, 20261005 + origin)
        lower, upper = np.quantile(distribution, [.1, .9])
        actual = float(prices[origin + HORIZON])
        interval_backtest.append({'origin_date': dates[origin].isoformat(),
                                 'target_date': dates[origin + HORIZON].isoformat(),
                                 'actual_cny_per_t': actual, 'lower_10pct_cny_per_t': float(lower),
                                 'upper_90pct_cny_per_t': float(upper),
                                 'inside_interval': int(lower <= actual <= upper)})
    write_csv('interval_backtest.csv', interval_backtest)

    last = len(prices) - 1
    distribution = simulated_terminal(prices, last, 100000, 20261005)
    quantiles = np.quantile(distribution, [.05, .1, .5, .9, .95])
    future_days = [row['date'] for row in calendar
                   if row['is_trading_day'] and row['date'] > dates[-1].isoformat()][:HORIZON]
    write_csv('forecast_trading_dates.csv', [{'trading_day_ahead': i + 1, 'date': d}
                                           for i, d in enumerate(future_days)])
    forecast = {
        'asof': ASOF, 'origin_date': dates[-1].isoformat(), 'last_close_cny_per_t': float(prices[-1]),
        'horizon_trading_days': HORIZON, 'target_date': future_days[-1], 'adopted_model': adopted,
        'point_forecast_cny_per_t': prediction(prices, last, adopted),
        'simulated_mean_cny_per_t': float(distribution.mean()),
        'simulated_median_cny_per_t': float(quantiles[2]),
        'q05_cny_per_t': float(quantiles[0]), 'q10_cny_per_t': float(quantiles[1]),
        'q90_cny_per_t': float(quantiles[3]), 'q95_cny_per_t': float(quantiles[4]),
        'nominal_interval_coverage': .8,
        'test_interval_coverage': sum(row['inside_interval'] for row in interval_backtest) / len(test),
        'n_simulations': 100000, 'bootstrap_block_days': 5, 'return_window_days': WINDOW,
        'simulation_seed': 20261005,
    }
    write_csv('forecast_summary.csv', [forecast])
    contract = {
        'name': '研究假设：CEA综合价格20交易日现金结算合约',
        'underlying': '全国CEA现货综合收盘价；现金结算不指定实物配额年度',
        'origin_date': dates[-1].isoformat(), 'maturity': future_days[-1],
        'horizon_trading_days': HORIZON, 'tonnes_per_contract_assumption': 1000,
        'settlement': '到期当天交易所公布的CEA现货综合收盘价',
        'payoff_long': '1000 * (到期结算价 - 约定参考价)',
        'unmodeled': ['逐日盯市及其融资影响', '保证金机会成本', '手续费', '违约风险'],
        'pricing_formula': '假设期货参考价 = 真实概率模拟的到期价格均值 + 假设调整delta',
        'adjustment_meaning': 'delta汇总真实概率与定价概率及未建模项的差异，仅为敏感性参数',
        'adjustment_estimated_from_market': False,
    }
    scenarios = [{'delta_assumption_cny_per_t': delta,
                  'reference_price_cny_per_t': float(distribution.mean() + delta),
                  'notional_at_reference_cny': float((distribution.mean() + delta) * 1000),
                  'tonnes_per_contract_assumption': 1000,
                  'source': 'research_assumption_not_observed_market_quote'}
                 for delta in [-2., 0., 2.]]
    write_csv('hypothetical_contract_scenarios.csv', scenarios)

    # Source sensitivity: rerun on the 149 official 2026 observations only.
    official = [row for row in rows if row['source_type'] == 'official_daily']
    official_prices = np.array([float(row['close_cny_per_t']) for row in official])
    official_dist = simulated_terminal(official_prices, len(official_prices) - 1, 100000, 20261005)
    sensitivity = {'origin_date': official[-1]['date'], 'n_official_rows': len(official),
                   'point_persistence_cny_per_t': float(official_prices[-1]),
                   'q10_cny_per_t': float(np.quantile(official_dist, .1)),
                   'q90_cny_per_t': float(np.quantile(official_dist, .9)),
                   'comparison_note': '不同预测起点；仅显示官方数据独立运行可行，不能视为同一预测的来源差异估计'}
    write_csv('official_only_sensitivity.csv', [sensitivity])
    summary = {'calendar_expected_and_observed_rows': len(rows), 'missing_2026_trading_dates': [],
               'validation_origins': len(validation), 'purged_origins': 19, 'test_origins': len(test),
               'validation_first_origin': dates[validation[0]].isoformat(),
               'validation_last_origin': dates[validation[-1]].isoformat(),
               'last_validation_label_date': dates[validation[-1] + HORIZON].isoformat(),
               'test_first_origin': dates[test[0]].isoformat(),
               'test_last_origin': dates[test[-1]].isoformat(),
               'candidate_chosen_in_validation': chosen, 'adopted_model': adopted,
               'metrics': metrics, 'forecast': forecast, 'contract': contract,
               'scenarios': scenarios, 'official_only_sensitivity': sensitivity,
               'input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest()}
    (OUT / 'analysis_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    table = '\n'.join(f"| {MODEL_NAMES[model]} | " + ' | '.join(
        f"{next(m['mae_cny_per_t'] for m in metrics if m['model']==model and m['partition']==partition):.2f}"
        for partition in ['validation', 'test']) + ' |' for model in MODEL_NAMES)
    scenario_table = '\n'.join(f"| {s['delta_assumption_cny_per_t']:+.2f} | {s['reference_price_cny_per_t']:.2f} |"
                               for s in scenarios)
    covered = sum(row['inside_interval'] for row in interval_backtest)
    report = f'''# 国内CEA 20交易日预测与假设期货参考价

分析日期：{ASOF}。预测信息截至：{forecast['origin_date']}。到期日：{forecast['target_date']}。

## 可用数据和交易日

使用2026年1月5日至9月30日的181个收盘价；149条来自原官方日公告，32条为第三方转载。日期与[全国碳市场官方2026年休市安排]({CALENDAR_URL})完全吻合。价格单位为元/吨。只使用收盘价，未用存在口径疑问的最高、最低价。早期年份有缺口，因此未纳入本轮模型。

本轮采用综合收盘价指数作为现金结算的研究标的，不把不同配额年度的报价轮流拼接成单一实物配额价格。

## 时间顺序回测

每个起点仅使用已知的过去数据。60日收益率窗口需要61个收盘价，共建立101个带20交易日结果的样本。

- 验证期42个起点：{dates[validation[0]]}至{dates[validation[-1]]}，最晚标签在{dates[validation[-1]+HORIZON]}已知。
- 测试期40个起点：{dates[test[0]]}至{dates[test[-1]]}，对应到期结果截至9月30日。
- 中间19个起点从模型选择中剔除，防止跨越分界的20日标签泄露。
- 验证期选择候选模型，测试期用于检验候选模型。候选失败后采用价格不变基准；测试集的基准误差属于已测量结果，不是回退决策后的新增独立评估。

| 模型 | 验证期MAE（元/吨） | 测试期MAE（元/吨） |
|---|---:|---:|
{table}

候选模型为“{MODEL_NAMES[chosen]}”，测试期未优于价格不变基准。价格不变基准的测试期RMSE为{baseline_test['rmse_cny_per_t']:.2f}元/吨。40个起点的20日标签重叠，不能把它们视作40个独立月度实验，未进行独立样本显著性检验。

## 到期价格预测

价格不变点预测：**{forecast['point_forecast_cny_per_t']:.2f}元/吨**。

以最近60日对数收益率减去其样本均值，按5日循环区块重抽样，模拟20交易日累计收益率。采用零对数收益率漂移假设，模拟100000次，随机种子20261005。该分布在真实概率建模框架下构造，不是已校准的风险中性分布。

| 统计量 | 元/吨 |
|---|---:|
| 模拟均值 | {forecast['simulated_mean_cny_per_t']:.2f} |
| 模拟中位数 | {forecast['simulated_median_cny_per_t']:.2f} |
| 10%分位数 | {forecast['q10_cny_per_t']:.2f} |
| 90%分位数 | {forecast['q90_cny_per_t']:.2f} |
| 5%分位数 | {forecast['q05_cny_per_t']:.2f} |
| 95%分位数 | {forecast['q95_cny_per_t']:.2f} |

名义80%的模拟区间在测试期覆盖{covered}/40次，即{forecast['test_interval_coverage']:.1%}。实际覆盖不足，区间不具备80%的经验保证；可能受到趋势变化、尾部变化和有限样本影响。每次区间回测也只使用当时之前的60日收益率。

## 假设合约与敏感性分析

研究合约假设：起点9月30日，20交易日后于11月4日现金结算；每份1000吨；结算价为到期当天CEA综合收盘价。数量与结算设计均为项目假设，不是交易所公布的期货条款。买方到期结果为1000×（结算价−约定价）。本轮不模拟逐日盯市、保证金、手续费或违约。

参考价公式：F_hyp = E_hat_P[S_T] + delta。delta为未识别的定价调整，本轮取−2、0、+2元/吨，全部是敏感性假设，未根据国内期货成交估计。它不建立无套利复制，E_hat_P[S_T]也不能直接视为唯一公允期货价。

| 假设调整delta（元/吨） | 模拟参考价（元/吨） |
|---|---:|
{scenario_table}

模拟期货参考价的三个情景与现货预测区间含义不同：前者描述定价假设，后者描述统计模型下的到期现货不确定性。

## 文件和复现

`samples_20d.csv`保留每个起点、信息窗口、目标日期、来源和预测值；`model_backtest_metrics.csv`与`interval_backtest.csv`保存回测结果；`forecast_summary.csv`和`hypothetical_contract_scenarios.csv`保存预测与假设报价。`trading_calendar_2026.csv`保存全年交易日及来源。

`official_only_sensitivity.csv`显示只用149条官方2026日数据、从8月14日出发的独立计算。因预测起点不同，不把它当成同一时点的来源敏感性对照。

32条近期价格仍是第三方转载；其中56个历史重叠收盘价与官方数据一致，月报汇总核对保留9月挂牌成交量多20吨的差异。本模型不使用成交量，故该差异不进入计算，但来源的不确定性仍然存在。回测假设每个交易日收盘后可获得当日收盘价；转载网页逐日实际发布时间未核实，本轮未证明可以按转载发布时间实时执行。未见上市国内碳期货，无法检验模拟参考价与真实期货成交的偏离。

复现：在工作目录用具有numpy的Python运行`run_cea_20d_analysis.py`。输入不被修改。模型改进应使用下一段新数据进行检验，避免继续用本次测试期调参后仍称其独立测试。
'''
    (OUT / '研究报告.md').write_text(report, encoding='utf-8')
    print(json.dumps({'forecast': forecast, 'selected_candidate': chosen,
                      'test_mae_baseline': baseline_test['mae_cny_per_t'],
                      'test_mae_candidate': candidate_test['mae_cny_per_t'],
                      'scenarios': scenarios}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
