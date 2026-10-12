"""Research demonstration of procurement, partial insurance and surplus sales.

All enterprise quantities, budgets and execution costs are illustrative.
Uses frozen 2026-09-30 inputs and the unvalidated preceding insurance model.
"""
from pathlib import Path
from datetime import date,timedelta
import csv,json,math,hashlib,shutil
import numpy as np
from run_cea_product_review_20261009 import paths,params,claim,components,tvar,MODELS,MODEL_NAMES
from run_cea_20d_analysis import is_open

ROOT=Path(__file__).resolve().parent
OUT=ROOT if ROOT.name=='cea_enterprise_strategy_20261009' else ROOT/'cea_enterprise_strategy_20261009'
INPUT=ROOT/'domestic_cea_data_20261005/cea_daily_research_combined.csv'
if not INPUT.exists():INPUT=ROOT/'CEA输入日行情.csv'
M=100000;SEED=2026100902

def save(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def free_surplus(available,reserve,restricted=0):
    # Reserve includes verified obligations and the enterprise's safety buffer.
    for x in [available,reserve,restricted]:
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0:
            raise ValueError('Invalid quantity')
    return max(available-restricted-reserve,0.)

def sale_path(pp,s0,days,kind,threshold=None):
    """Close-based trigger executes NEXT session; never use a future maximum.

    Deadline sale is committed before that session. Execution at a daily close
    is a research proxy and does not assure a live fill at the published close.
    """
    n,h=pp.shape
    if kind=='now':idx=np.zeros(n,dtype=int)
    elif kind=='day10':idx=np.full(n,10,dtype=int)
    elif kind=='day20':idx=np.full(n,20,dtype=int)
    else:
        known=np.column_stack([np.full(n,s0),pp[:,:h-1]])
        crossed=known>=threshold
        any_hit=crossed.any(axis=1)
        # Column 0 is known day0. A day19 observation can execute on day20.
        idx=np.where(any_hit,crossed.argmax(axis=1)+1,h)
    panel=np.column_stack([np.full(n,s0),pp])
    executed=panel[np.arange(n),idx]
    return executed,idx,np.array(days)[idx]

def main():
    OUT.mkdir(exist_ok=True)
    with INPUT.open(encoding='utf-8-sig') as f:raw=[x for x in csv.DictReader(f) if x['date'].startswith('2026')]
    prices=np.array([float(x['close_cny_per_t']) for x in raw]);origin=len(prices)-1
    assert len(prices)==181
    cutoff=date.fromisoformat(raw[-1]['date']);dt=[cutoff];d=cutoff
    while len(dt)<21:
        d+=timedelta(days=1)
        if is_open(d):dt.append(d)
    days=[(x-cutoff).days for x in dt];s0=float(prices[-1]);par=params(prices)
    sims={m:paths(prices,origin,20,M,SEED,m) for m in MODELS}
    # Synthetic buyer: 10,000 t unpriced need. No surplus in this profile.
    gap=10000.;cash0=550000.;premium_budget=3500.;fund=.06;fee=.001;capacity_t=10000.
    plans=[];priced={};counter=0
    for h in [10,20]:
        index=sims['zero60'][:,:h][:,-5:].mean(axis=1)
        for buy_fraction in [0.,.25,.5,.75,1.]:
            b0=gap*buy_fraction;b_later=gap-b0
            for alpha in [0.,.25,.5,.75,1.]:
                if b_later==0 and alpha!=0:continue
                q_ins=alpha*b_later
                premium=components(claim(index,**par,q=q_ins),days[h]/365)['gross_premium_cny'] if q_ins else 0.
                initial_cash=b0*s0*(1+fee)+premium
                feasible=initial_cash<=cash0 and premium<=premium_budget and q_ins<=capacity_t
                plan_id=f'P{counter:02d}';counter+=1
                term_cost=[];plan_rows=[]
                for model,pp in sims.items():
                    x=pp[:,:h][:,-5:].mean(axis=1)
                    payout=claim(x,**par,q=q_ins)
                    # All alternatives measured at day20, avoid mixed time values.
                    c=(initial_cash*math.exp(fund*days[20]/365)
                       +(b_later*x*(1+fee)-payout)*math.exp(fund*(days[20]-days[h])/365))
                    assert np.all(payout<=par['eta']*q_ins*par['c']+1e-6)
                    row=dict(plan_id=plan_id,model=model,purchase_window_end=dt[h].isoformat(),
                        future_horizon_trading_days=h,buy_now_t=b0,buy_later_t=b_later,
                        insured_t=q_ins,insured_fraction_of_later_purchase=alpha,insured_fraction_of_total_gap=q_ins/gap,
                        premium_cny=premium,initial_cash_cny=initial_cash,feasible=feasible,
                        mean_net_cost_cny=float(c.mean()),q95_net_cost_cny=float(np.quantile(c,.95)),
                        cvar95_net_cost_cny=tvar(c,.95),mean_payout_cny=float(payout.mean()),
                        probability_net_cost_above_one_million=float((c>1000000).mean()))
                    plan_rows.append(row);term_cost.append(c)
                plans.extend(plan_rows)
                priced[plan_id]=plan_rows
    choices=[]
    for label,lam in [('期望成本优先',0.),('适度控制尾部成本',.15),('尾部成本优先',.5)]:
        scores=[]
        for pid,rr in priced.items():
            if not rr[0]['feasible']:continue
            vals=[r['mean_net_cost_cny']+lam*(r['cvar95_net_cost_cny']-r['mean_net_cost_cny']) for r in rr]
            scores.append((max(vals),pid))
        score,pid=min(scores)
        chosen=priced[pid][0]
        choices.append(dict(profile=label,tail_risk_weight=lam,robust_objective_cny=score,
            **{k:v for k,v in chosen.items() if k not in ['model','feasible']}))
    # Conditional example where enterprise procurement date cannot move earlier.
    restricted=[]
    for pid,rr in priced.items():
        if rr[0]['feasible'] and rr[0]['future_horizon_trading_days']==20:
            restricted.append((max(r['mean_net_cost_cny']+.5*(r['cvar95_net_cost_cny']-r['mean_net_cost_cny']) for r in rr),pid))
    score,pid=min(restricted)
    fixed20=dict(profile='采购日期固定20日且控制尾部成本',tail_risk_weight=.5,robust_objective_cny=score,
        **{k:v for k,v in priced[pid][0].items() if k not in ['model','feasible']})
    # Gross-price sensitivities are assumptions, not inferred calibration factors.
    sensitivity=[]
    for multiplier in [1.,1.5,2.]:
        candidates=[]
        for pid,rr in priced.items():
            base=rr[0];additional=base['premium_cny']*(multiplier-1)
            if base['initial_cash_cny']+additional>cash0 or base['premium_cny']*multiplier>premium_budget:continue
            increment=additional*math.exp(fund*days[20]/365)
            objective=max(r['mean_net_cost_cny']+.5*(r['cvar95_net_cost_cny']-r['mean_net_cost_cny'])+increment for r in rr)
            candidates.append((objective,pid))
        objective,pid=min(candidates);base=priced[pid][0]
        sensitivity.append(dict(gross_premium_multiplier_assumption=multiplier,plan_id=pid,
            buy_now_t=base['buy_now_t'],buy_later_t=base['buy_later_t'],insured_t=base['insured_t'],
            future_horizon_trading_days=base['future_horizon_trading_days'],
            adjusted_premium_cny=base['premium_cny']*multiplier,robust_objective_cny=objective))
    # Synthetic seller, separate from buyer: 2,000 t verified freely tradable surplus.
    seller_available=22000.;seller_reserve=20000.;surplus=free_surplus(seller_available,seller_reserve)
    policy_defs=[('立即出售','now',None),('第10交易日出售','day10',None),('第20交易日出售','day20',None),
        ('收盘达98后次交易日出售，最迟第20日','trigger',98.),
        ('收盘达100后次交易日出售，最迟第20日','trigger',100.),
        ('收盘达105后次交易日出售，最迟第20日','trigger',105.)]
    sales=[]
    for label,kind,threshold in policy_defs:
        for model,pp in sims.items():
            execution,when,elapsed=sale_path(pp,s0,days,kind,threshold)
            rev=surplus*execution*(1-fee)*np.exp(-fund*elapsed/365)
            immediate=surplus*s0*(1-fee)
            sales.append(dict(policy=label,model=model,sellable_t=surplus,
                mean_pv_net_revenue_cny=float(rev.mean()),q05_pv_net_revenue_cny=float(np.quantile(rev,.05)),
                lower_cvar95_revenue_cny=-tvar(-rev,.95),
                mean_advantage_vs_now_cny=float(rev.mean()-immediate),
                probability_underperform_now=float((rev<immediate).mean()),
                mean_sale_trading_day=float(when.mean())))
    salechoices=[]
    for label,lam in [('期望回款优先',0.),('控制回款下行风险',.5)]:
        candidates=[]
        for policy,_,_ in policy_defs:
            rr=[r for r in sales if r['policy']==policy]
            scores=[r['mean_pv_net_revenue_cny']-lam*(r['mean_pv_net_revenue_cny']-r['lower_cvar95_revenue_cny']) for r in rr]
            candidates.append((min(scores),policy))
        score,policy=max(candidates)
        salechoices.append(dict(profile=label,risk_weight=lam,policy=policy,robust_score_cny=score))
    # Counterfactual stress cases: no use of future realized information in selection.
    stress=[]
    selected_ids=set(x['plan_id'] for x in choices)|{fixed20['plan_id']}
    for pid in sorted(selected_ids):
        r=priced[pid][0];h=r['future_horizon_trading_days']
        for x in [85.,90.,95.,100.,105.,115.]:
            payout=float(claim(x,**par,q=r['insured_t']))
            c=r['initial_cash_cny']*math.exp(fund*days[20]/365)+(r['buy_later_t']*x*(1+fee)-payout)*math.exp(fund*(days[20]-days[h])/365)
            stress.append(dict(plan_id=pid,procurement_index_cny_per_t=x,payout_cny=payout,
                terminal_net_cost_cny=c))
    testpp=np.array([[95.,99.,94.,96.],[94.,94.,94.,94.]])
    execution,when,_=sale_path(testpp,94.,[0,1,2,3,4],'trigger',98.)
    assert when.tolist()==[3,4] and execution.tolist()==[94.,94.]
    assert free_surplus(22000,20000)==2000 and free_surplus(19000,20000)==0
    assert all(x['insured_t']<=x['buy_later_t'] for x in choices)
    save('全部采购投保候选.csv',plans);save('演示采购方案.csv',choices+[fixed20]);save('保费变动对方案的影响.csv',sensitivity)
    save('富余配额出售比较.csv',sales);save('演示出售方案.csv',salechoices);save('采购成本压力情景.csv',stress)
    config=dict(research_date='2026-10-09',data_cutoff=cutoff.isoformat(),real_enterprise_data_available=False,
        live_recommendation_allowed=False,calibration_status='previous exploratory replay underestimates payouts; independent validation pending',
        scenarios=M,contract=par,buyer_assumptions=dict(unpriced_gap_t=gap,initial_cash_budget_cny=cash0,
            premium_budget_cny=premium_budget,annual_funding_cost=fund,one_way_fee_rate=fee,
            assumed_capacity_t=capacity_t,all_remaining_purchase_executed_in_last_five_days=True,
            procurement_index_equals_execution_price=True),
        seller_assumptions=dict(available_t=seller_available,reserved_for_compliance_and_buffer_t=seller_reserve,
            verified_sellable_surplus_t=surplus),buyer_choices=choices,fixed20_example=fixed20,
        gross_premium_sensitivity=sensitivity,seller_choices=salechoices,
        actual_inputs_needed=['企业履约需求与不确定性','可交易库存及受限库存','采购交付与阶段数量',
            '预留履约量和安全缓冲','现金预算及融资成本','保费预算','采购价与指数价差','卖出手续费与滑点',
            '富余配额未来用途及可结转条件','企业风险偏好','保险人条款、容量及价格'],
        boundaries=['已提前买入或锁价的吨数不再分配给未来采购保险',
            '现有上涨保障不覆盖富余配额的卖价下跌',
            '实际出售或采购变更后需重核被保险风险与数量，不能机械保留原赔款',
            '买方与卖方是两种独立演示企业，不能相加解释为无风险套利',
            '未模拟订单簿、滑点、需求误差或年份可用性；研究结果不构成可执行交易指令'],
        verification=dict(next_session_trigger='passed',surplus_reserve='passed',partial_coverage_quantity='passed',
            independent_strategy_validation='not_completed'))
    (OUT/'summary.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    for src,dst in [(Path(__file__),OUT/Path(__file__).name),
        (ROOT/'run_cea_product_review_20261009.py',OUT/'run_cea_product_review_20261009.py'),
        (ROOT/'run_cea_20d_analysis.py',OUT/'run_cea_20d_analysis.py'),(INPUT,OUT/'CEA输入日行情.csv')]:
        if src.resolve()!=dst.resolve():shutil.copy2(src,dst)
    print(json.dumps(dict(buyer_choices=choices,seller_choices=salechoices,baseline_sales=[r for r in sales if r['model']=='zero60']),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
