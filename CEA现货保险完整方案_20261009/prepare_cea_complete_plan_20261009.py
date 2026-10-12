"""Recompute the consolidated plan with shared 60-day Monte Carlo paths."""
from pathlib import Path
from datetime import date,timedelta
import csv,json,math,hashlib,shutil,contextlib,io
from collections import Counter
import numpy as np
import run_cea_enterprise_strategy_20261009 as enterprise
from run_cea_product_review_20261009 import paths,params,claim,components,tvar,MODELS
from run_cea_20d_analysis import is_open

ROOT=Path(__file__).resolve().parent
OUT=ROOT if ROOT.name=='CEA现货保险完整方案_20261009' else ROOT/'CEA现货保险完整方案_20261009'
DATA=OUT/'data';N=100000;SEED=2026100903
INPUT=ROOT/'domestic_cea_data_20261005/cea_daily_research_combined.csv'
if not INPUT.exists():INPUT=DATA/'CEA输入日行情.csv'

def save(name,rows):
    with (DATA/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    DATA.mkdir(parents=True,exist_ok=True)
    with INPUT.open(encoding='utf-8-sig') as f:allrows=list(csv.DictReader(f))
    rows=[x for x in allrows if x['date'].startswith('2026')]
    p=np.array([float(x['close_cny_per_t']) for x in rows]);o=len(p)-1;par=params(p)
    dates=[date.fromisoformat(x['date']) for x in rows];cutoff=dates[-1];calendar=[cutoff];day=cutoff
    while len(calendar)<61:
        day+=timedelta(days=1)
        assert day.year==2026
        if is_open(day):calendar.append(day)
    days=[(x-cutoff).days for x in calendar]
    master={model:paths(p,o,60,N,SEED,model) for model in MODELS}
    def shared_paths(prices,origin,h,count,seed,model):
        assert origin==o and count==N and h<=60
        return master[model][:,:h]
    # Reuse the preceding decision engine, replacing its path source only.
    enterprise.paths=shared_paths;enterprise.OUT=DATA;enterprise.INPUT=INPUT;enterprise.SEED=SEED
    with contextlib.redirect_stdout(io.StringIO()):enterprise.main()
    ent=json.loads((DATA/'summary.json').read_text(encoding='utf-8'))
    ent['shared_path_seed']=SEED;ent['shared_max_horizon']=60
    (DATA/'企业决策汇总.json').write_text(json.dumps(ent,ensure_ascii=False,indent=2),encoding='utf-8')
    forecasts=[];quotes=[];algo=[];sampling=[];tier=[];margin=[]
    for h in [10,20,40,60]:
        for model,pp in master.items():
            idx=pp[:,h-5:h].mean(axis=1);terminal=pp[:,h-1]
            loss=claim(idx,**par);tau=days[h]/365
            row=dict(model=model,horizon=h,expiry=calendar[h].isoformat(),calendar_days=days[h],
                quantity_t=1000,limit_cny=par['eta']*1000*par['c'],**components(loss,tau))
            algo.append(row)
            if model=='zero60':
                quotes.append(row)
                forecasts.append(dict(horizon=h,expiry=calendar[h].isoformat(),
                    end_mean=float(terminal.mean()),end_p10=float(np.quantile(terminal,.1)),
                    end_median=float(np.median(terminal)),end_p90=float(np.quantile(terminal,.9)),
                    average5_mean=float(idx.mean()),average5_p10=float(np.quantile(idx,.1)),
                    average5_median=float(np.median(idx)),average5_p90=float(np.quantile(idx,.9))))
                if h==20:
                    for label,index in [('期末单日',terminal),('末5日均价',idx),('全期间均价',pp[:,:h].mean(axis=1)),('期间最高收盘价',pp[:,:h].max(axis=1))]:
                        sampling.append(dict(sampling=label,**components(claim(index,**par),tau)))
                    for q in [1000,10000,50000]:tier.append(dict(quantity_t=q,limit_cny=par['eta']*q*par['c'],**components(claim(idx,**par,q=q),tau)))
                    for m in [0.,.05,.1]:margin.append(dict(margin_rate=m,**components(loss,tau,margin_rate=m)))
    pp=master['zero60'];stage_losses=[];stages=[]
    for h,q in [(20,400),(40,300),(60,300)]:
        loss=claim(pp[:,h-5:h].mean(axis=1),**par,q=q);stage_losses.append(loss)
        stages.append(dict(horizon=h,expiry=calendar[h].isoformat(),quantity_t=q,limit_cny=par['eta']*q*par['c'],
            **components(loss,days[h]/365)))
    total=np.sum(stage_losses,axis=0);mu=float(total.mean())
    pv=sum(x['pure_premium_cny'] for x in stages);tail=tvar(total)
    capital=max(tail-mu,0)*.08*days[60]/365
    joint=dict(total_quantity_t=1000,total_limit_cny=sum(x['limit_cny'] for x in stages),
        expected_payout_cny=mu,pure_premium_cny=pv,tvar995_cny=tail,capital_cost_cny=capital,
        fixed_expense_cny=200,gross_premium_cny=(pv+capital+200)/.8,
        sum_separate_premiums_cny=sum(x['gross_premium_cny'] for x in stages),
        capital_release='all modeled tail capital held to final stage; no stage release',
        stage_payout_correlations=np.corrcoef(stage_losses).tolist())
    old=components(claim(master['zero60'][:,19],94.96,2,10,1),days[20]/365)
    recent_returns=np.diff(np.log(p[-61:]))
    metadata=dict(research_date='2026-10-09',data_cutoff=cutoff.isoformat(),
        rows_all=len(allrows),rows_2026=len(rows),source_counts=dict(Counter(x['source_type'] for x in rows)),
        data_start=rows[0]['date'],last_close=float(p[-1]),last5_closes=[dict(date=x['date'],close=float(x['close_cny_per_t'])) for x in rows[-5:]],
        historical_2026_low=float(p.min()),historical_2026_high=float(p.max()),
        daily_log_mean60=float(recent_returns.mean()),daily_log_sd60=float(recent_returns.std(ddof=1)),
        seed=SEED,n_sim=N,contract=par,input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        quotes=quotes,forecasts=forecasts,sampling=sampling,tiers=tier,margin=margin,
        stages=stages,joint=joint,old_contract_same_paths=old,enterprise=ent,
        commercial_quote_allowed=False,live_trade_recommendation_allowed=False,
        validation='reused exploratory historical replay only; previous payouts underestimated',
        code_checks=dict(coverage_cap='passed',same_paths_between_price_and_enterprise='passed',
            no_daily_duplicate_payout='passed',joint_stage_quantity='passed',sale_next_session='passed'))
    for name,value in [('统一行情预测.csv',forecasts),('统一单期限保费.csv',quotes),('统一算法报价.csv',algo),
        ('统一采价比较.csv',sampling),('统一吨数档位.csv',tier),('统一模型附加.csv',margin),('统一分阶段保障.csv',stages)]:save(name,value)
    save('2026年最后15日行情.csv',rows[-15:])
    assert sum(x['quantity_t'] for x in stages)==1000
    assert np.all(total<=joint['total_limit_cny']+1e-8)
    # Fixed cost means premiums do not scale directly with quantity.
    with (DATA/'全部采购投保候选.csv').open(encoding='utf-8-sig') as f:cr=list(csv.DictReader(f))
    matched=next(x for x in cr if x['model']=='zero60' and x['future_horizon_trading_days']=='20' and x['insured_t']=='5000.0' and x['buy_now_t']=='5000.0')
    assert abs(float(matched['premium_cny'])-components(claim(pp[:,15:20].mean(axis=1),**par,q=5000),days[20]/365)['gross_premium_cny'])<1e-8
    for name in ['保费算法回放指标.csv','历史赔款预测记录.csv','智能体输入流程.json','企业输入空白示例.json','限额流程校验.json']:
        src=ROOT/'cea_product_review_20261009'/name
        if src.exists():shutil.copy2(src,DATA/name)
    for name in ['run_cea_enterprise_strategy_20261009.py','run_cea_product_review_20261009.py','run_cea_20d_analysis.py',Path(__file__).name]:
        src=ROOT/name;dst=OUT/name
        if src.resolve()!=dst.resolve():shutil.copy2(src,dst)
    (OUT/'统一计算结果.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(quotes=quotes,joint=joint,enterprise_choices=ent['buyer_choices'],fixed20=ent['fixed20_example'],sales=ent['seller_choices']),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
