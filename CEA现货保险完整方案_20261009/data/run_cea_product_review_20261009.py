"""Contract limits, agent inputs, price standards and prospective-loss replay."""
from pathlib import Path
from datetime import date, timedelta
import csv, json, math, hashlib, shutil, zipfile
import numpy as np
from run_cea_20d_analysis import is_open

ROOT=Path(__file__).resolve().parent
OUT=ROOT if ROOT.name=='cea_product_review_20261009' else ROOT/'cea_product_review_20261009'
INPUT=ROOT/'domestic_cea_data_20261005/cea_daily_research_combined.csv'
if not INPUT.exists():INPUT=ROOT/'CEA输入日行情.csv'
MODELS=['zero60','shrink_drift25','recent_vol_scale']
MODEL_NAMES={'zero60':'60日零漂移区块基准','shrink_drift25':'25%历史漂移收缩','recent_vol_scale':'近期波动调整'}
N=100000;BACKTEST_N=20000;SEED=20261009

def save(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def tvar(loss,alpha=.995):
    # Exact upper probability mass; supports a fractional boundary sample.
    x=np.sort(loss);mass=(1-alpha)*len(x);whole=int(math.floor(mass+1e-10));part=mass-whole
    total=float(x[-whole:].sum()) if whole else 0.
    if part>1e-10:total+=part*float(x[-whole-1])
    return total/mass

def params(prices):
    k=float(np.mean(prices[-5:]));d=max(2.,.02*k);c=.05*k
    return dict(k=k,d=d,c=c,eta=.8)

def paths(prices,origin,h,count,seed,model):
    r=np.diff(np.log(prices[origin-60:origin+1]));mean=float(r.mean());z=r-mean
    ratio=float(np.std(r[-20:],ddof=1)/np.std(r,ddof=1)) if np.std(r,ddof=1)>0 else 1.
    scale=float(np.clip(ratio,.5,2)) if model=='recent_vol_scale' else 1.
    drift=.25*mean if model=='shrink_drift25' else 0.
    j=np.random.default_rng(seed).integers(0,60,size=(count,math.ceil(h/5)))
    ret=z[(j[:,:,None]+np.arange(5))%60].reshape(count,-1)[:,:h]*scale+drift
    return prices[origin]*np.exp(np.cumsum(ret,axis=1))

def claim(index,k,d,c,eta,q=1000.):
    return eta*q*np.clip(index-k-d,0,c)

def components(loss,tau,fixed=200,expense=.15,profit=.05,margin_rate=0.):
    mean=float(loss.mean());tail=tvar(loss);pv=math.exp(-.02*tau)*mean
    cap=max(0,tail-mean)*.08*tau;model_margin=margin_rate*pv
    premium=(pv+cap+fixed+model_margin)/(1-expense-profit)
    return dict(expected_payout_cny=mean,pure_premium_cny=pv,tvar995_cny=tail,capital_cost_cny=cap,
      fixed_expense_cny=fixed,model_margin_rate_assumption=margin_rate,model_margin_cny=model_margin,
      gross_premium_cny=premium,probability_claim=float((loss>0).mean()),
      mc_se_expected_payout_cny=float(loss.std(ddof=1)/math.sqrt(len(loss))))

def limit_assessment(data,approved_capacity_t=None):
    fields=['industry','compliance_year','required_allowance_t','eligible_resources_t',
      'locked_future_t_not_in_resources','reserved_unpriced_t','stage_requested_t','stage_purchase_date']
    missing=[k for k in fields if data.get(k) is None or data.get(k)=='']
    if missing:return dict(status='needs_enterprise_inputs',missing=missing,insured_t=None,limit_cny=None)
    if not isinstance(data['industry'],str) or not data['industry'].strip():
        return dict(status='invalid_input',invalid_field='industry',insured_t=None,limit_cny=None)
    if isinstance(data['compliance_year'],bool) or not isinstance(data['compliance_year'],int) or not 1900<=data['compliance_year']<=2100:
        return dict(status='invalid_input',invalid_field='compliance_year',insured_t=None,limit_cny=None)
    try:
        date.fromisoformat(data['stage_purchase_date'])
    except (ValueError,TypeError):
        return dict(status='invalid_input',invalid_field='stage_purchase_date',insured_t=None,limit_cny=None)
    for key in fields[2:7]:
        value=data[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            return dict(status='invalid_input',invalid_field=key,insured_t=None,limit_cny=None)
    net=max(0.,data['required_allowance_t']-data['eligible_resources_t']-
              data['locked_future_t_not_in_resources']-data['reserved_unpriced_t'])
    if net==0:
        return dict(status='no_remaining_unpriced_need',remaining_net_t=0,insured_t=None,limit_cny=None)
    if data['stage_requested_t']==0:
        return dict(status='no_stage_quantity_requested',remaining_net_t=net,insured_t=None,limit_cny=None)
    if data['stage_requested_t']>net:
        return dict(status='requested_t_exceeds_remaining_net_need',remaining_net_t=net,
          requested_t=data['stage_requested_t'],insured_t=None,limit_cny=None)
    if approved_capacity_t is None:
        return dict(status='needs_insurer_capacity_review',remaining_net_t=net,
          requested_t=data['stage_requested_t'],insured_t=None,limit_cny=None)
    if isinstance(approved_capacity_t,bool) or not isinstance(approved_capacity_t,(int,float)) or not math.isfinite(approved_capacity_t) or approved_capacity_t<=0:
        return dict(status='invalid_insurer_capacity',insured_t=None,limit_cny=None)
    q=min(net,data['stage_requested_t'],approved_capacity_t)
    return dict(status='provisional_quantity_only',remaining_net_t=net,insured_t=q,
      limit_cny=None,evidence_status='enterprise disclosures and overlapping coverage require verification')

def main():
    OUT.mkdir(exist_ok=True)
    with INPUT.open(encoding='utf-8-sig') as f:rows=[r for r in csv.DictReader(f) if r['date'].startswith('2026')]
    dates=[date.fromisoformat(r['date']) for r in rows];p=np.array([float(r['close_cny_per_t']) for r in rows])
    assert len(p)==181 and np.all(p>0)
    origin=len(p)-1;contract=params(p);k,d,c,eta=[contract[key] for key in ['k','d','c','eta']]
    # Loss replay: same prior partition, 19 purged origins, all labels mature.
    # These dates were already viewed in earlier work: exploratory, not new holdout.
    validation=range(60,102);test=range(121,161);records=[]
    for part,origins in [('validation_replay',validation),('test_replay',test)]:
        for o in origins:
            par=params(p[:o+1]);actual_index=float(p[o+16:o+21].mean())
            actual=float(claim(actual_index,**par))
            for model in MODELS:
                sim=paths(p,o,20,BACKTEST_N,SEED+o,model)
                loss=claim(sim[:,-5:].mean(axis=1),**par)
                records.append(dict(partition=part,model=model,origin=dates[o].isoformat(),
                  label_date=dates[o+20].isoformat(),budget_k=par['k'],deductible_d=par['d'],cap_band_c=par['c'],
                  actual_index=actual_index,actual_payout_cny=actual,predicted_mean_payout_cny=float(loss.mean()),
                  predicted_claim_probability=float((loss>0).mean())))
    metrics=[]
    for part in ['validation_replay','test_replay']:
        for model in MODELS:
            rr=[r for r in records if r['partition']==part and r['model']==model]
            actual=np.array([r['actual_payout_cny'] for r in rr]);pred=np.array([r['predicted_mean_payout_cny'] for r in rr])
            prob=np.array([r['predicted_claim_probability'] for r in rr]);err=pred-actual
            metrics.append(dict(partition=part,model=model,n=len(rr),
              rmse_mean_payout_cny=float(np.sqrt(np.mean(err**2))),mae_mean_payout_cny=float(np.abs(err).mean()),
              mean_bias_cny=float(err.mean()),brier_claim_probability=float(np.mean((prob-(actual>0))**2)),
              expected_payout_sum_cny=float(pred.sum()),actual_payout_sum_cny=float(actual.sum()),
              actual_over_expected=float(actual.sum()/pred.sum()) if pred.sum()>0 else None))
    validation_selected=min([r for r in metrics if r['partition']=='validation_replay'],key=lambda x:x['rmse_mean_payout_cny'])['model']
    # Do not auto-deploy a model using a historically reused test sample.
    deployed='zero60'
    trading=[];day=dates[-1]
    while len(trading)<60:
        day+=timedelta(days=1)
        assert day.year==2026
        if is_open(day):trading.append(day)
    quotes=[];algorithm=[];sampling=[];margin=[];packs=[]
    for h in [10,20,40,60]:
        tau=(trading[h-1]-dates[-1]).days/365
        for model in MODELS:
            pp=paths(p,origin,h,N,SEED+h,model)
            index=pp[:,-5:].mean(axis=1);loss=claim(index,**contract)
            row=dict(model=model,horizon_trading_days=h,expiry=trading[h-1].isoformat(),
              budget_k=k,deductible_d=d,cap_band_c=c,participation=eta,insured_t=1000,limit_cny=eta*1000*c,
              **components(loss,tau))
            algorithm.append(row)
            if model==deployed:
                quotes.append(row)
                if h==20:
                    for kind,idx in [('期末单日',pp[:,-1]),('末5日均价',index),('全期间均价',pp.mean(axis=1)),('期间最高收盘价',pp.max(axis=1))]:
                        sampling.append(dict(sampling=kind,**components(claim(idx,**contract),tau)))
                    for m in [0.,.05,.10]:margin.append(dict(model_margin_rate=m,**components(loss,tau,margin_rate=m)))
                    for q in [1000.,10000.,50000.]:
                        packs.append(dict(demo_tier_t=q,coverage_limit_cny=eta*q*c,
                          status='illustration_not_approved_capacity',**components(loss*q/1000,tau)))
    agent=dict(product='短期CEA现货采购价格保障',collection_order=[
      dict(step=1,fields=['industry','compliance_year'],question='企业属于哪个行业，本次对应哪一履约年度？'),
      dict(step=2,fields=['required_allowance_t','eligible_resources_t'],question='预计需清缴多少吨，已有多少吨可用于该年度履约？可用量需避免获配与库存重复计算。'),
      dict(step=3,fields=['locked_future_t_not_in_resources','reserved_unpriced_t'],question='还有多少未计入可用量的锁价采购？尚未锁价的需求中，有多少吨已分配给其他有效保险或待确认申请？同一吨只能扣除一次。'),
      dict(step=4,fields=['stage_requested_t','stage_purchase_date'],question='本次计划保障多少吨，采购或采价截止日期是什么？'),
      dict(step=5,fields=['budget_price_choice','participation','deductible_choice','cap_band_choice'],question='选择预算价格、免赔、赔付比例及每吨最高保障价差。')],
      default_contract=dict(budget='研究起点最近5个已知交易日收盘均价',deductible='max(2元/吨,2%预算价)',
        cap_band='5%预算价',participation=.8,sampling='每阶段末5个交易日均价'),
      capacity_policy='行业用于核算与核验分组，不以行业均值推定企业额度；容量参数来自保险人授权',
      missing_input_policy='missing or blank remains unknown; never convert to zero or auto-approve',
      over_request_policy='ask to reduce stage quantity or reverify annual need; do not silently clip',
      evidence_fields=['enterprise_id','information_as_of','evidence_ids','verification_status'],
      field_semantics=dict(eligible_resources_t='截至信息时点可用于目标履约年度的资源；获配量与账户库存去重，仅纳入合法可用且经核验的资源',
        locked_future_t_not_in_resources='采购价格已确定、资源未计入可用量且可按期交付的采购数量',
        reserved_unpriced_t='尚未锁价需求中已分配给其他阶段、有效保险或待确认申请的数量；去重，不与库存或锁价采购重复扣减'),
      ledger_requirements=['同一企业与履约年度统一数量台账','待确认申请也预占额度','分阶段数量之和不超过未锁价净需求','确认时原子预占，取消时释放，状态变化后重新核验'],
      final_quote_requirements=['企业资料及净需求核验','最新已核验CEA行情','保险人容量批准','条款与费用参数确认','独立赔款校准通过'])
    demo=dict(industry='发电',compliance_year=2025,required_allowance_t=1000000,
      eligible_resources_t=980000,locked_future_t_not_in_resources=10000,reserved_unpriced_t=0,
      stage_requested_t=10000,stage_purchase_date='2026-11-04')
    audit=dict(empty=limit_assessment({}),missing_capacity=limit_assessment(demo),
      complete_demo=limit_assessment(demo,approved_capacity_t=10000),
      excess_request=limit_assessment(demo|dict(stage_requested_t=20000),10000),
      negative_quantity=limit_assessment(demo|dict(stage_requested_t=-1),10000),
      invalid_date=limit_assessment(demo|dict(stage_purchase_date='2026-02-30'),10000),
      exhausted_need=limit_assessment(demo|dict(reserved_unpriced_t=10000),10000),
      reduced_capacity=limit_assessment(demo,5000),
      invalid_capacity=limit_assessment(demo,True),
      unknown_resources=limit_assessment(demo|dict(eligible_resources_t=None),10000))
    assert audit['empty']['insured_t'] is None and audit['complete_demo']['insured_t']==10000
    assert audit['excess_request']['insured_t'] is None and audit['negative_quantity']['status']=='invalid_input'
    assert audit['invalid_date']['status']=='invalid_input' and audit['exhausted_need']['insured_t'] is None
    assert audit['reduced_capacity']['insured_t']==5000 and audit['invalid_capacity']['insured_t'] is None
    assert audit['unknown_resources']['status']=='needs_enterprise_inputs'
    boundary=np.array([k+d-1,k+d,k+d+1,k+d+c,k+d+c+10])
    assert np.allclose(claim(boundary,**contract),[0,0,eta*1000,eta*1000*c,eta*1000*c])
    # Confirm the old 20d case with the original seed and original contract.
    old=paths(p,origin,20,N,20261005,'zero60')
    oldloss=claim(old[:,-1],94.96,2,10,1)
    assert abs(components(oldloss,35/365)['gross_premium_cny']-1605.6095943585708)<1e-8
    assert min(r['gross_premium_cny'] for r in quotes)>0
    save('历史赔款预测记录.csv',records);save('保费算法回放指标.csv',metrics)
    save('候选算法同条款保费.csv',algorithm);save('修订条款多期限保费.csv',quotes)
    save('采价方式比较.csv',sampling);save('模型附加敏感性.csv',margin);save('示例吨数档位.csv',packs)
    (OUT/'智能体输入流程.json').write_text(json.dumps(agent,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'限额流程校验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    summary=dict(research_date='2026-10-09',data_cutoff=dates[-1].isoformat(),input_rows=181,
      input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),contract=contract,
      validation_selected=validation_selected,deployed_model=deployed,
      deployment_status='research baseline only; commercial quotation blocked by missing inputs and unvalidated calibration',
      quote_gate=dict(commercial_quote_allowed=False,enterprise_inputs_complete=False,
        latest_market_data_verified=False,insurer_capacity_authorized=False,independent_payout_validation_passed=False,
        blocked_reasons=['没有真实企业输入','行情仅核验至2026-09-30','缺少保险人容量及费用参数','历史回放出现赔款低估，尚无新的独立验证']),
      metrics=metrics,quotes=quotes,sampling=sampling,margin_sensitivity=margin,
      pure_premium_model_margin='5% and 10% are assumptions, not calibrated loadings',
      capacity_status='no real enterprise inputs or insurer capital authorization',
      scopes=['short_term_CEA_spot_insurance'],future_work=['fresh data verification','actual enterprise input validation','independent claims backtest','portfolio liquidity and capital release'])
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    for source,target in [(Path(__file__),OUT/Path(__file__).name),(INPUT,OUT/'CEA输入日行情.csv'),
       (ROOT/'run_cea_20d_analysis.py',OUT/'run_cea_20d_analysis.py')]:
        if source.resolve()!=target.resolve():shutil.copy2(source,target)
    (OUT/'verification.json').write_text(json.dumps(dict(agent_missing_input_behavior='passed',over_request_and_negative_input='passed',date_and_capacity_validation='passed',reservation_and_exhausted_need='passed',payoff_boundary_and_cap='passed',previous_quote_reproduction='passed',statistical_calibration='not_passed; exploratory replay underestimates payouts'),indent=2),encoding='utf-8')
    print(json.dumps(dict(contract=contract,validation_selected=validation_selected,metrics=metrics,quotes=quotes,sampling=sampling),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
