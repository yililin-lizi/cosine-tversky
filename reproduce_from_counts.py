"""Reproduce frozen ranking summaries and paired bootstrap CIs from abstract counts.
Run from any directory: python reproduce_from_counts.py
Dependencies: numpy and pandas. No terminology source file is required.
"""
import os,math,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent
OUT=BASE/'reproduced';OUT.mkdir(exist_ok=True)
METHODS=('cosine','tversky_0.1','tversky_0.3','tversky_0.5','jaccard','dice')
METRICS=['parent_over_child','top1','top5','mrr']
def stable_seed(*parts):return int(hashlib.sha256('|'.join(str(x) for x in parts).encode('utf-8')).hexdigest()[:16],16)%(2**32)
def metrics(scores,ref,ancestors,legacy):
 target=scores[ref];eq=np.isclose(scores,target,rtol=1e-12,atol=1e-12);greater=int(((scores>target)&~eq).sum());tied=int(eq.sum())
 if legacy:top1=float(greater==0);top5=float(greater<5);mrr=1.0/(greater+1)
 else:top1=1.0/tied if greater==0 else 0.;top5=max(0.,min(1.,(5.-greater)/tied));mrr=sum(1.0/r for r in range(greater+1,greater+tied+1))/tied
 ancestor_better=bool(np.any((scores>target)&~eq&ancestors))
 return [int(ancestor_better),top1,top5,mrr],tied
counts=pd.read_csv(BASE/'candidate_counts.csv.gz')
design=pd.read_csv(BASE/'trial_design.csv').set_index(['trial_id','scenario','condition_id'])
rows=[]
for (trial,scenario,condition),g in counts.groupby(['trial_id','scenario','condition_id'],sort=False):
 meta=design.loc[(trial,scenario,condition)];k=g.k.to_numpy(float);e=g.e.to_numpy(float);c=g.c.to_numpy(float)
 assert g.is_reference.sum()==1
 ref=int(np.flatnonzero(g.is_reference.to_numpy())[0]);ancestors=g.is_ancestor.to_numpy(bool)
 for method in METHODS:
  if method=='cosine':den=np.sqrt((k+e)*(k+c))
  else:
   alpha,beta=(1.,1.) if method=='jaccard' else ((.5,.5) if method=='dice' else (1.,float(method.split('_')[1])))
   den=k+alpha*e+beta*c
  scores=np.divide(k,den,out=np.zeros(len(k)),where=den!=0)
  values,tied=metrics(scores,ref,ancestors,scenario=='legacy_same_release')
  row={'trial_id':trial,'scenario':scenario,'condition_id':condition,'method':method}
  row.update(meta.to_dict());row.update(dict(zip(METRICS,values)));row['tie_size']=tied;rows.append(row)
tr=pd.DataFrame(rows);tr.to_csv(OUT/'trial_results.csv.gz',index=False,compression='gzip')
keys=['scenario','hierarchy_retention','attribute_retention','condition_id'];summaries=[];intervals=[]
for key,g in tr.groupby(keys,sort=True):
 scenario,hr,ar,condition=key;seeds=sorted(g.root_concept_id.unique());n=len(seeds)
 rng=np.random.RandomState(stable_seed(7,'bootstrap',scenario,condition));weights=rng.multinomial(n,[1./n]*n,size=2000)/float(n)
 arrays={m:g[g.method==m].groupby('root_concept_id')[METRICS].mean().reindex(seeds).values for m in METHODS}
 for method in METHODS:
  q=g[g.method==method];mean=q[METRICS].mean();bs=weights.dot(arrays[method]);delta=arrays[method]-arrays['cosine'];bd=weights.dot(delta)
  row=dict(zip(keys,key));row.update(method=method,n_trials=q.trial_id.nunique(),n_seeds=n,mean_candidates=q.n_candidates.mean(),reference_tie_rate=(q.tie_size>1).mean(),empty_expression_rate=q.empty_expression.mean());row.update(mean.to_dict());summaries.append(row)
  for j,metric in enumerate(METRICS):
   lo,hi=np.percentile(bs[:,j],[2.5,97.5]);dl,dh=np.percentile(bd[:,j],[2.5,97.5]);row=dict(zip(keys,key));row.update(method=method,metric=metric,estimate=mean[metric],ci_low=lo,ci_high=hi,difference_vs_cosine=delta[:,j].mean(),difference_ci_low=dl,difference_ci_high=dh,bootstrap_replicates=2000,n_seeds=n);intervals.append(row)
s=pd.DataFrame(summaries);ci=pd.DataFrame(intervals)
s.to_csv(OUT/'condition_summary.csv',index=False);ci.to_csv(OUT/'paired_bootstrap_confidence_intervals.csv',index=False)
for file,df,sort in [('condition_summary.csv',s,keys+['method']),('paired_bootstrap_confidence_intervals.csv',ci,keys+['method','metric'])]:
 expected=pd.read_csv(BASE/'expected'/file).sort_values(sort).reset_index(drop=True);actual=df.sort_values(sort).reset_index(drop=True)[expected.columns]
 for col in expected.columns:
  if pd.api.types.is_numeric_dtype(expected[col]):assert np.allclose(expected[col],actual[col],rtol=1e-11,atol=1e-12),col
  else:assert expected[col].equals(actual[col]),col
json.dump({'status':'passed','candidate_rows':len(counts),'reproduced_summaries':len(s),'reproduced_intervals':len(ci),'bootstrap_replicates':2000},open(OUT/'verification.json','w'),indent=2)
print('PASS: all condition summaries and 95% paired bootstrap intervals reproduced.')
