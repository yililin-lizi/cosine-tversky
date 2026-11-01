import gzip
SCENARIOS = ('primary_no_identity','identity_retained','legacy_same_release')
METHODS = ('cosine','tversky_0.1','tversky_0.3','tversky_0.5','jaccard','dice')
BOOTSTRAPS = 2000

def scores_for(counts, method):
    if method == 'cosine': return [cosine_kec(*x) for x in counts]
    if method == 'jaccard': return [tversky_kec(*x,alpha=1,beta=1) for x in counts]
    if method == 'dice': return [tversky_kec(*x,alpha=.5,beta=.5) for x in counts]
    return [tversky_kec(*x,alpha=1,beta=float(method.split('_')[1])) for x in counts]

def retrieval_metrics(scores, true_index):
    arr = np.array(scores); target = arr[true_index]
    equal = np.isclose(arr,target,rtol=SCORE_TIE_TOLERANCE,atol=SCORE_TIE_TOLERANCE)
    greater = int(((arr > target) & ~equal).sum()); tied = int(equal.sum())
    return {'top1': (1.0/tied if greater == 0 else 0.0),
            'top5': max(0.0,min(1.0,(5.0-greater)/tied)),
            'mrr': sum(1.0/r for r in range(greater+1,greater+tied+1))/tied,
            'tie_size':tied,'optimistic_top1':int(greater==0),
            'optimistic_top5':int(greater<5),'optimistic_mrr':1.0/(greater+1)}
assert retrieval_metrics([1]*10,0)['top1'] == .1
assert retrieval_metrics([1]*10,0)['top5'] == .5
assert math.isclose(retrieval_metrics([1,1],0)['mrr'],.75)
assert retain_features(set(range(2)),.1,7)  # legacy behavior explicitly preserved only in legacy scenario

# Theory audit is descriptive: direct attributes need not be inherited.
audit_rows=[]
for _, plan in trial_plan_df.drop_duplicates(['root_concept_id','true_concept_id']).iterrows():
    root=int(plan.root_concept_id); child=int(plan.true_concept_id)
    p=feature_export[str(root)]; q=feature_export[str(child)]
    ph=set(p['hierarchy']); ch=set(q['hierarchy']); pa=set(p['attributes']); ca=set(q['attributes'])
    audit_rows.append({'root_concept_id':root,'reference_concept_id':child,
                      'hierarchy_contained':ph <= ch,'full_features_contained':(ph|pa) <= (ch|ca),
                      'missing_parent_attributes':len(pa-ca)})
audit_df=pd.DataFrame(audit_rows); audit_df.to_csv(os.path.join(OUT_DIR,'feature_containment_audit.csv'),index=False)
assert audit_df.hierarchy_contained.all()

trial_rows=[]; expression_rows=[]
score_path=os.path.join(OUT_DIR,'candidate_scores.csv.gz')
score_handle=gzip.open(score_path,'wt',newline=''); writer=None
for ix,plan in trial_plan_df.iterrows():
    ids=json.loads(plan.candidate_ids_json); ti=ids.index(int(plan.true_concept_id)); ref=feature_export[str(int(plan.true_concept_id))]
    ancestor_ids=set(json.loads(plan.ancestor_candidate_ids_json))
    expr_seed=stable_seed(int(plan.trial_seed),'expression-order')
    for scenario in SCENARIOS:
        legacy=scenario=='legacy_same_release'
        hier_key='legacy_hierarchy' if legacy else 'hierarchy'
        hier=set(ref[hier_key]); attrs=set(ref['attributes'])
        if scenario=='primary_no_identity': hier.discard('concept:{}'.format(int(plan.true_concept_id)))
        # Uniforms are shared across conditions and corrected scenarios.
        hierarchy_uniform={f:random.Random(stable_seed(expr_seed,'bernoulli',f)).random() for f in hier}
        attribute_uniform={f:random.Random(stable_seed(expr_seed,'bernoulli',f)).random() for f in attrs}
        for hr in HIERARCHY_RETENTION_VALUES:
            for ar in ATTRIBUTE_RETENTION_VALUES:
                condition='H{:.1f}_A{:.1f}'.format(hr,ar)
                if legacy:
                    kh=retain_features(hier,hr,stable_seed(expr_seed,'hierarchy-order'))
                    ka=retain_features(attrs,ar,stable_seed(expr_seed,'attribute-order'))
                else:
                    kh={f for f,u in hierarchy_uniform.items() if u<hr}
                    ka={f for f,u in attribute_uniform.items() if u<ar}
                expr=set(kh)|set(ka)
                csets=[set(feature_export[str(cid)][hier_key])|set(feature_export[str(cid)]['attributes']) for cid in ids]
                counts=[kec(expr,cset) for cset in csets]
                rootset=set(feature_export[str(int(plan.root_concept_id))][hier_key])|set(feature_export[str(int(plan.root_concept_id))]['attributes'])
                meta={'trial_id':plan.trial_id,'source_row':int(plan.source_row),'root_concept_id':int(plan.root_concept_id),
                      'true_concept_id':int(plan.true_concept_id),'scenario':scenario,'condition_id':condition,
                      'hierarchy_retention':hr,'attribute_retention':ar,'n_candidates':len(ids),
                      'hierarchy_total':len(hier),'hierarchy_kept':len(kh),'attribute_total':len(attrs),'attribute_kept':len(ka),
                      'empty_expression':int(not expr),'reference_identity_retained':int('concept:{}'.format(int(plan.true_concept_id)) in expr),
                      'n_features_distinguishing_reference_from_root':len(expr-rootset)}
                expression_rows.append(dict(meta,expression_features_json=json.dumps(sorted(expr))))
                all_scores={m:scores_for(counts,m) for m in METHODS}
                assert competition_ranks(all_scores['jaccard']).tolist()==competition_ranks(all_scores['dice']).tolist()
                for method,scores in all_scores.items():
                    metrics=retrieval_metrics(scores,ti)
                    ancestor_better=any(scores[j]>scores[ti] and not math.isclose(scores[j],scores[ti],rel_tol=SCORE_TIE_TOLERANCE,abs_tol=SCORE_TIE_TOLERANCE) for j,cid in enumerate(ids) if cid in ancestor_ids)
                    if legacy:
                        for metric in ('top1','top5','mrr'):metrics[metric]=metrics['optimistic_'+metric]
                    trial_rows.append(dict(meta,method=method,parent_over_child=int(ancestor_better),**metrics))
                for j,cid in enumerate(ids):
                    row={'trial_id':plan.trial_id,'scenario':scenario,'condition_id':condition,'candidate_concept_id':cid,
                         'is_reference':int(j==ti),'is_ancestor':int(cid in ancestor_ids),'k':counts[j][0],'e':counts[j][1],'c':counts[j][2]}
                    row.update({m:all_scores[m][j] for m in METHODS})
                    if writer is None: writer=csv.DictWriter(score_handle,fieldnames=list(row)); writer.writeheader()
                    writer.writerow(row)
    if (ix+1)%100==0:print('Evaluated {}/{} paired base trials'.format(ix+1,len(trial_plan_df)),flush=True)
score_handle.close()
tr=pd.DataFrame(trial_rows); ex=pd.DataFrame(expression_rows)
tr.to_csv(os.path.join(OUT_DIR,'trial_results.csv.gz'),index=False,compression='gzip')
ex.to_csv(os.path.join(OUT_DIR,'simulated_expressions.csv.gz'),index=False,compression='gzip')
keys=['scenario','hierarchy_retention','attribute_retention','condition_id']
metrics=['parent_over_child','top1','top5','mrr']
summary=[]; intervals=[]
for group_key,group in tr.groupby(keys,sort=True):
    scenario,hr,ar,condition=group_key
    seed_ids=sorted(group.root_concept_id.unique()); n=len(seed_ids)
    rng=np.random.RandomState(stable_seed(RANDOM_SEED,'bootstrap',scenario,condition))
    weights=rng.multinomial(n,[1.0/n]*n,size=BOOTSTRAPS)/float(n)
    seed_values={method:group[group.method==method].groupby('root_concept_id')[metrics].mean().reindex(seed_ids).values for method in METHODS}
    for method in METHODS:
        part=group[group.method==method]; mean=part[metrics].mean(); boot=weights.dot(seed_values[method]); delta=seed_values[method]-seed_values['cosine']; bd=weights.dot(delta)
        row=dict(zip(keys,group_key));row.update(method=method,n_trials=part.trial_id.nunique(),n_seeds=n,mean_candidates=part.n_candidates.mean(),reference_tie_rate=(part.tie_size>1).mean(),empty_expression_rate=part.empty_expression.mean())
        row.update(mean.to_dict());summary.append(row)
        for j,metric in enumerate(metrics):
            lo,hi=np.percentile(boot[:,j],[2.5,97.5]); dl,dh=np.percentile(bd[:,j],[2.5,97.5])
            ci=dict(zip(keys,group_key));ci.update(method=method,metric=metric,estimate=float(mean[metric]),ci_low=float(lo),ci_high=float(hi),difference_vs_cosine=float(delta[:,j].mean()),difference_ci_low=float(dl),difference_ci_high=float(dh),bootstrap_replicates=BOOTSTRAPS,n_seeds=n)
            intervals.append(ci)
summary_df=pd.DataFrame(summary); ci_df=pd.DataFrame(intervals)
summary_df.to_csv(os.path.join(OUT_DIR,'condition_summary.csv'),index=False)
ci_df.to_csv(os.path.join(OUT_DIR,'paired_bootstrap_confidence_intervals.csv'),index=False)
for scenario in SCENARIOS:
    table=summary_df[summary_df.scenario==scenario].pivot(index=['hierarchy_retention','attribute_retention'],columns='method',values=metrics)
    table.columns=['{} | {}'.format(a,b) for a,b in table.columns]
    table.reset_index().to_csv(os.path.join(OUT_DIR,'table4_{}.csv'.format(scenario)),index=False)
# Descriptive evidence strata; does not substitute for independent clinical labels.
tr['observed_refinement_evidence']=tr.n_features_distinguishing_reference_from_root>0
strata=tr.groupby(['scenario','condition_id','method','observed_refinement_evidence'])[metrics].agg(['mean','count'])
strata.to_csv(os.path.join(OUT_DIR,'refinement_evidence_strata.csv'))
ret=ex.groupby(['scenario','hierarchy_retention','attribute_retention']).apply(lambda g:pd.Series({
 'realized_hierarchy_retention':g.hierarchy_kept.sum()/g.hierarchy_total.sum() if g.hierarchy_total.sum() else np.nan,
 'realized_attribute_retention':g.attribute_kept.sum()/g.attribute_total.sum() if g.attribute_total.sum() else np.nan,
 'empty_expression_rate':g.empty_expression.mean(),'n_expressions':len(g)}))
ret.to_csv(os.path.join(OUT_DIR,'realized_retention.csv'))
# Integrity checks: paired methods and nine conditions per scenario/base trial.
assert tr.groupby(['scenario','trial_id','condition_id']).method.nunique().eq(len(METHODS)).all()
assert ex.groupby(['scenario','trial_id']).condition_id.nunique().eq(9).all()
assert len(tr)==len(trial_plan_df)*9*len(SCENARIOS)*len(METHODS)
assert ex[ex.scenario=='primary_no_identity'].reference_identity_retained.eq(0).all()
assert np.all((tr[metrics].values>=0)&(tr[metrics].values<=1))
# Figure for primary analysis; the connected x-axis enumerates the experimental grid.
fig,axes=plt.subplots(2,2,figsize=(12,8))
p=summary_df[summary_df.scenario=='primary_no_identity']
for ax,metric,title in zip(axes.flat,metrics,['Ancestor ranked above reference','Expected Top-1','Expected Top-5','Expected reciprocal rank']):
    for method in METHODS[:4]:
        q=p[p.method==method].sort_values(['hierarchy_retention','attribute_retention'])
        ax.plot(range(9),q[metric],marker='o',label=method)
    ax.set_xticks(range(9));ax.set_xticklabels(['{:.1f}/{:.1f}'.format(h,a) for h in HIERARCHY_RETENTION_VALUES for a in ATTRIBUTE_RETENTION_VALUES],rotation=45)
    ax.set_title(title);ax.set_xlabel('Hierarchy / attribute retention');ax.set_ylabel('Proportion');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8);fig.suptitle('Corrected ontology-feature recovery without reference identity');fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR,'figure1_primary.png'),dpi=180);plt.close(fig)
metadata={'rf2_release':'US1000124_20260301','inputs':input_signatures,'n_source_rows':len(source_df),'n_unique_seeds':len(terms_df),
 'n_eligible_seeds':int(trial_plan_df.root_concept_id.nunique()),'trials_per_condition':len(trial_plan_df),
 'mean_candidates':float(trial_plan_df.n_candidates.mean()),'max_candidates':int(trial_plan_df.n_candidates.max()),
 'n_unique_evaluated_candidates':len(used_ids),'characteristic_type':CHARACTERISTIC_TYPEID,
 'scenarios':list(SCENARIOS),'methods':list(METHODS),'bootstrap_replicates':BOOTSTRAPS,'random_seed':RANDOM_SEED,
 'hierarchy_containment_fraction':float(audit_df.hierarchy_contained.mean()),'full_containment_fraction':float(audit_df.full_features_contained.mean()),
 'software':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__},'completed_utc':datetime.utcnow().isoformat()+'Z'}
json.dump(metadata,open(os.path.join(OUT_DIR,'run_metadata.json'),'w'),indent=2)
print(json.dumps(metadata,indent=2))
print(summary_df[(summary_df.scenario=='primary_no_identity') & summary_df.method.isin(['cosine','tversky_0.1'])].to_string(index=False))
