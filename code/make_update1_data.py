"""Frozen temporal rows/bootstrap and production cubes -> small update-1 plotting inputs.
Needs paths.FINAL/source_data and paths.PRODUCTION; writes data/ plus a local receipt.
Runs: cap2048_v3_nested11_20261108 and cap2048_production_nn_temporal_20261112.
No teacher stores, training, inference or new bootstrap draws are used.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import paths
import make_metrics_core as metrics
ORDER=['LAI','LMA','Chl','Car','EWT','N','C','NSC','Lignin','Fibre','Cellulose']
RAW=['LMA','N','LAI','C','Chl','EWT','Car','Lignin','Cellulose','Fibre','NSC']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
 if not a.run:p.error('Use --run to derive small figure summaries from frozen inputs')
 assignment=pd.read_parquet(paths.ASSIGNMENT);allowed=set(assignment.loc[assignment.new_role.eq('development'),'event_id'])
 source=paths.FINAL/'source_data/formal_temporal_with_pixel_baselines.parquet'
 ids=pd.read_parquet(source,columns=['target_event_id']);assert set(ids.target_event_id)<=allowed
 f=pd.read_parquet(source);budget=pd.read_parquet(paths.DATA/'Fig3b_budget.parquet');rows=[]
 receipt={'assignment_sha256':sha(paths.ASSIGNMENT),'development_rows':len(f),'all_events_development':True,'sealed_value_reads':0,'formal_rows_sha256':sha(source),'bootstrap_draws':1000,'bootstrap_seed':20261111,'bootstrap_source_sha256':{},'production':{}}
 for trait in ORDER:
  j=RAW.index(trait);canonical=pd.DataFrame(dict(site_id=f.site_id,event_id=f.target_event_id,date_utc=pd.to_datetime(f.target_event_id.str.split(':',n=1).str[1],utc=True),pixel_id=f.sample_id,reference=f[f'ref__{j}'],qc=True,oos=True,pixel_seasonal=f[f'pixel_season__{j}']))
  canonical=canonical[np.isfinite(canonical.pixel_seasonal)]
  result=metrics.evaluate(canonical,prediction_columns={'baseline':'pixel_seasonal'},data_version=paths.RUN_TEMPORAL,pixel_ids_stable=False,bootstrap_draws=0)
  point=result['per_seed'].iloc[0]
  path=paths.FINAL/f'source_data/Fig3b_bootstrap_{trait}.parquet';boot=pd.read_parquet(path);assert len(boot)==1000
  totals=boot[[f'recovered_share_{l}' for l in metrics.LEVELS]].sum(axis=1,min_count=3)
  lo,hi=np.nanquantile(totals,[.025,.975]);b=budget[budget.trait.eq(trait)]
  for level in metrics.LEVELS:
   original=b[b.level.eq(level)].iloc[0];ci=np.nanquantile(boot[f'recovered_share_{level}'],[.025,.975]);assert np.allclose(ci,[original.recovered_low,original.recovered_high],atol=1e-14,rtol=0)
  assert np.isclose(point.recovered_share_date,b.iloc[0].pixel_seasonal_recovered_date,atol=1e-14)
  rows.append(dict(trait=trait,total_recovered=float(b.recovered.sum()),total_low=lo,total_high=hi,pixel_seasonal_total=float(sum(point[f'recovered_share_{l}'] for l in metrics.LEVELS)),baseline_n=len(result['prepared']['data'])))
  receipt['bootstrap_source_sha256'][trait]=sha(path)
 pd.DataFrame(rows).to_parquet(paths.DATA/'Fig3b_totals.parquet',index=False)
 for site,sid in [('Hesse',45),('Castelporziano',32)]:
  path=paths.PRODUCTION/site/'trait_cube.npz';manifest=json.loads((path.parent/'manifest.json').read_text());assert sha(path)==manifest['files']['trait_cube.npz']
  original=pd.read_parquet(paths.DATA/f'Fig4_trajectories_{site}.parquet');original=original[original.kind.eq('Production NN')].copy();original['stamp']=pd.to_datetime(original.date,utc=True)
  out=[];errors={};seed_errors={}
  with np.load(path) as z:
   names=[str(t).split(' ')[0] for t in z['traits']];assert z['seeds'].tolist()==list(range(42,47));ps=z['per_seed'];mean=z['mean'];support=z['w7_support'];forest=z['forest_mask'];fort=z['is_fortnightly'];dates=pd.to_datetime(z['dates'],utc=True,format='mixed');assert fort.sum()==113
   assert np.array_equal(mean,np.mean(ps,axis=0,dtype='f4'),equal_nan=True)
   for short,trait in [('LMA','LMA'),('CHL','Chl'),('EWT','EWT'),('N','N')]:
    j=names.index(trait);errors[trait]=0.;seed_errors[trait]=0.
    for i in np.flatnonzero(fort):
     keep=forest&support[i];coverage=float(keep.sum()/forest.sum());good=coverage>=.8
     v=float(mean[i,j][keep].mean()) if good else np.nan
     perseed=np.array([float(ps[s,i,j][keep].mean()) for s in range(5)]) if good else np.full(5,np.nan)
     old=float(original.loc[original.trait.eq(short)&original.stamp.eq(dates[i]),'mean'].iloc[0]);assert np.isclose(v,old,atol=1e-12,rtol=0,equal_nan=True)
     if good:
      error=abs(float(perseed.mean())-v);assert error<=4*np.finfo('f4').eps*max(abs(v),1)
      errors[trait]=max(errors[trait],abs(v-old));seed_errors[trait]=max(seed_errors[trait],error)
     out.append(dict(site=site,site_id=sid,date=dates[i].isoformat(),trait=short,mean=v,seed_min=float(perseed.min()),seed_max=float(perseed.max()),coverage=coverage,gate80=good,forest_pixels=int(forest.sum()),supported_forest_pixels=int(keep.sum())))
  pd.DataFrame(out).to_csv(paths.DATA/f'Fig4_ensemble_range_{site}.csv',index=False,float_format='%.17g')
  receipt['production'][site]={'source':'upstream://V6/production_nn/'+site+'/trait_cube.npz','sha256':sha(path),'mean_identity_max_abs':errors,'mean_of_seed_means_vs_pixel_mean_max_abs':seed_errors,'dates':113,'labels_read':False}
 receipt['outputs_sha256']={name:sha(paths.DATA/name) for name in ['Fig3b_totals.parquet','Fig4_ensemble_range_Hesse.csv','Fig4_ensemble_range_Castelporziano.csv']}
 (paths.REPO/'.public_update1_data_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
