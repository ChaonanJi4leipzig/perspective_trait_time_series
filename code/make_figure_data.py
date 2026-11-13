"""Needs frozen figure-data outputs at paths.FINAL/V6; writes only data/ plot inputs.
Copies selected columns into lossless Parquet and arrays into compressed NPZ.
No training, inference or metric calculation. Runs: temporal, spatial and production.
"""
from make_runtime import cli
if __name__=='__main__':cli(__doc__)
from pathlib import Path
import numpy as np,pandas as pd
import paths
V=paths.V6;F=paths.FINAL;D=paths.DATA
# Copy only plotting tables; strip paths/irrelevant metadata, never numeric values.
columns={
 'Fig3a_metrics.csv':['trait','scope','model','metric','mean'],
 'Fig3b_budget.csv':['trait','level','share','recovered','recovered_low','recovered_high','pixel_seasonal_recovered_date'],
 'Fig3a_pixel_baselines.csv':['trait','model','nrmse_P95_P05'],
 'SI1_S2_observations.csv':['site_id','date_utc','valid_observation'],
 'SI1_development_events.csv':['event_id','site_id','date','new_role'],
 'SI1_pc_fill_dates.csv':['site_id','date_utc'],
 'SI1_site_classes_filled.csv':['site_id','display_class','plot_label']}
for name,cols in columns.items():
 frame=pd.read_csv(F/'source_data'/name)
 if name=='Fig3a_metrics.csv':frame=frame[frame.model.isin(['NN','pixel_mean','pixel_seasonal_mean'])].reset_index(drop=True)
 frame[cols].to_parquet(D/Path(name).with_suffix('.parquet'),index=False)
for site in ['Hesse','Castelporziano']:
 for prefix,cols in [('Fig4_trajectories',['date','trait','kind','mean']),('Fig4_multi_year_mean',['trait','date','mean']),('Fig4_S2_valid_dates',['year','plot','date_utc','valid_fraction'])]:
  name=f'{prefix}_{site}.csv';pd.read_csv(F/'source_data'/name)[cols].to_parquet(D/Path(name).with_suffix('.parquet'),index=False)
 src=V/'figure_data/Fig4_cube_Hesse.npz' if site=='Hesse' else paths.PRODUCTION/'Castelporziano/Fig4_cube_Castelporziano.npz'
 with np.load(src) as z:
  fields={k:z[k] for k in ['data','trait_names','t0','display_dates']};np.savez_compressed(D/f'Fig4_cube_{site}.npz',**fields)
  with np.load(D/f'Fig4_cube_{site}.npz') as out:
   for k,v in fields.items():assert np.array_equal(v,out[k],equal_nan=True) if v.dtype.kind in 'fc' else np.array_equal(v,out[k])
 src=V/'evaluation_sites/Hesse/scene_maps.npz' if site=='Hesse' else F/'source_data/si3_work/evaluation_sites/Castelporziano/scene_maps.npz'
 with np.load(src) as z:
  fields={k:z[k] for k in ['reference','prediction','mask','event']};np.savez_compressed(D/f'SI3_maps_{site}.npz',**fields)
  with np.load(D/f'SI3_maps_{site}.npz') as out:
   for k,v in fields.items():assert np.array_equal(v,out[k],equal_nan=True) if v.dtype.kind in 'fc' else np.array_equal(v,out[k])
 for prefix in ['model_series','daily_GPP','field_PAI']:
  name=f'{prefix}_{site}.csv';df=pd.read_csv(F/'si4'/name)
  cols=['date','GPP','Chl','Chl_lo','Chl_hi','CCC','CCC_lo','CCC_hi','LAI','LAI_lo','LAI_hi'] if prefix=='model_series' else ['date','GPP'] if prefix=='daily_GPP' else ['date','PAI_mean','PAI_sd']
  df[cols].to_parquet(D/Path(name).with_suffix('.parquet'),index=False)
pd.read_csv(F/'si4/spearman_GPP.csv')[['region','site','quantity','season','spearman_rho','n']].to_parquet(D/'spearman_GPP.parquet',index=False)
