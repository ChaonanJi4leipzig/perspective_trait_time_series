"""Derive only CP2-dependent paper inputs from a validated versioned product.

Uses the existing production, S2 inventory, ICOS alignment and seasonality
kernels. --run enables the standard WORK/data write guard. Other sites' rows
are reused verbatim. No training or sealed value reads.
"""
from make_runtime import cli
if __name__=='__main__':cli(__doc__)
from pathlib import Path
import json,hashlib,importlib.util
import numpy as np,pandas as pd
import paths
import make_production_outputs as products
import make_s2_dates as s2
import make_icos_align as icos
import make_si1_inventory as inventory
import make_si1_stac_fill as stac

SITE='Castelporziano'
TASK=paths.PROJECT/'analysis/0_perspective/figures_add_samples/cp2_2022_extension_20261116'
BACKUP=TASK/'backup'
W=paths.WORK;D=paths.DERIVED
BASE=json.loads((TASK/'PAPER_BASELINE_SHA256.json').read_text())
OUTPUTS={}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,default=str)+'\n')
def parquet(frame,name):
    dest=paths.DATA/name
    if dest.exists():assert sha(dest)==BASE['data/'+name], 'Paper data changed since backup: '+name
    frame.to_parquet(dest,index=False)
    OUTPUTS[name]=sha(dest)

def multi_year(frame):
    """Exact render_fig4.area_baseline: month/day, <=16d interpolation, >=2y."""
    frame=frame.copy();frame.date=pd.to_datetime(frame.date,utc=True,format='mixed')
    frame['year']=frame.date.dt.year
    frame['seasonal_date']=pd.to_datetime(dict(year=2000,month=frame.date.dt.month,day=frame.date.dt.day)).to_numpy()
    grid=np.arange(1,367);rows=[]
    for trait in ['LMA','CHL','EWT','N']:
        annual=[]
        for year in range(2022,2027):
            f=frame[frame.trait.eq(trait)&frame.kind.eq('Production NN')&frame.year.eq(year)].sort_values('date')
            days=pd.DatetimeIndex(f.seasonal_date).dayofyear.to_numpy();values=f['mean'].to_numpy(float);series=np.full(366,np.nan)
            for day,value in zip(days,values):
                if np.isfinite(value):series[int(day)-1]=value
            for i in range(len(f)-1):
                gap=(f.date.iloc[i+1]-f.date.iloc[i]).total_seconds()/86400
                if gap<=16 and np.isfinite(values[i:i+2]).all():
                    use=(grid>=days[i])&(grid<=days[i+1]);series[use]=np.interp(grid[use],days[i:i+2],values[i:i+2])
            annual.append(series)
        a=np.asarray(annual);n=np.isfinite(a).sum(0);mean=np.divide(np.nansum(a,axis=0),n,out=np.full(366,np.nan),where=n>=2)
        for i,day in enumerate(grid):rows.append(dict(site=SITE,trait=trait,day_of_year=int(day),date=str((pd.Timestamp('2000-01-01')+pd.Timedelta(days=int(day)-1)).date()),mean=mean[i],n_years=int(n[i]),**{f'year_{y}':a[j,i] for j,y in enumerate(range(2022,2027))}))
    return pd.DataFrame(rows)

def figure_data():
    products.main(SITE)
    frame=pd.read_csv(D/f'Fig4_trajectories_{SITE}.csv')
    parquet(frame[['date','trait','kind','mean']],f'Fig4_trajectories_{SITE}.parquet')
    baseline=multi_year(frame)
    baseline.to_csv(D/f'Fig4_multi_year_mean_{SITE}.csv',index=False)
    parquet(baseline[['trait','date','mean']],f'Fig4_multi_year_mean_{SITE}.parquet')
    with np.load(paths.production_site(SITE)/'trait_cube.npz') as z:
        names=[str(t).split(' ')[0] for t in z['traits']];rows=[]
        for short,trait in [('LMA','LMA'),('CHL','Chl'),('EWT','EWT'),('N','N')]:
            j=names.index(trait)
            for i in np.flatnonzero(z['is_fortnightly']):
                keep=z['forest_mask']&z['w7_support'][i];coverage=float(keep.sum()/z['forest_mask'].sum());good=coverage>=.8
                v=float(z['mean'][i,j][keep].mean()) if good else np.nan
                perseed=np.asarray([float(a[keep].mean()) for a in z['per_seed'][:,i,j]]) if good else np.full(5,np.nan)
                old=frame[frame.kind.eq('Production NN')&frame.trait.eq(short)&frame.date.eq(str(z['dates'][i]))]['mean'].item()
                assert np.isclose(v,old,atol=1e-12,rtol=0,equal_nan=True)
                rows.append(dict(site=SITE,site_id=32,date=str(z['dates'][i]),trait=short,mean=v,seed_min=float(perseed.min()),seed_max=float(perseed.max()),coverage=coverage,gate80=good,forest_pixels=int(z['forest_mask'].sum()),supported_forest_pixels=int(keep.sum())))
    ensemble=pd.DataFrame(rows);name=f'Fig4_ensemble_range_{SITE}.csv';target=paths.DATA/name
    assert sha(target)==BASE['data/'+name]
    ensemble.to_csv(target,index=False,float_format='%.17g');OUTPUTS[name]=sha(target)
    candidate=W/f'production/{SITE}/Fig4_cube_{SITE}.npz'
    with np.load(candidate) as z:fields={k:z[k] for k in ['data','trait_names','t0','display_dates']}
    target=paths.DATA/f'Fig4_cube_{SITE}.npz'
    with np.load(target) as z:
        same=all(np.array_equal(v,z[k],equal_nan=True) if v.dtype.kind in 'fc' else np.array_equal(v,z[k]) for k,v in fields.items())
    if not same:
        assert sha(target)==BASE['data/'+target.name]
        np.savez_compressed(target,**fields);OUTPUTS[target.name]=sha(target)
    save(W/'CUBE_IDENTITY.json',dict(unchanged=same,paper_cube_sha256=sha(target),reason='2024 outside new 2022 context'))
    s2.ticks([SITE])
    ticks=pd.read_csv(D/f'Fig4_S2_valid_dates_{SITE}.csv')
    parquet(ticks[['year','plot','date_utc','valid_fraction']],f'Fig4_S2_valid_dates_{SITE}.parquet')
    medians=[]
    old=pd.read_parquet(BACKUP/f'data/Fig4_trajectories_{SITE}.parquet')
    for label,f in [('old',old),('new',frame)]:
        f=f[f.kind.eq('Production NN')].copy();date=pd.to_datetime(f.date,utc=True,format='mixed')
        f['season']=np.asarray(['DJF','DJF','MAM','MAM','MAM','JJA','JJA','JJA','SON','SON','SON','DJF'])[date.dt.month.to_numpy()-1]
        for (trait,season),part in f.groupby(['trait','season'],sort=False):
            unit={'LMA':'g m^-2','CHL':'ug cm^-2','EWT':'mg cm^-2','N':'mg cm^-2'}[trait]
            medians.append(dict(version=label,site=SITE,trait=trait,season=season,median=part['mean'].median(),n=int(part['mean'].notna().sum()),unit=unit))
    medians=pd.DataFrame(medians);medians.to_csv(W/'seasonal_medians_old_new.csv',index=False)
    medians[medians.version.eq('new')].drop(columns='version').to_csv(W/'seasonal_medians.csv',index=False)

def flux_data():
    icos.main([SITE])
    m=pd.read_csv(W/f'model_series_{SITE}.csv')
    cols=['date','GPP','Chl','Chl_lo','Chl_hi','CCC','CCC_lo','CCC_hi','LAI','LAI_lo','LAI_hi']
    parquet(m[cols],f'model_series_{SITE}.parquet')
    new=pd.read_csv(W/'spearman_GPP.csv');old=pd.read_parquet(paths.DATA/'spearman_GPP.parquet')
    keep=old[~old.site.eq(SITE)]
    combined=pd.concat([keep,new[old.columns]],ignore_index=True)
    combined.to_csv(D/'spearman_GPP.csv',index=False)
    parquet(combined,'spearman_GPP.parquet')
    events=pd.read_csv(W/'seasonal_timing.csv');rows=[]
    for (site,q,metric),f in events.groupby(['site','quantity','metric'],sort=False):
        for scope in ['all_applicable','resolved_SOS_or_complete_peak']:
            s=f[~f.template_not_applicable&f.delta_days.notna()]
            if scope=='resolved_SOS_or_complete_peak':
                s=s[s.GPP_bracket_days.le(16)&s.trait_bracket_days.le(16)] if metric=='SOS' else s[~s.partial_record]
            rows.append(dict(site=site,quantity=q,metric=metric,scope=scope,n=len(s),mean_delta_days=s.delta_days.mean(),median_delta_days=s.delta_days.median(),min_delta_days=s.delta_days.min(),max_delta_days=s.delta_days.max()))
    pd.DataFrame(rows).to_csv(W/'seasonal_timing_summary.csv',index=False)
    # Refresh the other CP2 text rows using exactly the original LOYO,
    # fixed-bootstrap-bank and threshold-template functions. No new draws.
    source=paths.FLUX/'work_v4/run_new_sites.py'
    spec=importlib.util.spec_from_file_location('cp2_original_seasonality',source)
    diagnostic=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostic)
    H=diagnostic.H
    traj=pd.read_csv(D/f'Fig4_trajectories_{SITE}.csv');traj.trait=traj.trait.replace({'CHL':'Chl'})
    traj.date=pd.to_datetime(traj.date,utc=True,format='mixed').dt.tz_localize(None)
    f=traj[traj.kind.eq('Production NN')].pivot(index='date',columns='trait',values='mean').sort_index()
    aligned=pd.read_csv(paths.FLUX/f'flux_aligned_{SITE}.csv',parse_dates=['date']).set_index('date')
    assert f.index.equals(aligned.index)
    f['GPP']=aligned.GPP
    contributions=[]
    for v in ['Chl','N','GPP']:
        ref,n,c=H.loyo(f[v]);contributions.extend(c)
        f[v+'_anomaly_LOYO']=f[v]-ref;f[v+'_n_reference_years']=n
    f.to_csv(W/f'anomaly_series_{SITE}.csv')
    pd.DataFrame(contributions).to_csv(W/'LOYO_contributions.csv',index=False)
    daily=pd.read_csv(paths.FLUX/f'flux_daily_{SITE}.csv',parse_dates=['date']).set_index('date')
    ph=diagnostic.new_phenology(SITE,f,daily);summary=diagnostic.summarize_phenology(ph)
    ph.to_csv(W/'phenology_dates_Castelporziano.csv',index=False);summary.to_csv(W/'phenology_summary_Castelporziano.csv',index=False)
    policy=diagnostic.PHASE1['IT-Cp2']['climatology']['GPP']
    assert not policy['climatology_available'] and policy['primary_anomaly_reference'].startswith('LOYO')
    bank=diagnostic.BANK;assert bank.shape==(10000,113)
    rows=[]
    for trait in ['Chl','N']:
        a=H.bootstrap_spearman(f[trait+'_anomaly_LOYO'].to_numpy(),f.GPP_anomaly_LOYO.to_numpy(),np.ones(len(f),bool),bank)
        b=H.bootstrap_spearman(f[trait].to_numpy(),f.GPP.to_numpy(),f.index.month.isin([6,7,8]),bank)
        row=dict(site=SITE,trait=trait,anomaly_rho=a[0],anomaly_low=a[2],anomaly_high=a[3],anomaly_n=a[1],anomaly_reference='LOYO',JJA_raw_rho=b[0],JJA_raw_low=b[2],JJA_raw_high=b[3],JJA_n=b[1])
        for phase in ['SOS','EOS']:
            q=summary[summary.trait.eq(trait)&summary.GPP_variant.eq('GPP')&summary.threshold.eq(.2)&summary.phase.eq(phase)&summary.scope.eq('primary_unflagged')].iloc[0]
            row.update({phase+'_mean_delta':q.mean_delta,phase+'_n':int(q.n_used),phase+'_flags':q.excluded_flags})
        rows.append(row)
    pd.DataFrame(rows).to_csv(W/'cross_site_summary.csv',index=False)
    save(W/'SEASONALITY_KERNELS.json',dict(source=str(source),sha256=sha(source),core_sha256=sha(paths.FLUX/'tests/run_seasonality_tests.py'),bootstrap_bank_sha256=sha(paths.FLUX/'tests/bootstrap_product_indices.npz'),new_draws=0))

def si1_data():
    inv=pd.read_csv(paths.INVENTORY_SOURCE/'s2_store_metadata_inventory.csv')
    stores=inv[inv.site_id.eq(32)].copy()
    new=paths.s2_store(32,2022);attrs=json.loads((new/'.zattrs').read_text())
    row=stores.iloc[0].to_dict();row.update(path=str(new),site_id=32,year=2022,zattrs_sha256=sha(new/'.zattrs'),time_header_sha256=sha(new/'time/.zarray'),rebuild_run=attrs['s2_full_rebuild_run'],source_version=attrs['s2_source_version'])
    row.update(listed_for_rebuild=False,listed_for_extension=False)
    import zarr
    row['dates']=zarr.open_group(new,mode='r')['time'].shape[0]
    stores=pd.concat([pd.DataFrame([row]),stores],ignore_index=True)
    inventory.CACHE.mkdir(parents=True,exist_ok=True)
    inventory.site_job((32,stores))
    obs=pd.read_csv(inventory.CACHE/'s2_site_32.csv')
    stamp=pd.to_datetime(obs.date_utc,utc=True,format='mixed')
    obs=obs[stamp.ge(stac.START)&stamp.lt(stac.STOP)].copy()
    old=pd.read_csv(paths.FINAL/'source_data/SI1_S2_observations.csv')
    for c in old.columns:
        if c not in obs:obs[c]=old.loc[old.site_id.eq(32),c].iloc[0]
    merged=pd.concat([old[~old.site_id.eq(32)],obs[old.columns]],ignore_index=True)
    merged.to_csv(D/'SI1_S2_observations.csv',index=False)
    counts=pd.read_csv(paths.FINAL/'source_data/SI1_counts_site_year.csv')
    for year in range(2022,2027):
        ix=counts.site_id.eq(32)&counts.year.eq(year);part=obs[obs.year.eq(year)]
        counts.loc[ix,'S2_store_present']=paths.s2_store(32,year).exists()
        counts.loc[ix,'n_s2_valid']=int(part.valid_observation.sum())
    counts.to_csv(D/'SI1_counts_site_year.csv',index=False)
    stac.WORK.mkdir(parents=True,exist_ok=True)
    c,s,local,gaps,centres=stac.inventory([32])
    items=pd.read_csv(paths.FINAL/'source_data/SI1_pc_stac_items.csv.gz')
    stac.derive(c,s,local,gaps,items[items.site_id.eq(32)].copy())
    # stac.derive writes CP2-only tables; retain all other sites' frozen rows.
    for name in ['SI1_counts_site_year_pc_fill.csv','SI1_pc_dates.csv','SI1_pc_fill_dates.csv','SI1_S2_gap_periods.csv']:
        original=pd.read_csv(paths.FINAL/'source_data'/name);current=pd.read_csv(D/name)
        pd.concat([original[~original.site_id.eq(32)],current.reindex(columns=original.columns)],ignore_index=True).to_csv(D/name,index=False)
    original=pd.read_parquet(paths.DATA/'SI1_S2_observations.parquet')
    newrows=obs[['site_id','date_utc','valid_observation']].copy()
    # Public table stores timestamp strings; preserve all unrelated rows/dtypes.
    combined=pd.concat([original[~original.site_id.eq(32)],newrows],ignore_index=True)
    parquet(combined,'SI1_S2_observations.parquet')
    fill=pd.read_csv(D/'SI1_pc_fill_dates.csv')
    parquet(fill[['site_id','date_utc']],'SI1_pc_fill_dates.parquet')
    totals=pd.read_csv(D/'SI1_counts_site_year_pc_fill.csv')
    receipt=dict(n_local_valid=int(totals.n_local_valid.sum()),n_pc_fill=int(totals.n_pc_fill.sum()),n_s2_combined=int(totals.n_combined.sum()),median_s2_per_site_year=float(totals.n_combined.median()))
    save(W/'SI1_counts_receipt.json',receipt)
    save(W/'SI1_EXTENSION_RECEIPT.json',dict(recomputed_site=32,stores=stores.to_dict('records'),catalogue_cache_sha256=sha(paths.FINAL/'source_data/SI1_pc_stac_items.csv.gz'),frozen_other_sites_reused=True,outputs_sha256={p.name:sha(p) for p in D.glob('SI1*')},sealed_value_reads=0))

def main(resume_inventory=False):
    product=paths.production_site(SITE)
    assert product.name=='Castelporziano_2022ext_20261116'
    manifest=json.loads((product/'manifest.json').read_text())
    assert manifest['sealed_value_reads']==0 and manifest['new_fits']==0 and manifest['GPU_hours']==0
    if not resume_inventory:
        figure_data();flux_data()
    else:
        assert (W/'CUBE_IDENTITY.json').exists() and (W/'cross_site_summary.csv').exists()
    si1_data()
    for name in ['Fig4_trajectories_Castelporziano.parquet','Fig4_ensemble_range_Castelporziano.csv','Fig4_multi_year_mean_Castelporziano.parquet','Fig4_S2_valid_dates_Castelporziano.parquet','model_series_Castelporziano.parquet','spearman_GPP.parquet']:
        OUTPUTS[name]=sha(paths.DATA/name)
    save(W/'PAPER_DATA_RECEIPT.json',dict(product=str(product),product_manifest_sha256=sha(product/'manifest.json'),outputs_sha256=OUTPUTS,all_events_development=True,sealed_value_reads=0,fold0_rows_loaded=0,forest_pixels=2171,coverage_gate=.8,support_days=7))
    # Keep update1's public input receipt consistent without altering Hesse.
    p=paths.REPO/'.public_update1_data_receipt.json'
    receipt=json.loads(p.read_text());receipt['outputs_sha256'][f'Fig4_ensemble_range_{SITE}.csv']=OUTPUTS[f'Fig4_ensemble_range_{SITE}.csv']
    receipt['production'][SITE].update(source=str(product/'trait_cube.npz'),sha256=sha(product/'trait_cube.npz'),extension_receipt=str(W/'PAPER_DATA_RECEIPT.json'))
    save(W/'PUBLIC_UPDATE1_EXTENSION_RECEIPT.json',receipt)
    print(json.dumps(OUTPUTS,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-inventory',action='store_true',help='Resume after completed figure/flux derivations in the same WORK')
    args=parser.parse_args()
    main(args.resume_inventory)
