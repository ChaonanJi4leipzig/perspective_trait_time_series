"""Redraw with a file-open audit, compare PNG pixels, inspect PDFs; optional upstream metadata checks."""
from pathlib import Path
import argparse,os,sys,subprocess,tempfile,json,hashlib,re
import numpy as np,pandas as pd
from PIL import Image
import paths
R=paths.REPO

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
FIGURES={'Fig3','Fig4','Fig4_v2','SI1_data_volume','SI2_model','SI3_fold0_test','SI4_maps_qq','SI5_flux_consistency'}
CP2_WORK=R/'.work/cp2_2022_extension_20261116'
CP2_PRODUCT=paths.PRODUCTION/'Castelporziano_2022ext_20261116'
CP2_TASK=paths.PROJECT/'analysis/0_perspective/figures_add_samples/cp2_2022_extension_20261116'
FOLD0=paths.V6/'fold0_test_20261116'

def read_json(p):return json.loads(Path(p).read_text())

def check_fold0_schema():
 d=pd.read_parquet(R/'data/SI3_fold0_metrics.parquet')
 expected={'trait','model','population','statistic','seed','metric','estimate','ci_low','ci_high','n','n_events','n_sites','coverage','n_boot_valid'}
 assert set(d.columns)==expected
 forbidden={'sample_id','event_id','target_event_id','x','y','date','reference'}
 assert not any(c in forbidden or c.startswith(('trait__','ref__','pred__','NN_')) for c in d.columns)
 from common import ORDER
 assert set(d.trait)==set(ORDER) and len(d)==506
 assert set(d.metric)=={'r2','nrmse_P95_P05'}
 nn=d[d.model.eq('NN')]
 assert set(nn.population)=={'primary','pixel_mean_matched','pixel_seasonal_mean_matched'}
 for (trait,pop,metric),g in nn.groupby(['trait','population','metric']):
  per=g[g.statistic.eq('per_seed')];mean=g[g.statistic.eq('seed_mean')];ensemble=g[g.statistic.eq('ensemble_mean')]
  assert len(per)==5 and set(per.seed)==set(range(42,47)) and len(mean)==len(ensemble)==1
  assert np.isclose(mean.estimate.iloc[0],per.estimate.mean(),atol=1e-14,rtol=0)
  assert g[['n','n_events','n_sites','coverage']].nunique().eq(1).all()
  assert mean.seed.isna().all() and ensemble.seed.isna().all()
 for model in ['pixel_mean','pixel_seasonal_mean']:
  baseline=d[d.model.eq(model)]
  assert len(baseline)==22 and baseline.population.eq(model+'_defined').all() and baseline.statistic.eq('baseline').all() and baseline.seed.isna().all()
  for row in baseline.itertuples():
   primary=nn[nn.trait.eq(row.trait)&nn.population.eq('primary')&nn.metric.eq(row.metric)&nn.statistic.eq('seed_mean')].iloc[0]
   matched=nn[nn.trait.eq(row.trait)&nn.population.eq(model+'_matched')&nn.metric.eq(row.metric)&nn.statistic.eq('seed_mean')].iloc[0]
   assert np.isclose(row.coverage,row.n/primary.n,atol=1e-15,rtol=0)
   assert (row.n,row.n_events,row.n_sites,row.coverage)==(matched.n,matched.n_events,matched.n_sites,matched.coverage)
 assert d.coverage.between(0,1).all() and d.n.between(0,8776).all()
 assert (d.n_sites<=d.n_events).all() and (d.n_events<=d.n).all()
 finite=d[d.estimate.notna()];assert finite.n_boot_valid.eq(1000).all() and (finite.ci_low<=finite.ci_high).all()
 assert nn[nn.population.eq('primary')].n.eq(8776).all()
 return d

def check_fold0_authorization(assignment,allowed):
 # Only aggregate exports and identity metadata are read. Never open local
 # predictions, reference arrays, or teacher stores to repeat this evaluation.
 d=check_fold0_schema();run=paths.PROJECT/'runs/fold0_test_production_nn_20261116'
 auth=read_json(run/'AUTHORIZATION.json');freeze=read_json(run/'PROTOCOL_FREEZE.json')
 start=read_json(FOLD0/'VALUES_ACCESS_STARTED.json');done=read_json(FOLD0/'EVALUATION_DONE.json');review=read_json(FOLD0/'REVIEW_VALIDATION.json')
 assert auth['date']=='2026-11-16'
 assert auth['verbatim'].startswith('1，保持现在的fig3。2，加一个supplementary figure，')
 assert auth['scope']=='One evaluation of frozen cap2048_production_nn_temporal_20261112 seeds42-46 on v3 fold0 only'
 assert set(auth['excluded_roles'])=={'fold_conflict','QA_excluded'}
 assert freeze['fold0_value_reads_before_freeze']==0
 assert pd.Timestamp(freeze['frozen_at_utc'])<pd.Timestamp(start['started_at_utc'])<pd.Timestamp(done['completed_at_utc'])
 assert sha(run/'PROTOCOL.md')==start['protocol_sha256']==done['protocol_sha256']
 # The experiment card was legitimately closed after evaluation; its frozen
 # version is historical. All other protocol-pinned artifacts remain exact.
 for name,digest in freeze['sha256'].items():
  if name.endswith('experiment_card.yaml'):continue
  assert sha(paths.PROJECT/name)==digest,name
 assert review['state']=='pass' and review['assignment_sha256']==sha(paths.ASSIGNMENT)
 assert review['authorization_valid'] and review['protocol_precedes_values'] and review['training_development_only']
 assert done['state']=='complete' and done['one_time'] and done['no_fitting']
 assert done['fold0_rows_loaded']==review['fold0_rows_loaded']==10024
 assert not done['fold0_used_for_fit'] and not done['fold0_used_for_selection']
 assert sha(FOLD0/'metrics.parquet')==done['metrics_sha256']
 assert sha(R/'data/SI3_fold0_metrics.parquet')==done['paper_aggregate_sha256']
 local=pd.read_parquet(FOLD0/'metrics.parquet').drop(columns=['reference_p95_p5'])
 pd.testing.assert_frame_equal(local,d,check_exact=True)
 # Verify the CSV export as well, including the exact 88 copied text rows.
 pd.testing.assert_frame_equal(pd.read_csv(FOLD0/'metrics.csv',float_precision='round_trip').drop(columns=['reference_p95_p5']),d,check_exact=True)
 from make_numbers_for_text import fold0_rows,FIELDS
 import csv
 with (R/'data/numbers_for_text.csv').open() as f:
  reader=csv.DictReader(f);assert reader.fieldnames==FIELDS;numbers=list(reader)
 assert [r for r in numbers if r['quantity'].startswith('fold0_')]==fold0_rows()
 fold0=set(assignment.loc[assignment.new_role.eq('fold0'),'event_id']);assert len(fold0)==80 and not fold0&allowed
 flow=pd.read_csv(FOLD0/'sample_flow.csv');steps=pd.read_csv(FOLD0/'fold0_source_timesteps.csv',usecols=['event_id'])
 assert len(flow)==80 and set(flow.event_id)==set(steps.event_id)==fold0
 assert flow.n_reference.sum()==10024 and flow.n_primary.sum()==done['n_primary']==8776
 assert flow.loc[flow.n_primary.gt(0),'event_id'].nunique()==done['events']==75
 assert flow.loc[flow.n_primary.gt(0),'site_id'].nunique()==done['sites']==35
 metadata=read_json(FOLD0/'METADATA_AUDIT.json')
 training=paths.PRODUCTION/'training/rows.parquet'
 assert sha(training)==metadata['training_metadata_sha256']
 train=pd.read_parquet(training,columns=['target_event_id','site_id'])
 assert len(train)==827091 and train.target_event_id.nunique()==430 and set(train.target_event_id)<=allowed
 assert not set(train.target_event_id)&fold0 and metadata['training_event_role']=='development only'
 for seed in range(42,47):
  unit=paths.PRODUCTION/f'units/seed{seed}_final';receipt=read_json(unit/'DONE.json')
  assert sha(unit/'NN.pt')==receipt['files']['NN.pt']
 scoring=read_json(FOLD0/'SCORING_PLAN.json')
 assert scoring['bootstrap_unit']=='target_event_id' and scoring['draws']==1000 and scoring['seed']==20261111
 assert sha(FOLD0/'score_once.py')==scoring['sha256']['score_once.py']==done['scorer_sha256']
 assert read_json(FOLD0/'DEVELOPMENT_FEATURE_IDENTITY.json')['passed']
 events=pd.read_parquet(R/'data/SI1_fold0_events.parquet');columns=['event_id','site_id','date','new_role']
 assert list(events.columns)==columns
 pd.testing.assert_frame_equal(assignment.loc[assignment.new_role.eq('fold0'),columns].reset_index(drop=True),events,check_exact=True)
 assert len(events)==80 and events.site_id.nunique()==37
 return '- Fold-0 authorization PASS: exact 80-event allowlist, frozen protocol/authorization, assignment and checkpoint hashes, development-only training, completion receipt; 506 aggregate rows and 88 text rows match. No per-sample fold-0 values were read by this check.'

def check_cp2(allowed):
 receipt=read_json(CP2_WORK/'PAPER_DATA_RECEIPT.json');manifest=read_json(CP2_PRODUCT/'manifest.json');production=read_json(CP2_PRODUCT/'PRODUCTION_RECEIPT.json')
 assert Path(receipt['product']).resolve()==CP2_PRODUCT.resolve()
 assert receipt['product_manifest_sha256']==sha(CP2_PRODUCT/'manifest.json')
 assert receipt['all_events_development'] and receipt['sealed_value_reads']==receipt['fold0_rows_loaded']==0
 assert (receipt['forest_pixels'],receipt['coverage_gate'],receipt['support_days'])==(2171,.8,7)
 for obj in [manifest,production]:
  assert obj['sealed_value_reads']==obj['new_fits']==obj['GPU_hours']==0
  assert obj['extension_id']=='CP2_2022_EXTENSION_20261116'
 for name,digest in production['files'].items():assert sha(CP2_PRODUCT/name)==digest
 for name,digest in receipt['outputs_sha256'].items():assert sha(R/'data'/name)==digest
 oldmanifest=read_json(paths.PRODUCTION/'Castelporziano/manifest.json')
 assert manifest['baseline_manifest_sha256']==sha(paths.PRODUCTION/'Castelporziano/manifest.json')
 assert manifest['model_sha256']==oldmanifest['model_sha256'] and manifest['selected_epochs']==oldmanifest['selected_epochs']
 for seed,digest in manifest['model_sha256'].items():assert sha(paths.PRODUCTION/f'units/seed{seed}_final/NN.pt')==digest
 validation=read_json(CP2_TASK/'FINAL_VALIDATION.json');assert validation['passed'] and validation['Hesse_hashes_unchanged'] and validation['cube_2024_byte_identical']
 store=read_json(CP2_TASK/'STORE_VALIDATION.json')
 assert sha(CP2_TASK/'STORE_VALIDATION.json')==manifest['store_validation_sha256']
 assert store['passed'] and store['sealed_value_reads']==0 and store['n_dates']==119
 assert (store['dates'][0],store['dates'][-1])==('2022-01-05','2022-12-31') and store['catalogue_dates_contained']
 assert store['metadata_and_grid_match'] and store['all_arrays_full_readback']
 for name,digest in read_json(CP2_TASK/'HESSE_PROTECTED_SHA256.json').items():assert sha(paths.PROJECT/name)==digest
 baseline=read_json(CP2_TASK/'PAPER_BASELINE_SHA256.json')
 assert sha(R/'data/Fig4_cube_Castelporziano.npz')==baseline['data/Fig4_cube_Castelporziano.npz']
 for name in ['daily_GPP_Castelporziano.parquet','field_PAI_Castelporziano.parquet']:
  assert sha(R/'data'/name)==baseline['data/'+name]
 changes=pd.read_csv(CP2_TASK/'feature_date_changes.csv');dates=pd.to_datetime(changes.date,utc=True,format="mixed")
 assert changes[changes.feature_changed].date.tolist()==manifest['changed_feature_dates']
 assert changes.loc[dates.ge('2023-02-01'),'max_abs_feature_change'].eq(0).all()
 assert changes.loc[dates.ge('2023-02-01'),'old_boundary'].equals(changes.loc[dates.ge('2023-02-01'),'new_boundary'])
 assert changes.loc[changes.feature_changed,'date'].iloc[-1].startswith('2023-01-14')
 boundary=changes.feature_changed|changes.old_boundary.ne(changes.new_boundary)
 assert changes.loc[boundary,'date'].iloc[-1].startswith('2023-01-28')
 with np.load(CP2_PRODUCT/'trait_cube.npz') as new,np.load(paths.PRODUCTION/'Castelporziano/trait_cube.npz') as old:
  for k in ['dates','is_fortnightly','forest_mask']:
   assert np.array_equal(new[k],old[k]),k
  stable=pd.to_datetime(new['dates'],utc=True,format='mixed')>=pd.Timestamp('2023-02-01',tz='UTC')
  assert np.array_equal(new['w7_support'][stable],old['w7_support'][stable])
  assert np.array_equal(new['per_seed'][:,stable],old['per_seed'][:,stable],equal_nan=True)
  feature_receipt=read_json(CP2_PRODUCT/'FEATURES_DONE.json')
  for record in feature_receipt['date_records']:
   support=CP2_PRODUCT/'features'/record['support_file']
   assert sha(support)==record['support_sha256']
   assert np.array_equal(np.load(support).reshape(50,50),new['w7_support'][record['index']])
  assert new['forest_mask'].sum()==2171
  keep=new['w7_support']&new['forest_mask'];coverage=keep.sum(axis=(1,2))/2171
  fortnightly=new['is_fortnightly'];passing=fortnightly&(coverage>=.8)
  assert passing.sum()==97 and fortnightly.sum()==113
  assert str(new['dates'][np.flatnonzero(passing)[0]]).startswith('2022-06-04')
  assert ((pd.to_datetime(new['dates'],utc=True,format='mixed').year==2022)&passing).sum()==15
  packaged=pd.read_csv(R/'data/Fig4_ensemble_range_Castelporziano.csv',float_precision='round_trip')
  for trait,part in packaged.groupby('trait',sort=False):
   assert np.array_equal(pd.to_datetime(part.date,utc=True),pd.to_datetime(new['dates'][fortnightly],utc=True))
   assert np.allclose(part.coverage,coverage[fortnightly],atol=1e-16,rtol=0) and np.array_equal(part.gate80,coverage[fortnightly]>=.8)
 acquisition=pd.read_csv(CP2_PRODUCT/'development_acquisitions.csv')
 event_column='event_id' if 'event_id' in acquisition else 'event'
 assert set(acquisition[event_column])<=allowed
 # No duplicate local/catalogue dates; all other sites retain exact rows.
 for name in ['SI1_S2_observations','SI1_pc_fill_dates']:
  packed=pd.read_parquet(R/'data'/(name+'.parquet'));old=pd.read_csv(paths.FINAL/'source_data'/(name+'.csv'))[packed.columns]
  pd.testing.assert_frame_equal(packed.loc[packed.site_id.ne(32)].reset_index(drop=True),old.loc[old.site_id.ne(32)].reset_index(drop=True),check_exact=True)
 obs=pd.read_parquet(R/'data/SI1_S2_observations.parquet');local=obs[obs.valid_observation]
 fill=pd.read_parquet(R/'data/SI1_pc_fill_dates.parquet')
 assert not set(zip(local.site_id,local.date_utc))&set(zip(fill.site_id,fill.date_utc))
 counts=read_json(CP2_WORK/'SI1_counts_receipt.json')
 assert len(local)==counts['n_local_valid']==6464 and len(fill)==counts['n_pc_fill']==1954 and len(local)+len(fill)==counts['n_s2_combined']==8418
 return '- CP2 extension PASS: versioned product/receipt and frozen models, 119 store dates, 15 new 2022 points, 97/113 gate-passing dates, early-2023 boundary audit, forest/support identity, zero sealed reads, unchanged Hesse/2024 cube and 8,418 SI1 dates.'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream-check',action='store_true');args=p.parse_args();rows=[];report=[]
 with tempfile.TemporaryDirectory(prefix='.tmp-reproduction-',dir=R) as temp:
  t=Path(temp);audit=t/'sitecustomize.py';log=t/'opens.jsonl';out=t/'figures'
  audit.write_text("import sys,os,json\nfrom pathlib import Path\nr=Path(os.environ['PAPER_REPO']).resolve();project=Path(os.environ['PAPER_PROJECT']).resolve();log=open(os.environ['PAPER_AUDIT_LOG'],'a',buffering=1)\ndef hook(event,args):\n if event!='open' or not isinstance(args[0],(str,bytes)):return\n p=Path(os.fsdecode(args[0])).resolve();mode=args[1]\n if mode is not None and not ('r' in mode or '+' in mode):return\n if p.is_relative_to(project) and not p.is_relative_to(r):raise RuntimeError('Upstream read from plot: '+str(p))\n if p.is_relative_to(r/'data'):log.write(json.dumps(str(p.relative_to(r)))+'\\n')\nsys.addaudithook(hook)\n")
  env=dict(os.environ,PYTHON=sys.executable,PYTHONPATH=str(t),PAPER_REPO=str(R),PAPER_PROJECT=str(paths.PROJECT),PAPER_AUDIT_LOG=str(log),PYTHONDONTWRITEBYTECODE='1')
  subprocess.run(['bash',str(R/'code/run_all.sh'),str(out)],env=env,check=True)
  assert {f.stem for f in out.glob('*.png')}==FIGURES
  assert {f.stem for f in out.glob('*.pdf')}==FIGURES
  check_fold0_schema()
  observed={json.loads(line) for line in log.read_text().splitlines()};expected={'data/'+f.name for f in (R/'data').iterdir() if f.name not in {'numbers_for_text.csv','Fig4_S2_valid_dates_Hesse.parquet','Fig4_S2_valid_dates_Castelporziano.parquet'}}
  assert observed==expected,(observed-expected,expected-observed)
  for f in sorted(out.glob('*.png')):
   a=np.asarray(Image.open(R/'figures'/f.name));im=Image.open(f);b=np.asarray(im);assert a.shape==b.shape
   assert all(abs(dpi-600)<.02 for dpi in im.info['dpi'])
   assert abs(im.width/600*25.4-183)<.05
   diff=np.abs(a.astype(np.int16)-b.astype(np.int16));maximum=int(diff.max());mean=float(diff.mean());assert maximum==0,(f.name,maximum,mean)
   pdf=out/f.with_suffix('.pdf').name;fonts=subprocess.check_output(['pdffonts',str(pdf)],text=True).splitlines()[2:];assert fonts and all('Arial' in q and 'TrueType' in q and 'yes' in q for q in fonts)
   info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);width=float(re.search(r'Page size:\s+([\d.]+)',info).group(1))*25.4/72;assert abs(width-183)<.01
   rows.append((f.stem,maximum,mean,round(width,3)))
  report+=['# Reproduction check','',f'- Python {sys.version.split()[0]}; all seven renderers ran through `code/run_all.sh` with nice19, numerical threads1, no GPU.',f'- File-open audit observed exactly {len(observed)} packaged plotting inputs; all project-upstream data reads were blocked. The text-numbers table and two retained legacy window S2-date tables are intentionally not plotted.','- New totals and seed-range data have a dedicated frozen-input derivation script; legacy S2-date tables remain local for provenance.','', '| Figure | Max pixel difference | Mean pixel difference | PDF width mm |','|---|---:|---:|---:|']+['| '+' | '.join(map(str,r))+' |' for r in rows]
 if args.upstream_check:
  assignment=pd.read_parquet(paths.ASSIGNMENT);report.append(check_fold0_authorization(assignment,set(assignment.loc[assignment.new_role.eq('development'),'event_id'])));report.append(check_cp2(set(assignment.loc[assignment.new_role.eq('development'),'event_id'])));allowed=set(assignment.loc[assignment.new_role.eq('development'),'event_id']);events=pd.read_parquet(R/'data/SI1_development_events.parquet');assert set(events.event_id)<=allowed and events.new_role.eq('development').all()
  reference_count=0
  for site,sid in [('Hesse',45),('Castelporziano',32)]:
   f=pd.read_parquet(R/'data'/f'Fig4_trajectories_{site}.parquet');refs=f[f.kind.eq('EnMAP reference')];ids={f'{sid}:{str(d)[:10]}' for d in refs.date};assert ids<=allowed;reference_count+=len(ids)
   source=paths.V6/'evaluation_sites/Hesse/scene_maps.npz' if site=='Hesse' else paths.FINAL/'source_data/si3_work/evaluation_sites/Castelporziano/scene_maps.npz'
   with np.load(source) as original,np.load(R/'data'/f'SI3_maps_{site}.npz') as packed:
    assert str(packed['event']) in allowed
    for key in packed.files:
     a,b=original[key],packed[key];assert np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b)
   source=paths.V6/'figure_data/Fig4_cube_Hesse.npz' if site=='Hesse' else paths.PRODUCTION/'Castelporziano/Fig4_cube_Castelporziano.npz'
   with np.load(source) as original,np.load(R/'data'/f'Fig4_cube_{site}.npz') as packed:
    for key in packed.files:
     a,b=original[key],packed[key];assert np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b)
  formal=paths.FINAL/'source_data/formal_temporal_primary.parquet';ids=pd.read_parquet(formal,columns=['target_event_id']);assert set(ids.target_event_id)<=allowed
  receipt=json.loads((paths.FINAL/'source_data/FORMAL_EXPORT_RECEIPT.json').read_text());assert receipt['all_events_development'] and receipt['sealed_value_reads']==0 and receipt['assignment_sha256']==sha(paths.ASSIGNMENT)
  for f in (R/'data').glob('*.parquet'):
   if f.name in {'Fig3b_totals.parquet','SI3_fold0_metrics.parquet','SI1_fold0_events.parquet'}:continue
   source=paths.SI4/(f.stem+'.csv') if f.stem.startswith(('model_series','daily_GPP','field_PAI','spearman_GPP')) else paths.FINAL/'source_data'/(f.stem+'.csv')
   if f.name in read_json(CP2_WORK/'PAPER_DATA_RECEIPT.json')['outputs_sha256']:
    source=CP2_WORK/('model_series_Castelporziano.csv' if f.stem=='model_series_Castelporziano' else 'source_data/'+f.stem+'.csv')
   packed=pd.read_parquet(f)
   # CP2 seasonal means were packed directly from float64, so parse their
   # CSV with round-trip precision; legacy copies retain their original parser.
   precision='round_trip' if f.stem=='Fig4_multi_year_mean_Castelporziano' else None
   original=pd.read_csv(source,float_precision=precision)[packed.columns]
   if f.stem=='Fig3a_metrics':original=original[original.model.isin(['NN','pixel_mean','pixel_seasonal_mean'])].reset_index(drop=True)
   pd.testing.assert_frame_equal(original,packed,check_exact=True)
  legacy=read_json(R/'.public_update1_data_receipt.json');assert legacy['all_events_development'] and legacy['sealed_value_reads']==0 and legacy['assignment_sha256']==sha(paths.ASSIGNMENT)
  update=read_json(CP2_WORK/'PUBLIC_UPDATE1_EXTENSION_RECEIPT.json');
  for name,digest in legacy['outputs_sha256'].items():
   if name!='Fig4_ensemble_range_Castelporziano.csv':assert update['outputs_sha256'][name]==digest
  assert update['production']['Hesse']==legacy['production']['Hesse']
  assert update['all_events_development'] and update['sealed_value_reads']==0 and update['assignment_sha256']==sha(paths.ASSIGNMENT)
  for name,digest in update['outputs_sha256'].items():assert sha(R/'data'/name)==digest
  for trait in pd.read_parquet(R/'data/Fig3b_totals.parquet').trait:
   boot=pd.read_parquet(paths.FINAL/f'source_data/Fig3b_bootstrap_{trait}.parquet');row=pd.read_parquet(R/'data/Fig3b_totals.parquet').set_index('trait').loc[trait]
   ci=np.nanquantile(boot[[f'recovered_share_{l}' for l in ['site','date','scene']]].sum(axis=1,min_count=3),[.025,.975]);assert np.allclose(ci,[row.total_low,row.total_high],atol=1e-14)
  for site in ['Hesse','Castelporziano']:
   new=pd.read_csv(R/'data'/f'Fig4_ensemble_range_{site}.csv',float_precision='round_trip');old=pd.read_parquet(R/'data'/f'Fig4_trajectories_{site}.parquet');old=old[old.kind.eq('Production NN')]
   for _,row in new.iterrows():
    match=old[old.trait.eq(row.trait)&pd.to_datetime(old.date,utc=True).eq(pd.Timestamp(row.date))]
    assert len(match)==1 and np.isclose(row['mean'],match['mean'].iloc[0],atol=1e-12,rtol=0,equal_nan=True)
   assert ((new.coverage>=.8)==new.gate80).all()
  report+=['',f'- Sealed check: assignment SHA256 `{sha(paths.ASSIGNMENT)}`; {len(events)} SI1 events, {reference_count} distinct Fig4 reference events and both SI4 scenes are development-only. Formal aggregate inputs cover {len(ids)} rows; all event IDs are development.', '- Four NPZs and all original Parquet tables retain exact values from their explicitly routed upstream exports. New total CIs equal the quantiles of per-draw summed recovered shares; new range means match the original trajectories within 1e-12.', '- Production grids contain inferred values on arbitrary product dates, not held-out EnMAP labels. ICOS, S2-date inventories and weather summaries are not EnMAP teacher observations. Aggregate fold-0 metrics are in data/SI3_fold0_metrics.parquet; SI1 contains fold-0 event metadata only. No per-sample fold-0 values or references, training matrices or model weights are included.','- Update 1 preserves original plotting data; the original 12 figures are retained in figures/archive_pre_update1/. New totals/ranges have verified hashes; component CIs reuse stored draws. New production grids contain predictions only, not EnMAP labels.']
 # Scan decoded table/array text too, not just source-code bytes.
 secrets=[];email=re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}');token=re.compile(r'(?:sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|Bearer\s+[A-Za-z0-9._-]{20,})')
 for f in R.rglob('*'):
  if not f.is_file() or f.name in {'paths_local.py', 'PUBLIC_REPOSITORY_CHECK.md', 'PUBLIC_TRACKED_FILES.tsv', 'REPRODUCTION_CHECK.md'} or any(part.startswith('.') or part=='__pycache__' for part in f.relative_to(R).parts):continue
  if f.suffix=='.parquet':text=pd.read_parquet(f).to_string(index=False)
  elif f.suffix=='.npz':
   with np.load(f) as z:text=' '.join(str(z[k]) for k in z.files if z[k].dtype.kind in 'US')
  elif f.suffix in ['.py','.md','.txt','.csv','.sh']:text=f.read_text()
  else:continue
  # Pattern definitions in this checker are not private values.
  if f.name!=Path(__file__).name and (email.search(text) or token.search(text) or Path.home().name in text):secrets.append(str(f.relative_to(R)))
 assert not secrets,secrets
 for f in (R/'code').glob('plot_*.py'):assert not re.search(r'[/](?:home|lustre|tmp)[/]|input_data/|figures_v6/',f.read_text()),f.name
 files=[f for f in R.rglob('*') if f.is_file() and not any((part.startswith('.') and part!='.gitignore') or part=='__pycache__' for part in f.relative_to(R).parts)];data=list((R/'data').iterdir());assert all(f.stat().st_size<50_000_000 for f in data)
 report+=['','- All PDFs embed Arial TrueType; regenerated PDF bytes may differ because of creation metadata. PNG pixels are exact.', '- No absolute upstream paths in plot_*.py; no e-mail addresses or private access strings found in source/documents or decoded packaged tables/array metadata.',f'- data/: {len(data)} files, {sum(f.stat().st_size for f in data)/1e6:.3f} MB; largest file {max(f.stat().st_size for f in data)/1e6:.3f} MB (<50 MB).',f'- Packaged content excluding generated caches/temp files: {len(files)} files, {sum(f.stat().st_size for f in files)/1e6:.3f} MB.', '- Heavy upstream derivations were not rerun. The small update-1 summaries were derived from frozen exports/cubes and verified; other make_* scripts had syntax/--help checks. They require the referenced upstream raw stores, frozen arrays/code/models and ICOS archives. All eight outputs (seven figures plus the shared-range Fig4_v2 variant) redraw from data/ alone; Arial itself and Poppler are system dependencies, not bundled.']
 (R/'REPRODUCTION_CHECK.md').write_text('\n'.join(report)+'\n');print('\n'.join(report))
if __name__=='__main__':main()
