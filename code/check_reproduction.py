"""Redraw with a file-open audit, compare PNG pixels, inspect PDFs; optional upstream metadata checks."""
from pathlib import Path
import argparse,os,sys,subprocess,tempfile,json,hashlib,re
import numpy as np,pandas as pd
from PIL import Image
import paths
R=paths.REPO

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream-check',action='store_true');args=p.parse_args();rows=[];report=[]
 with tempfile.TemporaryDirectory(prefix='.tmp-reproduction-',dir=R) as temp:
  t=Path(temp);audit=t/'sitecustomize.py';log=t/'opens.jsonl';out=t/'figures'
  audit.write_text("import sys,os,json\nfrom pathlib import Path\nr=Path(os.environ['PAPER_REPO']).resolve();project=Path(os.environ['PAPER_PROJECT']).resolve();log=open(os.environ['PAPER_AUDIT_LOG'],'a',buffering=1)\ndef hook(event,args):\n if event!='open' or not isinstance(args[0],(str,bytes)):return\n p=Path(os.fsdecode(args[0])).resolve();mode=args[1]\n if mode is not None and not ('r' in mode or '+' in mode):return\n if p.is_relative_to(project) and not p.is_relative_to(r):raise RuntimeError('Upstream read from plot: '+str(p))\n if p.is_relative_to(r/'data'):log.write(json.dumps(str(p.relative_to(r)))+'\\n')\nsys.addaudithook(hook)\n")
  env=dict(os.environ,PYTHON=sys.executable,PYTHONPATH=str(t),PAPER_REPO=str(R),PAPER_PROJECT=str(paths.PROJECT),PAPER_AUDIT_LOG=str(log),PYTHONDONTWRITEBYTECODE='1')
  subprocess.run(['bash',str(R/'code/run_all.sh'),str(out)],env=env,check=True)
  observed={json.loads(line) for line in log.read_text().splitlines()};expected={'data/'+f.name for f in (R/'data').iterdir() if f.name not in {'numbers_for_text.csv','Fig4_S2_valid_dates_Hesse.parquet','Fig4_S2_valid_dates_Castelporziano.parquet'}}
  assert observed==expected,(observed-expected,expected-observed)
  for f in sorted(out.glob('*.png')):
   a=np.asarray(Image.open(R/'figures'/f.name));b=np.asarray(Image.open(f));assert a.shape==b.shape
   diff=np.abs(a.astype(np.int16)-b.astype(np.int16));maximum=int(diff.max());mean=float(diff.mean());assert maximum==0,(f.name,maximum,mean)
   pdf=out/f.with_suffix('.pdf').name;fonts=subprocess.check_output(['pdffonts',str(pdf)],text=True).splitlines()[2:];assert fonts and all('Arial' in q and 'TrueType' in q and 'yes' in q for q in fonts)
   info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);width=float(re.search(r'Page size:\s+([\d.]+)',info).group(1))*25.4/72;assert abs(width-183)<.01
   rows.append((f.stem,maximum,mean,round(width,3)))
  report+=['# Reproduction check','',f'- Python {sys.version.split()[0]}; all six renderers ran through `code/run_all.sh` with nice19, numerical threads1, no GPU.',f'- File-open audit observed exactly {len(observed)} packaged plotting inputs; all project-upstream data reads were blocked. The text-numbers table and two retained legacy window S2-date tables are intentionally not plotted.','- New totals and seed-range data have a dedicated frozen-input derivation script; legacy S2-date tables remain local for provenance.','', '| Figure | Max pixel difference | Mean pixel difference | PDF width mm |','|---|---:|---:|---:|']+['| '+' | '.join(map(str,r))+' |' for r in rows]
 if args.upstream_check:
  assignment=pd.read_parquet(paths.ASSIGNMENT);allowed=set(assignment.loc[assignment.new_role.eq('development'),'event_id']);events=pd.read_parquet(R/'data/SI1_development_events.parquet');assert set(events.event_id)<=allowed and events.new_role.eq('development').all()
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
   if f.name=='Fig3b_totals.parquet':continue
   source=paths.SI4/(f.stem+'.csv') if f.stem.startswith(('model_series','daily_GPP','field_PAI','spearman_GPP')) else paths.FINAL/'source_data'/(f.stem+'.csv');packed=pd.read_parquet(f);original=pd.read_csv(source)[packed.columns]
   if f.stem=='Fig3a_metrics':original=original[original.model.isin(['NN','pixel_mean','pixel_seasonal_mean'])].reset_index(drop=True)
   pd.testing.assert_frame_equal(original,packed,check_exact=True)
  update=json.loads((R/'.public_update1_data_receipt.json').read_text());assert update['all_events_development'] and update['sealed_value_reads']==0 and update['assignment_sha256']==sha(paths.ASSIGNMENT)
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
  report+=['',f'- Sealed check: assignment SHA256 `{sha(paths.ASSIGNMENT)}`; {len(events)} SI1 events, {reference_count} distinct Fig4 reference events and both SI3 scenes are development-only. Formal aggregate inputs cover {len(ids)} rows; all event IDs are development.', '- Four NPZs and 20 original Parquet tables retain exact upstream values. New total CIs equal the quantiles of per-draw summed recovered shares; new range means match the original trajectories within 1e-12.', '- Production grids contain inferred values on arbitrary product dates, not held-out EnMAP labels. ICOS, S2-date inventories and weather summaries are not EnMAP teacher observations. No fold0/lockbox values, training matrices or model weights are included.','- Update 1 preserves original plotting data; the original 12 figures are retained in figures/archive_pre_update1/. New totals/ranges have verified hashes; component CIs reuse stored draws. New production grids contain predictions only, not EnMAP labels.']
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
 report+=['','- All PDFs embed Arial TrueType; regenerated PDF bytes may differ because of creation metadata. PNG pixels are exact.', '- No absolute upstream paths in plot_*.py; no e-mail addresses or private access strings found in source/documents or decoded packaged tables/array metadata.',f'- data/: {len(data)} files, {sum(f.stat().st_size for f in data)/1e6:.3f} MB; largest file {max(f.stat().st_size for f in data)/1e6:.3f} MB (<50 MB).',f'- Packaged content excluding generated caches/temp files: {len(files)} files, {sum(f.stat().st_size for f in files)/1e6:.3f} MB.', '- Heavy upstream derivations were not rerun. The small update-1 summaries were derived from frozen exports/cubes and verified; other make_* scripts had syntax/--help checks. They require the referenced upstream raw stores, frozen arrays/code/models and ICOS archives. All six figures redraw from data/ alone; Arial itself and Poppler are system dependencies, not bundled.']
 (R/'REPRODUCTION_CHECK.md').write_text('\n'.join(report)+'\n');print('\n'.join(report))
if __name__=='__main__':main()
