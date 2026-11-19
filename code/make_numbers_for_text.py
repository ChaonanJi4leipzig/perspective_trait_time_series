"""Needs frozen upstream result CSV/JSONs in paths.V6; writes data/numbers_for_text.csv.
Copies published values verbatim; no statistics are recomputed. Runs: temporal,
spatial G_single, production NN, flux seasonality and Hesse drought2022/2026.
"""
import csv,json,argparse
from pathlib import Path
import paths

FIELDS=['quantity','site/trait','value','unit','n','CI_low','CI_high','source_path','note']

def fold0_rows():
 """Copy the authorised aggregate export, preserving its decimal strings."""
 source='fold0_test_20261116/metrics.csv'
 with (paths.V6/source).open() as f:metrics=list(csv.DictReader(f))
 out=[]
 for trait in ['LAI','LMA','Chl','Car','EWT','N','C','NSC','Lignin','Fibre','Cellulose']:
  for model in ['NN','pixel_mean','pixel_seasonal_mean']:
   statistic='seed_mean' if model=='NN' else 'baseline'
   population='primary' if model=='NN' else model+'_defined'
   for metric in ['r2','nrmse_P95_P05']:
    matches=[r for r in metrics if (r['trait'],r['model'],r['population'],r['statistic'],r['metric'])==(trait,model,population,statistic,metric)]
    assert len(matches)==1
    r=matches[0]
    out.append(dict(quantity=f'fold0_{model}_{metric}',**{'site/trait':trait},value=r['estimate'],unit='dimensionless',n=r['n'],CI_low=r['ci_low'],CI_high=r['ci_high'],source_path='upstream://V6/'+source,note=f'{statistic}; {population}; event-cluster bootstrap1000 seed20261111; historical-exposure caveat'))
    if metric=='r2' and model!='NN':coverage=r
   if model!='NN':
    primary=[r for r in metrics if (r['trait'],r['model'],r['population'],r['statistic'],r['metric'])==(trait,'NN','primary','seed_mean','r2')];assert len(primary)==1
    out.append(dict(quantity=f'fold0_{model}_coverage',**{'site/trait':trait},value=coverage['coverage'],unit='fraction',n=primary[0]['n'],CI_low='',CI_high='',source_path='upstream://V6/'+source,note='fraction of primary NN observations with defined development history'))
 with (paths.V6/'fold0_test_20261116/numbers_for_text_additions.csv').open() as f:expected=list(csv.DictReader(f))
 assert len(out)==88 and out==expected
 return out

def append_fold0(output):
 """Idempotently append aggregates without changing existing CP2/SI1 rows."""
 with (paths.DATA/'numbers_for_text.csv').open() as f:
  reader=csv.DictReader(f);assert reader.fieldnames==FIELDS;before=list(reader)
 additions=fold0_rows();old=[r for r in before if r['quantity'].startswith('fold0_')]
 assert not old or old==additions
 out=[r for r in before if not r['quantity'].startswith('fold0_')]+additions
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(out)
 print('Copied fold0 aggregates:',len(additions),'; total rows:',len(out))

def update_cp2(work, output):
 """Replace only rows derived from CP2/updated SI1; keep other text numbers."""
 import pandas as pd
 with (paths.DATA/'numbers_for_text.csv').open() as f:out=list(csv.DictReader(f))
 def frame(name):return pd.read_csv(work/name)
 def replace(quantity,identity,note,value,n='',lo='',hi='',source=''):
  matches=[r for r in out if r['quantity']==quantity and r['site/trait']==identity and r['note']==note]
  assert len(matches)==1,(quantity,identity,note,len(matches))
  r=matches[0];r.update(value=str(value),n=str(n),CI_low=str(lo),CI_high=str(hi),source_path='derivation://CP2_2022_EXTENSION_20261116/'+source)
 for r in frame('spearman_GPP.csv').itertuples():
  if r.site=='Castelporziano' and r.region=='Fig4_window':replace('SI4_GPP_rho',r.site+'/'+r.quantity,r.season,r.spearman_rho,r.n,source='spearman_GPP.csv')
 for r in frame('seasonal_timing_summary.csv').itertuples():
  replace('SI4_'+r.metric+'_mean_delta',r.site+'/'+r.quantity,r.scope+'; trait minus GPP',r.mean_delta_days,r.n,source='seasonal_timing_summary.csv')
 for r in frame('cross_site_summary.csv').itertuples():
  identity=r.site+'/'+r.trait
  replace('anomaly_rho',identity,'reference='+r.anomaly_reference,r.anomaly_rho,r.anomaly_n,r.anomaly_low,r.anomaly_high,'cross_site_summary.csv')
  replace('Jun_Aug_rho',identity,'raw Jun–Aug',r.JJA_raw_rho,r.JJA_n,r.JJA_raw_low,r.JJA_raw_high,'cross_site_summary.csv')
  for phase in ['SOS','EOS']:
   matches=[x for x in out if x['quantity']==phase+'_mean_delta' and x['site/trait']==identity];assert len(matches)==1
   oldnote=matches[0]['note'];replace(phase+'_mean_delta',identity,oldnote,getattr(r,phase+'_mean_delta'),getattr(r,phase+'_n'),source='cross_site_summary.csv')
   matches[0]['note']=str(getattr(r,phase+'_flags'))+'; trait minus GPP'
 receipt=json.loads((work/'SI1_counts_receipt.json').read_text())
 for key,value in receipt.items():
  matches=[r for r in out if r['quantity']=='SI1_'+key];assert len(matches)==1
  matches[0].update(value=str(value),source_path='derivation://CP2_2022_EXTENSION_20261116/SI1_counts_receipt.json')
 for r in frame('seasonal_medians.csv').itertuples():
  # This descriptive statistic had no row in the old text-number export.
  matches=[x for x in out if x['quantity']=='Fig4_seasonal_median' and x['site/trait']==r.site+'/'+r.trait and x['note']==r.season]
  assert len(matches)<=1
  row=dict(quantity='Fig4_seasonal_median',**{'site/trait':r.site+'/'+r.trait},value=str(r.median),unit=r.unit,n=str(r.n),CI_low='',CI_high='',source_path='derivation://CP2_2022_EXTENSION_20261116/seasonal_medians.csv',note=r.season)
  if matches:matches[0].update(row)
  else:out.append(row)
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['quantity','site/trait','value','unit','n','CI_low','CI_high','source_path','note']);w.writeheader();w.writerows(out)
 print('Updated CP2 text numbers:',len(out))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=paths.DATA/'numbers_for_text.csv');p.add_argument('--run',action='store_true');p.add_argument('--cp2-work-dir',type=Path);p.add_argument('--fold0-audit',action='store_true',help='Copy 88 authorised aggregate rows, preserving all existing numbers');args=p.parse_args();out=[]
 if args.fold0_audit:
  if not args.run or args.cp2_work_dir:p.error('--fold0-audit requires --run and cannot combine with --cp2-work-dir')
  append_fold0(args.output.resolve());return
 if args.cp2_work_dir:
  if not args.run:p.error('--cp2-work-dir requires --run')
  import sys
  from make_runtime import cli
  sys.argv=[sys.argv[0],'--run','--work-dir',str(args.cp2_work_dir)]
  cli(__doc__)
  update_cp2(args.cp2_work_dir.resolve(),args.output.resolve());return
 def rows(relative):
  with (paths.V6/relative).open() as f:return list(csv.DictReader(f))
 def add(quantity,identity,value,unit='',n='',lo='',hi='',source='',note=''):
  out.append(dict(quantity=quantity,**{'site/trait':identity},value=value,unit=unit,n=n,CI_low=lo,CI_high=hi,source_path='upstream://V6/'+source,note=note))
 source='final_figures_20261113/source_data/Fig3a_metrics.csv'
 for r in rows(source):
  if r['model']=='NN':add(r['scope']+'_NN_'+r['metric'],r['trait'],r['mean'],'dimensionless',r['n'],source=source,note='5-seed mean; seed range is not a CI')
 source='final_figures_20261113/source_data/Fig3a_pixel_baselines.csv'
 for r in rows(source):
  for metric in ['r2','nrmse_P95_P05','coverage']:add(r['model']+'_'+metric,r['trait'],r[metric],'fraction' if metric=='coverage' else 'dimensionless',r['n_defined'],source=source,note='temporal defined subset; coverage denominator='+r['n_full'])
 source='final_figures_20261113/source_data/Fig3b_budget.csv'
 for r in rows(source):add('variance_recovered_'+r['level'],r['trait'],r['recovered'],'fraction',r['n_pixels'],r['recovered_low'],r['recovered_high'],source,'temporal;1000 site-cluster draws;5-seed mean predictions')
 source='final_figures_20261113/source_data/SI3_variance_compression.csv'
 for r in rows(source):
  for key in ['qq_slope','SD_ratio']:add(key,r['site']+'/'+r['trait'],r[key],'dimensionless',r['n_pixels'],source=source,note=r['event_id'])
 source='final_figures_20261113/si4/spearman_GPP.csv'
 for r in rows(source):
  if r['region']=='Fig4_window':add('SI4_GPP_rho',r['site']+'/'+r['quantity'],r['spearman_rho'],'dimensionless',r['n'],source=source,note=r['season'])
 source='final_figures_20261113/si4/seasonal_timing_summary.csv'
 for r in rows(source):add('SI4_'+r['metric']+'_mean_delta',r['site']+'/'+r['quantity'],r['mean_delta_days'],'day',r['n'],source=source,note=r['scope']+'; trait minus GPP')
 source='flux_compare/tests/cross_site_summary.csv'
 for r in rows(source):
  identity=r['site']+'/'+r['trait']
  add('anomaly_rho',identity,r['anomaly_rho'],'dimensionless',r['anomaly_n'],r['anomaly_low'],r['anomaly_high'],source,'reference='+r['anomaly_reference'])
  add('Jun_Aug_rho',identity,r['JJA_raw_rho'],'dimensionless',r['JJA_n'],r['JJA_raw_low'],r['JJA_raw_high'],source,'raw Jun–Aug')
  for phase in ['SOS','EOS']:add(phase+'_mean_delta',identity,r[phase+'_mean_delta'],'day',r[phase+'_n'],source=source,note=r[phase+'_flags']+'; trait minus GPP')
 source='hesse_lai_compare/pairs_within7d.csv'
 for r in rows(source):
  for key in ['PAI_field','LAI_model']:add('Hesse_PAI_LAI_'+key,'Hesse',r[key],'m² m⁻²',source=source,note='field='+r['field_date']+';model='+r['model_date']+';lag_days='+r['lag_days'])
 source='hesse_drought_2022_2026/drought_yearly_metrics.csv'
 for r in rows(source):
  if r['year'] in ['2022','2026'] and r['scope']=='Jun_Sep_available':
   for key,unit in [('drought_onset','date'),('first_episode_days','day')]:add(key,'Hesse/'+r['year'],r[key],unit,r['SWC5_n_days'],source=source,note='available-period;2026 right-censored')
 source='hesse_drought_2022_2026/table_key_Chl.csv'
 for r in rows(source):
  if r['model']=='LOYO NN':add('out_of_year_Chl_anomaly','Hesse/'+r['held_out_year'],r['Chl_anomaly'],'µg cm⁻²',source=source,note=r['date'])
 source='hesse_drought_2022_2026/dwd_matched_summary.csv'
 for r in rows(source):
  for year in ['2022','2026']:add('DWD_median_DOY','Hesse region/'+year,r['median'+year],'day of year',r['n'],source=source,note='phase='+r['phase']+';matched stations')
 source='hesse_drought_2022_2026/table_swiss_phenology.csv'
 for r in rows(source):
  if r['year'] in ['2022','2026']:add('MeteoSwiss_median_DOY','Hesse region/'+r['year'],r['median_doy'],'day of year',r['n_stations'],source=source,note='phase='+r['phase']+';sparse/incomplete2026')
 source='hesse_drought_2022_2026/FINAL_AUDIT.json';add('drought_verdict','Hesse 2022/2026',json.loads((paths.V6/source).read_text())['verdict'],source=source,note='existing preregistered conclusion; no new inference')
 source='final_figures_20261113/si1_pc/journal/render_receipt.json';r=json.loads((paths.V6/source).read_text())
 for key in ['n_enmap','n_local_valid','n_pc_fill','n_s2_combined','median_enmap_per_site_year','median_s2_per_site_year']:add('SI1_'+key,'all sites',r[key],'count',source=source,note='local validity and PC catalogue cloud criteria differ')
 args.output.parent.mkdir(parents=True,exist_ok=True)
 with args.output.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['quantity','site/trait','value','unit','n','CI_low','CI_high','source_path','note']);w.writeheader();w.writerows(out)
 print('Copied numbers:',len(out))
if __name__=='__main__':main()
