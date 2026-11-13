"""Needs frozen upstream result CSV/JSONs in paths.V6; writes data/numbers_for_text.csv.
Copies published values verbatim; no statistics are recomputed. Runs: temporal,
spatial G_single, production NN, flux seasonality and Hesse drought2022/2026.
"""
import csv,json,argparse
from pathlib import Path
import paths

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=paths.DATA/'numbers_for_text.csv');args=p.parse_args();out=[]
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
