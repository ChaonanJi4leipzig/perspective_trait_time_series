"""Only upstream-location configuration; no original data/models are copied or written."""
from pathlib import Path
import os
# Roots may be supplied by the environment or an ignored local configuration.
# An explicit environment value takes precedence over a local default.
import importlib.util
_CODE_ROOT = Path(__file__).resolve().parent
_local = None
_local_file = _CODE_ROOT / 'paths_local.py'
if _local_file.is_file():
    _spec = importlib.util.spec_from_file_location('_paper_local_paths', _local_file)
    _local = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_local)

def _root(name, default):
    value = os.environ.get(name, getattr(_local, name, default))
    return Path(value).expanduser().resolve()

REPO = _root('PAPER_ROOT', _CODE_ROOT.parent)
PROJECT = _root('ECOSPECTRA_ROOT', REPO / 'ECOSPECTRA_ROOT_NOT_CONFIGURED')
JZ_ROOT = _root('ECOSPECTRA_JZ_ROOT', REPO / 'JZ_ROOT_NOT_CONFIGURED')
V6=Path(os.environ.get('ECOSPECTRA_V6',PROJECT/'analysis/0_perspective/figures_v6')).resolve()
FINAL=V6/'final_figures_20261113'
FORMAL=PROJECT/'analysis/0_perspective/figures_add_samples/final_pipeline_20261105/training_2048/harness/final_pipeline_20261105/training_2048'
SPATIAL=PROJECT/'runs/cap2048_spatial_gsingle_20261112'
PRODUCTION=V6/'production_nn'
MODEL_CODE=V6/'code'
RAW=PROJECT/'input_data'
WORLD_COVER=PROJECT/'experiments/enmap_v2_worldcover_source_topup_20261022/worldcover_zarr'
ASSIGNMENT=V6/'inputs/fold_assignment_v3.parquet'
FLUX=V6/'flux_compare'
SI1=FINAL/'si1_pc'
SI4=FINAL/'si4'
EARLIER_CLASSES=PROJECT/'analysis/1_method/figures_0921/fig_2/enmap_study_sites_land_cover.csv'
JZ_FORMAL=JZ_ROOT/'training2048_v3_nested11_20261108/harness/final_pipeline_20261105/training_2048'
WORK=Path(os.environ.get('PAPER_DERIVATION_WORK',REPO/'.work/derivation')).resolve()
DERIVED=WORK/'source_data'
DATA=REPO/'data'
RUN_TEMPORAL='cap2048_v3_nested11_20261108'
RUN_SPATIAL='cap2048_spatial_gsingle_20261112'
RUN_PRODUCTION='cap2048_production_nn_temporal_20261112'
ICOS_PIDS={
 'FR-Hes flux':'11676/3FV_P_h69OveJojAgQKrPEZD',
 'IT-Cp2 flux':'11676/Wc9LMp9RCv59Jkmc2tDR7ilP',
 'FR-Hes ancillary':'11676/hnh5IHKw9yDIC3RbGWvvFidz',
 'IT-Cp2 ancillary':'11676/fFH_pQ3l3LLjQpw2QEGv2hWF'}
STAC_API='https://planetarycomputer.microsoft.com/api/stac/v1'
def teacher_store(site):return RAW/f'enmap/traits_region_v2/site_{site}_enmap_traits_region_30km_30m.zarr'
def s2_store(site,year):return RAW/f's2/region/site_{site}/site_{site}_s2_region_{year}.zarr'
def worldcover_store(site):return WORLD_COVER/f'site_{site}_worldcover2021_30m.zarr'

INVENTORY_SOURCE=V6.parent/'figures_v4/tables'
V4_CONFIG=V6.parent/'figures_v4/config.json'

def derived_input(name):
    """Prefer explicitly rebuilt staging, otherwise read the frozen figure-data export."""
    candidate=DERIVED/name
    return candidate if candidate.exists() else FINAL/'source_data'/name
ICOS_PRECEDENCE=FLUX/'work_v3/SOURCE_PRECEDENCE.json'
