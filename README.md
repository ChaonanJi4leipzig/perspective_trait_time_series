# EcoSpectra paper figures

Code for Fig. 3, Fig. 4 and SI 1–4 (final 2026-11-13 versions).
Original files were copied, never moved; raw stores, training arrays and model weights are not included.

## Layout and redraw

- Public repository: `code/`, this README, `requirements.txt` and `.gitignore`.
- `data/` (39.14 MB) and `figures/` (16.94 MB) are local-only and git-ignored.
- Figure data and final figures are held by the authors and will be deposited separately: **DOI: TBD**.
- The data deposit includes lossless plot inputs, `numbers_for_text.csv` (293 results) and six captions.

Tested with Python 3.11.6 and `requirements.txt`; install Arial separately (not redistributed).
Poppler (`pdfinfo`, `pdffonts`) is needed only for PDF verification.

```bash
python -m pip install -r requirements.txt
bash code/run_all.sh                         # once data/ is available locally
bash code/run_all.sh redraw_output    # optional output directory
python code/check_reproduction.py            # independent redraw + audit
```

Renderers use relative paths and only `data/`; exports are 183 mm wide, 600 dpi PNG and embedded-font PDF.
`common.py` holds Arial, mathtext, style and 1-mm outer padding. Numeric values/selections match the frozen final figures.
Local checks and the SHA256 manifest are retained by the authors and git-ignored.

## Data-processing pipeline

1. EnMAP L2A spectra were processed with the frozen teacher to obtain QA-masked trait maps.
2. The frozen development assignment excludes fold0/lockbox events before reading teacher values.
3. Sentinel-2 L2A stores supply target-date and surrounding-window features; WorldCover selects forest pixels.
4. The temporal run `cap2048_v3_nested11_20261108` supplies frozen LOYO evaluation predictions and metrics.
5. The spatial run `cap2048_spatial_gsingle_20261112` supplies held-out-block evaluation metrics.
6. `make_formal_export.py`, `make_pixel_baselines.py` and `make_variance_budget.py` prepare Fig. 3 inputs.
7. `cap2048_production_nn_temporal_20261112` supplies all-data forest-window cubes/trajectories; `make_production_*` and `make_s2_dates.py` prepare Fig. 4.
8. `make_si1_inventory.py`, `make_si1_stac_fill.py` and `make_si1_classes.py` combine inventories, Planetary Computer metadata and site classes.
9. `make_scene_*` selects development scenes by coverage and reconstructs them using frozen held-out-year models for SI 3.
10. `make_icos_*` and `make_flux_core.py` process ICOS flux/ancillary products and align daily comparisons for SI 4.
11. `make_figure_data.py` losslessly packs selected existing exports; `make_numbers_for_text.py` copies existing text results without recalculating them.
12. `plot_*.py` render the six figures; captions define masks, baselines, gates, windows and statistics.

The `make_*` bodies reuse original scripts: source hashes are in `code/DERIVATION_SOURCES.md`.
They require upstream inputs and explicit `--run`; only syntax/`--help` were checked, not heavy derivation.
Writes are guarded to a separate working directory and package `data/`; upstream stores remain read-only.
Helper modules are not separate commands. Stage derivations under `.work/derivation/` before packing.

## Upstream locations (referenced, not included)

Configure upstream roots only when running derivations (redrawing needs the local data deposit):

```bash
export ECOSPECTRA_ROOT="${HOME}/path-to-ecospectra" # contains input_data/, analysis/, model/
export ECOSPECTRA_JZ_ROOT="${HOME}/path-to-jz-mirror" # frozen mirror or mounted JZ project
export PAPER_ROOT="${PWD}"                         # repository root; automatic by default
```

`code/paths_local.py`, if present, provides ignored workstation defaults; environment values override them.
Without configuration, upstream roots use explicit NOT_CONFIGURED placeholders, never another user's paths.
All upstream locations are built in `code/paths.py`; `ECOSPECTRA_V6` optionally relocates the archived V6 tree.
`V6` below means `$ECOSPECTRA_ROOT/analysis/0_perspective/figures_v6`.

| Input | Location relative to ECOSPECTRA_ROOT, unless stated |
|---|---|
| Raw EnMAP / Sentinel-2 | `input_data/enmap/traits_region_v2/`; `input_data/s2/region/` |
| Frozen development assignment | `V6/inputs/fold_assignment_v3.parquet` |
| Temporal code, arrays and models | `analysis/0_perspective/figures_add_samples/final_pipeline_20261105/training_2048/harness/final_pipeline_20261105/training_2048/`; feature/model code `V6/code/` |
| Spatial evaluation | `runs/cap2048_spatial_gsingle_20261112/` |
| Production model and products | `V6/production_nn/` |
| WorldCover | `experiments/enmap_v2_worldcover_source_topup_20261022/worldcover_zarr/` |
| Inventories / final export | `analysis/0_perspective/figures_v4/tables/`; `V6/final_figures_20261113/` |
| ICOS archives / extra text results | `V6/flux_compare/`; `V6/hesse_lai_compare/`; `V6/hesse_drought_2022_2026/` |
| JZ frozen mirror (read-only) | `$ECOSPECTRA_JZ_ROOT/training2048_v3_nested11_20261108/harness/final_pipeline_20261105/training_2048/` |

`source_path` URIs in the numbers table use `upstream://V6/…`, resolved against `paths.V6`.
Hesse PAI/LAI numbers retain the existing matched pairs; no new comparison statistic was invented.
The drought verdict is copied as “not supported”. Formal scorecard interpretation remains subject to scientific review.

## Sources, licences and attribution

The separate data deposit contains derived maps/cubes/tables, not original spectral images or training data.

| Source | Licence / citation and attribution |
|---|---|
| EnMAP, DLR | [EnMAP data licence v1.1](https://www.enmap.org/data/resources/EnMAP_Data_License.pdf); value-added trait products: **contains modified EnMAP data © DLR (2022–2026)**. Original imagery is not redistributed. |
| Sentinel-2 / Copernicus | [Sentinel legal notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice); contains modified Copernicus Sentinel data (2022–2026). Metadata via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a). |
| ICOS FLUXNET/ETC | CC BY 4.0; cite the four dataset PIDs below, including their authors and product versions. Derived flux/ancillary tables belong in the separate data deposit. |
| ESA WorldCover | [CC BY 4.0](https://esa-worldcover.org/en/data-access); © ESA WorldCover project 2021 / contains modified Copernicus Sentinel data (2021), processed by the ESA WorldCover consortium. |
| DWD | [CDC terms](https://opendata.dwd.de/climate_environment/CDC/Nutzungsbedingungen_German.pdf), CC BY 4.0; source DWD. Regional summary numbers only. |
| MeteoSwiss | [OGD terms](https://opendatadocs.meteoswiss.ch/general/terms-of-use), CC BY 4.0; source MeteoSwiss. Regional summary numbers only. |

ICOS citations: FR-Hes [flux](https://hdl.handle.net/11676/3FV_P_h69OveJojAgQKrPEZD), [ancillary](https://hdl.handle.net/11676/hnh5IHKw9yDIC3RbGWvvFidz); IT-Cp2 [flux](https://hdl.handle.net/11676/Wc9LMp9RCv59Jkmc2tDR7ilP), [ancillary](https://hdl.handle.net/11676/fFH_pQ3l3LLjQpw2QEGv2hWF).
No sealed teacher values are in the local figure data; arbitrary prediction dates and independent S2/ICOS observations are not teacher labels.
No Git repository was initialized and nothing was uploaded.

## Paper figures update 1

Fig. 3b uses separate component bars; Total and the pixel-seasonal tick sum all three share × max(R², 0) terms (baseline on its defined subset). Component CIs reuse stored draws; Total CI sums each draw before taking percentiles (1,000 site-cluster draws, seed 20261111).
Fig. 4 uses continuous dates, five-network min–max forest-mean bands and fixed Nov–Feb winter shading, not measured error.
`make_update1_data.py --run` creates the small totals/range inputs from frozen exports and cubes; no training or inference. Caption definitions remain with the author-held figures.
