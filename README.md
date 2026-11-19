# From trait maps to trait time series: a physiological state observable for vegetation from space

**Ji, C. et al. (in review).** *[journal]*. [DOI to be added on publication].

This study asks whether sparse imaging-spectroscopy observations can be turned into trait time series that describe vegetation's physiological state. EnMAP provides detailed trait maps on relatively few dates, whereas Sentinel-2 provides much denser observations. We apply the multi-trait model of Cherif et al. (2023) to EnMAP spectra to obtain reference trait maps, then use these model-derived references (the “teacher”) to train neural networks that reconstruct 11 traits at 30 m from Sentinel-2 time windows and static soil context. Withheld-year and withheld-block evaluations test reconstruction of the reference traits; forest trajectories demonstrate how sporadic maps can be extended to regularly spaced dates. This repository accompanies the study with figure renderers, input-derivation scripts and provenance documentation.

Public code: <https://github.com/ChaonanJi4leipzig/perspective_trait_time_series>.
Figure inputs and final figures: **[data DOI to be added]**.

## The study in brief

### Sites, observations and traits

The study covers 41 European sites (forest, grassland, cropland, wetland, shrubland/savanna and one lake station), 37 of which have a validated Integrated Carbon Observation System (ICOS)/FLUXNET eddy-covariance tower identifier. Each site is a 30 × 30 km area on a 30 m grid. EnMAP Level-2A scenes span June 2022–September 2026: 919 site-associated scenes (870 distinct products), grouped into 518 acquisition events. An event is one site on one UTC calendar date.

The events are split into 438 development events (777 scenes) and 80 held-out events (142 scenes; "fold 0"). Dates linked by shared EnMAP products or datatakes stay on the same side of the split. Training uses reference pixels of all permitted land-cover types (not only forest) that have valid Sentinel-2 data nearby in time.

The targets, in the display order used in Fig. 3 (`code/common.py`), are:

| Trait | Figure abbreviation or label |
|---|---|
| Leaf area index | LAI |
| Leaf mass per area | LMA |
| Leaf chlorophyll content | Chl |
| Leaf carotenoid content | Car |
| Equivalent water thickness | EWT |
| Leaf nitrogen content | N |
| Leaf carbon content | C |
| Non-structural carbohydrate content | NSC |
| Leaf lignin content | Lignin |
| Leaf fibre content | Fibre |
| Leaf cellulose content | Cellulose |

### Model

The input has 804 features: a base branch of 436 (346 Sentinel-2-derived features within ±7 days and 90 static, site-level SoilGrids descriptors), 137 features for the selected observation nearest the target date, and 231 broader-context features from 8–21 days on either side. The centre observation need not fall on the target date. Soil context is shared within each site; site identifiers and coordinates are not predictors.

Three shared hidden layers each have 128 neurons, GELU activation and 10% dropout. A shared 64-neuron head predicts all 11 standardised traits. A training-only auxiliary branch predicts 16 hyperspectral principal components from 224 bands; the loss is trait mean-squared error plus 0.3 times principal-component mean-squared error. Hyperspectral input is not required at inference. Five independently fitted networks use the same training data; Supplementary Fig. 2 shows the architecture.

### Evaluation design

- **Temporal reconstruction:** leave-one-year-out evaluation over 2022–2026, excluding training dates within 43 days of the withheld year to prevent overlap of input windows. The use of observations on either side of a target date makes this retrospective reconstruction.
- **Spatial reconstruction:** withheld 600 × 600 m blocks within sites, with training pixels excluded within a 600 m Chebyshev distance of withheld evaluation-pixel centres. This tests within-site spatial transfer.
- **Pixel-history baselines:** the mean of development references at the same 30 m pixel in other years (pixel mean), and the same restricted to the same meteorological season (pixel seasonal mean). For the fold-0 audit, the history uses all development data except the same site and date. Pixels without history get no substitute value; each baseline is scored on the subset where it is defined, and network scores on the same subsets are given in the source data.
- **Variance decomposition:** differences between site means, between dates within sites, and between pixels within scenes are assessed separately after averaging network predictions. Fig. 3b Total intervals sum the recovered components within each stored bootstrap draw before taking percentiles.
- **Held-out audit:** the frozen all-development-data networks used for Fig. 4 were evaluated once on fold 0 under a protocol fixed before value access (Supplementary Fig. 3).

See the paper and source data for accuracy estimates, uncertainty intervals and population-specific comparisons.

### Forest demonstration

Fortnightly trajectories at FR-Hes Hesse and IT-Cp2 Castelporziano cover June 2022–September 2026. Fixed 50 × 50-pixel windows at 30 m cover 1.5 × 1.5 km. Means use supported WorldCover tree-cover pixels; a date is retained only when at least 80% of the fixed forest pixels have Sentinel-2 support within ±7 days. Unsupported dates remain gaps.

Fig. 4 shows LMA, Chl, EWT and N, together with a multi-year seasonal mean. Supplementary Fig. 5 compares reconstructed Chl, canopy chlorophyll content (CCC) and LAI with ICOS gross primary production (GPP) and ground overstory plant area index (PAI). CCC is computed per pixel and network before averaging. GPP is aligned as a centred 15-day mean sampled fortnightly, requiring at least 8 valid days. PAI was measured by SunScan at Hesse and digital hemispherical photography at Castelporziano.

The seasonal mean interpolates only between valid points at most 16 days apart and requires at least two contributing years; it is displayed only within supported trajectory segments. EnMAP reference means require at least 80% reference coverage of the forest window. Full averaging and masking rules accompany the deposited captions.

## Evidence boundaries

- References are model-derived EnMAP trait maps, not independent field measurements.
- Fold 0 is a held-out audit, not an untouched lockbox: 35 events were reassigned from development on 5 November 2026, and 32 overlap an earlier evaluation of a superseded model. They were excluded from fitting, selection and tuning of the current networks.
- Fig. 4 EnMAP references are in-sample for the model fitted to all development data. Production trajectories are predictions on arbitrary target dates, not reference labels.
- GPP/PAI comparisons are consistency checks, not validation. Flux footprints, ground plots and reconstruction windows differ; GPP also responds to environmental forcing and shares seasonality with traits. PAI includes woody area and is not directly equivalent to leaf-only LAI.
- Fig. 4 bands show the minimum–maximum of five individual network forest means, not confidence intervals. November–February shading is a seasonal cue, not measured error; winter values are retained.

## Figures in the paper

Input names below are relative to `data/`. Paired files refer to Hesse and Castelporziano. Full definitions accompany the final figures in the separate deposit.

| Figure | What it shows | Renderer script | Packaged input files in `data/` |
|---|---|---|---|
| Fig. 3 | Withheld-year/block accuracy, pixel-history baselines and three-level reference-variance recovery. | `code/plot_fig3.py` | `Fig3a_metrics.parquet`, `Fig3a_pixel_baselines.parquet`, `Fig3b_budget.parquet`, `Fig3b_totals.parquet` |
| Fig. 4 | Forest-trait cubes and continuous fortnightly trajectories at both sites. | `code/plot_fig4.py` | `Fig4_cube_Hesse.npz`, `Fig4_cube_Castelporziano.npz`, `Fig4_trajectories_Hesse.parquet`, `Fig4_trajectories_Castelporziano.parquet`, `Fig4_ensemble_range_Hesse.csv`, `Fig4_ensemble_range_Castelporziano.csv`, `Fig4_multi_year_mean_Hesse.parquet`, `Fig4_multi_year_mean_Castelporziano.parquet` |
| Fig4_v2 (alternative) | Fig. 4 with one shared range per trait across both sites' colour scales and trajectory axes. | `code/plot_fig4.py` | Same inputs as Fig. 4. |
| Supplementary Fig. 1 | Development/held-out EnMAP events and Sentinel-2 temporal coverage, grouped by ecosystem. | `code/plot_si1.py` | `SI1_development_events.parquet`, `SI1_fold0_events.parquet`, `SI1_S2_observations.parquet`, `SI1_pc_fill_dates.parquet`, `SI1_site_classes_filled.parquet` |
| Supplementary Fig. 2 | Shared neural-network architecture and training-only auxiliary branch. | `code/plot_si2.py` | None; schematic defined in code. |
| Supplementary Fig. 3 | Aggregate fold-0 audit of the frozen production networks and pixel-history baselines. | `code/plot_si3.py` | `SI3_fold0_metrics.parquet` |
| Supplementary Fig. 4 | Held-out-year reference/reconstruction maps and quantile–quantile comparisons of spatial variability. | `code/plot_si4.py` | `SI3_maps_Hesse.npz`, `SI3_maps_Castelporziano.npz` |
| Supplementary Fig. 5 | Consistency of reconstructed canopy traits with tower GPP and ground PAI. | `code/plot_si5.py` | `model_series_Hesse.parquet`, `model_series_Castelporziano.parquet`, `daily_GPP_Hesse.parquet`, `daily_GPP_Castelporziano.parquet`, `field_PAI_Hesse.parquet`, `field_PAI_Castelporziano.parquet`, `spearman_GPP.parquet` |

Historical internal names differ from paper numbering: `SI3_maps_Hesse.npz` and `SI3_maps_Castelporziano.npz` feed Supplementary Fig. 4, and some flux derivations retain `SI4` names. `Fig4_v2` rounds the containing range outwards to multiples of 5, 2, 0.5 and 0.02 for LMA, Chl, EWT and N, respectively, including both sites' colour limits and all plotted trajectory values.

## Repository layout

- `code/`: renderers, shared styling and paths, the reproduction checker, and adapted input-derivation scripts. `code/DERIVATION_SOURCES.md` records original source locations and SHA-256 hashes.
- `README.md`, `LICENSE`, `requirements-redraw.txt`, `requirements.txt` and `.gitignore`: documentation, code licence, dependency stacks and exclusions.
- `data/`: lossless plotting inputs (about 39.20 MB) and `numbers_for_text.csv` with 397 existing text-result rows and provenance; distributed through the separate deposit and git-ignored here.
- `figures/`: final PNG/PDF figures and captions, also supplied separately and git-ignored. Local historical archives are excluded from git.
- Raw EnMAP/Sentinel-2 stores, training arrays and model weights are not included because of product-licence conditions and/or size. The source scripts describe dependencies but do not bundle a standalone training workflow.
- Local audit reports, hash manifests, workstation defaults, caches and working directories are git-ignored.

## Redrawing the figures

Run commands from the repository root after obtaining the separate deposit and placing its inputs in `data/`. The redraw environment is **Python 3.13.9**, `requirements-redraw.txt`, **FreeType 2.13.3**, and system **Arial**. Arial is not redistributed. Poppler (`pdfinfo`, `pdffonts`) is needed only for the checker's PDF checks.

```bash
python -m pip install -r requirements-redraw.txt
bash code/run_all.sh
```

Figures are written to `figures/` by default; pass another directory as the first argument (`bash code/run_all.sh <output-dir>`). A single figure can be drawn with its renderer, for example `python code/plot_fig3.py --output-dir <output-dir>`.

`code/run_all.sh` runs the seven renderers (`plot_fig4.py` writes both Fig. 4 variants) with single-threaded numerical libraries and no GPU. Exports are **183 mm wide**, **600 dpi PNG**, and **PDF with embedded Arial TrueType fonts**. `code/common.py` defines the shared style and 1 mm outer padding. Renderers read only the packaged inputs in `data/`, through repository-relative paths.

To verify a redraw against the deposited final figures:

```bash
python code/check_reproduction.py
```

The checker redraws all figures into a temporary directory, records which files are opened, checks the expected input set, and compares PNG pixels, dimensions, resolution and PDF fonts with the final figures in `figures/`. It writes a local audit report. PDF bytes can differ because of creation metadata.

`python code/check_reproduction.py --upstream-check` is an author-side provenance check that needs the upstream archive (see [Upstream configuration](#upstream-configuration)). It verifies that only development events feed the reference displays, that packed inputs equal the upstream exports, and the provenance of the fold-0 audit and the Castelporziano 2022 extension, using metadata and aggregates only.

Author-side checks report exact PNG reproduction for all seven figures and the alternative Fig. 4, correct dimensions and embedded fonts; the upstream provenance checks also passed. The scope of these checks is redraw and consistency with frozen products; heavy upstream derivations have not been rerun end to end for this package.

`requirements.txt` retains the earlier Python 3.11.6 derivation environment, including upstream-processing dependencies. Its Matplotlib 3.8.4 / FreeType 2.14.1 rendering stack produces known text-pixel differences in Fig. 4 and the supplementary maps. Use the current redraw stack for exact reproduction.

## From raw data to figure inputs

The following stages condense the upstream workflow. The frozen run identifiers identify provenance; redrawing uses their packaged outputs and does not execute these stages.

### Reference traits and Sentinel-2 features

EnMAP Level-2A spectra were processed with the Cherif et al. (2023) reference model and EnMAP quality masks. The frozen development assignment excludes held-out and excluded events from development derivations before reference values are read. Quality-screened, radiometrically harmonised Sentinel-2 Level-2A stores provide target-window and surrounding-window features, with static SoilGrids context. WorldCover supplies land-cover exclusions and the fixed forest masks used in the demonstrations.

### Evaluation runs and example maps

The temporal run `cap2048_v3_nested11_20261108` supplies frozen leave-one-year-out predictions and metrics; `cap2048_spatial_gsingle_20261112` supplies withheld-block metrics. `code/make_formal_export.py`, `code/make_formal_metrics.py`, `code/make_pixel_baselines.py` and `code/make_variance_budget.py` prepare Fig. 3 inputs. `code/make_update1_data.py` derives Total intervals and ensemble-range summaries from frozen exports and cubes, reusing stored bootstrap draws.

The once-only `fold0_test_production_nn_20261116` audit supplies aggregate metrics for Supplementary Fig. 3; its protocol preceded value access. Packaging copies the existing aggregate export rather than repeating evaluation. `code/make_figure_data.py` copies only fold-0 event metadata for Supplementary Fig. 1.

`code/make_scene_selection.py` selects development scenes by valid coverage before inference; `code/make_scene_reconstruction.py` reconstructs them using frozen networks trained without the selected year. `code/make_scene_collect.py` handles frozen upstream panel extraction. These products feed Supplementary Fig. 4.

### Production trajectories

`cap2048_production_nn_temporal_20261112` supplies the networks fitted to all eligible development data and the forest-window products. `code/make_production_features.py`, `code/make_production_window.py`, `code/make_production_outputs.py`, `code/make_production_hesse.py` and `code/make_s2_dates.py` cover feature preparation, frozen-model application and output assembly. `code/make_soil_reference.py` extracts the frozen site-level soil vectors.

The versioned `CP2_2022_EXTENSION_20261116` product adds the validated Castelporziano Sentinel-2 store for 2022 and applies the same frozen production networks. `code/make_cp2_extension.py` stages the affected Fig. 4 and Supplementary Figs 1 and 5 inputs from that product. This extension adds no model fit or accuracy evidence. Its upstream product directory is `production_nn/Castelporziano_2022ext_20261116/` within the archived analysis tree defined below.

### Inventories and flux data

`code/make_si1_inventory.py`, `code/make_si1_stac_fill.py` and `code/make_si1_classes.py` combine local observations, Planetary Computer catalogue metadata and site classifications. The Sentinel-2 inventory combines dates with valid local coverage of at least 50% of vegetation pixels and catalogue-filled dates outside local-store coverage with scene cloud cover ≤50%; these are different quality criteria. Catalogue filling accesses metadata rather than imagery assets.

`code/make_icos_hesse.py`, `code/make_icos_cp2.py`, `code/make_icos_align.py` and `code/make_flux_core.py` process the ICOS flux/ancillary products and align daily comparisons. The invalid Hesse PAI aggregate of 9 July 2025 is excluded, as in the original processing, and ground comparisons use the existing matched pairs. `numbers_for_text.csv` also records the pre-registered Hesse 2022/2026 drought comparison, whose verdict is "not supported".

### Packaging and execution safeguards

`code/make_figure_data.py` losslessly packs selected columns and arrays from existing exports. `code/make_numbers_for_text.py` copies existing text results without recalculating statistics. Its `--run --fold0-audit` mode appends the 88 aggregate fold-0 audit rows while keeping the Castelporziano and inventory results already in the table.

Derivation entry points require upstream inputs; use explicit `--run` after reviewing their arguments and locations. Most use `code/make_runtime.py`, whose write guard restricts writes to the selected work directory and package `data/`, rejecting upstream writes. Small summary/copy utilities use explicit output destinations instead; inspect those destinations before execution. Helper modules are not separate commands. The default staging directory is `.work/derivation/`; `--work-dir` is available to entry points using the shared runtime.

Only syntax/`--help` checks were performed for the heavy derivations when the package was assembled; they were not rerun end to end. The small Total/range summaries were previously derived and verified from frozen inputs. Provenance is documented in `code/DERIVATION_SOURCES.md`.

## Upstream configuration

Redrawing needs the deposited inputs only. Derivations require separately obtained upstream stores, arrays, frozen model code/weights and ICOS archives. Configure these roots for your own installation through the variables read by `code/paths.py`:

| Environment variable | Meaning |
|---|---|
| `ECOSPECTRA_ROOT` | EcoSpectra project root containing `input_data/`, `analysis/` and `model/`. |
| `ECOSPECTRA_JZ_ROOT` | Root of a read-only mirror of frozen training arrays on a high-performance computing (HPC) archive. The variable name is retained for compatibility. |
| `PAPER_ROOT` | This repository's root; inferred automatically by default. |
| `ECOSPECTRA_V6` | Optional relocation of the archived analysis tree, normally `analysis/0_perspective/figures_v6/` under the project root. |
| `ECOSPECTRA_CP2_PRODUCTION` | Versioned Castelporziano production-product directory; set to the extension below when deriving the packaged extended series. |
| `PAPER_DERIVATION_WORK` | Optional staging directory; defaults to `.work/derivation/` under the repository root. |

For the versioned Castelporziano product, after setting the project root:

```bash
export ECOSPECTRA_CP2_PRODUCTION="${ECOSPECTRA_ROOT}/analysis/0_perspective/figures_v6/production_nn/Castelporziano_2022ext_20261116"
```

An optional ignored `code/paths_local.py` provides workstation defaults; environment values override them. Unconfigured upstream roots use explicit `NOT_CONFIGURED` placeholders. Staging directories are created on execution. The table lists expected external dependencies, not files supplied by this repository; their availability depends on your installation. It uses **analysis archive** for the directory selected by `ECOSPECTRA_V6` (called `V6` in the path module); this is a historical directory name.

| Input | Location relative to `ECOSPECTRA_ROOT`, unless stated |
|---|---|
| EnMAP reference / Sentinel-2 stores | `input_data/enmap/traits_region_v2/`; `input_data/s2/region/` |
| Frozen development assignment | Analysis archive: `inputs/fold_assignment_v3.parquet` |
| Temporal code, arrays and models | `analysis/0_perspective/figures_add_samples/final_pipeline_20261105/training_2048/harness/final_pipeline_20261105/training_2048/`; shared feature/model code in analysis archive: `code/` |
| Spatial evaluation | `runs/cap2048_spatial_gsingle_20261112/` |
| Production model/products | Analysis archive: `production_nn/`; extended product: `production_nn/Castelporziano_2022ext_20261116/` |
| Frozen fold-0 audit | `runs/fold0_test_production_nn_20261116/`; aggregate exports in analysis archive: `fold0_test_20261116/` |
| WorldCover | `experiments/enmap_v2_worldcover_source_topup_20261022/worldcover_zarr/` |
| Inventories / frozen figure exports | `analysis/0_perspective/figures_v4/tables/`; analysis archive: `final_figures_20261113/` |
| ICOS / additional text results | Analysis archive: `flux_compare/`, `hesse_lai_compare/`, `hesse_drought_2022_2026/` |
| Frozen training-array mirror | Relative to `ECOSPECTRA_JZ_ROOT`: `training2048_v3_nested11_20261108/harness/final_pipeline_20261105/training_2048/` |

In `numbers_for_text.csv`, `source_path` values beginning `upstream://V6/` resolve against that analysis archive. Values beginning `derivation://CP2_2022_EXTENSION_20261116/` resolve against the local `.work/cp2_2022_extension_20261116/` exports. These are provenance identifiers, not web download links.

## Data availability and sources/licences

Figure inputs, derived maps/cubes/tables, final figures and captions will be deposited separately at **[data DOI to be added]**. Original spectral imagery and training data are not part of that deposit. Obtain EnMAP from DLR, Sentinel-2 through Copernicus/Planetary Computer, and ancillary SoilGrids and WorldCover products from their providers; ICOS datasets are identified below.

| Source | Licence / citation and attribution |
|---|---|
| EnMAP, DLR | [EnMAP data licence v1.1](https://www.enmap.org/data/resources/EnMAP_Data_License.pdf); value-added trait products: **contains modified EnMAP data © DLR (2022–2026)**. Original imagery is not redistributed. |
| Sentinel-2 / Copernicus | [Sentinel legal notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice); contains modified Copernicus Sentinel data (2022–2026). Metadata via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a). |
| ICOS FLUXNET/ETC | CC BY 4.0; cite the four dataset PIDs below, including their authors and product versions. Derived flux/ancillary tables belong in the separate data deposit. |
| ESA WorldCover | [CC BY 4.0](https://esa-worldcover.org/en/data-access); © ESA WorldCover project 2021 / contains modified Copernicus Sentinel data (2021), processed by the ESA WorldCover consortium. |
| DWD | [CDC terms](https://opendata.dwd.de/climate_environment/CDC/Nutzungsbedingungen_German.pdf), CC BY 4.0; source DWD. Regional summary numbers only. |
| MeteoSwiss | [OGD terms](https://opendatadocs.meteoswiss.ch/general/terms-of-use), CC BY 4.0; source MeteoSwiss. Regional summary numbers only. |

ICOS dataset PIDs (cite the dataset authors and product versions):

- FR-Hes flux: <https://hdl.handle.net/11676/3FV_P_h69OveJojAgQKrPEZD>
- FR-Hes ancillary: <https://hdl.handle.net/11676/hnh5IHKw9yDIC3RbGWvvFidz>
- IT-Cp2 flux: <https://hdl.handle.net/11676/Wc9LMp9RCv59Jkmc2tDR7ilP>
- IT-Cp2 ancillary: <https://hdl.handle.net/11676/fFH_pQ3l3LLjQpw2QEGv2hWF>

Aggregate fold-0 evaluation metrics are in `data/SI3_fold0_metrics.parquet`; Supplementary Fig. 1 includes fold-0 event metadata only. No per-sample fold-0 values or references are packaged. Arbitrary prediction dates and independent Sentinel-2/ICOS observations are not EnMAP reference labels.

## How to cite

Ji, C. et al. (in review). *From trait maps to trait time series: a physiological state observable for vegetation from space*. [journal]. [DOI to be added on publication].

Please also cite the figure-data deposit, **[data DOI to be added]**, and the original data/model sources used in your work. Replace the placeholders below with publication metadata when available:

```bibtex
@article{Ji_trait_time_series,
  author  = {Ji, C. and others},
  title   = {From trait maps to trait time series: a physiological state observable for vegetation from space},
  journal = {[journal]},
  year    = {[year to be added on publication]},
  volume  = {[volume]},
  doi     = {[DOI to be added on publication]},
  note    = {In review}
}
```

## Licence

Repository code is released under the **MIT licence**; see [LICENSE](LICENSE). Copyright (c) 2026 Chaonan Ji and co-authors. Derived data and third-party products remain under their own licences and attribution requirements; the MIT code licence does not relicense them.

## Contact

For scientific questions, data access or correspondence, contact the corresponding author identified in the paper.
