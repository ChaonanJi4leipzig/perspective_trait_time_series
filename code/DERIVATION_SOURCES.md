Original derivation bodies reused; only location bindings/import names/CLI/write protection were adapted. No training code or model weights are copied.

| Copy | Upstream V6-relative source | Original SHA256 |
|---|---|---|
| make_formal_export | final_figures_20261113/scripts/export_formal_jz.py | f1b3be3cbeedddc2827da9caefe25582df4d527f9e3d19ea2320379d5d581e84 |
| make_pixel_baselines | final_figures_20261113/scripts/pixel_baselines.py | 75181eb62e5f104304eec6dc30d416d1d3674054967bd2ca4e9a354b7d5ec35c |
| make_variance_budget | final_figures_20261113/scripts/analyze_fig3.py | d24bb6efdd408c37c7466d2427b5e58bd0fe3ae3d7ab3637d130bd6aa698731e |
| make_metrics_core | final_figures_20261113/scripts/hierarchy_metrics.py | 723c2e71e0b6feb767465e0e688e0da9bf08bed7c8bf1d436f277eec7dc8f71e |
| make_metrics_fast | final_figures_20261113/scripts/hierarchy_fast.py | 611855e95206646eb1adf12edd7a8ab6a6e043fb6046b8445b7421e12e92549e |
| make_s2_dates | final_figures_20261113/scripts/s2_sources.py | e988c0952f8b8130852342ac80f9d652859691de347d8888abd1b263805e2455 |
| make_scene_selection | final_figures_20261113/scripts/select_si3_scene.py | 7b3ba68c58a2cada4065b41aee3140a2f8e822ec3aeae24b439c1e5aa1246e85 |
| make_scene_reconstruction | final_figures_20261113/scripts/infer_si3_scene.py | c0ff2993874e2c61abab50c8d1a739857b1f48d4675ecefa78775f21b400370b |
| make_scene_collect | final_figures_20261113/scripts/collect_si3_jz.py | 00e43f8854bca3c0043f9d0f386c8d239d3f3fb49ae87c7e5141c6a158e550b8 |
| make_site_support | figure_sites.py | a3bc88d0c0d4cc22a9558efe6fea00f465647ce295c0f06094ba6da3bf1eb633 |
| make_si1_stac_fill | final_figures_20261113/si1_pc/build_pc_fill.py | 52d919f90b26a7c2d58b04df11052f2f3ddee1f1d9dbe59d8624f1759549ab8b |
| make_si1_classes | final_figures_20261113/si1_pc/journal/build_classes.py | 33bc0106334d724458a3b16fab4aba22febcee4c05b5196296a04cd4d9d43bd9 |
| make_si1_inventory | inventories.py | a3975600c68027c4764bfce5e7fb2a79f2a590b1d66f4657c2f8b470d4a31bd0 |
| make_production_features | production_nn/prepare_features.py | 06359a907aff6df96e6b12ba24ddddcc418fae2bcc871cae3de817cf99eba2f4 |
| make_production_common | production_nn/common.py | 8999bec41f289e54d3e1b965c911c14eabbee358ec26628272cf9ac7773680f4 |
| make_production_window | production_nn/new_sites_20261113/run_window.py | 60674023f52dba30ec84fc9f76127d9e3a4c88d1dee62c1328cfe46051354f38 |
| make_production_outputs | production_nn/new_sites_20261113/assemble_fig4_inputs.py | a8f64ae9cd7f430941daf65f2a9f17d3ead79324ce401463ccc7008d7926749c |
| make_soil_reference | production_nn/new_sites_20261113/extract_soil_jz.py | 6c59f0ccfcec21c34497ff9cc8374c47af0f72010ba1d16f0f1f9cc788279100 |
| make_flux_core | flux_compare/tests/run_seasonality_tests.py | 50432f89f00c8e1c7cc71ab41cd48236cc5877c37cda911f4fb06df4164f9199 |
| make_icos_align | final_figures_20261113/si4/build_si4_data.py | c191e7c58ce794db451f5c72ec1391ceb345124d35a2ee00e1a9999690991465 |
| make_icos_hesse | flux_compare/plot_flux_compare.py | 43f8637888bebe8a6c7a37b0f49bea8ee71a82de2a657ec33884bff16f2f3e62 |
| make_icos_cp2 | flux_compare/work_v3/process_three.py | 1e7aa4e5d5d83e9afe6f75e029f894a3655a886dbe43594ca5f075b2cc41e849 |
| make_formal_metrics | prepare_panel_a.py | aedb6077e40b9dc84832e991208b8dfaccc90d9faea9b382f647319051bbd395 |
| make_production_hesse | adapt_production_fig4.py | 589b849e939724462b84956de27b7d6d1c3904bee7c07b8e7c3eee9be3bd34ce |

Update 1: `make_update1_data.py` reuses `make_metrics_core.evaluate` with the original pixel-seasonal rows, aggregates the stored per-draw recovered components for Total CIs, and applies the forest/support means from `make_icos_align.py` to the frozen production per-seed cubes. It neither draws a new bootstrap nor reads teacher stores. Input/output SHA256 receipts are local-only.
