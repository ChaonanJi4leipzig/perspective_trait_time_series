# Literature Review And Manuscript Actions, 2026-05-30

Scope: high-quality and Nature-family literature most relevant to the current Figure 1-5 package and the paper's core claim that greenness recovery is not functional recovery.

## One-line diagnosis

The manuscript's central claim is literature-defensible if framed narrowly: greenness/SIF/NIRv are strong integrative indicators of canopy activity and GPP, but trait trajectories add a functional-state layer for leaf economics, nitrogen, water status, stress strategy, uncertainty and ecological transferability.

## Priority literature map

| Paper | Quality signal | Main support for this paper | Manuscript action |
|---|---:|---|---|
| Jetz et al. 2016, Nature Plants | Nature-family, foundational | Plant functional diversity can be monitored from space. | Keep as Fig. 1/intro anchor. |
| Skidmore et al. 2021, Nature Ecology & Evolution | Nature-family, EBV framing | Prioritizes biodiversity variables observable from space. | Keep as EBV context; clarify that our contribution is the state variable, not another metric list. |
| Cavender-Bares et al. 2022, Nature Ecology & Evolution | Nature-family Perspective | Remote sensing must be integrated with field ecology/evolution. | Keep as the nearest conceptual predecessor; emphasize our dynamic trait-trajectory extension. |
| Schweiger et al. 2018, Nature Ecology & Evolution | Nature-family, high relevance | Spectral diversity integrates functional/phylogenetic biodiversity and predicts ecosystem function. | Use for dynamic functional diversity and Fig. 2/Fig. 5 logic. |
| Schweiger & Laliberte 2022, Nature Communications | Nature-family | Imaging spectroscopy captures beta-diversity across biomes. | Keep for distribution/turnover claims; do not overstate alpha diversity. |
| Wright et al. 2004, Nature | Very high-cited canon | Leaf economics spectrum state space. | Keep as core trait-space foundation. |
| Diaz et al. 2016, Nature | Very high-cited canon | Global plant form-function spectrum. | Keep as trait-space foundation. |
| Joswig et al. 2022, Nature Ecology & Evolution | Nature-family, trait-environment | Climate/soil controls on global trait spectrum. | Keep for trait-environment dynamics. |
| Smith et al. 2022, Nature Climate Change | Nature-family | Greenness-derived resilience metrics can diagnose vegetation resilience but remain index-based. | Keep in resilience section; contrast with trait-state endpoints. |
| Yan et al. 2025, Nature Plants | Nature-family, recent | Global forest recovery from mortality events by satellites. | Keep as recovery-monitoring context; frame as greenness/canopy recovery, not trait recovery. |
| van der Woude et al. 2023, Nature Communications | Nature-family, direct 2022 Europe evidence | 2022 European drought/heat reduced summer forest carbon uptake by 56-62 TgC over drought area. | Added to intro, Box 1 note and 2022 drought section. |
| Forzieri et al. 2021, Nature Communications | Nature-family, high relevance | European forest vulnerability depends on climate/forest context and disturbance type. | Corrected citation metadata and used for disturbance vulnerability. |
| Li et al. 2024, Nature Reviews Earth & Environment | Nature-family, current greenness context | 2023 global greenness was very high despite extreme-event impacts. | Added to 2022 drought section to support greenness-recovery context. |
| Ploton et al. 2020, Nature Communications | Nature-family, validation warning | Spatial validation can reveal poor transferability hidden by random CV. | Kept and strengthened in guardrails. |
| Meyer & Pebesma 2022, Nature Communications | Nature-family, validation warning | Naive random validation can give biased, overly optimistic assessment for ecological maps. | Added to guardrails. |
| Roberts et al. 2017, Ecography | Highly cited methods | Cross-validation must respect temporal/spatial/hierarchical structure. | Added to guardrails. |

## Figure-specific recommendations

| Figure | Current role | Literature-backed improvement | Risk if ignored |
|---|---|---|---|
| Fig. 1 | Conceptual state-space transition | Add visual language that separates "index archive", "hyperspectral anchor" and "trait-state trajectory"; cite Jetz, Skidmore, Cavender-Bares, Wright, Diaz. | Reviewers may see it as a generic remote-sensing schematic. |
| Fig. 2 | Trait-readiness / measurement contract | Make direct/indirect/exploratory trait status visually explicit; tie to spectral biology and EBV literature. | It may look like an internal taxonomy rather than a community-useful contract. |
| Fig. 3 | Empirical model readiness | Keep the split-specific panels. Add caption/text warning that random or naive splits overestimate ecological transferability. | Experts will question whether R2 is inflated by spatial/temporal autocorrelation. |
| Fig. 4 | Real-data drought anchor | This is the strongest empirical figure. Keep the 2022-2023 aligned time axes and explicitly connect the ERA5/VPD/soil-water driver, spectral-index recovery and trait displacement. | Without stronger literature citation, the trait legacy claim can sound model-driven rather than ecology-driven. |
| Fig. 5 | Roadmap / scenario figure | Anchor the scenarios in resilience theory plus Smith/Yan remote-sensing resilience papers. | It may read as a speculative ending rather than a testable research programme. |

## Manuscript edits made

| File | Change | Reason |
|---|---|---|
| `main.tex` | Rewrote the opening evidence chain around the 2022 drought, carbon uptake loss and functional legacy risk. | Avoids overclaiming "ICP/FLUXNET prove trait legacy"; makes the claim narrower and stronger. |
| `main.tex` | Strengthened Fig. 3 caption with split/transferability warning. | Matches current experiment results and Nature-style validation expectations. |
| `main.tex` | Added van der Woude 2023, Li 2024, Tai 2023, Roberts 2017, Meyer & Pebesma 2022 citations in the relevant sections. | Better Nature-family and methods support. |
| `main.tex` | Revised Data and code availability to say empirical panels are illustrative analyses, not final benchmark claims. | Reduces reviewer risk because current Fig. 3 uses internal results. |
| `references.bib` | Corrected the Forzieri European-forest vulnerability entry from an incorrect 2022/volume 13 metadata record to the Nature Communications 2021 record. | Prevents a wrong DOI/citation. |
| `references.bib` | Added four new reference entries: `vanderwoude2023carbon`, `li2024greenness`, `roberts2017crossvalidation`, `meyer2022maps`. | Supports 2022 drought, 2023 greenness and validation guardrails. |

## Citation advice for next manuscript pass

| Claim type | Best citation set | How to phrase |
|---|---|---|
| "Greenness is useful but incomplete" | Zeng 2022; Sun 2017; Badgley 2017; Smith 2022 | Say greenness/SIF/NIRv are strong integrative proxies, not wrong variables. |
| "Traits define functional state space" | Wright 2004; Diaz 2016; Reich 2014; Joswig 2022 | Use these as ecological foundations, not remote-sensing citations. |
| "Remote sensing can monitor biodiversity/function" | Jetz 2016; Skidmore 2021; Cavender-Bares 2022; Schweiger 2018; Schweiger 2022 | Say this paper specifies the dynamic state variable those efforts should produce. |
| "2022 Europe is a strong test case" | van der Woude 2023; Bastos 2023; Toreti/JRC; Rakovec/Schumacher | Separate event severity, carbon-cycle impact and greenness-index behavior. |
| "Validation must be split-aware" | Roberts 2017; Ploton 2020; Meyer & Pebesma 2022 | This is a guardrail, not a side note. Use it to justify Fig. 3. |

## Remaining review risks

| Risk | Severity | Suggested next action |
|---|---:|---|
| Fig. 4 trait trajectories are model-derived and may be challenged as artefacts. | High | Add uncertainty bands or a supplementary panel showing prediction-domain flags and sample support. |
| The "mid-2023 recovery" statement needs a site-specific quantitative date for Le Bray/Bilos. | High | Extract NDVI/NDWI/MSI recovery dates from the Fig. 4 source table and include them in caption or text. |
| The paper may still sound like it claims operational trait monitoring is solved. | Medium | Keep the "design illustration, not operational benchmark" language and make uncertainty/domain masks visible in figures. |
| Trait names such as nitrogen/chlorophyll/LMA can imply leaf-level biology even when inferred at canopy/pixel scale. | Medium | Use "canopy-inferred" or "hyperspectral-derived" where precision matters. |
| Literature around tree-ring/GPP decoupling and drought legacy could further strengthen the legacy argument. | Medium | Add one focused carbon-growth decoupling reference in the next pass if the Discussion expands. |
