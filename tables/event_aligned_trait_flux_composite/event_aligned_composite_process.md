# Event-aligned trait-flux composite process

This document is the preregistered process for the superposed-epoch composite
that aligns drought/heat events across forest flux sites. The figure target is
not a larger-n version of the 2022 flux scatter. It is a separate, event-level
time-alignment check: greenness, reconstructed chlorophyll, and measured GPP
are deseasonalized, aligned to driver-defined event onset, and composited with
site-clustered uncertainty.

## Phase 0: Figure target

Default conclusion tested: after drought/heat onset, NDVI returns toward its
seasonal expectation earlier than reconstructed chlorophyll and measured GPP.
The evidentiary claim is temporal corroboration of a chlorophyll legacy, not
pixel-level inflation of the flux sample size.

## Phase 1: Data assembly

Scope: forest PFT sites with eddy-covariance flux, usable Sentinel-2 dense
reflectance, reconstructed traits, and driver coverage over the target event
years. Default event years are 2019, 2020, and 2022.

Inputs per site:

- Sentinel-2 footprint-matched reflectance/index series, preferably 2017/2018
  through 2023.
- Reconstructed traits: chlorophyll primary; LAI, N, LMA, and EWT optional.
- Gap-filled GPP, using GPP_NT_VUT_REF as the default and GPP_DT as a
  sensitivity where available.
- ERA5-Land or flux-met drivers: root-zone/upper-layer SWC, VPD, and Tmax.

Spatial support default: tower footprint pixels, or a fixed tower-near radius
when footprints are unavailable. The full 30 km tile is reserved for the
pixel-scale decoupling claim, not for flux corroboration.

## Phase 2: Deseasonalization

For each site and variable, build a day-of-year climatology from baseline years
while excluding target event years. Default center is a robust median; default
scale is 1.4826 * MAD with a small positive floor. Smooth the climatology with a
harmonic fit of order 2, with rolling-DOY/LOESS as a sensitivity.

Default anomaly:

```text
z_V(t) = (V(t) - mu_V(DOY[t])) / sigma_V(DOY[t])
```

All plotted variables, including drivers and GPP, must be converted to z
anomalies before event alignment.

## Phase 3: Driver-based event detection

Events are detected only from climate drivers. NDVI, reconstructed traits, and
flux outcomes are never used to choose events.

Default thresholds:

- Drought: z_SWC <= -1.0 for at least 28 consecutive days in the growing season.
- Heat: z_VPD >= +1.5 for at least 14 consecutive days, or Tmax above its 90th
  percentile as a sensitivity.
- Compound event: overlapping drought and heat spells.

Each spell records onset, peak, type, and severity. Default severity is
cumulative positive VPD anomaly during the spell; min z_SWC is a sensitivity.
DoA gate: exclude or flag site-event periods whose reconstructed traits fall
outside the hyperspectral-anchored domain of applicability.

Thresholds are preregistered and run as sensitivity settings:
z_SWC <= -1.0, -1.28, -1.5; z_VPD >= +1.5 and +2.0.

## Phase 4: Event-relative extraction

Default event grid is monthly tau = -6 ... +18 months relative to event onset.
For each site-event-variable, resample deseasonalized z values to this grid.
Mask bins with insufficient observations. Default minimum coverage is at least
one valid biweekly optical observation per monthly bin for S2 variables, and at
least 50 percent daily coverage for flux/driver variables.

## Phase 5: Composite and uncertainty

Events are nested within site. The primary uncertainty method is cluster
bootstrap by site: resample sites with replacement, carrying all events for each
selected site. Report both n_sites and n_site_events.

Mixed-model sensitivity:

```text
z_V ~ ns(tau, df=4) + (1 | site) + (1 | site:event)
```

## Phase 6: Legacy statistics

Recovery time for variable V is the first tau where the composite remains within
|z| < 0.5 for at least two consecutive bins. Primary legacy offset is the
chlorophyll and GPP residual anomaly at the NDVI recovery time.

Secondary statistics:

- Cross-correlation or concordance between chlorophyll and GPP composite curves.
- Severity scaling: per-event legacy_offset ~ severity + (1 | site).

All intervals use site-clustered bootstrap or mixed-model uncertainty.

## Phase 7: Stratification

Default main text is pooled forest events. Stratify by event type
(drought/heat/compound) and biome (Mediterranean/temperate) only when each
stratum has enough sites for a readable interval. Otherwise, stratified panels
belong in the Supplement.

Mechanistic consistency checks are optional: EWT with ET, chlorophyll with GPP.

## Phase 8: Figure layout

Top panel: driver anomalies, SWC and/or VPD, with event onset marked at tau = 0.

Main panel: NDVI, reconstructed chlorophyll, and GPP z-anomaly composites with
site-clustered confidence bands. Mark the zero line, onset, NDVI recovery
crossing, and chlorophyll/GPP legacy offsets at that crossing.

Figure annotation must include n_sites, n_site_events, event-type composition,
event thresholds, recovery band, and bootstrap type.

## Phase 9: Reporting rules

Always state that flux is site-level. Pixel data can show the decoupling
distribution, but cannot increase the effective flux n. Event detection is
driver-based and non-circular. If the composite legacy is weak or the confidence
interval crosses zero, report it as suggestive rather than consistent.

## Pseudocode skeleton

```python
for site in sites:
    for V in [NDVI, chl, LAI, N, LMA, EWT, GPP, SWC, VPD]:
        mu, sigma = robust_doy_climatology(
            V,
            baseline_years,
            smoother="harmonic",
            exclude=event_years,
        )
        z[V] = (V - mu[doy]) / sigma[doy]

events = []
for site in sites:
    spells = detect_spells(
        z_swc_threshold=-1.0,
        drought_min_days=28,
        z_vpd_threshold=1.5,
        heat_min_days=14,
        season="growing",
    )
    for spell in spells:
        if doa_mask(site, spell.period) == "in_domain":
            events.append(spell)

tau = monthly_grid(-6, +18)
for event in events:
    for V in variables:
        A[V][event] = resample_to_tau(z[V], around=event.onset, grid=tau)

for V in variables:
    composite[V], ci[V] = cluster_bootstrap_by_site(A[V], B=10000)

rt = {V: first_stable_tau(abs(composite[V]) < 0.5, persist=2)
      for V in ["NDVI", "Chlorophyll", "GPP"]}
legacy_offset = {
    "Chlorophyll": composite["Chlorophyll"][rt["NDVI"]],
    "GPP": composite["GPP"][rt["NDVI"]],
}
```

