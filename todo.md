# TODO — Chaonan's outstanding tasks before NEE submission

Created 2026-05-25 after Sprint 6.1 (3-reviewer feedback addressed in manuscript).

**Strategic context**: per the synthesis of three role-played reviewers (Editor / Functional Ecologist / RS specialist), if you submit the current manuscript as-is the desk-reject probability is ~55%. If you complete **U1–U6** (Tier 0) before submission, that drops to ~15% and acceptance becomes plausible. Tier 1 (U7–U12) tightens compile cleanliness and relationship-management before sending the pre-submission inquiry.

Companion files:
- `cover_letters/NEE_presubmission_inquiry.md` — inquiry letter Chaonan sends (PC/Miguel/Hannes cc'd)
- `outreach/cavenderbares_invitation.md` — fallback senior-author outreach
- `AUTHOR_NOTES.md` — author block, ORCID, grant ID checklist
- `comments.txt` — latest 3-reviewer feedback (838 lines, integrated into Sprint 6.1)

---

## 🔴 Tier 0 — Deal-breakers (must complete before submission)

The single biggest lever is U1+U2+U4 (delete all placeholder figures). Doing those three drops desk-reject probability from ~55% to ~25%.

### U1 — Regenerate Fig 4 (LES state-space trajectory)
**Why critical**: Fig 4 panel b is the *conceptual punch* of the whole Perspective—the "same greenness, different function" visual claim. Without a publication-quality panel b, the central ecology claim has no visual support.

**Action**:
- Keep **panel a** (multi-trait time series with disturbance / resistance / recovery / legacy stages)
- **Build panel b**: nitrogen–LMA LES plane showing three time points of a community trajectory (pre-event → drought trough → apparent greenness recovery), with the post-recovery position visibly shifted from pre-event despite greenness baseline. Use real EnMAP-retrieved trait values where possible, or illustrative-but-grounded values (not pure schematic).
- **Delete panel c** entirely (the 2022 European drought concrete anchor)—no real data exists, schematic c would reveal the empirical thinness rather than support the claim. Cleaner to have a 2-panel figure.
- Update caption to remove `[Author note: panels b and c to be added...]` and reflect the final a+b structure
- Ping me when panel structure changes — I'll re-tune the caption text

**Owner**: Chaonan + Eya / David Montero (Python figure generation)
**Estimated effort**: 2–3 days

### U2 — Regenerate Fig 5 (trait-space recovery scenarios)
**Why critical**: Fig 5 currently still ships the old monitoring-roadmap PDF despite the caption being completely rewritten. Caption-figure mismatch is an immediate signal of unfinished manuscript.

**Action**:
- Per the new caption: three idealized community trajectories in nitrogen–LMA trait space following a climate extreme:
  - (i) **functional recovery** — return to pre-disturbance trait state within uncertainty
  - (ii) **functional legacy** — greenness recovers but LES position remains shifted
  - (iii) **regime shift / reorganization** — persistent displacement to new trait-space region
- LES-plane schematic; clean, publication-quality
- Remove `[Author note: figure to be regenerated...]` from caption after rebuild

**Owner**: Chaonan + Eya / David Montero
**Estimated effort**: 1–2 days

### U3 — Regenerate Fig 3 / clarify its status
**Why critical**: Fig 3 in main.tex is currently a LaTeX table styled as a figure (the trait-readiness × ecological-process matrix). The old Python-generated model-performance heatmap PDF still lives under `figures/generated/fig3_proof_of_concept_performance.pdf` but is no longer referenced. The Supplementary needs it as Supp Fig S1.

**Action**:
- Decision: either (a) keep the current LaTeX table-as-figure and rename it `Table 2` (since Table 1 already exists), or (b) make a real graphical version of the readiness matrix
- Move the old `fig3_proof_of_concept_performance.pdf` to a `supplementary/` folder as Supp Fig S1
- Caption for Supp Fig S1: model performance heatmap (Linear / RF / MLP / Transformer × 5 traits) supporting the readiness ratings in Table 1
- Remove `[Author note: model-performance heatmaps... moved to Supplementary Fig. S1]` from the current Fig 3 caption (since the move is now actually done)

**Owner**: Chaonan + Eya
**Estimated effort**: 0.5 day

### U4 — Strip all `[Author note: ...]` markers from main.tex
**Why critical**: Every `[Author note]` is a desk-reject signal. NEE editors interpret these as "manuscript not ready for submission."

**Action**: After U1, U2, U3 are done, grep for all remaining `[Author note` instances in `main.tex` and remove them. Known locations after Sprint 6.1:
- Fig 3 caption ("model-performance heatmaps moved to Supplementary Fig. S1")
- Fig 4 caption ("panels b and c to be added in figure regeneration")
- Fig 5 caption ("figure to be regenerated as a three-scenario LES-plane trajectory schematic")
- §5 ("confirm exact R² ranges from companion RSE paper before submission")
- Acknowledgements ("final grant numbers, MSCA project acronym...")
- Data availability ("GitHub release link to be confirmed")

Some require U6 / acknowledgements finalization first.

**Owner**: Chaonan
**Estimated effort**: 0.5 hour after U1–U3 and U6 done

### U5 — Create Supplementary materials
**Why critical**: Main text now references several supplementary items that don't yet exist as files. Submission system requires the supplementary file to be uploaded.

**Action**: Create `supplementary.tex` (or PDF) containing:
- **Supp Fig S1**: model-performance heatmap for the European forest illustration (Linear / RF / MLP / Transformer × 5 traits), with $R^2$, RMSE, nRMSE panels. Referenced from Table 1 caption and §5.
- **Supp Note 1**: expanded six principles for credible functional ecological observables (the dissolved Box 4 content). Referenced from §8 final paragraph.
- **Supp Table S1**: CHIME / SBG mission specifications (spectral / spatial / revisit / SNR / swath). Referenced from §9 first paragraph.
- *Optional but recommended*: **Supp Table S2** with European 2022 drought test-case methods box (number of EnMAP/PRISMA/EMIT scenes, date ranges, sites, forest types, field labels, validation split) — addresses Reviewer 2 Major concern 1. Even a compact 1-page methods statement here would shift Section 5 from "design illustration" to "documented mini-illustration."

**Owner**: Chaonan + Eya
**Estimated effort**: 1–2 days

### U6 — Replace placeholder $R^2$ ranges in §5 with real numbers
**Why critical**: §5 currently reads "in-domain $R^2$ is highest for LAI (approximately 0.6–0.8), intermediate for LMA, nitrogen and lignin (approximately 0.3–0.5), and lowest for chlorophyll (approximately 0.4 in-domain, dropping to <0.1 under cross-site transfer)." These ranges are my estimates based on RS-trait-retrieval literature norms—they need to be replaced with actual values from the companion RSE paper.

**Action**:
- Pull the actual $R^2$ ranges (in-domain and cross-site, for each of LAI / chlorophyll / N / LMA / lignin / leaf EWT) from the companion RSE methods paper
- Update §5 with real numbers
- Remove the `[Author note: confirm exact R² ranges from companion RSE paper before submission]` marker
- If actual cross-site $R^2$ for chlorophyll is *not* <0.1 (i.e. if my estimate was too pessimistic), revise the readiness ratings in Table 1 accordingly to maintain internal consistency

**Owner**: Chaonan (RSE paper data is yours)
**Estimated effort**: 0.5 day

---

## 🟡 Tier 1 — Strongly recommended (before sending pre-submission inquiry)

### U7 — Confirm RSE companion paper submission status
**Why important**: Per the 3-reviewer synthesis, NEE submission should happen with the RSE paper *at least* submitted (preferably "in review"). This gives the Perspective's measurement claims verifiable backing.

**Action**:
- If RSE not yet submitted: submit RSE first, before NEE inquiry
- If RSE submitted but not yet under review: wait until "in review" before NEE inquiry
- If RSE already in review or accepted: proceed with NEE inquiry
- Add RSE manuscript ID / DOI to Perspective citation list (currently "in preparation")

**Owner**: Chaonan + Ciais
**Estimated effort**: Depends on RSE state

### U8 — Verify `ploton2020spatial` citation accuracy
**Why important**: I added this bib entry in §8 to support the leave-site/year/biome-out validation argument. Need to double-check.

**Action**: Verify:
- Ploton P, Mortier F, Réjou-Méchain M, Barbier N, Picard N, Rossi V, Dormann C, et al. (2020)
- "Spatial validation reveals poor predictive performance of large-scale ecological mapping models"
- *Nature Communications* 11: 4540
- DOI: 10.1038/s41467-020-18321-y

Confirmation via Google Scholar or crossref takes 2 minutes.

**Owner**: Chaonan
**Estimated effort**: 5 minutes

### U9 — Verify Donohue 2016 "seventeen dimensions" claim
**Why important**: §6 now claims "Donohue et al. catalogued at least seventeen distinct dimensions of ecological stability." This number comes from the reviewer comment, not my direct read.

**Action**: Pull Donohue I et al. (2016) "Navigating the complexity of ecological stability" *Ecology Letters* 19:1172–1185 and confirm the dimension count. If the number is different (e.g. 15, 19), update §6 accordingly. If they don't list a discrete number, soften to "catalogued many distinct dimensions of ecological stability."

**Owner**: Chaonan
**Estimated effort**: 15 minutes

### U10 — LaTeX clean compile verification
**Why important**: Need to confirm no missing references, no broken cross-references (especially after Sprint 6.1's table/figure renumbering), no overfull boxes.

**Action**:
```bash
cd ~/data/projects/ecospectra/docs/paper0_perspective
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Check the log for:
- `[?]` in PDF (missing bib entries)
- "undefined reference" warnings (Table~\ref / Fig~\ref / Box~\ref)
- "overfull \hbox" warnings (LaTeX overflow)
- Citation order issues (NEE uses numerical citations; verify order matches first-mention order)

**Owner**: Chaonan
**Estimated effort**: 30 min (compile is fast; fixing issues takes longer if any surface)

### U11 — Git push current state to GitHub
**Why important**: Backup + collaboration sync. The remote is `git@github.com:ChaonanJi4leipzig/perspective_paper.git` on branch `main`.

**Action**:
```bash
! cd ~/data/projects/ecospectra/docs/paper0_perspective && git status
```

Then once status looks right:
```bash
! cd ~/data/projects/ecospectra/docs/paper0_perspective && git add -A && git commit -m "Sprint 6.1: 3-reviewer feedback addressed

- §1 reorganized ecology-first (trait ecology canon before sensor)
- Differentiation paragraph (three precursors + we add a fourth) moved to §1
- §3.1 acknowledges trait ecology partly dynamic (no strawman)
- §3.3 trait-state trajectory disambiguated (CWM / distribution / dispersion)
- Table 1: EWT split into leaf-level vs canopy-water; lignin lowered; added retrospective feasibility column
- Fig 3 inline matrix: EWT split; lignin lowered
- §5: real R² ranges placeholder (to confirm from RSE paper); design-illustration framing kept honest
- §6: Donohue 2016 17-dimension acknowledgement; baseline-under-climate-drift caveat
- §7: ITV reframed as one component of community trait distribution (between-species + canopy + ITV)
- §8: leave-out validation; uncertainty quantification candidate methods (conformal/ensemble/RTM-LUT) + calibration requirement
- Box 1: thresholds anchored to indicative ranges; detectability-floor disclaimer
- Abstract: 'how much they grow' → 'how much canopy area they display'
- Fig 5 caption: reframed to trait-space recovery scenarios (functional recovery / legacy / regime shift)
- New bib: ploton2020spatial
- New file: todo.md tracking U1-U12 outstanding tasks" && git push origin main
```

**Owner**: Chaonan
**Estimated effort**: 5 minutes

### U12 — Get Ciais / Mahecha / Feilhauer sign-off on pre-submission inquiry
**Why important**: The inquiry letter (`cover_letters/NEE_presubmission_inquiry.md`) is configured as Chaonan-sent with three senior cc'd. All three must explicitly approve before send—not just be told they will be cc'd.

**Action** (sequential, not parallel — preserve senior-PI courtesy):
1. **Step 1**: Send letter draft to Philippe Ciais alone for review. Wait for explicit "go" + any edits he wants.
2. **Step 2**: After Ciais approves, send to Miguel Mahecha (with the latest version including Ciais's edits if any). Wait for explicit "go."
3. **Step 3**: After Mahecha approves, send to Hannes Feilhauer. Wait for explicit "go."
4. Only **after all three have explicitly agreed** (and ideally after U1–U10 done so the manuscript is submission-ready), send the inquiry to NEE.

Each "step" email can be short — paste the inquiry letter, ask "happy for me to send this with you on cc?", let them respond.

**Owner**: Chaonan
**Estimated effort**: 1 week of elapsed time (waiting for 3 senior PIs); your active time ~30 min total

---

## Suggested execution order

If you have ~2 weeks of focused work:

**Week 1**:
- Day 1: U11 (git push current state) → U10 (compile check) → U9 (Donohue verify) → U8 (Ploton verify) — quick wins, ~2 hours total
- Day 2–3: U6 (real R² ranges) → U5 (Supplementary draft)
- Day 4–5: U3 (Fig 3 status) + start Fig 4/Fig 5 design with Eya/David

**Week 2**:
- Day 1–3: U1 (Fig 4 publication-quality) + U2 (Fig 5 three-scenario LES)
- Day 4: U4 (strip all `[Author note]`) + final compile check
- Day 5: U12 Step 1 (send to Ciais)

**Week 3**:
- U12 Steps 2–3 (Mahecha, Hannes) in parallel with U7 (RSE coordination)
- Send NEE pre-submission inquiry when all approvals + U1–U6 complete

If RSE is far from submission, target the submission within Week 4 instead.

---

## Reference: what I (Claude) completed in Sprint 6.1

For traceability, the manuscript changes I made in response to the 3-reviewer feedback are detailed at the bottom of this folder's chat history. Key files modified:
- `main.tex` (substantial — §1, §3, §5, §6, §7, §8, Box 1, Fig 5 caption, Table 1, Abstract)
- `references.bib` (+`ploton2020spatial`)

No new boxes were added; box count remains 1 (Box 1: falsifiable predictions). Display item count remains 1 Table + 5 Figures (Fig 1, 2, 3-inline, 4, 5) + 1 Box + 2 inline definition boxes.
