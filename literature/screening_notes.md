# Literature screening record (OpenAlex, run 2026-10-07)

Script: `lit_01_openalex_search.py`. Query strings and hit counts are in `search_log.csv`.
Matching: title + abstract; years 2018–2026. **OpenAlex only.** Scopus and Web of Science were not available, so the search must be repeated there before submission.

## Counts

| Query | Purpose | Hits |
|---|---|---|
| Q1 | (LST / SUHI / UHI) AND (Pakistani cities) AND (ML / XAI terms) | 117 |
| Q2 | (LST / SUHI / UHI) AND (Islamabad OR Rawalpindi), any method | 50 |
| Q3 | (LST OR UHI) AND (SHAP / Shapley / explainable / interpretable) | 2,059 |
| Q3b | (LST OR UHI) AND (SHAP OR Shapley) | 662 total: 2020: 1 · 2021: 3 · 2022: 6 · 2023: 13 · 2024: 46 · 2025: 205 · 2026 (to 7 Oct): 388 |

Q1 + Q2 gave 156 unique records. Screening was keyword-assisted on title and abstract, with manual reading of every record that mentioned ML or prediction together with LST/UHI (50 records). 37 Q1 records had no abstract in OpenAlex, so they were screened on title only.

## Records that bear directly on the gap

| Study (verified via Crossref) | City | ML / XAI | Relevance |
|---|---|---|---|
| Baqa, M.F. et al. 2026, *GIScience & Remote Sensing* 63:2640263. doi:10.1080/15481603.2026.2640263 | Urban cores of Karachi, Lahore, Faisalabad, Gujranwala | Stepwise MLR + **XGBoost–SHAP**; SDGSAT-1 high-res LST; 11 2D/3D morphology metrics | **Closest competitor.** SHAP-based LST attribution already exists for Pakistan. All four cities are flat plains; Islamabad–Rawalpindi is not included; validation scheme unknown (abstract only) |
| Tang, N. et al. 2025, *IEEE JSTARS* 18:9279–9296. doi:10.1109/jstars.2025.3554529 | Multan | GRU deep learning + **SHAP**; NDVI, NDBI; monthly Landsat 2001–2023 | SHAP used for *temporal* UHI forecasting, not spatial attribution |
| Zafar, Z. et al. 2026, *Phys. Chem. Earth* 145:104887. doi:10.1016/j.pce.2026.104887 | Lahore | "wavelet coherence and machine learning" | **No abstract available.** Must be read: it may use SHAP |
| Zafar, Z. et al. 2026, *Sci. China Earth Sci.* 69:528–551 | 5 Punjab cities | RF, XGBoost, AdaBoost | Already in the assessment; full text still unread |
| Khan, M. et al. 2023, *Heliyon* 9:e23043. doi:10.1016/j.heliyon.2023.e23043 | **Islamabad**, planned vs unplanned areas | SVM classification; CA-Markov; ANN projection | **Planned-vs-unplanned LST contrast in Islamabad already studied** (descriptive, change-based; no driver attribution) |
| Syed, N. et al. 2026, *Earth Syst. Environ.* 10:7853–7869 | "Arid heartland" | GEE + ML for LULC/LST | No abstract; probably classification-based; check |

## Consequences for the gap statement

1. **Drop any implication that XGBoost/SHAP attribution of LST is new for Pakistan.** Baqa et al. (2026) did exactly this for four cities. Globally, SHAP–LST studies have grown roughly 30-fold since 2023 (Q3b), so the method itself carries no novelty.
2. **Drop "planned vs unplanned" as novel in itself.** Khan et al. (2023) compared LST between planned and unplanned areas of Islamabad. What remains new is comparing *driver response functions* (SHAP dependence) between the two fabrics, rather than mean LST.
3. **What still survives screening:**
   - (a) Islamabad–Rawalpindi: no ML/XAI driver-attribution study found;
   - (b) **topographic complexity**: every Pakistani XAI–LST study found is on flat plains, and this would be the first test of whether terrain effects (elevation, solar incidence) confound vegetation attribution in a Pakistani city. Phrase this as "to our knowledge" and confirm with Scopus/WoS;
   - (c) **spatially independent validation** and **explanation-stability testing**, neither visible in any Pakistani abstract screened;
   - (d) cross-city comparison: our results can be set against Baqa et al. (2026), which strengthens the Discussion.

### Revised gap (replaces D.3 draft in the assessment)

> Explainable machine learning has recently been applied to land surface temperature in Pakistan's plains cities (e.g., Karachi, Lahore, Faisalabad, Gujranwala, Multan). However, no study we identified has applied it to the Islamabad–Rawalpindi metropolitan area. In that setting, strong relief co-varies with vegetation, so terrain could confound the attribution of cooling to green cover. Existing Pakistani studies also report performance without spatially independent validation and do not test whether explanations are stable across years, folds or model families. This study addresses these gaps and compares driver response functions between Islamabad's planned and Rawalpindi's organically grown urban fabric.

## To do (cannot be completed via open APIs)
- [ ] Repeat Q1/Q2 in Scopus and WoS; merge and deduplicate; report a PRISMA flow.
- [ ] Obtain full texts: Baqa et al. 2026; Zafar et al. 2026 (both papers); Syed et al. 2026. Record their validation scheme and SHAP use.
