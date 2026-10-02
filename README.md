# Biodiversity Forecasting Decision Framework

An interactive decision-support tool for choosing biodiversity forecasting methods, based on the four-tier framework in Watson et al. (2026).

**Live app:** [edna_forecast_app.py](https://mldecisionframework.streamlit.app/) <!-- replace with your Streamlit URL -->

> **Status:** beta (version 4.0). The method scores and thresholds are practical heuristics under review with co-authors. Feedback is welcome via [Issues](https://github.com/artizanlakes/forecasting_framework/issues).

---

## What the tool does

Users describe their forecasting problem and data, and the app returns a set of candidate methods with the reasoning behind each, plus a plan for evaluating and reporting the forecast. It follows the four tiers of the framework:

| Tier | Purpose | User provides | App returns |
|---|---|---|---|
| 1 · Problem definition | Define what is predicted, where and over what time horizon | Ecological objective, spatial domain, temporal mode, number of sites, time points per site | The capabilities the problem requires (e.g. spatial transfer, extrapolation, uncertainty) |
| 2 · Data and analytical constraints | Characterise the data across seven dimensions (Box 1) | Number of features, temporal correlation, taxonomic resolution, missing data and noise, interpretability need, computing resources | Effective sample size, n_eff : p ratio and Table B1 band |
| 3 · Method filtering and prioritisation | Classify 23 candidate methods | – | Methods grouped as **recommended**, **emerging**, **use with caution** or **excluded**, with reasons, Python/R packages and references |
| 4 · Evaluation, uncertainty and reporting | Plan how forecasts are tested and reported | Whether an independent dataset is available | Validation design, performance metrics, uncertainty outputs, plausibility checks and a downloadable report |

The framework does not assume one method meets every requirement, so several candidates are usually retained for comparison.

## How the calculation works

1. **Required capabilities.** The Tier 1 choices switch on capabilities a method must provide, such as spatial transfer or extrapolation.
2. **Effective sample size.** Repeated samples from the same site are correlated, so they count as fewer independent samples. Each site's series is adjusted with the AR(1) variance-inflation formula and then reduced by the proportion of missing values:

   $$n_{eff} = n_{sites} \cdot \frac{m}{1 + 2\sum_{k=1}^{m-1}\left(1-\frac{k}{m}\right)\rho^{k}} \cdot (1 - f_{missing})$$

   where *m* is time points per site, *ρ* the correlation between consecutive samples and *f_missing* the proportion of missing values. The ratio *n_eff : p* is placed in the Table B1 bands (< 1, 1–5, 5–10, > 10).
3. **Method classification.** Each method is checked against hard rules (response type, minimum n_eff : p band, computing, interpretability, temporal data) and scored on how many required capabilities it provides (*S*, 0–1). *S* ≥ 0.67 with no penalties gives **recommended** (or **emerging** for newer methods); 0.33–0.67 or any penalty gives **use with caution**; below 0.33 gives **excluded**.
4. **Evaluation plan.** The validation design is chosen from the prediction domain (e.g. spatial-block cross-validation for new sites, forward-chaining for future states), then from n_eff : p.

Full rules are shown in the app's **About the calculation** tab.

## Run it locally

Requires Python 3.9 or later.

```bash
git clone https://github.com/artizanlakes/forecasting_framework.git
cd forecasting_framework
pip install -r requirements.txt
python -m streamlit run edna_forecast_app.py
```

The app opens in your browser at `http://localhost:8501`. Running `python edna_forecast_app.py` also works.

## Repository contents

| File | Description |
|---|---|
| `edna_forecast_app.py` | The Streamlit app. The method catalogue, decision logic and charts are kept separate from the interface code |
| `requirements.txt` | Python packages needed (Streamlit, Plotly, pandas) |
| `README.md` | This file |

## Limitations

- Thresholds and method scores are decision-support heuristics, not universal statistical rules.
- Spatial correlation between sites is not corrected in the effective sample size.
- The tool suggests methods; it does not fit models or assess ecological plausibility, which remains the user's judgement.

## Citation

If you use this tool, please cite:

> Watson A., Naksukpaiboon P., Kopec Harding K.R., Wingfield C., Orsini L. & Zhou J. (2026). Rethinking biodiversity forecasting: from fragmented methods to integrated predictions. *[Journal, in prep.]*



## Licence


To be confirmed.

## Contact

Arron Watson, School of Biosciences, University of Birmingham – axw213@student.bham.ac.uk

## Version history

| Version | Date | Changes |
|---|---|---|
| 4.0 | October 2026 | Restructured around the four framework tiers; effective sample size calculation; method catalogue with capability scoring |
