# Tutorial 3: Atlantic salmon returns (responses to environmental change and management)

Long-term counts of returning salmon from rivers across Europe, monitored for different reasons, used to project returns under warmer, drier summers and test whether riparian tree planting offsets the effect. It shows how large spatial scales and data of mixed origin affect method choice and ecological plausibility.

**Data:** [`tutorial3_salmon_returns.xlsx`](tutorial3_salmon_returns.xlsx) · **Shortcut:** in the app, open the **Tutorials** tab and click **Load Tutorial 3**.

> River names and locations are real, but **all counts and measurements are simulated**. They do not describe the real stocks of these rivers.

## The question

> *How will adult returns change under warmer, lower summer flows, and does riparian tree planting offset it?*

**Context:** Atlantic salmon are listed in Annex II of the Habitats Directive (in fresh water), and many rivers are designated Special Areas of Conservation for them. Internationally, North Atlantic stocks are managed through NASCO (North Atlantic Salmon Conservation Organization), advised by ICES. Not all rivers here are monitored for legislative reasons; some data come from research projects.

## The data

| | |
|---|---|
| Design | **34 salmon rivers in 11 countries**, from the Alta (northern Norway) to the Narcea (Spain), 1995–2024; series length 14–30 years (average 25); 834 river-years |
| Measurements | Adult returns (counter data), summer maximum water temperature, days above 20 °C, summer low-flow anomaly, riparian canopy cover (shading), marine survival index; smolt estimates for index and research rivers only |
| River information | Nearest town or city, latitude, monitoring purpose (16 statutory, 4 NASCO/ICES index, 7 national monitoring, 7 research), counter type, hatchery stocking, tree-planting programme, counter upgrade year |
| Built-in patterns | Southern rivers are warmer and decline faster; summers above about 20 °C reduce returns three years later; shading lowers water temperature; Europe-wide droughts (2003, 2006, 2018, 2022) lower flows; marine survival declines over time |
| Missing data | About **10%** (counter failures, lost temperature loggers, canopy not surveyed every year) |
| Sheets | `data`, `rivers`, `variables`, `app_inputs` |

## Step 1 · Tier 1: problem definition

| App input | Enter | Why |
|---|---|---|
| Ecological objective | Responses to environmental change or management interventions | Climate pressure and a management measure |
| Spatial domain | Interpolation – same sites or conditions as the data | Forecasts for the monitored rivers |
| Number of sites (*n*_sites) | **34** | Rows in `rivers` |
| Temporal mode | Long-term scenario projection | Returns under future summer conditions |
| Time points per site (*m*) | **25** | Average years per river |

Required capabilities: **extrapolation, representation of temporal dynamics, process knowledge, uncertainty propagation**.

## Step 2 · Tier 2: data and analytical constraints

| App input | Enter | Why |
|---|---|---|
| Number of features (*p*) | **7** | Temperature, days above 20 °C, low flow, canopy cover, marine survival, latitude, stocking |
| Correlation between samples (ρ) | High (ρ ≈ 0.8) | Returns in successive years are strongly linked |
| Level of identification | Species | |
| Missing values (%) | **10** | Calculated in `app_inputs` |
| Interpretability | Medium | |
| Computing | Moderate (workstation) | |

**What the app calculates:** each river's 25 years are worth 3.4 independent samples, so *n*_eff = 34 × 3.38 × 0.90 = **103**; *n*:*p* = **14.8** (band > 10). Plenty of data relative to predictors: the constraint here is extrapolation, not dimensionality.

## Step 3 · Tier 3: method filtering and prioritisation

| Class | Methods | Why |
|---|---|---|
| ✅ Recommended | Process-based models (e.g. salmon life-cycle models) | Represent temperature effects on juveniles, flow and marine survival mechanistically; better suited to conditions not yet observed |
| ⚠️ Cautious use | Bayesian hierarchical models | Share information across rivers, but only partly represent mechanisms and extrapolation |
| ⚠️ Cautious use | State-space models, data assimilation, ARIMA | Capture year-to-year population dynamics, but do not extrapolate to new conditions on their own |
| ⚠️ Cautious use | GLM, GAM, ensemble forecasting, regularised VAR | Partial capabilities only |
| ❌ Excluded | Random forest, gradient boosting, PLS, SVM | Cannot extrapolate or represent processes |
| ❌ Excluded | Hybrid models, PINNs, LSTM, deep learning | Need HPC-level computing |

A practical combination is a process-based life-cycle model whose parameters are estimated in a Bayesian hierarchical framework across rivers.

## Step 4 · Tier 4: forecast evaluation, uncertainty and reporting

- **Validation:** forward-chaining validation; train on earlier years and test on later ones.
- **Metrics:** error in predicted change; RMSE and MAE; CRPS; prediction-interval coverage; skill against persistence.
- **Uncertainty:** separate future climate scenarios, parameter uncertainty and model structure.
- **Plausibility (priority):** predictions go beyond observed conditions. Use the `rivers` sheet to check before trusting a pattern: stocked rivers inflate returns, counter upgrades create step changes, and research series are short. Compare predicted temperature effects with published thermal tolerances for salmon.

## Try changing one input

| Change | Result |
|---|---|
| Temporal mode → **Near-term iterative** (next year's returns) | State-space models become recommended and data assimilation emerging, as in operational stock forecasts |
| Spatial domain → **Transfer** (rivers without counters) | Bayesian hierarchical models join process-based models as recommended |
| Computing → **High** | Hybrid process + ML models and PINNs appear as emerging |
| Interpretability → **High** | Recommendations unchanged; black-box methods were already excluded |

*Results from app version 4.2.*
