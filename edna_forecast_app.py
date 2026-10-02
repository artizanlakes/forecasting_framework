"""
Biodiversity Forecasting Decision Framework  (Watson et al., 2026)
====================================================================

Interactive implementation of the four-tier decision-support framework:

    Tier 1  Problem definition            (ecological objective + spatial/temporal structure)
    Tier 2  Data and analytical constraints (seven dimensions, Box 1)
    Tier 3  Method filtering and prioritisation
    Tier 4  Forecast evaluation, uncertainty and reporting

HOW TO RUN (Windows / macOS / Linux)
------------------------------------
    python edna_forecast_app.py                    <- launches Streamlit for you (see bottom of file)
    python -m streamlit run edna_forecast_app.py   <- use this if `streamlit run ...` is not recognised

FILE LAYOUT
-----------
    1  Imports and configuration
    2  METHOD_CATALOGUE  (single source of truth for Tier 3, charts, export)
    3  Constants (Tier 1 options, worked examples, Table B1 bands, thresholds)
    4  Tier 1 logic   - required capabilities
    5  Tier 2 logic   - effective sample size, n_eff:p ratio, constraint profile
    6  Tier 3 logic   - score_methods()
    7  Tier 4 logic   - evaluation plan and report
    8  Charts (plotly only)
    9  Streamlit UI (one tab per tier)
    10 Export
    11 Help text, footer and launcher

Sections 2-8 contain no Streamlit calls, so the decision logic can be imported and
tested on its own (see test_framework.py).
"""

# =============================================================================
# 1. IMPORTS AND CONFIGURATION
# =============================================================================
import os
import sys

import pandas as pd
import plotly.graph_objects as go

APP_VERSION = "4.0"
APP_YEAR = "2026"
CITATION = ("Watson A., Naksukpaiboon P., Kopec Harding K.R., Wingfield C., Orsini L. & Zhou J. "
            "(2026). Rethinking biodiversity forecasting: from fragmented methods to integrated "
            "predictions. [Journal, in prep.]")
SELECT = "— select —"

# Capability vocabulary used by Tier 1 (required) and the catalogue (provided)
CAPABILITIES = {
    "multivariate": "Models many taxa / a community response jointly",
    "interactions": "Represents species interactions or associations",
    "spatial_transfer": "Transfers information to new or poorly sampled sites",
    "temporal_dynamics": "Represents temporal dynamics (lags, trends, autocorrelation)",
    "extrapolation": "Predicts beyond observed environmental conditions",
    "uncertainty": "Produces probabilistic output / quantified uncertainty",
    "iterative_updating": "Updates forecasts as new observations arrive",
    "mechanistic": "Encodes ecological or physical process knowledge",
}

RESPONSE_LABELS = {
    "univariate": "single (univariate) response, e.g. a diversity index or ecosystem variable",
    "per_taxon": "per-taxon response (occurrence or abundance of each taxon)",
    "multivariate": "multivariate community response",
    "any": "any response type",
}

# =============================================================================
# 2. METHOD CATALOGUE
# -----------------------------------------------------------------------------
# Attribute scales
#   method_class     : Tier 3 method classes named in the manuscript (lines 339-342)
#   response         : response types the method can model (see RESPONSE_LABELS)
#   min_band         : lowest Table B1 band the method can be used in (below = excluded)
#   pref_band        : band from which no dimensionality penalty applies
#                      bands: 0 = n:p < 1, 1 = 1-5, 2 = 5-10, 3 = > 10
#   min_ratio        : optional extra n_eff:p threshold (deep models only) -> penalty if below
#   caps             : capability -> level (1 = full, 0.5 = partial, absent = none)
#   interpretability : 1 low, 2 medium, 3 high
#   compute          : 1 laptop, 2 workstation, 3 HPC / GPU
#   robustness       : robustness to missing or noisy data, 1 low, 2 medium, 3 high
#   temporal_only    : time-series method; needs repeated observations and a temporal mode
#   min_timepoints   : time points per site below which a penalty applies
#   maturity         : "established" or "emerging" in biodiversity forecasting
#   packages, refs, manuscript_examples : reporting fields
# All scores are author-assigned heuristics and are reported as such (Supplementary Methods).
# =============================================================================
METHOD_CATALOGUE = {
    # ---------------- Statistical models ----------------
    "Generalised linear model (GLM)": dict(
        method_class="Statistical", response={"univariate", "per_taxon"},
        min_band=2, pref_band=3,
        caps={"uncertainty": 1, "extrapolation": 0.5, "spatial_transfer": 0.5},
        interpretability=3, compute=1, robustness=1, maturity="established",
        description="Interpretable baseline; parametric response shapes extrapolate more predictably than flexible learners.",
        packages="statsmodels.api.GLM; sklearn.linear_model.LogisticRegression / PoissonRegressor",
        refs="Nelder & Wedderburn, 1972",
        manuscript_examples="Species distribution models (Elith & Leathwick, 2009)"),
    "Generalised additive model (GAM)": dict(
        method_class="Statistical", response={"univariate", "per_taxon"},
        min_band=2, pref_band=3,
        caps={"uncertainty": 1, "temporal_dynamics": 0.5, "spatial_transfer": 0.5},
        interpretability=3, compute=1, robustness=1, maturity="established",
        description="Smooth non-linear responses; limit basis dimension to avoid overfitting. Splines extrapolate poorly.",
        packages="pygam.LogisticGAM / PoissonGAM / LinearGAM; mgcv (R)",
        refs="Hastie & Tibshirani, 1990; Wood, 2017",
        manuscript_examples="Species distribution models (Elith & Leathwick, 2009)"),
    "Regularised regression (Lasso / Ridge / Elastic Net)": dict(
        method_class="Statistical", response={"univariate", "per_taxon"},
        min_band=0, pref_band=0,
        caps={"uncertainty": 0.5, "extrapolation": 0.5, "spatial_transfer": 0.5},
        interpretability=3, compute=1, robustness=1, maturity="established",
        description="Penalised coefficients control overfitting when p approaches or exceeds n; L1 performs feature selection.",
        packages="sklearn.linear_model.LogisticRegressionCV / LassoCV / ElasticNetCV; glmnet (R)",
        refs="Tibshirani, 1996; Zou & Hastie, 2005; Hastie et al., 2009",
        manuscript_examples="Box 1 / Table B1 (n:p < 1)"),
    "Partial least squares (PLS) regression": dict(
        method_class="Statistical", response={"univariate", "per_taxon", "multivariate"},
        min_band=0, pref_band=0,
        caps={"multivariate": 1, "uncertainty": 0.5, "spatial_transfer": 0.5},
        interpretability=2, compute=1, robustness=1, maturity="established",
        description="Built-in dimension reduction for many correlated predictors or responses.",
        packages="sklearn.cross_decomposition.PLSRegression",
        refs="Wold et al., 2001",
        manuscript_examples="Box 1 / Table B1 (dimensionality reduction)"),
    "Constrained ordination (RDA / CCA)": dict(
        method_class="Statistical", response={"multivariate"},
        min_band=0, pref_band=1,
        caps={"multivariate": 1, "uncertainty": 0.5},
        interpretability=3, compute=1, robustness=1, maturity="established",
        description="Links community composition to environmental gradients; permutation tests give significance.",
        packages="skbio.stats.ordination.rda / cca; vegan::rda / cca (R)",
        refs="ter Braak, 1986; Legendre & Legendre, 2012",
        manuscript_examples="—"),
    "Distance-based methods (nMDS / PERMANOVA)": dict(
        method_class="Statistical", response={"multivariate"},
        min_band=0, pref_band=0,
        caps={"multivariate": 1},
        interpretability=3, compute=1, robustness=2, maturity="established",
        description="Describes and tests compositional differences; descriptive rather than predictive.",
        packages=("sklearn.manifold.MDS(metric=False, dissimilarity='precomputed') on Bray–Curtis "
                  "(scipy.spatial.distance); skbio.stats.distance.permanova; vegan::metaMDS / adonis2 (R)"),
        refs="Kruskal, 1964; Anderson, 2001",
        manuscript_examples="—"),
    # ---------------- Bayesian hierarchical ----------------
    "Bayesian hierarchical models": dict(
        method_class="Bayesian hierarchical", response={"univariate", "per_taxon"},
        min_band=0, pref_band=0,
        caps={"uncertainty": 1, "spatial_transfer": 1, "extrapolation": 0.5,
              "temporal_dynamics": 0.5, "iterative_updating": 0.5, "mechanistic": 0.5},
        interpretability=3, compute=2, robustness=3, maturity="established",
        description="Partial pooling shares information across sites; shrinkage priors (e.g. horseshoe) handle n:p < 1; explicit observation models.",
        packages="PyMC; cmdstanpy (Stan); brms (R)",
        refs="Gelman & Hill, 2007; Clark, 2005",
        manuscript_examples="Bayesian state-space models (Table S1, Fig. 2)"),
    "Joint species distribution models (e.g. HMSC)": dict(
        method_class="Bayesian hierarchical", response={"per_taxon", "multivariate"},
        min_band=1, pref_band=2,
        caps={"multivariate": 1, "interactions": 0.5, "spatial_transfer": 1, "uncertainty": 1},
        interpretability=2, compute=3, robustness=2, maturity="established",
        description="Models all taxa jointly; residual associations indicate (not prove) interactions.",
        packages="Hmsc (R); hmsc-hpc (GPU back end)",
        refs="Ovaskainen et al., 2017",
        manuscript_examples="—"),
    # ---------------- State-space and dynamic occupancy ----------------
    "State-space / dynamic occupancy models": dict(
        method_class="State-space and dynamic occupancy", response={"univariate", "per_taxon"},
        min_band=0, pref_band=1,
        caps={"temporal_dynamics": 1, "uncertainty": 1, "iterative_updating": 1, "spatial_transfer": 0.5},
        interpretability=3, compute=2, robustness=3, maturity="established",
        temporal_only=True, min_timepoints=10,
        description="Separates observation error (e.g. eDNA detection) from true dynamics; Kalman filtering supports iterative updating.",
        packages="statsmodels.tsa.statespace; PyMC; unmarked::colext (R)",
        refs="MacKenzie et al., 2003; Durbin & Koopman, 2012",
        manuscript_examples="Iterative near-term forecasting (Dietze et al., 2018; Thomas et al., 2023)"),
    "ARIMA": dict(
        method_class="State-space and dynamic occupancy", response={"univariate"},
        min_band=1, pref_band=2,
        caps={"temporal_dynamics": 1, "uncertainty": 1, "iterative_updating": 0.5},
        interpretability=3, compute=1, robustness=1, maturity="established",
        temporal_only=True, min_timepoints=30,
        description="Classical univariate time-series model; needs long, regular series.",
        packages="statsmodels.tsa.arima.model.ARIMA",
        refs="Box & Jenkins, 1970",
        manuscript_examples="—"),
    "Regularised vector autoregression (VAR)": dict(
        method_class="State-space and dynamic occupancy", response={"per_taxon", "multivariate"},
        min_band=0, pref_band=1,
        caps={"temporal_dynamics": 1, "multivariate": 1, "interactions": 0.5, "uncertainty": 0.5},
        interpretability=2, compute=1, robustness=1, maturity="established",
        temporal_only=True, min_timepoints=20,
        description="Multivariate time series with L1/L2 penalties; lagged cross-effects approximate interactions.",
        packages="statsmodels.tsa.api.VAR (unpenalised); BigVAR (R) for penalised VAR",
        refs="Basu & Michailidis, 2015",
        manuscript_examples="—"),
    # ---------------- Data assimilation ----------------
    "Data assimilation (Kalman / ensemble / particle filters)": dict(
        method_class="Data assimilation", response={"univariate", "per_taxon", "multivariate"},
        min_band=1, pref_band=2,
        caps={"iterative_updating": 1, "uncertainty": 1, "temporal_dynamics": 1, "mechanistic": 0.5},
        interpretability=2, compute=2, robustness=3, maturity="emerging",
        temporal_only=True, min_timepoints=10,
        description="Sequentially combines model forecasts with new observations, propagating uncertainty.",
        packages="filterpy; statsmodels Kalman filter; custom ensemble Kalman / particle filters",
        refs="Carrassi et al., 2018; Dietze, 2017",
        manuscript_examples="Climate and hydrological DA (Bauer et al., 2015; Liu et al., 2012)"),
    # ---------------- Machine learning ----------------
    "Random forest": dict(
        method_class="Machine learning", response={"univariate", "per_taxon", "multivariate"},
        min_band=0, pref_band=0,
        caps={"uncertainty": 0.5, "multivariate": 0.5, "spatial_transfer": 0.5, "temporal_dynamics": 0.5},
        interpretability=2, compute=1, robustness=2, maturity="established",
        description="Robust non-linear learner; random feature subsets cope with high p. Does not extrapolate beyond training range.",
        packages="sklearn.ensemble.RandomForestClassifier / RandomForestRegressor; quantile-forest",
        refs="Breiman, 2001; Cutler et al., 2007",
        manuscript_examples="Species distribution models (Elith & Leathwick, 2009)"),
    "Gradient boosting (XGBoost / LightGBM)": dict(
        method_class="Machine learning", response={"univariate", "per_taxon"},
        min_band=1, pref_band=2,
        caps={"uncertainty": 0.5, "spatial_transfer": 0.5, "temporal_dynamics": 0.5},
        interpretability=2, compute=2, robustness=3, maturity="established",
        description="High accuracy; tune learning rate, depth and early stopping. Handles missing values natively.",
        packages="sklearn.ensemble.HistGradientBoostingClassifier / Regressor; xgboost; lightgbm",
        refs="Friedman, 2001; Elith et al., 2008; Chen & Guestrin, 2016",
        manuscript_examples="—"),
    "Support vector machine (SVM)": dict(
        method_class="Machine learning", response={"univariate", "per_taxon"},
        min_band=0, pref_band=1,
        caps={"uncertainty": 0.5, "spatial_transfer": 0.5},
        interpretability=1, compute=2, robustness=1, maturity="established",
        description="Kernel classifier/regressor; usable with n < p but sensitive to kernel and scaling.",
        packages="sklearn.svm.SVC / SVR",
        refs="Cortes & Vapnik, 1995",
        manuscript_examples="—"),
    "Shallow neural network (1–2 hidden layers)": dict(
        method_class="Machine learning", response={"univariate", "per_taxon", "multivariate"},
        min_band=2, pref_band=3,
        caps={"multivariate": 1, "uncertainty": 0.5, "spatial_transfer": 0.5},
        interpretability=1, compute=2, robustness=1, maturity="established",
        description="Captures interactions among predictors; needs regularisation and dropout.",
        packages="sklearn.neural_network.MLPClassifier / MLPRegressor",
        refs="Olden et al., 2008",
        manuscript_examples="—"),
    "Deep learning (CNN / transformer / graph networks)": dict(
        method_class="Machine learning", response={"univariate", "per_taxon", "multivariate"},
        min_band=3, pref_band=3, min_ratio=100,
        caps={"multivariate": 1, "spatial_transfer": 0.5, "temporal_dynamics": 0.5, "uncertainty": 0.5},
        interpretability=1, compute=3, robustness=1, maturity="emerging",
        description="Learns complex spatio-temporal structure from very large datasets.",
        packages="torch.nn (PyTorch); tensorflow.keras",
        refs="LeCun et al., 2015",
        manuscript_examples="GraphCast (Lam et al., 2023); FourCastNet (Pathak et al., 2022); transformer forecasting (Table S1)"),
    "LSTM / GRU networks": dict(
        method_class="Machine learning", response={"univariate", "per_taxon", "multivariate"},
        min_band=3, pref_band=3, min_ratio=50,
        caps={"temporal_dynamics": 1, "multivariate": 1, "spatial_transfer": 0.5,
              "iterative_updating": 0.5, "uncertainty": 0.5},
        interpretability=1, compute=3, robustness=1, maturity="emerging",
        temporal_only=True, min_timepoints=50,
        description="Recurrent networks for long, dense time series; multi-site training supports transfer.",
        packages="torch.nn.LSTM / GRU; tensorflow.keras.layers.LSTM; neuralhydrology",
        refs="Hochreiter & Schmidhuber, 1997",
        manuscript_examples="Multi-catchment LSTM (Kratzert et al., 2019)"),
    "Transfer learning (pre-trained / multi-site models)": dict(
        method_class="Machine learning", response={"univariate", "per_taxon", "multivariate"},
        min_band=1, pref_band=2,
        caps={"spatial_transfer": 1, "extrapolation": 0.5, "multivariate": 0.5, "uncertainty": 0.5},
        interpretability=1, compute=3, robustness=1, maturity="emerging",
        description="Fine-tunes models learned in data-rich systems for data-poor sites or regions.",
        packages="torch / tensorflow.keras (fine-tuning pre-trained models)",
        refs="Pan & Yang, 2010",
        manuscript_examples="Regionalisation and transfer learning in hydrology (Beven & Cloke, 2012; Kratzert et al., 2019)"),
    "Ensemble / multi-model forecasting": dict(
        method_class="Machine learning", response={"univariate", "per_taxon"},
        min_band=1, pref_band=2,
        caps={"uncertainty": 1, "extrapolation": 0.5, "spatial_transfer": 0.5},
        interpretability=2, compute=2, robustness=2, maturity="established",
        description="Combines several model structures; ensemble spread expresses structural uncertainty.",
        packages="sklearn.ensemble.StackingRegressor / VotingClassifier; biomod2 (R)",
        refs="Araújo & New, 2007",
        manuscript_examples="Multi-model ensembles (Kirtman et al., 2013; Bedia et al., 2026)"),
    # ---------------- Process-based ----------------
    "Process-based ecosystem models": dict(
        method_class="Process-based", response={"univariate", "per_taxon", "multivariate"},
        min_band=0, pref_band=1,
        caps={"mechanistic": 1, "extrapolation": 1, "temporal_dynamics": 1, "interactions": 1,
              "spatial_transfer": 0.5, "multivariate": 0.5, "uncertainty": 0.5},
        interpretability=3, compute=2, robustness=2, maturity="established",
        description="Represents the processes generating patterns; less data-hungry and better under novel conditions, if mechanisms are known.",
        packages="Standalone model code (e.g. PCLake, SWAT/SWAT+, VIC); eDITH (R) for eDNA transport",
        refs="Mooij et al., 2007; Arnold et al., 1998; Liang et al., 1994",
        manuscript_examples=("PCLake (Mooij et al., 2007); cyanobacterial thresholds (Zhang et al., 2021); "
                             "eDNA transport (Song et al., 2017; Carraro et al., 2023); SWAT; VIC; "
                             "decadal prediction (Meehl et al., 2009); EFAS (Alfieri et al., 2013)")),
    # ---------------- Hybrid ----------------
    "Hybrid process + ML models": dict(
        method_class="Hybrid", response={"univariate", "per_taxon", "multivariate"},
        min_band=1, pref_band=2,
        caps={"mechanistic": 1, "extrapolation": 1, "temporal_dynamics": 1, "multivariate": 0.5,
              "spatial_transfer": 0.5, "uncertainty": 0.5, "iterative_updating": 0.5, "interactions": 0.5},
        interpretability=2, compute=3, robustness=2, maturity="emerging",
        description="Couples process structure with learned components (emulators, ML for poorly resolved processes).",
        packages="Custom: process model + torch / sklearn components",
        refs="Wang et al., 2019",
        manuscript_examples="ML emulators (Wang et al., 2019); process-informed SDMs (Riaz et al., 2020)"),
    "Physics- / ecology-informed neural networks (PINNs)": dict(
        method_class="Hybrid", response={"univariate", "per_taxon"},
        min_band=1, pref_band=2,
        caps={"mechanistic": 1, "extrapolation": 1, "temporal_dynamics": 1,
              "spatial_transfer": 0.5, "uncertainty": 0.5},
        interpretability=2, compute=3, robustness=2, maturity="emerging",
        description="Embeds governing equations in the loss function to keep predictions process-consistent.",
        packages="DeepXDE; custom torch.autograd implementations",
        refs="Raissi et al., 2019",
        manuscript_examples="PINNs in hydrology (Qi et al., 2024)"),
}

# Fill defaults so every entry has every field
for _m in METHOD_CATALOGUE.values():
    _m.setdefault("temporal_only", False)
    _m.setdefault("min_timepoints", None)
    _m.setdefault("min_ratio", None)

# =============================================================================
# 3. CONSTANTS
# =============================================================================
# ---- Tier 1, part 1: ecological objectives (manuscript lines 244-247) ----
OBJECTIVES = {
    "Species occurrence and abundance": dict(
        response="per_taxon", requires=[],
        p_help="All features entering the model: environmental predictors, plus taxa/ASVs if used as predictors."),
    "Community composition": dict(
        response="multivariate", requires=["multivariate"],
        p_help="Number of taxa/ASVs in the community matrix after filtering, plus environmental predictors."),
    "Taxonomic, functional or phylogenetic diversity": dict(
        response="univariate", requires=[],
        p_help="Number of predictors used to explain the diversity index."),
    "Ecological interactions": dict(
        response="multivariate", requires=["multivariate", "interactions"],
        p_help="Number of interacting taxa modelled, plus environmental predictors."),
    "Ecosystem state and function": dict(
        response="univariate", requires=[],
        p_help="Number of predictors (biological features, drivers) for the state or function variable."),
    "Responses to environmental change or management interventions": dict(
        response="any", requires=["extrapolation"],
        p_help="Number of predictors, including the drivers or interventions whose effects are projected."),
}
OCCURRENCE_SUBTYPES = ["Occurrence (presence/absence)", "Abundance (counts / read counts / density)"]

# ---- Tier 1, part 2: spatial domain (lines 272-278) ----
SPATIAL_DOMAINS = {
    "Interpolation – same sites or conditions as the data": [],
    "Transfer – new sites in a similar region": ["spatial_transfer"],
    "Extrapolation – unsampled regions or novel conditions": ["spatial_transfer", "extrapolation"],
}
# ---- Tier 1, part 2: temporal modes (lines 264-270) ----
TEMPORAL_MODES = {
    "State inference – estimate the current, incompletely observed state": [],
    "Near-term iterative forecasting – update as new data arrive": ["temporal_dynamics", "iterative_updating", "uncertainty"],
    "Long-term scenario projection – responses under future conditions": ["extrapolation", "mechanistic", "uncertainty"],
    "Hindcasting / trajectory reconstruction – reconstruct past dynamics": ["temporal_dynamics"],
    "Early-warning forecasting – anticipate critical change": ["temporal_dynamics", "uncertainty"],
}
FUTURE_MODES = [k for k in TEMPORAL_MODES if k.startswith(("Near-term", "Long-term", "Early-warning"))]
TIME_UNITS = ["Hours", "Days", "Weeks", "Months", "Seasons", "Years", "Sediment layers / decades"]

# ---- Tier 2 ----
RHO_OPTIONS = {"Low (ρ ≈ 0.2)": 0.2, "Moderate (ρ ≈ 0.5)": 0.5, "High (ρ ≈ 0.8)": 0.8}
INTERP_OPTIONS = {"High (regulatory / policy)": 3, "Medium (scientific understanding)": 2,
                  "Low (exploratory / predictive)": 1}
COMPUTE_OPTIONS = {"Limited (laptop, < 8 GB RAM)": 1, "Moderate (workstation, 8–32 GB)": 2,
                   "High (HPC / GPU, > 32 GB)": 3}
MISSING_PENALTY_THRESHOLD = 0.20      # Box 1: > 20 % missing
TAXONOMY_FLAG_THRESHOLD = 50          # % assigned to species level

# ---- Table B1 bands ----
NP_BANDS = [
    dict(label="< 1", name="High-dimensional", risk="High",
         preferred="Regularised regression, feature selection, dimensionality reduction, constrained machine learning"),
    dict(label="1–5", name="Moderate high-dimensional", risk="Moderate",
         preferred="Parsimonious machine learning, ensemble methods, regularised approaches"),
    dict(label="5–10", name="Moderate", risk="Moderate to low",
         preferred="Statistics, machine learning alone or combined with process-based models"),
    dict(label="> 10", name="Low-dimensional", risk="Low",
         preferred="Most forecasting approaches feasible"),
]

# ---- Tier 3 classification thresholds (heuristics) ----
SCORE_RECOMMEND = 0.67
SCORE_MINIMUM = 0.33
MIN_TIMEPOINTS_TEMPORAL = 3

STATUS_ORDER = ["Recommended", "Emerging", "Use with caution", "Excluded"]

# ---- Worked examples for Tier 1 (load into the form) ----
_S = list(SPATIAL_DOMAINS)
_T = list(TEMPORAL_MODES)
_O = list(OBJECTIVES)
EXAMPLES = {
    "Occurrence of an invasive species from eDNA at unsampled lakes in the same catchment": dict(
        objective=_O[0], subtype=OCCURRENCE_SUBTYPES[0], spatial=_S[1], temporal=_T[0],
        n_sites=30, timepoints=1, time_unit="Months", p=25, rho="Moderate (ρ ≈ 0.5)"),
    "Monthly abundance of cyanobacterial taxa at a monitored reservoir, next 3 months": dict(
        objective=_O[0], subtype=OCCURRENCE_SUBTYPES[1], spatial=_S[0], temporal=_T[1],
        n_sites=3, timepoints=60, time_unit="Months", p=12, rho="High (ρ ≈ 0.8)"),
    "Bacterial community composition across lakes from environmental pressures": dict(
        objective=_O[1], spatial=_S[0], temporal=_T[0],
        n_sites=60, timepoints=1, time_unit="Months", p=500, rho="Moderate (ρ ≈ 0.5)"),
    "Macroinvertebrate community composition under 2050 warming scenarios": dict(
        objective=_O[1], spatial=_S[2], temporal=_T[2],
        n_sites=40, timepoints=10, time_unit="Years", p=150, rho="Moderate (ρ ≈ 0.5)"),
    "Phylogenetic diversity over the past century from sediment-core sedaDNA": dict(
        objective=_O[2], spatial=_S[0], temporal=_T[3],
        n_sites=5, timepoints=30, time_unit="Sediment layers / decades", p=15, rho="High (ρ ≈ 0.8)"),
    "Functional diversity of fish at unsurveyed river reaches": dict(
        objective=_O[2], spatial=_S[1], temporal=_T[0],
        n_sites=80, timepoints=3, time_unit="Years", p=20, rho="Low (ρ ≈ 0.2)"),
    "Plankton association networks under changing nutrient loads": dict(
        objective=_O[3], spatial=_S[0], temporal=_T[1],
        n_sites=10, timepoints=52, time_unit="Weeks", p=80, rho="Moderate (ρ ≈ 0.5)"),
    "Early warning of a shift to a turbid, cyanobacteria-dominated lake state": dict(
        objective=_O[4], spatial=_S[0], temporal=_T[4],
        n_sites=1, timepoints=365, time_unit="Days", p=12, rho="High (ρ ≈ 0.8)"),
    "Nutrient-cycling function from microbial eDNA across catchments": dict(
        objective=_O[4], spatial=_S[1], temporal=_T[0],
        n_sites=50, timepoints=4, time_unit="Seasons", p=300, rho="Moderate (ρ ≈ 0.5)"),
    "Fish community response to weir removal (management intervention)": dict(
        objective=_O[5], spatial=_S[0], temporal=_T[1],
        n_sites=12, timepoints=8, time_unit="Years", p=30, rho="Moderate (ρ ≈ 0.5)"),
    "Species range shifts under climate change in data-poor catchments": dict(
        objective=_O[5], spatial=_S[2], temporal=_T[2],
        n_sites=25, timepoints=1, time_unit="Years", p=10, rho="Moderate (ρ ≈ 0.5)"),
}

# =============================================================================
# 4. TIER 1 LOGIC
# =============================================================================
def required_capabilities(objective, spatial, temporal):
    """Union of capabilities demanded by the objective, spatial domain and temporal mode."""
    req = []
    for cap in (OBJECTIVES[objective]["requires"] + SPATIAL_DOMAINS[spatial] + TEMPORAL_MODES[temporal]):
        if cap not in req:
            req.append(cap)
    return req


# =============================================================================
# 5. TIER 2 LOGIC
# =============================================================================
def ar1_effective_n(m, rho):
    """Effective number of independent observations in a series of m points with lag-1
    autocorrelation rho (AR(1) variance-inflation formula). m = 1 -> 1; rho = 0 -> m."""
    m = int(m)
    if m <= 1:
        return float(max(m, 0))
    inflation = 1 + 2 * sum((1 - k / m) * rho ** k for k in range(1, m))
    return m / inflation


def effective_sample_size(n_sites, timepoints, rho, missing_frac):
    n_raw = int(n_sites) * int(timepoints)
    per_site = ar1_effective_n(timepoints, rho)
    n_eff = n_sites * per_site * (1 - missing_frac)
    return dict(n_raw=n_raw, n_eff_per_site=per_site, n_eff=n_eff)


def np_band(ratio):
    if ratio < 1:
        return 0
    if ratio < 5:
        return 1
    if ratio < 10:
        return 2
    return 3


def feature_label(p):
    """Plain-language description of feature dimensionality (display only)."""
    if p < 10:
        return "Very low (< 10 features)"
    if p < 50:
        return "Low (10–49 features)"
    if p < 200:
        return "Moderate (50–199 features)"
    if p < 1000:
        return "Moderate–high (200–999 features)"
    return "High (≥ 1,000 features, typical of ASV tables)"


def build_profile(t1, t2):
    """Combine Tier 1 and Tier 2 inputs into the constraint profile used by Tiers 3-4."""
    rho = RHO_OPTIONS[t2["rho"]] if t1["timepoints"] > 1 else 0.0
    missing = t2["missing_pct"] / 100.0
    ess = effective_sample_size(t1["n_sites"], t1["timepoints"], rho, missing)
    p = max(int(t2["p"]), 1)
    ratio = ess["n_eff"] / p
    band = np_band(ratio)
    return dict(
        objective=t1["objective"], subtype=t1.get("subtype"), spatial=t1["spatial"],
        temporal=t1["temporal"], time_unit=t1["time_unit"],
        n_sites=int(t1["n_sites"]), m=int(t1["timepoints"]),
        response=OBJECTIVES[t1["objective"]]["response"],
        required=required_capabilities(t1["objective"], t1["spatial"], t1["temporal"]),
        rho=rho, missing=missing, noise=bool(t2["noise"]), p=p,
        n_raw=ess["n_raw"], n_eff=ess["n_eff"], n_eff_per_site=ess["n_eff_per_site"],
        ratio=ratio, band=band,
        interp=INTERP_OPTIONS[t2["interp"]], interp_label=t2["interp"],
        compute=COMPUTE_OPTIONS[t2["compute"]], compute_label=t2["compute"],
        taxonomy=int(t2["taxonomy"]), external=bool(t2["external"]),
    )


# =============================================================================
# 6. TIER 3 LOGIC
# =============================================================================
def _cap_names(caps):
    return ", ".join(c.replace("_", " ") for c in caps)


def assess_method(name, m, prof):
    """Return a result dict for one method: status, capability match S, and reasons."""
    hard, penalties = [], []
    # --- hard exclusions (assumptions or data requirements cannot be met) ---
    if prof["response"] != "any" and prof["response"] not in m["response"]:
        hard.append(f"does not model a {RESPONSE_LABELS[prof['response']]}")
    if prof["band"] < m["min_band"]:
        hard.append(f"needs n_eff:p in band {NP_BANDS[m['min_band']]['label']} or above "
                    f"(yours: {prof['ratio']:.2f})")
    if m["compute"] > prof["compute"]:
        hard.append("needs more computing resources than available")
    if prof["interp"] == 3 and m["interpretability"] == 1:
        hard.append("black-box method excluded by the high interpretability requirement")
    if m["temporal_only"]:
        if "temporal_dynamics" not in prof["required"]:
            hard.append("time-series method, but the forecasting mode has no temporal component")
        elif prof["m"] < MIN_TIMEPOINTS_TEMPORAL:
            hard.append(f"needs ≥ {MIN_TIMEPOINTS_TEMPORAL} time points per site")

    # --- capability match ---
    req = prof["required"]
    score = (sum(m["caps"].get(c, 0) for c in req) / len(req)) if req else 1.0
    lacking = [c for c in req if m["caps"].get(c, 0) == 0]
    partial = [c for c in req if m["caps"].get(c, 0) == 0.5]

    # --- penalties (usable, but needs safeguards) ---
    if prof["band"] < m["pref_band"]:
        penalties.append(f"n_eff:p below the preferred band ({NP_BANDS[m['pref_band']]['label']}); "
                         "regularise and validate stringently")
    if m["min_ratio"] and prof["ratio"] < m["min_ratio"]:
        penalties.append(f"n_eff:p below the ~{m['min_ratio']} typically needed for this architecture")
    if (prof["missing"] > MISSING_PENALTY_THRESHOLD or prof["noise"]) and m["robustness"] == 1:
        penalties.append("sensitive to missing or noisy data; impute or use an explicit observation model")
    if prof["interp"] == 2 and m["interpretability"] == 1:
        penalties.append("limited interpretability; pair with explanation tools (e.g. SHAP, partial dependence)")
    if ("temporal_dynamics" in req and m["min_timepoints"] and prof["m"] < m["min_timepoints"]):
        penalties.append(f"fewer time points per site ({prof['m']}) than the ~{m['min_timepoints']} usually needed")

    # --- classification ---
    if hard:
        status = "Excluded"
        reasons = hard
    elif score < SCORE_MINIMUM:
        status = "Excluded"
        reasons = [f"lacks required capabilities: {_cap_names(lacking)}"]
    elif score >= SCORE_RECOMMEND and not penalties:
        status = "Recommended" if m["maturity"] == "established" else "Emerging"
        reasons = ["meets the problem's requirements and data constraints"]
        if m["maturity"] == "emerging":
            reasons = ["meets the requirements, but has limited application in biodiversity "
                       "forecasting or needs specialised expertise"]
    else:
        status = "Use with caution"
        reasons = penalties[:]
        if score < SCORE_RECOMMEND:
            reasons.insert(0, f"only partly meets required capabilities "
                              f"(missing: {_cap_names(lacking) or 'none'}; partial: {_cap_names(partial) or 'none'})")

    return dict(name=name, status=status, score=score, reasons=reasons, penalties=penalties,
                lacking=lacking, partial=partial, method_class=m["method_class"],
                maturity=m["maturity"], interpretability=m["interpretability"],
                compute=m["compute"], robustness=m["robustness"], pref_band=m["pref_band"],
                packages=m["packages"], refs=m["refs"], description=m["description"],
                manuscript_examples=m["manuscript_examples"])


def score_methods(prof, catalogue=None):
    """Classify every catalogue method. Returns {status: [result, ...]} sorted by score."""
    catalogue = catalogue or METHOD_CATALOGUE
    out = {s: [] for s in STATUS_ORDER}
    for name, m in catalogue.items():
        r = assess_method(name, m, prof)
        out[r["status"]].append(r)
    for s in out:
        out[s].sort(key=lambda r: (-r["score"], -r["interpretability"], r["name"]))
    return out


# =============================================================================
# 7. TIER 4 LOGIC
# =============================================================================
def validation_design(prof):
    """Validation strategy from prediction domain first, then n_eff:p (lines 373-387)."""
    spatial_new = "spatial_transfer" in prof["required"]
    future = prof["temporal"] in FUTURE_MODES
    hindcast = prof["temporal"].startswith("Hindcasting")
    design, notes = None, []

    if spatial_new and future and prof["m"] > 1:
        design = ("Spatio-temporal block cross-validation: hold out whole sites and later periods together "
                  "(sklearn GroupKFold on sites combined with TimeSeriesSplit; blockCV in R).")
    elif spatial_new:
        if prof["n_sites"] >= 10:
            design = "Spatial-block cross-validation (blockCV in R; sklearn GroupKFold on spatial blocks)."
        else:
            design = "Leave-one-site-out cross-validation (sklearn LeaveOneGroupOut)."
        if prof["n_sites"] < 5:
            notes.append("Fewer than 5 sites: spatial transferability can only be weakly tested.")
    elif future and prof["m"] > 1:
        design = ("Forward-chaining (rolling-origin) validation: train on earlier periods, test on later ones "
                  "(sklearn.model_selection.TimeSeriesSplit).")
    elif future:
        design = ("No repeated observations: forecasts cannot be tested in time. Use the best available "
                  "spatial hold-out and plan follow-up sampling to evaluate forecasts.")
    elif hindcast:
        design = ("Temporal hold-out of a past period, or reconstruction of a known trajectory from an "
                  "independent record (e.g. monitoring data overlapping the core).")
    else:
        if prof["band"] == 0:
            design = ("Nested cross-validation (outer loop: evaluation; inner loop: tuning); "
                      + ("leave-one-out outer loop because n_eff < 30." if prof["n_eff"] < 30
                         else "repeated k-fold (k = 5–10) outer loop."))
        elif prof["band"] < 3:
            design = "Repeated k-fold cross-validation (k = 5–10)."
        else:
            design = "k-fold cross-validation or a train–test split (e.g. 70:30)."

    if prof["band"] == 0 and design and not design.startswith("Nested"):
        notes.append("n_eff:p < 1: tune hyperparameters inside the training folds only (nested design) "
                     "and use permutation tests to check skill against chance.")
    if prof["m"] > 1 and not future and not hindcast:
        notes.append("Repeated samples per site: keep all samples from a site in the same fold.")
    if prof["external"]:
        notes.insert(0, "An independent dataset is available: use it as the final test of transferability.")
    return design, notes


def performance_metrics(prof):
    obj, metrics = prof["objective"], []
    if obj == "Species occurrence and abundance":
        if prof["subtype"] == OCCURRENCE_SUBTYPES[0]:
            metrics += ["AUC (discrimination)", "Calibration plot / calibration slope", "Brier score"]
        else:
            metrics += ["RMSE and MAE (on the modelled scale)", "Deviance or log-likelihood for count models"]
    elif obj == "Community composition":
        metrics += ["Bray–Curtis dissimilarity between predicted and observed communities",
                    "Procrustes correlation", "Per-taxon AUC or RMSE summarised across taxa"]
    elif obj == "Taxonomic, functional or phylogenetic diversity":
        metrics += ["RMSE and MAE", "R² between predicted and observed diversity"]
    elif obj == "Ecological interactions":
        metrics += ["Precision and recall of recovered associations against known interactions",
                    "Predictive skill for held-out co-occurrences"]
    elif obj == "Ecosystem state and function":
        metrics += ["RMSE and MAE for continuous state variables", "Accuracy / Brier score for categorical states"]
    else:
        metrics += ["Error in predicted change or effect size", "RMSE and MAE of the response variable"]
    if prof["temporal"].startswith("Early-warning"):
        metrics += ["Hit rate and false-alarm rate", "Warning lead time"]
    if "uncertainty" in prof["required"]:
        metrics += ["Continuous ranked probability score (CRPS)", "Prediction-interval coverage"]
    baseline = ("persistence or climatology forecast" if prof["temporal"] in FUTURE_MODES
                else "null or intercept-only model")
    metrics.append(f"Skill relative to a baseline ({baseline})")
    return metrics


UNCERTAINTY_BY_CLASS = {
    "Statistical": "Confidence and prediction intervals",
    "Bayesian hierarchical": "Posterior credible intervals and posterior predictive distributions",
    "State-space and dynamic occupancy": "Filtered/forecast state distributions with separate observation and process error",
    "Data assimilation": "Ensemble or particle spread updated at each assimilation step",
    "Machine learning": "Quantile regression, bootstrap or conformal prediction intervals; ensemble spread",
    "Process-based": "Parameter and scenario ensembles; sensitivity analysis",
    "Hybrid": "Ensembles over process parameters and learned components",
}


def uncertainty_plan(prof, results):
    retained = results["Recommended"] + results["Emerging"] + results["Use with caution"]
    classes = []
    for r in retained:
        if r["method_class"] not in classes:
            classes.append(r["method_class"])
    outputs = [f"{c}: {UNCERTAINTY_BY_CLASS[c]}" for c in classes]
    sources = ["Observation (including eDNA detection and taxonomic assignment)", "Environmental drivers",
               "Parameter estimation", "Model structure (compare or ensemble retained methods)"]
    if prof["temporal"].startswith("Long-term"):
        sources.append("Future scenarios")
    if "mechanistic" in prof["required"]:
        sources.append("Incomplete ecological knowledge")
    return outputs, sources


def plausibility_flag(prof, results):
    reasons = []
    if "extrapolation" in prof["required"]:
        reasons.append("predictions extend beyond observed spatial, temporal or environmental conditions")
    data_driven = [r["name"] for r in results["Recommended"] + results["Emerging"] + results["Use with caution"]
                   if r["method_class"] == "Machine learning"]
    if data_driven:
        reasons.append("predominantly data-driven methods are retained")
    return bool(reasons), reasons


def taxonomy_flag(prof):
    if prof["taxonomy"] < TAXONOMY_FLAG_THRESHOLD:
        return (f"Only {prof['taxonomy']}% of sequences assigned to species: consider functional, "
                "phylogenetic or community-level responses rather than species-level forecasts (Box 1).")
    return None


def evaluation_plan(prof, results):
    design, notes = validation_design(prof)
    outputs, sources = uncertainty_plan(prof, results)
    flag, flag_reasons = plausibility_flag(prof, results)
    return dict(design=design, notes=notes, metrics=performance_metrics(prof),
                uncertainty_outputs=outputs, uncertainty_sources=sources,
                plausibility=flag, plausibility_reasons=flag_reasons, taxonomy=taxonomy_flag(prof))


# =============================================================================
# 8. CHARTS
# =============================================================================
STATUS_COLOURS = {"Recommended": "#2e7d32", "Emerging": "#1565c0",
                  "Use with caution": "#ef6c00", "Excluded": "#c62828"}


def create_ratio_gauge(ratio):
    band = NP_BANDS[np_band(ratio)]
    colours = ["#e53935", "#fb8c00", "#fdd835", "#43a047"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=min(ratio, 20),
        number={"font": {"size": 36}, "valueformat": ".2f"},
        title={"text": f"Effective n:p = {ratio:.2f}<br><span style='font-size:0.8em'>{band['name']} "
                       f"(band {band['label']})</span>", "font": {"size": 16}},
        gauge={"axis": {"range": [0, 20], "tickvals": [0, 1, 5, 10, 20]},
               "bar": {"color": colours[np_band(ratio)], "thickness": 0.6},
               "steps": [{"range": [0, 1], "color": "#ffcdd2"}, {"range": [1, 5], "color": "#ffe0b2"},
                         {"range": [5, 10], "color": "#fff9c4"}, {"range": [10, 20], "color": "#c8e6c9"}]}))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=70, b=10))
    return fig


def create_decision_path(prof, results, plan):
    """Four boxes, one per tier, summarising what each tier decided."""
    short = lambda s: s.split(" – ")[0]
    counts = {s: len(results[s]) for s in STATUS_ORDER}
    boxes = [
        ("Tier 1<br>Problem definition",
         f"{prof['objective'][:34]}<br>{short(prof['spatial'])}<br>{short(prof['temporal'])}"),
        ("Tier 2<br>Constraints",
         f"n = {prof['n_raw']}, n_eff = {prof['n_eff']:.1f}<br>p = {prof['p']}<br>"
         f"n_eff:p = {prof['ratio']:.2f} ({NP_BANDS[prof['band']]['label']})"),
        ("Tier 3<br>Method filtering",
         f"Recommended {counts['Recommended']}<br>Emerging {counts['Emerging']}<br>"
         f"Caution {counts['Use with caution']} · Excluded {counts['Excluded']}"),
        ("Tier 4<br>Evaluation",
         plan["design"].split(":")[0].split("(")[0][:40]
         + ("<br>Plausibility check flagged" if plan["plausibility"] else "")),
    ]
    fig = go.Figure()
    fills = ["#ffebee", "#fff3e0", "#e8f5e9", "#e3f2fd"]
    for i, (title, body) in enumerate(boxes):
        x0 = i * 0.25 + 0.01
        fig.add_shape(type="rect", x0=x0, x1=x0 + 0.22, y0=0.05, y1=0.95,
                      fillcolor=fills[i], line=dict(color="#555", width=1.5))
        fig.add_annotation(x=x0 + 0.11, y=0.82, text=f"<b>{title}</b>", showarrow=False, font=dict(size=12))
        fig.add_annotation(x=x0 + 0.11, y=0.42, text=body, showarrow=False, font=dict(size=10.5))
        if i < 3:
            fig.add_annotation(x=x0 + 0.245, y=0.5, ax=x0 + 0.222, ay=0.5, xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=2)
    fig.update_layout(height=260, margin=dict(l=5, r=5, t=10, b=10), showlegend=False,
                      xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False, range=[0, 1]),
                      plot_bgcolor="white")
    return fig


RADAR_AXES = ["Capability match", "Interpretability", "Data efficiency",
              "Low compute demand", "Robustness to missing data"]


def radar_values(r):
    return [r["score"] * 100, r["interpretability"] / 3 * 100, (3 - r["pref_band"]) / 3 * 100,
            (4 - r["compute"]) / 3 * 100, r["robustness"] / 3 * 100]


def create_method_radar(results, top=4):
    """Radar of the top retained methods, on axes derived from the Tier 2 dimensions."""
    shown = (results["Recommended"] + results["Emerging"] + results["Use with caution"])[:top]
    fig = go.Figure()
    for r in shown:
        v = radar_values(r)
        fig.add_trace(go.Scatterpolar(r=v + [v[0]], theta=RADAR_AXES + [RADAR_AXES[0]],
                                      name=r["name"], fill="toself", opacity=0.5))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=480,
                      title="Top retained methods across Tier 2 dimensions")
    return fig


def create_suitability_matrix(results):
    """Interpretability vs capability match for all retained methods, coloured by class."""
    fig = go.Figure()
    for status in STATUS_ORDER[:3]:
        rs = results[status]
        if not rs:
            continue
        x = [r["interpretability"] + (i % 5 - 2) * 0.07 for i, r in enumerate(rs)]
        fig.add_trace(go.Scatter(
            x=x, y=[r["score"] for r in rs], mode="markers+text", name=status,
            text=[r["name"].split(" (")[0] for r in rs], textposition="top center", textfont=dict(size=9),
            marker=dict(size=14, color=STATUS_COLOURS[status], opacity=0.75),
            hovertemplate="<b>%{text}</b><br>Capability match: %{y:.2f}<extra></extra>"))
    fig.update_layout(height=500, title="Retained methods: interpretability vs capability match",
                      xaxis=dict(title="Interpretability", tickvals=[1, 2, 3],
                                 ticktext=["Low", "Medium", "High"], range=[0.5, 3.5]),
                      yaxis=dict(title="Capability match (S)", range=[0, 1.1]))
    fig.add_hline(y=SCORE_RECOMMEND, line_dash="dash", line_color="grey")
    return fig


# =============================================================================
# 10. EXPORT  (kept beside the logic so it can be tested without Streamlit)
# =============================================================================
def results_table(results):
    rows = []
    for status in STATUS_ORDER:
        for r in results[status]:
            rows.append({"Status": status, "Method": r["name"], "Method class": r["method_class"],
                         "Capability match (S)": round(r["score"], 2), "Reasons": "; ".join(r["reasons"]),
                         "Implementation": r["packages"], "Key references": r["refs"],
                         "Manuscript examples": r["manuscript_examples"]})
    return pd.DataFrame(rows)


def build_report(prof, results, plan):
    L = []
    add = L.append
    add("BIODIVERSITY FORECASTING DECISION FRAMEWORK – REPORT")
    add("=" * 64)
    add(f"Framework: {CITATION}")
    add(f"App version {APP_VERSION}\n")
    add("TIER 1  PROBLEM DEFINITION")
    add(f"  Ecological objective : {prof['objective']}" + (f" – {prof['subtype']}" if prof.get("subtype") else ""))
    add(f"  Spatial domain       : {prof['spatial']}")
    add(f"  Temporal mode        : {prof['temporal']}")
    add(f"  Required capabilities: {_cap_names(prof['required']) or 'none beyond data constraints'}\n")
    add("TIER 2  DATA AND ANALYTICAL CONSTRAINTS")
    add(f"  Sites × time points  : {prof['n_sites']} × {prof['m']} ({prof['time_unit']}) = n {prof['n_raw']}")
    add(f"  Lag-1 autocorrelation: {prof['rho']}")
    add(f"  Missing data         : {prof['missing']*100:.0f}%" + (" ; high noise flagged" if prof["noise"] else ""))
    add(f"  Effective n (n_eff)  : {prof['n_eff']:.1f}")
    add(f"  Features (p)         : {prof['p']}")
    add(f"  n_eff:p              : {prof['ratio']:.2f}  (Table B1 band {NP_BANDS[prof['band']]['label']}, "
        f"{NP_BANDS[prof['band']]['risk'].lower()} overfitting risk)")
    add(f"  Taxonomic resolution : {prof['taxonomy']}% assigned to species")
    add(f"  Interpretability     : {prof['interp_label']}")
    add(f"  Computing resources  : {prof['compute_label']}\n")
    add("TIER 3  METHOD FILTERING AND PRIORITISATION")
    add("  No single method is assumed to meet all requirements; several candidates are retained for evaluation.")
    for status in STATUS_ORDER:
        add(f"\n  {status.upper()}")
        if not results[status]:
            add("    (none)")
        for r in results[status]:
            add(f"    - {r['name']}  [S = {r['score']:.2f}]")
            add(f"        {'; '.join(r['reasons'])}")
            if status != "Excluded":
                add(f"        Implementation: {r['packages']}")
                add(f"        References: {r['refs']}")
    add("\nTIER 4  EVALUATION, UNCERTAINTY AND REPORTING")
    add(f"  Validation design: {plan['design']}")
    for n in plan["notes"]:
        add(f"    * {n}")
    add("  Performance metrics:")
    for mtr in plan["metrics"]:
        add(f"    - {mtr}")
    add("  Uncertainty outputs:")
    for u in plan["uncertainty_outputs"]:
        add(f"    - {u}")
    add("  Uncertainty sources to distinguish: " + "; ".join(plan["uncertainty_sources"]))
    if plan["plausibility"]:
        add("  ECOLOGICAL PLAUSIBILITY: priority check (" + "; ".join(plan["plausibility_reasons"]) + "). "
            "Compare predicted relationships with published knowledge, expert judgement or process models.")
    if plan["taxonomy"]:
        add(f"  NOTE: {plan['taxonomy']}")
    add("\n  Report: objective and domain; data provenance; n, n_eff, p and dependence structure; "
        "rationale for method selection; validation design and metrics; uncertainty; degree of "
        "extrapolation; limitations; code and data availability.")
    return "\n".join(L)


# =============================================================================
# 9. STREAMLIT UI
# =============================================================================
def _defaults():
    return {"example": SELECT, "objective": SELECT, "subtype": OCCURRENCE_SUBTYPES[0],
            "spatial": SELECT, "temporal": SELECT, "n_sites": 10, "timepoints": 1, "time_unit": "Months",
            "p": 50, "rho": "Moderate (ρ ≈ 0.5)", "missing_pct": 0, "noise": False, "taxonomy": 70,
            "interp": SELECT, "compute": SELECT, "external": False}


def _load_example():
    import streamlit as st
    ex = EXAMPLES.get(st.session_state.get("example"))
    if ex:
        for k, v in ex.items():
            st.session_state[k] = v


def render_tier1(st):
    st.markdown("Define **what** is predicted, **where** and **over what time horizon**, before considering "
                "data or methods.")
    st.selectbox("Load a worked example (optional)", [SELECT] + list(EXAMPLES), key="example",
                 on_change=_load_example)
    st.markdown("#### Part 1 · Ecological objective")
    st.selectbox("What is to be predicted?", [SELECT] + list(OBJECTIVES), key="objective")
    obj = st.session_state["objective"]
    if obj != SELECT:
        exs = [q for q, e in EXAMPLES.items() if e["objective"] == obj]
        if exs:
            st.caption("Example questions: " + " · ".join(exs))
        if obj == "Species occurrence and abundance":
            st.radio("Response measured as", OCCURRENCE_SUBTYPES, key="subtype", horizontal=True)

    st.markdown("#### Part 2 · Spatial and temporal structure")
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Spatial domain of the prediction", [SELECT] + list(SPATIAL_DOMAINS), key="spatial")
        st.number_input("Number of sites", min_value=1, max_value=100000, step=1, key="n_sites")
    with c2:
        st.selectbox("Temporal mode", [SELECT] + list(TEMPORAL_MODES), key="temporal")
        st.selectbox("Temporal resolution (sampling interval)", TIME_UNITS, key="time_unit")
        st.number_input("Time points per site", min_value=1, max_value=100000, step=1, key="timepoints",
                        help="1 = a single survey per site (snapshot).")

    ss = st.session_state
    t1 = {k: ss[k] for k in ("objective", "subtype", "spatial", "temporal", "n_sites", "timepoints", "time_unit")}
    if obj != "Species occurrence and abundance":
        t1["subtype"] = None
    if SELECT not in (ss["objective"], ss["spatial"], ss["temporal"]):
        req = required_capabilities(ss["objective"], ss["spatial"], ss["temporal"])
        st.info("**Capabilities this problem requires:** "
                + ("; ".join(CAPABILITIES[c] for c in req) if req
                   else "none beyond the data constraints in Tier 2."))
    return t1


def render_tier2(st, t1):
    ss = st.session_state
    st.markdown("Characterise the data across the seven dimensions of Box 1.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1 · Sample size** (from Tier 1)")
        st.write(f"{t1['n_sites']} sites × {t1['timepoints']} time points = **{t1['n_sites'] * t1['timepoints']} samples**")
        st.markdown("**2 · Feature dimensionality**")
        p_help = (OBJECTIVES[ss['objective']]['p_help'] if ss["objective"] != SELECT
                  else "Number of features (taxa/ASVs, environmental variables) entering the model.")
        st.number_input("Number of features (p)", min_value=1, max_value=1000000, step=1, key="p", help=p_help)
        st.caption(f"Dimensionality: {feature_label(ss['p'])}. {p_help}")
        st.markdown("**3 · Temporal resolution** (from Tier 1)")
        st.selectbox("Correlation between consecutive samples at a site", list(RHO_OPTIONS), key="rho",
                     disabled=t1["timepoints"] <= 1,
                     help="Lag-1 autocorrelation. Repeated samples at one site are not independent, so they "
                          "count as fewer effective samples.")
        st.markdown("**4 · Taxonomic resolution** (reported, not scored)")
        st.slider("Sequences assigned to species level (%)", 0, 100, key="taxonomy")
    with c2:
        st.markdown("**5 · Data quality**")
        st.slider("Missing values (%)", 0, 80, step=5, key="missing_pct")
        st.checkbox("High measurement noise or detection uncertainty", key="noise")
        st.markdown("**6 · Interpretability requirement**")
        st.selectbox("Interpretability", [SELECT] + list(INTERP_OPTIONS), key="interp")
        st.markdown("**7 · Computational resources**")
        st.selectbox("Computing", [SELECT] + list(COMPUTE_OPTIONS), key="compute")
        st.markdown("**Validation data**")
        st.checkbox("An independent dataset (other sites or periods) is available for testing", key="external")
    return {k: ss[k] for k in ("p", "rho", "missing_pct", "noise", "taxonomy", "interp", "compute", "external")}


def render_tier2_summary(st, prof):
    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(create_ratio_gauge(prof["ratio"]))
    with c2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Samples (n)", prof["n_raw"])
        m2.metric("Effective n", f"{prof['n_eff']:.1f}")
        m3.metric("Features (p)", prof["p"])
        b = NP_BANDS[prof["band"]]
        st.markdown(f"**Table B1 band {b['label']}** – {b['risk'].lower()} overfitting risk.  \n"
                    f"Preferred approaches: {b['preferred']}.")
        if prof["m"] > 1:
            st.caption(f"Each site's {prof['m']} time points count as {prof['n_eff_per_site']:.1f} "
                       f"independent samples (ρ = {prof['rho']}).")


def render_method(st, r, status):
    with st.expander(f"**{r['name']}**  ·  S = {r['score']:.2f}  ·  {r['method_class']}"):
        st.write(r["description"])
        st.markdown("**Why:** " + "; ".join(r["reasons"]))
        if status == "Recommended" and r["partial"]:
            st.caption("Partial capabilities: " + _cap_names(r["partial"]))
        st.code(f"Implementation: {r['packages']}", language="text")
        st.caption(f"References: {r['refs']}  |  Manuscript examples: {r['manuscript_examples']}")


def render_tier3(st, prof, results, plan):
    st.plotly_chart(create_decision_path(prof, results, plan))
    st.info("The framework does not assume a single method meets every requirement: several candidates "
            "are retained so they can be compared or combined in Tier 4.")
    headers = {"Recommended": "✅ Recommended", "Emerging": "⚡ Emerging",
               "Use with caution": "⚠️ Use with caution", "Excluded": "❌ Excluded"}
    blurbs = {"Recommended": "Assumptions and requirements align with your problem; established in comparable conditions.",
              "Emerging": "Could meet the requirements, but limited application in biodiversity forecasting or needs specialised expertise.",
              "Use with caution": "Potentially informative, with identifiable risks that need safeguards.",
              "Excluded": "Fundamental assumptions or data requirements cannot be supported."}
    for status in STATUS_ORDER:
        st.subheader(f"{headers[status]} ({len(results[status])})")
        st.caption(blurbs[status])
        if status == "Excluded":
            for r in results[status]:
                st.markdown(f"- **{r['name']}**: {'; '.join(r['reasons'])}")
        else:
            for r in results[status]:
                render_method(st, r, status)
    if any(results[s] for s in STATUS_ORDER[:3]):
        t_a, t_b = st.tabs(["Method comparison", "Suitability matrix"])
        with t_a:
            st.plotly_chart(create_method_radar(results))
            st.caption("Axes: capability match (S); interpretability; data efficiency (lower preferred n:p band "
                       "= higher); low compute demand; robustness to missing data. Scores are heuristics.")
        with t_b:
            st.plotly_chart(create_suitability_matrix(results))
            st.caption("Dashed line: S threshold for recommendation (0.67).")


def render_tier4(st, prof, results, plan):
    st.markdown("#### Validation design")
    st.write(plan["design"])
    for n in plan["notes"]:
        st.markdown(f"- {n}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Performance metrics")
        for mtr in plan["metrics"]:
            st.markdown(f"- {mtr}")
    with c2:
        st.markdown("#### Uncertainty")
        for u in plan["uncertainty_outputs"]:
            st.markdown(f"- {u}")
        st.caption("Distinguish: " + "; ".join(plan["uncertainty_sources"]))
    st.markdown("#### Ecological plausibility")
    if plan["plausibility"]:
        st.warning("Priority check: " + "; ".join(plan["plausibility_reasons"]) + ". Compare predicted "
                   "relationships with published knowledge, expert judgement or process-based models. "
                   "This assessment is made by the user, not the tool.")
    else:
        st.write("Standard check: confirm predicted relationships are ecologically sensible.")
    if plan["taxonomy"]:
        st.warning(plan["taxonomy"])
    st.markdown("#### Reporting profile and export")
    st.write("Download a record of every choice, the n_eff:p calculation, each method's classification "
             "and reasons, and the evaluation plan.")
    d1, d2 = st.columns(2)
    d1.download_button("📥 Download report (TXT)", build_report(prof, results, plan),
                       file_name="forecasting_framework_report.txt", mime="text/plain")
    d2.download_button("📥 Download method table (CSV)", results_table(results).to_csv(index=False),
                       file_name="forecasting_framework_methods.csv", mime="text/csv")


# =============================================================================
# 11. HELP TEXT, FOOTER AND LAUNCHER
# =============================================================================
def render_about(st):
    st.markdown("#### Effective sample size and n_eff:p")
    st.write("Repeated samples from one site are autocorrelated, so they carry less independent information. "
             "Each site's series is converted to an effective number of samples using the AR(1) "
             "variance-inflation formula, then reduced by the proportion of missing values:")
    st.latex(r"n_{eff} = n_{sites}\cdot\frac{m}{1 + 2\sum_{k=1}^{m-1}\left(1-\frac{k}{m}\right)\rho^{k}}\cdot(1-f_{missing})")
    st.write("where m is time points per site and ρ the lag-1 autocorrelation. n_eff:p is then placed in the "
             "Table B1 bands (< 1, 1–5, 5–10, > 10). Spatial autocorrelation between sites is not corrected.")
    st.markdown("#### Classification rules (Tier 3)")
    st.markdown(
        "- **Capability match S** = mean level (1 full, 0.5 partial, 0 none) of the capabilities required by Tier 1.\n"
        "- **Excluded**: wrong response type, n_eff:p below the method's minimum band, computing above available, "
        "black-box under a high interpretability need, time-series method without a temporal mode or with < 3 time "
        "points, or S < 0.33.\n"
        "- **Recommended**: S ≥ 0.67, no penalties, established. **Emerging**: same, but emerging maturity.\n"
        "- **Use with caution**: 0.33 ≤ S < 0.67, or any penalty (below preferred band; low robustness when > 20 % of "
        "data are missing or noise is high; medium interpretability need with a black box; too few time points).")
    st.caption("All thresholds and method scores are practical decision-support heuristics, not universal "
               "statistical rules (Box 1).")


def render_footer(st):
    st.markdown("---")
    st.markdown(f"<div style='text-align:center'><p>Developed by Arron Watson · School of Biosciences, "
                f"University of Birmingham · Version {APP_VERSION} · {APP_YEAR}</p>"
                f"<p>Contact: axw213@student.bham.ac.uk · Report issues: [GitHub link]</p>"
                f"<p style='font-size:0.85em'>{CITATION}</p></div>", unsafe_allow_html=True)


def main():
    import streamlit as st
    st.set_page_config(page_title="Biodiversity Forecasting Decision Framework", page_icon="🧬", layout="wide")
    for k, v in _defaults().items():
        st.session_state.setdefault(k, v)

    st.title("Biodiversity Forecasting Decision Framework")
    st.markdown("Work through the four tiers of the framework (Watson et al., 2026). Results update as you "
                "change inputs; Tiers 3 and 4 appear once Tiers 1 and 2 are complete.")
    tabs = st.tabs(["Tier 1 · Problem definition", "Tier 2 · Data and constraints",
                    "Tier 3 · Method filtering", "Tier 4 · Evaluation and reporting", "About the calculation"])
    with tabs[0]:
        t1 = render_tier1(st)
    with tabs[1]:
        t2 = render_tier2(st, t1)
    missing = [lbl for lbl, v in (("objective", t1["objective"]), ("spatial domain", t1["spatial"]),
                                  ("temporal mode", t1["temporal"]), ("interpretability", t2["interp"]),
                                  ("computing", t2["compute"])) if v == SELECT]
    prof = results = plan = None
    if not missing:
        prof = build_profile(t1, t2)
        results = score_methods(prof)
        plan = evaluation_plan(prof, results)
        with tabs[1]:
            render_tier2_summary(st, prof)
    msg = "Complete Tiers 1 and 2 first. Still needed: " + ", ".join(missing) + "."
    with tabs[2]:
        render_tier3(st, prof, results, plan) if prof else st.info(msg)
    with tabs[3]:
        render_tier4(st, prof, results, plan) if prof else st.info(msg)
    with tabs[4]:
        render_about(st)
    render_footer(st)


def _running_inside_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:  # very old or very new Streamlit layouts
        try:
            from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
        except ImportError:
            return False
    return get_script_run_ctx() is not None


if __name__ == "__main__":
    if _running_inside_streamlit():
        main()
    else:
        # `python edna_forecast_app.py` -> relaunch this file under Streamlit
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())
