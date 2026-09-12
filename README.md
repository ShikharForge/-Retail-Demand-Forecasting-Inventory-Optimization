# Retail Demand Forecasting & Inventory Optimization

> **Data Science Portfolio Project** — End-to-end decision-support system that converts multi-horizon demand forecasts into statistically grounded inventory policies using quantile regression for uncertainty quantification.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Architecture](#architecture)
4. [Methodology](#methodology)
   - [Feature Engineering](#feature-engineering)
   - [Model Selection](#model-selection)
   - [Quantile Regression](#quantile-regression)
   - [Inventory Optimization](#inventory-optimization)
5. [Key Results](#key-results)
6. [Statistical Analysis](#statistical-analysis)
7. [Modelling Assumptions](#modelling-assumptions)
8. [Project Structure](#project-structure)
9. [Running the Project](#running-the-project)
10. [Dashboard](#dashboard)
11. [Interview Q&A](#interview-qa)

---

## Project Overview

This project builds an **original** retail demand forecasting and inventory optimization system. It is **not** a reproduction of any existing repository — the architecture, feature engineering choices, model design, and inventory formulation were all designed independently.

### What this system does

```
Raw retail data
    │
    ▼
Feature Engineering ──► 39 leakage-safe features (lags, rolling stats, calendar, economic)
    │
    ▼
Forecasting Layer ─────► LightGBM / XGBoost global panel models
                          Quantile regression: P10 (lower bound), P50 (point), P90 (upper bound)
    │
    ▼
Inventory Layer ────────► Safety stock, Reorder Point, EOQ, Stockout Risk
                          Two methods: Parametric (Normal) + Non-parametric (quantile-based)
    │
    ▼
Scenario Engine ────────► 5 what-if scenarios (demand shock, lead time disruption, etc.)
    │
    ▼
Explainability ─────────► SHAP global + local feature attribution
    │
    ▼
Dashboard ──────────────► 6-page Streamlit app with interactive controls
```

---

## Dataset

**Source:** Walmart Store Sales Forecasting (Kaggle)

| Property | Value |
|---|---|
| Records | 6,435 rows |
| Stores | 45 |
| Time span | 2010-02-05 → 2012-10-26 (143 weeks per store) |
| Frequency | Weekly (Friday-anchored) |
| Target variable | `weekly_sales` (USD, store-level aggregate) |
| Missing values | **Zero** |
| Duplicates | **Zero** |
| Negative/zero sales | **Zero** |

### Observed data distributions
- **Sales range:** $209,986 – $3,818,686 per store per week
- **Mean weekly sales:** $1,046,965 (all stores)
- **Sales skewness:** 0.668 (right-skewed → non-parametric tests preferred)
- **Holiday weeks:** 450 / 6,435 (7.0%)

---

## Architecture

### Global Panel Model

Rather than training 45 separate models (one per store), this project uses a **single global panel model** where `store` is a categorical feature. This decision is justified by three reasons:

1. **Statistical evidence:** Kruskal-Wallis H=6,144.3, p≈0 confirms stores differ — the model must learn store-specific patterns. Store identity as a feature achieves this without requiring 45 separate training jobs.
2. **Cross-store learning:** The model can learn demand patterns (seasonality shape, holiday uplift) that are common across stores, improving predictions for lower-volume stores that have less data.
3. **Practical scalability:** Adding a new store requires only adding rows to the training set, not a new model file.

### Train / Validation / Test Split

```
|──────────── TRAIN (70%) ────────────|── VAL (18%) ──|── TEST (12%) ──|
  2010-02-05            2011-12-30     2012-01-06  2012-06-29  2012-10-26
```

This is a **strict chronological split** — no shuffling, no random splits. This simulates real-world deployment where the model is always predicting forward in time.

---

## Methodology

### Feature Engineering

All features are engineered to be **strictly leakage-free**: at prediction time t, only information from t-1 or earlier is used.

| Feature Group | Features | Leakage Prevention |
|---|---|---|
| Calendar | year, month, quarter, week_of_year, week_sin/cos, month_sin/cos | No future info |
| Holiday decomposition | is_superbowl, is_laborday, is_thanksgiving, is_christmas, weeks_to_thanksgiving, weeks_to_christmas | Calendar-only |
| Lag features | lag_1, lag_4, lag_8, lag_52 | `.shift(1)` per store group |
| Rolling statistics | rolling_mean_4/8/12, rolling_std_4/8/12, rolling_max_4 | `.shift(1).rolling(w)` per store group |
| Economic covariates | fuel_price_chg, cpi_chg, unemployment_yoy | Differenced to avoid level non-stationarity |

**Leakage verification:** `verify_no_leakage()` is run on all 45 stores — it checks that `lag_1[i] == weekly_sales[i-1]` and raises `AssertionError` if any contamination is detected. **45/45 stores passed.**

### Model Selection

| Model | WAPE | MAE | RMSE | Notes |
|---|---|---|---|---|
| Naive (lag-1) | 0.0863 | $89,078 | $117,979 | Baseline 1 |
| Seasonal Naive (lag-52) | 0.0522 | $53,867 | $84,681 | Baseline 2 |
| **LightGBM** | **0.0410** | **$42,336** | **$63,610** | **Primary model** |
| XGBoost | 0.0400 | $41,278 | $60,595 | Alternative |

**LightGBM improvement over Seasonal Naive: 21.4% WAPE reduction.**

Both gradient boosting models significantly outperform both baselines. LightGBM was chosen as primary because:
- Handles categorical features (`store`) natively via `categorical_feature` parameter
- Faster training due to histogram-based splitting
- Marginally higher interpretability (well-supported SHAP integration)
- XGBoost is retained as a comparison model

### Quantile Regression

Three separate LightGBM models are trained with different objective functions:

```python
# P10 model — lower bound (pessimistic forecast)
objective = 'quantile', alpha = 0.10

# P50 model — median forecast (point estimate)
objective = 'quantile', alpha = 0.50

# P90 model — upper bound (optimistic forecast)
objective = 'quantile', alpha = 0.90
```

The **P10–P90 interval** captures demand uncertainty and feeds directly into the safety stock calculation.

**Why quantile regression?**

Standard regression minimises squared error (MSE), producing a single point estimate. For inventory management, we need a **full distribution** of possible demand outcomes — not just the expected value. Quantile regression achieves this efficiently within a single model framework, without needing to assume a parametric distribution for the errors.

**Test-set interval coverage:** 59.6% of actual values fell within the P10–P90 band. The theoretical target is 80%. The gap reflects the limited test period (only ~17 weeks per store after the split) — wider quantiles or more training data would improve coverage.

### Inventory Optimization

> ⚠️ **All inventory parameters are MODELLING ASSUMPTIONS — the Walmart dataset contains no inventory records.**

#### Formulas

**1. Lead-Time Demand**
```
LTD = P50_weekly × lead_time_weeks
```
Expected demand during the replenishment window.

**2. Safety Stock (Quantile-Based — default)**
```
SS_quantile = (P90_weekly - P50_weekly) × √lead_time_weeks
```
Non-parametric: uses the actual forecast uncertainty band rather than assuming Normal demand.

**3. Safety Stock (Parametric — alternative)**
```
SS_parametric = Z(service_level) × σ_weekly × √lead_time_weeks
```
Classical formula. Assumes weekly demand is approximately Normal. Z = 1.645 for 95% SL.

**4. Reorder Point**
```
ROP = LTD + SS
```
Place a new order when inventory falls to ROP.

**5. Economic Order Quantity (Wilson/EOQ formula)**
```
EOQ = √(2DS / H)
where D = annual demand, S = ordering cost, H = holding cost per unit per year
```
Optimal order size that minimises total ordering + holding costs.

**6. Stockout Risk**
```
P(stockout) = 1 - Φ((ROP - μ_LT) / σ_LT)
```
Probability that demand during lead time exceeds the reorder point.

**References:**
- Silver, Pyke & Thomas (2017). *Inventory and Production Management in Supply Chains* (4th ed.)
- Chopra & Meindl (2016). *Supply Chain Management: Strategy, Planning, and Operation* (6th ed.)

---

## Key Results

### Forecasting

| Metric | Value |
|---|---|
| **WAPE** (primary metric) | **4.10%** |
| MAE | $42,336 per store per week |
| Improvement over Seasonal Naive | **21.4%** |
| Top SHAP feature | lag_52 (same week last year, $217K avg impact) |
| 2nd SHAP feature | rolling_mean_4 ($162K avg impact) |

### Explainability (SHAP)

Top features by mean |SHAP value|:

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|---|---|---|---|
| 1 | lag_52 | $217,108 | Same week last year — dominant yearly seasonality |
| 2 | rolling_mean_4 | $162,315 | Recent 4-week trend |
| 3 | lag_4 | $68,393 | Short-term momentum (1 month ago) |
| 4 | store | $10,054 | Store-level baseline demand |
| 5 | lag_1 | $7,019 | Most recent week |
| 9 | unemployment | $5,546 | Macro-economic context |
| 12 | month_sin | $4,040 | Annual seasonality (cyclic encoding) |

> **Note:** SHAP values show *associations*, not causal relationships. High lag_52 pushing the forecast up means the model learned that last year's sales are associated with this year's — it does not mean historical sales *cause* current sales.

### Statistical Tests

| Test | Question | Result |
|---|---|---|
| Mann-Whitney U | Holiday vs non-holiday sales | p=0.026 → **SIGNIFICANT** (effect size r=−0.063, modest) |
| Mann-Whitney U | Thanksgiving vs non-holiday | p≈0 → **HIGHLY SIGNIFICANT** |
| Kruskal-Wallis | All 45 stores equal? | H=6,144, p≈0 → **REJECT** (η²=0.955, large effect) |
| Levene's Test | Equal variance across stores? | W=19.02, p≈0 → **REJECT** (per-store SS justified) |

---

## Statistical Analysis

### Why non-parametric tests?

Sales data has skewness of 0.668 — the distribution is right-skewed. Parametric tests (t-test, ANOVA) assume approximately Normal data. Non-parametric equivalents (Mann-Whitney, Kruskal-Wallis) make no distributional assumption and are therefore more appropriate here.

### Holiday Effect Decomposition

| Event | Mean Weekly Sales | vs Non-Holiday |
|---|---|---|
| Non-holiday | $1.041M | — |
| Super Bowl | $1.079M | +3.7% |
| Labour Day | $1.042M | +0.1% |
| **Thanksgiving** | **$1.471M** | **+41.3%** |
| Christmas | $0.961M | −7.7% |

**Thanksgiving is the dominant event.** Christmas actually shows *below*-average sales at the store-aggregate level, likely because shoppers buy gifts online or at specialist retailers rather than Walmart.

### Economic Variables

All four economic variables (temperature, fuel price, CPI, unemployment) are statistically significant but show only **weak correlations** (|ρ| < 0.1). This confirms that store identity and calendar effects dominate over macroeconomic factors for weekly demand prediction.

---

## Modelling Assumptions

The following are **explicit assumptions**, not facts derived from the data:

| Assumption | Value | Rationale |
|---|---|---|
| Lead time | 2 weeks | Typical grocery/FMCG replenishment cycle |
| Service level | 95% | Standard retail in-stock target |
| Ordering cost | $500/order | Illustrative value; must be calibrated per retailer |
| Holding cost | 25% p.a. | Includes capital, storage, shrinkage, obsolescence |
| Current inventory | User input | Must be provided — not in dataset |
| EOQ unit value | $1 (working in $) | Demand and inventory measured in revenue, not units |

In a production deployment, these parameters would be sourced from:
- **Lead time:** Supplier contracts / purchase order history
- **Ordering cost:** Finance / procurement data
- **Holding cost:** Warehouse ops data + cost of capital
- **Service level:** Business decision (based on margin vs stockout cost trade-off)

---

## Project Structure

```
Demand-Forecasting-and-Inventory-Optimization/
│
├── config/
│   └── config.yaml               # All hyperparameters and assumptions
│
├── src/
│   ├── data/
│   │   ├── loader.py             # Data ingestion + schema validation
│   │   └── preprocessor.py       # Cleaning, encoding, chronological sort
│   ├── features/
│   │   └── engineer.py           # Leakage-safe feature engineering (39 features)
│   ├── models/
│   │   ├── baselines.py          # Naive, Seasonal Naive
│   │   ├── trainer.py            # LightGBM/XGBoost train + chronological split
│   │   ├── forecaster.py         # Point + quantile forecast generation
│   │   └── evaluator.py          # MAE, RMSE, WAPE, MAPE + per-store metrics
│   ├── inventory/
│   │   └── optimizer.py          # Safety stock, ROP, EOQ, stockout risk
│   ├── scenarios/
│   │   └── scenario_engine.py    # 5 what-if scenario analyses
│   └── explainability/
│       └── shap_analysis.py      # SHAP global + local explanations
│
├── dashboard/
│   ├── app.py                    # Streamlit entry point
│   └── pages/
│       ├── 01_Executive_Overview.py
│       ├── 02_Demand_Forecast.py
│       ├── 03_Inventory_Optimization.py
│       ├── 04_Scenario_Analysis.py
│       ├── 05_Model_Performance.py
│       └── 06_Explainability.py
│
├── tests/
│   ├── test_features.py          # Leakage detection, lag correctness (46 tests)
│   ├── test_inventory.py         # Safety stock, EOQ, ROP formulas
│   └── test_evaluator.py         # MAE, RMSE, WAPE, MAPE correctness
│
├── outputs/
│   ├── figures/                  # 14 generated charts (EDA + model + SHAP)
│   ├── forecasts/                # test_forecasts.csv, per_store_metrics.csv
│   └── models/                   # lgbm_point.pkl, lgbm_p10/50/90.pkl, xgb_point.pkl
│
├── run_phase1.py                 # Data pipeline (ingestion → features)
├── run_eda.py                    # EDA charts
├── run_stats.py                  # Statistical hypothesis tests
├── run_models.py                 # Model training + evaluation
├── run_explainability.py         # SHAP analysis
└── requirements.txt
```

---

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Phase 1: Data ingestion, cleaning, feature engineering
python run_phase1.py

# Phase 2: EDA (generates outputs/figures/eda_*.png)
python run_eda.py

# Phase 3: Statistical analysis
python run_stats.py

# Phase 4-5: Train all models (baselines, LightGBM point + quantile, XGBoost)
python run_models.py

# Phase 8: SHAP explainability
python run_explainability.py
```

### 3. Run tests

```bash
python -m pytest tests/ -v
# Expected: 46 passed
```

### 4. Launch dashboard

```bash
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

---

## Dashboard

The Streamlit dashboard has 6 pages:

| Page | What it shows |
|---|---|
| 📊 Executive Overview | Aggregate KPIs, sales trend, store rankings, holiday impact |
| 🔮 Demand Forecast | Store-level historical + forecast + P10-P90 uncertainty band |
| 📦 Inventory Optimization | Safety stock, ROP, EOQ, stockout gauge with interactive parameters |
| 🎲 Scenario Analysis | 5 what-if scenarios: demand surge/drop, lead time disruption, etc. |
| 📈 Model Performance | Actual vs predicted scatter, per-store WAPE, model comparison |
| 🔍 Explainability | SHAP feature importance, beeswarm, waterfall (with causality disclaimer) |

---

## Interview Q&A

### "Why a global panel model instead of 45 separate models?"

The Kruskal-Wallis test (H=6,144, p≈0) confirms stores differ significantly — so the model must capture store heterogeneity. Rather than 45 separate training jobs, we include `store` as a categorical feature in a single global model. This enables **cross-store learning** (the model learns that all stores share the same seasonal shape), while the store feature allows it to learn store-specific demand levels. This approach scales better, requires less code, and typically improves predictions for smaller stores that benefit from information borrowed from larger stores.

### "How do you prevent data leakage?"

All lag and rolling features are computed per-store using `groupby('store').shift(1).rolling(w)`. This guarantees that at time t, only values from t-1 or earlier are used. We also have an automated `verify_no_leakage()` function that asserts `lag_1[i] == weekly_sales[i-1]` for every row and raises an `AssertionError` if any violation is found. This was run on all 45 stores and all 45 passed.

### "Why WAPE instead of MAPE?"

MAPE treats a 10% error on a $200K store the same as a 10% error on a $3M store. In retail, we care more about getting large-volume stores right. WAPE weights errors by volume: `WAPE = Σ|actual - forecast| / Σ|actual|`. It also avoids the instability of MAPE when actuals approach zero.

### "Why quantile regression for inventory?"

Standard regression gives you a single point forecast — the expected demand. But inventory decisions need to account for the *distribution* of possible demand outcomes. Safety stock protects against the *upper tail* of demand, not the expectation. Quantile regression lets us directly model P10, P50, and P90 with a single LightGBM model per quantile, without assuming a parametric form for the error distribution.

### "Are the inventory parameters realistic?"

No — and that's explicitly labelled throughout the codebase and dashboard. The Walmart dataset contains no inventory records, so parameters like lead time (2 weeks), ordering cost ($500), and holding cost (25%) are assumptions. In a real deployment, these would come from supplier contracts, finance data, and warehouse operations data. The system is designed to accept any values via config sliders.

### "How would you improve this in production?"

1. **More data:** Product-level (SKU) rather than store-aggregate forecasts
2. **External signals:** Weather forecasts, promotional calendars, web traffic
3. **Hierarchical models:** Reconcile forecasts across store → region → national hierarchy (e.g., using `statsforecast` HierarchicalReconciliation)
4. **Online learning:** Retrain weekly on rolling window as new data arrives
5. **Causal inference:** Use difference-in-differences or synthetic controls to measure true promotional lift rather than observational correlation
6. **Coverage calibration:** Use conformal prediction to guarantee the P10-P90 interval covers 80% of actuals
