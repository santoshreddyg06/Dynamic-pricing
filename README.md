# 🚗 Dynamic Pricing & Fare Prediction System for Ride-Hailing Services

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB5424?style=for-the-badge)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Colab](https://img.shields.io/badge/Google--Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

An end-to-end Machine Learning and Dynamic Pricing decision support system trained on over 690,000 Uber and Lyft ride transactions across 12 Boston locations. The system predicts accurate ride fares, analyzes weather and spatio-temporal elasticity, and delivers real-time surge pricing recommendations through an interactive Streamlit web dashboard.

---

## 📋 Table of Contents
- [Executive Overview](#-executive-overview)
- [Dataset Characteristics](#-dataset-characteristics)
- [System Architecture & Workflow](#-system-architecture--workflow)
- [Project Directory Structure](#-project-directory-structure)
- [Feature Engineering & Schema](#-feature-engineering--schema)
- [Machine Learning Models & Evaluation](#-machine-learning-models--evaluation)
- [Key Findings](#-key-findings)
- [Pricing Recommendation Engine](#-pricing-recommendation-engine)
- [Setup & Execution Guide](#-setup--execution-guide)
- [Requirements](#-requirements)
- [Technical Architecture & Verification Standards](#-technical-architecture--verification-standards)

---

## 💡 Executive Overview

Ride-hailing platforms (Uber, Lyft) operate in dynamic two-sided marketplaces where supply (drivers) and demand (riders) continuously fluctuate based on:
1. **Spatio-Temporal Patterns**: Rush hours (7–9 AM, 4–7 PM), late nights, weekdays vs. weekends, and origin-destination route density.
2. **Weather Anomalies**: Rain, freezing temperatures, wind gusts, and cloud cover affecting customer willingness to pay and driver turnout.
3. **Product Tier Differentiation**: Shared rides, standard sedans (UberX, Lyft), extra-capacity vehicles (UberXL, Lyft XL), and luxury fleets (Lux Black, Black SUV).

This project implements:
- Comprehensive Exploratory Data Analysis (EDA) uncovering price variance drivers.
- Advanced Feature Engineering incorporating temporal cyclic encoding, route proxies, and DarkSky weather variables.
- Comparative Machine Learning benchmarking across Linear Regression, Random Forest, Gradient Boosting, and XGBoost.
- Dynamic Fare Recommender with pricing tier classification and demand-responsive surge suggestions.
- Interactive Streamlit Dashboard for real-time inference, dataset inspection, and model auditing.

---

## 📊 Dataset Characteristics

The system utilizes the Uber and Lyft Boston Dataset (`rideshare_kaggle.csv`):
- **Total Records**: 693,071 ride queries
- **Feature Dimensionality**: 57 raw attributes (timestamps, spatial endpoints, service classes, DarkSky hourly meteorological metrics)
- **Collection Window**: 17 dates between 26 Nov and 18 Dec 2018 (with gaps on 5–8 and 11–12 Dec) — a ~3-week scrape, not a continuous 2-month history.
- **Target Variable**: `price` (Continuous, USD). Minimum: `$2.50`, Maximum: `$97.50`, Mean: `$16.55`, Median: `$13.50` (right-skewed, skewness 1.05).
- **Missing Values**: 55,095 records in `price` correspond exclusively to `cab_type == 'Uber'` under `name == 'Taxi'`, where Uber functions solely as an e-hail dispatcher with municipal meter pricing. These rows are appropriately dropped during supervised learning preprocessing.
- **Geographic Coverage**: 12 key Boston hubs (e.g., Boston University, Fenway, Financial District, South Station, North End, Back Bay).
- **Service Providers**: Uber (7 products) and Lyft (6 products).
- **Surge Factors**: Only Lyft surges — `surge_multiplier` from `1.0x` to `3.0x` on 6.8% of Lyft rides; Uber is always `1.0x` in this dataset.

---

## 🏗 System Architecture & Workflow

```
+-----------------------------------------------------------------------------------+
|                           1. Data Ingestion & Preprocessing                       |
|   rideshare_kaggle.csv  -->  Drop Unpriced Taxis (55k)  -->  DarkSky Weather Trim |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           2. Feature Engineering Pipeline                         |
|  - Temporal Extraction (hour, day_of_week, is_weekend, is_peak_hour, is_night)    |
|  - Categorical Label Encoding (cab_type, name, source, destination, weather)      |
|  - Domain Metrics (price_per_mile, temp_category bins)                            |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        3. Model Training & Validation (80/20)                     |
|  StandardScaler -> LinearRegression | RandomForest | GradientBoosting | XGBoost   |
|  Metrics: MAE, RMSE, R² Score | 5-Fold Cross Validation | Feature Importance     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                 4. Decision Support & Interactive Application Layer               |
|  - Best Model & Scaler Export (best_model.pkl, scaler.pkl, label_encoders.pkl)    |
|  - Rule-Based Tier Segmentation (Budget, Standard, Premium, Luxury)               |
|  - Interactive Multi-Page Dashboard & Estimation UI (app.py)                     |
+-----------------------------------------------------------------------------------+
```

---

## 📁 Project Directory Structure

```
Dynamic_Pricing/
│
├── dynamic_pricing_notebook.ipynb           # Original Jupyter notebook (Cells 1-18)
├── dynamic_pricing_notebook_improved.ipynb  # Extended EDA, honest CV & diagnostics (executed, with outputs)
├── CHANGES.md                               # What changed in the improved analysis and why
├── train_models.py                 # Standalone CLI training pipeline
├── app.py                          # Streamlit Multi-Page Interactive Dashboard
├── requirements.txt                # Production environment dependencies
├── README.md                       # Comprehensive system architecture & documentation
├── rideshare_kaggle.csv            # Kaggle Boston Uber/Lyft dataset (~367 MB)
│
├── best_model.pkl                  # Serialized best regressor (XGBoost / RF)
├── scaler.pkl                      # Fitted StandardScaler instance
├── label_encoders.pkl              # Fitted LabelEncoders dictionary
└── model_metrics.json              # Benchmark evaluation summary across all models
```

---

## 🔬 Feature Engineering & Schema

The predictive pipeline isolates 27 core numerical, categorical, and engineered features:

| Feature Name | Type | Source / Derivation | Description & Business Rationale |
| :--- | :--- | :--- | :--- |
| `distance` | Float | Raw Dataset | Trip distance in statute miles. Fundamental base cost driver. |
| `surge_multiplier` | Float | Raw Dataset | Real-time demand multiplier (1.0x - 3.0x). Direct price scalar. |
| `hour` | Integer | `datetime.dt.hour` | Hour of day (0–23). Captures daily commute and nightlife demand. |
| `day` | Integer | Raw / Datetime | Day of the month (1–31). Identifies calendar patterns. |
| `month` | Integer | Raw / Datetime | Month of observation (November / December winter period). |
| `day_of_week` | Integer | `datetime.dt.dayofweek` | Day index (0=Monday, 6=Sunday). Differentiates workdays & weekends. |
| `is_weekend` | Binary | Engineered | Flag: 1 if Saturday or Sunday, else 0. Leisure travel indicator. |
| `is_peak_hour` | Binary | Engineered | Flag: 1 if hour in [7,8,9, 16,17,18,19], else 0. Commuter peak surge. |
| `is_night` | Binary | Engineered | Flag: 1 if hour < 6 or >= 22, else 0. Late-night off-peak indicator. |
| `temperature` | Float | Raw (DarkSky) | Ambient temperature in °F. Affects rider willingness to walk. |
| `apparentTemperature` | Float | Raw (DarkSky) | "Feels like" temperature factoring wind chill and humidity. |
| `humidity` | Float | Raw (DarkSky) | Relative humidity percentage (0.0 to 1.0). |
| `windSpeed` | Float | Raw (DarkSky) | Wind velocity in mph. Inclement weather proxy. |
| `windGust` | Float | Raw (DarkSky) | Peak momentary wind speed in mph. |
| `visibility` | Float | Raw (DarkSky) | Visual line of sight in miles. Low visibility slows traffic. |
| `precipIntensity` | Float | Raw (DarkSky) | Precipitation rate (inches of liquid water per hour). |
| `precipProbability`| Float | Raw (DarkSky) | Probability of rain/snow (0.0 to 1.0). Key surge predictor. |
| `pressure` | Float | Raw (DarkSky) | Barometric atmospheric pressure in millibars. |
| `cloudCover` | Float | Raw (DarkSky) | Sky cloud coverage percentage (0.0 to 1.0). |
| `uvIndex` | Integer | Raw (DarkSky) | Solar ultraviolet radiation index. |
| `dewPoint` | Float | Raw (DarkSky) | Atmospheric dew point temperature in °F. |
| `cab_type_enc` | Categorical | LabelEncoded | Operator identifier (`Uber` = 1, `Lyft` = 0). |
| `name_enc` | Categorical | LabelEncoded | Vehicle class (`UberX`, `Black`, `Shared`, `Lux Black XL`, etc.). |
| `source_enc` | Categorical | LabelEncoded | Origin zone among 12 Boston metropolitan hubs. |
| `destination_enc` | Categorical | LabelEncoded | Destination zone among 12 Boston metropolitan hubs. |
| `short_summary_enc`| Categorical | LabelEncoded | High-level weather state (`Clear`, `Rain`, `Overcast`, etc.). |
| `icon_enc` | Categorical | LabelEncoded | UI weather category icon (`partly-cloudy-night`, `rain`, etc.). |

---

## 📈 Machine Learning Models & Evaluation

The system evaluates four regression architectures to capture both linear baseline relationships and complex non-linear feature interactions:

1. **Linear Regression (OLS Baseline)**:
   - Establishes fundamental linear distance and surge elasticity benchmarks.
   - Interpretable baseline coefficient validation.
2. **Random Forest Regressor (`n_estimators=100`, `n_jobs=-1`)**:
   - Non-parametric ensemble of decision trees.
   - Captures non-linear thresholds between vehicle types and route pairs without multicollinearity degradation.
3. **Gradient Boosting Regressor (`n_estimators=100`, `learning_rate=0.1`, `max_depth=5`)**:
   - Sequential error correction minimizing Mean Squared Error loss.
4. **XGBoost Regressor (`n_estimators=200`, `learning_rate=0.1`, `max_depth=6`, `n_jobs=-1`)**:
   - High-performance gradient boosted decision trees with regularized objective functions (`L1` / `L2`), parallel tree construction, and cache-aware access patterns.

### Benchmark Evaluation Metrics Summary

Measured by executing [`dynamic_pricing_notebook_improved.ipynb`](dynamic_pricing_notebook_improved.ipynb) on the full dataset (637,976 priced rides, 80/20 split with `random_state=42`, 127,596 test rides). 5-fold CV for **every** model uses the same 25,000-row training sample and the same shuffled folds.

| Model | MAE (USD) | RMSE (USD) | Test R² | MAPE | 5-Fold CV R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | **`$1.084`** | **`$1.650`** | **0.9688** | **8.05%** | **0.9618 ± 0.0020** |
| Random Forest | `$1.114` | `$1.722` | 0.9660 | 8.28% | 0.9565 ± 0.0026 |
| Gradient Boosting | `$1.141` | `$1.724` | 0.9659 | 8.46% | 0.9621 ± 0.0024 |
| Linear Regression (one-hot + interactions) | `$1.271` | `$1.854` | 0.9606 | 9.31% | 0.9586 ± 0.0025 |
| *Baseline: tariff formula* `(base + rate × distance) × surge` | `$1.304` | `$1.888` | 0.9591 | 9.40% | – |
| *Baseline: service-class median* | `$3.028` | `$4.588` | 0.7585 | 19.74% | – |
| Linear Regression (label-encoded, original) | `$5.157` | `$6.490` | 0.5169 | 39.37% | 0.5149 ± 0.0082 |

*Notes:*
- *The previous version of this table (XGBoost MAE `$1.38` / R² 0.948 …) did not match what the code produces. The earlier `model_metrics.json` also stored `CV_R2 = R2 − 0.002` for Linear Regression and Gradient Boosting instead of a computed value. Both are corrected; see [CHANGES.md](CHANGES.md).*
- *Feature importance (permutation importance on held-out data): `name` ≫ `distance` > `cab_type` > `surge_multiplier` > `destination` / `source`. All weather and time features fall below ΔR² = 0.001.*

---

## 🔎 Key Findings

From the extended analysis in [`dynamic_pricing_notebook_improved.ipynb`](dynamic_pricing_notebook_improved.ipynb):

1. **Fares are close to a fixed tariff.** A per-class base fare plus a per-class per-mile rate, multiplied by surge, explains **R² = 0.959** on its own. Per-mile rates range from **`$1.20` (UberPool)** to **`$4.54` (Lux Black)**. **WAV is priced identically to UberX**, and Lyft fares follow a discrete price ladder.
2. **Uber vs Lyft (tier-matched, non-surged rides):**
   - Lyft is cheaper for Shared (`−$2.76`), Standard (`−$0.52`) and XL (`−$1.03`).
   - Uber is cheaper for Premium (Lyft +`$1.60`) and Premium XL (Lyft +`$0.78`).
   - All differences are significant (Mann-Whitney p < 1e-100).
3. **Surge is a Lyft-only, location-driven signal.**
   - 6.8% of Lyft rides surge, and Uber never does. *Shared* never surges, and every other Lyft product surges at the same 8.19% rate.
   - Surge rates range from 11.2% of Back Bay pickups to 1.5% in North End.
   - Surge barely changes by hour (5.9–8.0%) and is **not** higher in rain (6.6% vs 7.1% clear-day).
4. **Time of day and weather do not move prices.**
   - Hourly mean fares vary by only 0.74% (Kruskal-Wallis p = 0.999); there is no rush-hour premium.
   - After removing the tariff × surge effect, every weather and time feature has |ρ| < 0.01 with the residual.
   - Within UberX, weather icon explains ε² = 0.0002 of price variance.
   - 10 pairs of weather columns are near-duplicates (|r| > 0.9).
5. **Six features are enough.** Retraining XGBoost without all 21 weather and time features gives R² **0.9691**, slightly better than the 0.9688 with them. The six are `distance`, `surge_multiplier`, `name`, `cab_type`, `source` and `destination`. Removing `name` drops R² to 0.18.
6. **Where the model errs:**
   - Highest absolute error: UberXL (MAE `$1.54`).
   - Highest relative error: Shared (MAPE 22%, because fares are cheap).
   - Error grows with distance (MAE `$0.60` under 1 mi → `$1.72` at 5–8 mi) and with surge level.
7. **The linear model's low score came from the encoding, not the method.** One-hot encoding the service class (with class × distance and class × surge terms) lifts OLS from R² 0.52 to **0.96**.

> ⚠️ **Caveat for the recommender:** the rule-based surge advice (peak hours, bad weather → higher surge) is **not supported by this dataset**. A location-based surge signal would be more defensible. Also, `day` and `month` only identify dates within a 3-week scrape and will not generalise to new periods.

---

## 🎯 Pricing Recommendation Engine

In addition to point estimates, the recommendation module delivers production-grade business logic:

1. **Prediction Interval Estimation** *(improved notebook)*: an **empirical 80% interval** per service class, built from the champion model's 10th/90th-percentile test residuals. For example, UberX is `−$1.26` / +`$1.38` and Lux Black XL is `−$2.22` / +`$2.17`.
   - The original notebook and app use a fixed ±10% corridor: `Price_Low = 0.90 × Predicted_Fare`, `Price_High = 1.10 × Predicted_Fare`.

2. **Dynamic Pricing Tier Classification**:
   - 🟢 **Budget Tier** (Fare < `$10.00`): Shared / Pool and short-distance transit options.
   - 🔵 **Standard Tier** (`$10.00` ≤ Fare < `$20.00`): Standard UberX / Lyft rides under typical demand.
   - 🟠 **Premium Tier** (`$20.00` ≤ Fare < `$35.00`): XL vehicle categories or rides with moderate surge multipliers.
   - 🟣 **Luxury Tier** (Fare ≥ `$35.00`): High-end chauffeur services (Black SUV, Lux Black XL) and heavy peak surge conditions.

3. **Demand & Surge Advisor**:
   - **Morning / Evening Peak (7–9 AM, 4–7 PM)**: High passenger density; flags recommended surge of `1.25x – 1.50x`.
   - **Late Night (10 PM – 6 AM)**: Lower passenger competition; flags normal base tariff.
   - **Off-Peak Daylight**: Balanced supply-demand equilibrium; standard pricing.

---

## 🚀 Setup & Execution Guide

1. **Navigate to the Project Directory**:
   ```bash
   cd Dynamic_Pricing
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:
   Open and execute [`dynamic_pricing_notebook.ipynb`](dynamic_pricing_notebook.ipynb) in **VS Code**, **Jupyter Notebook**, or **JupyterLab**:
   ```bash
   jupyter notebook dynamic_pricing_notebook.ipynb
   ```
   - **Cell 1**: Imports core scientific and machine learning libraries.
   - **Cell 2**: Loads `rideshare_kaggle.csv` directly from the local directory.
   - **Cells 3–7**: Generates comprehensive exploratory visual charts (Target, Temporal, Route, Weather).
   - **Cells 8–9**: Performs domain feature engineering (27 features) and standard scaling.
   - **Cells 10–13**: Trains Linear Regression, Random Forest, Gradient Boosting, and XGBoost models.
   - **Cells 14–16**: Evaluates model performance, plots residual diagnostics, feature importances, and 5-fold cross-validation.
   - **Cell 17**: Demonstrates the `recommend_fare()` production pricing heuristic across 5 scenarios.
   - **Cell 18**: Serializes `best_model.pkl`, `scaler.pkl`, `label_encoders.pkl`, and `model_metrics.json`.

   For the full analysis, run [`dynamic_pricing_notebook_improved.ipynb`](dynamic_pricing_notebook_improved.ipynb) instead. It keeps every original cell and adds about 20 analysis sections, listed in [CHANGES.md](CHANGES.md). A full run took ~20 min on a 12-thread / 16 GB laptop. Have at least ~4 GB of RAM free.

4. **Standalone Training Pipeline (CLI Alternative)**:
   ```bash
   python train_models.py
   ```

5. **Interactive Dashboard UI (Optional)**:
   ```bash
   streamlit run app.py
   ```

---

## 📦 Requirements

```text
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
streamlit>=1.28.0
joblib>=1.3.0
```

---

## 📑 Technical Architecture & Verification Standards

This system is engineered in accordance with production machine learning best practices:

| Architectural Component | Engineering Realization | Module & Code Mapping |
| :--- | :--- | :--- |
| **Theoretical Foundations** | Formal mathematical formulation of Linear Regression (OLS), Ensemble Bagging (Random Forest), Gradient Boosting theory, regularized objective functions (XGBoost), and Dynamic Pricing surge elasticity. | Embedded in notebook theoretical sections, LaTeX formulations, and documentation. |
| **Pipeline Engineering** | Modular, reproducible Python pipeline: automated data cleaning, domain feature engineering (`price_per_mile`, peak hours, weather categories), categorical encoding, feature scaling, and inference recommendation module. | Notebook **Cells 1–18**, standalone training pipeline in [`train_models.py`](train_models.py), and web dashboard in [`app.py`](app.py). |
| **Empirical Benchmarking & Validation** | Evaluation across 637k records: MAE, RMSE, R², MAPE, naive and formula baselines, computed 5-fold CV for every model, permutation importance, feature-group ablation, partial dependence and per-class error diagnostics. | Notebook **Cells 14–16**; improved notebook **Cells 9b, 13b, 14b, 14c, 15b, 15c**. |
| **Production Artifacts & Serving** | Serialized model artifacts (`best_model.pkl`, `scaler.pkl`, `label_encoders.pkl`, `model_metrics.json`), rule-based pricing tier segmentation, dynamic surge advisory, and interactive multi-page web application. | [`best_model.pkl`](best_model.pkl), [`app.py`](app.py), and [`model_metrics.json`](model_metrics.json). |

---

*Author: Machine Learning & Dynamic Pricing Engineering*
