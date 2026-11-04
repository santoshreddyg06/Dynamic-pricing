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
- **Target Variable**: `price` (Continuous, USD). Minimum: \$3.00, Maximum: \$67.50, Mean: ~\$16.22.
- **Missing Values**: 55,095 records in `price` correspond exclusively to `cab_type == 'Uber'` under `name == 'Taxi'`, where Uber functions solely as an e-hail dispatcher with municipal meter pricing. These rows are appropriately dropped during supervised learning preprocessing.
- **Geographic Coverage**: 12 key Boston hubs (e.g., Boston University, Fenway, Financial District, South Station, North End, Back Bay).
- **Service Providers**: Uber (7 products) and Lyft (6 products).
- **Surge Factors**: Lyft `surge_multiplier` ranging from `1.0x` to `3.0x`.

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
├── dynamic_pricing_notebook.ipynb  # Primary Jupyter notebook (Cells 1-18)
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

| Model | MAE (\$) | RMSE (\$) | $R^2$ Score | 5-Fold CV $R^2$ | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | ~\$4.78 | ~\$6.28 | ~0.528 | 0.527 ± 0.003 | < 1 ms |
| **Random Forest** | ~\$1.42 | ~\$2.18 | ~0.943 | 0.941 ± 0.002 | ~18 ms |
| **Gradient Boosting** | ~\$1.48 | ~\$2.25 | ~0.939 | 0.938 ± 0.002 | ~8 ms |
| **XGBoost (Champion)**| **~\$1.38** | **~\$2.09** | **~0.948** | **0.946 ± 0.002** | **~4 ms** |

*Note: The primary predictive drivers identified by tree feature importances are vehicle class (`name_enc`), trip `distance`, `surge_multiplier`, and route combination (`source_enc`, `destination_enc`).*

---

## 🎯 Pricing Recommendation Engine

In addition to point estimates, the recommendation module delivers production-grade business logic:

1. **Confidence Interval Estimation**: Computes a dynamic bounded price corridor:
   $$\text{Price}_{\text{Low}} = 0.90 \times \hat{y}, \quad \text{Price}_{\text{High}} = 1.10 \times \hat{y}$$
2. **Dynamic Pricing Tier Classification**:
   - 🟢 **Budget Tier** ($\hat{y} < \$10.00$): Shared/Pool and short-distance transit options.
   - 🔵 **Standard Tier** ($\$10.00 \le \hat{y} < \$20.00$): Standard UberX / Lyft rides under typical demand.
   - 🟠 **Premium Tier** ($\$20.00 \le \hat{y} < \$35.00$): XL vehicle categories or rides with moderate surge multipliers.
   - 🟣 **Luxury Tier** ($\hat{y} \ge \$35.00$): High-end chauffeur services (Black SUV, Lux Black XL) and heavy peak surge conditions.
3. **Demand & Surge Advisor**:
   - **Morning / Evening Peak (7–9 AM, 4–7 PM)**: High passenger density; flags recommended surge of `1.25x - 1.50x`.
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
| **Empirical Benchmarking & Validation** | Rigorous evaluation across 637k records: MAE, RMSE, $R^2$, 5-Fold Cross-Validation with variance analysis, top-15 feature importance rankings, and actual-vs-predicted residual diagnostics. | Notebook **Cells 14, 15, 16**; Model comparison tables, benchmark bar charts, residual scatter plots, and CV results. |
| **Production Artifacts & Serving** | Serialized model artifacts (`best_model.pkl`, `scaler.pkl`, `label_encoders.pkl`, `model_metrics.json`), rule-based pricing tier segmentation, dynamic surge advisory, and interactive multi-page web application. | [`best_model.pkl`](best_model.pkl), [`app.py`](app.py), and [`model_metrics.json`](model_metrics.json). |

---

*Author: Machine Learning & Dynamic Pricing Engineering*
