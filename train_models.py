"""
Dynamic Pricing Model Training Script
Executes the data preprocessing, feature engineering, model training,
evaluation, cross-validation, and artifact serialization.
"""

import os
import sys
import json
import time

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def main():
    start_total = time.time()
    print("=" * 60)
    print("🚀 STARTING DYNAMIC PRICING MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load Data
    data_path = 'rideshare_kaggle.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"\n[1/7] Loading dataset from {data_path}...")
    t0 = time.time()
    df = pd.read_csv(data_path)
    print(f"      Loaded {df.shape[0]:,} rows and {df.shape[1]} columns in {time.time()-t0:.2f}s")

    # 2. Preprocessing & Null Removal
    print("\n[2/7] Preprocessing and handling missing values...")
    initial_len = len(df)
    df = df.dropna(subset=['price']).copy()
    print(f"      Filtered unpriced taxi records: dropped {initial_len - len(df):,} rows.")
    print(f"      Active records with price target: {len(df):,}")

    # 3. Feature Engineering
    print("\n[3/7] Performing feature engineering...")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if x in [7, 8, 9, 16, 17, 18, 19] else 0)
    df['is_night'] = df['hour'].apply(lambda x: 1 if x < 6 or x >= 22 else 0)
    df['price_per_mile'] = df['price'] / (df['distance'] + 1e-5)

    df['icon'] = df['icon'].astype(str).str.strip()
    df['short_summary'] = df['short_summary'].astype(str).str.strip()

    df['temp_category'] = pd.cut(
        df['temperature'],
        bins=[-np.inf, 32, 50, 65, 80, np.inf],
        labels=['freezing', 'cold', 'mild', 'warm', 'hot']
    )

    # Encode categoricals
    le_dict = {}
    cat_cols = ['cab_type', 'name', 'source', 'destination', 'short_summary', 'icon', 'temp_category']
    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    print("      Categorical encoding completed for:", cat_cols)

    # 4. Feature Selection & Train/Test Split
    print("\n[4/7] Preparing feature matrix and scaling...")
    FEATURES = [
        'distance', 'surge_multiplier', 'hour', 'day', 'month',
        'day_of_week', 'is_weekend', 'is_peak_hour', 'is_night',
        'temperature', 'apparentTemperature', 'humidity', 'windSpeed',
        'windGust', 'visibility', 'precipIntensity', 'precipProbability',
        'pressure', 'cloudCover', 'uvIndex', 'dewPoint',
        'cab_type_enc', 'name_enc', 'source_enc', 'destination_enc',
        'short_summary_enc', 'icon_enc'
    ]
    TARGET = 'price'

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"      Features count: {len(FEATURES)}")
    print(f"      Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 5. Model Training & Evaluation
    print("\n[5/7] Training machine learning models...")
    results = {}

    # Model 1: Linear Regression
    print("      -> Training Model 1: Linear Regression (Baseline)...")
    t_start = time.time()
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)
    results['Linear Regression'] = {'model': lr, 'MAE': mae_lr, 'RMSE': rmse_lr, 'R2': r2_lr}
    print(f"         [Linear Regression] in {time.time()-t_start:.2f}s | MAE: ${mae_lr:.3f} | RMSE: ${rmse_lr:.3f} | R²: {r2_lr:.4f}")

    # Model 2: Random Forest Regressor
    print("      -> Training Model 2: Random Forest Regressor...")
    t_start = time.time()
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        max_samples=0.25 if len(X_train) > 100000 else None
    )
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    results['Random Forest'] = {'model': rf, 'MAE': mae_rf, 'RMSE': rmse_rf, 'R2': r2_rf}
    print(f"         [Random Forest] in {time.time()-t_start:.2f}s | MAE: ${mae_rf:.3f} | RMSE: ${rmse_rf:.3f} | R²: {r2_rf:.4f}")

    # Model 3: Gradient Boosting Regressor
    print("      -> Training Model 3: Gradient Boosting Regressor...")
    t_start = time.time()
    gb_sample_size = min(50000, len(X_train_scaled))
    gb_idx = np.random.RandomState(42).choice(len(X_train_scaled), gb_sample_size, replace=False)
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb.fit(X_train_scaled[gb_idx], y_train.iloc[gb_idx])
    y_pred_gb = gb.predict(X_test_scaled)
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    r2_gb = r2_score(y_test, y_pred_gb)
    results['Gradient Boosting'] = {'model': gb, 'MAE': mae_gb, 'RMSE': rmse_gb, 'R2': r2_gb}
    print(f"         [Gradient Boosting] in {time.time()-t_start:.2f}s | MAE: ${mae_gb:.3f} | RMSE: ${rmse_gb:.3f} | R²: {r2_gb:.4f}")

    # Model 4: XGBoost Regressor
    print("      -> Training Model 4: XGBoost Regressor (Champion)...")
    t_start = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    xgb_model.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)
    results['XGBoost'] = {'model': xgb_model, 'MAE': mae_xgb, 'RMSE': rmse_xgb, 'R2': r2_xgb}
    print(f"         [XGBoost] in {time.time()-t_start:.2f}s | MAE: ${mae_xgb:.3f} | RMSE: ${rmse_xgb:.3f} | R²: {r2_xgb:.4f}")

    # Select Champion Model
    best_model_name = max(results.keys(), key=lambda m: results[m]['R2'])
    best_model = results[best_model_name]['model']
    print(f"\n🏆 Champion Model Selected: {best_model_name} (R² = {results[best_model_name]['R2']:.4f})")

    # 6. Cross Validation
    print("\n[6/7] Running 5-fold cross-validation...")
    cv_sample_n = min(20000, len(X_train_scaled))
    cv_idx = np.random.RandomState(42).choice(len(X_train_scaled), cv_sample_n, replace=False)
    X_cv = X_train_scaled[cv_idx]
    y_cv = y_train.iloc[cv_idx]

    # Cross-validate EVERY model on the same sample and folds (previously only XGB/RF were computed
    # and the other CV scores were written as R2 - 0.002).
    cv_results = {}
    for name, v in results.items():
        cv_results[name] = cross_val_score(v['model'], X_cv, y_cv, cv=5, scoring='r2', n_jobs=-1)
        print(f"      {name:18s} 5-Fold Mean R²: {cv_results[name].mean():.4f} (± {cv_results[name].std():.4f})")

    # 7. Serialize Artifacts
    print("\n[7/7] Serializing production artifacts...")
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le_dict, 'label_encoders.pkl')

    metrics_export = {
        "models": {
            k: {
                "MAE": float(v['MAE']),
                "RMSE": float(v['RMSE']),
                "R2": float(v['R2']),
                "CV_R2": float(cv_results[k].mean()),
                "CV_R2_std": float(cv_results[k].std())
            } for k, v in results.items()
        },
        "best_model_name": best_model_name,
        "sample_size": len(df),
        "features_count": len(FEATURES),
        "test_size": 0.2,
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics_export, f, indent=2)

    print("      Saved: best_model.pkl")
    print("      Saved: scaler.pkl")
    print("      Saved: label_encoders.pkl")
    print("      Saved: model_metrics.json")

    print("\n" + "=" * 60)
    print(f"✅ PIPELINE SUCCESSFULLY COMPLETED IN {time.time()-start_total:.2f}s")
    print("=" * 60)

if __name__ == '__main__':
    main()
