"""
GNA Analytics Showdown — Model Selection Benchmark
Compares 7 candidate models with 5-fold CV on log_total_weight.

Models tested:
  1. Ridge regression (linear baseline)
  2. Lasso regression
  3. Random Forest
  4. Histogram-based Gradient Boosting (sklearn, handles NaN natively)
  5. XGBoost
  6. LightGBM
  7. CatBoost

Preprocessing per model class:
  - Linear: StandardScaler + SimpleImputer (median) + OneHotEncoder
  - Tree (sklearn RF, HistGB): SimpleImputer (median) + OrdinalEncoder
  - Gradient boosting (XGB, LGBM, CatBoost): native NaN + categorical handling

Metrics:
  - Primary: RMSE on log_total_weight (model selection)
  - Secondary: MAE on log_total_weight
  - Interpretive: R²
  - Business: RMSE on raw total_weight_kg (expm1 of predictions)
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("master_features.csv")

NUM_FEATURES = [
    "age", "is_female", "is_organic", "number_seasons", "days_as_member", "zone_ordinal",
    "total_hectares", "n_loan_packages", "n_crop_types_loaned",
    "has_fertilizer", "has_fungicide", "has_gypsum", "has_inoculant",
    "has_insecticide", "has_lime", "has_seed_guard",
    "input_count", "input_richness_score",
    "has_source_program", "has_seed_program", "has_organic_program", "has_partnership_program",
    "total_inkind_repayment", "total_cash_repayment", "total_down_payment",
    "has_asset_loan", "has_preharvest_loan", "has_family_package",
    "qty_kgs_planted", "n_crops_planted", "avg_spacing",
    "pct_spacing_optimal", "has_training", "pct_multi_seed",
    "any_late_planting", "planting_doy", "planting_spread_days",
    "days_loan_to_plant", "seed_density_kg_per_ha",
    "fertilizer_x_zone", "lime_x_zone", "experience_x_training", "rich_inputs_x_hectares",
]
CAT_FEATURES = ["dominant_crop", "agroecological_zone", "region_name"]

TARGET      = "log_total_weight"
TARGET_RAW  = "total_weight_kg"

X = df[NUM_FEATURES + CAT_FEATURES]
y = df[TARGET]
y_raw = df[TARGET_RAW]

print(f"Dataset: {X.shape[0]} rows × {X.shape[1]} features")
print(f"Target: {TARGET}  (mean={y.mean():.3f}, std={y.std():.3f})")
print(f"Numeric features: {len(NUM_FEATURES)}")
print(f"Categorical features: {CAT_FEATURES}")


# ─────────────────────────────────────────────────────────────
# PREPROCESSING PIPELINES
# ─────────────────────────────────────────────────────────────

# For linear models: impute + scale numerics, one-hot encode categoricals
linear_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]), NUM_FEATURES),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CAT_FEATURES),
])

# For sklearn tree models: impute numerics + ordinal encode categoricals
tree_preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord",     OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), CAT_FEATURES),
])


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────
models = {
    "Ridge": Pipeline([
        ("pre",   linear_preprocessor),
        ("model", Ridge(alpha=1.0)),
    ]),

    "Lasso": Pipeline([
        ("pre",   linear_preprocessor),
        ("model", Lasso(alpha=0.01, max_iter=5000)),
    ]),

    "Random Forest": Pipeline([
        ("pre",   tree_preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            n_jobs=-1, random_state=42,
        )),
    ]),

    "HistGradientBoosting": Pipeline([
        ("pre",   tree_preprocessor),
        ("model", HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=6,
            min_samples_leaf=20, random_state=42,
        )),
    ]),

    "XGBoost": xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        enable_categorical=True,
        tree_method="hist", random_state=42, verbosity=0,
        n_jobs=-1,
    ),

    "LightGBM": lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1, n_jobs=-1,
    ),

    "CatBoost": CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        min_data_in_leaf=20, random_seed=42,
        cat_features=CAT_FEATURES,
        verbose=0,
    ),
}

# Prepare separate encoded versions for native GBM models
cat_ordinal_encoder = ColumnTransformer([
    ("num", "passthrough",  NUM_FEATURES),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_FEATURES),
])
X_encoded = cat_ordinal_encoder.fit_transform(X)
X_encoded = pd.DataFrame(X_encoded, columns=NUM_FEATURES + CAT_FEATURES)

# XGBoost: needs integer dtype for enable_categorical (not float, not "category")
X_for_xgb = X_encoded.copy()
for col in CAT_FEATURES:
    X_for_xgb[col] = X_for_xgb[col].fillna(-1).astype(int)
    X_for_xgb[col] = X_for_xgb[col].astype("category")

# LightGBM: accepts pandas "category" dtype natively (handles NaN internally)
X_for_lgbm = X_encoded.copy()
for col in CAT_FEATURES:
    X_for_lgbm[col] = X_for_lgbm[col].astype("category")

# CatBoost: accepts raw strings natively; fill NaN cats with "Unknown"
X_for_catboost = X.copy()
for col in CAT_FEATURES:
    X_for_catboost[col] = X_for_catboost[col].fillna("Unknown")


# ─────────────────────────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_score(name, model, X_data):
    """Run 5-fold CV and return mean/std of RMSE, MAE, R² on log target
    plus RMSE on raw kg (business metric)."""
    rmse_scores, mae_scores, r2_scores, raw_rmse_scores = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_data)):
        X_tr, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        y_raw_val   = y_raw.iloc[val_idx]

        model.fit(X_tr, y_tr)
        preds_log = model.predict(X_val)

        rmse_scores.append(root_mean_squared_error(y_val, preds_log))
        mae_scores.append( mean_absolute_error(y_val, preds_log))
        r2_scores.append(  r2_score(y_val, preds_log))

        # Convert log predictions back to kg
        preds_kg = np.expm1(preds_log).clip(min=0)
        raw_rmse_scores.append(root_mean_squared_error(y_raw_val, preds_kg))

    return {
        "RMSE_log":     np.mean(rmse_scores),
        "RMSE_log_std": np.std(rmse_scores),
        "MAE_log":      np.mean(mae_scores),
        "R2":           np.mean(r2_scores),
        "RMSE_kg":      np.mean(raw_rmse_scores),
    }

print("\n" + "=" * 70)
print("RUNNING 5-FOLD CROSS-VALIDATION")
print("=" * 70)

results = {}
X_DATA_MAP = {
    "Ridge":               X,
    "Lasso":               X,
    "Random Forest":       X,
    "HistGradientBoosting":X,
    "XGBoost":             X_for_xgb,
    "LightGBM":            X_for_lgbm,
    "CatBoost":            X_for_catboost,
}

for name, model in models.items():
    X_data = X_DATA_MAP[name]
    t0 = time.time()
    print(f"  {name:<25}", end="", flush=True)
    scores = cv_score(name, model, X_data)
    elapsed = time.time() - t0
    results[name] = scores
    print(f"  RMSE_log={scores['RMSE_log']:.4f} ± {scores['RMSE_log_std']:.4f}"
          f"  MAE_log={scores['MAE_log']:.4f}"
          f"  R²={scores['R2']:.4f}"
          f"  RMSE_kg={scores['RMSE_kg']:.0f}"
          f"  ({elapsed:.1f}s)")


# ─────────────────────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS SUMMARY (ranked by RMSE on log target)")
print("=" * 70)

res_df = pd.DataFrame(results).T.sort_values("RMSE_log")
res_df["RMSE_log_fmt"] = res_df.apply(
    lambda r: f"{r['RMSE_log']:.4f} ± {r['RMSE_log_std']:.4f}", axis=1
)
print(res_df[["RMSE_log_fmt", "MAE_log", "R2", "RMSE_kg"]].rename(columns={
    "RMSE_log_fmt": "RMSE (log) ± std",
    "MAE_log":      "MAE (log)",
    "R2":           "R²",
    "RMSE_kg":      "RMSE (kg)",
}).round(4).to_string())

best = res_df.index[0]
print(f"\n→ Best model: {best}")
print(f"   RMSE_log = {res_df.loc[best, 'RMSE_log']:.4f}")
print(f"   R²       = {res_df.loc[best, 'R2']:.4f}")
print(f"   RMSE_kg  = {res_df.loc[best, 'RMSE_kg']:.0f} kg")

# Relative improvement over linear baseline
ridge_rmse = res_df.loc["Ridge", "RMSE_log"]
best_rmse  = res_df.loc[best, "RMSE_log"]
print(f"\n   Improvement over Ridge (linear baseline): {(ridge_rmse - best_rmse)/ridge_rmse:.1%}")

res_df.to_csv("model_comparison_results.csv")
print("\nSaved → model_comparison_results.csv")
