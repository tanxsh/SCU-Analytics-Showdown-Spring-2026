"""
GNA Analytics Showdown — XGBoost Hyperparameter Tuning
Uses Optuna (Bayesian / TPE search) with 5-fold CV + early stopping.

Strategy
--------
• Early stopping resolves n_estimators automatically — no need to search it.
• Each Optuna trial trains 5 CV folds; objective = mean RMSE on log_total_weight.
• Pruning (MedianPruner) cuts unpromising trials early to save time.
• After the search, the best params are validated with a clean 5-fold CV
  and compared against the untuned baseline from model_comparison.py.

Outputs
-------
  best_xgb_params.json        — best hyperparameters
  tuning_results.csv          — all trial results
  best_xgb_model.json         — final model (trained on full dataset)
"""

import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
ALL_FEATURES  = NUM_FEATURES + CAT_FEATURES

TARGET     = "log_total_weight"
TARGET_RAW = "total_weight_kg"

# Encode categoricals as integers → cast to pandas "category" for XGBoost
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_raw = df[ALL_FEATURES].copy()
X_raw[CAT_FEATURES] = enc.fit_transform(X_raw[CAT_FEATURES].fillna("Unknown"))
for col in CAT_FEATURES:
    X_raw[col] = X_raw[col].astype(int).astype("category")

y     = df[TARGET].values
y_raw = df[TARGET_RAW].values

print(f"Dataset: {X_raw.shape[0]} rows × {X_raw.shape[1]} features")
print(f"Target : {TARGET}  (mean={y.mean():.3f}, std={y.std():.3f})")

UNTUNED_RMSE_LOG = 1.8845   # baseline from model_comparison.py
UNTUNED_R2       = 0.4686


# ─────────────────────────────────────────────────────────────
# OPTUNA OBJECTIVE
# ─────────────────────────────────────────────────────────────
N_FOLDS  = 5
N_TRIALS = 75
EARLY_STOP_ROUNDS = 50
MAX_ESTIMATORS    = 2000

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

def objective(trial):
    params = {
        # Tree structure
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "gamma":            trial.suggest_float("gamma", 0.0, 3.0),

        # Sampling
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel":  trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "colsample_bynode":   trial.suggest_float("colsample_bynode", 0.4, 1.0),

        # Regularisation
        "reg_alpha":  trial.suggest_float("reg_alpha",  1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),

        # Learning rate — n_estimators resolved by early stopping
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),

        # Fixed
        "n_estimators":    MAX_ESTIMATORS,
        "tree_method":     "hist",
        "enable_categorical": True,
        "random_state":    42,
        "verbosity":       0,
        "n_jobs":          -1,
    }

    fold_rmses = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_raw)):
        X_tr, X_val = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(**params, early_stopping_rounds=EARLY_STOP_ROUNDS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        rmse  = root_mean_squared_error(y_val, preds)
        fold_rmses.append(rmse)

        # Pruning: report intermediate value after each fold
        trial.report(np.mean(fold_rmses), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_rmses)


# ─────────────────────────────────────────────────────────────
# RUN SEARCH
# ─────────────────────────────────────────────────────────────
print(f"\nRunning Optuna TPE search: {N_TRIALS} trials × {N_FOLDS}-fold CV ...")
print(f"(Baseline untuned RMSE_log = {UNTUNED_RMSE_LOG})\n")

study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=42, n_startup_trials=15),
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params
best_rmse   = study.best_value
n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
n_pruned    = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

print(f"\nCompleted: {n_completed} trials  |  Pruned: {n_pruned} trials")
print(f"Best RMSE_log (CV):  {best_rmse:.4f}  (was {UNTUNED_RMSE_LOG})")
print(f"Improvement:         {(UNTUNED_RMSE_LOG - best_rmse) / UNTUNED_RMSE_LOG:.2%}")
print(f"\nBest hyperparameters:")
for k, v in sorted(best_params.items()):
    print(f"  {k:<25} {v}")


# ─────────────────────────────────────────────────────────────
# DETERMINE OPTIMAL n_estimators VIA FULL EARLY-STOPPING CV
# ─────────────────────────────────────────────────────────────
print("\nFinding optimal n_estimators with best params via full CV early stopping ...")

best_iters = []
for train_idx, val_idx in kf.split(X_raw):
    X_tr, X_val = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    m = xgb.XGBRegressor(
        **{k: v for k, v in best_params.items()},
        n_estimators=MAX_ESTIMATORS,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=EARLY_STOP_ROUNDS,
        random_state=42, verbosity=0, n_jobs=-1,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    best_iters.append(m.best_iteration)

optimal_n = int(np.mean(best_iters)) + 1   # +1: 0-indexed → count
print(f"  Per-fold best iterations: {best_iters}")
print(f"  Optimal n_estimators:     {optimal_n}")


# ─────────────────────────────────────────────────────────────
# FINAL 5-FOLD VALIDATION WITH FIXED n_estimators
# ─────────────────────────────────────────────────────────────
print("\nFinal validation with fixed n_estimators ...")

final_params = {
    **best_params,
    "n_estimators":       optimal_n,
    "tree_method":        "hist",
    "enable_categorical": True,
    "random_state":       42,
    "verbosity":          0,
    "n_jobs":             -1,
}

rmse_log_folds, mae_log_folds, r2_folds, rmse_kg_folds = [], [], [], []
for train_idx, val_idx in kf.split(X_raw):
    X_tr, X_val = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    y_raw_val   = y_raw[val_idx]

    m = xgb.XGBRegressor(**final_params)
    m.fit(X_tr, y_tr)
    preds_log = m.predict(X_val)
    preds_kg  = np.expm1(preds_log).clip(min=0)

    rmse_log_folds.append(root_mean_squared_error(y_val, preds_log))
    mae_log_folds.append(mean_absolute_error(y_val, preds_log))
    r2_folds.append(r2_score(y_val, preds_log))
    rmse_kg_folds.append(root_mean_squared_error(y_raw_val, preds_kg))

tuned_rmse = np.mean(rmse_log_folds)
tuned_r2   = np.mean(r2_folds)

print("\n" + "=" * 60)
print("TUNING RESULTS")
print("=" * 60)
print(f"{'Metric':<22} {'Untuned':>10} {'Tuned':>10} {'Δ':>8}")
print("-" * 52)
print(f"{'RMSE_log (mean)':<22} {UNTUNED_RMSE_LOG:>10.4f} {tuned_rmse:>10.4f} {(UNTUNED_RMSE_LOG-tuned_rmse):>+8.4f}")
print(f"{'RMSE_log std':<22} {'0.0182':>10} {np.std(rmse_log_folds):>10.4f}")
print(f"{'MAE_log':<22} {'1.3469':>10} {np.mean(mae_log_folds):>10.4f}")
print(f"{'R²':<22} {UNTUNED_R2:>10.4f} {tuned_r2:>10.4f} {(tuned_r2-UNTUNED_R2):>+8.4f}")
print(f"{'RMSE_kg':<22} {'592':>10} {np.mean(rmse_kg_folds):>10.0f}")
print(f"{'n_estimators':<22} {'500':>10} {optimal_n:>10}")
print(f"{'Total improvement':<22} {'':>10} {'':>10} {(UNTUNED_RMSE_LOG-tuned_rmse)/UNTUNED_RMSE_LOG:>+7.2%}")


# ─────────────────────────────────────────────────────────────
# TRAIN FINAL MODEL ON FULL DATASET
# ─────────────────────────────────────────────────────────────
print("\nTraining final model on full dataset ...")
final_model = xgb.XGBRegressor(**final_params)
final_model.fit(X_raw, y)
final_model.save_model("best_xgb_model.json")
print("Saved → best_xgb_model.json")

# Save best params (include n_estimators)
with open("best_xgb_params.json", "w") as f:
    json.dump(final_params, f, indent=2)
print("Saved → best_xgb_params.json")

# Save all trial results
trials_df = study.trials_dataframe()
trials_df.to_csv("tuning_results.csv", index=False)
print("Saved → tuning_results.csv")

print("\n✅  Done.")
