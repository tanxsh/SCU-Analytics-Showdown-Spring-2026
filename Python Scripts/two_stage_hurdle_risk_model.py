"""
GNA Analytics Showdown — Two-Stage Hurdle Model

Stage 1: XGBoost binary classifier — P(farmer sells back)
         Tuned with Optuna, 75 trials × 5-fold CV
         Target: non_seller (1 = didn't sell back)

Stage 2: XGBoost regressor — E[yield | farmer sells back]
         Tuned with Optuna, 50 trials × 5-fold CV
         Target: log_total_weight, trained on sellers only

Combined: Expected yield = P(sell) × E[yield | sell]
          Compared against single-stage model baseline

Risk tiers derived from combined expected yield, validated
against actual outcomes and profiled for GNA field use.

Outputs
-------
  stage1_classifier.json
  stage1_params.json
  stage2_regressor.json
  stage2_params.json
  hurdle_evaluation.txt
  risk_tiers.csv           — farmer-level risk scores + tier labels
  risk_tier_profiles.csv   — tier summary statistics
"""

import json, os
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, classification_report,
    root_mean_squared_error, mean_absolute_error, r2_score,
)
import warnings
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
CAT_FEATURES  = ["dominant_crop", "agroecological_zone", "region_name"]
ALL_FEATURES  = NUM_FEATURES + CAT_FEATURES

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_full = df[ALL_FEATURES].copy()
X_full[CAT_FEATURES] = enc.fit_transform(X_full[CAT_FEATURES].fillna("Unknown"))
for col in CAT_FEATURES:
    X_full[col] = X_full[col].astype(int).astype("category")

# Stage 1: full dataset
y_stage1 = df["non_seller"].values          # 1 = didn't sell back

# Stage 2: sellers only
seller_mask   = df["has_buyback"].values.astype(bool)
X_stage2      = X_full[seller_mask].reset_index(drop=True)
y_stage2      = df.loc[seller_mask, "log_total_weight"].values

# For final evaluation
y_raw_full    = df["total_weight_kg"].values

SCALE_POS_WEIGHT = (seller_mask.sum()) / (~seller_mask).sum()   # ~3.3

print(f"Stage 1  — full dataset:   {len(X_full)} rows, {y_stage1.mean():.1%} non-sellers")
print(f"Stage 2  — sellers only:   {len(X_stage2)} rows, "
      f"log_yield mean={y_stage2.mean():.3f} std={y_stage2.std():.3f}")


# ─────────────────────────────────────────────────────────────
# SHARED CV SPLITS  (stratified for Stage 1 consistency)
# ─────────────────────────────────────────────────────────────
N_FOLDS   = 5
skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
kf        = KFold(            n_splits=N_FOLDS, shuffle=True, random_state=42)

S1_SPLITS = list(skf.split(X_full, y_stage1))
S2_SPLITS = list(kf.split(X_stage2))


# ─────────────────────────────────────────────────────────────
# STAGE 1 — TUNE CLASSIFIER
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 1 — CLASSIFIER TUNING (75 trials × 5-fold CV)")
print("=" * 60)

def s1_objective(trial):
    params = {
        "max_depth":         trial.suggest_int(  "max_depth",         3, 8),
        "min_child_weight":  trial.suggest_int(  "min_child_weight",  1, 30),
        "gamma":             trial.suggest_float( "gamma",            0.0, 3.0),
        "subsample":         trial.suggest_float( "subsample",        0.5, 1.0),
        "colsample_bytree":  trial.suggest_float( "colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float( "colsample_bylevel",0.4, 1.0),
        "colsample_bynode":  trial.suggest_float( "colsample_bynode", 0.4, 1.0),
        "reg_alpha":         trial.suggest_float( "reg_alpha",  1e-8, 5.0, log=True),
        "reg_lambda":        trial.suggest_float( "reg_lambda", 1e-8, 5.0, log=True),
        "learning_rate":     trial.suggest_float( "learning_rate", 0.01, 0.3, log=True),
        "scale_pos_weight":  SCALE_POS_WEIGHT,
        "n_estimators":      2000,
        "objective":         "binary:logistic",
        "eval_metric":       "auc",
        "tree_method":       "hist",
        "enable_categorical":True,
        "random_state":      42,
        "verbosity":         0,
        "n_jobs":            -1,
    }
    fold_aucs = []
    for fold_idx, (tr_idx, val_idx) in enumerate(S1_SPLITS):
        model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
        model.fit(X_full.iloc[tr_idx], y_stage1[tr_idx],
                  eval_set=[(X_full.iloc[val_idx], y_stage1[val_idx])],
                  verbose=False)
        prob  = model.predict_proba(X_full.iloc[val_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y_stage1[val_idx], prob))
        trial.report(np.mean(fold_aucs), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return np.mean(fold_aucs)

study_s1 = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42, n_startup_trials=15),
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2),
)
study_s1.optimize(s1_objective, n_trials=75, show_progress_bar=True)

best_s1 = study_s1.best_params
print(f"\nBest AUC (CV): {study_s1.best_value:.4f}")
print("Best params:")
for k, v in sorted(best_s1.items()):
    print(f"  {k:<25} {v}")

# Find optimal n_estimators via early stopping
print("\nFinding optimal n_estimators for Stage 1 ...")
s1_iters = []
for tr_idx, val_idx in S1_SPLITS:
    m = xgb.XGBClassifier(
        **best_s1, n_estimators=2000, scale_pos_weight=SCALE_POS_WEIGHT,
        objective="binary:logistic", eval_metric="auc",
        tree_method="hist", enable_categorical=True,
        early_stopping_rounds=50, random_state=42, verbosity=0, n_jobs=-1,
    )
    m.fit(X_full.iloc[tr_idx], y_stage1[tr_idx],
          eval_set=[(X_full.iloc[val_idx], y_stage1[val_idx])], verbose=False)
    s1_iters.append(m.best_iteration)

s1_n = int(np.mean(s1_iters)) + 1
print(f"  Fold best iters: {s1_iters}  →  n_estimators={s1_n}")

# Final Stage 1 params
s1_final_params = {
    **best_s1,
    "n_estimators":       s1_n,
    "scale_pos_weight":   SCALE_POS_WEIGHT,
    "objective":          "binary:logistic",
    "eval_metric":        "auc",
    "tree_method":        "hist",
    "enable_categorical": True,
    "random_state":       42,
    "verbosity":          0,
    "n_jobs":             -1,
}

# Full CV evaluation for Stage 1
print("\nFull CV evaluation — Stage 1 ...")
s1_aucs, s1_aps, s1_f1s = [], [], []
for tr_idx, val_idx in S1_SPLITS:
    m = xgb.XGBClassifier(**s1_final_params)
    m.fit(X_full.iloc[tr_idx], y_stage1[tr_idx])
    prob  = m.predict_proba(X_full.iloc[val_idx])[:, 1]
    pred  = (prob >= 0.5).astype(int)
    s1_aucs.append(roc_auc_score(y_stage1[val_idx], prob))
    s1_aps.append(average_precision_score(y_stage1[val_idx], prob))
    s1_f1s.append(f1_score(y_stage1[val_idx], pred))

print(f"  AUC-ROC:          {np.mean(s1_aucs):.4f} ± {np.std(s1_aucs):.4f}")
print(f"  Avg Precision:    {np.mean(s1_aps):.4f} ± {np.std(s1_aps):.4f}")
print(f"  F1 (thresh=0.5):  {np.mean(s1_f1s):.4f} ± {np.std(s1_f1s):.4f}")

# Train final Stage 1 model
print("\nTraining final Stage 1 model on full dataset ...")
stage1_model = xgb.XGBClassifier(**s1_final_params)
stage1_model.fit(X_full, y_stage1)
stage1_model.save_model("stage1_classifier.json")
with open("stage1_params.json", "w") as f:
    json.dump(s1_final_params, f, indent=2)
print("Saved → stage1_classifier.json + stage1_params.json")


# ─────────────────────────────────────────────────────────────
# STAGE 2 — TUNE REGRESSOR ON SELLERS ONLY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 2 — REGRESSOR TUNING (50 trials × 5-fold CV, sellers only)")
print("=" * 60)

# Warm-start search space around existing best params
with open("best_xgb_params.json") as f:
    base_params = json.load(f)

def s2_objective(trial):
    params = {
        "max_depth":         trial.suggest_int(  "max_depth",         3, 8),
        "min_child_weight":  trial.suggest_int(  "min_child_weight",  1, 30),
        "gamma":             trial.suggest_float( "gamma",            0.0, 3.0),
        "subsample":         trial.suggest_float( "subsample",        0.5, 1.0),
        "colsample_bytree":  trial.suggest_float( "colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float( "colsample_bylevel",0.4, 1.0),
        "colsample_bynode":  trial.suggest_float( "colsample_bynode", 0.4, 1.0),
        "reg_alpha":         trial.suggest_float( "reg_alpha",  1e-8, 5.0, log=True),
        "reg_lambda":        trial.suggest_float( "reg_lambda", 1e-8, 5.0, log=True),
        "learning_rate":     trial.suggest_float( "learning_rate", 0.01, 0.3, log=True),
        "n_estimators":      2000,
        "tree_method":       "hist",
        "enable_categorical":True,
        "random_state":      42,
        "verbosity":         0,
        "n_jobs":            -1,
    }
    fold_rmses = []
    for fold_idx, (tr_idx, val_idx) in enumerate(S2_SPLITS):
        model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
        model.fit(X_stage2.iloc[tr_idx], y_stage2[tr_idx],
                  eval_set=[(X_stage2.iloc[val_idx], y_stage2[val_idx])],
                  verbose=False)
        preds = model.predict(X_stage2.iloc[val_idx])
        fold_rmses.append(root_mean_squared_error(y_stage2[val_idx], preds))
        trial.report(np.mean(fold_rmses), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return np.mean(fold_rmses)

study_s2 = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=42, n_startup_trials=10),
    pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=2),
)
# Seed first trial with known-good params from full-dataset tuning
study_s2.enqueue_trial({
    k: v for k, v in base_params.items()
    if k in ["max_depth","min_child_weight","gamma","subsample",
             "colsample_bytree","colsample_bylevel","colsample_bynode",
             "reg_alpha","reg_lambda","learning_rate"]
})
study_s2.optimize(s2_objective, n_trials=50, show_progress_bar=True)

best_s2 = study_s2.best_params
print(f"\nBest RMSE_log (CV, sellers): {study_s2.best_value:.4f}")

# Optimal n_estimators
print("Finding optimal n_estimators for Stage 2 ...")
s2_iters = []
for tr_idx, val_idx in S2_SPLITS:
    m = xgb.XGBRegressor(
        **best_s2, n_estimators=2000,
        tree_method="hist", enable_categorical=True,
        early_stopping_rounds=50, random_state=42, verbosity=0, n_jobs=-1,
    )
    m.fit(X_stage2.iloc[tr_idx], y_stage2[tr_idx],
          eval_set=[(X_stage2.iloc[val_idx], y_stage2[val_idx])], verbose=False)
    s2_iters.append(m.best_iteration)

s2_n = int(np.mean(s2_iters)) + 1
print(f"  Fold best iters: {s2_iters}  →  n_estimators={s2_n}")

s2_final_params = {
    **best_s2,
    "n_estimators":       s2_n,
    "tree_method":        "hist",
    "enable_categorical": True,
    "random_state":       42,
    "verbosity":          0,
    "n_jobs":             -1,
}

# Full CV evaluation for Stage 2
print("\nFull CV evaluation — Stage 2 ...")
s2_rmses, s2_r2s = [], []
for tr_idx, val_idx in S2_SPLITS:
    m = xgb.XGBRegressor(**s2_final_params)
    m.fit(X_stage2.iloc[tr_idx], y_stage2[tr_idx])
    preds = m.predict(X_stage2.iloc[val_idx])
    s2_rmses.append(root_mean_squared_error(y_stage2[val_idx], preds))
    s2_r2s.append(r2_score(y_stage2[val_idx], preds))

print(f"  RMSE_log (sellers): {np.mean(s2_rmses):.4f} ± {np.std(s2_rmses):.4f}")
print(f"  R² (sellers):       {np.mean(s2_r2s):.4f}  ± {np.std(s2_r2s):.4f}")

# Train final Stage 2 model
print("\nTraining final Stage 2 model on all sellers ...")
stage2_model = xgb.XGBRegressor(**s2_final_params)
stage2_model.fit(X_stage2, y_stage2)
stage2_model.save_model("stage2_regressor.json")
with open("stage2_params.json", "w") as f:
    json.dump(s2_final_params, f, indent=2)
print("Saved → stage2_regressor.json + stage2_params.json")


# ─────────────────────────────────────────────────────────────
# COMBINED MODEL EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMBINED MODEL EVALUATION (vs single-stage baseline)")
print("=" * 60)

SINGLE_STAGE_RMSE_KG = 593.0   # tuned XGBoost from tune_xgboost.py
SINGLE_STAGE_R2      = 0.4771

# Produce out-of-fold predictions for the full dataset
# We need OOF from both stages on their respective samples
oof_prob_sell  = np.zeros(len(df))    # P(farmer sells)
oof_yield_pred = np.zeros(len(df))    # E[yield | seller]

# Build seller index mapping
seller_global_idx = np.where(seller_mask)[0]

for fold_idx, ((s1_tr, s1_val), (s2_tr, s2_val)) in enumerate(zip(S1_SPLITS, S2_SPLITS)):
    # Stage 1
    m1 = xgb.XGBClassifier(**s1_final_params)
    m1.fit(X_full.iloc[s1_tr], y_stage1[s1_tr])
    prob = m1.predict_proba(X_full.iloc[s1_val])[:, 1]
    oof_prob_sell[s1_val] = 1 - prob    # prob of SELLING = 1 - P(non_seller)

    # Stage 2 (apply to all farmers, not just sellers in this fold)
    m2 = xgb.XGBRegressor(**s2_final_params)
    m2.fit(X_stage2.iloc[s2_tr], y_stage2[s2_tr])
    # Predict for ALL farmers (for the combined score)
    yield_log_all = m2.predict(X_full.iloc[s1_val])
    oof_yield_pred[s1_val] = np.expm1(yield_log_all).clip(min=0)

# Combined expected yield
combined_expected_kg = oof_prob_sell * oof_yield_pred

rmse_combined = root_mean_squared_error(y_raw_full, combined_expected_kg)
r2_combined   = r2_score(y_raw_full, combined_expected_kg)
mae_combined  = mean_absolute_error(y_raw_full, combined_expected_kg)

print(f"\n{'Metric':<30} {'Single-stage':>14} {'Hurdle model':>14} {'Δ':>10}")
print("-" * 70)
print(f"{'RMSE (kg)':<30} {SINGLE_STAGE_RMSE_KG:>14.1f} {rmse_combined:>14.1f} "
      f"{SINGLE_STAGE_RMSE_KG - rmse_combined:>+10.1f}")
print(f"{'R²':<30} {SINGLE_STAGE_R2:>14.4f} {r2_combined:>14.4f} "
      f"{r2_combined - SINGLE_STAGE_R2:>+10.4f}")
print(f"{'MAE (kg)':<30} {'—':>14} {mae_combined:>14.1f}")

print(f"\nStage 1 AUC-ROC:   {np.mean(s1_aucs):.4f}  (identifies non-sellers)")
print(f"Stage 1 Avg Prec:  {np.mean(s1_aps):.4f}  (precision-recall area)")
print(f"Stage 2 RMSE_log:  {np.mean(s2_rmses):.4f}  (yield prediction on sellers)")
print(f"Stage 2 R²:        {np.mean(s2_r2s):.4f}  (variance explained, sellers only)")


# ─────────────────────────────────────────────────────────────
# RISK TIERS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RISK TIERS")
print("=" * 60)

# Use P(non-sell) directly from Stage 1 as primary risk signal
# — more interpretable than combined score for the risk question
prob_nonseller_full = 1 - oof_prob_sell   # P(non_seller)

# Tier thresholds — driven by percentiles of expected yield
t33 = np.percentile(combined_expected_kg, 33)
t67 = np.percentile(combined_expected_kg, 67)

tiers = np.where(combined_expected_kg <= t33, "HIGH",
        np.where(combined_expected_kg <= t67, "MEDIUM", "LOW"))

df_out = df[["farmer_id", "non_seller", "total_weight_kg", "yield_per_ha",
             "dominant_crop", "region_name", "agroecological_zone",
             "total_hectares", "number_seasons", "has_fertilizer", "has_lime",
             "any_late_planting", "has_training"]].copy()
df_out["p_non_seller"]       = prob_nonseller_full
df_out["p_seller"]           = oof_prob_sell
df_out["predicted_yield_kg"] = combined_expected_kg
df_out["risk_tier"]          = tiers

# Profile
print(f"\nTier thresholds:  HIGH ≤ {t33:.0f} kg  |  MEDIUM ≤ {t67:.0f} kg  |  LOW > {t67:.0f} kg\n")
print(f"{'Metric':<35} {'HIGH':>10} {'MEDIUM':>10} {'LOW':>10}")
print("-" * 67)

for metric, col, fmt in [
    ("Farmer count",          None,                  "d"),
    ("Actual non-seller rate",None,                  ".1%"),
    ("Actual median yield (kg)", "total_weight_kg",  ".0f"),
    ("Actual mean yield (kg)",   "total_weight_kg",  ".0f"),
    ("P(non-seller) mean",    None,                  ".3f"),
    ("Median farm size (ha)", "total_hectares",      ".2f"),
    ("Seasons with GNA (med)","number_seasons",      ".1f"),
    ("Late planting rate",    "any_late_planting",   ".1%"),
    ("Has fertilizer rate",   "has_fertilizer",      ".1%"),
    ("Training rate",         "has_training",        ".1%"),
]:
    vals = []
    for tier in ["HIGH", "MEDIUM", "LOW"]:
        g = df_out[df_out["risk_tier"] == tier]
        if col is None and metric == "Farmer count":
            vals.append(len(g))
        elif col is None and "non-seller" in metric:
            vals.append(g["non_seller"].mean())
        elif col is None and "P(non-seller)" in metric:
            vals.append(g["p_non_seller"].mean())
        else:
            vals.append(g[col].median() if "median" in metric.lower() else g[col].mean())

    row = f"  {metric:<33}"
    for v in vals:
        if fmt == "d":
            row += f" {int(v):>10,}"
        elif fmt == ".1%":
            row += f" {v:>10.1%}"
        else:
            row += f" {v:>10{fmt}}"
    print(row)

# Region breakdown per tier
print("\nTop regions per risk tier:")
for tier in ["HIGH", "MEDIUM", "LOW"]:
    top = df_out[df_out["risk_tier"]==tier]["region_name"].value_counts().head(3)
    print(f"  {tier}: {', '.join(top.index.tolist())}")

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
df_out.to_csv("risk_tiers.csv", index=False)

tier_profiles = df_out.groupby("risk_tier").agg(
    n_farmers             = ("farmer_id",          "count"),
    actual_nonseller_rate = ("non_seller",          "mean"),
    actual_median_yield   = ("total_weight_kg",     "median"),
    actual_mean_yield     = ("total_weight_kg",     "mean"),
    p_nonseller_mean      = ("p_non_seller",        "mean"),
    predicted_yield_mean  = ("predicted_yield_kg",  "mean"),
    median_hectares       = ("total_hectares",      "median"),
    median_seasons        = ("number_seasons",      "median"),
    late_planting_rate    = ("any_late_planting",   "mean"),
    fertilizer_rate       = ("has_fertilizer",      "mean"),
    training_rate         = ("has_training",        "mean"),
).round(3)
tier_profiles.to_csv("risk_tier_profiles.csv")
print("\nSaved → risk_tiers.csv")
print("Saved → risk_tier_profiles.csv")

# Evaluation summary
summary = f"""
HURDLE MODEL EVALUATION SUMMARY
================================
Stage 1 (Classifier — non-seller detection)
  AUC-ROC:          {np.mean(s1_aucs):.4f} ± {np.std(s1_aucs):.4f}
  Avg Precision:    {np.mean(s1_aps):.4f} ± {np.std(s1_aps):.4f}
  F1 (thresh=0.5):  {np.mean(s1_f1s):.4f} ± {np.std(s1_f1s):.4f}
  n_estimators:     {s1_n}

Stage 2 (Regressor — yield on sellers only)
  RMSE_log:         {np.mean(s2_rmses):.4f} ± {np.std(s2_rmses):.4f}
  R²:               {np.mean(s2_r2s):.4f}  ± {np.std(s2_r2s):.4f}
  n_estimators:     {s2_n}

Combined vs Single-Stage
  RMSE (kg) single: {SINGLE_STAGE_RMSE_KG:.1f}
  RMSE (kg) hurdle: {rmse_combined:.1f}
  R² single:        {SINGLE_STAGE_R2:.4f}
  R² hurdle:        {r2_combined:.4f}

Risk Tiers
  HIGH   ≤ {t33:.0f} kg expected
  MEDIUM ≤ {t67:.0f} kg expected
  LOW    > {t67:.0f} kg expected
"""
with open("hurdle_evaluation.txt", "w") as f:
    f.write(summary)
print(summary)
print("✅  Done.")
