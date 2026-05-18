"""
GNA Analytics Showdown — SHAP Analysis
Answers all three competition questions using the tuned XGBoost model.

Outputs (all saved to shap_plots/):
  01_feature_importance_bar.png    — top 20 features by mean |SHAP|
  02_beeswarm.png                  — direction + magnitude for all farmers
  03_input_effectiveness.png       — input package features only
  04_dependence_*.png              — how key features relate to yield
  05_interaction_fertilizer_zone.png
  06_risk_profile.png              — SHAP signature of non-sellers vs sellers
  07_waterfall_high_low.png        — individual farmer explanations

  shap_feature_importance.csv      — numeric SHAP importance table
  shap_values_full.csv             — SHAP matrix (farmer × feature)
"""

import json, os
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.filterwarnings("ignore")

os.makedirs("shap_plots", exist_ok=True)

PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        11,
}
plt.rcParams.update(PLOT_STYLE)


# ─────────────────────────────────────────────────────────────
# DATA + MODEL
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

# Readable display names
DISPLAY_NAMES = {
    "total_hectares":          "Farm size (ha)",
    "total_inkind_repayment":  "Expected in-kind repayment (kg)",
    "total_cash_repayment":    "Expected cash repayment",
    "total_down_payment":      "Down payment value",
    "input_richness_score":    "Input richness score",
    "input_count":             "Input count",
    "rich_inputs_x_hectares":  "Input richness × farm size",
    "has_fertilizer":          "Has fertilizer",
    "has_lime":                "Has lime",
    "has_fungicide":           "Has fungicide",
    "has_inoculant":           "Has inoculant",
    "has_insecticide":         "Has insecticide",
    "has_gypsum":              "Has gypsum",
    "has_seed_guard":          "Has seed guard",
    "fertilizer_x_zone":       "Fertilizer × agro-zone",
    "lime_x_zone":             "Lime × agro-zone",
    "has_source_program":      "SOURCE program",
    "has_seed_program":        "SEED program",
    "has_organic_program":     "ORGANIC program",
    "has_partnership_program": "Partnership program",
    "number_seasons":          "Seasons with GNA",
    "age":                     "Farmer age",
    "is_female":               "Female farmer",
    "days_as_member":          "Days as GNA member",
    "zone_ordinal":            "Agro-zone (I→III)",
    "dominant_crop":           "Dominant crop",
    "region_name":             "Region",
    "agroecological_zone":     "Agro-zone (raw)",
    "planting_doy":            "Planting day of season",
    "any_late_planting":       "Any late planting",
    "planting_spread_days":    "Planting spread (days)",
    "days_loan_to_plant":      "Days from loan to planting",
    "qty_kgs_planted":         "Seed quantity planted (kg)",
    "seed_density_kg_per_ha":  "Seed density (kg/ha)",
    "avg_spacing":             "Row spacing (cm)",
    "pct_spacing_optimal":     "% plots with optimal spacing",
    "has_training":            "Received crop training",
    "pct_multi_seed":          "% holes with multiple seeds",
    "n_crops_planted":         "Crops planted",
    "n_crop_types_loaned":     "Crop types in loan",
    "n_loan_packages":         "Loan packages",
    "experience_x_training":   "Experience × training",
    "is_organic":              "Organic farmer",
    "has_asset_loan":          "Has asset loan",
    "has_preharvest_loan":     "Has pre-harvest loan",
    "has_family_package":      "Has family package",
}

# Encode categoricals
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X = df[ALL_FEATURES].copy()
X[CAT_FEATURES] = enc.fit_transform(X[CAT_FEATURES].fillna("Unknown"))
for col in CAT_FEATURES:
    X[col] = X[col].astype(int).astype("category")

y = df["log_total_weight"].values

# Load best params and retrain on full dataset
with open("best_xgb_params.json") as f:
    params = json.load(f)

print("Training model on full dataset ...")
model = xgb.XGBRegressor(**params)
model.fit(X, y)
print("Done.\n")


# ─────────────────────────────────────────────────────────────
# SHAP VALUES
# ─────────────────────────────────────────────────────────────
print("Computing SHAP values ...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer(X)          # shap.Explanation object
sv           = shap_values.values   # (n_farmers, n_features)
base_value   = shap_values.base_values[0]
X_display    = X.rename(columns=DISPLAY_NAMES)
feature_names_display = [DISPLAY_NAMES.get(f, f) for f in ALL_FEATURES]

print(f"SHAP matrix: {sv.shape}  base value: {base_value:.3f}\n")


# ─────────────────────────────────────────────────────────────
# IMPORTANCE TABLE
# ─────────────────────────────────────────────────────────────
importance = pd.DataFrame({
    "feature":       ALL_FEATURES,
    "display_name":  feature_names_display,
    "mean_abs_shap": np.abs(sv).mean(axis=0),
    "mean_shap":     sv.mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
importance["rank"] = importance.index + 1

importance.to_csv("shap_feature_importance.csv", index=False)
print("Top 20 features by mean |SHAP|:")
print(importance[["rank","display_name","mean_abs_shap","mean_shap"]].head(20).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# PLOT 1 — FEATURE IMPORTANCE BAR
# ─────────────────────────────────────────────────────────────
top_n = 20
top_idx = importance.head(top_n)

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(
    top_idx["display_name"][::-1],
    top_idx["mean_abs_shap"][::-1],
    color="#E87722", edgecolor="none",
)
ax.set_xlabel("Mean |SHAP value|  (impact on log yield)", fontsize=11)
ax.set_title("Feature Importance — GNA Yield Model\n(top 20 features)", fontsize=13, pad=12)
ax.axvline(0, color="black", lw=0.5)
for bar, val in zip(bars, top_idx["mean_abs_shap"][::-1]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8.5)
plt.tight_layout()
plt.savefig("shap_plots/01_feature_importance_bar.png", dpi=150)
plt.close()
print("\nSaved → shap_plots/01_feature_importance_bar.png")


# ─────────────────────────────────────────────────────────────
# PLOT 2 — BEESWARM (top 20 features)
# ─────────────────────────────────────────────────────────────
top_features_idx = [ALL_FEATURES.index(f) for f in importance["feature"].head(top_n)]
sv_top    = sv[:, top_features_idx]
Xv_top    = X.iloc[:, top_features_idx].copy()
names_top = [DISPLAY_NAMES.get(f, f) for f in importance["feature"].head(top_n)]

shap_exp_top = shap.Explanation(
    values       = sv_top,
    base_values  = shap_values.base_values,
    data         = Xv_top.values,
    feature_names= names_top,
)

fig, ax = plt.subplots(figsize=(10, 8))
shap.plots.beeswarm(shap_exp_top, max_display=top_n, show=False, plot_size=None)
plt.title("SHAP Beeswarm — Direction & Magnitude\n(each dot = one farmer)", fontsize=12, pad=10)
plt.tight_layout()
plt.savefig("shap_plots/02_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → shap_plots/02_beeswarm.png")


# ─────────────────────────────────────────────────────────────
# PLOT 3 — INPUT EFFECTIVENESS (loan inputs only)
# ─────────────────────────────────────────────────────────────
INPUT_FEATURES = [
    "has_fertilizer", "has_lime", "has_inoculant", "has_fungicide",
    "has_insecticide", "has_gypsum", "has_seed_guard",
    "input_richness_score", "fertilizer_x_zone", "lime_x_zone",
]
inp_idx  = [ALL_FEATURES.index(f) for f in INPUT_FEATURES]
sv_inp   = sv[:, inp_idx]
inp_names= [DISPLAY_NAMES.get(f, f) for f in INPUT_FEATURES]

# Mean SHAP for has=1 vs has=0 (direction of effect)
rows = []
for f, idx in zip(INPUT_FEATURES, inp_idx):
    col_vals = X[f].cat.codes if hasattr(X[f], "cat") else X[f]
    mask1 = (X[f].astype(float) == 1.0)
    mask0 = (X[f].astype(float) == 0.0)
    if mask1.sum() < 5 or mask0.sum() < 5:
        continue
    shap_with    = sv[:, idx][mask1].mean()
    shap_without = sv[:, idx][mask0].mean()
    rows.append({
        "feature": DISPLAY_NAMES.get(f, f),
        "mean_shap_with":    shap_with,
        "mean_shap_without": shap_without,
        "net_effect":        shap_with - shap_without,
        "mean_abs_shap":     np.abs(sv[:, idx]).mean(),
    })

inp_df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#2166ac" if v >= 0 else "#d6604d" for v in inp_df["net_effect"]]
ax.barh(inp_df["feature"], inp_df["net_effect"], color=colors, edgecolor="none")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Net SHAP effect of having input (log yield units)", fontsize=11)
ax.set_title("Input Package Effectiveness\n(positive = input raises predicted yield)", fontsize=12, pad=10)
for i, (_, row) in enumerate(inp_df.iterrows()):
    x = row["net_effect"]
    ax.text(x + (0.005 if x >= 0 else -0.005), i,
            f"{x:+.3f}", va="center", ha="left" if x >= 0 else "right", fontsize=9)
plt.tight_layout()
plt.savefig("shap_plots/03_input_effectiveness.png", dpi=150)
plt.close()
print("Saved → shap_plots/03_input_effectiveness.png")


# ─────────────────────────────────────────────────────────────
# PLOT 4 — DEPENDENCE PLOTS (key continuous features)
# ─────────────────────────────────────────────────────────────
DEPENDENCE_FEATURES = [
    ("total_hectares",      "Farm size (ha)"),
    ("planting_doy",        "Planting day of season (days since Nov 1)"),
    ("input_richness_score","Input richness score"),
    ("number_seasons",      "Seasons with GNA"),
    ("avg_spacing",         "Row spacing (cm)"),
    ("seed_density_kg_per_ha", "Seed density (kg/ha)"),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for ax, (feat, label) in zip(axes, DEPENDENCE_FEATURES):
    fidx  = ALL_FEATURES.index(feat)
    xvals = X[feat].astype(float).values
    svals = sv[:, fidx]

    # Remove NaN
    mask  = ~np.isnan(xvals)
    xv, sv_ = xvals[mask], svals[mask]

    # Scatter with alpha
    ax.scatter(xv, sv_, alpha=0.07, s=6, color="#2166ac", rasterized=True)

    # Lowess smoothing
    from scipy.stats import binned_statistic
    try:
        bins   = min(40, len(np.unique(xv)))
        bstat, bedges, _ = binned_statistic(xv, sv_, statistic="mean", bins=bins)
        bcenters = (bedges[:-1] + bedges[1:]) / 2
        mask_b = ~np.isnan(bstat)
        ax.plot(bcenters[mask_b], bstat[mask_b], color="#E87722", lw=2.5, zorder=5)
    except Exception:
        pass

    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("SHAP value", fontsize=10)
    ax.set_title(f"Effect of {label.split(' (')[0]}", fontsize=10)

plt.suptitle("Feature Dependence Plots — How Key Variables Affect Predicted Yield",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("shap_plots/04_dependence_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → shap_plots/04_dependence_plots.png")


# ─────────────────────────────────────────────────────────────
# PLOT 5 — INTERACTION: FERTILIZER × AGRO-ZONE
# ─────────────────────────────────────────────────────────────
fert_idx = ALL_FEATURES.index("has_fertilizer")
zone_idx = ALL_FEATURES.index("zone_ordinal")
zone_names = {1: "Zone I", 2: "Zone IIa", 3: "Zone IIb", 4: "Zone III"}

fig, ax = plt.subplots(figsize=(8, 5))
zone_vals  = X["zone_ordinal"].astype(float).values
fert_vals  = X["has_fertilizer"].astype(float).values
shap_fert  = sv[:, fert_idx]

for zone_val, zone_label in sorted(zone_names.items()):
    for fert_val, ls, marker in [(0, "--", "o"), (1, "-", "s")]:
        mask = (np.round(zone_vals) == zone_val) & (fert_vals == fert_val)
        if mask.sum() < 10:
            continue
        mean_shap = shap_fert[mask].mean()
        ax.scatter(zone_val, mean_shap, marker=marker, s=80,
                   label=f"{zone_label}, fert={'yes' if fert_val else 'no'}")

# Compute mean SHAP of fertilizer by zone for with/without
zone_data = []
for zone_val in sorted(zone_names.keys()):
    mask_with    = (np.round(zone_vals) == zone_val) & (fert_vals == 1)
    mask_without = (np.round(zone_vals) == zone_val) & (fert_vals == 0)
    if mask_with.sum() > 5 and mask_without.sum() > 5:
        zone_data.append({
            "zone": zone_val,
            "with": shap_fert[mask_with].mean(),
            "without": shap_fert[mask_without].mean(),
        })

zdf = pd.DataFrame(zone_data)
zone_labels = [zone_names[z] for z in zdf["zone"]]
x = np.arange(len(zdf))
width = 0.35

ax.cla()
bars1 = ax.bar(x - width/2, zdf["with"],    width, label="With fertilizer",    color="#E87722")
bars2 = ax.bar(x + width/2, zdf["without"], width, label="Without fertilizer", color="#2166ac")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(zone_labels)
ax.set_ylabel("Mean SHAP value of fertilizer feature", fontsize=11)
ax.set_title("Fertilizer Effect by Agroecological Zone\n(interaction effect)", fontsize=12)
ax.legend()
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=8.5)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + (0.01 if h >= 0 else -0.04),
            f"{h:.2f}", ha="center", fontsize=8.5)
plt.tight_layout()
plt.savefig("shap_plots/05_interaction_fertilizer_zone.png", dpi=150)
plt.close()
print("Saved → shap_plots/05_interaction_fertilizer_zone.png")


# ─────────────────────────────────────────────────────────────
# PLOT 6 — RISK PROFILE: non-sellers vs sellers
# ─────────────────────────────────────────────────────────────
non_seller_mask = df["non_seller"].values.astype(bool)
seller_mask     = ~non_seller_mask

# Mean SHAP for top-15 features for each group
top15_features = importance["feature"].head(15).tolist()
top15_idx      = [ALL_FEATURES.index(f) for f in top15_features]
top15_names    = [DISPLAY_NAMES.get(f, f) for f in top15_features]

shap_sellers     = sv[seller_mask][:, top15_idx].mean(axis=0)
shap_nonsellers  = sv[non_seller_mask][:, top15_idx].mean(axis=0)
diff             = shap_nonsellers - shap_sellers

risk_df = pd.DataFrame({
    "feature": top15_names,
    "sellers":     shap_sellers,
    "non_sellers": shap_nonsellers,
    "diff":        diff,
}).sort_values("diff")

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(risk_df))
ax.barh(risk_df["feature"], risk_df["sellers"],    label="Sellers",     color="#2166ac", alpha=0.85)
ax.barh(risk_df["feature"], risk_df["non_sellers"],label="Non-sellers", color="#d6604d", alpha=0.85)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Mean SHAP value (contribution to predicted log yield)", fontsize=11)
ax.set_title("Risk Profile: SHAP Signatures of Non-sellers vs Sellers\n"
             "(red = non-seller average, blue = seller average)", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("shap_plots/06_risk_profile.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → shap_plots/06_risk_profile.png")

# Print top differentiating features for non-sellers
print("\nTop features separating non-sellers from sellers (SHAP diff):")
print(risk_df.sort_values("diff")[["feature","sellers","non_sellers","diff"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────
# PLOT 7 — WATERFALL: one high-yield vs one low-yield farmer
# ─────────────────────────────────────────────────────────────
# Pick a representative high and low yield farmer (among sellers)
yields = df.loc[seller_mask, "total_weight_kg"].values
seller_indices = np.where(seller_mask)[0]

# High: 80th percentile; Low: 20th percentile — avoid extreme outliers
high_thresh = np.percentile(yields, 80)
low_thresh  = np.percentile(yields, 20)

high_candidates = seller_indices[yields >= high_thresh]
low_candidates  = seller_indices[yields <= low_thresh]

# Pick the one closest to the percentile value
high_idx = high_candidates[np.argmin(np.abs(df.loc[high_candidates, "total_weight_kg"].values - high_thresh))]
low_idx  = low_candidates[np.argmin( np.abs(df.loc[low_candidates,  "total_weight_kg"].values - low_thresh))]

for label, idx, fname in [
    (f"High-yield farmer (≈{int(df.loc[high_idx,'total_weight_kg'])} kg)", high_idx, "07a_waterfall_high.png"),
    (f"Low-yield farmer  (≈{int(df.loc[low_idx, 'total_weight_kg'])} kg)", low_idx,  "07b_waterfall_low.png"),
]:
    exp_single = shap.Explanation(
        values        = shap_values.values[idx],
        base_values   = shap_values.base_values[idx],
        data          = shap_values.data[idx],
        feature_names = feature_names_display,
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(exp_single, max_display=15, show=False)
    plt.title(f"SHAP Waterfall — {label}", fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(f"shap_plots/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → shap_plots/{fname}")


# ─────────────────────────────────────────────────────────────
# SAVE FULL SHAP MATRIX
# ─────────────────────────────────────────────────────────────
shap_df = pd.DataFrame(sv, columns=ALL_FEATURES)
shap_df.insert(0, "farmer_id", df["farmer_id"].values)
shap_df.to_csv("shap_values_full.csv", index=False)
print("\nSaved → shap_values_full.csv")

# ─────────────────────────────────────────────────────────────
# SUMMARY PRINT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)

print("\n1. TOP 10 YIELD DRIVERS:")
for _, row in importance.head(10).iterrows():
    direction = "↑" if row["mean_shap"] > 0 else "↓"
    print(f"  {row['rank']:>2}. {row['display_name']:<35} |SHAP|={row['mean_abs_shap']:.4f}  {direction}")

print("\n2. INPUT EFFECTIVENESS (net SHAP effect of having input):")
inp_df_sorted = pd.DataFrame(rows).sort_values("net_effect", ascending=False)
for _, row in inp_df_sorted.iterrows():
    sign = "+" if row["net_effect"] > 0 else ""
    print(f"  {row['feature']:<35} {sign}{row['net_effect']:.4f}")

print("\n3. NON-SELLER RISK SIGNALS (top 5 features where non-sellers differ most):")
risk_top = risk_df.sort_values("diff").head(5)
for _, row in risk_top.iterrows():
    print(f"  {row['feature']:<35} diff={row['diff']:+.4f}  "
          f"(sellers={row['sellers']:.3f}, non-sellers={row['non_sellers']:.3f})")

print("\n✅  All plots saved to shap_plots/")
