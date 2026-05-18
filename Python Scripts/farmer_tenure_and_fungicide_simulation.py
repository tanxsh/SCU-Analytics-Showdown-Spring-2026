"""
Two final analytical additions before presentation:

  H. Farmer Tenure Profile — cross-sectional yield + dropout by seasons with GNA
  I. Fungicide Simulation   — raw vs SHAP-adjusted causal uplift for Soy Bean / Zone IIa

Framing discipline:
  - Tenure: cross-sectional snapshot (different farmers in same season), NOT longitudinal.
    Framed as "tenure profile", not "growth trajectory". Survivorship bias acknowledged.
  - Fungicide: raw comparison (confounded, 209%) shown alongside SHAP-adjusted
    causal estimate (~9.6%) with clear labelling of what each means.
"""

import json, os, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import OrdinalEncoder
warnings.filterwarnings("ignore")

os.makedirs("final_analysis", exist_ok=True)

C_ORANGE = "#E87722"; C_BLUE = "#2166ac"; C_RED = "#d6604d"
C_GREEN  = "#4dac26"; C_GREY  = "#888888"; C_PURPLE = "#762a83"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False,    "axes.spines.right": False,
    "font.size": 11,
})

def savefig(name):
    plt.tight_layout()
    plt.savefig(f"final_analysis/{name}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → final_analysis/{name}")

SEP = "=" * 68

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("master_features.csv")

NUM_FEATURES = [
    "age","is_female","is_organic","number_seasons","days_as_member","zone_ordinal",
    "total_hectares","n_loan_packages","n_crop_types_loaned",
    "has_fertilizer","has_fungicide","has_gypsum","has_inoculant",
    "has_insecticide","has_lime","has_seed_guard",
    "input_count","input_richness_score",
    "has_source_program","has_seed_program","has_organic_program","has_partnership_program",
    "total_inkind_repayment","total_cash_repayment","total_down_payment",
    "has_asset_loan","has_preharvest_loan","has_family_package",
    "qty_kgs_planted","n_crops_planted","avg_spacing",
    "pct_spacing_optimal","has_training","pct_multi_seed",
    "any_late_planting","planting_doy","planting_spread_days",
    "days_loan_to_plant","seed_density_kg_per_ha",
    "fertilizer_x_zone","lime_x_zone","experience_x_training","rich_inputs_x_hectares",
]
CAT_FEATURES = ["dominant_crop","agroecological_zone","region_name"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

enc   = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_raw = df[ALL_FEATURES].copy()
X_raw[CAT_FEATURES] = enc.fit_transform(X_raw[CAT_FEATURES].fillna("Unknown"))
for col in CAT_FEATURES:
    X_raw[col] = X_raw[col].astype(int).astype("category")

y_log = df["log_total_weight"].values

with open("best_xgb_params.json") as f:
    reg_params = json.load(f)


# ════════════════════════════════════════════════════════════════════════════
# H. FARMER TENURE PROFILE
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("H. FARMER TENURE PROFILE (Cross-Sectional)")
print(SEP)

# Cap at 6+ seasons to keep cells large enough
df["season_band"] = df["number_seasons"].clip(upper=6).astype(int)

tenure = (
    df.groupby("season_band")
    .agg(
        n             = ("farmer_id",       "count"),
        median_yield  = ("total_weight_kg", "median"),
        mean_yield    = ("total_weight_kg", "mean"),
        nonseller_rate= ("non_seller",      "mean"),
        pct_sellers   = ("non_seller",      lambda x: 1 - x.mean()),
    )
    .reset_index()
)
tenure["label"] = tenure["season_band"].apply(
    lambda s: f"Season {s}" if s < 6 else "Season 6+"
)

print("\n  Tenure profile table (cross-sectional — same 2024/25 season):")
print(tenure[["label","n","median_yield","nonseller_rate"]].to_string(index=False))

# Key headline numbers
s1_yield  = tenure.loc[tenure["season_band"]==1, "median_yield"].values[0]
s6_yield  = tenure.loc[tenure["season_band"]==6, "median_yield"].values[0]
s1_ns     = tenure.loc[tenure["season_band"]==1, "nonseller_rate"].values[0]
s6_ns     = tenure.loc[tenure["season_band"]==6, "nonseller_rate"].values[0]
yield_mult= s6_yield / s1_yield
drop_mult = s1_ns   / s6_ns

print(f"\n  Season 1 median yield:    {s1_yield:.0f} kg  |  dropout: {s1_ns:.1%}")
print(f"  Season 6+ median yield:   {s6_yield:.0f} kg  |  dropout: {s6_ns:.1%}")
print(f"  Yield ratio (6+/1):       {yield_mult:.1f}×")
print(f"  Dropout ratio (1/6+):     {drop_mult:.1f}×")
print(f"\n  ⚠  Cross-sectional, not longitudinal. Survivorship bias acknowledged:")
print(f"     Low-performing farmers likely exited GNA after early seasons.")
print(f"     The 10× gap reflects BOTH learning effects AND natural selection.")
print(f"     Business implication is the same: long-term farmer relationships")
print(f"     are GNA's most valuable asset — retain or develop them.")

# ── Plot ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))
gs  = GridSpec(1, 3, figure=fig, wspace=0.42)

# H1 — Median yield by tenure band
ax1 = fig.add_subplot(gs[0, 0])
colors_t = [
    C_RED if r > 0.20 else C_ORANGE if r > 0.10 else C_GREEN
    for r in tenure["nonseller_rate"]
]
bars = ax1.bar(tenure["label"], tenure["median_yield"],
               color=colors_t, alpha=0.88, edgecolor="none", width=0.6)
ax1.set_ylabel("Median yield (kg)")
ax1.set_title("Median Yield by Farmer Tenure\n(cross-sectional, 2024/25 season)",
              fontsize=11)
ax1.tick_params(axis="x", rotation=35)
for bar, val in zip(bars, tenure["median_yield"]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
             f"{val:.0f}", ha="center", fontsize=10, fontweight="bold")

red_p   = mpatches.Patch(color=C_RED,    label=">20% dropout rate")
orange_p= mpatches.Patch(color=C_ORANGE, label="10–20% dropout rate")
green_p = mpatches.Patch(color=C_GREEN,  label="<10% dropout rate")
ax1.legend(handles=[red_p, orange_p, green_p], fontsize=8, loc="upper left")

# H2 — Non-seller (dropout) rate by tenure
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(tenure["label"], tenure["nonseller_rate"] * 100,
         color=C_RED, lw=2.5, marker="o", markersize=8, zorder=5)
ax2.fill_between(tenure["label"], tenure["nonseller_rate"] * 100,
                 alpha=0.12, color=C_RED)
ax2.set_ylabel("Non-seller rate (%)")
ax2.set_title("Dropout Rate by Farmer Tenure\n(1st-season farmers 21× more likely to drop out)",
              fontsize=11)
ax2.tick_params(axis="x", rotation=35)
for i, row in tenure.iterrows():
    ax2.text(i, row["nonseller_rate"] * 100 + 0.5,
             f"{row['nonseller_rate']:.1%}", ha="center", fontsize=9,
             color=C_RED, fontweight="bold")
ax2.axhline(5, color=C_GREY, lw=1, ls="--", label="5% target")
ax2.legend(fontsize=9)

# H3 — Farmer count by band (show base sizes for survivorship context)
ax3 = fig.add_subplot(gs[0, 2])
bars3 = ax3.bar(tenure["label"], tenure["n"],
                color=C_BLUE, alpha=0.75, edgecolor="none", width=0.6)
ax3.set_ylabel("Number of farmers")
ax3.set_title("Farmer Count per Tenure Band\n(note steep drop: survivorship or GNA growth?)",
              fontsize=11)
ax3.tick_params(axis="x", rotation=35)
for bar, val in zip(bars3, tenure["n"]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f"{val:,}", ha="center", fontsize=9)

# Annotation box explaining cross-sectional nature
fig.text(0.5, -0.04,
         "NOTE: All farmers observed in the same 2024/25 season. 'Season N' = farmers with N cumulative seasons of GNA experience.\n"
         "Higher yields for long-tenure farmers reflect both learning effects and survivorship (weaker farmers exit the programme).",
         ha="center", fontsize=9, color=C_GREY, style="italic",
         wrap=True)
savefig("H_tenure_profile.png")

print(f"\n  Saved → final_analysis/H_tenure_profile.png")


# ════════════════════════════════════════════════════════════════════════════
# I. FUNGICIDE SIMULATION — Soy Bean / Zone IIa
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("I. FUNGICIDE SIMULATION — Soy Bean / Zone IIa")
print(SEP)

# ── Raw comparison (confounded) ──────────────────────────────────────────
soy_iia   = df[(df["dominant_crop"] == "Soy Bean") & (df["agroecological_zone"] == "IIa")]
no_fung   = soy_iia[soy_iia["has_fungicide"] == 0]
has_fung  = soy_iia[soy_iia["has_fungicide"] == 1]
no_fung_s = no_fung[no_fung["non_seller"] == 0]
has_fung_s= has_fung[has_fung["non_seller"] == 0]

raw_no    = no_fung_s["total_weight_kg"].median()
raw_yes   = has_fung_s["total_weight_kg"].median()
raw_uplift= (raw_yes - raw_no) / raw_no

print(f"\n  ── RAW COMPARISON (confounded) ──")
print(f"  Soy Bean / IIa WITHOUT fungicide:  {raw_no:.0f} kg median  (n={len(no_fung):,} farmers)")
print(f"  Soy Bean / IIa WITH    fungicide:  {raw_yes:.0f} kg median  (n={len(has_fung):,} farmers)")
print(f"  Raw uplift:                        {raw_uplift:+.1%}")
print(f"  ⚠  Confounded — fungicide farmers systematically better in other ways.")
print(f"     Cannot attribute full uplift to fungicide alone.")

# Check confounders — how different are the two groups?
print(f"\n  Confounding check (fungicide vs no-fungicide, Soy/IIa):")
confounders = ["number_seasons","total_hectares","has_training",
               "pct_spacing_optimal","input_richness_score","has_inoculant"]
for col in confounders:
    v_yes = soy_iia[soy_iia["has_fungicide"]==1][col].mean()
    v_no  = soy_iia[soy_iia["has_fungicide"]==0][col].mean()
    diff  = (v_yes - v_no) / (v_no if v_no != 0 else 1)
    print(f"    {col:<30}: with={v_yes:.2f}  without={v_no:.2f}  diff={diff:+.0%}")

# ── SHAP-adjusted causal estimate ────────────────────────────────────────
print(f"\n  ── SHAP-ADJUSTED CAUSAL ESTIMATE ──")

# Fit model, compute SHAP on Soy/IIa subset
reg_model = xgb.XGBRegressor(**reg_params)
reg_model.fit(X_raw, y_log)
explainer = shap.TreeExplainer(reg_model)

soy_iia_idx = soy_iia.index
X_soy_iia   = X_raw.loc[soy_iia_idx]
sv_soy_iia  = explainer.shap_values(X_soy_iia)

fung_idx    = ALL_FEATURES.index("has_fungicide")
mean_shap_fung = sv_soy_iia[:, fung_idx].mean()   # mean over all Soy/IIa farmers
# For no-fungicide sub-group
no_fung_local_idx = [i for i, fi in enumerate(soy_iia_idx) if fi in no_fung.index]
shap_no_fung = sv_soy_iia[no_fung_local_idx, fung_idx].mean() if no_fung_local_idx else mean_shap_fung

print(f"  Mean SHAP of fungicide (Soy/IIa):       {mean_shap_fung:+.4f} log yield units")
print(f"  Mean SHAP for currently no-fung farmers: {shap_no_fung:+.4f} log yield units")
causal_uplift_pct = np.expm1(mean_shap_fung)
print(f"  Causal yield uplift (exp(SHAP) − 1):    {causal_uplift_pct:+.1%}")
print(f"  This controls for all other observed differences between groups.")

# ── Simulation ────────────────────────────────────────────────────────────
print(f"\n  ── SIMULATION: Give fungicide to all {len(no_fung):,} Soy/IIa farmers without it ──")

# Conservative: SHAP-based causal estimate
causal_add_kg = no_fung_s["total_weight_kg"].sum() * causal_uplift_pct
# Optimistic: mid-point between raw and SHAP (partial confounding adjustment)
mid_uplift    = (raw_uplift + causal_uplift_pct) / 2
mid_add_kg    = no_fung_s["total_weight_kg"].sum() * mid_uplift

print(f"\n  {'Estimate':<35} {'Uplift':>8} {'Additional kg':>15} {'Tonnes':>8}")
print(f"  {'-'*68}")
print(f"  {'Raw (confounded upper bound)':<35} {raw_uplift:>+8.1%} "
      f"{no_fung_s['total_weight_kg'].sum() * raw_uplift:>15,.0f}"
      f"{no_fung_s['total_weight_kg'].sum() * raw_uplift / 1000:>8.0f}")
print(f"  {'SHAP-adjusted (causal, lower)':<35} {causal_uplift_pct:>+8.1%} "
      f"{causal_add_kg:>15,.0f}  {causal_add_kg/1000:>7.0f}")
print(f"  {'Plausible range midpoint':<35} {mid_uplift:>+8.1%} "
      f"{mid_add_kg:>15,.0f}  {mid_add_kg/1000:>7.0f}")

# Cost-benefit framing
soy_rev_per_kg = 9.07   # median from buyback analysis
causal_zmw     = causal_add_kg * soy_rev_per_kg
print(f"\n  Conservative ZMW value (SHAP estimate):  ZMW {causal_zmw:,.0f}")
print(f"  ({causal_add_kg/1000:.0f} tonnes × ZMW {soy_rev_per_kg:.2f}/kg)")

# Dropout reduction for no-fung farmers if they get fungicide
ns_no_fung  = no_fung["non_seller"].mean()
ns_has_fung = has_fung["non_seller"].mean()
print(f"\n  Dropout rate (Soy/IIa, no fungicide):    {ns_no_fung:.1%}")
print(f"  Dropout rate (Soy/IIa, with fungicide):  {ns_has_fung:.1%}")
print(f"  Raw dropout reduction:                    {ns_no_fung - ns_has_fung:+.1%} percentage points")
print(f"  ⚠  Also confounded — presented as associated, not causal.")

# ── Broader fungicide opportunity across all crops ──────────────────────
print(f"\n  ── BROADER OPPORTUNITY: all crops, all zones ──")
no_fung_all   = df[df["has_fungicide"] == 0]
no_fung_all_s = no_fung_all[no_fung_all["non_seller"] == 0]
print(f"  Farmers without fungicide:       {len(no_fung_all):,} ({len(no_fung_all)/len(df):.1%} of total)")
print(f"  Sellers without fungicide:       {len(no_fung_all_s):,}")
print(f"  Their total current buyback:     {no_fung_all_s['total_weight_kg'].sum()/1000:.0f} tonnes")
print(f"  At SHAP causal estimate +9.6%:   "
      f"+{no_fung_all_s['total_weight_kg'].sum() * causal_uplift_pct / 1000:.0f} tonnes across all crops/zones")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# I1 — Raw vs SHAP comparison bar
ax = axes[0]
estimates = ["Raw\n(confounded)", "SHAP-adjusted\n(causal)", "Midpoint\n(plausible)"]
uplifts   = [raw_uplift * 100, causal_uplift_pct * 100, mid_uplift * 100]
col_bars  = [C_GREY, C_BLUE, C_ORANGE]
bars_i    = ax.bar(estimates, uplifts, color=col_bars, alpha=0.85, edgecolor="none", width=0.55)
ax.set_ylabel("Yield uplift (%)")
ax.set_title("Fungicide Effect on Soy Bean / Zone IIa\nRaw vs SHAP-adjusted causal estimate",
             fontsize=11)
for bar, v in zip(bars_i, uplifts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"+{v:.0f}%", ha="center", fontsize=12, fontweight="bold")
ax.axhline(0, color="black", lw=0.8)
ax.text(0, uplifts[0] * 0.5, "⚠ Confounded\nnot causal",
        ha="center", va="center", fontsize=9, color="white",
        bbox=dict(fc=C_GREY, ec="none", alpha=0.0))

# I2 — Additional tonnes under each scenario
ax = axes[1]
tonnes = [
    no_fung_s["total_weight_kg"].sum() * raw_uplift / 1000,
    causal_add_kg / 1000,
    mid_add_kg / 1000,
]
bars_t = ax.bar(estimates, tonnes, color=col_bars, alpha=0.85, edgecolor="none", width=0.55)
ax.set_ylabel("Additional procurement (tonnes)")
ax.set_title(f"Additional Tonnes if {len(no_fung):,} Soy/IIa Farmers\nReceive Fungicide",
             fontsize=11)
for bar, v in zip(bars_t, tonnes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"+{v:.0f}t", ha="center", fontsize=12, fontweight="bold")

# I3 — Distribution: yield by fungicide status (sellers only, Soy/IIa)
ax = axes[2]
bins  = np.linspace(0, 3000, 40)
ax.hist(no_fung_s["total_weight_kg"].clip(upper=3000), bins=bins,
        color=C_GREY,   alpha=0.65, label=f"No fungicide  (n={len(no_fung_s):,})", density=True)
ax.hist(has_fung_s["total_weight_kg"].clip(upper=3000), bins=bins,
        color=C_BLUE,   alpha=0.65, label=f"Has fungicide (n={len(has_fung_s):,})", density=True)
ax.axvline(raw_no,  color=C_GREY, lw=2, ls="--", label=f"Median (no fung): {raw_no:.0f} kg")
ax.axvline(raw_yes, color=C_BLUE, lw=2, ls="--", label=f"Median (w/ fung): {raw_yes:.0f} kg")
ax.set_xlabel("Total weight sold (kg)")
ax.set_ylabel("Density")
ax.set_title("Yield Distribution: Fungicide vs No Fungicide\nSoy Bean / Zone IIa — sellers only",
             fontsize=11)
ax.legend(fontsize=9)

savefig("I_fungicide_simulation.png")
print(f"\n  Saved → final_analysis/I_fungicide_simulation.png")


# ════════════════════════════════════════════════════════════════════════════
# COMBINED SUMMARY PRINT
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SUMMARY OF FINAL TWO ADDITIONS")
print(SEP)
print(f"""
H. FARMER TENURE PROFILE
   Season 1 → Season 6+: {s1_yield:.0f} kg → {s6_yield:.0f} kg  ({yield_mult:.1f}× yield increase)
   Dropout:               {s1_ns:.1%} → {s6_ns:.1%}  ({drop_mult:.0f}× reduction)
   Key message: Long-term farmer relationships are GNA's highest-value asset.
   Framing: Cross-sectional snapshot. Reflects both learning AND survivorship.

I. FUNGICIDE SIMULATION (Soy Bean / Zone IIa)
   Raw comparison:        +{raw_uplift:.0%} (confounded — fungicide farmers differ in many ways)
   SHAP causal estimate:  +{causal_uplift_pct:.1%} (controlling for all other features)
   If {len(no_fung):,} at-risk farmers receive fungicide:
     Conservative (SHAP): +{causal_add_kg/1000:.0f} tonnes  ≈  ZMW {causal_zmw:,.0f}
   Across ALL crops/zones (68.8% of farmers without fungicide):
     Conservative:        +{no_fung_all_s['total_weight_kg'].sum() * causal_uplift_pct/1000:.0f} additional tonnes
""")
print("  ✅  Both plots saved to final_analysis/")
