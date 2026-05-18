"""
GNA Analytics Showdown — Feature Engineering Pipeline
Produces: master_features.csv (farmer-level, one row per farmer)

Season timeline:
  - Loan registration:  Dec 2024 – Jan 2025
  - Planting:           Sep 2024 – Mar 2025 (median Jan 1 2025)
  - Buyback / harvest:  Apr 2025 – Dec 2025 (median Jun 2025)

Target variable: total_weight_kg (kg sold back to GNA, 0 for non-sellers)
Risk target:     non_seller (1 = took loan but never sold back)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

RAW = "Datasets"
OUT = "."

SEASON_REF = pd.Timestamp("2024-11-01")  # season start for age / tenure calculations


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def parse_dates(series):
    """Parse DD/MM/YYYY or ISO dates robustly."""
    return pd.to_datetime(series, dayfirst=True, errors="coerce")

def yes_no(series):
    """Convert Yes/No strings to 1/0 int."""
    return (series == "Yes").astype(int)

def validate(label, series, *, min_val=None, max_val=None, max_null_pct=0.3):
    nulls = series.isna().mean()
    flag = "⚠️ " if nulls > max_null_pct else "  "
    msg = f"{flag}[{label}]  nulls={nulls:.1%}"
    if min_val is not None:
        below = (series < min_val).sum()
        if below:
            msg += f"  below_{min_val}={below}"
    if max_val is not None:
        above = (series > max_val).sum()
        if above:
            msg += f"  above_{max_val}={above}"
    print(msg)

print("=" * 65)
print("LOADING RAW DATA")
print("=" * 65)

farmer  = pd.read_csv(f"{RAW}/farmer_details.csv")
loan    = pd.read_csv(f"{RAW}/loan_details.csv")
buyback = pd.read_csv(f"{RAW}/buyback_details.csv")
planting = pd.read_csv(f"{RAW}/planting_survey.csv")

# Strip leading/trailing spaces from buyback column names
buyback.columns = buyback.columns.str.strip()

print(f"farmer:   {farmer.shape}")
print(f"loan:     {loan.shape}")
print(f"buyback:  {buyback.shape}")
print(f"planting: {planting.shape}")


# ─────────────────────────────────────────────────────────────
# 1. FARMER DETAILS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("1. FARMER DETAILS")
print("=" * 65)

farmer = farmer.rename(columns={"farmer_fea_id": "farmer_id"})

farmer["farmer_created_at"] = parse_dates(farmer["farmer_created_at"])
farmer["dob"]               = parse_dates(farmer["dob"])

# Age at season start
farmer["age"] = (SEASON_REF - farmer["dob"]).dt.days / 365.25

# Clamp implausible ages (< 18 or > 90 → NaN so model ignores them)
farmer.loc[farmer["age"] < 18, "age"] = np.nan
farmer.loc[farmer["age"] > 90, "age"] = np.nan

# Days as a GNA member by season start
farmer["days_as_member"] = (SEASON_REF - farmer["farmer_created_at"]).dt.days
farmer.loc[farmer["days_as_member"] < 0, "days_as_member"] = np.nan

# Binary demographics
farmer["is_female"] = (farmer["gender"] == "female").astype(int)

# Association type
farmer["is_certified_organic"]    = (farmer["association"] == "Organic, certified").astype(int)
farmer["is_converting_organic"]   = (farmer["association"] == "Organic, in conversion").astype(int)
farmer["is_organic"]              = (farmer["is_certified_organic"] | farmer["is_converting_organic"]).astype(int)

# Agroecological zone — ordinal encoding (higher = wetter = better for crops)
# Zone I < IIa < IIb < III based on Zambian rainfall classification
zone_map = {
    "I": 1, "IIa": 2, "IIb": 3, "III": 4,
    "I and IIa": 1.5, "IIa and III": 3.0, "IIb and III": 3.5,
}
farmer["zone_ordinal"] = farmer["agroecological_zone"].map(zone_map)

# Region / district — kept as raw strings for later target-encoding or one-hot
# (don't encode here to avoid leakage into train/test splits)

farmer_feat = farmer[[
    "farmer_id", "age", "days_as_member", "number_seasons",
    "is_female", "is_organic", "is_certified_organic", "is_converting_organic",
    "zone_ordinal", "agroecological_zone", "region_name", "district_name", "camp_name",
    "default_payment_method",
]]

print("Validation:")
validate("age",            farmer_feat["age"],            min_val=18, max_val=90)
validate("days_as_member", farmer_feat["days_as_member"], min_val=0)
validate("number_seasons", farmer_feat["number_seasons"], min_val=0, max_val=20)
validate("zone_ordinal",   farmer_feat["zone_ordinal"])
print(f"  Unique farmers: {farmer_feat['farmer_id'].nunique()}")


# ─────────────────────────────────────────────────────────────
# 2. LOAN DETAILS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("2. LOAN DETAILS")
print("=" * 65)

loan["account_package_created_at"] = parse_dates(loan["account_package_created_at"])

# Separate crop vs non-crop programs
# Non-crop: ASSETS, PRE-HARVEST LOANS, FAMILY PACKAGE (no hectares, no yield relevance)
CROP_PROG_PATTERN = r"SEED|SOURCE|ORGANIC|Partner"
loan["is_crop_program"] = loan["program_name"].str.contains(
    CROP_PROG_PATTERN, case=False, na=False
)
loan_crop    = loan[loan["is_crop_program"]].copy()
loan_noncrop = loan[~loan["is_crop_program"]].copy()

print(f"Crop program rows:     {len(loan_crop)}")
print(f"Non-crop program rows: {len(loan_noncrop)}")
print(f"Null hectares in crop: {loan_crop['package_hectares'].isna().sum()}  (should be 0)")

# Binary input columns
INPUT_COLS = ["fertilizer", "fungicide", "gypsum", "inoculant", "insecticide", "lime", "seed_guard"]
for col in INPUT_COLS:
    loan_crop[f"has_{col}"] = yes_no(loan_crop[col])

# Program flags per row (before aggregation)
loan_crop["is_source"]      = loan_crop["program_name"].str.contains("SOURCE",   case=False, na=False).astype(int)
loan_crop["is_seed"]        = loan_crop["program_name"].str.contains("SEED",     case=False, na=False).astype(int)
loan_crop["is_organic_prg"] = loan_crop["program_name"].str.contains("ORGANIC",  case=False, na=False).astype(int)
loan_crop["is_partnership"] = loan_crop["program_name"].str.contains("Partner",  case=False, na=False).astype(int)

# Dominant crop class (most hectares)
dominant_crop = (
    loan_crop.groupby(["farmer_id", "crop_class_name"])["package_hectares"]
    .sum()
    .reset_index()
    .sort_values("package_hectares", ascending=False)
    .groupby("farmer_id")
    .first()["crop_class_name"]
    .rename("dominant_crop")
)

# Aggregate crop loan to farmer level
agg_fns = {
    "package_hectares":         ("total_hectares",           "sum"),
    "id":                       ("n_loan_packages",          "count"),
    "crop_class_name":          ("n_crop_types_loaned",      "nunique"),
    "package_in_kind_repayment":("total_inkind_repayment",   "sum"),
    "package_cash_repayment":   ("total_cash_repayment",     "sum"),
    "initial_down_payment_value":("total_down_payment",      "sum"),
    "has_fertilizer":           ("has_fertilizer",           "max"),
    "has_fungicide":            ("has_fungicide",            "max"),
    "has_gypsum":               ("has_gypsum",               "max"),
    "has_inoculant":            ("has_inoculant",            "max"),
    "has_insecticide":          ("has_insecticide",          "max"),
    "has_lime":                 ("has_lime",                 "max"),
    "has_seed_guard":           ("has_seed_guard",           "max"),
    "is_source":                ("has_source_program",       "max"),
    "is_seed":                  ("has_seed_program",         "max"),
    "is_organic_prg":           ("has_organic_program",      "max"),
    "is_partnership":           ("has_partnership_program",  "max"),
    "account_package_created_at":("earliest_loan_date",      "min"),
}
loan_agg = loan_crop.groupby("farmer_id").agg(**{
    new: pd.NamedAgg(column=col, aggfunc=fn)
    for col, (new, fn) in agg_fns.items()
}).reset_index()

loan_agg = loan_agg.merge(dominant_crop.reset_index(), on="farmer_id", how="left")

# Input richness: weighted by effect size seen in EDA
# fertilizer=3, lime=3, fungicide=2, inoculant=2, others=1
WEIGHTS = {
    "has_fertilizer": 3, "has_lime": 3,
    "has_fungicide":  2, "has_inoculant": 2,
    "has_gypsum":     1, "has_insecticide": 1, "has_seed_guard": 1,
}
loan_agg["input_richness_score"] = sum(
    loan_agg[col] * w for col, w in WEIGHTS.items()
)
loan_agg["input_count"] = loan_agg[[f"has_{c}" for c in INPUT_COLS]].sum(axis=1)

# Non-crop loan features (asset / pre-harvest loans as additional financial signals)
noncrop_agg = loan_noncrop.groupby("farmer_id").agg(
    has_asset_loan     = ("program_name", lambda x: x.str.contains("ASSET", case=False).any()),
    has_preharvest_loan= ("program_name", lambda x: x.str.contains("PRE",   case=False).any()),
    has_family_package = ("program_name", lambda x: x.str.contains("FAMILY",case=False).any()),
).astype(int).reset_index()

print("\nValidation:")
validate("total_hectares",     loan_agg["total_hectares"],       min_val=0, max_val=5)
validate("input_richness_score",loan_agg["input_richness_score"],min_val=0)
validate("n_loan_packages",    loan_agg["n_loan_packages"],      min_val=1)
print(f"  Large farms (>5 ha): {(loan_agg['total_hectares'] > 5).sum()} — legitimate, kept as-is")
print(f"  dominant_crop distribution:\n{loan_agg['dominant_crop'].value_counts().to_string()}")


# ─────────────────────────────────────────────────────────────
# 3. PLANTING SURVEY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("3. PLANTING SURVEY")
print("=" * 65)

planting = planting.dropna(subset=["farmer_id"]).copy()
planting["farmer_id"]     = planting["farmer_id"].astype(int)
planting["planting_date"] = parse_dates(planting["planting_date"])
planting["has_training"]  = yes_no(planting["rcvd_crop_training"])
planting["multi_seed"]    = yes_no(planting["more_than_one_seed_in_hole"])

# Spacing: cap suspicious values
# <10cm likely data-entry error (cm keyed as mm or typo); >120cm clearly wrong
planting["spacing_suspicious"] = (
    (planting["spacing_cm_btwn_rows"] < 10) | (planting["spacing_cm_btwn_rows"] > 120)
).astype(int)
planting["spacing_clean"] = planting["spacing_cm_btwn_rows"].clip(lower=10, upper=120)
# Optimal spacing for legumes in Zambia is 45–60 cm between rows
planting["spacing_optimal"] = (
    planting["spacing_clean"].between(45, 60)
).astype(int)

# Planting dates before Oct 2024 are suspicious (very early / data errors)
planting.loc[planting["planting_date"] < "2024-10-01", "planting_date"] = pd.NaT

# Late planting: after Jan 31 2025 → higher frost / dry-spell risk at end of season
LATE_PLANTING_CUTOFF = pd.Timestamp("2025-01-31")
planting["is_late_planting"] = (planting["planting_date"] > LATE_PLANTING_CUTOFF).astype(int)

planting_agg = planting.groupby("farmer_id").agg(
    qty_kgs_planted      = ("Qty_kgs_planted",       "sum"),
    n_crops_planted      = ("crop_planted",          "nunique"),
    planting_date_first  = ("planting_date",         "min"),
    planting_date_last   = ("planting_date",         "max"),
    avg_spacing          = ("spacing_clean",         "mean"),
    pct_spacing_optimal  = ("spacing_optimal",       "mean"),
    pct_spacing_suspicious=("spacing_suspicious",    "mean"),
    has_training         = ("has_training",          "max"),
    pct_multi_seed       = ("multi_seed",            "mean"),
    any_late_planting    = ("is_late_planting",      "max"),
).reset_index()

# Planting season spread (days between first and last plot planted)
planting_agg["planting_spread_days"] = (
    planting_agg["planting_date_last"] - planting_agg["planting_date_first"]
).dt.days.fillna(0)

# Day of season for first planting (days since Nov 1 2024 = season start)
planting_agg["planting_doy"] = (
    planting_agg["planting_date_first"] - SEASON_REF
).dt.days
# Negative means planted before loan — informative, keep as-is

print("Validation:")
validate("qty_kgs_planted",        planting_agg["qty_kgs_planted"],     min_val=0)
validate("avg_spacing",            planting_agg["avg_spacing"],         min_val=10, max_val=120)
validate("pct_spacing_suspicious", planting_agg["pct_spacing_suspicious"])
validate("planting_doy",           planting_agg["planting_doy"],        min_val=-60, max_val=180)
print(f"  Farmers with late planting: {planting_agg['any_late_planting'].sum()}")
print(f"  Farmers with training:      {planting_agg['has_training'].sum()}")


# ─────────────────────────────────────────────────────────────
# 4. BUYBACK (TARGET)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("4. BUYBACK — TARGET VARIABLE")
print("=" * 65)

for col in ["total_weight", "total_net_weight", "net_owed_to_farmer"]:
    buyback[col] = pd.to_numeric(buyback[col], errors="coerce")

buyback["bbk_created_at"] = parse_dates(buyback["bbk_created_at"])

# Aggregate to farmer level
buyback_agg = buyback.groupby("farmer_id").agg(
    total_weight_kg    = ("total_weight",       "sum"),
    total_net_weight   = ("total_net_weight",   "sum"),
    net_owed_farmer    = ("net_owed_to_farmer", "sum"),
    grade_a_weight     = ("grade_a_weight",     "sum"),
    grade_b_weight     = ("grade_b_weight",     "sum"),
    grade_c_weight     = ("grade_c_weight",     "sum"),
    n_sellback_events  = ("id",                 "count"),
    n_crops_sold       = ("crop_class",         "nunique"),
    first_sellback_date= ("bbk_created_at",     "min"),
).reset_index()

# Grade quality distribution — use sum of all grade weights as denominator
# (total_weight has NaN rows that inflate the ratio if used as denominator)
grade_total = (
    buyback_agg["grade_a_weight"] +
    buyback_agg["grade_b_weight"] +
    buyback_agg["grade_c_weight"]
).replace(0, np.nan)
buyback_agg["grade_a_pct"] = (buyback_agg["grade_a_weight"] / grade_total).clip(0, 1)
buyback_agg["grade_b_pct"] = (buyback_agg["grade_b_weight"] / grade_total).clip(0, 1)
buyback_agg["grade_c_pct"] = (buyback_agg["grade_c_weight"] / grade_total).clip(0, 1)

# Days from season start to first sellback (harvest speed)
buyback_agg["days_to_sellback"] = (
    buyback_agg["first_sellback_date"] - SEASON_REF
).dt.days

print("Validation:")
validate("total_weight_kg", buyback_agg["total_weight_kg"], min_val=0)
validate("grade_a_pct",     buyback_agg["grade_a_pct"],     min_val=0, max_val=1)
print(f"  Farmers with 0 kg buyback:  {(buyback_agg['total_weight_kg']==0).sum()}")
print(f"  total_weight_kg > 10000:    {(buyback_agg['total_weight_kg']>10000).sum()}")


# ─────────────────────────────────────────────────────────────
# 5. MERGE INTO MASTER DATASET
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("5. MERGING")
print("=" * 65)

master = (
    farmer_feat
    .merge(loan_agg,     on="farmer_id", how="left")
    .merge(noncrop_agg,  on="farmer_id", how="left")
    .merge(planting_agg, on="farmer_id", how="left")
    .merge(buyback_agg,  on="farmer_id", how="left")
)

print(f"Master shape after merge: {master.shape}")

# Fill non-crop loan flags as 0 (farmer didn't take those loans)
for col in ["has_asset_loan", "has_preharvest_loan", "has_family_package"]:
    master[col] = master[col].fillna(0).astype(int)


# ─────────────────────────────────────────────────────────────
# 6. CROSS-DATASET FEATURES (require merged table)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("6. CROSS-DATASET FEATURES")
print("=" * 65)

# Days from loan signup to first planting (delay signal)
master["days_loan_to_plant"] = (
    master["planting_date_first"] - master["earliest_loan_date"]
).dt.days
# Negative = planted before loan (backfilled loan date, or early planter)
# < -180 = planting from a different season entirely → null out
master.loc[master["days_loan_to_plant"] < -180, "days_loan_to_plant"] = np.nan

# Seed density: kg of seed planted per hectare contracted
# Cap at 200 kg/ha — max realistic for beans; values above are likely data errors
master["seed_density_kg_per_ha"] = (
    master["qty_kgs_planted"] / master["total_hectares"].replace(0, np.nan)
).clip(upper=200)

# Yield (target) and yield per hectare
master["has_buyback"]    = master["total_weight_kg"].notna().astype(int)
master["non_seller"]     = (master["has_buyback"] == 0).astype(int)   # RISK TARGET

# Fill target with 0 for non-sellers (they produced 0 kg for GNA)
master["total_weight_kg"] = master["total_weight_kg"].fillna(0)

# Yield per hectare — meaningful measure normalised for farm size
master["yield_per_ha"] = (
    master["total_weight_kg"] / master["total_hectares"].replace(0, np.nan)
)

# Log-transformed target for modelling (highly right-skewed; use log1p)
master["log_total_weight"] = np.log1p(master["total_weight_kg"])

# Repayment ratio: how much of expected in-kind repayment was actually delivered
# (proxy for loan compliance; only meaningful for buyers)
master["inkind_coverage_ratio"] = (
    master["total_weight_kg"] / master["total_inkind_repayment"].replace(0, np.nan)
).clip(upper=20)   # cap at 20x to bound extreme ratios

# Interaction features with domain backing
master["fertilizer_x_zone"]    = master["has_fertilizer"] * master["zone_ordinal"]
master["lime_x_zone"]          = master["has_lime"]       * master["zone_ordinal"]
master["experience_x_training"]= master["number_seasons"] * master["has_training"].fillna(0)
master["rich_inputs_x_hectares"]= master["input_richness_score"] * master["total_hectares"]

print("Validation:")
validate("days_loan_to_plant",    master["days_loan_to_plant"],   min_val=-30, max_val=120)
validate("seed_density_kg_per_ha",master["seed_density_kg_per_ha"], min_val=0, max_val=300)
validate("yield_per_ha",          master["yield_per_ha"],          min_val=0, max_val=30000)
validate("inkind_coverage_ratio", master["inkind_coverage_ratio"], min_val=0)
print(f"  non_seller rate: {master['non_seller'].mean():.1%}")
print(f"  has planting survey: {master['planting_date_first'].notna().mean():.1%}")


# ─────────────────────────────────────────────────────────────
# 7. FINAL VALIDATION REPORT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("7. FINAL VALIDATION REPORT")
print("=" * 65)

FEATURE_COLS = [
    # Farmer demographics
    "age", "is_female", "is_organic", "number_seasons", "days_as_member",
    "zone_ordinal",
    # Loan / input package
    "total_hectares", "n_loan_packages", "n_crop_types_loaned",
    "has_fertilizer", "has_fungicide", "has_gypsum", "has_inoculant",
    "has_insecticide", "has_lime", "has_seed_guard",
    "input_count", "input_richness_score",
    "has_source_program", "has_seed_program", "has_organic_program", "has_partnership_program",
    "total_inkind_repayment", "total_cash_repayment", "total_down_payment",
    "has_asset_loan", "has_preharvest_loan", "has_family_package",
    # Planting behaviour
    "qty_kgs_planted", "n_crops_planted", "avg_spacing",
    "pct_spacing_optimal", "has_training", "pct_multi_seed",
    "any_late_planting", "planting_doy", "planting_spread_days",
    # Cross-dataset
    "days_loan_to_plant", "seed_density_kg_per_ha",
    "fertilizer_x_zone", "lime_x_zone",
    "experience_x_training", "rich_inputs_x_hectares",
]

TARGET_COLS = [
    "total_weight_kg", "log_total_weight", "yield_per_ha",
    "non_seller", "has_buyback",
    "grade_a_pct", "n_sellback_events",
]

all_cols = ["farmer_id"] + FEATURE_COLS + TARGET_COLS + [
    "dominant_crop", "agroecological_zone", "region_name", "district_name", "camp_name",
    "default_payment_method", "earliest_loan_date",
]

master_out = master[all_cols].copy()

print(f"\nFinal shape:  {master_out.shape}")
print(f"Farmers:      {master_out['farmer_id'].nunique()}")
print(f"\nNull rates for features (top 15 highest):")
null_rates = master_out[FEATURE_COLS].isna().mean().sort_values(ascending=False)
print(null_rates.head(15).to_string())

print(f"\nTarget summary:")
print(master_out[TARGET_COLS].describe().to_string())

# Leakage check: no buyback-derived columns in FEATURE_COLS
buyback_derived = {"grade_a_pct", "grade_b_pct", "grade_c_pct",
                   "net_owed_farmer", "n_sellback_events",
                   "total_net_weight", "days_to_sellback"}
leaked = set(FEATURE_COLS) & buyback_derived
assert not leaked, f"LEAKAGE DETECTED in feature cols: {leaked}"
print("\n✅  No leakage: buyback-derived columns excluded from FEATURE_COLS")

# Duplicate farmer check
assert master_out["farmer_id"].is_unique, "DUPLICATE farmer_ids in master!"
print("✅  No duplicate farmer_ids")

# All input flags are binary
for col in [f"has_{c}" for c in ["fertilizer","fungicide","gypsum","inoculant","insecticide","lime","seed_guard"]]:
    assert master_out[col].dropna().isin([0, 1]).all(), f"{col} has non-binary values"
print("✅  All input flags are 0/1")

# ─────────────────────────────────────────────────────────────
# 8. SAVE
# ─────────────────────────────────────────────────────────────
out_path = f"{OUT}/master_features.csv"
master_out.to_csv(out_path, index=False)
print(f"\n✅  Saved → {out_path}")
print(f"    {master_out.shape[0]} rows × {master_out.shape[1]} columns")
