# **🏆 Winner — SCU Analytics Showdown, Spring 2026**

## Roots & Returns — The Story

Every growing season, Good Nature Agro (GNA) extends agricultural input loans — seeds, fungicide, fertilizer, to thousands of smallholder farmers across Zambia. In return, farmers are expected to sell their harvest back to GNA at the end of the season.

But **23.3% of farmers never sell back**. They take the inputs and disappear.

This project answers the question GNA faces every season: **which farmers will we lose — and what can we do about it before it's too late?**

---

## The Model

Built a **two-stage XGBoost hurdle model** that separates the problem into two distinct questions:

```
Stage 1: Will this farmer sell back at all?     → XGBoost Classifier  (AUC: 0.880)
Stage 2: If they do sell back, how much?        → XGBoost Regressor   (RMSE: 594 kg)

Expected Yield = P(sells back) × E[yield | sells back]
```

This architecture handles the 23.3% zero-yield non-sellers that break a standard regression model.

### Risk Tiers

Farmers are assigned to one of three risk tiers based on their expected yield:

| Tier | Expected Yield | Actual Non-Seller Rate | Action |
|------|---------------|----------------------|--------|
| 🔴 **HIGH** | ≤ 83 kg | 54.5% | Immediate intervention |
| 🟠 **MEDIUM** | 84 – 216 kg | 11.7% | Monitor and support |
| 🟢 **LOW** | > 216 kg | 3.9% | Reliable |

> By flagging just the top 27% of farmers by risk, the model catches **74% of all non-sellers**.

---

## Key Findings

| # | Insight | Impact |
|---|---------|--------|
| 1 | Season 1 farmers have a **27.7% dropout rate** — 20× higher than Season 6+ farmers | Prioritise early-season outreach for first-timers |
| 2 | **Fungicide** is the #1 controllable yield driver — Soy Bean farmers with fungicide yield 463 kg vs 160 kg without | Expand fungicide access; 69% of Soy Bean farmers don't receive it |
| 3 | **Western and Southern regions** have 50%+ non-seller rates — drought-impacted crisis zones | Region-specific intervention strategies needed |
| 4 | **Partnership Programme** farmers have 57% non-seller rate and 0% fungicide adoption | Pairing with even one additional input is the highest-leverage investment |
| 5 | HIGH-risk tier flagging recovers an estimated **ZMW 2.3M** in at-risk procurement per season | Direct financial case for proactive risk management |

---

## Repository Structure

```
├── Jupyter Notebooks/
│   ├── 00_Data_Preparation.ipynb               ← Feature engineering & dataset construction
│   ├── 01_Predictive_Model.ipynb               ← Model selection, training & validation
│   ├── 02_Yield_Prediction_Input_Effectiveness ← SHAP analysis & input effectiveness
│   └── 03_Risk_Identification.ipynb            ← Risk tier assignment & profiling
│
├── Python Scripts/
│   ├── data_feature_engineering.py             ← Full feature engineering pipeline
│   ├── two_stage_hurdle_risk_model.py          ← Two-stage model implementation
│   ├── xgboost_hyperparameter_tuning.py        ← Bayesian hyperparameter search
│   ├── shap_input_importance_analysis.py       ← SHAP value computation
│   ├── model_selection_benchmark.py            ← 7-model benchmark comparison
│   ├── model_validation_and_narrative.py       ← Validation & business narrative
│   ├── farmer_tenure_and_fungicide_simulation  ← Tenure analysis & simulations
│   └── advanced_analytics_and_scorecard.py     ← Operational scorecard
│
├── JSON Files/
│   ├── best_xgb_params.json                    ← Tuned XGBoost hyperparameters
│   ├── stage1_params.json                      ← Stage 1 classifier params
│   └── stage2_params.json                      ← Stage 2 regressor params
│
└── Visualizations/
    └── 01–15 publication-ready plots
```

---

> **Note:** The `Datasets/` folder is not included in this repository due to data privacy. 

---

## Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)

`pandas` · `numpy` · `scikit-learn` · `matplotlib` · `XGBoost` · `SHAP` · `Streamlit`

---
