# **Competition Winner — SCU Leavey School of Business Analytics Showdown (Spring 2026)**

### Roots & Returns - SCU Analytics Showdown Spring 2026

A data science competition project analyzing farmer non-seller risk for **Good Nature Agro (GNA)**, an agricultural input loan company operating across Zambia. The goal was to predict which farmers would not sell their harvest back to GNA and translate that into actionable business recommendations.

---

## The Problem

GNA extends agricultural inputs (seeds, fungicide, fertilizer) on credit to smallholder farmers across Zambia. A **non-seller** is a farmer who takes the loan but does not sell their harvest back to GNA — representing both a financial loss and a missed development opportunity. 23.3% of farmers in the 2024/25 season were non-sellers.

---

## Our Approach

We built a **two-stage XGBoost hurdle model**:

- **Stage 1 — Classifier**: Predicts the probability a farmer will sell back (AUC: 0.880)
- **Stage 2 — Regressor**: Predicts yield in kg for farmers who do sell back (RMSE: 594 kg)
- **Expected Yield** = P(sells back) × E[yield | sells back]

Farmers are segmented into three risk tiers based on expected yield:

| Tier | Expected Yield | Actual Non-Seller Rate |
|------|---------------|----------------------|
| 🔴 HIGH | ≤ 83 kg | 54.5% |
| 🟠 MEDIUM | 84 – 216 kg | 11.7% |
| 🟢 LOW | > 216 kg | 3.9% |

By flagging the top 27% of farmers by risk, the model catches **74% of all non-sellers**.

---

## Repository Structure

```
├── Jupyter Notebooks/
│   ├── 00_Data_Preparation.ipynb
│   ├── 01_Predictive_Model.ipynb
│   ├── 02_Yield_Prediction_Input_Effectiveness.ipynb
│   └── 03_Risk_Identification.ipynb
├── Python Scripts/
│   ├── data_feature_engineering.py
│   ├── two_stage_hurdle_risk_model.py
│   ├── xgboost_hyperparameter_tuning.py
│   ├── shap_input_importance_analysis.py
│   ├── model_selection_benchmark.py
│   ├── model_validation_and_narrative.py
│   ├── farmer_tenure_and_fungicide_simulation.py
│   └── advanced_analytics_and_scorecard.py
├── JSON Files/
│   ├── best_xgb_params.json
│   ├── stage1_params.json
│   └── stage2_params.json
└── Visualizations/
    └── 01–15 publication-ready plots
```

---

## Key Findings

- **Fungicide** is the #1 controllable driver of yield — Soy Bean farmers with fungicide yield 463 kg vs 160 kg without
- **Season 1 farmers** have a 27.7% dropout rate — the highest-risk cohort by far
- **Western and Southern regions** are crisis zones with 50%+ non-seller rates
- **Partnership Programme** farmers are the most vulnerable cohort — 57% non-seller rate, 0% fungicide adoption
- Flagging the HIGH-risk tier alone recovers an estimated **ZMW 2.3M** in at-risk procurement per season

---

## Tools & Technologies

- **Python** — pandas, numpy, XGBoost, scikit-learn, SHAP, matplotlib
- **Deployed App** — [Good Nature Agro Farmer Risk Assessment Tool](https://github.com/tanxsh/gna-risk-tool)

---

## How to Run

1. Clone the repository
2. Upload all folders to the same directory (Datasets folder required separately — not included due to data privacy)
3. Run notebooks in order: `00` → `01` → `02` → `03`

---

*SCU Leavey School of Business · Group 9 · Spring 2026*
