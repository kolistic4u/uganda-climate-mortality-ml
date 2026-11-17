# Data Science & Machine Learning Capstone Projects

This repository contains two complete, end-to-end machine learning solutions developed for the Data Science & ML Capstone Project.

---

## Project 1: Predicting Climate-Sensitive Mortality in Uganda
* **Objective:** Predict whether mortality records in Uganda (`N = 3,146` training records) are climate-sensitive (`is_climate_sensitive`), and generalize predictions to `1,030` records across unseen villages.
* **Methodology & Key Results:** Engineered 22 domain features (cyclical seasonality, diurnal temperature range, anomalies, and pediatric age groups) and validated via leak-free 5-fold `StratifiedGroupKFold` grouped by `location`. A tuned **Random Forest Classifier** (`n_estimators=400`, `max_depth=4`, `max_features='log2'`) achieved **ROC-AUC = 0.7810** and **PR-AUC = 0.8524**, with SHAP identifying under-five vulnerability and elevated minimum temperatures as top drivers.
* **Files:** `Capstone_DSA_2.0.ipynb`, `uganda_mortality_predictions.xls`

---

## Project 2: Predicting 30-Day Customer Liquidity Stress
* **Objective:** Predict whether mobile money customers (`N = 40,000` training records) will experience liquidity stress over the next 30 days (`liquidity_stress_next_30d`), and score `30,000` unseen test customers.
* **Methodology & Key Results:** Audited 183 behavioral and balance features (`85%` vs. `15%` class imbalance), engineered 20 trajectory and cash-burn features (`202` total predictors), and evaluated models via 5-fold `StratifiedKFold`. A tuned **LightGBM Classifier** (`n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=100`) achieved **ROC-AUC = 0.8842** and **PR-AUC = 0.6376**, with SHAP confirming that `bal_drop_m3_m4` and `bal_ratio_m1_m6` are the top predictors.
* **Files:** `Financial_Stress_Capstone.ipynb`, `financial_stress_predictions.xls`
