# Predicting Climate-Sensitive Mortality in Uganda

## Project Overview
This repository contains an end-to-end machine learning pipeline developed to identify whether mortality records in Uganda (`N = 3,146` training records) are climate-sensitive (`is_climate_sensitive`), and generalize predictions to `1,030` records across unseen geographic locations.

## Key Methodology & Results
* **Spatial Generalization & Leak-Free Validation:** Evaluated using 5-fold `StratifiedGroupKFold` grouped by `location` after removing village identifiers (`location`, `latitude`, `longitude`) to prevent spatial overfitting.
* **Feature Engineering:** Engineered 22 domain-grounded features capturing cyclical seasonality (`month_sin`, `month_cos`), diurnal temperature range, temperature anomalies, warm-wet interactions, and clinical age brackets.
* **Best Model:** A hyperparameter-tuned **Random Forest Classifier** (`n_estimators=400`, `max_depth=4`, `max_features='log2'`) achieved an out-of-fold **ROC-AUC of 0.7810** and **PR-AUC of 0.8524**.
* **Explainability:** SHAP summary analysis confirmed that under-five pediatric vulnerability (`age <= 4`) and elevated minimum nighttime temperatures (`min_temperature`) are the primary drivers of climate-sensitive mortality.

## Repository Contents
* `Capstone_DSA_2.0.ipynb` — Complete Jupyter Notebook containing EDA, visualizations, preprocessing pipelines, model benchmarking, hyperparameter optimization, SHAP interpretation, and policy recommendations.
* `uganda_mortality_predictions.xls` — Final test set predictions (`ID`, `predicted_probability`, and `is_climate_sensitive`).
