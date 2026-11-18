# Predicting Climate-Sensitive Mortality in Uganda

### Business & Public Health Objective
Predict whether mortality records in Uganda (`N = 3,146` training records) are climate-sensitive (`is_climate_sensitive`), and generalize predictions to `1,030` records across completely unseen villages and geographic locations.

### Methodology & Spatial Feature Engineering
* **Leak-Free Spatial Validation:** Implemented 5-fold `StratifiedGroupKFold` grouped strictly by `location` so no village ever overlapped between training and validation folds.
* **Domain Feature Engineering:** Created 22 domain features capturing cyclical seasonality (`sin`/`cos` month transforms), diurnal temperature range (`tmax - tmin`), local temperature anomalies, and under-five pediatric vulnerability.
* **Model Selection & Tuning:** Evaluated Logistic Regression, Random Forest, and Gradient Boosting inside leak-free `scikit-learn` pipelines using `RandomizedSearchCV`.

### Key Results & SHAP Explainability
* **Top Model:** A regularized **Random Forest Classifier** (`n_estimators=400`, `max_depth=4`, `max_features='log2'`) achieved **ROC-AUC = 0.7810** and **PR-AUC = 0.8524** across unseen geographic locations.
* **Key Risk Drivers:** SHAP explainability revealed that **under-five pediatric age groups** and **elevated nighttime minimum temperatures (`tmin`)** are the strongest predictors of climate-sensitive mortality.

### Repository Contents
* `Capstone_DSA_2.0.ipynb` — Complete end-to-end modeling & spatial validation notebook
* `Capstone_DSA_2.0.html` — Rendered interactive HTML report
* `uganda_mortality_predictions.xls` — Out-of-sample risk probabilities & binary classifications (`N = 1,030`)
* `requirements.txt` — Python package dependencies
* *(Note: Capstone Project 2 files are also retained in this repository for LMS grading reference; see the dedicated repository at https://github.com/kolistic4u/customer-liquidity-stress-ml)*
