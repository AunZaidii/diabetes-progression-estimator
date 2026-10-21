"""
Train the two linear regression models used by the Streamlit app (app.py)
Same data split as models.py (the original assignment), but features are kept in real units
(years, kg/m2, mmHg, mg/dL) so users can type in normal values.
"""
import os
import numpy as np
import joblib
import sklearn
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "diabetes_models.joblib")

# ---------------------------------------------------------------
# STEP 1: LOAD THE DATASET (unscaled = real units)
# ---------------------------------------------------------------
diabetes = datasets.load_diabetes(as_frame=True, scaled=False)
X, y = diabetes.data, diabetes.target

# ---------------------------------------------------------------
# STEP 2: SAME TRAIN / TEST SPLIT AS models.py
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------
# STEP 3: TRAIN BOTH MODELS
# ---------------------------------------------------------------
m_bmi = linear_model.LinearRegression().fit(X_train[["bmi"]], y_train)
m_all = linear_model.LinearRegression().fit(X_train, y_train)
p_bmi = m_bmi.predict(X_test[["bmi"]])
p_all = m_all.predict(X_test)
p_base = np.full(len(y_test), y_train.mean())  # always guess the average

# ---------------------------------------------------------------
# STEP 4: EVALUATE ON THE TEST SET
# ---------------------------------------------------------------
metrics = {}
for name, pred in [("baseline", p_base), ("bmi", p_bmi), ("all", p_all)]:
    metrics[name] = {"mae": mean_absolute_error(y_test, pred),
                     "r2": r2_score(y_test, pred)}
    print("%-9s Test MAE: %.2f | Test R2: %.4f" % (name, metrics[name]["mae"], metrics[name]["r2"]))

expected = {"baseline": 64.01, "bmi": 52.26, "all": 42.79}  # from results.csv
for name, mae in expected.items():
    if abs(round(metrics[name]["mae"], 2) - mae) > 0.01:
        print("WARNING: %s MAE does not match results.csv (%.2f)" % (name, mae))

# ---------------------------------------------------------------
# STEP 5: SAVE EVERYTHING THE APP NEEDS
# ---------------------------------------------------------------
bundle = {
    "model_bmi": m_bmi,
    "model_all": m_all,
    "features": list(X.columns),
    "feature_stats": X.describe().T[["min", "mean", "max"]].to_dict("index"),
    "target": {"min": float(y.min()), "max": float(y.max()), "mean": float(y.mean()),
               "p33": float(np.percentile(y, 100 / 3)), "p67": float(np.percentile(y, 200 / 3))},
    "train_mean": float(y_train.mean()),
    "metrics": metrics,
    # 10th / 90th percentile of test errors (actual - predicted) -> "likely range"
    "resid_q": {"bmi": tuple(np.percentile(y_test - p_bmi, [10, 90])),
                "all": tuple(np.percentile(y_test - p_all, [10, 90]))},
    "sklearn_version": sklearn.__version__,
}
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(bundle, MODEL_PATH)
print("Saved models to", MODEL_PATH)
