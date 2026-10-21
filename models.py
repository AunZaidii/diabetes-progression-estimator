"""
Supervised Learning Assignment: Training a model on the Diabetes dataset
Follows Lecture 2 structure: Dataset + Learning Algorithm (Model Class + Objective + Optimizer) -> Predictive Model
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)

# ---------------------------------------------------------------
# STEP 1: LOAD THE DATASET
# ---------------------------------------------------------------
diabetes = datasets.load_diabetes(as_frame=True)
X, y = diabetes.data, diabetes.target
print("=== STEP 1: Dataset ===")
print("Number of patients (n):", X.shape[0])
print("Number of features (d):", X.shape[1])
print("Features:", list(X.columns))
print("Target: disease progression score (continuous) -> REGRESSION problem")
print(X.head(), "\n")

# ---------------------------------------------------------------
# STEP 2: EXPLORE THE DATA
# ---------------------------------------------------------------
print("=== STEP 2: Exploration ===")
print(y.describe(), "\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y, bins=20, color="steelblue")
axes[0].set_xlabel("Diabetes risk score")
axes[0].set_ylabel("Number of patients")
axes[0].set_title("Target distribution")
axes[1].scatter(X["bmi"], y, s=10, color="black")
axes[1].set_xlabel("BMI (normalized)")
axes[1].set_ylabel("Diabetes risk")
axes[1].set_title("BMI vs risk")
plt.tight_layout()
plt.savefig("fig1_exploration.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# STEP 3: TRAIN / TEST SPLIT
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("=== STEP 3: Split ===")
print("Train size:", len(X_train), "| Test size:", len(X_test), "\n")

# ---------------------------------------------------------------
# STEP 4: MODEL 1 - SIMPLE LINEAR REGRESSION (BMI only, as in lecture)
#   Model class:  f(x) = theta1 * x + theta0
#   Objective:    MSE
#   Optimizer:    sklearn LinearRegression (least squares)
# ---------------------------------------------------------------
print("=== STEP 4: Model 1 - BMI only ===")
m1 = linear_model.LinearRegression()
m1.fit(X_train[["bmi"]], y_train)
p1_train = m1.predict(X_train[["bmi"]])
p1_test = m1.predict(X_test[["bmi"]])
print("Slope (theta1): %.3f" % m1.coef_[0])
print("Intercept (theta0): %.3f" % m1.intercept_, "\n")

# ---------------------------------------------------------------
# STEP 5: MODEL 2 - MULTIPLE LINEAR REGRESSION (all 10 features)
#   f(x) = theta0 + theta1*x1 + ... + theta10*x10
# ---------------------------------------------------------------
print("=== STEP 5: Model 2 - all features ===")
m2 = linear_model.LinearRegression()
m2.fit(X_train, y_train)
p2_train = m2.predict(X_train)
p2_test = m2.predict(X_test)
coefs = pd.Series(m2.coef_, index=X.columns).sort_values()
print("Intercept: %.3f" % m2.intercept_)
print("Coefficients:\n", coefs, "\n")

# ---------------------------------------------------------------
# STEP 6: EVALUATE (training vs test error) + random baseline
# ---------------------------------------------------------------
def report(name, ytr, ptr, yte, pte):
    return {
        "Model": name,
        "Train MSE": mean_squared_error(ytr, ptr),
        "Test MSE": mean_squared_error(yte, pte),
        "Test MAE": mean_absolute_error(yte, pte),
        "Test R2": r2_score(yte, pte),
    }

baseline_pred = np.full(len(y_test), y_train.mean())  # always predict the mean
rows = [
    report("Baseline (predict mean)", y_train, np.full(len(y_train), y_train.mean()),
           y_test, baseline_pred),
    report("Linear: BMI only", y_train, p1_train, y_test, p1_test),
    report("Linear: all features", y_train, p2_train, y_test, p2_test),
]
results = pd.DataFrame(rows).round(2)
print("=== STEP 6: Evaluation ===")
print(results.to_string(index=False), "\n")
results.to_csv("results.csv", index=False)

# ---------------------------------------------------------------
# STEP 7: VISUALIZE
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# (a) BMI-only fit
order = np.argsort(X_test["bmi"].values)
axes[0].scatter(X_train["bmi"], y_train, s=10, label="Train", alpha=0.5)
axes[0].scatter(X_test["bmi"], y_test, s=14, color="red", label="Test")
axes[0].plot(X_test["bmi"].values[order], p1_test[order], color="black",
             linewidth=2, label="Model (BMI only)")
axes[0].set_xlabel("BMI (normalized)")
axes[0].set_ylabel("Diabetes risk")
axes[0].set_title("Model 1: BMI only")
axes[0].legend()

# (b) predicted vs actual for the all-features model
axes[1].scatter(y_test, p2_test, color="red", s=15)
lims = [y.min(), y.max()]
axes[1].plot(lims, lims, "k--", label="Perfect prediction")
axes[1].set_xlabel("True risk")
axes[1].set_ylabel("Predicted risk")
axes[1].set_title("Model 2: all features (test set)")
axes[1].legend()
plt.tight_layout()
plt.savefig("fig2_models.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 4))
coefs.plot(kind="barh", color="steelblue")
plt.title("Model 2 coefficients")
plt.tight_layout()
plt.savefig("fig3_coefficients.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# STEP 8: PREDICT FOR NEW PATIENTS
# ---------------------------------------------------------------
print("=== STEP 8: Predictions on 3 patients ===")
new = X_test.iloc[:3]
out = pd.DataFrame({
    "True risk": y_test.iloc[:3].values,
    "Pred (BMI only)": m1.predict(new[["bmi"]]).round(1),
    "Pred (all features)": m2.predict(new).round(1),
})
print(out)