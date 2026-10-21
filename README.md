---
title: Diabetes Progression Estimator
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
short_description: Linear regression on the scikit-learn diabetes dataset
---

# Diabetes Progression – ML Assignment

- `models.py` – the original assignment (training, evaluation, figures, `results.csv`).
- `train.py` – retrains the same two linear regression models on the same split, using real units
  (years, kg/m², mmHg, mg/dL), and saves them to `models/diabetes_models.joblib`.
- `app.py` – a Streamlit page where you enter values and get an estimate from both models.

## Run (PowerShell, in this folder)

```
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

`train.py` should print test MAE 64.01 (baseline), 52.26 (BMI only) and 42.79 (all features), matching `results.csv`.

Educational project only, not medical advice.
