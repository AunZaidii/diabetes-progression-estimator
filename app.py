"""Streamlit app: estimate diabetes progression with the models from train.py (run: python -m streamlit run app.py)"""
import os
import numpy as np
import pandas as pd
import joblib
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "diabetes_models.joblib")
LOW, HIGH = 25, 346  # smallest / largest score in the study

# ---------------------------------------------------------------
# INPUT SETTINGS: (label, min, max, default, step, help) - ranges = dataset min/max
# ---------------------------------------------------------------
INPUTS = {
    "age": ("Age (years)", 19, 79, 49, 1, "Your age in years. Patients in the study were 19 to 79."),
    "bmi": ("Body Mass Index, BMI (kg/m²)", 18.0, 42.2, 26.4, 0.1, "Weight in kg divided by height in m squared. 18.5–25 is a healthy range."),
    "bp": ("Average blood pressure (mmHg)", 62, 133, 95, 1, "Mean arterial pressure, roughly (systolic + 2 × diastolic) / 3. For 120/80, this is about 93."),
    "s1": ("Total cholesterol (mg/dL)", 97, 301, 189, 1, "Total cholesterol in your blood. Under 200 is considered desirable."),
    "s2": ('LDL ("bad") cholesterol (mg/dL)', 41.6, 242.4, 115.4, 1.0, "Low-density lipoprotein. Under 100 is optimal."),
    "s3": ('HDL ("good") cholesterol (mg/dL)', 22, 99, 50, 1, "High-density lipoprotein. Higher is better; 60 or more is protective."),
    "tg": ("Triglycerides (mg/dL)", 26, 449, 104, 1, "Blood fat level. Under 150 is normal. The model uses its logarithm internally."),
    "s6": ("Blood sugar / glucose (mg/dL)", 58, 124, 91, 1, "Blood glucose level. 70–99 fasting is normal."),
}
GENDERS = {"Female": 1, "Male": 2}  # study codes; inferred from the data (group 1 has higher HDL, lower BP)
GENDER_HELP = ("The study recorded this as 1 or 2 without saying which is which. We assume 1 = female and "
               "2 = male, based on typical differences in the data. Switching it changes the result by about 23 points.")

def ask(key, widget):  # widget is st.slider or st.number_input
    label, lo, hi, default, step, help_text = INPUTS[key]
    return widget(label, lo, hi, default, step, help=help_text)

def split_panels(path):
    """Cut a two-panel figure at the blank gap in the middle (so the panels can stack on phones)."""
    img = Image.open(path)
    gray = np.asarray(img.convert("L"))
    w = gray.shape[1]
    blank = [x for x in range(int(w * 0.4), int(w * 0.6)) if (gray[:, x] < 250).sum() == 0]
    if not blank:  # no clean gap (e.g. a single chart): keep the image whole
        return [img]
    cut = (blank[0] + blank[-1]) // 2
    return [img.crop((0, 0, cut, img.height)), img.crop((cut, 0, w, img.height))]

# Small-screen tweaks (phones / small tablets): less padding, smaller title and numbers
MOBILE_CSS = """
<style>
@media (max-width: 640px) {
  .block-container { padding: 2.5rem 1rem 2rem; }
  h1 { font-size: 1.7rem !important; }
  [data-testid="stMetricValue"] { font-size: 1.8rem; }
  button[data-baseweb="tab"] p { font-size: 1rem; }
}
</style>
"""

# ---------------------------------------------------------------
# STEP 1: LOAD THE TRAINED MODELS
# ---------------------------------------------------------------
@st.cache_resource
def load_bundle(file_time):  # file_time = when the file was saved, so re-running train.py reloads it
    return joblib.load(MODEL_PATH)

st.set_page_config(page_title="Diabetes Progression Estimator", layout="centered")
st.markdown(MOBILE_CSS, unsafe_allow_html=True)
st.title("Diabetes Progression Estimator")
st.caption("Estimates how much diabetes may progress over one year, using a linear regression "
           "model trained on 442 patients (Efron et al., 2004).")
st.warning("Educational project only, not medical advice. This model is approximate (it explains about "
           "45% of the variation between patients) and must not be used for diagnosis or treatment. "
           "Talk to a doctor about your health.")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Train the models first by running this in the project folder:")
    st.code("python train.py")
    st.stop()
try:
    bundle = load_bundle(os.path.getmtime(MODEL_PATH))
except Exception as e:
    st.error(f"Could not load the model file ({type(e).__name__}: {e}). Re-run python train.py to rebuild it.")
    st.stop()
if bundle["sklearn_version"] != sklearn.__version__:
    st.warning(f"Models were trained with scikit-learn {bundle['sklearn_version']} but you have "
               f"{sklearn.__version__}. Results may differ; re-run python train.py to be safe.")

# ---------------------------------------------------------------
# TWO TABS: the predictor, and the results from models.py
# ---------------------------------------------------------------
tab_pred, tab_cmp = st.tabs(["Predictor", "Model comparison"])

with tab_pred:
    # ---------------------------------------------------------------
    # STEP 2: COLLECT THE INPUTS
    # ---------------------------------------------------------------
    st.subheader("About you")
    st.caption("Each input is limited to the range of values seen in the study.")
    col1, col2 = st.columns(2)
    with col1:
        age = ask("age", st.slider)
        gender = st.radio("Gender", list(GENDERS), index=0, horizontal=True, help=GENDER_HELP)
        sex = GENDERS[gender]  # the model still uses the study code
    with col2:
        bmi, bp = ask("bmi", st.slider), ask("bp", st.slider)

    with st.expander("Blood test results (optional)"):
        st.caption("Leave these at the study average if you don't know them. Values come from a standard "
                   "blood test (lipid panel and fasting glucose).")
        c1, c2 = st.columns(2)
        with c1:
            s1, s2, s3 = [ask(k, st.number_input) for k in ["s1", "s2", "s3"]]
        with c2:
            tg, s6 = [ask(k, st.number_input) for k in ["tg", "s6"]]
        s4 = float(np.clip(s1 / s3, 2.0, 9.09))
        st.caption(f"Cholesterol ratio (total/HDL) calculated for you: {s4:.2f}")

    row = {"age": age, "sex": sex, "bmi": bmi, "bp": bp, "s1": s1, "s2": s2,
           "s3": s3, "s4": s4, "s5": np.log(tg), "s6": s6}
    X_new = pd.DataFrame([row], columns=bundle["features"])

    # ---------------------------------------------------------------
    # STEP 3: PREDICT
    # ---------------------------------------------------------------
    raw_all = bundle["model_all"].predict(X_new)[0]
    raw_bmi = bundle["model_bmi"].predict(X_new[["bmi"]])[0]
    pred_all, pred_bmi = np.clip([raw_all, raw_bmi], LOW, HIGH)
    avg, t = bundle["train_mean"], bundle["target"]
    lo, hi = np.clip(raw_all + np.array(bundle["resid_q"]["all"]), LOW, HIGH)

    st.subheader("Your estimate")
    m1, m2, m3 = st.columns(3)
    m1.metric("Full model", f"{pred_all:.0f}", f"{pred_all - avg:+.0f} vs avg", delta_color="inverse",
              help="Uses all 10 inputs.")
    m2.metric("BMI only", f"{pred_bmi:.0f}", f"{pred_bmi - avg:+.0f} vs avg", delta_color="inverse",
              help="Uses only your BMI.")
    m3.metric("Average patient", f"{avg:.0f}")
    if raw_all != pred_all or raw_bmi != pred_bmi:
        st.caption(f"Note: a raw prediction fell outside the study range and was clipped to {LOW}–{HIGH}.")
    st.markdown(f"Likely range: **{lo:.0f} – {hi:.0f}** (about 8 in 10 test patients were inside a range this wide)")

    if pred_all < t["p33"]:
        st.success("Lower than most patients in the study (bottom third).")
    elif pred_all <= t["p67"]:
        st.info("Around the middle of the study group.")
    else:
        st.error("Higher than most patients in the study (top third).")

    # ---------------------------------------------------------------
    # STEP 4: SCALE CHART (where you sit between 25 and 346)
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 1.8))
    for a, b, color in [(LOW, t["p33"], "#c8e6c9"), (t["p33"], t["p67"], "#fff3c4"), (t["p67"], HIGH, "#ffcdd2")]:
        ax.axvspan(a, b, color=color)
    ax.barh(0, hi - lo, left=lo, height=0.35, color="steelblue", alpha=0.35, label="Likely range")
    ax.plot(pred_all, 0, "o", color="black", markersize=10, label="Full model")
    ax.plot(pred_bmi, 0, "o", markerfacecolor="none", markeredgecolor="black", markersize=10, label="BMI only")
    ax.axvline(avg, color="gray", linestyle="--", label="Average")
    ax.set(xlim=(LOW, HIGH), ylim=(-0.5, 0.5), yticks=[])
    ax.tick_params(labelsize=10)
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=10, frameon=False)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    word = "higher" if pred_all > avg else "lower"
    st.write(f"A score of {pred_all:.0f} means the model expects your disease progression to be {word} than the "
             f"average patient ({avg:.0f}). The score is a research measure from 25 (least progression) to 346 "
             "(most). It is not a blood sugar value or a probability.")

    # ---------------------------------------------------------------
    # STEP 5: EXPLANATIONS
    # ---------------------------------------------------------------
    with st.expander("How accurate is this model?"):
        names = {"baseline": "Always guess the average", "bmi": "BMI only", "all": "All features"}
        table = pd.DataFrame([{"Model": names[k], "Test MAE": round(v["mae"], 2), "Test R2": round(v["r2"], 2)}
                              for k, v in bundle["metrics"].items()])
        st.table(table)
        st.write(f"On patients it had never seen, the full model was off by about "
                 f"{bundle['metrics']['all']['mae']:.0f} points on average.")

    with st.expander("What do these inputs mean?"):
        st.markdown(
            "- **age**: age in years. **sex**: gender (1 = female, 2 = male, assumed). **bmi**: body mass index (kg/m²).\n"
            "- **bp**: average (mean arterial) blood pressure, mmHg.\n"
            '- **s1**: total cholesterol. **s2**: LDL ("bad") cholesterol. **s3**: HDL ("good") cholesterol.\n'
            "- **s4**: total / HDL cholesterol ratio (calculated). **s5**: log of triglycerides. **s6**: blood sugar (glucose).")


# ---------------------------------------------------------------
# STEP 6: MODEL COMPARISON TAB (results.csv and figures from models.py)
# ---------------------------------------------------------------
FIGURES = [
    ("fig1_exploration.png", "1. Exploring the data",
     "Left: how the disease progression scores are spread across the 442 patients. "
     "Right: patients with a higher BMI tend to have a higher score, so BMI is a useful feature."),
    ("fig2_models.png", "2. How well the models fit",
     "Left: Model 1 is a straight line through BMI vs score; many points sit far from the line. "
     "Right: Model 2 (all features) on the test set. Points closer to the dashed line are better predictions."),
    ("fig3_coefficients.png", "3. Which features matter in Model 2",
     "The weight the model gives each feature (features were scaled in models.py, so the bars can be compared). "
     "Long bars push the score up (right) or down (left); s5 (triglycerides) and bmi push it up the most. "
     "s1 and s2 (total and LDL cholesterol) have large opposite weights because they are strongly related, "
     "so they partly cancel each other out."),
]
with tab_cmp:
    st.subheader("Comparing the three models")
    st.caption("Results from models.py: trained on 80% of the patients, tested on the other 20%.")
    csv_path = os.path.join(BASE_DIR, "results.csv")
    if os.path.exists(csv_path):
        st.dataframe(pd.read_csv(csv_path), hide_index=True, width="stretch")
        st.markdown(
            "- **MSE / MAE**: how far predictions are from the truth (lower is better).\n"
            "- **R2**: share of the differences between patients the model explains (1 = perfect, 0 = no better than guessing the average).\n"
            "- Using **all 10 features** gives the lowest error and the highest R2. Train and test errors are close, "
            "so the model is not overfitting.")
    else:
        st.warning("results.csv not found. Run python models.py to create it.")

    for file, title, text in FIGURES:
        st.subheader(title)
        path = os.path.join(BASE_DIR, file)
        if os.path.exists(path):
            panels = split_panels(path)
            for col, panel in zip(st.columns(len(panels)), panels):
                col.image(panel, width="stretch")
            st.caption(text)
        else:
            st.warning(f"{file} not found. Run python models.py to create it.")

st.caption("Not medical advice. Data: scikit-learn diabetes dataset (Efron, Hastie, Johnstone & Tibshirani, 2004).")
