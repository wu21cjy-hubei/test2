from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier, Pool


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "catboost_model.cbm"
SCALER_PATH = APP_DIR / "training_scaler.pkl"
META_PATH = APP_DIR / "public_metadata.json"

st.set_page_config(
    page_title="Postoperative Spinal Infection Risk Calculator",
    page_icon="🩺",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px;}
h1 {font-size: 2.25rem !important; line-height: 1.18 !important; color: #202638; margin-bottom: .35rem;}
h2, h3, h4 {color: #202638;}
.subtitle {font-size: 1.05rem; color: #667085; margin-bottom: 1.5rem;}
.input-note {font-size: .98rem; color: #667085; margin-bottom: .85rem;}
.result-card {border: 1px solid #EAECF0; border-radius: 18px; padding: 24px 26px; background: #FFFFFF; box-shadow: 0 1px 5px rgba(16,24,40,.06); margin-bottom: 16px;}
.prob {font-size: 4rem; font-weight: 820; color: #202638; line-height: 1.0; margin: .1rem 0 .3rem 0;}
.prob-label {font-size: 1.02rem; color: #667085;}
.context-box {background: #F9FAFB; border: 1px solid #EAECF0; border-radius: 14px; padding: 16px 18px; color: #344054; font-size: 1.02rem; line-height: 1.6;}
.badge-low {background: #ECFDF3; border: 1px solid #ABEFC6; color: #067647; border-radius: 16px; padding: 18px 20px; font-size: 1.28rem; font-weight: 780;}
.badge-mild {background: #F0F9FF; border: 1px solid #B9E6FE; color: #026AA2; border-radius: 16px; padding: 18px 20px; font-size: 1.28rem; font-weight: 780;}
.badge-concerning {background: #FFFAEB; border: 1px solid #FEDF89; color: #B54708; border-radius: 16px; padding: 18px 20px; font-size: 1.28rem; font-weight: 780;}
.badge-high {background: #FEF3F2; border: 1px solid #FDA29B; color: #B42318; border-radius: 16px; padding: 18px 20px; font-size: 1.28rem; font-weight: 780;}
.clinical-text {font-size: 1.04rem; color: #344054; line-height: 1.68; margin-top: .75rem;}
.factor-card {border: 1px solid #EAECF0; border-radius: 14px; padding: 17px 19px; background: #FFFFFF; height: 100%; box-shadow: 0 1px 4px rgba(16,24,40,.04);}
.factor-title {font-size: 1.08rem; font-weight: 780; color: #202638; margin-bottom: .55rem;}
.factor-item {font-size: 1.0rem; color: #344054; margin: .45rem 0; line-height: 1.45;}
.footer-note {font-size: .92rem; color: #667085; border-top: 1px solid #EAECF0; margin-top: 2rem; padding-top: 1rem;}
.stButton > button {border-radius: 10px; padding: .65rem 1.25rem; font-weight: 750;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    scaler = joblib.load(SCALER_PATH)
    return model, scaler, meta


def format_pct(p):
    return f"{100 * p:.1f}%"


def risk_category(prob):
    """Clinician-facing probability bands. These are not diagnostic labels."""
    if prob < 0.15:
        return (
            "Low estimated probability",
            "badge-low",
            "The estimate is low and close to or below the average rate observed in the development cohort. Continue routine postoperative monitoring and reassess if new clinical signs appear.",
        )
    if prob < 0.30:
        return (
            "Mildly increased probability",
            "badge-mild",
            "The estimate is above the cohort average but not strongly suspicious by itself. Review the wound, symptoms, and inflammatory trend, and repeat assessment if the clinical picture changes.",
        )
    if prob < 0.60:
        return (
            "Clinically concerning probability",
            "badge-concerning",
            "This is not a confirmed diagnosis, but the estimated probability is clearly higher than the average postoperative infection rate in the development cohort. Closer reassessment, wound evaluation, repeat inflammatory markers, and microbiological or imaging work-up should be considered when clinically indicated.",
        )
    return (
        "High estimated probability",
        "badge-high",
        "The estimated probability is high. Prompt infection-focused reassessment is recommended, including wound review, repeat laboratory testing, and microbiological or imaging evaluation when appropriate.",
    )


def plain_probability_explanation(prob, baseline):
    n_out_of_100 = int(round(prob * 100))
    baseline_out_of_100 = int(round(baseline * 100))
    ratio = prob / baseline if baseline > 0 else np.nan
    if np.isfinite(ratio):
        ratio_text = f"about {ratio:.1f} times"
    else:
        ratio_text = "higher than"
    return (
        f"In plain language, this result means that among patients with a similar early postoperative laboratory pattern, "
        f"approximately {n_out_of_100} out of 100 would be estimated as having postoperative infection. "
        f"For context, the average infection rate in the development cohort was about {baseline_out_of_100} out of 100. "
        f"Therefore, this is {ratio_text} the cohort average, so it should be treated as clinically concerning rather than ignored as 'only around half'."
    )


def predict_and_explain(model, scaler, meta, raw_values):
    features = meta["features"]
    x_raw = pd.DataFrame([raw_values], columns=features)
    x_scaled = scaler.transform(x_raw)
    prob = float(model.predict_proba(x_scaled)[0, 1])
    pool = Pool(x_scaled, feature_names=features)
    shap_values = model.get_feature_importance(pool, type="ShapValues")[0]
    return prob, shap_values[:-1]


def ranked_factors(meta, raw_values, contributions, positive=True, top_n=3):
    features = meta["features"]
    info = meta["feature_info"]
    rows = []
    for i, f in enumerate(features):
        value = raw_values[f]
        unit = info[f].get("unit", "")
        label = info[f].get("short", info[f].get("display", f))
        rows.append({"feature": f, "label": label, "value": value, "unit": unit, "effect": float(contributions[i])})
    if positive:
        rows = [r for r in rows if r["effect"] > 0]
    else:
        rows = [r for r in rows if r["effect"] < 0]
    rows = sorted(rows, key=lambda r: abs(r["effect"]), reverse=True)[:top_n]
    return rows


def factor_sentence(row):
    value = row["value"]
    unit = row["unit"]
    value_text = f"{value:.3g} {unit}" if unit else f"{value:.3g}"
    label = row["label"]
    return f"{label}: {value_text}"


model, scaler, meta = load_artifacts()
features = meta["features"]
feature_info = meta["feature_info"]
stats = meta["stats"]
baseline = float(meta.get("baseline_risk", 0.1621))

st.title(meta["app_title"])
st.markdown(f'<div class="subtitle">{meta["subtitle"]}</div>', unsafe_allow_html=True)

st.markdown("### Patient laboratory values")
st.markdown('<div class="input-note">Enter original clinical values. For a change variable, enter the later value minus the earlier value, using the same unit as the laboratory report.</div>', unsafe_allow_html=True)

raw_values = {}
col1, col2 = st.columns(2, gap="large")

for idx, f in enumerate(features):
    info = feature_info[f]
    s = stats[f]
    iqr = abs(float(s["q3"]) - float(s["q1"]))
    step = iqr / 100 if iqr > 0 else max(abs(float(s["max"]) - float(s["min"])) / 100, 0.01)
    step = max(step, 0.01)
    target_col = col1 if idx % 2 == 0 else col2
    with target_col:
        raw_values[f] = st.number_input(
            label=info.get("display", f),
            value=float(s["median"]),
            step=float(step),
            format="%.4f",
            help=f"Unit: {info.get('unit', '')}. {info.get('clinical_meaning', '')}",
        )

calculate = st.button("Calculate infection estimate", type="primary", use_container_width=True)

if calculate:
    prob, contributions = predict_and_explain(model, scaler, meta, raw_values)
    category, css_class, clinical_text = risk_category(prob)

    st.markdown("---")
    st.markdown("## Result")
    left, right = st.columns([1, 1.08], gap="large")

    with left:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="prob-label">Estimated probability of postoperative infection</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prob">{format_pct(prob)}</div>', unsafe_allow_html=True)
        st.progress(min(max(prob, 0.0), 1.0))
        st.markdown(f'<div class="context-box"><b>How to read this:</b><br>{plain_probability_explanation(prob, baseline)}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown(f'<div class="{css_class}">{category}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="clinical-text"><b>Clinical interpretation:</b> {clinical_text}</div>', unsafe_allow_html=True)

    inc = ranked_factors(meta, raw_values, contributions, positive=True, top_n=3)
    dec = ranked_factors(meta, raw_values, contributions, positive=False, top_n=3)

    f1, f2 = st.columns(2, gap="large")
    with f1:
        st.markdown('<div class="factor-card">', unsafe_allow_html=True)
        st.markdown('<div class="factor-title">Findings that increased this estimate</div>', unsafe_allow_html=True)
        if inc:
            for row in inc:
                st.markdown(f'<div class="factor-item">- {factor_sentence(row)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="factor-item">- No major risk-raising finding was identified from the entered values.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="factor-card">', unsafe_allow_html=True)
        st.markdown('<div class="factor-title">Findings that lowered this estimate</div>', unsafe_allow_html=True)
        if dec:
            for row in dec:
                st.markdown(f'<div class="factor-item">- {factor_sentence(row)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="factor-item">- No major risk-lowering finding was identified from the entered values.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Enter values and click **Calculate infection estimate**.")

st.markdown(f'<div class="footer-note">{meta["disclaimer"]}</div>', unsafe_allow_html=True)
