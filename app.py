import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Load trained model + column list saved from the notebook
# ---------------------------------------------------------
model = joblib.load("best_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Startup Success Predictor", page_icon="🚀", layout="centered")
st.title("🚀 AI Startup Success Predictor")
st.write("Enter your startup's details below to estimate its 5-year survival probability.")

# ---------------------------------------------------------
# Inputs (mirrors the raw columns used in the notebook)
# ---------------------------------------------------------
st.subheader("Funding & Network")
funding_total_usd = st.number_input("Total Funding Raised (USD)", min_value=0, value=1_000_000, step=10_000)
funding_rounds = st.number_input("Number of Funding Rounds", min_value=0, value=2, step=1)
relationships = st.number_input("Number of Relationships / Investor Connections", min_value=0, value=5, step=1)
avg_participants = st.number_input("Avg. Participants per Round", min_value=0.0, value=2.0, step=0.5)

st.subheader("Timeline (in years since founding)")
age_first_funding_year = st.number_input("Age at First Funding (years)", value=1.0, step=0.5)
age_last_funding_year = st.number_input("Age at Last Funding (years)", value=3.0, step=0.5)
age_first_milestone_year = st.number_input("Age at First Milestone (years)", value=1.5, step=0.5)
age_last_milestone_year = st.number_input("Age at Last Milestone (years)", value=3.5, step=0.5)
milestones = st.number_input("Number of Milestones Achieved", min_value=0, value=2, step=1)

st.subheader("Funding Round Types")
col1, col2, col3 = st.columns(3)
with col1:
    has_VC = st.checkbox("Has VC Funding")
    has_angel = st.checkbox("Has Angel Funding")
with col2:
    has_roundA = st.checkbox("Has Round A")
    has_roundB = st.checkbox("Has Round B")
with col3:
    has_roundC = st.checkbox("Has Round C")
    has_roundD = st.checkbox("Has Round D")

is_top500 = st.checkbox("Ranked in Top 500 (industry ranking)")

st.subheader("Location & Category")
state_code = st.selectbox("State", ["CA", "NY", "MA", "TX", "WA", "other"])
category_code = st.selectbox(
    "Category",
    ["software", "web", "mobile", "enterprise", "advertising", "games_video", "ecommerce", "biotech", "consulting", "other"],
)

# ---------------------------------------------------------
# Feature engineering (same formulas as the notebook)
# ---------------------------------------------------------
funding_per_round = funding_total_usd / max(funding_rounds, 1)
total_round_types = sum([has_VC, has_angel, has_roundA, has_roundB, has_roundC, has_roundD])
funding_duration = age_last_funding_year - age_first_funding_year
milestone_duration = age_last_milestone_year - age_first_milestone_year
relationships_per_round = relationships / max(funding_rounds, 1)

# ---------------------------------------------------------
# Build the raw input row (before one-hot encoding)
# ---------------------------------------------------------
raw_row = {
    "funding_total_usd": funding_total_usd,
    "funding_rounds": funding_rounds,
    "relationships": relationships,
    "avg_participants": avg_participants,
    "age_first_funding_year": age_first_funding_year,
    "age_last_funding_year": age_last_funding_year,
    "age_first_milestone_year": age_first_milestone_year,
    "age_last_milestone_year": age_last_milestone_year,
    "milestones": milestones,
    "has_VC": int(has_VC),
    "has_angel": int(has_angel),
    "has_roundA": int(has_roundA),
    "has_roundB": int(has_roundB),
    "has_roundC": int(has_roundC),
    "has_roundD": int(has_roundD),
    "is_top500": int(is_top500),
    "funding_per_round": funding_per_round,
    "total_round_types": total_round_types,
    "funding_duration": funding_duration,
    "milestone_duration": milestone_duration,
    "relationships_per_round": relationships_per_round,
    "state_code": state_code,
    "category_code": category_code,
}

input_df = pd.DataFrame([raw_row])

# One-hot encode state_code / category_code exactly like the notebook did
input_encoded = pd.get_dummies(input_df, columns=["state_code", "category_code"], drop_first=True)

# Align to the exact columns/order the model was trained on.
# Any column the model expects but we didn't generate (e.g. a state/category
# that wasn't selected, or a raw column dropped during training) is filled with 0.
input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
if st.button("Predict Success"):
    proba = model.predict_proba(input_final)[0][1]
    prediction = model.predict(input_final)[0]

    st.subheader("Result")
    if prediction == 1:
        st.success(f"✅ Predicted: Likely to Succeed (Acquired/Survives)")
    else:
        st.error(f"⚠️ Predicted: Likely to Fail (Closed)")

    st.metric("Probability of Success", f"{proba*100:.1f}%")
    st.progress(float(proba))
