import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import re

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model      = joblib.load("model/best_model.pkl")
    tfidf      = joblib.load("model/tfidf_vectorizer.pkl")
    feat_names = joblib.load("model/structured_features.pkl")
    return model, tfidf, feat_names

model, tfidf, STRUCTURED_FEATURES = load_artefacts()

# ── Red flag keywords ─────────────────────────────────────────────────────────
RED_FLAGS = [
    "urgent hiring", "immediate start", "no experience needed",
    "no experience required", "work from home", "earn money fast",
    "guaranteed income", "unlimited earning", "wire transfer",
    "send personal details", "investment required", "multi-level",
    "be your own boss", "no degree required", "no cv needed",
    "earn up to", "make money online", "weekly pay guaranteed",
    "apply now limited", "100% remote", "data entry",
]

def find_red_flags(text):
    found = []
    text_lower = text.lower()
    for flag in RED_FLAGS:
        if flag in text_lower:
            found.append(flag)
    return found

def count_red_flags(text):
    return len(find_red_flags(text))

# ── Feature engineering (mirrors Notebook 1) ─────────────────────────────────
def build_features(title, company_profile, description, requirements,
                   benefits, employment_type, required_experience,
                   required_education, telecommuting, has_logo, has_questions):

    text = f"{title} {company_profile} {description} {requirements} {benefits}"

    # Structured feature dict (same order as STRUCTURED_FEATURES)
    row = {
        "telecommuting":              int(telecommuting),
        "has_company_logo":           int(has_logo),
        "has_questions":              int(has_questions),
        "missing_salary":             1,   # user didn't provide salary
        "missing_company_profile":    int(company_profile.strip() == ""),
        "missing_requirements":       int(requirements.strip() == ""),
        "missing_benefits":           int(benefits.strip() == ""),
        "missing_location":           1,   # not collected in UI
        "desc_len":                   len(description),
        "title_len":                  len(title),
        "text_len":                   len(text),
        "red_flag_count":             count_red_flags(text),
        # Encoded categoricals — use simple ordinal map for UI
        "employment_type_enc":        {"Full-time":0,"Part-time":1,"Contract":2,
                                       "Temporary":3,"Other":4}.get(employment_type, 4),
        "required_experience_enc":    {"Not Applicable":0,"Internship":1,"Entry level":2,
                                       "Mid-Senior level":3,"Director":4,"Executive":5}.get(required_experience, 0),
        "required_education_enc":     {"Unspecified":0,"High School":1,"Some College":2,
                                       "Bachelor's Degree":3,"Master's Degree":4,"Doctorate":5}.get(required_education, 0),
        "industry_enc":               0,  # unknown from UI → 0
        "function_enc":               0,
    }

    # Keep only columns that were in training
    struct_values = [row.get(f, 0) for f in STRUCTURED_FEATURES]
    X_struct = csr_matrix(np.array(struct_values).reshape(1, -1))

    X_text_tfidf = tfidf.transform([text])
    X_final = hstack([X_text_tfidf, X_struct])
    return X_final, text

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 Fake Job Posting Detector")
st.markdown(
    "Paste a job posting below and the model will predict whether it's **Real** or **Fake**, "
    "along with a confidence score and red flag analysis."
)
st.divider()

# Instructions
with st.expander("ℹ️ How to use this tool"):
    st.markdown("""
    1. Go to any job posting on LinkedIn, Indeed, or any job board
    2. Copy the **job title**, **description**, and any other available details
    3. Paste them into the fields below
    4. Hit **Analyse Job Posting**
    
    The model was trained on the EMSCAD dataset (17,880 postings).  
    It uses TF-IDF text features + structured signals to detect scams.
    """)

# Input form
st.subheader("Job Posting Details")

col1, col2 = st.columns(2)
with col1:
    title = st.text_input("Job Title *", placeholder="e.g. Data Analyst – Remote")
with col2:
    employment_type = st.selectbox(
        "Employment Type",
        ["Full-time", "Part-time", "Contract", "Temporary", "Other"]
    )

col3, col4 = st.columns(2)
with col3:
    required_experience = st.selectbox(
        "Required Experience",
        ["Not Applicable","Internship","Entry level","Mid-Senior level","Director","Executive"]
    )
with col4:
    required_education = st.selectbox(
        "Required Education",
        ["Unspecified","High School","Some College","Bachelor's Degree","Master's Degree","Doctorate"]
    )

company_profile = st.text_area(
    "Company Description (if available)",
    placeholder="Paste what the job says about the company...",
    height=100
)

description = st.text_area(
    "Job Description *",
    placeholder="Paste the full job description here...",
    height=200
)

requirements = st.text_area(
    "Requirements / Qualifications (if available)",
    placeholder="Paste the requirements section...",
    height=100
)

benefits = st.text_area(
    "Benefits (if available)",
    placeholder="Paste the benefits section...",
    height=80
)

st.markdown("**Additional Signals**")
col5, col6, col7 = st.columns(3)
with col5:
    telecommuting = st.checkbox("Remote / Telecommuting")
with col6:
    has_logo = st.checkbox("Company Logo Present")
with col7:
    has_questions = st.checkbox("Screening Questions Listed")

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Analyse Job Posting", use_container_width=True, type="primary"):

    if not title.strip() or not description.strip():
        st.warning("Please fill in at least the **Job Title** and **Job Description**.")
    else:
        with st.spinner("Analysing..."):
            X_final, combined_text = build_features(
                title, company_profile, description, requirements,
                benefits, employment_type, required_experience,
                required_education, telecommuting, has_logo, has_questions
            )
            prediction = model.predict(X_final)[0]
            proba      = model.predict_proba(X_final)[0]
            confidence = proba[prediction] * 100
            fake_prob  = proba[1] * 100

        # ── Verdict ──────────────────────────────────────────────────────────
        st.subheader("Result")

        if prediction == 1:
            st.error(f"### ⚠️ Likely FAKE  —  {confidence:.1f}% confidence")
        else:
            if fake_prob > 30:
                st.warning(f"### 🟡 Likely REAL but suspicious  —  {confidence:.1f}% confidence")
            else:
                st.success(f"### ✅ Likely REAL  —  {confidence:.1f}% confidence")

        # Confidence bar
        st.markdown("**Fake probability score**")
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.progress(int(fake_prob))
        with col_b:
            st.markdown(f"**{fake_prob:.1f}%**")

        st.divider()

        # ── Red flag analysis ─────────────────────────────────────────────────
        st.subheader("🚩 Red Flag Analysis")
        flags_found = find_red_flags(combined_text)

        if flags_found:
            st.markdown(f"**{len(flags_found)} red flag(s) detected:**")
            cols = st.columns(min(len(flags_found), 3))
            for i, flag in enumerate(flags_found):
                cols[i % 3].error(f"🚩 {flag}")
        else:
            st.success("No common red flag phrases detected in this posting.")

        # ── Feature signals ───────────────────────────────────────────────────
        st.divider()
        st.subheader("📊 Signal Breakdown")

        signals = {
            "Company logo present":       "✅ Yes" if has_logo else "⚠️ No",
            "Screening questions listed":  "✅ Yes" if has_questions else "⚠️ No",
            "Remote / telecommuting":      "🌐 Yes" if telecommuting else "🏢 No",
            "Company profile filled":      "✅ Yes" if company_profile.strip() else "⚠️ Missing",
            "Requirements section filled": "✅ Yes" if requirements.strip() else "⚠️ Missing",
            "Benefits section filled":     "✅ Yes" if benefits.strip() else "⚠️ Missing",
            "Description length":          f"{len(description)} chars {'✅' if len(description) > 300 else '⚠️ Short'}",
            "Red flag keyword count":      f"{len(flags_found)} {'🚩' * min(len(flags_found), 5)}",
        }

        for signal, value in signals.items():
            col_s, col_v = st.columns([2, 3])
            col_s.markdown(f"**{signal}**")
            col_v.markdown(value)

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.divider()
        st.caption(
            "⚠️ This tool is a machine learning classifier, not a legal authority. "
            "Always verify job postings independently before sharing personal information "
            "or making any payment."
        )
