import streamlit as st
import joblib
import pandas as pd
import google.generativeai as genai
from datetime import datetime

# --- 1. CONFIG ---
try:
    model = joblib.load('credit_model.pkl')
except:
    st.error("Model file not found!")

genai.configure(api_key="AIzaSyDLgOGqGxNrATYwTn3sXtJLQuh1ecE1TN0")
chat_model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="AI Credit Risk Agent", layout="wide")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🤖 Risk Advisor")
    if prompt := st.chat_input("Ask about risk..."):
        with st.chat_message("assistant"):
            response = chat_model.generate_content(f"Context: Credit Risk Analyst. Answer: {prompt}")
            st.markdown(response.text)

    st.header("Applicant Profile")
    applicant_name = st.text_input("Applicant Name", "John Doe")
    cibil = st.slider("CIBIL Score", 300, 900, 750)
    income = st.number_input("Annual Income ($)", value=5000000)
    loan = st.number_input("Loan Amount ($)", value=10000000)
    term = st.slider("Loan Term (Months)", 2, 20, 12)
    edu = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    emp = st.selectbox("Self Employed?", ["No", "Yes"])

# --- 3. MAIN LOGIC ---
st.title("🏦 AI Credit Risk Agent")
st.markdown("### Strict Financial Assessment Mode")

if st.button("Run Assessment", use_container_width=True):
    input_df = pd.DataFrame([[
        2, 1 if edu == "Graduate" else 0, 1 if emp == "Yes" else 0,
        income, loan, term, cibil
    ]], columns=['no_of_dependents', 'education_encoded', 'self_employed_encoded', 
                 'income_annum', 'loan_amount', 'loan_term', 'cibil_score'])
    
    prob = model.predict_proba(input_df)[0][1]
    
    # --- NEW: THE "STRICT" OVERRIDE ---
    # If loan is greater than 100% of annual income for a short term, 
    # we manually lower the probability score.
    if loan > income:
        prob = prob * 0.5  # Heavy penalty for high debt-to-income
        st.warning("⚠️ **Alert:** Loan amount exceeds annual income. High probability of repayment distress.")

    decision = "APPROVED" if prob > 0.5 else "REJECTED"
    
    col1, col2 = st.columns(2)
    with col1:
        if decision == "APPROVED":
            st.success(f"### {decision}")
            st.metric("Final Confidence Score", f"{prob*100:.1f}%")
        else:
            st.error(f"### {decision}")
            st.metric("Adjusted Risk Score", f"{(1-prob)*100:.1f}%")

    with col2:
        st.subheader("Why this decision?")
        if loan > income:
            st.write("❌ **Debt-to-Income Crisis:** The applicant is trying to borrow more than they earn in a year. Even with a high CIBIL, the cash flow is insufficient.")
        if cibil < 600:
            st.write("❌ **Historical Default Risk:** Past behavior indicates poor repayment reliability.")
        if prob > 0.7:
            st.write("✅ **Stable Profile:** Income sufficiently covers the debt obligation.")

    # DOWNLOAD REPORT
    report = f"REPORT FOR {applicant_name}\nDECISION: {decision}\nCIBIL: {cibil}\nDTI Ratio: {round(loan/income, 2)}"
    st.download_button("📥 Download Final Audit Report", data=report, file_name=f"Loan_{applicant_name}.txt")
