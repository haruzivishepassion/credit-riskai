import streamlit as st
import joblib
import pandas as pd
import google.generativeai as genai
import os

# --- 1. CONFIG & SETUP ---
st.set_page_config(page_title="AI Credit Risk Agent", layout="wide")

# Securely load the model
model_path = os.path.join(os.path.dirname(__file__), 'credit_model.pkl')

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Model file not found. Error: {e}")
    st.stop()

# Configure GenAI
try:
    # Ensure 'GOOGLE_API_KEY' is in your Streamlit Secrets
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    # Using 'gemini-pro' for maximum compatibility
    chat_model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    st.error(f"AI Configuration Error: {e}")
    chat_model = None

# --- 2. SIDEBAR (Inputs) ---
with st.sidebar:
    st.title("🤖 Risk Advisor Settings")
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

# Chat Interface (Integrated with error handling)
if prompt := st.chat_input("Ask the Risk Advisor about this profile..."):
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        if chat_model:
            with st.spinner("Analyzing..."):
                try:
                    full_prompt = f"Context: Credit Risk Analyst. Profile: {applicant_name}, CIBIL: {cibil}, Income: {income}, Loan: {loan}. Answer: {prompt}"
                    response = chat_model.generate_content(full_prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error("AI service is currently unreachable.")
                    st.write(f"Diagnostic info: {e}")
        else:
            st.error("AI Advisor is not configured.")

st.markdown("---")
st.markdown("### Strict Financial Assessment Mode")

if st.button("Run Assessment", use_container_width=True):
    input_df = pd.DataFrame([[
        2, 1 if edu == "Graduate" else 0, 1 if emp == "Yes" else 0,
        income, loan, term, cibil
    ]], columns=['no_of_dependents', 'education_encoded', 'self_employed_encoded', 
                 'income_annum', 'loan_amount', 'loan_term', 'cibil_score'])
    
    # Predict
    prob = model.predict_proba(input_df)[0][1]
    
    # Strict Override Logic
    if loan > income:
        prob = prob * 0.5
        st.warning("⚠️ **Alert:** Loan amount exceeds annual income.")

    decision = "APPROVED" if prob > 0.5 else "REJECTED"
    
    col1, col2 = st.columns(2)
    with col1:
        if decision == "APPROVED":
            st.success(f"### {decision}")
        else:
            st.error(f"### {decision}")
        st.metric("Probability of Repayment", f"{prob*100:.1f}%")

    with col2:
        st.subheader("Decision Insight")
        if loan > income:
            st.write("❌ Debt-to-Income Ratio is too high.")
        if cibil < 600:
            st.write("❌ Low CIBIL score indicates high risk.")
        if prob > 0.7:
            st.write("✅ Profile is within healthy risk parameters.")

    report = f"REPORT FOR {applicant_name}\nDECISION: {decision}\nCIBIL: {cibil}\nDTI Ratio: {round(loan/income, 2)}"
    st.download_button("📥 Download Final Audit Report", data=report, file_name=f"Loan_{applicant_name}.txt")
