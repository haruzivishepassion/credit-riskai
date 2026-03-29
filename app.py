import streamlit as st
import joblib
import pandas as pd
import google.generativeai as genai
from datetime import datetime

# --- 1. MODEL & AI CONFIGURATION ---
try:
    model = joblib.load('credit_model.pkl')
except:
    st.error("Model file (credit_model.pkl) not found! Please ensure it is uploaded to GitHub.")

# YOUR ACTUAL API KEY
genai.configure(api_key="AIzaSyDLgOGqGxNrATYwTn3sXtJLQuh1ecE1TN0")
chat_model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="AI Credit Risk Agent", page_icon="🏦", layout="wide")

# --- 2. THE CHAT SIDEBAR ---
with st.sidebar:
    st.title("🤖 Risk Advisor Chat")
    st.info("Ask me about credit risk, CIBIL scores, or the model's logic!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            context = "You are a professional credit risk analyst. Keep answers concise and expert."
            try:
                response = chat_model.generate_content(f"{context} User asked: {prompt}")
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error("The AI Chat is currently busy. Please try again in a moment.")

    st.divider()
    st.header("Applicant Profile")
    applicant_name = st.text_input("Applicant Name", "John Doe")
    cibil = st.slider("CIBIL Score", 300, 900, 750)
    income = st.number_input("Annual Income ($)", value=5000000)
    loan = st.number_input("Loan Amount ($)", value=10000000)
    term = st.slider("Loan Term (Months)", 2, 20, 12)
    edu = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    emp = st.selectbox("Self Employed?", ["No", "Yes"])

# --- 3. THE MAIN DASHBOARD ---
st.title("🏦 AI Credit Risk Agent")
st.markdown("### Advanced Loan Risk Analysis & Reporting")
st.divider()

if st.button("Run Detailed Risk Assessment", use_container_width=True):
    # Prepare data for prediction
    input_df = pd.DataFrame([[
        2, 1 if edu == "Graduate" else 0, 1 if emp == "Yes" else 0,
        income, loan, term, cibil
    ]], columns=['no_of_dependents', 'education_encoded', 'self_employed_encoded', 
                 'income_annum', 'loan_amount', 'loan_term', 'cibil_score'])
    
    prob = model.predict_proba(input_df)[0][1]
    decision = "APPROVED" if prob > 0.5 else "REJECTED"
    
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.subheader("Final Decision")
        if decision == "APPROVED":
            st.success(f"### {decision}")
            st.metric("Approval Confidence", f"{prob*100:.1f}%")
        else:
            st.error(f"### {decision}")
            st.metric("Risk Level", f"{(1-prob)*100:.1f}%")

    with res_col2:
        st.subheader("AI Insight Breakdown")
        reasons = []
        if cibil < 600:
            reasons.append("- High Risk: Low CIBIL Score")
        if loan > (income * 2):
            reasons.append("- Risk: High Debt-to-Income Ratio")
        
        if not reasons:
            st.info("✅ Financial indicators are within standard safety thresholds.")
        else:
            for r in reasons:
                st.warning(r)

    # CREATE REPORT
    report_text = f"APPLICANT: {applicant_name}\nDECISION: {decision}\nCONFIDENCE: {prob*100:.1f}%\nDATE: {datetime.now()}"
    st.divider()
    st.download_button(
        label="📥 Download Assessment Report",
        data=report_text,
        file_name=f"Loan_Report_{applicant_name}.txt",
        use_container_width=True
    )
