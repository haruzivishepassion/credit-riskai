import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Load the brain
try:
    model = joblib.load('credit_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")

st.set_page_config(page_title="AI Credit Risk Agent", page_icon="🏦", layout="wide")

st.title("🏦 AI Credit Risk Agent")
st.markdown("### Advanced Loan Risk Analysis & Reporting")
st.divider()

# Input Section
with st.sidebar:
    st.header("Applicant Profile")
    applicant_name = st.text_input("Applicant Full Name", "John Doe")
    cibil = st.slider("CIBIL Score", 300, 900, 750)
    income = st.number_input("Annual Income ($)", value=5000000)
    loan = st.number_input("Loan Amount ($)", value=10000000)
    term = st.slider("Loan Term (Months)", 2, 20, 12)
    edu = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    emp = st.selectbox("Self Employed?", ["No", "Yes"])

# Processing
if st.button("Run Detailed Risk Assessment", use_container_width=True):
    input_df = pd.DataFrame([[
        2, 1 if edu == "Graduate" else 0, 1 if emp == "Yes" else 0,
        income, loan, term, cibil
    ]], columns=['no_of_dependents', 'education_encoded', 'self_employed_encoded', 
                 'income_annum', 'loan_amount', 'loan_term', 'cibil_score'])
    
    prob = model.predict_proba(input_df)[0][1]
    decision = "APPROVED" if prob > 0.5 else "REJECTED"
    
    # UI Results
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
        st.subheader("Risk Breakdown")
        reasons = []
        if cibil < 600:
            reasons.append("- Low Credit Score (High Risk Factor)")
        if loan > (income * 2):
            reasons.append("- High Debt-to-Income Ratio")
        
        if not reasons:
            st.info("✅ All primary financial indicators are within safe limits.")
        else:
            for r in reasons:
                st.warning(r)

    # CREATE DOWNLOADABLE REPORT
    report_text = f"""
    LOAN RISK ASSESSMENT REPORT
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    -------------------------------------------
    APPLICANT: {applicant_name}
    DECISION: {decision}
    CONFIDENCE: {prob*100:.1f}%
    
    FINANCIAL DATA:
    - Annual Income: ${income:,}
    - Requested Loan: ${loan:,}
    - Loan Term: {term} Months
    - CIBIL Score: {cibil}
    
    SYSTEM NOTES:
    {chr(10).join(reasons) if reasons else "Profile meets standard stability requirements."}
    -------------------------------------------
    Official AI Agent Output
    """

    st.divider()
    st.download_button(
        label="📥 Download Assessment Report (.txt)",
        data=report_text,
        file_name=f"Loan_Report_{applicant_name}.txt",
        mime="text/plain",
        use_container_width=True
    )
