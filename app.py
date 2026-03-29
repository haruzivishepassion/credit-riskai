import streamlit as st
import joblib
import pandas as pd

# 1. Load the "Brain" (The file you got from Colab)
try:
    model = joblib.load('credit_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")

# 2. Page Styling
st.set_page_config(page_title="AI Credit Risk Agent", page_icon="🏦")
st.title("🏦 AI Credit Risk Agent")
st.markdown("Enter applicant details below to get an instant AI-powered loan decision.")

# 3. Create Two Columns for Inputs
col1, col2 = st.columns(2)

with col1:
    cibil = st.slider("CIBIL Score", 300, 900, 750)
    income = st.number_input("Annual Income ($)", value=5000000)
    loan = st.number_input("Loan Amount ($)", value=10000000)

with col2:
    term = st.slider("Loan Term (Months)", 2, 20, 12)
    edu = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    emp = st.selectbox("Self Employed?", ["No", "Yes"])

st.divider()

# 4. The Decision Logic
if st.button("Generate AI Decision", use_container_width=True):
    # Prepare the data exactly how the model saw it in Colab
    input_df = pd.DataFrame([[
        2, # Default dependents
        1 if edu == "Graduate" else 0,
        1 if emp == "Yes" else 0,
        income, loan, term, cibil
    ]], columns=['no_of_dependents', 'education_encoded', 'self_employed_encoded', 
                 'income_annum', 'loan_amount', 'loan_term', 'cibil_score'])
    
    # Get Probability
    prob = model.predict_proba(input_df)[0][1]
    
    # Display Result
    if prob > 0.5:
        st.success(f"### ✅ APPROVED\n**Confidence Level:** {prob*100:.1f}%")
        st.balloons() # Added a little celebration for approvals!
    else:
        st.error(f"### ❌ REJECTED\n**Confidence Level:** {(1-prob)*100:.1f}%")
