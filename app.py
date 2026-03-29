import streamlit as st
import joblib
import pandas as pd
import google.generativeai as genai

# --- 1. SET UP THE BRAIN & THE CHAT ---
try:
    model = joblib.load('credit_model.pkl')
except:
    st.error("Model file not found!")

# PASTE YOUR API KEY HERE
genai.configure(api_key="AIzaSyDLgOGqGxNrATYwTn3sXtJLQuh1ecE1TN0")
chat_model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="AI Credit Risk Agent", layout="wide")

# --- 2. THE CHAT SIDEBAR ---
with st.sidebar:
    st.title("🤖 AI Finance Advisor")
    st.markdown("Ask me anything about this loan or finance theory!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # We tell Gemini to act like a Finance Professor
            context = "You are a senior credit risk analyst. Answer briefly and professionally."
            response = chat_model.generate_content(f"{context} User asked: {prompt}")
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

    st.divider()
    st.header("Input Data")
    cibil = st.slider("CIBIL Score", 300, 900, 750)
    income = st.number_input("Annual Income ($)", value=5000000)
    loan = st.number_input("Loan Amount ($)", value=10000000)
    edu = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    emp = st.selectbox("Self Employed?", ["No", "Yes"])

# --- 3. THE MAIN DASHBOARD ---
st.title("🏦 AI Credit Risk Agent")

if st.button("Run Detailed Assessment", use_container_width=True):
    input_df = pd.DataFrame([[
        2, 1 if edu == "Graduate" else 0, 1 if emp == "Yes" else 0,
        income, loan, 12, cibil # Using 12 months as default
    ]], columns=['no_of_dependents', 'education_encoded', 'self_employed_encoded', 
                 'income_annum', 'loan_amount', 'loan_term', 'cibil_score'])
    
    prob = model.predict_proba(input_df)[0][1]
    
    col1, col2 = st.columns(2)
    with col1:
        if prob > 0.5:
            st.success(f"### APPROVED ({prob*100:.1f}%)")
        else:
            st.error(f"### REJECTED ({(1-prob)*100:.1f}%)")
    
    with col2:
        st.info("**AI Insight:** Use the sidebar chat to ask why this decision was made or to discuss the macroeconomic implications of this interest rate environment.")
