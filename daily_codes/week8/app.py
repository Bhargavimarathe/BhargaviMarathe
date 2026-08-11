import streamlit as st
import numpy as np
import joblib

# Load FINAL model and scaler

model = joblib.load('xgb_final_model.pkl')
scaler = joblib.load('xgb_final_scaler.pkl')

# Page settings

st.set_page_config(page_title='Loan Risk Prediction', page_icon='💰')

# Title

st.title('💰 Loan Risk Prediction System')
st.write('Enter applicant details below to predict loan default risk.')

# Input fields

loan_amnt = st.number_input('Loan Amount', min_value=0, step=1000)
funded_amnt = st.number_input('Funded Amount', min_value=0, step=1000)
int_rate = st.number_input('Interest Rate (%)', min_value=0.0, step=0.1)
annual_inc = st.number_input('Annual Income', min_value=0, step=1000)

fico_range_low = st.number_input(
    'FICO Range Low',
    min_value=300,
    max_value=850,
    step=1
)

fico_range_high = st.number_input(
    'FICO Range High',
    min_value=300,
    max_value=850,
    step=1
)

revol_bal = st.number_input('Revolving Balance', min_value=0, step=1000)
tot_cur_bal = st.number_input('Total Current Balance', min_value=0, step=1000)
total_rev_hi_lim = st.number_input('Total Revolving High Limit', min_value=0, step=1000)

dti = st.number_input(
    'Debt-to-Income Ratio',
    min_value=0.0,
    step=0.1
)

# Predict button

if st.button('🔍 Predict Loan Risk'):

    # Create input array
    input_data = np.array([[
        loan_amnt,
        funded_amnt,
        int_rate,
        annual_inc,
        fico_range_low,
        fico_range_high,
        revol_bal,
        tot_cur_bal,
        total_rev_hi_lim,
        dti
    ]])

    # Show input data
    st.write('Input Data:')
    st.write(input_data)

    # Scale input
    input_scaled = scaler.transform(input_data)

    # ML model probability
    model_probability = model.predict_proba(input_scaled)[0][1] * 100

    # Business risk score
    risk_score = 0

    if int_rate >= 30:
        risk_score += 30
    elif int_rate >= 20:
        risk_score += 20

    if annual_inc <= 50000:
        risk_score += 20

    if fico_range_low <= 550:
        risk_score += 25

    if dti >= 40:
        risk_score += 25

    if loan_amnt >= 500000:
        risk_score += 15

    # Final combined risk percentage
    risk_percent = round((model_probability + risk_score) / 2, 2)

    # Result section
    st.subheader('Prediction Result')
    st.metric('Risk Probability', f'{risk_percent}%')

    if risk_percent >= 50:
        st.error('🚨 High Risk Loan Applicant')
        st.write('This applicant has a high probability of loan default.')
    else:
        st.success('✅ Low Risk Loan Applicant')
        st.write('This applicant appears financially safer according to the trained model.')

    # Progress bar
    st.progress(min(int(risk_percent), 100))