import joblib
import streamlit as st
import pandas as pd
import os


model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)


st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide"
)

st.title("📊 Telco Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn.")


# Sidebar
st.sidebar.header("About")
st.sidebar.write("ML model predicts churn risk based on customer details.")


def user_input():
    with st.form("churn_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Select", "Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["Select", "Yes", "No"])
            phone = st.selectbox("Phone Service", ["Select", "Yes", "No"])
            internet = st.selectbox(
                "Internet Service", ["Select", "DSL", "Fiber optic", "No"]
            )
            onlinesecurity = st.selectbox(
                "Online Security", ["Select", "Yes", "No", "No internet service"]
            )
            onlinebackup = st.selectbox(
                "Online Backup", ["Select", "Yes", "No", "No internet service"]
            )
            device = st.selectbox(
                "Device Protection", ["Select", "Yes", "No", "No internet service"]
            )
            techsupport = st.selectbox(
                "Tech Support", ["Select", "Yes", "No", "No internet service"]
            )

        with col2:
            streamingtv = st.selectbox(
                "Streaming TV", ["Select", "Yes", "No", "No internet service"]
            )
            streamingmovies = st.selectbox(
                "Streaming Movies", ["Select", "Yes", "No", "No internet service"]
            )
            paperless = st.selectbox("Paperless Billing", ["Select", "Yes", "No"])
            contract = st.selectbox(
                "Contract", ["Select", "Month-to-month", "One year", "Two year"]
            )
            payment = st.selectbox(
                "Payment Method",
                [
                    "Select",
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            monthly = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
            total = st.slider("Total Charges", 0.0, 10000.0, 2000.0)


        submit = st.form_submit_button("🔍 Predict Churn")

        categorical_fields = [
            gender, senior,  phone,
            internet, onlinesecurity, onlinebackup, device,
            techsupport, streamingtv, streamingmovies,
            paperless, contract, payment
        ]

        data = {
            "Gender": gender,
            "Senior Citizen": senior ,
            "Phone Service": phone,
            "Internet Service": internet,
            "Online Security": onlinesecurity,
            "Online Backup": onlinebackup,
            "Device Protection": device,
            "Tech Support": techsupport,
            "Streaming TV": streamingtv,
            "Streaming Movies": streamingmovies,
            "Paperless Billing": paperless,
            "Contract": contract,
            "Payment Method": payment,
            "Tenure Months": tenure,
            "Monthly Charges": monthly,
            "Total Charges": total,

        }

    return submit, pd.DataFrame([data]), categorical_fields


submit, input_df, fields = user_input()

# Fix numeric columns
numeric_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']
input_df[numeric_cols] = input_df[numeric_cols].apply(
    pd.to_numeric, errors="coerce"
)

# Fix categorical columns
categorical_cols = [
    'Gender', 'Senior Citizen', 'Phone Service',
       'Internet Service', 'Online Security',
       'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV',
       'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method'
]
input_df[categorical_cols] = input_df[categorical_cols].astype(str)

st.write(input_df)

# Prediction
if submit:
    if any(str(v).startswith("Select") for v in fields):
        st.warning("⚠ Please fill all fields.")
        st.stop()


    probs = model.predict_proba(input_df)[0]
    stay_prob = probs[0]
    churn_prob = probs[1]

    st.subheader("📌 Prediction Result")

    st.error(f"⚠ High Churn Risk: {churn_prob:.2f}")
    st.success(f"✅ Likely to Stay: {stay_prob:.2f}")








