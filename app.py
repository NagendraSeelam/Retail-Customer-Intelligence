
import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("xgboost_high_value_model.pkl")
features = joblib.load("features.pkl")

# Page title
st.title("Retail Customer Intelligence System")

st.write(
    "Predict whether a customer is likely to become a high-value customer "
    "based on their historical purchasing behavior."
)

st.divider()

st.subheader("Enter Customer Information")

# Customer inputs
total_spend = st.number_input(
    "Total Spend",
    min_value=0.0,
    value=1000.0
)

total_orders = st.number_input(
    "Total Orders",
    min_value=1,
    value=5
)

total_items = st.number_input(
    "Total Items Purchased",
    min_value=1,
    value=20
)

average_order_value = st.number_input(
    "Average Order Value",
    min_value=0.0,
    value=200.0
)

average_item_price = st.number_input(
    "Average Item Price",
    min_value=0.0,
    value=10.0
)

recency = st.number_input(
    "Recency (Days Since Last Purchase)",
    min_value=0,
    value=30
)

customer_lifetime = st.number_input(
    "Customer Lifetime (Days)",
    min_value=0,
    value=100
)

# Create input dataframe
input_data = pd.DataFrame([[
    total_spend,
    total_orders,
    total_items,
    average_order_value,
    average_item_price,
    recency,
    customer_lifetime
]], columns=features)

# Prediction
if st.button("Predict Customer Value"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.success("High-Value Customer")
    else:
        st.info("Not High-Value Customer")

    st.write(
        f"Probability of becoming a high-value customer: "
        f"{probability:.2%}"
    )
