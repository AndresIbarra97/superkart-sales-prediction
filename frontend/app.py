
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend: inside the Docker network the backend container is reachable by its name
BACKEND_URL = "http://backend:7860"

# Page title and description
st.title("SuperKart Sales Forecasting System")
st.write("Enter the product and store details below to predict the total sales of the product in the store.")

# ---------------------------------------------------------------- Single prediction
st.subheader("Single Prediction")

# Input fields for product data
col1, col2 = st.columns(2)
with col1:
    Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
    Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.027, format="%.3f")
    Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
    Product_Id_char = st.selectbox("Product ID Character (FD = Food, DR = Drinks, NC = Non-Consumable)", ["FD", "DR", "NC"])
    Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

# Input fields for store data
with col2:
    Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"], index=1)
    Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"], index=1)
    Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"], index=1)
    Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=16)

# Create the JSON payload expected by the API
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category,
}

if st.button("Predict", type="primary"):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data, timeout=30)
        if response.status_code == 200:
            predicted_sales = response.json()["Sales"]
            st.success(f"Predicted Product Store Sales Total: {predicted_sales:,.2f}")
        else:
            st.error(f"The prediction API returned an error: {response.text}")
    except requests.exceptions.RequestException:
        st.error("Unable to connect to the prediction API.")

# ---------------------------------------------------------------- Batch prediction
st.subheader("Batch Prediction")
st.write(
    "Upload a CSV file with the columns: Product_Weight, Product_Sugar_Content, Product_Allocated_Area, Product_MRP, "
    "Store_Size, Store_Location_City_Type, Store_Type, Product_Id_char, Store_Age_Years, Product_Type_Category."
)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    if st.button("Predict for Batch", type="primary"):
        try:
            response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": uploaded_file}, timeout=60)
            if response.status_code == 200:
                results = response.json()
                st.success("Predictions completed successfully!")

                # The API returns {row_index: predicted_sales}; show it next to the uploaded data
                uploaded_file.seek(0)
                input_df = pd.read_csv(uploaded_file)
                input_df["Predicted_Sales"] = [results[str(i)] for i in range(len(input_df))]
                st.dataframe(input_df, use_container_width=True)
            else:
                st.error(f"The prediction API returned an error: {response.text}")
        except requests.exceptions.RequestException:
            st.error("Unable to connect to the prediction API.")
