
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Load the trained model (preprocessing + estimator pipeline serialized earlier)
model = joblib.load("superkart_model.joblib")

# Features the model was trained on - the API validates the incoming payload against this list
FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]


# Define a route for the home page
@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart System"


# Define an endpoint to predict sales for a single product-store combination
@superkart_api.post("/v1/predict")
def predict_sales():
    # Get JSON data from the request
    data = request.get_json()

    # Reject the request if any feature is missing
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    # Extract relevant features from the input data (keeps the column order used in training)
    sample = {feature: data[feature] for feature in FEATURES}

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Make a prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]

    # Return the prediction as a JSON response
    return jsonify({"Sales": round(prediction, 2)})


# Define an endpoint to predict sales for a batch of products (csv file upload)
@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files["file"]

    # Read the file into a DataFrame
    input_data = pd.read_csv(file)

    # Reject the file if any feature column is missing
    missing = [f for f in FEATURES if f not in input_data.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    # Make predictions for the batch data (columns re-ordered as in training)
    predictions = model.predict(input_data[FEATURES]).tolist()

    # Create an output dictionary mapping row index to predicted sales
    output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}

    return jsonify(output_dict)


# Run the Flask app in debug mode (in the container the app is served by Gunicorn instead)
if __name__ == "__main__":
    superkart_api.run(debug=True)
