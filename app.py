# To run , open terminal and write streamlit run app.py
import streamlit as st
import joblib
import numpy as np

# Load the trained model
knn_model = joblib.load('iris_classifier.pkl')
# Define target names manually
iris_species = ['setosa', 'versicolor', 'virginica']

# Define the Streamlit app
st.title("Iris Species Prediction (Streamlit)")

# Input fields for user input
sepal_length = st.number_input('Sepal Length (cm)')
sepal_width = st.number_input('Sepal Width (cm)')
petal_length = st.number_input('Petal Length (cm)')
petal_width = st.number_input('Petal Width (cm)')

# Prediction button
if st.button('Predict'):
    # Convert input values to numpy array
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]], dtype=float)
    # Make prediction
    prediction = knn_model.predict(features)[0]
    species = iris_species[prediction]
    result = f"The input belongs to the species '{species}'."
    # Display prediction result
    st.write(result)
