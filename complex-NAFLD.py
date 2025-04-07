import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('ET.pkl')  # Load the trained model

# Streamlit UI
st.title("NAFLD Disease Predictor")  # NAFLD Prediction Tool

# Sidebar for input options
st.sidebar.header("Input Sample Data")  # Sidebar for input sample data

# GGT input
GGT = st.sidebar.number_input("GGT:", min_value=1, max_value=70, value=50)  

# Insulin input
Insulin = st.sidebar.number_input("Insulin:", min_value=0.5, max_value=250.0, value=100.0)  

# WC input
WC = st.sidebar.number_input("WC:", min_value=50, max_value=170, value=100)  

# RBC input
RBC = st.sidebar.number_input("RBC:", min_value=3.5, max_value=7.5, value=5.5)  

# WBC input
WBC = st.sidebar.number_input("WBC:", min_value=0, max_value=20, value=5)  

# Process the input and make a prediction
feature_values = [[GGT, Insulin, WC, RBC, WBC]]  # Collect all input features
features = np.array(feature_values)  # Convert to NumPy array

if st.button("Make Prediction"):  # If clicked the prediction button
    # Predict the class and probabilities
    predicted_proba = model.predict_proba(features)[0]  # Predict probabilities
    predicted_class = model.predict(features)[0]  # Get the predicted class

    # Display the prediction results
    st.write(f"**Prediction Probabilities:** {predicted_proba}")  # Display probabilities

    # Generate advice based on the prediction result
    probability = predicted_proba[predicted_class] * 100  # Convert to percentage

    if predicted_class == 1:  # If predicted as ND
        advice = (
            f"According to our model, your risk of ND is high. "
            f"The probability of you having LIVER disease is {probability:.1f}%. "
            "This suggests that you might have a higher risk of ND disease. "
            "I recommend that you contact a cardiologist for a further examination and assessment, "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )  # Advice for high risk
    else:  # If predicted as not having ND
        advice = (
            f"According to our model, your risk of ND disease is low. "
            f"The probability of you not having ND disease is {probability:.1f}%. "
            "Nevertheless, maintaining a healthy lifestyle is still very important. "
            "I suggest that you have regular health check-ups to monitor your ND health, "
            "and seek medical attention if you experience any discomfort."
        )  # Advice for low risk

    st.write(advice)  # Display advice

    # Visualize the prediction probabilities
    sample_prob = {
        'Class_0': predicted_proba[0],  # Probability for Class 0
        'Class_1': predicted_proba[1]   # Probability for Class 1
    }

    # Set figure size
    plt.figure(figsize=(10, 3))  # Set figure size

    # Create bar chart
    bars = plt.barh(['Not Sick', 'Sick'], 
                    [sample_prob['Class_0'], sample_prob['Class_1']], 
                    color=['#512b58', '#fe346e'])  # Draw horizontal bar chart

    # Add title and labels with increased font size
    plt.title("Prediction Probability for Patient", fontsize=20, fontweight='bold')
    plt.xlabel("Probability", fontsize=14, fontweight='bold')
    plt.ylabel("Classes", fontsize=14, fontweight='bold')

    # Add probability labels to the bars
    for i, v in enumerate([sample_prob['Class_0'], sample_prob['Class_1']]):
        plt.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=14, color='black', fontweight='bold')

    # Show plot in Streamlit
    st.pyplot(plt)  # Render the plot in Streamlit



    # Hide other axes (top, right, bottom)
    plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
    plt.gca().spines['right'].set_visible(False)  # 隐藏右边框

    # Show the plot
    st.pyplot(plt)  # 显示图表