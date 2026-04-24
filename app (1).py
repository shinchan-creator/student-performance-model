import streamlit as st
import pandas as pd
import joblib

# 1. Page Styling
st.set_page_config(page_title="Pima Diabetes Predictor", layout="centered")

st.title("🩺 Pima Diabetes Prediction")
st.write("Enter patient details to predict diabetes outcome (0 = No Diabetes, 1 = Diabetes).")

try:
    # 2. Load the trained model
    model = joblib.load('advertising_model.pkl') # Corrected model filename

    # 3. Create a Layout for User Input (5 features)
    col1, col2, col3 = st.columns(3)
    col4, col5, col_empty = st.columns(3) # Use a 3rd column as empty to maintain layout consistency

    with col1:
        glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
    with col2:
        bloodpressure = st.number_input("Blood Pressure", min_value=0, max_value=122, value=70)
    with col3:
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, value=20)
    with col4:
        insulin = st.number_input("Insulin", min_value=0, max_value=846, value=79)
    with col5:
        bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=30.0)

    # 4. Create a 'Predict' button
    if st.button("Predict Diabetes"):
        # Create a DataFrame from the dynamic user input
        user_input = pd.DataFrame([[glucose, bloodpressure, skin_thickness, insulin, bmi]],
                                  columns=['Glucose', 'Bloodpressure', 'skinThickness', 'insulin', 'BMI']) # Corrected column names

        # Get prediction
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)

        # 5. Display Result in a nice box
        st.divider()
        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error(f"The model predicts: Diabetes (Probability: {prediction_proba[0][1]*100:.2f}%) ❌")
        else:
            st.success(f"The model predicts: No Diabetes (Probability: {prediction_proba[0][0]*100:.2f}%) ✅")

        st.write("Input Features:")
        st.dataframe(user_input)

except Exception as e:
    st.error(f"An error occurred: {e}. Please ensure the model file exists and is correctly structured.")
