import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("cancer_patient_acc_lr_model.pkl")

st.title("Lung Cancer Risk Prediction System")

st.write("Enter patient details")

# Inputs
age = st.number_input("Age", 1, 100, 30)
gender = st.selectbox("Gender", [1, 2])
air_pollution = st.slider("Air Pollution", 1, 8, 3)
alcohol_use = st.slider("Alcohol Use", 1, 8, 3)
dust_allergy = st.slider("Dust Allergy", 1, 8, 3)
occupational_hazards = st.slider("Occupational Hazards", 1, 8, 3)
genetic_risk = st.slider("Genetic Risk", 1, 8, 3)
chronic_lung_disease = st.slider("Chronic Lung Disease", 1, 8, 3)
balanced_diet = st.slider("Balanced Diet", 1, 8, 3)
obesity = st.slider("Obesity", 1, 8, 3)
smoking = st.slider("Smoking", 1, 8, 3)
passive_smoker = st.slider("Passive Smoker", 1, 8, 3)
chest_pain = st.slider("Chest Pain", 1, 8, 3)
coughing_of_blood = st.slider("Coughing of Blood", 1, 8, 3)
fatigue = st.slider("Fatigue", 1, 8, 3)
weight_loss = st.slider("Weight Loss", 1, 8, 3)
shortness_of_breath = st.slider("Shortness of Breath", 1, 8, 3)
wheezing = st.slider("Wheezing", 1, 8, 3)
swallowing_difficulty = st.slider("Swallowing Difficulty", 1, 8, 3)
clubbing_of_finger_nails = st.slider("Clubbing of Finger Nails", 1, 8, 3)
frequent_cold = st.slider("Frequent Cold", 1, 8, 3)
dry_cough = st.slider("Dry Cough", 1, 8, 3)
snoring = st.slider("Snoring", 1, 8, 3)

# Prediction button
if st.button("Predict Cancer Risk"):

    features = np.array([[age, gender, air_pollution, alcohol_use,
                          dust_allergy, occupational_hazards,
                          genetic_risk, chronic_lung_disease,
                          balanced_diet, obesity, smoking,
                          passive_smoker, chest_pain,
                          coughing_of_blood, fatigue,
                          weight_loss, shortness_of_breath,
                          wheezing, swallowing_difficulty,
                          clubbing_of_finger_nails,
                          frequent_cold, dry_cough, snoring]])

    prediction = model.predict(features)

    if prediction[0] == 0:
        st.success("Low Cancer Risk")
    elif prediction[0] == 1:
        st.warning("Medium Cancer Risk")
    else:
        st.error("High Cancer Risk")
