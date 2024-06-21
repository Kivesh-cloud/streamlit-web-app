import streamlit as st
import joblib
import numpy as np

def load_model(model_path):
    try:
        model = joblib.load(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model_path = 'heart_disease_prediction_model.pkl' 
scaler_path = 'scaler.pkl'  
model = load_model(model_path)
scaler = joblib.load(scaler_path)


def preprocess_input(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):

    cp_mapping = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    restecg_mapping = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
    slope_mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    thal_mapping = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}


    sex = 1 if sex == 'Male' else 0
    fbs = 1 if fbs == 'Yes' else 0
    exang = 1 if exang == 'Yes' else 0


    input_data = np.array([[
        age, sex, cp_mapping[cp], trestbps, chol, fbs, restecg_mapping[restecg], thalach, exang,
        oldpeak, slope_mapping[slope], int(ca), thal_mapping[thal]
    ]])


    input_data_scaled = scaler.transform(input_data)


    if input_data_scaled.shape[1] < 30:
        input_data_scaled = np.pad(input_data_scaled, ((0, 0), (0, 30 - input_data_scaled.shape[1])), 'constant')

    return input_data_scaled


def predict_heart_disease(input_data):
    try:
        if model:
            # Predict the target
            prediction = model.predict(input_data)
            return prediction[0]
        else:
            st.error("Model not loaded. Cannot make predictions.")
            return None
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


st.title('Heart Disease Prediction')
st.markdown('Fill in the details below to predict whether a patient likely suffers from heart disease.')


age = st.slider('Age', 20, 100, 50)
sex = st.radio('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
trestbps = st.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
chol = st.slider('Cholesterol (mg/dl)', 100, 600, 200)
fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
thalach = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
exang = st.radio('Exercise Induced Angina', ['Yes', 'No'])
oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 0.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
ca = st.selectbox('Number of Major Vessels Colored by Flouroscopy', ['0', '1', '2', '3'])
thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])


if st.button('Predict'):
    input_data = preprocess_input(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    
    if input_data.shape[1] == 30:  
        prediction = predict_heart_disease(input_data)
        
        if prediction is not None:
            # Display prediction
            if prediction == 0:
                st.error('The patient is predicted to NOT have heart disease.')
            else:
                st.success('The patient is predicted to HAVE heart disease.')
    else:
        st.error(f"Input data shape {input_data.shape} does not match model's expected shape (1, 30).")
