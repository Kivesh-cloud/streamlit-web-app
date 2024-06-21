import streamlit as st
import pandas as pd
import joblib

def download_model_from_url(url):
    response = requests.get(url)
    model_bytes = BytesIO(response.content)
    return joblib.load(model_bytes)

# URL where the model is hosted
model_url = 'https://1drv.ms/u/s!AnoSK-UPXGfigvUEgHUqMjaFNUMjfg?e=aVTDNx'

# Download the model
st.write('Downloading model from remote location...')
model = download_model_from_url(model_url)
st.write('Model downloaded successfully.')


# Function to predict heart disease
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol],
        'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach], 'exang': [exang],
        'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
    })
    prediction = model.predict(input_data)
    return prediction[0]



# Application title and description
st.title('Heart Disease Prediction')
st.write('Enter patient details to predict the likelihood of heart disease.')

# Sidebar with input fields
st.sidebar.title('Patient Details')
age = st.sidebar.number_input('Age', min_value=1, max_value=100)
sex = st.sidebar.selectbox('Sex', ['0', '1'])  # Assuming '0' for female, '1' for male
cp = st.sidebar.selectbox('Chest Pain Type', ['0', '1', '2', '3'])
trestbps = st.sidebar.number_input('Resting Blood Pressure (mm Hg)', min_value=0, max_value=300)
chol = st.sidebar.number_input('Cholesterol (mg/dL)', min_value=0, max_value=600)
fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ['0', '1'])
restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', ['0', '1', '2'])
thalach = st.sidebar.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=300)
exang = st.sidebar.selectbox('Exercise Induced Angina', ['0', '1'])
oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0)
slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', ['0', '1', '2'])
ca = st.sidebar.selectbox('Number of Major Vessels Colored by Flouroscopy', ['0', '1', '2', '3'])
thal = st.sidebar.selectbox('Thalassemia', ['0', '1', '2', '3'])

# Predict button
if st.sidebar.button('Predict'):
    prediction = predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca,
                                       thal)
    st.write('## Prediction:')
    if prediction == 1:
        st.write('The patient is likely to have heart disease.')
    else:
        st.write('The patient is not likely to have heart disease.')
