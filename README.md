# Heart Disease Prediction App

This Streamlit web application predicts the likelihood of heart disease based on user input. It uses a logistic regression model trained on the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [Deployment](#deployment)
- [Dependencies](#dependencies)

## Overview

The application allows users to input various parameters related to their health (such as age, sex, cholesterol levels, etc.) and predicts whether they are likely to have heart disease based on these inputs. The prediction model is trained using machine learning techniques and is deployed using Streamlit, a Python library for creating interactive web applications.

## Setup

To set up the application locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone Kivesh-cloud/streamlit-web-app
   cd streamlit-web-app
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   streamlit run main.py
   ```

   This will start the Streamlit server locally. Open the provided URL in a web browser to access the application.

## Usage

Once the application is running, follow these steps to predict heart disease:

1. Fill in the details in the input fields provided:
   - Age
   - Sex (Male/Female)
   - Chest Pain Type
   - Resting Blood Pressure
   - Cholesterol
   - Fasting Blood Sugar
   - Resting Electrocardiographic Results
   - Maximum Heart Rate Achieved
   - Exercise Induced Angina
   - ST Depression Induced by Exercise Relative to Rest
   - Slope of the Peak Exercise ST Segment
   - Number of Major Vessels Colored by Fluoroscopy
   - Thalassemia

2. Click on the Predict button to see the prediction result.

## Deployment

This application can be deployed on various platforms that support Python applications, such as:

- Streamlit Cloud

Ensure to include the requirements.txt file in your deployment setup to ensure all dependencies are installed correctly. This is currently deployed at https://medicalwebapp.streamlit.app/

## Dependencies

The application uses the following Python libraries:

- streamlit
- joblib
- numpy
- pandas
- scikit-learn

These dependencies are listed in the requirements.txt file for easy installation.