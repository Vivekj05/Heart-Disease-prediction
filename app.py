import streamlit as st
import pandas as pd
import joblib

model= joblib.load("KNN_heart.pkl")
scaler= joblib.load("scaler.pkl")
expected_columns= joblib.load("columns.pkl")

st.title("HEART STROKE PREDICTION ")
# st.markdown()
st.markdown("Provide the following details")

age=st.slider("Age",18,100,40)
sex=st.selectbox("SEX",['M','F'])
chest_pain=st.selectbox("Cheast Pain type",["ATA","NAP","TA","ASY"])
resting_bp = st.slider("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, step=1, format="%d")
cholesterol = st.slider("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200, step=1, format="%d")
fasting_bs=st.selectbox("Fasting Blood Sugar>120 mg/dL",[0,1])
resting_ecg=st.selectbox("Resting ECG",["Normal","ST","LVH"])
max_hr=st.slider("Max Heart Rate",60,220,100)
exercise_angina=st.selectbox("Exercise-Induced Angina",["Y","N"])
oldpeak=st.slider("Oldpeak(ST depression)",0.0,6.0,1.0)
st_slope=st.selectbox("ST slope",["Up","Flat","Down"])


if st.button("Predict"):
    raw_input={
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df=pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0    

    input_df=input_df[expected_columns]

    scaled_input=scaler.transform(input_df)
    prediction=model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]
    # st.write(f"Model Confidence (High Risk): {prob:.2%}")
    # Show result
    # Test multiple cases and print model prediction
    st.write("Raw prediction:", model.predict(scaled_input)[0])
    st.write("Probability of class 0:", model.predict_proba(scaled_input)[0][0])
    st.write("Probability of class 1:", prob)

    # if prediction == 1:
    #     st.error("⚠️ High Risk of Heart Disease")
    # else:
    #     st.success("✅ Low Risk of Heart Disease") 
    if prob >= 0.5:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
