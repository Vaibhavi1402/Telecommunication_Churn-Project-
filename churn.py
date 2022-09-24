#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from xgboost import XGBClassifier
import pandas as pd
import streamlit as st

st.title('Model Deployment:XGBC')
st.sidebar.header('user input parameters')
def user_input_features():
    account_length=st.sidebar.number_input("Insert the account_length")
    voice_mail_plan=st.sidebar.number_input("Insert the voice_mail_plan")
    voice_mail_messages=st.sidebar.number_input("Insert the voice_mail_messages")
    day_mins=st.sidebar.number_input("Insert the day_mins")
    evening_mins=st.sidebar.number_input("Insert the evening_mins")
    night_mins=st.sidebar.number_input("Insert the night_mins")
    international_mins=st.sidebar.number_input("Insert the international_mins")
    customer_service_calls=st.sidebar.number_input("Insert the customer_service_calls")
    international_plan=st.sidebar.number_input("Insert the international_plan")
    day_calls=st.sidebar.number_input("Insert the day_calls")
    day_charge=st.sidebar.number_input("Insert the day_charge")
    evening_calls=st.sidebar.number_input("Insert the evening_calls")
    evening_charge=st.sidebar.number_input("Insert the evening_charge")
    night_calls=st.sidebar.number_input("Insert the night_calls")
    night_charge=st.sidebar.number_input("Insert the night_charge")
    international_calls=st.sidebar.number_input("Insert the international_calls")
    international_charge=st.sidebar.number_input("Insert the international_charge")
    total_charge=st.sidebar.number_input("Insert the total_charge")
    
    data={'account_length':account_length,
          'voice_mail_plan':voice_mail_plan,
          'voice_mail_messages':voice_mail_messages,
          'day_mins':day_mins,
          'evening_mins':evening_mins,
          'night_mins':night_mins,
          'international_mins':international_mins,
          'customer_service_calls':customer_service_calls,
          'international_plan':international_plan,
          'day_calls':day_calls,
          'day_charge':day_charge,
          'evening_calls':evening_calls,
          'evening_charge':evening_charge,
          'night_calls':night_calls,
          'night_charge':night_charge,
          'international_calls':international_calls,
          'international_charge':international_charge,
          'total_charge':total_charge}
    features=pd.DataFrame(data,index=[0])
    return features

dff=user_input_features()
st.subheader('user Input Parameters')
st.write(dff)

churn=pd.read_csv("C:\\Users\\RAJ\\telecommunications_churn.csv")
churn=churn.dropna()

x=churn.iloc[:,0:18]
y=churn.iloc[:,18]
xgb=XGBClassifier()
xgb.fit(x,y)
prediction=xgb.predict(dff)
prediction_proba=xgb.predict_proba(dff)

st.subheader('predicted Result')
st.write("churn" if prediction_proba[0][1]>0.5 else "Non-churn")

st.subheader('Prediction Probability')
st.write(prediction_proba)

