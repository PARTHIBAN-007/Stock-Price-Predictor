import streamlit as st
import numpy as np
import pandas as pd
from predictions import make_predictions
st.title("Time Series Forecasting Input")

features = ['close', 'high', 'low', 'open', 'volume', 'adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume']
input_data = {}

st.sidebar.header("Enter Feature Values")

for feature in features:
    input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0, step=0.1)

input_df = pd.DataFrame([input_data])
print(input_df)
st.subheader("User Input Data")
st.write(input_df)

if st.button("Predict"):
    predicion = make_predictions(input_df)
    st.write(predicion)

