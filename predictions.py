import torch
import pandas as pd
import joblib
from train import LSTMModel
import numpy as np

seq_length = 60
input_size = 10
hidden_size = 64
num_layers = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size, hidden_size, num_layers).to(device)
model.load_state_dict(torch.load(r"D:\Projects\ML projects\Stock Price Prediction\Model\LSTM.pkl", map_location=device))
model.eval()

scaler = joblib.load(r"D:\Projects\ML projects\Stock Price Prediction\Model\scaler.pkl")

def make_predictions(inputs):
    if not isinstance(inputs, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    input_data = scaler.transform(inputs)

    input_tensor = torch.tensor(input_data, dtype=torch.float32, device=device)

    input_tensor = input_tensor.unsqueeze(0)  
    with torch.no_grad():
        scaled_prediction = model(input_tensor).cpu().numpy()  
    scaled_prediction = scaled_prediction.reshape(-1, 1)  
    unscaled_prediction = scaler.inverse_transform(np.hstack((scaled_prediction, np.zeros((len(scaled_prediction), 9)))))[:, 0]

    return unscaled_prediction[0]  

