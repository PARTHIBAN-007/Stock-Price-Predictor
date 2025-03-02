# Stock Price Prediction using LSTM

## Goal

Predict stock prices using a deep learning model based on LSTM (Long Short-Term Memory) networks.

## Tech Stack

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit 

## Features of the Data

- `close`: Closing stock price
- `high`: Highest stock price of the day
- `low`: Lowest stock price of the day
- `open`: Opening stock price
- `volume`: Number of shares traded
- `adjClose`: Adjusted closing price
- `adjHigh`: Adjusted highest price
- `adjLow`: Adjusted lowest price
- `adjOpen`: Adjusted opening price
- `adjVolume`: Adjusted volume of shares traded

## How to Clone and Run the Model

### Clone the Repository

```sh
git clone https://github.com/your-username/Stock-Price-Prediction.git
cd Stock-Price-Prediction
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Train the Model

Run the training script to train the LSTM model:

```sh
python train.py
```

### Make Predictions

Use the following script to make predictions on new data:

```sh
python predict.py
```

### Run Streamlit App (Optional)

If you have a web interface for visualization, start the Streamlit app:

```sh
streamlit run app.py
```

---



