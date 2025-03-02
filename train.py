import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv(r"D:\Projects\ML projects\Stock Price Prediction\Data\GOOG.csv")
dff = df.drop(columns=['date', 'divCash', 'splitFactor'])
dff = dff[['close', 'high', 'low', 'open', 'volume', 'adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume']]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dff)
joblib.dump(scaler, "scaler.pkl")

torch.manual_seed(42)

seq_length = 60
input_size = 10
hidden_size = 64
num_layers = 3
num_epochs = 40
batch_size = 32
learning_rate = 0.001

def create_sequences(data, sequence_length):
    sequences, targets = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        targets.append(data[i+sequence_length, 0])  
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

x, y = create_sequences(scaled_data, seq_length)

train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

class StockDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32) 
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = StockDataset(x_train, y_train)
test_dataset = StockDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        return self.fc(lstm_out[:, -1, :])  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = LSTMModel(input_size, hidden_size, num_layers).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def training():
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.5f}")

def evaluate(model, dataloader):
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())

    return np.array(predictions), np.array(actuals)

if __name__ == "__main__":
    training()

    preds, actuals = evaluate(model, test_loader)

    preds = scaler.inverse_transform(np.column_stack((preds, np.zeros((len(preds), 9)))))[:, 0]
    actuals = scaler.inverse_transform(np.column_stack((actuals.reshape(-1, 1), np.zeros((len(actuals), 9)))))[:, 0]

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    def evaluation_metrics(preds, actuals):
        n = len(actuals)
        p = input_size  

        mae = mean_absolute_error(actuals, preds)
        mse = mean_squared_error(actuals, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals, preds)
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        print(f"R2 Score: {r2:.6f}")
        print(f"Adjusted R2 Score: {adjusted_r2:.6f}")

    evaluation_metrics(preds, actuals)

    torch.save(model.state_dict(), "LSTM.pkl")
    print("Model saved successfully.")
