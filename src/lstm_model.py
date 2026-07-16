import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data
df = pd.read_csv("data/JPM.csv")

# Clean
df = df.iloc[1:].copy()
df.rename(columns={"Price": "Date"}, inplace=True)

for col in ["Close", "High", "Low", "Open", "Volume"]:
    df[col] = pd.to_numeric(df[col])

# Close price
data = df["Close"].values.reshape(-1,1)

# Scale
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Sequences
window_size = 60

X = []
y = []

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# Train-Test Split
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# Model
model = Sequential()

model.add(
    LSTM(
        units=50,
        return_sequences=False,
        input_shape=(60,1)
    )
)

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

model.summary()
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)
print(predictions.shape)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

mae = mean_absolute_error(
    y_test_actual,
    predictions
)

rmse = mean_squared_error(
    y_test_actual,
    predictions
) ** 0.5

print("LSTM MAE :", mae)
print("LSTM RMSE:", rmse)

pd.DataFrame({
    "Actual": y_test_actual.flatten(),
    "LSTM_Prediction": predictions.flatten()
}).to_csv(
    "outputs/lstm_predictions.csv",
    index=False
)

print("LSTM predictions saved.")

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(
    y_test_actual,
    label="Actual"
)

plt.plot(
    predictions,
    label="LSTM Prediction"
)

plt.title("LSTM Forecast - JPM")
plt.legend()

plt.show()