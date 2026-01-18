import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


symbol = "BTC-USD"  # İstersen 'AAPL', 'GOOGL', 'THYAO.IS' yapabilirsin.
print(f"{symbol} verisi indiriliyor...")
df = yf.download(symbol, start="2021-01-01", end="2026-02-01")


data = df[['Close']].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[0:train_len]


def create_sequences(dataset, look_back=60):
    x, y = [], []
    for i in range(look_back, len(dataset)):
        x.append(dataset[i-look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)

x_train, y_train = create_sequences(train_data)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2)) # Aşırı öğrenmeyi (Overfitting) engellemek için %20 nöronu kapat


model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')


print("Model eğitiliyor (Bu işlem biraz sürebilir)...")
model.fit(x_train, y_train, batch_size=32, epochs=25)

test_data = scaled_data[train_len - 60:]
x_test, y_test = create_sequences(test_data)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


predictions = model.predict(x_test)

predictions = scaler.inverse_transform(predictions)

real_values = data[train_len:]


plt.figure(figsize=(14, 6))
plt.title(f'{symbol} Fiyat Tahmin Modeli')
plt.xlabel('Tarih', fontsize=12)
plt.ylabel('Fiyat (USD)', fontsize=12)
plt.plot(df.index[train_len:], real_values, color='blue', label='Gerçek Fiyat')
plt.plot(df.index[train_len:], predictions, color='red', label='Yapay Zeka Tahmini')
plt.legend()
plt.show()