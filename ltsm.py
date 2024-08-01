import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_sequences(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def plot_stock_data(dates, closing_prices, predicted_prices, future_dates, ema_50, ema_200, support, resistance):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dates, closing_prices, label="Actual Closing Prices", color='blue')
    ax.plot(future_dates, predicted_prices, label="Predicted Prices", linestyle='dashed', marker='o', color='red')
    ax.plot(dates, ema_50, label="EMA 50", linestyle='--', color='green')
    ax.plot(dates, ema_200, label="EMA 200", linestyle='--', color='purple')
    ax.axhline(y=support, color='black', linestyle='--', label="Support Level")
    ax.axhline(y=resistance, color='orange', linestyle='--', label="Resistance Level")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"Bitcoin Price Prediction (Last {len(dates)} days)")
    ax.legend()
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def analyze_stock(num_days):
    ticker = "BTC-USD"
    time_period = "2y"  # Increased to ensure we have enough data

    @st.cache_data
    def load_data():
        return yf.download(ticker, period=time_period, interval="1d")

    data = load_data()
    
    if data.empty or len(data) < num_days:
        st.error(f"Not enough data. Available data length: {len(data)}.")
        return
    
    closing_prices = data['Close'].values.reshape(-1, 1)[-num_days:]
    dates = data.index[-num_days:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(closing_prices)

    time_step = 20
    X, Y = create_sequences(data_normalized, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, Y, epochs=100, batch_size=1, verbose=0)

    predicted_prices = []
    current_sequence = data_normalized[-time_step:]

    for _ in range(10):
        current_sequence = current_sequence.reshape(1, time_step, 1)
        predicted_value = model.predict(current_sequence)
        predicted_prices.append(predicted_value[0][0])
        current_sequence = np.append(current_sequence[0][1:], predicted_value)
        current_sequence = current_sequence.reshape(time_step, 1)

    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

    future_dates = [dates[-1] + pd.DateOffset(days=i) for i in range(1, 11)]

    closing_prices_series = pd.Series(closing_prices.flatten())
    support = closing_prices_series.rolling(window=10).min().iloc[-1]
    resistance = closing_prices_series.rolling(window=10).max().iloc[-1]

    ema_50 = closing_prices_series.ewm(span=50, adjust=False).mean()
    ema_200 = closing_prices_series.ewm(span=200, adjust=False).mean()

    fig = plot_stock_data(dates, closing_prices, predicted_prices, future_dates, ema_50, ema_200, support, resistance)
    st.pyplot(fig)

def main():
    st.title("Bitcoin Price Analysis and Prediction")
    
    st.sidebar.header("Settings")
    num_days = st.sidebar.selectbox(
        "Select number of days for analysis:",
        options=[50, 100, 150, 200],
        index=1  # Default to 100 days
    )
    
    if st.sidebar.button("Analyze"):
        analyze_stock(num_days)

if __name__ == "__main__":
    main()