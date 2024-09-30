import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# List of top 50 stocks (you may want to update this list periodically)
TOP_50_STOCKS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "BRK-B", "V", "JPM", "JNJ",
    "WMT", "PG", "UNH", "MA", "NVDA", "HD", "DIS", "BAC", "ADBE", "CMCSA",
    "XOM", "VZ", "NFLX", "INTC", "PFE", "T", "CRM", "KO", "PEP", "CSCO",
    "ABT", "CVX", "MRK", "ORCL", "NKE", "ACN", "LLY", "TMO", "COST", "MCD",
    "NEE", "ABBV", "AVGO", "DHR", "WFC", "TXN", "BMY", "UNP", "PM", "HON"
]

@st.cache_data
def load_data(ticker, period="2y", interval="1d"):
    return yf.download(ticker, period=period, interval=interval)

def create_sequences(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def lstm_prediction(data, num_days):
    closing_prices = data['Close'].values.reshape(-1, 1)[-num_days:]
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
    return predicted_prices

def plot_price_prediction_and_rsi(data, ticker, num_days, lstm_prices):
    closing_prices = data['Close'].values.reshape(-1, 1)[-num_days:]
    dates = data.index[-num_days:]
    future_dates = [dates[-1] + pd.DateOffset(days=i) for i in range(1, 11)]

    closing_prices_series = pd.Series(closing_prices.flatten())
    support = closing_prices_series.rolling(window=10).min().iloc[-1]
    resistance = closing_prices_series.rolling(window=10).max().iloc[-1]

    ema_50 = closing_prices_series.ewm(span=50, adjust=False).mean()
    ema_200 = closing_prices_series.ewm(span=200, adjust=False).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Price Prediction Chart
    ax1.plot(dates, closing_prices, label="Actual Closing Prices", color='blue')
    ax1.plot(future_dates, lstm_prices, label="LSTM Prediction", linestyle='dashed', marker='o', color='red')
    ax1.plot(dates, ema_50, label="EMA 50", linestyle='--', color='green')
    ax1.plot(dates, ema_200, label="EMA 200", linestyle='--', color='purple')
    ax1.axhline(y=support, color='black', linestyle='--', label="Support Level")
    ax1.axhline(y=resistance, color='orange', linestyle='--', label="Resistance Level")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.set_title(f"{ticker} Price Prediction (Last {len(dates)} days)")
    ax1.legend()
    ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # RSI Chart
    rsi = calculate_rsi(data['Close'])[-num_days:]
    ax2.plot(dates, rsi, label='RSI', color='blue')
    ax2.axhline(y=70, color='red', linestyle='--')
    ax2.axhline(y=30, color='green', linestyle='--')
    ax2.set_ylabel('RSI')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig

def analyze_stock(ticker, num_days):
    data = load_data(ticker)
    
    if data.empty or len(data) < num_days:
        st.error(f"Not enough data for {ticker}. Available data length: {len(data)}.")
        return
    
    lstm_prices = lstm_prediction(data, num_days)

    st.subheader("Price Prediction and RSI")
    fig = plot_price_prediction_and_rsi(data, ticker, num_days, lstm_prices)
    st.pyplot(fig)

def main():
    st.title("Stock Price Analysis and Prediction")
    
    st.sidebar.header("Settings")
    selected_stock = st.sidebar.selectbox(
        "Select a stock for analysis:",
        options=TOP_50_STOCKS,
        index=0  # Default to the first stock in the list
    )
    
    num_days = st.sidebar.selectbox(
        "Select number of days for analysis:",
        options=[50, 100, 150, 200],
        index=1  # Default to 100 days
    )
    
    if st.sidebar.button("Analyze"):
        analyze_stock(selected_stock, num_days)

if __name__ == "__main__":
    main()