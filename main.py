from flask import Flask, jsonify, render_template, request, redirect, url_for
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math
from datetime import datetime
from sklearn.linear_model import LinearRegression
import os
import requests
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

app = Flask(__name__)

@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return render_template('index.html')

def get_historical(quote):
    try:
        url = f'https://api.twelvedata.com/time_series?symbol={quote}&interval=1day&outputsize=1700&apikey=cda2e65c28ee4d44a12f6d37d3e4667b'
        response = requests.get(url)
        data = response.json()

        if 'values' not in data:
            raise ValueError("Twelve Data response is invalid")

        df = pd.DataFrame(data['values'])
        df = df.rename(columns={
            'datetime': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['adj close'] = df['close']

        # Add technical indicators
        df['ma7'] = df['close'].rolling(window=7).mean()
        df['ma21'] = df['close'].rolling(window=21).mean()
        df['ema'] = df['close'].ewm(span=20, adjust=False).mean()
        df['stddev'] = df['close'].rolling(7).std()
        df['return'] = df['close'].pct_change()
        df['rsi'] = compute_rsi(df['close'], 14)

        df = df.dropna()
        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'adj close', 'ma7', 'ma21', 'ema', 'stddev', 'return', 'rsi']]
        df.to_csv(f'{quote}.csv', index=False)
        return df

    except Exception as e:
        print(f"Twelve Data Error: {e}")
        return pd.DataFrame()
        

def get_current_price(symbol, twelve_key='', alpha_key=''):
    try:
        # First try Twelve Data
        url = f'https://api.twelvedata.com/price?symbol={symbol}&apikey={twelve_key}'
        response = requests.get(url)
        data = response.json()
        if 'price' in data:
            return round(float(data['price']), 2)
        raise ValueError("No price in Twelve Data response")
    except Exception as e:
        print(f"Twelve Data current price error: {e}")
        try:
            # Fallback to Alpha Vantage
            url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={alpha_key}'
            response = requests.get(url)
            data = response.json()
            price = data['Global Quote']['05. price']
            return round(float(price), 2)
        except Exception as e:
            print(f"Alpha Vantage current price error: {e}")
            return "Unavailable"

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@app.route('/insertintotable', methods=['POST'])
@app.route('/insertintotable', methods=['POST'])
def insertintotable():
    quote = request.form.get('nm')
    if not quote:
        return redirect(url_for('index'))

    try:
        print("[INFO] Fetching historical data...")
        df_full = get_historical(quote)
        print("[OK] Got historical data, fetching current price...")
        current_price = get_current_price(quote)

        if df_full.empty:
            return render_template('index.html', error=True)

        # ==== Prepare datasets for different models ====
        df_arima = df_full.copy()
        df_lstm = df_full.tail(5000).copy() if len(df_full) > 5000 else df_full.copy()
        df_lr = df_full.tail(2000).copy() if len(df_full) > 2000 else df_full.copy()
        df_xgb = df_full.tail(300).copy() if len(df_full) > 300 else df_full.copy()

        # Add Code column for displaying info later
        df_display = df_full.copy()
        df_display['Code'] = quote
        df_display = df_display[['Code'] + [col for col in df_display.columns if col != 'Code']]

        # ==== Run each model ====
        print("[INFO] Running models...")
        arima_pred, error_arima = ARIMA_ALGO(df_arima)
        lstm_pred, error_lstm, lstm_forecast_list = LSTM_ALGO(df_lstm)
        df_lr, lr_pred, forecast_set, mean, error_lr = LIN_REG_ALGO(df_lr)
        df_xgb, xgb_pred, xgb_forecast, xgb_mean, error_xgb = XGBOOST_ALGO(df_xgb)

        today_stock = df_display.iloc[-1:].round(2)
        idea, decision = recommending(df_display, 0, today_stock, mean)

        # Prepare forecast lists for frontend
        forecast_list = forecast_set.flatten().tolist()
        forecast_set_lstm = lstm_forecast_list
        xgb_forecast_list = xgb_forecast.flatten().tolist()

        return render_template('results.html',
            quote=quote,
            arima_pred=round(arima_pred, 2),
            lstm_pred=round(lstm_pred, 2),
            lr_pred=round(lr_pred, 2),
            xgb_pred=round(xgb_pred, 2),
            open_s=today_stock['open'].to_string(index=False),
            close_s=today_stock['close'].to_string(index=False),
            adj_close=today_stock['adj close'].to_string(index=False),
            high_s=today_stock['high'].to_string(index=False),
            low_s=today_stock['low'].to_string(index=False),
            vol=today_stock['volume'].to_string(index=False),
            current_price=current_price,
            forecast_set=forecast_list,
            xgb_forecast=xgb_forecast_list,
            forecast_set_lstm=forecast_set_lstm,
            error_lr=round(error_lr, 2),
            error_lstm=round(error_lstm, 2),
            error_arima=round(error_arima, 2),
            error_xgb=round(error_xgb, 2),
            idea=idea,
            decision=decision
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template('index.html', error=True)

    

# Keep your ARIMA_ALGO, LSTM_ALGO, LIN_REG_ALGO and recommending() functions unchanged from latest working version.

def ARIMA_ALGO(df):
    try:
        # Validate DataFrame structure
        required_columns = {'date', 'close'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Missing required columns. Found: {df.columns}")
        
        # Convert to datetime and sort
        df['date'] = pd.to_datetime(df['date'])
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])
        df = df.sort_values('date').set_index('date')
        
        # Plot trends
        plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(df['close'])
        plt.savefig('static/Trends.png')
        plt.close()

        # Train-test split
        values = df['close'].values
        split_idx = int(len(values) * 0.8)
        train, test = values[:split_idx], values[split_idx:]
        
        # ARIMA Model
        predictions = []
        history = list(train)
        for t in range(len(test)):
            model = ARIMA(history, order=(6,1,0))
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        
        # Plot results
        plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(test, label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.legend()
        plt.savefig('static/ARIMA.png')
        plt.close()

        return predictions[-1], np.sqrt(mean_squared_error(test, predictions))
    
    except Exception as e:
        print(f"ARIMA processing failed: {e}")
        return 0.0, 0.0

# ************* LSTM SECTION **********************
def LSTM_ALGO(df):
    import tensorflow as tf  # Ensure global import
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import math
    from sklearn.metrics import mean_squared_error

    if len(df) < 14:
        raise ValueError("Insufficient data for LSTM 7-day forecast")

    # GPU memory growth config
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth setting error: {e}")

    # === GPU Warm-up to reduce delay ===
    print("[INFO] Warming up GPU...")
    _ = tf.matmul(tf.constant([[1.0]]), tf.constant([[1.0]]))  # Forces early CUDA load

    dataset_train = df.iloc[0:int(0.8 * len(df)), :]
    dataset_test = df.iloc[int(0.8 * len(df)):, :]
    training_set = df['close'].values.reshape(-1, 1)

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train = []
    y_train = []
    for i in range(7, len(training_set_scaled)):
        X_train.append(training_set_scaled[i - 7:i, 0])
        y_train.append(training_set_scaled[i, 0])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    last_input = X_train[-1]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    last_input = last_input.reshape((1, 7, 1))

    # === LSTM Model ===
    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(7, 1)))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.1))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    print("[INFO] Starting LSTM training...")
    regressor.fit(X_train, y_train, epochs=5, batch_size=16, verbose=1)
    print("[INFO] LSTM training complete.")

    # === Prediction ===
    dataset_total = pd.concat((dataset_train['close'], dataset_test['close']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values.reshape(-1, 1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(7, len(inputs)):
        X_test.append(inputs[i - 7:i, 0])
    X_test = np.array(X_test).reshape((len(X_test), 7, 1))

    real_stock_price = dataset_test['close'].values.reshape(-1, 1)
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(real_stock_price, label='Actual Price')
    plt.plot(predicted_stock_price, label='Predicted Price')
    plt.legend(loc=4)
    plt.savefig('static/LSTM.png')
    plt.close(fig)

    error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

    # === Forecast 7 days ahead ===
    lstm_forecast_list = []
    current_input = last_input.copy()

    for _ in range(7):
        next_scaled = regressor.predict(current_input)
        next_price = sc.inverse_transform(next_scaled)[0][0]
        lstm_forecast_list.append(next_price)

        new_input = np.append(current_input[0, 1:], [[next_scaled[0][0]]], axis=0)
        current_input = new_input.reshape((1, 7, 1))

    lstm_pred = lstm_forecast_list[0]

    print()
    print("##############################################################################")
    print("Tomorrow's Closing Price Prediction by LSTM:", lstm_pred)
    print("LSTM RMSE:", error_lstm)
    print("##############################################################################")

    return lstm_pred, error_lstm, lstm_forecast_list



# ***************** LINEAR REGRESSION SECTION ******************       
def LIN_REG_ALGO(df):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    import math

    forecast_out = 7
    df = df.copy()

    # Technical Indicators
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA21'] = df['close'].rolling(window=21).mean()
    df['EMA'] = df['close'].ewm(span=20, adjust=False).mean()
    df['STDDEV'] = df['close'].rolling(window=7).std()
    df['Return'] = df['close'].pct_change()

    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))

    # Drop NaNs from indicator calculations
    df = df.dropna().reset_index(drop=True)

    # Future label
    df['close after n days'] = df['close'].shift(-forecast_out)

    # Final dataset for ML
    features = ['close', 'MA7', 'MA21', 'EMA', 'STDDEV', 'Return', 'RSI']
    df_model = df[features + ['close after n days']].dropna()

    # Train/test split
    X = df_model[features].values
    y = df_model['close after n days'].values.reshape(-1, 1)

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Future prediction input
    X_to_be_forecasted = df[features].tail(forecast_out).values

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_to_be_forecasted = scaler.transform(X_to_be_forecasted)

    # Train model
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict
    y_test_pred = model.predict(X_test)

    # Evaluate
    error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))

    # Forecast future
    forecast_set = model.predict(X_to_be_forecasted)
    lr_pred = forecast_set[0][0]
    mean = float(forecast_set.mean())

    # Plotting
    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(y_test, label='Actual Price')
    plt.plot(y_test_pred, label='Predicted Price')
    plt.legend(loc=4)
    plt.savefig('static/LR.png')
    plt.close(fig)

    print()
    print("##############################################################################")
    print("Tomorrow's Closing Price Prediction by Linear Regression: ", round(lr_pred, 2))
    print("Linear Regression RMSE:", round(error_lr, 2))
    print("##############################################################################")

    return df, lr_pred, forecast_set, mean, error_lr



def XGBOOST_ALGO(df):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from xgboost import XGBRegressor
    import math

    forecast_out = 7
    df = df.copy()

    # === Add technical indicators like in Linear Regression ===
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA21'] = df['close'].rolling(window=21).mean()
    df['EMA'] = df['close'].ewm(span=20, adjust=False).mean()
    df['STDDEV'] = df['close'].rolling(window=7).std()
    df['Return'] = df['close'].pct_change()

    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))

    df = df.dropna().reset_index(drop=True)

    df['close after n days'] = df['close'].shift(-forecast_out)

    # Prepare dataset
    features = ['close', 'MA7', 'MA21', 'EMA', 'STDDEV', 'Return', 'RSI']
    df_model = df[features + ['close after n days']].dropna()

    X = df_model[features].values
    y = df_model['close after n days'].values.reshape(-1, 1)

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    X_to_be_forecasted = df[features].tail(forecast_out).values

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_to_be_forecasted = scaler.transform(X_to_be_forecasted)

    # Train model (tune n_estimators & learning_rate)
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    error_xgb = math.sqrt(mean_squared_error(y_test, y_test_pred))

    # Forecast
    forecast_set = model.predict(X_to_be_forecasted)
    forecast_set = forecast_set.reshape(-1, 1)
    xgb_pred = forecast_set[0][0]
    mean = float(forecast_set.mean())

    # Plot
    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(y_test, label='Actual Price')
    plt.plot(y_test_pred, label='Predicted Price')
    plt.legend(loc=4)
    plt.savefig('static/XGB.png')
    plt.close(fig)

    print()
    print("##############################################################################")
    print("Tomorrow's Closing Price Prediction by XGBoost: ", round(xgb_pred, 2))
    print("XGBoost RMSE:", round(error_xgb, 2))
    print("##############################################################################")

    return df, xgb_pred, forecast_set, mean, error_xgb



# **************** RECOMMENDATION FUNCTION **************************
def recommending(df, global_polarity, today_stock, mean):
    if today_stock.empty:
        return "N/A", "Insufficient Data"
    if today_stock.iloc[-1]['close'] < mean:
        if global_polarity > 0:
            idea = "RISE"
            decision = "BUY"
        else:
            idea = "FALL"
            decision = "SELL"
    else:
        idea = "FALL"
        decision = "SELL"
    
    return idea, decision



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
