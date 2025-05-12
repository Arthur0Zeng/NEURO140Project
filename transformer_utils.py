import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Get data for past 7 days
def fetch_stock_info(stock_code = "NVDA", date_str = "2025-05-01", output_dir="outputs",passday = 7):
    try:
        end_date = (datetime.strptime(date_str, "%Y-%m-%d")
          if isinstance(date_str, str) else date_str)
        end_date = end_date + timedelta(days = 1)
        start_date = end_date - timedelta(days=passday)
        ticker = yf.Ticker(stock_code)
        stock_price_df = ticker.history(interval="1m", start=start_date, end=end_date)
        stock_price_df.reset_index(inplace=True)
        stock_price_df['stock_code'] = stock_code
        stock_price_df['data_fetched_on'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"{stock_code}_intraday_1min_data_{timestamp}.csv")
        stock_price_df.to_csv(path, index=False)
        print(f"✅ Intraday 1-minute data saved to: {path}")
        return stock_price_df.sort_values(by='Datetime')
    except Exception as e:
        print(f"Error fetching data for {stock_code}: {e}")
        return pd.DataFrame()

# scale data
def scale_data(df, feature_col='Close'):
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_col] = scaler.fit_transform(df[[feature_col]])
    return df_scaled, scaler

# prepare time windows
def prepare_data_for_transformer(df, feature_col='Close', time_steps=60, forecast_horizon=1):
    data = df[feature_col].values
    X, y = [], []
    for i in range(len(data) - time_steps - forecast_horizon + 1):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps:i + time_steps + forecast_horizon])
    return np.array(X), np.array(y)

def train_test_split(X, y, train_ratio=0.8):
    idx = int(len(X) * train_ratio)
    return X[:idx], X[idx:], y[:idx], y[idx:]

def make_tf_dataset(X, y, batch_size=32, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Transformer Block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, x, training=False):
        attn = self.att(x, x)
        x = self.norm1(x + self.drop1(attn, training=training))
        ffn = self.ffn(x)
        return self.norm2(x + self.drop2(ffn, training=training))

# Transformer Model
class TimeSeriesTransformer(Model):
    def __init__(self, d_model, num_heads, ff_dim, num_layers):
        super().__init__()
        self.input_proj = layers.Dense(d_model)
        self.blocks = [TransformerBlock(d_model, num_heads, ff_dim) for _ in range(num_layers)]
        self.out = layers.Dense(1)
        self.d_model = d_model

    def get_pos_encoding(self, seq_len):
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        i = tf.cast(tf.range(self.d_model)[tf.newaxis, :], tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / self.d_model)
        angles = pos * angle_rates
        sines = tf.sin(angles[:, 0::2])
        cosines = tf.cos(angles[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return pos_encoding[tf.newaxis, ...]

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        x = self.input_proj(x)
        x += self.get_pos_encoding(seq_len)
        for block in self.blocks:
            x = block(x, training=training)
        return self.out(x[:, -1, :])

# prediction
def predict_future(model, recent_data, future_steps=30, time_steps=60):
    predictions = []
    seq = np.array(recent_data)
    for _ in range(future_steps):
        inp = seq[-time_steps:].reshape(1, time_steps, 1)
        pred = model.predict(inp, verbose=0)[0][0]
        predictions.append(pred)
        seq = np.append(seq, pred)
    return np.array(predictions)


import matplotlib.pyplot as plt
import re
from PIL import Image
import google.generativeai as genai


def transformer_with_gemini(model: tf.keras.Model,
                            last_sequence: np.ndarray,
                            scaler: MinMaxScaler,
                            history_end: pd.Timestamp,
                            stock_code: str,
                            future_steps: int = 30,
                            time_steps: int = 60,
                            aimodel: str = "gemini-2.0-flash",
                            output_dir: str = "outputs") -> pd.DataFrame:


    preds_norm = predict_future(model, last_sequence, future_steps, time_steps)
    preds = scaler.inverse_transform(preds_norm.reshape(-1, 1)).flatten()

    future_times = pd.date_range(start=history_end + pd.Timedelta(minutes=1),
                                 periods=future_steps,
                                 freq='1min')

    future_df = pd.DataFrame({
        "Datetime": future_times,
        "Close": preds
    })

    os.makedirs(output_dir, exist_ok=True)
    chart_path = os.path.join(
        output_dir,
        f"{stock_code}_transformer_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )

    plt.figure(figsize=(12, 6))
    plt.plot(future_df["Datetime"], future_df["Close"],
             label="Transformer Forecast", linewidth=1.5)
    plt.title(f"{stock_code} – Transformer 30‑min Forecast (1‑min)", fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("Predicted Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"📈 Forecast chart saved to: {chart_path}")

    prompt_text = f"""
You are a professional technical stock analyst.
Based on the attached 1‑minute forecast chart for {stock_code}, analyze the projected movement technically.

Provide:
1. A technical sentiment score from 0 to 1 (0 = strong bearish, 0.5 = neutral, 1 = strong bullish).
2. A short paragraph explaining the reasoning behind the score.

Sentiment score must be the first thing in your answer.
"""

    model_gemini = genai.GenerativeModel(aimodel)
    with open(chart_path, "rb") as img_file:
        image_data = img_file.read()

    response = model_gemini.generate_content(
        [prompt_text, Image.open(chart_path)],
        stream=False
    )
    output = response.text.strip()

    m = re.search(r"\b([01](?:\.\d+)?)\b", output)
    score = float(m.group(1)) if m else None
    if score is None:
        print("❌ Gemini output does not contain a valid sentiment score.")

    result_df = pd.DataFrame([{
        "stock_code": stock_code,
        "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sentiment_score": score,
        "summary": output
    }])

    csv_path = os.path.join(
        output_dir,
        f"{stock_code}_transformer_gemini_sentiment.csv"
    )
    result_df.to_csv(csv_path, index=False)
    print(f"✅ Transformer‑Gemini sentiment saved to: {csv_path}")

    return result_df


# ========== main ==========
if __name__ == "__main__":
    df = fetch_stock_info(stock_code, date_str)
    if df.empty: exit()

    df_scaled, scaler = scale_data(df, 'Close')
    X, y = prepare_data_for_transformer(df_scaled, 'Close', 60, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    train_ds = make_tf_dataset(X_train, y_train)
    test_ds = make_tf_dataset(X_test, y_test, shuffle=False)

    model = TimeSeriesTransformer(d_model=32, num_heads=2, ff_dim=64, num_layers=2)
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_ds, epochs=10, validation_data=test_ds)

    preds = model.predict(X_test)
    preds_inv = scaler.inverse_transform(preds)
    y_test_inv = scaler.inverse_transform(y_test)

    plt.figure()
    plt.title("Test Set Prediction vs Actual")
    plt.plot(y_test_inv, label='Actual')
    plt.plot(preds_inv, label='Predicted')
    plt.legend()
    plt.show()

    # prediction
    last_seq = X_test[-1].reshape(-1)
    future = predict_future(model, last_seq, future_steps=30)
    future_inv = scaler.inverse_transform(future.reshape(-1, 1))

    print("✅ Future 30-minute prediction:")
    print(future_inv.reshape(-1))

    plt.figure()
    plt.title("Future 30-Minute Forecast")
    plt.plot(future_inv, label="Future Prediction")
    plt.legend()
    plt.show()

    transformer_with_gemini(model,
                            last_sequence=last_seq,
                            scaler=scaler,
                            history_end=df['Datetime'].max(),
                            stock_code=stock_code)
