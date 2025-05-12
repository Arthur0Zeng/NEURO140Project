import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

from news_utils import get_filtered_news_sample, analyze_sentiment_with_gemini
from transformer_utils import (fetch_stock_info,
                               scale_data,
    prepare_data_for_transformer,
    train_test_split,
    make_tf_dataset,
    TimeSeriesTransformer,
    transformer_with_gemini,
)
from technical_utils import analyze_technical_sentiment_with_gemini
from reco_utils import get_stock_recommendations, analyze_recommendation_sentiment_with_gemini
from summary_report import summarize_final_sentiment

def generate_and_backtest(stock_code: str,
                          date_str: str,
                          bullish_th: float = 0.6,
                          bearish_th: float = 0.4,
                          out_dir: str = "./outputs") -> dict:
    """
    4 sentiments
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1)

    sample_df = get_filtered_news_sample(stock_code, date_str, n=5)
    analyze_sentiment_with_gemini(sample_df, stock_code, output_dir=out_dir)

    # 2)
    df_price = fetch_stock_info(stock_code, date_str, output_dir=out_dir, passday=1)
    analyze_technical_sentiment_with_gemini(df_price, stock_code, output_dir=out_dir)

    # 3) Transformer
    try:
        df_scaled, scaler = scale_data(df_price, 'Close')
        X, y = prepare_data_for_transformer(df_scaled, 'Close', 60, 1)
        if len(X) < 70:
            raise ValueError("Not enough intraday points")
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        X_train = X_train[..., np.newaxis]
        X_test  = X_test[..., np.newaxis]
        train_ds = make_tf_dataset(X_train, y_train)
        test_ds  = make_tf_dataset(X_test,  y_test, shuffle=False)

        model = TimeSeriesTransformer(d_model=16, num_heads=2, ff_dim=32, num_layers=1)
        model.compile(optimizer='adam', loss='mse')
        model.fit(train_ds, epochs=3, validation_data=test_ds, verbose=0)

        last_seq = X_test[-1] if len(X_test) else X_train[-1]
        transformer_with_gemini(model,
                                last_sequence=last_seq.reshape(-1),
                                scaler=scaler,
                                history_end=df_price['Datetime'].max(),
                                stock_code=stock_code,
                                output_dir=out_dir)
    except Exception as e:
        print(f"⚠️ Transformer sentiment skipped: {e}")

    # 4) 
    rec_df = get_stock_recommendations(stock_code)
    analyze_recommendation_sentiment_with_gemini(rec_df, stock_code, output_dir=out_dir)

    # 5) summarized sentiment
    summary_df = summarize_final_sentiment(stock_code, output_dir=out_dir)
    score = summary_df['final_score'].iloc[0]
    pred_dir = "UP" if score >= bullish_th else "DOWN" if score <= bearish_th else "NEUTRAL"

    # 6) 
    date_dt = datetime.strptime(date_str, "%Y-%m-%d")
    price_df, tries, step_days = pd.DataFrame(), 0, 7
    while len(price_df) < 2 and tries < 3:
        next_day = date_dt + timedelta(days=step_days * (tries + 1))
        price_df = yf.download(stock_code,
                               start=date_dt.strftime("%Y-%m-%d"),
                               end=next_day.strftime("%Y-%m-%d"),
                               progress=False)
        tries += 1
    if len(price_df) < 2:
        print(f"⚠️ still couldn’t fetch two trading days for {stock_code} from {date_str}.")
        return {}

    price_df = price_df.sort_index()
    close_today = float(price_df['Close'].iloc[0])
    close_next  = float(price_df['Close'].iloc[1])
    act_dir = "UP" if close_next > close_today else "DOWN" if close_next < close_today else "NEUTRAL"
    hit = None if pred_dir == "NEUTRAL" else (pred_dir == act_dir)

    # 7) backtest result
    print(f"\n===== Backtest {stock_code} {date_str} =====")
    print(f"Predicted score : {score:.3f}  →  {pred_dir}")
    print(f"Actual movement : {close_today:.2f} → {close_next:.2f}  →  {act_dir}")
    if hit is True:
        print("✅ Direction matched.")
    elif hit is False:
        print("❌ Direction missed.")
    else:
        print("⚪ Neutral prediction; not counted.")

    return {
        "stock_code": stock_code,
        "date": date_str,
        "score": round(score, 3),
        "predicted_dir": pred_dir,
        "actual_dir": act_dir,
        "hit": hit
    }



