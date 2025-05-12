import matplotlib.pyplot as plt
import re
from PIL import Image
import pandas as pd
import os
import google.generativeai as genai
from datetime import datetime, timedelta
genai.configure(api_key="AIzaSyD_K9HocP_BzDiGuxkuzGeJXoXzlVb3gbQ")

def analyze_technical_sentiment_with_gemini(stock_df: pd.DataFrame, stock_code: str,
                                            aimodel="gemini-2.0-flash", output_dir="outputs"):
    if stock_df.empty:
        print(f"❌ No stock data to analyze for {stock_code}")
        return None

    # Image
    fig, ax = plt.subplots(figsize=(12, 6))
    stock_df['Datetime'] = pd.to_datetime(stock_df['Datetime'])
    stock_df = stock_df.sort_values('Datetime')

    ax.plot(stock_df['Datetime'], stock_df['Close'], label='Close Price', linewidth=1.5)
    ax.set_title(f"{stock_code} Intraday Price (1-min)", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"{stock_code}_tech_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    print(f"📊 Chart saved to: {image_path}")

    # Prompt
    prompt_text = f"""
You are a professional technical stock analyst.
Based on the attached 1-minute intraday stock chart for {stock_code}, analyze the chart technically. The straight line in the plot means market was closed.

Provide:
1. A technical sentiment score from 0 to 1:
    - 0 means strong bearish (e.g. downtrend, breakdown patterns),
    - 0.5 means neutral (sideways, consolidation),
    - 1 means strong bullish (uptrend, breakout signals).
2. A short paragraph explaining the reasoning based on technical patterns, trends, and potential signals.

Sentiment score must be the first thing in your answer.
"""

    # Image + Prompt
    model = genai.GenerativeModel(aimodel)
    with open(image_path, "rb") as f:
        image_data = f.read()

    response = model.generate_content(
        [prompt_text, Image.open(image_path)],
        stream=False
    )
    output = response.text.strip()

    match = re.search(r"\b([01](?:\.\d+)?)\b", output)
    if match:
        sentiment_score = float(match.group(1))
    else:
        print("❌ Gemini output does not contain a valid score.")
        sentiment_score = None

    result = pd.DataFrame([{
        'stock_code': stock_code,
        'generated_on': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'sentiment_score': sentiment_score,
        'summary': output
    }])

    filename = os.path.join(output_dir, f"{stock_code}_technical_sentiment.csv")
    result.to_csv(filename, index=False)
    print(f"✅ Technical sentiment saved to: {filename}")
    return result
