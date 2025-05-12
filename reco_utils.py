import yfinance as yf
import pandas as pd
import os
import pandas as pd
import os
import re
import unicodedata
from datetime import datetime
import google.generativeai as genai
from news_utils import clean_text
genai.configure(api_key="AIzaSyD_K9HocP_BzDiGuxkuzGeJXoXzlVb3gbQ")

def parse_period_to_date(period_str):
    """
    Convert yfinance 'period' format (e.g., '0', '-1m', '-2m') to actual dates.
    """
    if period_str == '0':
        return datetime.today().date()

    match = re.match(r"-(\d+)([mdy])", period_str)
    if not match:
        return None

    value, unit = int(match.group(1)), match.group(2)
    if unit == 'd':
        delta = pd.Timedelta(days=value)
    elif unit == 'm':
        delta = pd.DateOffset(months=value)
    elif unit == 'y':
        delta = pd.DateOffset(years=value)
    else:
        return None

    return (datetime.today() - delta).date()

def get_stock_recommendations(ticker_symbol: str) -> pd.DataFrame:
    """
    Fetches analyst recommendations for a given stock symbol,
    converts 'period' to actual date, and saves them as a CSV file.

    Parameters:
        ticker_symbol (str): Stock ticker (e.g., "AAPL", "TSLA")

    Returns:
        pd.DataFrame: Analyst recommendations with actual dates.
    """
    ticker = yf.Ticker(ticker_symbol)
    recommendations = ticker.recommendations

    if recommendations is None or recommendations.empty:
        print(f"No analyst recommendations found for {ticker_symbol}.")
        return pd.DataFrame()

    recommendations = recommendations.reset_index()
    recommendations['Date'] = recommendations['period'].apply(parse_period_to_date)

    os.makedirs("outputs", exist_ok=True)
    filename = f"{ticker_symbol.upper()}_analyst_recommendations.csv"
    filepath = os.path.join("outputs", filename)
    recommendations.to_csv(filepath, index=False)
    print(f"✅ Saved processed recommendations to {filepath}")

    return recommendations

def analyze_recommendation_sentiment_with_gemini(recommendation_df: pd.DataFrame, stock_code: str,
                                                 aimodel="gemini-2.0-flash", output_dir="outputs"):
    if recommendation_df.empty:
        print(f"❌ No recommendations to analyze for {stock_code}")
        return None

    recommendation_df = recommendation_df.reset_index(drop=True)

    rec_text = "\n\n".join(
        [f"Date: {row['Date']}\nStrong Buy: {row.get('strongBuy', '')}, Buy: {row.get('buy', '')}, Hold: {row.get('hold', '')}, Sell: {row.get('sell', '')}, Strong Sell: {row.get('strongSell', '')}"
         for _, row in recommendation_df.iterrows()]
    )

    raw_prompt = f"""
You are a financial analyst. Below is a chronological list of analyst recommendations for stock {stock_code}.
Please analyze the trend of these recommendations to determine the overall trading sentiment.
Give a score from 0 to 1 where:
- 0 means strong sell sentiment (many downgrades or sell ratings),
- 0.5 means neutral sentiment (mostly holds, mixed actions),
- 1 means strong buy sentiment (mostly upgrades or buy ratings).

Also provide a short summary explaining your score. I want you report sentiment score first before you report others.
Sentiment score must be the first thing in your answer

Recommendations:
{rec_text}
"""

    prompt = clean_text(raw_prompt)
    model = genai.GenerativeModel(aimodel)
    response = model.generate_content(prompt)
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
        'summary': output  # Entire unmodified Gemini output
    }])

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{stock_code}_recommendation_sentiment.csv"
    result.to_csv(filename, index=False)
    print(f"✅ Recommendation sentiment saved to: {filename}")
    return result