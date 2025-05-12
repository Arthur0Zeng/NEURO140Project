# ------------------- summary_report.py -------------------
import pandas as pd
import os
import re
import google.generativeai as genai
from datetime import datetime
from news_utils import clean_text

# --------------------------------------------------------
# --------------------------------------------------------
def summarize_final_sentiment(stock_code: str,
                               aimodel="gemini-2.0-flash",
                               output_dir="outputs"):
    # ---------- Read reports ----------
    rec_path = os.path.join("./outputs",f"{stock_code}_recommendation_sentiment.csv")
    tech_path = os.path.join("./outputs",f"{stock_code}_technical_sentiment.csv")
    news_path = os.path.join("./outputs",f"{stock_code}_sentiment.csv")
    trans_path = os.path.join("./outputs",f"{stock_code}_transformer_gemini_sentiment.csv")
    try:
        df_rec  = pd.read_csv(rec_path)
        df_tech = pd.read_csv(tech_path)
        df_news = pd.read_csv(news_path)
        df_trans = pd.read_csv(trans_path)            # ② 新增
    except Exception as e:
        print(f"❌ Failed to load sentiment files: {e}")
        return None

    # ---------- 抽取分数与摘要 ----------
    rec_score,  rec_summary  = df_rec['sentiment_score'][0],  df_rec['summary'][0]
    tech_score, tech_summary = df_tech['sentiment_score'][0], df_tech['summary'][0]
    news_score, news_summary = df_news['sentiment_score'][0], df_news['summary'][0]
    trans_score, trans_summary = df_trans['sentiment_score'][0], df_trans['summary'][0]   # ③ 新增

    # ---------- construct Gemini prompt ----------
    prompt = f"""
You are a financial analyst assistant.
Below are sentiment analysis results from four sources for stock {stock_code}:

1. Analyst Recommendations:
Score: {rec_score}
Summary: {clean_text(rec_summary)}

2. Technical Chart Analysis:
Score: {tech_score}
Summary: {clean_text(tech_summary)}

3. News Sentiment:
Score: {news_score}
Summary: {clean_text(news_summary)}

4. Transformer Model Forecast Sentiment:
Score: {trans_score}
Summary: {clean_text(trans_summary)}

Please integrate these scores and summaries to give:
- A final **buying sentiment score** (between 0 and 1)
- A final **concise recommendation**: one of ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
- A **short explanation** (3‑5 lines) why this recommendation is made

Start with the final score first, then the recommendation, and then the explanation.
"""

    model = genai.GenerativeModel(aimodel)
    output = model.generate_content(prompt).text.strip()

    match = re.search(r"([01](?:\.\d+)?)", output)
    sentiment_score = float(match.group(1)) if match else None

    result = pd.DataFrame([{
        'stock_code': stock_code,
        'generated_on': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'final_score': sentiment_score,
        'gemini_output': output
    }])

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir,
                            f"{stock_code}_final_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    result.to_csv(filename, index=False)
    print(f"✅ Final summary saved to: {filename}")
    return result

summarize_final_sentiment(
    stock_code="NVDA"
)
