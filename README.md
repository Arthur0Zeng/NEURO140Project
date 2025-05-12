
# Stock Sentiment Back‑testing Pipeline

Run **following function** and get:
1. Real‑time Google News scraping → Gemini news sentiment  
2. Intraday 1‑minute chart → Gemini technical sentiment  
3. Mini‑Transformer forecast → Gemini forecast sentiment  
4. Analyst recommendation table → Gemini recommendation sentiment  
5. Gemini summary integrating the four sources  
6. Back‑test: compare summary score vs. next‑day price movement


**The software is currently in ver1.1 beta, so it has lots of bugs**

```terminal
######optional
cd stock_sentiment_project
python -m venv venv 
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
######

export GEMINI_API_KEY="YOURKEY"  

python - <<'PY'
from news_utils import get_filtered_news_sample, analyze_sentiment_with_gemini
df_sample = get_filtered_news_sample("NVDA", "2025-04-29", n=5)
result_df = analyze_sentiment_with_gemini(df_sample, "NVDA")
from backtest import generate_and_backtest
generate_and_backtest("NVDA", "2025-04-29")
PY
```

**Set your Gemini API key once in the environment (`export GEMINI_API_KEY=...`).**

### Requirements
See `requirements.txt`.
# NEURO140Project
