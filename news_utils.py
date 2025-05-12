def clean_text(text):
    import unicodedata
    if isinstance(text, str):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return str(text)

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import re
import unicodedata
from datetime import datetime
from newspaper import Article
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import google.generativeai as genai

# Set your Gemini API key here
genai.configure(api_key="AIzaSyD_K9HocP_BzDiGuxkuzGeJXoXzlVb3gbQ")

# ----------- News Scraping and Processing ----------- #

def new_scraper(stock_code):
    keywords = stock_code
    url = f"https://news.google.com/rss/search?q={keywords.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, features='xml')
    items = soup.find_all('item')
    news_list = []
    for item in items:
        news = {
            'title': item.title.text,
            'link': item.link.text,
            'pubDate': item.pubDate.text
        }
        news_list.append(news)
    return news_list

def news_update(stock_code, csv_path='google_news.csv'):
    new_data = new_scraper(stock_code)
    try:
        old_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        old_df = pd.DataFrame(columns=['title', 'link', 'pubDate'])
    new_df = pd.DataFrame(new_data)
    combined_df = pd.concat([old_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(subset='link', inplace=True)
    combined_df.to_csv(csv_path, index=False)
    print(f"News updated. Total articles saved: {len(combined_df)}")

def get_real_url(encoded_url, delay=5):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(options=options)
    real_url = None
    try:
        driver.get(encoded_url)
        time.sleep(delay)
        real_url = driver.current_url
        if "news.google.com" in real_url:
            real_url = None
    except Exception as e:
        print(f"❌ Selenium error: {e}")
    finally:
        driver.quit()
    return real_url

def extract_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception as e:
        print(f"❌ Failed to extract content from {url}: {e}")
        return None

def update_news_dataframe(csv_path='google_news.csv', output_csv='google_news_with_content.csv', delay=5):
    df_new = pd.read_csv(csv_path)
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
    else:
        df_existing = pd.DataFrame(columns=list(df_new.columns) + ['content'])
    df_combined = pd.merge(df_new, df_existing[['link', 'content']], on='link', how='left')
    for idx, row in df_combined.iterrows():
        if pd.notnull(row.get('content')):
            print(f"[{idx+1}/{len(df_combined)}] ✅ Skipping processed.")
            continue
        print(f"[{idx+1}/{len(df_combined)}] 🔗 Fetching: {row['link']}")
        real_url = get_real_url(row['link'], delay=delay)
        if real_url:
            print(f"→ Resolved to: {real_url}")
            content = extract_article_content(real_url) or "FAILED"
        else:
            content = "FAILED"
        df_combined.at[idx, 'content'] = content
    df_combined.to_csv(output_csv, index=False)
    print(f"✅ Updated CSV saved to: {output_csv}")
    return df_combined

def get_filtered_news_sample(stock_code, date_str, n=5, delay=5):
    csv_path = f"{stock_code}_google_news.csv"
    output_csv = f"{stock_code}_google_news_with_content.csv"
    news_update(stock_code, csv_path=csv_path)
    df_combined = update_news_dataframe(csv_path=csv_path, output_csv=output_csv, delay=delay)
    input_date = datetime.strptime(date_str, "%Y-%m-%d")
    df_combined['parsed_date'] = pd.to_datetime(df_combined['pubDate'], errors='coerce')
    filtered_df = df_combined[
        (df_combined['parsed_date'] >= input_date) &
        (df_combined['content'].notnull()) &
        (df_combined['content'] != "FAILED") &
        (df_combined['content'].str.strip() != "")
    ]
    if len(filtered_df) < n:
        print(f"⚠️ Only {len(filtered_df)} articles available.")
        sampled_df = filtered_df
    else:
        sampled_df = filtered_df.sample(n=n, random_state=42)
    sampled_df = sampled_df.drop(columns=['parsed_date'])
    return sampled_df

# ----------- Sentiment Analysis with Gemini ----------- #

def clean_text(text):
    if isinstance(text, str):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return str(text)

def analyze_sentiment_with_gemini(sampled_df: pd.DataFrame, stock_code: str, aimodel="gemini-2.0-flash", output_dir="outputs"):
    articles_text = "\n\n".join(
        [f"Title: {clean_text(row['title'])}\nDate: {row['pubDate']}\nContent: {clean_text(row['content'])}"
         for _, row in sampled_df.iterrows()]
    )
    raw_prompt = f"""
You are a financial analyst. Based on the following news articles related to the stock {stock_code},
please assess the overall trading sentiment for this stock. Provide a single number between 0 and 1, where:
- 0 means strong sell,
- 0.5 means neutral,
- 1 means strong buy.

Only return the number and your summarize regarding to these news.

Articles:
{articles_text}
"""
    prompt = clean_text(raw_prompt)
    model = genai.GenerativeModel(aimodel)
    response = model.generate_content(prompt)
    output = response.text.strip()
    match = re.search(r"([01](?:\.\d+)?)", output)
    if match:
        sentiment_score = float(match.group(1))
        summary = output.replace(match.group(0), '').strip()
    else:
        print("❌ Gemini output does not contain a valid score.")
        sentiment_score = None
        summary = output
    result = pd.DataFrame([{
        'stock_code': stock_code,
        'min_date': pd.to_datetime(sampled_df['pubDate']).min().strftime("%Y-%m-%d"),
        'max_date': pd.to_datetime(sampled_df['pubDate']).max().strftime("%Y-%m-%d"),
        'generated_on': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'sentiment_score': sentiment_score,
        'summary': summary
    }])
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{stock_code}_sentiment.csv"
    result.to_csv(filename, index=False)
    print(f"✅ Sentiment saved to: {filename}")
    return result