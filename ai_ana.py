import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from textblob import TextBlob
from collections import Counter
import re
import feedparser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



from transformers import pipeline

from utils import classifier,tokenizer,model

# tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
# model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
# classifier = pipeline('zero-shot-classification', model="facebook/bart-large-mnli")


business_areas = {
    "Product Development": ["innovation", "research", "development", "product improvement", "AI-driven design"],
    "Customer Service": ["customer support", "chatbots", "personalization", "service automation"],
    "Operations Efficiency": ["automation", "process optimization", "efficiency", "cost reduction"],
    "Marketing and Sales": ["marketing", "advertising", "sales", "customer targeting", "AI-driven campaigns"],
    "Risk Management": ["risk assessment", "fraud detection", "security", "compliance", "predictive analytics"]
}
ai_impact = ""
def fetch_news(company):
    url = f"https://news.google.com/rss/search?q={company}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:1]:
        title = entry.title
        link = entry.link
        try:
            response = requests.get(link, timeout=5)
            article_text = response.text
        except:
            article_text = ""
        articles.append(title)
    return articles

def analyze_impact(articles, business_areas):
    impact_scores = {area: 0 for area in business_areas}
    for article in articles:
        for area, topics in business_areas.items():
            results = classifier(article, topics)
            score = np.mean([results['scores'][i] for i in range(len(results['labels'])) if results['labels'][i] in topics])
            impact_scores[area] += score
    return impact_scores

def normalize_scores(scores):
    max_score = max(scores.values())
    if max_score == 0:
        return {k: 1 for k in scores}
    return {k: int((v / max_score) * 10) for k, v in scores.items()}

def generate_ai_impact_analysis(company):
    articles = fetch_news(company)
    raw_scores = analyze_impact(articles, business_areas)
    normalized_scores = normalize_scores(raw_scores)
    return normalized_scores
# Set page config



def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    return stock.history(start=start_date, end=end_date)

def prepare_data_for_prophet(df):
    df_prophet = df.reset_index()[['Date', 'Close']]
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)
    return df_prophet

def train_and_predict(df, periods):
    model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

def get_financial_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.balance_sheet, stock.financials, stock.cashflow

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.detach().numpy()

# Function to get sentiment analysis of news articles for a given company
def get_sentiment_analysis(company):
    url = f"https://news.google.com/rss/search?q={company}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    
    sentiments = []
    for entry in feed.entries[:30]:
        title = entry.title
        print(title)  # Only use title for sentiment analysis
        sentiment_probs = analyze_sentiment(title)
        
        sentiments.append({
            "positive": sentiment_probs[0][1],
            "neutral": sentiment_probs[0][2],
            "negative": sentiment_probs[0][0]
        })
    
    df = pd.DataFrame(sentiments)
    sentiment_distribution = {
        "positive": np.mean(df['positive']),
        "neutral": np.mean(df['neutral']),
        "negative": np.mean(df['negative'])
    }
    
    avg_sentiment = np.mean(df[['positive', 'neutral', 'negative']].values, axis=0)
    
    return sentiment_distribution, avg_sentiment

def get_sustainability_data(ticker):
    stock = yf.Ticker(ticker)
    sustainability = stock.sustainability
    if sustainability is not None and not sustainability.empty:
        if isinstance(sustainability, pd.Series):
            sustainability = sustainability.to_frame().T
        sustainability = sustainability.reset_index()
        sustainability.columns = ['Metric', 'Value']
    return sustainability



def scrape_news(company, industry):
    url = f"https://www.google.com/search?q={company}+{industry}+future+trends&tbm=nws"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    news_items = soup.find_all('div', class_='g')
    return [item.find('h3').text for item in news_items if item.find('h3')]

def generate_future_trends(company):
    ticker = yf.Ticker(company)
    company_info = ticker.info
    industry = company_info.get('industry', 'Unknown')
    sector = company_info.get('sector', 'Unknown')

    news_headlines = scrape_news(company, industry)
    trend_words = ' '.join(news_headlines).lower()
    trend_words = re.findall(r'\w+', trend_words)
    common_words = set(['the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'is', 'are', 'will', 'be'])
    trend_words = [word for word in trend_words if word not in common_words and len(word) > 3]
    top_trends = [word for word, _ in Counter(trend_words).most_common(10)]

    sentiments = [TextBlob(headline).sentiment.polarity for headline in news_headlines]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    financial_data = ticker.history(start=start_date, end=end_date)
    price_change = (financial_data['Close'].iloc[-1] - financial_data['Close'].iloc[0]) / financial_data['Close'].iloc[0]
    volatility = financial_data['Close'].pct_change().std() * (252 ** 0.5)

    market_cap = company_info.get('marketCap', 'N/A')
    revenue = company_info.get('totalRevenue', 'N/A')
    employees = company_info.get('fullTimeEmployees', 'N/A')

    insights = []
    if avg_sentiment > 0.2:
        insights.append("Positive market sentiment indicates potential growth opportunities.")
    elif avg_sentiment < -0.2:
        insights.append("Negative market sentiment suggests caution and potential challenges ahead.")
    
    if price_change > 0.1:
        insights.append("Strong stock performance over the past year indicates positive market reception.")
    elif price_change < -0.1:
        insights.append("Weak stock performance over the past year suggests the need for strategic improvements.")
    
    if volatility > 0.3:
        insights.append("High volatility indicates potential for both significant gains and losses.")
    else:
        insights.append("Relatively low volatility suggests stability but potentially limited growth prospects.")

    return {
        'company': company,
        'industry': industry,
        'sector': sector,
        'market_cap': market_cap,
        'revenue': revenue,
        'employees': employees,
        'top_trends': top_trends,
        'sentiment': avg_sentiment,
        'price_change': price_change,
        'volatility': volatility,
        'insights': insights
    }

def generate_narrative_summary(company, current_price, predicted_price, sentiment_scores, sustainability_score, volatility, returns, sharpe_ratio, ai_impact):
    narrative = (
        f"Company Analysis Summary for {company}:\n\n"
        
        f"1. Stock Performance:\n"
        f"   Current Price: ${current_price:.2f}\n"
        f"   Predicted Price (5 years): ${predicted_price:.2f}\n"
        f"   This indicates a {'positive' if predicted_price > current_price else 'negative'} growth outlook "
        f"with an expected {'increase' if predicted_price > current_price else 'decrease'} "
        f"of {abs(predicted_price - current_price) / current_price:.2%} over the next five years.\n\n"
        
        f"2. Risk and Returns:\n"
        f"   Annualized Volatility: {volatility:.2%}\n"
        f"   Total Returns (5 years): {returns:.2%}\n"
        f"   Sharpe Ratio: {sharpe_ratio:.2f}\n"
        f"   The volatility indicates {'high' if volatility > 0.3 else 'moderate' if volatility > 0.15 else 'low'} price fluctuations. "
        f"The Sharpe ratio suggests {'excellent' if sharpe_ratio > 1 else 'good' if sharpe_ratio > 0.5 else 'poor'} risk-adjusted returns.\n\n"
        
        f"3. Market Sentiment:\n"
        f"   Positive: {sentiment_scores['positive']*100:.1f}%\n"
        f"   Neutral: {sentiment_scores['neutral']*100:.1f}%\n"
        f"   Negative: {sentiment_scores['negative']*100:.1f}%\n"
        f"   The market sentiment is predominantly {'positive' if sentiment_scores['positive'] > max(sentiment_scores['neutral'], sentiment_scores['negative']) else 'neutral' if sentiment_scores['neutral'] > max(sentiment_scores['positive'], sentiment_scores['negative']) else 'negative'}.\n\n"
    )
    
    
    
    narrative += (
        f"5. AI Impact Analysis:\n"
        f"   Highest impact area: {max(ai_impact)} (score: {max(ai_impact.values())}/10)\n"
        f"   {'This suggests significant AI-driven opportunities or challenges in this area.' if max(ai_impact.values()) > 7 else 'Moderate AI influence is expected in this area.' if max(ai_impact.values()) > 4 else 'Limited AI impact is anticipated in this area.'}\n\n"
        
        "Conclusion:\n"
        f"Based on these factors, {company} {'appears to have strong growth potential' if predicted_price > current_price and sentiment_scores['positive'] > 0.5 and sharpe_ratio > 1 else 'shows mixed indicators' if predicted_price > current_price or sentiment_scores['positive'] > 0.5 or sharpe_ratio > 0.5 else 'faces significant challenges'}. "
        "Investors should consider these quantitative projections, qualitative market sentiments, "
        "sustainability measures, and potential AI impacts in their decision-making process."
    )
    return narrative
def main():
    st.title("Comprehensive Company Analysis Dashboard")
   
    company = st.text_input("Enter company ticker symbol (e.g., AAPL for Apple):")

    if company:
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)
        df = get_stock_data(company, start_date, end_date)
        if not df.empty:
            df_prophet = prepare_data_for_prophet(df)
            periods = 365 * 5
            model, forecast = train_and_predict(df_prophet, periods)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_prediction = plot_plotly(model, forecast)
                fig_prediction.update_layout(title=f"{company} Stock Price Prediction (5 Years)", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_prediction, use_container_width=True)
            
            with col2:
                current_price = df['Close'].iloc[-1]
                predicted_price = forecast['yhat'].iloc[-1]
                st.metric("Current Price", f"${current_price:.2f}")
                st.metric("Predicted Price (in 5 years)", f"${predicted_price:.2f}")
                
                volatility = df['Close'].pct_change().std() * (252 ** 0.5)
                returns = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
                sharpe_ratio = (returns / volatility) * (252 ** 0.5)
                
                st.metric("Annualized Volatility", f"{volatility:.2%}")
                st.metric("Total Returns (5 years)", f"{returns:.2%}")
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            st.subheader("Interactive 3D Stock Analysis")
            # Get the column names for the dropdown options
            columns = df.columns.tolist()
            columns.insert(0, 'Date')  # Add 'Date' as an option

            # User inputs with default values
            x_param = st.selectbox('Select X-axis parameter:', columns, index=columns.index('Date'))
            y_param = st.selectbox('Select Y-axis parameter:', columns, index=columns.index('Close'))
            z_param = st.selectbox('Select Z-axis parameter:', columns, index=columns.index('Low'))

            # Handle 'Date' specially since it's the index
            x_data = df.index if x_param == 'Date' else df[x_param]
            y_data = df[y_param]
            z_data = df[z_param]

            # Create the 3D scatter plot
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=x_data,
                y=y_data,
                z=z_data,
                mode='markers',
                marker=dict(
                    size=5,
                    color=y_data,  # Color by the selected y-axis parameter
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=[f"Date: {date}<br>Close: ${close:.2f}<br>Volume: {volume}" for date, close, volume in zip(df.index, df['Close'], df['Volume'])],
                hoverinfo='text'
            )])
            fig_3d.update_layout(scene=dict(xaxis_title=x_param, yaxis_title=y_param, zaxis_title=z_param))

            st.plotly_chart(fig_3d)
                        
            st.subheader("Sentiment Analysis")
            sentiment_distribution, avg_sentiment = get_sentiment_analysis(company)

            fig_sentiment = go.Figure(data=[go.Pie(labels=list(sentiment_distribution.keys()), 
                                                   values=list(sentiment_distribution.values()))])
            fig_sentiment.update_layout(title="Sentiment Distribution")
            sentiment_distribution, avg_sentiment = get_sentiment_analysis(company)
    
            st.subheader("Sentiment Distribution")
            st.write(f"Positive: {sentiment_distribution['positive']:.2%}")
            st.write(f"Neutral: {sentiment_distribution['neutral']:.2%}")
            st.write(f"Negative: {sentiment_distribution['negative']:.2%}")
            
            st.subheader("Average Sentiment Scores")
            st.write(f"Positive: {avg_sentiment[0]:.2f}")
            st.write(f"Neutral: {avg_sentiment[1]:.2f}")
            st.write(f"Negative: {avg_sentiment[2]:.2f}")

            st.write("Sentiment scores range from -1 (very negative) to 1 (very positive).")
            st.write("0 indicates a neutral sentiment.")
            sentiment_score = {}
            sentiment_score['positive'] = sentiment_distribution['positive']
            sentiment_score['neutral'] = sentiment_distribution['neutral']
            sentiment_score['negative'] = sentiment_distribution['negative']
            # st.subheader("Sustainability Analysis")
            sustainability = 0
            sustainability_data = get_sustainability_data(company)
            
            if sustainability_data is not None and not sustainability_data.empty:
                esg_scores = ['totalEsg', 'environmentScore', 'socialScore', 'governanceScore']
                sustainability_score = None
                used_metric = None
                
                for metric in esg_scores:
                    score_row = sustainability_data[sustainability_data['Metric'] == metric]
                    if not score_row.empty:
                        sustainability_score = score_row['Value'].values[0]
                        used_metric = metric
                        sustainability = used_metric
                        break

                if sustainability_score is not None:
                    # st.metric(f"ESG Score ({used_metric})", f"{sustainability_score:.2f}")
                    pass
                else:
                    st.write("No specific ESG score available for this company.")

                numeric_data = sustainability_data[pd.to_numeric(sustainability_data['Value'], errors='coerce').notnull()]
            #     if not numeric_data.empty:
            #         fig_sustainability = go.Figure(data=[go.Bar(x=numeric_data['Metric'], y=numeric_data['Value'])])
            #         fig_sustainability.update_layout(title="ESG Scores Breakdown", xaxis_tickangle=-45)
            #         st.plotly_chart(fig_sustainability, use_container_width=True)
            #     else:
            #         st.write("No numeric sustainability data available for charting.")

            #     st.write("Raw Sustainability Data:")
            #     st.dataframe(sustainability_data)
            # else:
            #     st.write("No sustainability data available for this company.")
            #     sustainability_score = None
            
            st.title("AI Impact Analysis")
            impact_analysis = generate_ai_impact_analysis(company)
            st.write(f"AI Impact Analysis for {company}:")
            for area, score in impact_analysis.items():
                st.write(f"{area}: {score}/10")
            
            st.subheader("Future Trends Analysis")
            with st.spinner("Analyzing future trends... This may take a moment."):
                trends_data = generate_future_trends(company)
            
            if trends_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"*Industry:* {trends_data['industry']}")
                    st.write(f"*Sector:* {trends_data['sector']}")
                    st.write(f"*Market Cap:* ${trends_data['market_cap']:,}")
                    st.write(f"*Revenue:* ${trends_data['revenue']:,}")
                    st.write(f"*Employees:* {trends_data['employees']:,}")
                
                with col2:
                    st.write("*Top Trends:*")
                    for trend in trends_data['top_trends'][:5]:
                        st.write(f"- {trend}")
                
                st.write(f"*Average Sentiment:* {trends_data['sentiment']:.2f}")
                st.write(f"*Annual Price Change:* {trends_data['price_change']:.2%}")
                st.write(f"*Annualized Volatility:* {trends_data['volatility']:.2%}")
                
                st.write("*Insights:*")
                for insight in trends_data['insights']:
                    st.write(f"- {insight}")
            else:
                st.error("Failed to generate future trends analysis.")

            narrative_summary = generate_narrative_summary(company, current_price, predicted_price, sentiment_score, sustainability_score, volatility, returns, sharpe_ratio, impact_analysis)
            st.subheader("Narrative Summary")
            st.info(narrative_summary)

            st.subheader("Financial Statements")
            balance_sheet, income_statement, cash_flow = get_financial_data(company)

            tab1, tab2, tab3 = st.tabs(["Balance Sheet", "Income Statement", "Cash Flow"])

            with tab1:
                st.dataframe(balance_sheet)

            with tab2:
                st.dataframe(income_statement)

            with tab3:
                st.dataframe(cash_flow)
            

if __name__ == "__main__":
    main()
