from flask import Flask, render_template, request, redirect, url_for, send_file
import allocate_capital
import get_rankings
import pandas as pd
import plotly
import json
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
from transformers import pipeline
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
# from utils import classifier



app = Flask(__name__)
app.config['SECRET_KEY'] = 'hwedfbuyhedbfwe3484747erh'
app.debug = True

#Home page:
@app.route('/')
def index():
    return render_template('index.html')

#Allocation page getting user input to create portfolio:
@app.route('/allocation', methods=['GET', 'POST'])
def allocation():
    if request.method == 'POST':
        return redirect(url_for("portfolio"))
    else:
        return render_template('allocation.html')

#Allocation page giving the portfolio from user input:
@app.route("/allocation/portfolio", methods=['GET', 'POST'])
def portfolio():
    #Get user input data from form:
    e_scr = request.form["E"]                         #Environmental
    s_scr = request.form["S"]                         #Social
    g_scr = request.form["G"]                         #Governance
    risk = float(request.form["risk"])                #Risk-aversion
    del_sectors = request.form.getlist("sectors")     #Sectors to delete
    del_symbs = request.form.get("symb").split(',')   #Stocks to delete

    #Get portfolio charts and metrics with user inputs from the allocation algo:
    metrics, fig_stocks, fig_sectors = allocate_capital.get_portfolio(e_scr,
                                                                      s_scr,
                                                                      g_scr,
                                                                      risk,
                                                                      del_sectors,
                                                                      del_symbs)
    #Use JSON and Plotly to display charts on the webpage:
    port_stocks = json.dumps(fig_stocks, cls=plotly.utils.PlotlyJSONEncoder)
    port_sectors = json.dumps(fig_sectors, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('portfolio.html', graph_stocks=port_stocks,
                                             graph_sectors = port_sectors,
                                             e_scr=metrics['E_scr'],
                                             s_scr=metrics['S_scr'],
                                             g_scr=metrics['G_scr'],
                                             esg_scr=metrics['ESG_scr'],
                                             ret=metrics['Ret'],
                                             vol=metrics['Vol'],
                                             beta=metrics['beta'])

#Route to allow downloading option of CSV file of allocations:
@app.route("/download")
def download_csv():
    f = '\static\data\portfolio.csv'
    return send_file(f, as_attachment=True)

#Ratings search page:
@app.route("/ratings", methods=['GET', 'POST'])
def ratings():
    #Read in stock symbols and their ESG data:
    data_path = ".\static\data\stock_data.csv"
    data = pd.read_csv(data_path, index_col='ticker')

    #If user submits a stock symbol, show that new page with the ratings:
    if request.method == "POST":
        stock = request.form['symbol-rating']
        return redirect(url_for('stock_ratings', symbol=stock))

    #If user simply requests to see this page, show it instead:
    else:
        return render_template('ratings.html', data=data)

#Ratings page for each symbol searched above:
@app.route("/ratings/<symbol>")
def stock_ratings(symbol):
    #Get ESG rankings and stock info:
    symbol_info = get_rankings.get_company(symbol)

    return render_template('stock_ratings.html', symb_info=symbol_info, stock=symbol)

#Main methodologies page to see allocation and rating methods:
@app.route("/methodologies")
def methodologies():
    return render_template('methodologies.html')

#Methodology page for allocation model:
@app.route("/methodologies/allocation")
def allocation_methodology():
    return render_template('alloc_method.html')

#Methodology page for scoring firms on ESG criteria:
@app.route("/methodologies/esg-scores")
def scoring_methodology():
    return render_template('scoring_method.html')

#Methodology page for technology tools used:
@app.route("/methodologies/tech-used")
def tech_methodology():
    return render_template('tech_method.html')

#Contact me page:
@app.route("/contact")
def contact_me():
    return render_template('contact.html')




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

# def fetch_news(company):
#     url = f"https://news.google.com/rss/search?q={company}&hl=en-US&gl=US&ceid=US:en"
#     feed = feedparser.parse(url)
#     articles = []
#     for entry in feed.entries[:1]:
#         title = entry.title
#         link = entry.link
#         try:
#             response = requests.get(link, timeout=5)
#             article_text = response.text
#         except:
#             article_text = ""
#         articles.append(title)
#     return articles

# def analyze_impact(articles, business_areas):
#     impact_scores = {area: 0 for area in business_areas}
#     for article in articles:
#         for area, topics in business_areas.items():
#             results = classifier(article, topics)
#             score = np.mean([results['scores'][i] for i in range(len(results['labels'])) if results['labels'][i] in topics])
#             impact_scores[area] += score
#     return impact_scores

# def normalize_scores(scores):
#     max_score = max(scores.values())
#     if max_score == 0:
#         return {k: 1 for k in scores}
#     return {k: int((v / max_score) * 10) for k, v in scores.items()}

# def generate_ai_impact_analysis(company):
#     articles = fetch_news(company)
#     raw_scores = analyze_impact(articles, business_areas)
#     normalized_scores = normalize_scores(raw_scores)
#     return normalized_scores

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

# def analyze_sentiment(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     outputs = model(**inputs)
#     probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     return probs.detach().numpy()

# def get_sentiment_analysis(company):
#     url = f"https://news.google.com/rss/search?q={company}&hl=en-US&gl=US&ceid=US:en"
#     feed = feedparser.parse(url)
    
#     sentiments = []
#     for entry in feed.entries[:30]:
#         title = entry.title
#         sentiment_probs = analyze_sentiment(title)
        
#         sentiments.append({
#             "positive": sentiment_probs[0][1],
#             "neutral": sentiment_probs[0][2],
#             "negative": sentiment_probs[0][0]
#         })
    
#     df = pd.DataFrame(sentiments)
#     sentiment_distribution = {
#         "positive": np.mean(df['positive']),
#         "neutral": np.mean(df['neutral']),
#         "negative": np.mean(df['negative'])
#     }
    
#     avg_sentiment = np.mean(df[['positive', 'neutral', 'negative']].values, axis=0)
    
#     return sentiment_distribution, avg_sentiment

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


def calculate_statistics(df):
    current_price = df['Close'].iloc[-1]
    returns = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    volatility = df['Close'].pct_change().std() * (252 ** 0.5)
    sharpe_ratio = (returns / volatility) * (252 ** 0.5)
    return current_price, returns, volatility, sharpe_ratio


def prepare_data_for_plotting(df):
    df.reset_index(inplace=True)
    return df


def get_stock_data(company, start_date, end_date):
    df = yf.download(company, start=start_date, end=end_date)
    return df

@app.route('/indextemp')
def indextemp():
    stock = request.args.get('stock','')
    return render_template('index_temp.html',stock = stock)


@app.route('/analyze', methods=['POST','GET'])
def analyze():
    company = request.form['company']
    analysis_type = request.form['analysis_type']
    
    if analysis_type == 'financial':
        stock_data = get_stock_data(company, '2022-01-01', '2023-01-01')
        balance_sheet, financials, cashflow = get_financial_data(company)
        stock_data_html = stock_data.to_html(classes='table table-striped')
        balance_sheet_html = balance_sheet.to_html(classes='table table-striped')
        financials_html = financials.to_html(classes='table table-striped')
        cashflow_html = cashflow.to_html(classes='table table-striped')
        return render_template('financial.html', stock_data=stock_data_html, balance_sheet=balance_sheet_html, financials=financials_html, cashflow=cashflow_html)
    
    # elif analysis_type == 'sentiment':
    #     sentiment_distribution, avg_sentiment = get_sentiment_analysis(company)
    #     return render_template('sentiment.html', sentiment_distribution=sentiment_distribution, avg_sentiment=avg_sentiment)
    
    elif analysis_type == 'sustainability':
        sustainability_data = get_sustainability_data(company)
        if sustainability_data is not None:
            sustainability_html = sustainability_data.to_html(classes='table table-striped')
            return render_template('sustainability.html', sustainability_data=sustainability_html)
        else:
            return render_template('sustainability.html', sustainability_data="No sustainability data available")
    
    # elif analysis_type == 'ai_impact':
    #     ai_impact_scores = generate_ai_impact_analysis(company)
    #     return render_template('ai_impact.html', ai_impact_scores=ai_impact_scores)
    
    elif analysis_type == 'future_trends':
        future_trends = generate_future_trends(company)
        return render_template('future_trends.html', future_trends=future_trends)



















if __name__ == '__main__':
    app.run(debug=True)