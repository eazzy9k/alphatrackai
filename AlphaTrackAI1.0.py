import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from scipy.stats import norm
import time
import sys
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import ta
from tabulate import tabulate
import streamlit as st
from textblob import TextBlob
from joblib import Memory

# Initialize the scaler and classifier
scaler = StandardScaler()
classifier = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
memory = Memory("cache_dir", verbose=0)

@memory.cache
def preprocess_data(ticker):
    """Fetch and preprocess data with technical indicators."""
    data = fetch_data_with_technical_indicators(ticker)
    if data is None:
        raise ValueError(f"Failed to fetch data for ticker: {ticker}")
    data.dropna(inplace=True)  # Handle missing values
    return data

# Fetch Historical Market Data with Technical Indicators
def fetch_data_with_technical_indicators(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y")
    if data.empty:
        print(f"No historical data available for ticker {ticker}.")
        return None

    # technicals
    data['50_MA'] = data['Close'].rolling(50).mean()
    data['200_MA'] = data['Close'].rolling(200).mean()
    data['Volume_MA'] = data['Volume'].rolling(20).mean()
    data['BB_Upper'] = data['Close'].rolling(20).mean() + 2 * data['Close'].rolling(20).std()
    data['BB_Lower'] = data['Close'].rolling(20).mean() - 2 * data['Close'].rolling(20).std()
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).cumsum()
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    data['ATR'] = data['High'].rolling(14).max() - data['Low'].rolling(14).min()
    data['ROC'] = (data['Close'] / data['Close'].shift(14)) - 1
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()

    return data

# Sentiment Analysis
def analyze_news_sentiment(news_articles):
    sentiment_score = sum(TextBlob(article).sentiment.polarity for article in news_articles) / len(news_articles)
    return "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

#  Supply and Demand Levels by 3 months
def analyze_supply_demand(data):
    try:
        three_months_ago = datetime.now() - timedelta(days=90)
        filtered_data = data.loc[three_months_ago:]
        supply_level = filtered_data['High'].quantile(0.95)
        demand_level = filtered_data['Low'].quantile(0.05)
        return supply_level, demand_level
    except Exception as e:
        print(f"Error analyzing supply/demand levels: {e}")
        return None, None

#  Greeks using a chatgpt4.0 math model
def calculate_greeks(S, K, T, r, sigma, option_type):
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError(f"Invalid inputs: S={S}, K={K}, T={T}, sigma={sigma}")

        sigma = max(sigma, 1e-5)  # Avoid divide-by-zero
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
        theta = -((S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T)) if S * sigma * np.sqrt(T) > 0 else 0.0
        vega = S * norm.pdf(d1) * np.sqrt(T)
        return delta, theta, gamma, vega
    except Exception as e:
        print(f"Error calculating Greeks: {e}")
        return None, None, None, None

#  Market Sentiment and Expected Price Range
def analyze_market_sentiment(data):
    try:
        # Key metrics
        avg_rsi = data['RSI'].tail(14).mean()
        atr = data['ATR'].iloc[-1]
        roc = data['ROC'].iloc[-1]
    

        # Blended sentiment score
        sentiment_score = avg_rsi + (roc * 100) - (atr / data['Close'].mean())

        # Sentiment determination
        if sentiment_score > 100:
            sentiment = "Bullish"
        elif sentiment_score < 0:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"

        # Calculate expected price range
        recent_close = data['Close'].iloc[-1]
        upper_band = data['BB_Upper'].iloc[-1]
        lower_band = data['BB_Lower'].iloc[-1]
        expected_range = f"${lower_band:.2f} - ${upper_band:.2f}"

        return sentiment, expected_range
    except Exception as e:
        print(f"Error analyzing market sentiment: {e}")
        return "Unknown", "N/A"

     
# Feature Extraction
def extract_features(data, supply_level, demand_level):
    features = []
    for _, row in data.iterrows():
        S = row['Close']
        K = S
        T = max(30 / 365, 1e-5)
        r = 0.01
        sigma = data['Price_Change'].std() * np.sqrt(252) if not pd.isna(data['Price_Change'].std()) else 1e-5
        option_type = 'call' if row['Close'] > row['Open'] else 'put'
        delta, theta, gamma, vega = calculate_greeks(S, K, T, r, sigma, option_type)
        if delta is None:
            continue

        volatility_spike = 1 if sigma > 0.4 else 0

        features.append([
            row['Close'], row['Volume'], row['50_MA'], row['200_MA'], row['OBV'], row['RSI'],
            row['BB_Upper'], row['BB_Lower'], row['MACD'], row['VWAP'], delta, theta, gamma, vega,
            supply_level, demand_level, sigma
        ])
    return np.array(features)

 ###Train  using AutoML###
def train_model_with_automl(data, target):
    df = pd.DataFrame(data, columns=[f"Feature_{i}" for i in range(data.shape[1])])
    df['Target'] = target
    setup(data=df, target='Target', silent=True, verbose=False)
    best_model = compare_models()
    return best_model

###### NOT WORKING ###### 
#####SHAP DISABLED#####
def explain_model_with_shap(model, X_train):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train)

# Analyze Weekly Volume Trend
def analyze_weekly_volume(data):
    try:
        weekly_avg_volume = data['Volume'].tail(5).mean()
        historical_avg_volume = data['Volume'].mean()

        if weekly_avg_volume > 1.5 * historical_avg_volume:
            volume_strength = "Strong"
        elif weekly_avg_volume > historical_avg_volume:
            volume_strength = "Moderate"
        else:
            volume_strength = "Weak"

        return volume_strength, weekly_avg_volume
    except Exception as e:
        print(f"Error analyzing weekly volume: {e}")
        return "Unknown", 0
# Backtesting Function
# Updated Backtesting Function
# Simulate Expiration Dates
####BACKTESTING NOT EMITTING CORRECT RESULTS OR PERCENTAGE####
def simulate_expiration_dates(start_date, end_date, interval="weekly"):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    simulated_dates = []

    while start <= end:
        simulated_dates.append(start.strftime('%Y-%m-%d'))
        if interval == "weekly":
            start += timedelta(weeks=1)
        elif interval == "monthly":
            start += timedelta(weeks=4)
        else:
            raise ValueError("Unsupported interval. Use 'weekly' or 'monthly'.")

    return simulated_dates

# Backtesting Function
# Fetch Valid Expirations
def get_valid_expirations(ticker):
    stock = yf.Ticker(ticker)
    return stock.options

# Align Simulated Expirations with Real Data
def get_nearest_expiration(simulated_date, valid_expirations):
    simulated_date = pd.to_datetime(simulated_date)
    valid_expirations = pd.to_datetime(valid_expirations)
    nearest_expiration = valid_expirations[valid_expirations >= simulated_date].min()
    return nearest_expiration.strftime('%Y-%m-%d') if pd.notna(nearest_expiration) else None

# Backtesting Function
# Backtesting Tool
def backtest_tool(ticker, start_date, end_date):
    historical_data = fetch_data_with_technical_indicators(ticker)
    if historical_data is None:
        return None
    simulated_dates = pd.date_range(start_date, end_date, freq='W')
    results = []

    for simulated_date in simulated_dates:
        if simulated_date.strftime('%Y-%m-%d') in historical_data.index:
            close_price = historical_data.loc[simulated_date.strftime('%Y-%m-%d'), 'Close']
            results.append({"date": simulated_date, "close_price": close_price})

    return pd.DataFrame(results)
    # Calculate average hold duration
    average_hold_duration = results_df.apply(
        lambda x: (pd.to_datetime(x['expiration']) - pd.to_datetime(x['date'])).days, axis=1
    ).mean()

    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Average Hold Duration: {average_hold_duration:.2f} days")

    return results_df

# Simulate Exit Price
def simulate_exit_price(option):
    # Use a percentage change or an average price movement as a proxy
    return option["bid"] * 1.1  # Simulate a 10% increase in bid price

# Plot Backtesting Results
import plotly.graph_objects as go

def plot_backtest(results):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results['date'], y=results['cumulative_profit'], mode='lines', name='Cumulative Profit'))
    fig.update_layout(title='Backtesting Results', xaxis_title='Date', yaxis_title='Cumulative Profit ($)')
    fig.show()
# Recommend Options Contracts
def recommend_options(ticker, sentiment, expiration_date):
    try:
        stock = yf.Ticker(ticker)
        options_chain = stock.option_chain(expiration_date)
        calls = options_chain.calls
        puts = options_chain.puts
    except Exception as e:
        print(f"Error fetching option data: {e}")
        return []

    # Fetch historical data
    historical_data = fetch_data_with_technical_indicators(ticker)
    if historical_data is None:
        print("No historical data available for analysis.")
        return []

    # Calculate sentiment and expected range
    sentiment, expected_range = analyze_market_sentiment(historical_data)

    recommendations = []
    option_data = calls if sentiment == "Bullish" else puts

    for _, row in option_data.iterrows():
        try:
            # Handle missing or invalid values
            strike = row['strike'] if not pd.isna(row['strike']) else 0.0
            bid = row['bid'] if not pd.isna(row['bid']) else 0.0
            ask = row['ask'] if not pd.isna(row['ask']) else 0.0
            volume = int(row['volume']) if not pd.isna(row['volume']) and row['volume'] > 0 else 0
            open_interest = int(row['openInterest']) if not pd.isna(row['openInterest']) and row['openInterest'] > 0 else 0
            implied_volatility = row['impliedVolatility'] if not pd.isna(row['impliedVolatility']) else 0.0

            # Skip invalid rows
            if bid <= 0 or ask <= 0 or strike <= 0 or volume <= 0:
                continue

            mid_price = (bid + ask) / 2
            spread = ask - bid
            profitability = (mid_price - bid) / bid if bid > 0 else 0
            volatility_spike = 1 if implied_volatility > 0.4 else 0

            # Filter out contracts with high spreads
            if spread > 0.5 * mid_price:
                continue

            # Append valid recommendations
            recommendations.append({
                "type": "Call" if option_data is calls else "Put",
                "strike": float(strike),
                "expiration": expiration_date,
                "bid": float(bid),
                "ask": float(ask),
                "volume": volume,
                "openInterest": open_interest,
                "profitability": f"{profitability * 100:.2f}%",
                "volatility_spike": volatility_spike,
                "spread": spread,
                "overall_market_sentiment": sentiment,
                "expected_price_range": expected_range
            })
        except Exception as e:
            print(f"Error processing option data: {e}")
            continue

    # Sort and limit recommendations
    recommendations.sort(key=lambda x: (
        -float(x['profitability'].strip('%')),  # Higher profitability
        x['spread'],                           # Lower spread
        -x['volume']                           # Higher volume
    ))

    return recommendations[:5] # WILL Return  5 recommendations


############################################# MAIN FUNCTION #############################################
if __name__ == "__main__":
    print("Welcome to AlphaTrackAI (Vers. 1.1) - Your AI-Powered Trading Assistant\n")

    # Ask the user to select a mode
    mode = input("Select Mode: (1) Live Recommendations (2) Backtest\n").strip()

    if mode == "1":  # Live Recommendations Tool
        ticker = input("Enter Stock Ticker (e.g., AAPL): ").strip().upper()

        try:
            # Fetch available expirations
            stock = yf.Ticker(ticker)
            expirations = stock.options

            if not expirations:
                print("No options data available for the selected ticker.")
                sys.exit()

            print("\nAvailable Expirations:")
            for idx, exp in enumerate(expirations):
                print(f"{idx + 1}: {exp}")

            expiration_choice = int(input("Select an expiration date by number (or 0 for nearest expiration): ").strip())
            expiration_date = expirations[expiration_choice - 1] if expiration_choice > 0 else expirations[0]
        except (ValueError, IndexError):
            print("Invalid selection. Defaulting to the nearest expiration.")
            expiration_date = expirations[0]

        # Fetch historical data and analyze market sentiment
        try:
            historical_data = preprocess_data(ticker)  # Includes volume-based indicators
            sentiment, price_range = analyze_market_sentiment(historical_data)

            print(f"Market Sentiment: {sentiment}")
            print(f"Expected Price Range: {price_range}")

            # Fetch and display recommendations
            recommendations = recommend_options(ticker, sentiment, expiration_date)

            if recommendations:
                print("\nTop Options Recommendations:")
                print(tabulate(recommendations, headers="keys", tablefmt="pretty"))
            else:
                print("No valid options recommendations found.")
        except Exception as e:
            print(f"Error during live recommendations: {e}")

    elif mode == "2":  # Backtest Tool
        ticker = input("Enter Stock Ticker (e.g., AAPL): ").strip().upper()
        start_date = input("Enter Backtest Start Date (YYYY-MM-DD): ").strip()
        end_date = input("Enter Backtest End Date (YYYY-MM-DD): ").strip()

        try:
            # Run the backtest
            backtest_results = backtest_tool(ticker, start_date, end_date)

            if backtest_results is not None and not backtest_results.empty:
                print("\nBacktest Summary:")
                total_profit = backtest_results['cumulative_profit'].sum()
                avg_profit = backtest_results['cumulative_profit'].mean()
                sharpe_ratio = avg_profit / backtest_results['cumulative_profit'].std()
                win_rate = (backtest_results['cumulative_profit'] > 0).mean()

                print(f"Total Profit: ${total_profit:.2f}")
                print(f"Win Rate: {win_rate:.2%}")
                print(f"Average Profit: ${avg_profit:.2f}")
                print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

                # Plot backtesting results
                plot_backtest(backtest_results)
            else:
                print("No valid trades during the backtest period.")
        except Exception as e:
            print(f"Error during backtesting: {e}")
    else:
        print("Invalid mode selected. Please choose 1 or 2.")

    print("Thank you for using AlphaTrackAI. Exiting now...")
    sys.exit()
