import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error

# ETF tickers and Benchmark index
tickers = ['SMAX', 'SPLG', 'IVV', 'SPY', 'VOO']
index_ticker = '^GSPC'

# Expense ratios need to be manually added
expense_ratios = {
    'SMAX': 0.53,
    'SPLG': 0.02,
    'IVV': 0.03,
    'SPY': 0.09,
    'VOO': 0.03
} 

# Function to fetch adjusted closing prices
def get_data(ticker):
    try:
        data = yf.download(ticker, period="max", interval="1mo", progress=False)
        return data["Adj Close"].sort_index()
    except Exception as e:
        print(f"Failed to retrieve data for {ticker}: {e}")
        return pd.Series(dtype=float)

# Fetch data for all tickers
data = {ticker: get_data(ticker) for ticker in tickers + [index_ticker]}
data = pd.DataFrame(data).dropna()
returns = data.pct_change().dropna()

# Ensure aligned dates
returns = returns.dropna()

# Store results
results = {}

for etf in tickers:
    if etf not in returns.columns or index_ticker not in returns.columns:
        print(f"Warning: Missing data for {etf} or {index_ticker}, skipping...")
        continue
    
    etf_returns = returns[etf]
    index_returns = returns[index_ticker]
    
    if etf_returns.empty or index_returns.empty:
        print(f"Warning: No return data for {etf}, skipping...")
        continue
    
    # Correlation
    correlation = etf_returns.corr(index_returns)
    
    # Beta
    covariance_matrix = np.cov(etf_returns, index_returns)
    if covariance_matrix.shape == (2, 2):
        covariance = covariance_matrix[0, 1]
        market_variance = np.var(index_returns)
        beta = covariance / market_variance if market_variance != 0 else np.nan
    else:
        beta = np.nan
    
    beta_penalty = abs(1 - beta) if not np.isnan(beta) else np.nan
    
    # RMSE (Tracking Accuracy)
    if len(etf_returns) > 1:
        rmse = np.sqrt(mean_squared_error(etf_returns, index_returns))
    else:
        rmse = np.nan
    
    # Tracking Error (Standard deviation of excess return)
    tracking_error = np.std(etf_returns - index_returns) if len(etf_returns) > 1 else np.nan
    
    # Tracking Difference (Average excess return)
    tracking_difference = (etf_returns - index_returns).mean() if len(etf_returns) > 1 else np.nan
    
    # Expense Ratio Adjustment
    expense_ratio = expense_ratios.get(etf, np.nan)

    # Compute final score
    if not np.isnan(correlation) and not np.isnan(beta_penalty) and not np.isnan(rmse):
        score = (correlation + beta_penalty - rmse - tracking_error + tracking_difference - expense_ratio)
    else:
        score = np.nan
    
    results[etf] = {
        'Correlation': correlation,
        'Beta': beta,
        'Beta Penalty': beta_penalty,
        'RMSE': rmse,
        'Tracking Error': tracking_error,
        'Tracking Difference': tracking_difference,
        'Expense Ratio': expense_ratio,
        'Final Score': score
    }

# Convert to DataFrame for better visualization
results_df = pd.DataFrame(results).T
print(results_df.sort_values(by='Final Score', ascending=False, na_position='last'))