import yfinance as yf
import pandas as pd
import numpy as np

def calculate_ratios(ticker):
    stock = yf.Ticker(ticker)

    # Get fundamental data
    info = stock.info

    # Calculate ratios
    # Doesn't calulcate ROI, Interest Coverage, Working Capital or Adjusted EPS probably due to an inability to grab data from Fin Stmts   
    pe_ratio = info.get('trailingPE', None)
    pb_ratio = info.get('priceToBook', None)
    de_ratio = info.get('debtToEquity', None)
    fcf = info.get('freeCashflow', None)
    peg_ratio = info.get('pegRatio', None)
    eps = info.get('trailingEps', None)
    roe = info.get('returnOnEquity', None)
    ev_ebit = info.get('enterpriseToEbitda', None)
    quick_ratio = info.get('quickRatio', None)

    return pe_ratio, pb_ratio, de_ratio, fcf, peg_ratio, eps, roe, ev_ebit, quick_ratio

def calculate_momentum(data, period):
    # Calculate momentum using price changes over the specified period
    price_changes = data['Close'].pct_change(periods=period)
    momentum = price_changes.sum()
    return momentum

def calculate_risk_measures(data, benchmark):
    benchmark_returns = benchmark.pct_change().dropna()

    # Align lengths of benchmark returns and portfolio returns
    benchmark_returns = benchmark_returns[-len(data):]

    # Calculate risk measures
    returns = data.pct_change().dropna()
    excess_returns = returns - benchmark_returns
    excess_returns_mean = excess_returns.mean()
    excess_returns_cov = excess_returns.cov(benchmark_returns)
    if isinstance(excess_returns_cov, np.float64):
        excess_returns_cov = np.array([[excess_returns_cov]])
    if isinstance(excess_returns_mean, np.float64):
        excess_returns_mean = np.array([excess_returns_mean])
    weights = np.linalg.inv(excess_returns_cov) @ excess_returns_mean.reshape(-1, 1)
    weights /= np.sum(weights)
    returns_df = pd.DataFrame(returns)
    portfolio_returns = (returns_df * weights.flatten()).sum(axis=1)
    alpha = (portfolio_returns - benchmark_returns).mean() * 252 # 252 annualizes it
    beta = excess_returns_cov[0, 0] / benchmark_returns.var()
    r_squared = (beta * benchmark_returns.std()) / portfolio_returns.std()
    standard_deviation = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = alpha / standard_deviation

    return alpha, beta, r_squared, standard_deviation, sharpe_ratio

def pick_stocks(file_path, num_stocks, ratios, momentum_period, benchmark_ticker):
    selected_stocks = []

    # Read stocks from CSV file
    stocks_df = pd.read_csv(file_path)

    # Fetch benchmark data using yfinance
    benchmark_data = yf.download(benchmark_ticker, period='1y')['Close']

    for _, row in stocks_df.iterrows():
        ticker = row['Ticker']

        # Fetch stock data using yfinance
        data = yf.download(ticker, period='1y')

        if data.empty:
            print(f"No data available for {ticker}. Skipping...")
            continue

        # Calculate one-year return
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        one_year_return = ((end_price - start_price) / start_price) * 100

        # Calculate additional ratios
        ratios_values = calculate_ratios(ticker)

        # Calculate momentum
        momentum = calculate_momentum(data, momentum_period)

        # Calculate risk metrics
        alpha, beta, r_squared, std_dev, sharpe_ratio = calculate_risk_measures(data['Close'], benchmark_data)

        # Add stock to selected stocks
        selected_stocks.append((ticker, one_year_return, momentum, alpha, beta, r_squared, std_dev, sharpe_ratio, *ratios_values))

    # Sort stocks based on selected ratios, momentum, and alpha
    selected_stocks.sort(key=lambda x: tuple(x[3+i] for i, _ in enumerate(ratios + ['Momentum'])), reverse=True)

    # Select top N stocks based on the specified ratios, momentum, and alpha
    top_stocks = selected_stocks[:num_stocks]

    # Create a dataframe from selected stocks
    columns = ['Ticker', 'Return', 'Momentum', 'Alpha', 'Beta', 'R-squared', 'Standard Deviation', 'Sharpe Ratio',
               'Trailing P/E Ratio', 'P/B Ratio', 'D/E Ratio', 'Free Cash Flow', 'PEG Ratio',
               'Trailing EPS', 'ROE', 'Interest Coverage', 'EV/EBIT', 'Quick Ratio']
    df = pd.DataFrame(top_stocks, columns=columns)

    return df

# Usage
file_path = 'SP500.csv'  # CSV file containing stock tickers
num_stocks = 25  # Number of stocks to pick
ratios_to_use = ['ROI', 'P/E Ratio', 'P/B Ratio']  # Ratios to use for selecting stocks
momentum_period = 90  # Period for calculating momentum in days
benchmark_ticker = '^GSPC'  # Benchmark index ticker

top_picks_df = pick_stocks(file_path, num_stocks, ratios=ratios_to_use, momentum_period=momentum_period, benchmark_ticker=benchmark_ticker)

# Dataframe to CSV
output_file = 'top_picks.csv'  # Output CSV file
top_picks_df.to_csv(output_file, index=False)

print(f"Top picks saved to {output_file}")