# Stock Picker Script

import yfinance as yf

def pick_stocks(tickers, num_stocks):
    selected_stocks = []
    
    for ticker in tickers:
        # Fetch stock data using yfinance
        data = yf.download(ticker, period='1y')
        
        if data.empty:
            print(f"No data available for {ticker}. Skipping...")
            continue
        
        # Calculate one-year return
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        one_year_return = (end_price - start_price) / start_price
        
        # Add stock to selected stocks if positive return
        if one_year_return > 0:
            selected_stocks.append((ticker, one_year_return))
    
    # Sort stocks by return and select top N
    selected_stocks.sort(key=lambda x: x[1], reverse=True)
    top_stocks = selected_stocks[:num_stocks]
    
    return top_stocks

# Example usage
tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'META']  # List of stock tickers
num_stocks = 5  # Number of stocks to pick

top_picks = pick_stocks(tickers, num_stocks)
for stock in top_picks:
    ticker, return_pct = stock
    print(f"{ticker}: {return_pct * 100:.2f}% return in the past year")