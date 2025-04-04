
import numpy as np

def compute_advanced_risk_metrics(stock_returns, market_returns, beta, risk_free_rate=0.02, annual_factor=252):
    """
    Computes advanced risk metrics to supplement existing ones.
    """
    metrics = {}

    # Treynor Ratio
    try:
        annualized_return = (1 + stock_returns.mean()) ** annual_factor - 1
        metrics['Treynor Ratio'] = (annualized_return - risk_free_rate) / beta if beta != 0 else np.nan
    except:
        metrics['Treynor Ratio'] = np.nan

    # Skewness & Kurtosis
    metrics['Skewness'] = stock_returns.skew()
    metrics['Kurtosis'] = stock_returns.kurtosis()

    # Correlation with Market
    try:
        metrics['Correlation with Market'] = stock_returns.corr(market_returns)
    except:
        metrics['Correlation with Market'] = np.nan

    # Downside Capture Ratio
    try:
        market_down = market_returns[market_returns < 0]
        stock_down = stock_returns[market_returns < 0]
        metrics['Downside Capture Ratio'] = stock_down.mean() / market_down.mean() if market_down.mean() != 0 else np.nan
    except:
        metrics['Downside Capture Ratio'] = np.nan

    return metrics
