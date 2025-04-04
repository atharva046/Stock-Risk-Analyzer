import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px




st.set_page_config(
    page_title="Stock Risk Analyzer",
    page_icon="📈",
    layout="wide"
)


st.title("📊 Stock Risk Analyzer")
st.markdown("""
This application analyzes risk metrics for your selected stocks using real-time market data.
Enter a stock ticker to view comprehensive risk analysis including volatility, beta, Sharpe ratio, and more.
""")


st.sidebar.header("Parameters")




indian_tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
    "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "ITC.NS", "WIPRO.NS",
    "LT.NS", "MARUTI.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS", "NTPC.NS",
    "POWERGRID.NS", "TECHM.NS", "NESTLEIND.NS", "BHARTIARTL.NS", "COALINDIA.NS",
    "BAJAJ-AUTO.NS", "HCLTECH.NS", "ONGC.NS", "TITAN.NS", "HDFCLIFE.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "CIPLA.NS", "DIVISLAB.NS", "GRASIM.NS", "BPCL.NS", "BRITANNIA.NS",
    "TATASTEEL.NS", "EICHERMOT.NS", "SHREECEM.NS", "JSWSTEEL.NS", "HINDALCO.NS", "IOC.NS",
    "DRREDDY.NS", "BAJAJFINSV.NS", "HEROMOTOCO.NS", "M&M.NS", "INDUSINDBK.NS", "SBILIFE.NS",
    "UPL.NS", "HAVELLS.NS", "DMART.NS", "PIDILITIND.NS", "LTI.NS", "LTTS.NS", "TATACONSUM.NS",
    "GAIL.NS", "AMBUJACEM.NS", "DABUR.NS", "COLPAL.NS", "ICICIPRULI.NS", "HINDPETRO.NS",
    "BOSCHLTD.NS", "GODREJCP.NS", "TATAMOTORS.NS", "TVSMOTOR.NS", "PAGEIND.NS"
]

selected_ticker = st.sidebar.selectbox("Select Indian Stock", indian_tickers)
custom_ticker = st.sidebar.text_input("Or Enter Custom Ticker", selected_ticker)
ticker = custom_ticker if custom_ticker else selected_ticker



index_choices = {
    "Nifty 50 (^NSEI)": "^NSEI",
    "BSE Sensex (^BSESN)": "^BSESN",
    "Nifty Bank (^NSEBANK)": "^NSEBANK",
    "Nifty IT (^CNXIT)": "^CNXIT",
    "Nifty FMCG (^CNXFMCG)": "^CNXFMCG",
    "Nifty Pharma (^CNXPHARMA)": "^CNXPHARMA",
    "Nifty Financial Services (^CNXFINANCE)": "^CNXFINANCE",
    "Nifty Midcap 50 (^NSEMDCP50)": "^NSEMDCP50",
    "Nifty Smallcap (^CNXSMLCAP)": "^CNXSMLCAP",
    "Nifty Auto (^CNXAUTO)": "^CNXAUTO",
    "Nifty Realty (^CNXREALTY)": "^CNXREALTY",
    "S&P 500 (^GSPC)": "^GSPC",
    "Dow Jones (^DJI)": "^DJI",
    "Nasdaq Composite (^IXIC)": "^IXIC"
}
market_index_label = st.sidebar.selectbox("Select Market Index for Beta", list(index_choices.keys()))
market_ticker = index_choices[market_index_label]
  


time_periods = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "3 Years": 1095,
    "5 Years": 1825
}
selected_period = st.sidebar.selectbox("Select Time Period", list(time_periods.keys()))


risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=2.0) / 100


@st.cache_data(ttl=3600)  
def get_stock_data(ticker, period_days):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        if stock_data.empty:
            return None
        
      
        if 'Adj Close' not in stock_data.columns:
            if 'Close' in stock_data.columns:
                stock_data['Adj Close'] = stock_data['Close']
              
                pass
            else:
                st.error(f"Neither 'Adj Close' nor 'Close' data available for {ticker}.")
                return None
            
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def validate_data(data, ticker_name):
    if data is None:
        st.error(f"No data available for {ticker_name}. Please check the ticker symbol.")
        return False
    
    required_columns = ['Adj Close']
    for col in required_columns:
        if col not in data.columns:
            st.error(f"Required column '{col}' not found in {ticker_name} data.")
            return False
    
    return True


def calculate_risk_metrics(stock_data, market_data, risk_free_rate):
    try:
       
        stock_returns = stock_data['Adj Close'].pct_change().dropna()
        market_returns = market_data['Adj Close'].pct_change().dropna()
        
        
        common_index = stock_returns.index.intersection(market_returns.index)
        stock_returns = stock_returns.loc[common_index]
        market_returns = market_returns.loc[common_index]
        
        
        annual_factor = 252  
        
       
        metrics = {}
        
        
        daily_volatility = stock_returns.std()
        metrics['Daily Volatility'] = daily_volatility
        metrics['Annualized Volatility'] = daily_volatility * np.sqrt(annual_factor)
        
        
        covariance = np.cov(stock_returns, market_returns)[0, 1]
        market_variance = market_returns.var()
        metrics['Beta'] = covariance / market_variance if market_variance != 0 else 0
        
        
        metrics['Daily Mean Return'] = stock_returns.mean()
        metrics['Annualized Return'] = (1 + metrics['Daily Mean Return']) ** annual_factor - 1
        
        
        daily_risk_free = risk_free_rate / annual_factor
        metrics['Sharpe Ratio'] = (metrics['Annualized Return'] - risk_free_rate) / metrics['Annualized Volatility'] if metrics['Annualized Volatility'] != 0 else 0
        
        
        cumulative_returns = (1 + stock_returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / rolling_max) - 1
        metrics['Maximum Drawdown'] = drawdowns.min()
        
       
        metrics['VaR (95%)'] = np.percentile(stock_returns, 5)
        
        
        var_95 = metrics['VaR (95%)']
        metrics['CVaR (95%)'] = stock_returns[stock_returns <= var_95].mean()
        
       
        negative_returns = stock_returns[stock_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(annual_factor) if len(negative_returns) > 0 else 0
        metrics['Sortino Ratio'] = (metrics['Annualized Return'] - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
        
        
        metrics['Calmar Ratio'] = metrics['Annualized Return'] / abs(metrics['Maximum Drawdown']) if metrics['Maximum Drawdown'] != 0 else 0
        
        
        tracking_error = (stock_returns - market_returns).std() * np.sqrt(annual_factor)
        metrics['Information Ratio'] = (metrics['Annualized Return'] - (market_returns.mean() * annual_factor)) / tracking_error if tracking_error != 0 else 0
        
       
        return metrics, stock_returns, market_returns, drawdowns
    except Exception as e:
        st.error(f"Error calculating risk metrics: {e}")
        st.error("Please check if the data is valid and try again.")
        return None, None, None, None


def detect_anomalies(returns, contamination=0.05):
    if returns.empty or len(returns) < 10:
        st.warning("Not enough data points for anomaly detection.")
        return pd.Series()
    
    try:
        model = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = model.fit_predict(returns.values.reshape(-1, 1))
        anomalies = returns[anomaly_labels == -1]
        return anomalies
    except Exception as e:
        st.error(f"Error in anomaly detection: {e}")
        return pd.Series()


if ticker:
   
    with st.spinner(f"Fetching data for {ticker}..."):
        period_days = time_periods[selected_period]
        stock_data = get_stock_data(ticker, period_days)
        market_data = get_stock_data(market_ticker, period_days)
        
       
        stock_valid = validate_data(stock_data, ticker)
        market_valid = validate_data(market_data, market_ticker)
        
        if stock_valid and market_valid and len(stock_data) > 1 and len(market_data) > 1:
            
            
            
           
            result = calculate_risk_metrics(stock_data, market_data, risk_free_rate)
            
            if result[0] is None:
                st.error("Failed to calculate risk metrics. Please try another ticker or time period.")
            else:
                metrics, stock_returns, market_returns, drawdowns = result
                
                
                st.header(f"Analysis for {ticker}")
                
                
                last_price = stock_data['Adj Close'].iloc[-1]
                
              
                if len(stock_data) >= 2:
                    price_change = stock_data['Adj Close'].iloc[-1] - stock_data['Adj Close'].iloc[-2]
                    price_change_pct = (price_change / stock_data['Adj Close'].iloc[-2]) * 100
                    price_change_display = f"{price_change_pct:.2f}%"
                else:
                    price_change_display = "N/A"
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"${last_price:.2f}", price_change_display)
                col2.metric("Volatility (Annualized)", f"{metrics['Annualized Volatility']:.2%}")
                col3.metric("Beta", f"{metrics['Beta']:.2f}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                col2.metric("Maximum Drawdown", f"{metrics['Maximum Drawdown']:.2%}")
                col3.metric("VaR (95%)", f"{metrics['VaR (95%)']:.2%}")
                
               
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Price & Returns", "📊 Risk Metrics", "🔍 Anomaly Detection", "📉 Drawdown Analysis"])
                
                with tab1:
                    st.subheader("Price History & Returns")
               
                st.subheader("Price Comparison with Market Index")
                price_compare_df = pd.DataFrame({
                    ticker: stock_data['Adj Close'],
                    market_ticker: market_data['Adj Close']
                })
                price_compare_df = price_compare_df / price_compare_df.iloc[0]  # Normalize to 1
                fig_compare = px.line(price_compare_df, title=f"{ticker} vs {market_ticker} (Normalized Prices)")
                fig_compare.update_layout(xaxis_title='Date', yaxis_title='Normalized Price', height=500)
                st.plotly_chart(fig_compare, use_container_width=True)

                    
                   
                fig = px.line(stock_data, x=stock_data.index, y='Adj Close', title=f"{ticker} Price History")
                    
                fig.update_layout(xaxis_title='Date', yaxis_title='Price ($)', height=500)
                st.plotly_chart(fig, use_container_width=True)
                    
                    
                if not stock_returns.empty and len(stock_returns) > 1:
                    st.subheader("Returns Distribution")
                    fig = px.histogram(stock_returns, nbins=min(50, len(stock_returns)//2), title=f"{ticker} Daily Returns Distribution")
                    fig.update_layout(xaxis_title='Daily Return', yaxis_title='Frequency', height=400)
                        
                      
                if len(stock_returns) >= 10:
                    mu = stock_returns.mean()
                    sigma = stock_returns.std()
                    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
                    y = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
                    y = y * (len(stock_returns) * (stock_returns.max() - stock_returns.min()) / min(50, len(stock_returns)//2))  
                            
                    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Normal Distribution'))
                        
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough return data to display distribution.")
                    
                    
                if not stock_returns.empty and not market_returns.empty and len(stock_returns) > 1:
                        st.subheader(f"Beta Analysis (vs {market_ticker})")
                        scatter_df = pd.DataFrame({
                            'Market Returns': market_returns,
                            'Stock Returns': stock_returns
                        })
                        
                        fig = px.scatter(scatter_df, x='Market Returns', y='Stock Returns', 
                                        trendline='ols', trendline_color_override='red',
                                        title=f"Beta Visualization: {ticker} vs {market_ticker}")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data to display beta visualization.")
                
                with tab2:
                    st.subheader("Comprehensive Risk Metrics")
                    
                   
                    metrics_df = pd.DataFrame({
                        'Metric': list(metrics.keys()),
                        'Value': list(metrics.values())
                    })
                    
                   
                    metrics_df['Formatted Value'] = metrics_df.apply(
                        lambda x: f"{x['Value']:.2%}" if "Volatility" in x['Metric'] or "Return" in x['Metric'] 
                        or "Drawdown" in x['Metric'] or "VaR" in x['Metric'] or "CVaR" in x['Metric']
                        else f"{x['Value']:.2f}", axis=1
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    
                    half = len(metrics_df) // 2
                    with col1:
                        for i in range(half):
                            st.metric(metrics_df['Metric'][i], metrics_df['Formatted Value'][i])
                    
                   
                    with col2:
                        for i in range(half, len(metrics_df)):
                            st.metric(metrics_df['Metric'][i], metrics_df['Formatted Value'][i])
                    
                    
                    st.subheader("Risk Interpretation")
                    
                    
                    vol = metrics['Annualized Volatility']
                    if vol < 0.15:
                        vol_risk = "Low volatility"
                    elif vol < 0.25:
                        vol_risk = "Medium volatility"
                    else:
                        vol_risk = "High volatility"
                    
                    
                    beta = metrics['Beta']
                    if beta < 0.8:
                        beta_risk = "Defensive stock (less volatile than the market)"
                    elif beta < 1.2:
                        beta_risk = "Market-like risk"
                    else:
                        beta_risk = "Aggressive stock (more volatile than the market)"
                    
                   
                    sharpe = metrics['Sharpe Ratio']
                    if sharpe < 0:
                        sharpe_risk = "Poor risk-adjusted returns"
                    elif sharpe < 1:
                        sharpe_risk = "Below average risk-adjusted returns"
                    elif sharpe < 2:
                        sharpe_risk = "Good risk-adjusted returns"
                    else:
                        sharpe_risk = "Excellent risk-adjusted returns"
                    
                   
                    var = metrics['VaR (95%)']
                    var_interp = f"There is a 5% chance of losing {abs(var):.2%} or more in a single day"
                    
                    
                    interpretations = {
                        "Volatility": vol_risk,
                        "Beta": beta_risk,
                        "Sharpe Ratio": sharpe_risk,
                        "Value at Risk": var_interp,
                        "Maximum Drawdown": f"The stock lost up to {abs(metrics['Maximum Drawdown']):.2%} from peak to trough in the selected period"
                    }
                    
                    for metric, interp in interpretations.items():
                        st.info(f"**{metric}**: {interp}")
                    
                   
                    risk_factors = 0
                    risk_factors += 1 if vol >= 0.25 else 0
                    risk_factors += 1 if beta >= 1.2 else 0
                    risk_factors += 1 if sharpe < 1 else 0
                    risk_factors += 1 if abs(metrics['Maximum Drawdown']) > 0.2 else 0
                    
                    risk_levels = ["Low", "Moderate", "Substantial", "High", "Very High"]
                    overall_risk = risk_levels[min(risk_factors, 4)]
                    
                    st.subheader("Overall Risk Assessment")
                    st.markdown(f"### {overall_risk} Risk")
                    
                    st.progress(risk_factors / 4)  
                    
                    risk_descriptions = {
                        "Low": "This stock exhibits low volatility and drawdowns, with good risk-adjusted returns.",
                        "Moderate": "This stock shows average market risk with acceptable volatility and drawdowns.",
                        "Substantial": "This stock has above-average volatility or drawdowns, requiring careful monitoring.",
                        "High": "This stock is highly volatile with significant drawdowns and potentially poor risk-adjusted returns.",
                        "Very High": "This stock shows extreme volatility and severe drawdowns, indicating significant investment risk."
                    }
                    
                    st.markdown(f"*{risk_descriptions[overall_risk]}*")
                
                with tab3:
                    st.subheader("Anomaly Detection")
                    
                    if len(stock_returns) >= 10:  
                        
                        contamination = st.slider("Anomaly Sensitivity", 0.01, 0.1, 0.05, 0.01, 
                                                help="Lower values detect fewer, more extreme anomalies")
                        anomalies = detect_anomalies(stock_returns, contamination)
                        
                        if not anomalies.empty:
                            
                            fig = go.Figure()
                            
                           
                            fig.add_trace(go.Scatter(
                                x=stock_data.index,
                                y=stock_data['Adj Close'],
                                mode='lines',
                                name='Price'
                            ))
                            
                            
                            anomaly_dates = anomalies.index
                            anomaly_prices = stock_data.loc[anomaly_dates, 'Adj Close']
                            
                            fig.add_trace(go.Scatter(
                                x=anomaly_dates,
                                y=anomaly_prices,
                                mode='markers',
                                name='Anomalies',
                                marker=dict(color='red', size=10)
                            ))
                            
                            fig.update_layout(
                                title="Price Chart with Detected Anomalies",
                                xaxis_title="Date",
                                yaxis_title="Price ($)",
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            
                            st.subheader("Anomalous Returns")
                            if not anomalies.empty:
                                anomaly_df = pd.DataFrame({
                                    'Date': anomalies.index,
                                    'Return': anomalies.values,
                                    'Return (%)': anomalies.values * 100
                                })
                                anomaly_df = anomaly_df.sort_values('Return', ascending=False)
                                st.dataframe(anomaly_df)
                                
                                st.write(f"Found {len(anomalies)} anomalous trading days out of {len(stock_returns)} total days ({len(anomalies)/len(stock_returns):.1%}).")
                            else:
                                st.write("No anomalies detected with current settings.")
                            
                            
                            st.subheader("Anomaly Interpretation")
                            if not anomalies.empty:
                                extreme_pos = len(anomalies[anomalies > 0])
                                extreme_neg = len(anomalies[anomalies < 0])
                                
                                st.write(f"- {extreme_pos} extreme positive return events")
                                st.write(f"- {extreme_neg} extreme negative return events")
                                
                                if extreme_neg > extreme_pos * 2:
                                    st.warning("This stock shows significantly more extreme negative events than positive ones, indicating potential downside risk.")
                                elif extreme_pos > extreme_neg * 2:
                                    st.info("This stock shows significantly more extreme positive events than negative ones.")
                                
                                if not anomalies.empty:
                                    most_extreme = anomalies.abs().idxmax()
                                    st.write(f"Most extreme event occurred on {most_extreme.date()} with a return of {anomalies[most_extreme]:.2%}")
                        else:
                            st.warning("No anomalies detected with current settings. Try adjusting the sensitivity slider.")
                    else:
                        st.warning("Not enough data for anomaly detection. Try selecting a longer time period.")
                
                with tab4:
                    st.subheader("Drawdown Analysis")
                    
                    if not drawdowns.empty and len(drawdowns) > 5:
                       
                        fig = px.line(drawdowns, title=f"{ticker} Drawdowns Over Time")
                        fig.update_layout(xaxis_title='Date', yaxis_title='Drawdown (%)', height=500)
                        fig.update_yaxes(tickformat='.0%') 
                        
                        
                        fig.add_hline(y=metrics['Maximum Drawdown'], line_dash="dash", 
                                    annotation_text=f"Max Drawdown: {metrics['Maximum Drawdown']:.2%}")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                      
                        drawdowns_series = drawdowns.copy()
                        
                        
                        is_drawdown = drawdowns_series < -0.05  
                        
                        if not is_drawdown.empty and is_drawdown.any():
                            is_drawdown_change = is_drawdown.diff().fillna(is_drawdown.iloc[0])
                            drawdown_starts = is_drawdown_change[is_drawdown_change == True].index
                            drawdown_ends = is_drawdown_change[is_drawdown_change == False].index
                            
                           
                            if len(drawdown_starts) > 0 or len(drawdown_ends) > 0:
                              
                                if len(drawdown_starts) > 0 and (len(drawdown_ends) == 0 or drawdown_starts[0] > drawdown_ends[0]):
                                    drawdown_ends = pd.Index([drawdowns_series.index[0]]).append(drawdown_ends)
                                
                                if len(drawdown_ends) > 0 and (len(drawdown_starts) == 0 or drawdown_ends[-1] < drawdown_starts[-1]):
                                    drawdown_ends = drawdown_ends.append(pd.Index([drawdowns_series.index[-1]]))
                                
                           
                                periods = []
                                if len(drawdown_starts) > 0 and len(drawdown_ends) > 0:
                                    for i in range(min(len(drawdown_starts), len(drawdown_ends))):
                                        if i < len(drawdown_starts):
                                            start = drawdown_starts[i]
                                            
                                            valid_ends = drawdown_ends[drawdown_ends > start]
                                            if len(valid_ends) > 0:
                                                end = valid_ends[0]
                                                
                                        
                                                period_slice = drawdowns_series[start:end]
                                                if not period_slice.empty:
                                                    period_drawdown = period_slice.min()
                                                    
                                                    
                                                    if period_drawdown < -0.05:  
                                                        recovery_point = drawdowns_series[drawdowns_series.index >= end]
                                                        recovery_point = recovery_point[recovery_point >= 0]
                                                        
                                                        recovery_date = None
                                                        if not recovery_point.empty:
                                                            recovery_date = recovery_point.index[0]
                                                        
                                                        periods.append({
                                                            'Start': start.date(),
                                                            'End': end.date(),
                                                            'Recovery': recovery_date.date() if recovery_date else "Not Yet Recovered",
                                                            'Max Drawdown': period_drawdown,
                                                            'Duration (Days)': (end - start).days,
                                                            'Recovery Time (Days)': (recovery_date - end).days if recovery_date else None
                                                        })
                
                                if periods:
                                
                                    periods_df = pd.DataFrame(periods)
                                    
                                  
                                    periods_df['Max Drawdown'] = periods_df['Max Drawdown'].map('{:.2%}'.format)
                                    
                                    st.subheader("Major Drawdown Periods")
                                    st.dataframe(periods_df)
                                    
                                    
                                    recovery_times = [p['Recovery Time (Days)'] for p in periods if p['Recovery Time (Days)'] is not None]
                                    if recovery_times:
                                        avg_recovery = sum(recovery_times) / len(recovery_times)
                                        st.write(f"Average recovery time: {avg_recovery:.1f} days")
                                else:
                                    st.write("No significant drawdown periods (>5%) detected in the selected time range.")
                            else:
                                st.write("No significant drawdown transitions detected in the data.")
                        else:
                            st.write("No significant drawdowns detected in the data.")
                        
                        
                        st.subheader("Drawdown Risk Analysis")
                        
                     
                        avg_drawdown = drawdowns.mean()
                        drawdown_volatility = drawdowns.std()
                        time_in_drawdown = (drawdowns < 0).mean() * 100  
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Average Drawdown", f"{avg_drawdown:.2%}")
                        col2.metric("Drawdown Volatility", f"{drawdown_volatility:.2%}")
                        col3.metric("Time in Drawdown", f"{time_in_drawdown:.1f}%")
                        
                       
                        max_dd = abs(metrics['Maximum Drawdown'])
                        
                        if max_dd < 0.1:
                            dd_risk = "Low drawdown risk"
                        elif max_dd < 0.2:
                            dd_risk = "Moderate drawdown risk"
                        elif max_dd < 0.3:
                            dd_risk = "High drawdown risk"
                        else:
                            dd_risk = "Severe drawdown risk"
                        
                        st.info(f"**Drawdown Risk Assessment**: {dd_risk}")
                        st.write(f"This stock has experienced a maximum drawdown of {max_dd:.2%} during the selected period. {time_in_drawdown:.1f}% of the time was spent in drawdown.")
                    else:
                        st.warning("Not enough data for drawdown analysis. Try selecting a longer time period.")
        else:
            if not stock_valid:
                st.error(f"Could not retrieve valid data for {ticker}. Please check the ticker symbol and try again.")
            elif not market_valid:
                st.error(f"Could not retrieve valid data for market index {market_ticker}. Using a different index may help.")
            else:
                st.error("Not enough historical data points for analysis. Try selecting a longer time period.")
else:
    st.info("Please enter a stock ticker symbol in the sidebar to begin analysis.")


st.sidebar.markdown("---")
st.sidebar.caption("""
**Disclaimer**: This tool provides analysis for educational purposes only. It's not financial advice.
Market data provided by Yahoo Finance.
""")

