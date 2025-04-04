import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Health Check", layout="wide")
st.title("📊 Portfolio Health Check")

# --- Portfolio Input Section ---
st.subheader("Enter Your Portfolio")

default_data = {
    "Stock Ticker (e.g., TCS.NS)": ["TCS.NS", "INFY.NS"],
    "Quantity": [10, 5],
    "Buy Price": [3100, 1400]
}

portfolio_df = st.data_editor(pd.DataFrame(default_data), num_rows="dynamic", use_container_width=True)

if st.button("Analyze Portfolio"):

    st.subheader("📈 Portfolio Analysis")

    result_data = []
    total_value = 0
    sector_data = {}

    for _, row in portfolio_df.iterrows():
        ticker = row["Stock Ticker (e.g., TCS.NS)"]
        qty = row["Quantity"]
        buy_price = row["Buy Price"]

        try:
            stock = yf.Ticker(ticker)
            current_price = stock.history(period="1d")["Close"].iloc[-1]
            info = stock.info
            sector = info.get("sector", "Unknown")

            hist = stock.history(period="6mo")
            if hist.empty:
                raise Exception("No history available")

            # Moving Averages
            ma50 = hist["Close"].rolling(window=50).mean().iloc[-1]
            ma200 = hist["Close"].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else None

            # RSI
            delta = hist["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = round(rsi.iloc[-1], 2)

            # MACD
            ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
            ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_value = round(macd.iloc[-1], 2)
            signal_value = round(signal.iloc[-1], 2)
            macd_status = "Bullish (MACD > Signal)" if macd_value > signal_value else "Bearish (MACD < Signal)"

            # Bollinger Bands
            rolling_mean = hist["Close"].rolling(window=20).mean()
            rolling_std = hist["Close"].rolling(window=20).std()
            upper_band = rolling_mean + (rolling_std * 2)
            lower_band = rolling_mean - (rolling_std * 2)
            price_position = (
                "near upper band → Possible overbought" if current_price > upper_band.iloc[-1] * 0.97 else
                "near lower band → Possible oversold" if current_price < lower_band.iloc[-1] * 1.03 else
                "near middle band → Balanced"
            )

            # Combine technical insight
            trend = "Uptrend" if current_price > ma50 and (ma200 is None or current_price > ma200) else "Downtrend"
            rsi_status = (
                f"{rsi_value} → Overbought" if rsi_value > 70 else
                f"{rsi_value} → Oversold" if rsi_value < 30 else
                f"{rsi_value} → Neutral"
            )

            insight = (
                f"- Trend: {trend} (Price above MA50{' and MA200' if ma200 else ''})\n"
                f"- RSI: {rsi_status}\n"
                f"- MACD: {macd_status} (MACD={macd_value}, Signal={signal_value})\n"
                f"- Bollinger: {price_position}"
            )

        except Exception as e:
            st.warning(f"Could not fetch data for {ticker} — {e}")
            continue

        market_value = current_price * qty
        gain_loss = (current_price - buy_price) * qty

        result_data.append({
            "Ticker": ticker,
            "Sector": sector,
            "Quantity": qty,
            "Buy Price": buy_price,
            "Current Price": round(current_price, 2),
            "Market Value": round(market_value, 2),
            "Gain/Loss": round(gain_loss, 2),
            "Technical Insight": insight
        })

        total_value += market_value
        sector_data[sector] = sector_data.get(sector, 0) + market_value

    if not result_data:
        st.warning("No valid stocks to analyze.")
        st.stop()

    # Final portfolio dataframe
    result_df = pd.DataFrame(result_data)
    result_df["Allocation (%)"] = round(result_df["Market Value"] / total_value * 100, 2)

    # Show table
    st.dataframe(result_df[["Ticker", "Quantity", "Buy Price", "Current Price", "Market Value",
                            "Gain/Loss", "Allocation (%)"]], use_container_width=True)

    # Technical Insights Section
    st.subheader("📌 Technical Insights")
    for _, row in result_df.iterrows():
        with st.expander(f"{row['Ticker']}"):
            st.markdown(row["Technical Insight"])

    # Charts Section
    st.subheader("📊 Portfolio Allocation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Stock Allocation")
        fig1, ax1 = plt.subplots()
        ax1.pie(result_df["Market Value"], labels=result_df["Ticker"], autopct="%1.1f%%", startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)

    with col2:
        st.markdown("### Sector Exposure")
        sector_df = pd.DataFrame(list(sector_data.items()), columns=["Sector", "Value"])
        fig2, ax2 = plt.subplots()
        ax2.pie(sector_df["Value"], labels=sector_df["Sector"], autopct="%1.1f%%", startangle=90)
        ax2.axis("equal")
        st.pyplot(fig2)

    st.success("✅ Portfolio analyzed successfully!")
