from django.shortcuts import render
import yfinance as yf
import pandas as pd
import math

# --------------------------------
# BANK UNIVERSE (NSE)
# --------------------------------
BANK_STOCKS = {
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Bank of Baroda": "BANKBARODA.NS",
    "Punjab National Bank": "PNB.NS",
    "IDFC First Bank": "IDFCFIRSTB.NS",
    "Federal Bank": "FEDERALBNK.NS",
}

# --------------------------------
# HOME
# --------------------------------
def home(request):
    return render(request, "portfolio/home.html")

# --------------------------------
# BANK SECTOR TABLE
# --------------------------------
def bank_sector(request):
    banks = []

    for name, symbol in BANK_STOCKS.items():
        try:
            info = yf.Ticker(symbol).info
            banks.append({
                "name": name,
                "symbol": symbol.replace(".NS", ""),
                "ltp": info.get("currentPrice"),
                "high": info.get("dayHigh"),
                "low": info.get("dayLow"),
                "market_cap": round(info["marketCap"] / 1e7, 2) if info.get("marketCap") else None,
                "pe": info.get("trailingPE"),
            })
        except Exception:
            banks.append({"name": name, "symbol": symbol.replace(".NS", "")})

    return render(request, "portfolio/bank_sector.html", {"banks": banks})

# --------------------------------
# RSI CALCULATION
# --------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# --------------------------------
# STOCK DETAIL PAGE
# --------------------------------
def bank_detail(request, symbol):
    period = request.GET.get("period", "6mo")

    stock = yf.Ticker(symbol.upper() + ".NS")
    nifty = yf.Ticker("^NSEI")

    # -------- PRICE DATA --------
    df = stock.history(period=period)
    df_nifty = nifty.history(period=period)

    if df.empty or df_nifty.empty:
        return render(request, "portfolio/bank_detail.html", {
            "symbol": symbol.upper(),
            "period": period,
            "data": [],
            "quarterly": [],
            "pe_data": [],
            "quarterly_eps": [],
        })

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df["NIFTY_Close"] = df_nifty["Close"]
    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    df["MA30"] = df["Close"].rolling(30).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = calculate_rsi(df["Close"])
    df["RS_Line"] = df["Close"] / df["NIFTY_Close"]

    df.dropna(inplace=True)
    df["Date"] = df["Date"].astype(str)

    price_data = df.to_dict("records")

    # -------- PE RATIO --------
    pe_data = []
    eps = stock.info.get("trailingEps")

    if eps and isinstance(eps, (int, float)) and eps > 0:
        for r in price_data:
            pe = r["Close"] / eps
            if math.isfinite(pe):
                pe_data.append({"date": r["Date"], "pe": round(pe, 2)})

    # -------- QUARTERLY REVENUE --------
    quarterly = []
    q = stock.quarterly_income_stmt

    if q is not None and not q.empty:
        q = q.T.tail(12)
        for idx, row in q.iterrows():
            revenue = row.get("Total Revenue") or row.get("Interest Income")
            profit = row.get("Net Income")

            if pd.isna(revenue) or pd.isna(profit) or revenue <= 0:
                continue

            quarter = f"{idx.year}-Q{((idx.month - 1)//3) + 1}"

            quarterly.append({
                "quarter": quarter,
                "revenue": round(float(revenue) / 1e7, 2),
                "profit": round(float(profit) / 1e7, 2),
                "margin": round((profit / revenue) * 100, 2),
            })

    # -------- QUARTERLY EPS (LAST 3 YEARS) --------
    quarterly_eps = []
    shares = stock.info.get("sharesOutstanding")

    if q is not None and shares and shares > 0:
        for idx, row in q.iterrows():
            ni = row.get("Net Income")
            if pd.isna(ni):
                continue

            eps_val = ni / shares
            quarter = f"{idx.year}-Q{((idx.month - 1)//3) + 1}"

            quarterly_eps.append({
                "quarter": quarter,
                "eps": round(float(eps_val), 2)
            })

    return render(request, "portfolio/bank_detail.html", {
        "symbol": symbol.upper(),
        "period": period,
        "data": price_data,
        "quarterly": quarterly,
        "pe_data": pe_data,
        "quarterly_eps": quarterly_eps,
    })