#!/usr/bin/env python3
"""Simple script to fetch stock data via yfinance and save to CSV.

Usage example:
  python fetch_yfinance.py --tickers TCS.NS --period 6mo --interval 1d --output stocks.csv
"""
import argparse
from typing import List
import yfinance as yf
import pandas as pd


def fetch_and_save(tickers: List[str], period: str, interval: str, out_path: str) -> None:
    # Download; yfinance will return a DataFrame. For multiple tickers the DataFrame has a MultiIndex columns.
    data = yf.download(tickers=" ".join(tickers), period=period, interval=interval, group_by="ticker", auto_adjust=True, threads=True)

    if isinstance(data.columns, pd.MultiIndex):
        frames = []
        for t in tickers:
            try:
                df = data[t].copy()
            except Exception:
                # yfinance may return a different structure; try cross-section
                try:
                    df = data.xs(t, axis=1, level=0).copy()
                except Exception:
                    continue
            df = df.reset_index()
            df["Ticker"] = t
            frames.append(df)
        if not frames:
            raise RuntimeError("No data downloaded for given tickers")
        out_df = pd.concat(frames, ignore_index=True)
    else:
        out_df = data.copy().reset_index()
        # If single ticker was provided as a list, add ticker column
        out_df["Ticker"] = tickers[0] if len(tickers) == 1 else ",".join(tickers)

    out_df.to_csv(out_path, index=False)


def parse_args():
    p = argparse.ArgumentParser(description="Fetch stock data via yfinance and save CSV")
    p.add_argument("--tickers", "-t", nargs="+", required=True, help="One or more ticker symbols, e.g. AAPL MSFT")
    p.add_argument("--period", default="6mo", help="Data period (default: 6mo). Examples: 1d,5d,1mo,3mo,1y")
    p.add_argument("--interval", default="1d", help="Interval (default: 1d). Examples: 1m,5m,1d,1wk,1mo")
    p.add_argument("--output", "-o", default="stocks.csv", help="Output CSV path (default: stocks.csv)")
    return p.parse_args()


def main():
    args = parse_args()
    fetch_and_save(args.tickers, args.period, args.interval, args.output)
    print(f"Saved data to {args.output}")


if __name__ == "__main__":
    main()
