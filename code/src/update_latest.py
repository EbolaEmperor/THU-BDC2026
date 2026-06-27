"""Download latest stock data to supplement June 19-26."""
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime

# Read existing data to know what we have
existing = pd.read_csv('../../data/stock_data.csv', dtype={'股票代码': str})
existing['股票代码'] = existing['股票代码'].astype(str).str.zfill(6)
existing['日期'] = pd.to_datetime(existing['日期'])
max_date = existing['日期'].max()
print(f"Existing data up to: {max_date.date()}")

# Get stock codes
stock_codes = sorted(existing['股票代码'].unique())
print(f"Stocks: {len(stock_codes)}")

# Download new data from akshare (NetEase source)
start = (max_date + pd.Timedelta(days=1)).strftime('%Y%m%d')
end = datetime.now().strftime('%Y%m%d')
print(f"Downloading from {start} to {end}...")

all_new = []
failed = []
for i, code in enumerate(stock_codes):
    try:
        # akshare NetEase source: symbol format sh600000 or sz000001
        if code.startswith('6'):
            symbol = f'sh{code}'
        else:
            symbol = f'sz{code}'
        
        df = ak.stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end, adjust="qfq")
        
        if df is not None and len(df) > 0:
            n = len(df)
            code_int = int(code)
            mapped = pd.DataFrame({
                "股票代码": [code_int] * n,
                "日期": pd.to_datetime(df["date"]),
                "开盘": df["open"].astype(float).values,
                "收盘": df["close"].astype(float).values,
                "最高": df["high"].astype(float).values,
                "最低": df["low"].astype(float).values,
                "成交量": df["volume"].astype(float).values,
                "成交额": df.get("amount", pd.Series([0.0]*n)).astype(float).values,
                "振幅": df.get("amplitude", pd.Series([0.0]*n)).astype(float).values,
                "涨跌额": df.get("change", pd.Series([0.0]*n)).astype(float).values,
                "换手率": df.get("turnover", pd.Series([0.0]*n)).astype(float).values,
                "涨跌幅": df.get("pct_change", pd.Series([0.0]*n)).astype(float).values,
            })
            all_new.append(mapped)
        
        if (i+1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(stock_codes)}")
    except Exception as e:
        failed.append(code)
        if len(failed) <= 5:
            print(f"  Failed {code}: {e}")

print(f"\nDownloaded: {len(all_new)} stocks with new data")
print(f"Failed: {len(failed)} stocks")

if all_new:
    new_df = pd.concat(all_new, ignore_index=True)
    new_df['日期'] = pd.to_datetime(new_df['日期'])
    print(f"New data dates: {new_df['日期'].min().date()} ~ {new_df['日期'].max().date()}")
    print(f"New rows: {len(new_df)}")
    
    # Merge with existing
    existing_int = existing.copy()
    existing_int['股票代码'] = existing_int['股票代码'].astype(int)
    
    merged = pd.concat([existing_int, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=['股票代码', '日期'], keep='last')
    merged = merged.sort_values(['股票代码', '日期']).reset_index(drop=True)
    
    print(f"\nMerged: {len(merged)} rows")
    print(f"Date range: {merged['日期'].min().date()} ~ {merged['日期'].max().date()}")
    print(f"Stocks: {merged['股票代码'].nunique()}")
    
    # Save
    merged.to_csv('../../data/stock_data_updated.csv', index=False)
    # Also overwrite train.csv
    merged.to_csv('../../data/train.csv', index=False)
    print(f"\nSaved to data/stock_data_updated.csv and data/train.csv")
    
    # Show last 5 trading days
    last_dates = sorted(merged['日期'].unique())[-5:]
    print(f"Last 5 trading days: {[str(d)[:10] for d in last_dates]}")
