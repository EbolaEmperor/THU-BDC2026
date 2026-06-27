"""用 akshare 补充下载 stock_data.csv 中缺失的数据（带重试和限流保护）"""
import pandas as pd
import akshare as ak
import time
import random

# 读取已有数据
existing = pd.read_csv("data/stock_data.csv")
existing["日期"] = pd.to_datetime(existing["日期"])
last_date = existing["日期"].max()
print(f"已有数据最后一天: {last_date.date()}")

# 读取沪深300成分股列表
hs300 = pd.read_csv("data/hs300_stock_list.csv")
stock_codes = hs300["code"].str.replace("sh.", "").str.replace("sz.", "").str.zfill(6).tolist()

# 找出哪些股票数据不完整（需要补充）
need_update = []
for code in stock_codes:
    stock_data = existing[existing["股票代码"].astype(str).str.zfill(6) == code]
    if len(stock_data) == 0:
        need_update.append(code)
    else:
        stock_last = stock_data["日期"].max()
        if stock_last < pd.Timestamp("2026-06-19"):
            need_update.append(code)

print(f"需要更新的股票: {len(need_update)} / {len(stock_codes)}")

start_date = "20260314"
end_date = "20260619"
print(f"补充下载范围: {start_date} ~ {end_date}")

new_records = []
failed = []
total = len(need_update)
success_count = 0

for i, code in enumerate(need_update):
    retries = 3
    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="hfq"
            )
            if df is None or df.empty:
                break  # 没有新数据，正常跳过

            mapped = pd.DataFrame()
            mapped["股票代码"] = code
            mapped["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y/%-m/%-d")
            mapped["开盘"] = df["开盘"].astype(float)
            mapped["收盘"] = df["收盘"].astype(float)
            mapped["最高"] = df["最高"].astype(float)
            mapped["最低"] = df["最低"].astype(float)
            mapped["成交量"] = df["成交量"].astype(float)
            mapped["成交额"] = df["成交额"].astype(float)

            preclose = df["收盘"].shift(1)
            mapped["振幅"] = ((df["最高"] - df["最低"]) / preclose * 100).round(2)
            mapped["涨跌额"] = (df["收盘"] - preclose).round(2)
            mapped["换手率"] = df["换手率"].astype(float)
            mapped["涨跌幅"] = df["涨跌幅"].astype(float)

            mapped = mapped.iloc[1:] if len(mapped) > 1 else mapped
            new_records.append(mapped)
            success_count += 1
            break  # 成功，跳出重试
        except Exception as e:
            if attempt < retries - 1:
                wait = (attempt + 1) * 3 + random.uniform(0, 2)
                print(f"  [{i+1}/{total}] {code} 重试 {attempt+1}/{retries}，等待 {wait:.1f}s...")
                time.sleep(wait)
            else:
                failed.append((code, str(e)[:80]))

    # 每次请求后等待，避免限流
    time.sleep(random.uniform(1.0, 2.0))

    if (i + 1) % 20 == 0:
        print(f"[{i+1}/{total}] 成功 {success_count}，失败 {len(failed)}")

print(f"\n下载完成: 成功 {success_count} 只，失败 {len(failed)} 只")

if new_records:
    new_df = pd.concat(new_records, ignore_index=True)
    new_df["日期"] = pd.to_datetime(new_df["日期"])

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["股票代码", "日期"], keep="last")
    combined = combined.sort_values(["股票代码", "日期"]).reset_index(drop=True)

    combined.to_csv("data/stock_data.csv", index=False, encoding="utf-8-sig")
    combined_min = combined["日期"].min()
    combined_max = combined["日期"].max()
    print(f"\n合并后:")
    print(f"  总行数: {len(combined)}")
    print(f"  日期范围: {combined_min.date()} ~ {combined_max.date()}")
    print(f"  股票数: {combined['股票代码'].nunique()}")

if failed:
    print(f"\n失败列表 ({len(failed)} 只):")
    for code, err in failed[:20]:
        print(f"  {code}: {err}")
