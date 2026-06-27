"""用 akshare 的网易(163)数据源补充下载缺失数据"""
import pandas as pd
import akshare as ak
import time
import random

# 读取已有数据
existing = pd.read_csv("data/stock_data.csv")
existing["日期"] = pd.to_datetime(existing["日期"])
last_date = existing["日期"].max()
print(f"已有数据最后一天: {last_date.date()}")

# 读取成分股列表
hs300 = pd.read_csv("data/hs300_stock_list.csv")

# 构建代码映射: 纯6位代码 -> 带sh/sz前缀的代码
code_map = {}
for _, row in hs300.iterrows():
    bs_code = row["code"]  # e.g. sh.600000 or sz.000001
    pure = bs_code.replace("sh.", "").replace("sz.", "").zfill(6)
    prefix = "sh" if bs_code.startswith("sh") else "sz"
    code_map[pure] = f"{prefix}{pure}"

# 找出需要更新的股票
need_update = []
existing["股票代码_str"] = existing["股票代码"].astype(str).str.zfill(6)
for code in code_map:
    stock_data = existing[existing["股票代码_str"] == code]
    if len(stock_data) == 0:
        need_update.append(code)
    else:
        stock_last = stock_data["日期"].max()
        if stock_last < pd.Timestamp("2026-06-18"):
            need_update.append(code)
existing = existing.drop(columns=["股票代码_str"])

print(f"需要更新: {len(need_update)} / {len(code_map)}")

start_date = "20260314"
end_date = "20260618"
print(f"下载范围: {start_date} ~ {end_date}")

new_records = []
failed = []
total = len(need_update)

for i, code in enumerate(need_update):
    symbol = code_map[code]
    retries = 3
    success = False

    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="hfq"
            )
            if df is None or df.empty:
                success = True
                break

            # 转换为 stock_data.csv 的格式
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
                "成交额": df["amount"].astype(float).values,
            })

            # 计算 preclose (前一日收盘价)
            preclose = df["close"].shift(1).astype(float).values
            mapped["振幅"] = ((df["high"].astype(float).values - df["low"].astype(float).values) / preclose * 100).round(2)
            mapped["涨跌额"] = (df["close"].astype(float).values - preclose).round(2)

            # 换手率用 163 提供的 turnover 列
            mapped["换手率"] = df["turnover"].astype(float).values * 100
            mapped["涨跌幅"] = ((df["close"].astype(float).values - preclose) / preclose * 100).round(2)

            # 第一行没有 preclose，跳过
            if len(mapped) > 1:
                mapped = mapped.iloc[1:]

            new_records.append(mapped)
            success = True
            break
        except Exception as e:
            if attempt < retries - 1:
                wait = (attempt + 1) * 2 + random.uniform(0, 1)
                time.sleep(wait)
            else:
                failed.append((code, str(e)[:100]))

    time.sleep(random.uniform(0.5, 1.5))

    if (i + 1) % 20 == 0 or not success:
        print(f"[{i+1}/{total}] 已处理，成功 {len(new_records)}，失败 {len(failed)}")

print(f"\n下载完成: 成功 {len(new_records)} 只，失败 {len(failed)} 只")

if new_records:
    new_df = pd.concat(new_records, ignore_index=True)

    # 确保日期格式一致
    existing["日期"] = pd.to_datetime(existing["日期"])
    new_df["日期"] = pd.to_datetime(new_df["日期"])

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["股票代码", "日期"], keep="last")
    combined = combined.sort_values(["股票代码", "日期"]).reset_index(drop=True)

    # 输出日期格式和原始CSV一致
    combined_out = combined.copy()
    combined_out["日期"] = combined_out["日期"].dt.strftime("%Y/%-m/%-d")
    combined_out.to_csv("data/stock_data.csv", index=False, encoding="utf-8-sig")

    print(f"\n合并后:")
    print(f"  总行数: {len(combined)}")
    print(f"  日期范围: {combined['日期'].min().date()} ~ {combined['日期'].max().date()}")
    print(f"  股票数: {combined['股票代码'].nunique()}")

if failed:
    print(f"\n失败列表 ({len(failed)} 只):")
    for code, err in failed[:10]:
        print(f"  {code}: {err}")
