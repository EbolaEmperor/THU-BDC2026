"""
Quick evaluation: predict on June 12, buy June 15, evaluate returns.
Since June 19 is Dragon Boat Festival (no trading), evaluate 3-day return:
  buy June 16 open, sell June 18 open (3 trading days)
"""
import os
import sys
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import joblib
from config import config
from model import StockTransformer
from predict import preprocess_predict_data

feature_cloums_map = {
    '39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
           'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
           'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
           'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
           'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'],

    '158+39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0', 'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60', 'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60', 'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60', 'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60', 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60', 'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread']
}


def main():
    print("=" * 60)
    print("PREDICT JUNE 18 → BUY JUNE 22 → SELL JUNE 26")
    print("=" * 60)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    features = feature_cloums_map[config['feature_num']]
    seq_len = config['sequence_length']
    output_dir = config['output_dir']

    # Load model
    raw_df = pd.read_csv(os.path.join(config['data_path'], 'train.csv'), dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])

    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    num_stocks = len(stockid2idx)

    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device))
    model.to(device)
    model.eval()

    scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))

    # Feature engineering
    print("\nFeature engineering...")
    processed, _ = preprocess_predict_data(raw_df, stockid2idx)
    processed['instrument_raw'] = processed['instrument'].copy()
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    processed[features] = scaler.transform(processed[features])

    # Predict for June 18 (last trading day before June 22 buy)
    pred_date = pd.Timestamp('2026-06-18')
    print(f"\nPredicting for {pred_date.date()}...")

    day_sequences = []
    day_stock_codes = []
    for sid in stock_ids:
        idx = stockid2idx[sid]
        hist = processed[
            (processed['instrument_raw'] == idx) &
            (processed['日期'] <= pred_date)
        ].sort_values('日期').tail(seq_len)

        if len(hist) == seq_len:
            day_sequences.append(hist[features].values.astype(np.float32))
            day_stock_codes.append(sid)

    print(f"Stocks with valid {seq_len}-day window: {len(day_sequences)}")

    if len(day_sequences) < 5:
        print("ERROR: Not enough stocks for prediction")
        return

    sequences = torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)
    stock_idx_list = [stockid2idx[sid] for sid in day_stock_codes]
    stock_idx_tensor = torch.LongTensor(stock_idx_list).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = model(sequences, stock_indices=stock_idx_tensor).squeeze(0).detach().cpu().numpy()

    order = np.argsort(scores)[::-1]
    top5_codes = [day_stock_codes[i] for i in order[:5]]
    top5_scores = [scores[i] for i in order[:5]]

    print(f"\n{'='*60}")
    print(f"TOP-5 PICKS (predict date: June 18, buy: June 22, sell: June 26)")
    print(f"{'='*60}")
    for i, (code, score) in enumerate(zip(top5_codes, top5_scores)):
        print(f"  #{i+1}: {code} (score: {score:.4f})")

    # Evaluate returns using raw price data
    price_df = pd.read_csv(os.path.join(config['data_path'], 'train.csv'), dtype={'股票代码': str})
    price_df['股票代码'] = price_df['股票代码'].astype(str).str.zfill(6)
    price_df['日期'] = pd.to_datetime(price_df['日期'])

    # Predict date is June 18. Next trading days after June 18:
    # June 19 = Dragon Boat (holiday), June 20-21 = weekend
    # T+1 = June 22 (Mon), T+2 = June 23, T+3 = June 24, T+4 = June 25, T+5 = June 26 (Fri)
    # Buy at T+1 (June 22) open, sell at T+5 (June 26) open
    # Dynamically find the actual trading days from price data
    pred_ts = pd.Timestamp('2026-06-18')
    future_dates = sorted(price_df['日期'].unique())
    future_dates = [d for d in future_dates if pd.Timestamp(d) > pred_ts]

    if len(future_dates) < 2:
        print(f"\n{'='*60}")
        print(f"EVALUATION SKIPPED: Future trading day data not yet available")
        print(f"(Need at least 2 future trading days after {pred_ts.date()})")
        print(f"Data currently ends at: {price_df['日期'].max().date()}")
        print(f"Run this script again after June 22 to see evaluation results.")
        print(f"{'='*60}")
        return

    buy_date = pd.Timestamp(future_dates[0])   # T+1 open
    sell_date = pd.Timestamp(future_dates[4]) if len(future_dates) >= 5 else pd.Timestamp(future_dates[-1])  # T+5 open

    if len(future_dates) < 5:
        print(f"  NOTE: Only {len(future_dates)} future trading days available, using last available as sell date")

    print(f"\n{'='*60}")
    print(f"RETURN EVALUATION: buy {str(buy_date)[:10]} open → sell {str(sell_date)[:10]} open")
    print(f"{'='*60}")

    buy_prices = price_df[price_df['日期'] == buy_date][['股票代码', '开盘']].set_index('股票代码')
    sell_prices = price_df[price_df['日期'] == sell_date][['股票代码', '开盘']].set_index('股票代码')

    if buy_prices.empty or sell_prices.empty:
        print("ERROR: Missing price data")
        return

    # Returns for all stocks
    common = buy_prices.index.intersection(sell_prices.index)
    all_returns = {}
    for sc in common:
        o_buy = buy_prices.loc[sc, '开盘']
        o_sell = sell_prices.loc[sc, '开盘']
        if o_buy > 1e-4:
            all_returns[sc] = (o_sell - o_buy) / o_buy

    # Model's top-5 returns
    pick_returns = [all_returns.get(c, 0.0) for c in top5_codes]
    portfolio_return = sum(pick_returns)

    # Optimal top-5
    sorted_all = sorted(all_returns.items(), key=lambda x: x[1], reverse=True)
    optimal_codes = [x[0] for x in sorted_all[:5]]
    optimal_returns = [x[1] for x in sorted_all[:5]]
    max_return = sum(optimal_returns)

    # Random expected
    random_return = 5 * np.mean(list(all_returns.values()))

    # Final score
    denom = max_return - random_return
    final_score = (portfolio_return - random_return) / (denom + 1e-12) if abs(denom) > 1e-6 else 0.0

    print(f"\n--- MODEL PICKS ---")
    for i, (code, ret) in enumerate(zip(top5_codes, pick_returns)):
        print(f"  #{i+1} {code}: {ret*100:+.2f}%")
    print(f"  Portfolio total: {portfolio_return*100:+.2f}%")

    print(f"\n--- OPTIMAL TOP-5 ---")
    for i, (code, ret) in enumerate(zip(optimal_codes, optimal_returns)):
        print(f"  #{i+1} {code}: {ret*100:+.2f}%")
    print(f"  Optimal total: {max_return*100:+.2f}%")

    print(f"\n--- RANDOM ---")
    print(f"  Expected (5 × mean): {random_return*100:+.2f}%")

    print(f"\n--- FINAL SCORE ---")
    print(f"  Score: {final_score:.4f}")
    print(f"  (1.0 = perfect, 0.0 = random, <0 = worse than random)")

    # Also show top-10 and top-20 model picks for reference
    print(f"\n--- TOP-20 MODEL PICKS (for reference) ---")
    for i in range(min(20, len(order))):
        code = day_stock_codes[order[i]]
        score = scores[order[i]]
        ret = all_returns.get(code, 0.0)
        marker = " ***" if code in top5_codes else ""
        print(f"  #{i+1:2d} {code}: score={score:.3f}, return={ret*100:+.2f}%{marker}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
