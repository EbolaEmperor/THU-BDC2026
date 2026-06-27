"""
Strict train/test evaluation script.
- Model trained on data <= June 5 (labels end at May 29 due to shift(-5))
- Evaluation on June 8-11 pick dates (June 12 needs June 22 data, unavailable)
- No data leakage: features are causal, scaler fit on training only
"""
import os
import sys
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39
from predict import preprocess_predict_data

feature_cloums_map = {
    '39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
           'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
           'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
           'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
           'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'],
    '158+39': None  # not used in this experiment
}

feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39,
}


def predict_for_date(model, combined_data, features, sequence_length, stock_ids, stockid2idx, pred_date, device, top_k=5):
    """Predict top-k stocks for a given prediction date."""
    model.eval()
    pred_date = pd.Timestamp(pred_date)

    day_sequences = []
    day_stock_codes = []

    for stock_code in stock_ids:
        stock_history = combined_data[
            (combined_data['instrument_raw'] == stockid2idx[stock_code]) &
            (combined_data['日期'] <= pred_date)
        ].sort_values('日期').tail(sequence_length)

        if len(stock_history) == sequence_length:
            seq = stock_history[features].values.astype(np.float32)
            day_sequences.append(seq)
            day_stock_codes.append(stock_code)

    if len(day_sequences) < top_k:
        return None

    sequences = torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)

    with torch.no_grad():
        scores = model(sequences).squeeze(0).detach().cpu().numpy()

    order = np.argsort(scores)[::-1]
    top5_codes = [day_stock_codes[i] for i in order[:top_k]]
    top5_scores = [scores[i] for i in order[:top_k]]

    return top5_codes, top5_scores, [day_stock_codes[i] for i in order]


def compute_returns(price_df, pick_date_str, stock_codes, all_stock_codes=None):
    """
    Compute returns for a pick date.
    Buy at T+1 open, sell at T+5 open.
    Returns: dict with portfolio_return, max_return, random_return
    """
    price_df = price_df.copy()
    price_df['日期'] = pd.to_datetime(price_df['日期'])

    # Find trading days after pick_date
    pick_date = pd.Timestamp(pick_date_str)
    future_dates = sorted(price_df['日期'].unique())
    future_dates = [d for d in future_dates if d > pick_date]

    if len(future_dates) < 5:
        print(f"  WARNING: Only {len(future_dates)} future trading days after {pick_date_str}, need 5")
        if len(future_dates) < 2:
            return None
        # Use whatever we have
        t1_date = future_dates[0]
        t5_date = future_dates[min(4, len(future_dates)-1)]
        print(f"  Using T+1={str(t1_date)[:10]}, T+5={str(t5_date)[:10]} (adjusted)")
    else:
        t1_date = future_dates[0]
        t5_date = future_dates[4]

    print(f"  Buy: {str(t1_date)[:10]}, Sell: {str(t5_date)[:10]}")

    # Get open prices for T+1 and T+5 for all stocks
    t1_data = price_df[price_df['日期'] == t1_date][['股票代码', '开盘']].set_index('股票代码')
    t5_data = price_df[price_df['日期'] == t5_date][['股票代码', '开盘']].set_index('股票代码')

    if t1_data.empty or t5_data.empty:
        print(f"  WARNING: Missing price data for some dates")
        return None

    # Compute returns for ALL stocks (for max and random calculation)
    common_stocks = t1_data.index.intersection(t5_data.index)
    all_returns = {}
    for sc in common_stocks:
        open_t1 = t1_data.loc[sc, '开盘']
        open_t5 = t5_data.loc[sc, '开盘']
        if open_t1 > 1e-4:
            all_returns[sc] = (open_t5 - open_t1) / open_t1

    if not all_returns:
        return None

    # Return for picked stocks
    pick_returns = [all_returns.get(sc, 0.0) for sc in stock_codes]
    portfolio_return = sum(pick_returns)

    # Max return (top-5 true returns)
    sorted_returns = sorted(all_returns.values(), reverse=True)
    max_return = sum(sorted_returns[:5])

    # Random expected return
    random_return = 5 * np.mean(list(all_returns.values()))

    # Final score
    denominator = max_return - random_return
    final_score = (portfolio_return - random_return) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0

    return {
        'portfolio_return': portfolio_return,
        'max_return': max_return,
        'random_return': random_return,
        'final_score': final_score,
        'pick_returns': pick_returns,
        'pick_codes': stock_codes,
        't1_date': str(t1_date)[:10],
        't5_date': str(t5_date)[:10],
    }


def main():
    print("=" * 60)
    print("STRICT TRAIN/TEST EVALUATION")
    print("Training: data <= 2026-06-05, labels <= 2026-05-29")
    print("Testing: pick dates 2026-06-08 to 2026-06-11")
    print("=" * 60)

    # Setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    sequence_length = config['sequence_length']
    output_dir = config['output_dir']
    model_path = os.path.join(output_dir, 'best_model.pth')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    feature_num = config['feature_num']
    features = feature_cloums_map[feature_num]

    # Load model
    print(f"\nLoading model from {model_path}")
    raw_train = pd.read_csv(os.path.join(config['data_path'], 'train.csv'), dtype={'股票代码': str})
    raw_train['股票代码'] = raw_train['股票代码'].astype(str).str.zfill(6)
    stock_ids = sorted(raw_train['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    num_stocks = len(stockid2idx)

    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    scaler = joblib.load(scaler_path)

    # Load and prepare data
    print("\nPreparing data...")
    raw_train['日期'] = pd.to_datetime(raw_train['日期'])
    raw_test = pd.read_csv(os.path.join(config['data_path'], 'test.csv'), dtype={'股票代码': str})
    raw_test['股票代码'] = raw_test['股票代码'].astype(str).str.zfill(6)
    raw_test['日期'] = pd.to_datetime(raw_test['日期'])

    # Feature engineering on training data (for scaler fitting context - scaler already fit)
    print("Feature engineering on training data...")
    train_processed, _ = preprocess_predict_data(raw_train, stockid2idx)
    train_processed[features] = train_processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Keep unscaled instrument for filtering (scaler transforms instrument too!)
    train_processed['instrument_raw'] = train_processed['instrument'].copy()
    train_processed[features] = scaler.transform(train_processed[features])

    # Feature engineering on test data
    print("Feature engineering on test data...")
    test_processed, _ = preprocess_predict_data(raw_test, stockid2idx)
    test_processed[features] = test_processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test_processed['instrument_raw'] = test_processed['instrument'].copy()
    test_processed[features] = scaler.transform(test_processed[features])

    # Combine processed train + test for prediction
    combined = pd.concat([train_processed, test_processed]).sort_values(['instrument_raw', '日期']).reset_index(drop=True)

    # Load full price data for return computation (no features, just raw prices)
    full_price = pd.read_csv(os.path.join('../../data', 'stock_data.csv'), dtype={'股票代码': str})
    full_price['股票代码'] = full_price['股票代码'].astype(str).str.zfill(6)
    full_price['日期'] = pd.to_datetime(full_price['日期'])

    # Evaluation dates (pick dates)
    eval_dates = ['2026-06-08', '2026-06-09', '2026-06-10', '2026-06-11']
    # June 12 skipped: T+5 = June 22, data unavailable

    print(f"\n{'=' * 60}")
    print(f"Evaluating on {len(eval_dates)} pick dates")
    print(f"{'=' * 60}")

    results = []
    for pick_date in eval_dates:
        print(f"\n--- Pick Date: {pick_date} ---")

        pred_result = predict_for_date(
            model, combined, features, sequence_length,
            stock_ids, stockid2idx, pick_date, device, top_k=5
        )

        if pred_result is None:
            print(f"  FAILED: Could not predict for {pick_date}")
            continue

        top5_codes, top5_scores, all_ranked = pred_result
        print(f"  Top-5 picks: {top5_codes}")
        print(f"  Top-5 scores: {[f'{s:.4f}' for s in top5_scores]}")

        ret = compute_returns(full_price, pick_date, top5_codes)
        if ret:
            print(f"  Period: {ret['t1_date']} -> {ret['t5_date']}")
            print(f"  Portfolio return (sum of 5): {ret['portfolio_return']*100:.2f}%")
            print(f"  Max return (best 5): {ret['max_return']*100:.2f}%")
            print(f"  Random expected (5*mean): {ret['random_return']*100:.2f}%")
            print(f"  Final Score: {ret['final_score']:.4f}")
            for i, (code, r) in enumerate(zip(ret['pick_codes'], ret['pick_returns'])):
                print(f"    #{i+1} {code}: {r*100:.2f}%")
            results.append({**ret, 'pick_date': pick_date})

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    if results:
        avg_portfolio = np.mean([r['portfolio_return'] for r in results])
        avg_max = np.mean([r['max_return'] for r in results])
        avg_random = np.mean([r['random_return'] for r in results])
        avg_score = np.mean([r['final_score'] for r in results])

        print(f"Days evaluated: {len(results)}")
        print(f"Avg portfolio return: {avg_portfolio*100:.2f}%")
        print(f"Avg max return: {avg_max*100:.2f}%")
        print(f"Avg random return: {avg_random*100:.2f}%")
        print(f"Avg Final Score: {avg_score:.4f}")
        print(f"\nInterpretation: Final Score of {avg_score:.4f} means the model captures")
        print(f"{avg_score*100:.1f}% of the gap between random selection and theoretical maximum.")

        # Save results
        result_df = pd.DataFrame(results)
        os.makedirs('output', exist_ok=True)
        result_df.to_csv('output/eval_results.csv', index=False)
        print(f"\nResults saved to output/eval_results.csv")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
