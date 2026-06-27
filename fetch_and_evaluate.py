"""
Fetch 6.22-6.26 data for HS300 stocks and evaluate all 5 model predictions.
Stores data separately in data/eval_20260622_20260626.csv
"""
import baostock as bs
import pandas as pd
import numpy as np
import os, sys, json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
EVAL_FILE = os.path.join(DATA_DIR, 'eval_20260622_20260626.csv')

# All unique stocks predicted across 5 variants
PREDICTIONS = {
    'A_30epoch': {
        'top5': ['000977', '000938', '600188', '600026', '000338'],
        'top20': ['000977','000938','600188','600026','000338','600482','600875','600588',
                   '600460','002049','300502','000975','600489','002028','000063','000807',
                   '002493','600018','002384','605117'],
        'val_score': 0.1104,
    },
    'B_loss_fix': {
        'top5': ['000977', '600026', '000408', '600188', '000938'],
        'top20': ['000977','600026','000408','600188','000938','600460','000063','000975',
                   '600845','600875','002049','600011','605117','002384','000338','600999',
                   '601916','600183','000425','600018'],
        'val_score': 0.1178,
    },
    'C_stock_emb': {
        'top5': ['600584', '600183', '688981', '600018', '601298'],
        'top20': ['600584','600183','688981','600018','601298','600522','688256','688008',
                   '600039','603296','600176','603986','688041','601211','601901','002916',
                   '603993','300408','688126','601288'],
        'val_score': 0.1686,
    },
    'D_deep_cross': {
        'top5': ['601018', '601328', '601818', '601166', '601398'],
        'top20': ['601018','601328','601818','601166','601398','600019','002601','000708',
                   '601169','600926','600104','600016','600061','601825','601238','600372',
                   '601658','601318','600025','600000'],
        'val_score': 0.2066,
    },
    'E_combined': {
        'top5': ['600039', '601899', '600150', '000425', '600547'],
        'top20': ['600039','601899','600150','000425','600547','600845','000630','000983',
                   '600875','601288','600900','300803','002049','002916','600460','600176',
                   '000408','600183','002142','000792'],
        'val_score': 0.1823,
    },
}

def code_to_bs(code):
    """Convert 6-digit stock code to baostock format (sh./sz.)"""
    if code.startswith(('6', '9')):
        return f'sh.{code}'
    else:
        return f'sz.{code}'

def fetch_eval_data():
    """Fetch all HS300 stocks data for 6.22-6.26"""
    # Load HS300 stock list
    hs300_file = os.path.join(DATA_DIR, 'hs300_stock_list.csv')
    if os.path.exists(hs300_file):
        hs300 = pd.read_csv(hs300_file)
        codes = hs300.iloc[:, 0].astype(str).str.zfill(6).tolist()
    else:
        # Fallback: get from baostock
        lg = bs.login()
        rs = bs.query_hs300_stocks(date='2026-06-18')
        codes = []
        while rs.next():
            codes.append(rs.get_row_data()[1])  # code column
        bs.logout()
        codes = [c.split('.')[1] for c in codes]

    # Collect all unique predicted codes too
    all_pred_codes = set()
    for v in PREDICTIONS.values():
        all_pred_codes.update(v['top20'])

    # Merge: fetch all HS300 + all predicted
    fetch_codes = list(set(codes) | all_pred_codes)
    fetch_codes.sort()

    lg = bs.login()
    print(f'Fetching data for {len(fetch_codes)} stocks, 2026-06-22 to 2026-06-26...')

    all_data = []
    for i, code in enumerate(fetch_codes):
        bs_code = code_to_bs(code)
        rs = bs.query_history_k_data_plus(bs_code,
            'date,code,open,high,low,close,preclose,volume,amount,turn,pctChg',
            start_date='2026-06-22', end_date='2026-06-26',
            frequency='d', adjustflag='1')  # back-adjusted
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if rows:
            df = pd.DataFrame(rows, columns=rs.fields)
            df['股票代码'] = code
            all_data.append(df)
        if (i+1) % 50 == 0:
            print(f'  {i+1}/{len(fetch_codes)} done')

    bs.logout()
    result = pd.concat(all_data, ignore_index=True)
    result.to_csv(EVAL_FILE, index=False, encoding='utf-8-sig')
    print(f'Saved {len(result)} rows to {EVAL_FILE}')
    return result

def evaluate(df):
    """Evaluate all model predictions against actual returns"""
    df = df.copy()
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])

    # Get open prices for 6.22 (buy day) and 6.26 (sell day)
    buy_date = df['date'].min()  # 6.22
    sell_date = df['date'].max()  # 6.26

    print(f'\nBuy date: {buy_date.date()}, Sell date: {sell_date.date()}')

    buy_prices = df[df['date'] == buy_date].set_index('股票代码')['open'].to_dict()
    sell_prices = df[df['date'] == sell_date].set_index('股票代码')['open'].to_dict()

    # Also get all available returns for random/max baselines
    all_returns = {}
    for code in buy_prices:
        if code in sell_prices and buy_prices[code] > 0:
            all_returns[code] = (sell_prices[code] - buy_prices[code]) / buy_prices[code]

    print(f'Stocks with valid prices: {len(all_returns)}')

    # Random baseline: average return of all stocks
    random_return = np.mean(list(all_returns.values())) if all_returns else 0
    # Max possible: top-5 best returns
    sorted_returns = sorted(all_returns.items(), key=lambda x: x[1], reverse=True)
    max_top5 = sorted_returns[:5]
    max_return = np.mean([r for _, r in max_top5])

    print(f'Random baseline (avg all): {random_return*100:.2f}%')
    print(f'Max possible (best 5): {max_return*100:.2f}%')
    print(f'Max top-5 stocks: {[c for c,_ in max_top5]}\n')

    results = {}
    for variant_name, info in PREDICTIONS.items():
        top5 = info['top5']
        top20 = info['top20']

        # Calculate top-5 returns
        top5_returns = []
        top5_details = []
        for code in top5:
            if code in all_returns:
                ret = all_returns[code]
                top5_returns.append(ret)
                buy_p = buy_prices.get(code, 0)
                sell_p = sell_prices.get(code, 0)
                top5_details.append((code, ret, buy_p, sell_p))
            else:
                top5_details.append((code, None, 0, 0))

        pred_return = np.mean(top5_returns) if top5_returns else 0

        # Competition metric: (pred - random) / (max - random)
        if abs(max_return - random_return) > 1e-9:
            metric = (pred_return - random_return) / (max_return - random_return)
        else:
            metric = 0

        # Also evaluate top-20
        top20_returns = []
        for code in top20:
            if code in all_returns:
                top20_returns.append(all_returns[code])
        top20_avg = np.mean(top20_returns) if top20_returns else 0

        results[variant_name] = {
            'val_score': info['val_score'],
            'top5_return': pred_return,
            'top20_return': top20_avg,
            'metric': metric,
            'top5_details': top5_details,
            'n_valid': len(top5_returns),
        }

        print(f'=== {variant_name} (val={info["val_score"]:.4f}) ===')
        print(f'  Top-5 avg return: {pred_return*100:.2f}%')
        print(f'  Top-20 avg return: {top20_avg*100:.2f}%')
        print(f'  Competition metric: {metric:.4f}')
        for code, ret, buy_p, sell_p in top5_details:
            if ret is not None:
                print(f'    {code}: {ret*100:+.2f}% (buy={buy_p:.2f} -> sell={sell_p:.2f})')
            else:
                print(f'    {code}: NO DATA')
        print()

    # Save results
    summary_path = os.path.join(BASE_DIR, 'EVALUATION_RESULTS.json')
    save_results = {}
    for k, v in results.items():
        save_results[k] = {
            'val_score': v['val_score'],
            'top5_return_pct': round(v['top5_return'] * 100, 4),
            'top20_return_pct': round(v['top20_return'] * 100, 4),
            'metric': round(v['metric'], 4),
            'top5_details': [(c, round(r*100, 4) if r is not None else None) for c, r, _, _ in v['top5_details']],
        }
    save_results['_baselines'] = {
        'random_return_pct': round(random_return * 100, 4),
        'max_return_pct': round(max_return * 100, 4),
        'max_top5_stocks': [c for c, _ in max_top5],
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f'Results saved to {summary_path}')

    return results, random_return, max_return

if __name__ == '__main__':
    df = fetch_eval_data()
    evaluate(df)
