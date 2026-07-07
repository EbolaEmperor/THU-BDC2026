#!/usr/bin/env python3
"""
C4_high_lr 重新训练 + 预测脚本
1. 下载 6/19-6/26 的新数据 (baostock)
2. 合并到训练集
3. 用 C4_high_lr 配置训练
4. 预测 6/26 (买 6/29, 卖 7/3)
"""
import sys, os, time, json, copy
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace')

# ── 项目路径 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SRC_DIR = os.path.join(SCRIPT_DIR, 'src')
sys.path.insert(0, SRC_DIR)

import pandas as pd
import numpy as np

# ======================================================================
# STEP 1: 增量下载新数据
# ======================================================================
def download_new_data():
    """从 baostock 下载 2026-06-19 至 2026-06-26 的 HS300 数据"""
    import baostock as bs

    print("=" * 60)
    print("STEP 1: 下载新数据 (2026-06-19 ~ 2026-06-27)")
    print("=" * 60)

    lg = bs.login()
    if lg.error_code != '0':
        print(f"baostock 登录失败: {lg.error_msg}")
        return False

    try:
        # 获取 HS300 成分股
        rs = bs.query_hs300_stocks()
        stocks = []
        while rs.error_code == '0' and rs.next():
            stocks.append(rs.get_row_data())
        hs300_df = pd.DataFrame(stocks, columns=rs.fields)
        print(f"HS300 成分股: {len(hs300_df)} 只")

        start_date = "2026-06-19"
        end_date = "2026-06-27"  # extra day to be safe

        all_new = []
        failed = []

        for idx, row in hs300_df.iterrows():
            bs_code = row.get('code', '')
            try:
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
                    start_date=start_date, end_date=end_date,
                    frequency="d", adjustflag="1"
                )
                data_list = []
                while rs.error_code == '0' and rs.next():
                    data_list.append(rs.get_row_data())
                if not data_list:
                    continue

                df = pd.DataFrame(data_list, columns=rs.fields)
                numeric_cols = ['open','high','low','close','preclose','volume','amount','turn','pctChg']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df['振幅'] = ((df['high'] - df['low']) / df['preclose'] * 100).round(2)
                df['涨跌额'] = (df['close'] - df['preclose']).round(2)
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y/%-m/%-d')
                df['code'] = df['code'].str.replace('sh.','').str.replace('sz.','').str.zfill(6)

                df = df.rename(columns={
                    'code':'股票代码','date':'日期','open':'开盘','close':'收盘',
                    'high':'最高','low':'最低','volume':'成交量','amount':'成交额',
                    'turn':'换手率','pctChg':'涨跌幅'
                })
                columns = ['股票代码','日期','开盘','收盘','最高','最低',
                           '成交量','成交额','振幅','涨跌额','换手率','涨跌幅']
                df = df[columns]
                all_new.append(df)

                if (idx + 1) % 50 == 0:
                    print(f"  进度: {idx+1}/{len(hs300_df)}")
            except Exception as e:
                failed.append(bs_code)

        if all_new:
            new_df = pd.concat(all_new, ignore_index=True)
            print(f"\n下载完成: {len(new_df)} 条新记录, {len(new_df['股票代码'].unique())} 只股票")
            if failed:
                print(f"  失败: {len(failed)} 只股票")

            # 合并到现有数据
            stock_data_path = os.path.join(DATA_DIR, 'stock_data.csv')
            existing = pd.read_csv(stock_data_path, dtype={'股票代码': str})
            existing['股票代码'] = existing['股票代码'].astype(str).str.zfill(6)

            # 去重: 移除新数据中与现有数据重复的日期
            existing['日期_dt'] = pd.to_datetime(existing['日期'])
            new_df['日期_dt'] = pd.to_datetime(new_df['日期'])

            # 只保留真正新的数据
            existing_keys = set(zip(existing['股票代码'], existing['日期_dt']))
            new_df = new_df[~new_df.apply(lambda r: (r['股票代码'], r['日期_dt']) in existing_keys, axis=1)]

            if len(new_df) > 0:
                new_df = new_df.drop(columns=['日期_dt'])
                existing = existing.drop(columns=['日期_dt'])
                merged = pd.concat([existing, new_df], ignore_index=True)
                merged = merged.sort_values(['股票代码', '日期']).reset_index(drop=True)

                # 备份原文件
                backup_path = stock_data_path + '.bak'
                import shutil
                shutil.copy2(stock_data_path, backup_path)
                print(f"  已备份: {backup_path}")

                # 保存合并数据
                merged.to_csv(stock_data_path, index=False)
                # train.csv = stock_data.csv (保持一致)
                train_path = os.path.join(DATA_DIR, 'train.csv')
                shutil.copy2(stock_data_path, train_path)

                max_date = merged['日期'].max()
                print(f"  合并完成: {len(merged)} 条记录")
                print(f"  最新日期: {max_date}")
                return True
            else:
                print("  没有新的数据需要合并")
                return True
        else:
            print("  未下载到任何新数据")
            return False
    finally:
        bs.logout()


# ======================================================================
# STEP 2: 用 C4_high_lr 配置训练
# ======================================================================
def train_c4_model():
    """用 C4_high_lr 配置训练模型"""
    print("\n" + "=" * 60)
    print("STEP 2: C4_high_lr 训练 (全量数据)")
    print("=" * 60)

    from config import config
    from train import main as train_main

    # C4_high_lr 配置
    config['sequence_length'] = 60
    config['d_model'] = 256
    config['nhead'] = 4
    config['num_layers'] = 3
    config['dim_feedforward'] = 512
    config['batch_size'] = 2
    config['num_epochs'] = 30
    config['learning_rate'] = 5e-5
    config['dropout'] = 0.1
    config['feature_num'] = '158+39'
    config['max_grad_norm'] = 5.0
    config['pairwise_weight'] = 1
    config['base_weight'] = 1.0
    config['top5_weight'] = 2.0
    config['stock_emb_dim'] = 32
    config['seed'] = 42

    output_dir = os.path.join(SRC_DIR, 'model', '60_158+39', 'C4_retrain_full')
    config['output_dir'] = output_dir
    config['data_path'] = DATA_DIR

    os.makedirs(output_dir, exist_ok=True)

    scheduler_cfg = {
        'type': 'cosine',
        'T_max': 30,
        'eta_min': 1e-6,
    }

    print(f"  output_dir: {output_dir}")
    print(f"  data_path: {DATA_DIR}")
    print(f"  learning_rate: {config['learning_rate']}")
    print(f"  stock_emb_dim: {config['stock_emb_dim']}")
    print(f"  epochs: {config['num_epochs']}")

    start = time.time()
    train_main(scheduler_cfg=scheduler_cfg)
    elapsed = time.time() - start
    print(f"\n  训练完成! 耗时 {elapsed/60:.1f} 分钟")
    return output_dir


# ======================================================================
# STEP 3: 预测 6/26 (买 6/29, 卖 7/3)
# ======================================================================
def predict_new(output_dir):
    """用训练好的模型预测 2026-06-26"""
    print("\n" + "=" * 60)
    print("STEP 3: 预测 2026-06-26 (买 6/29, 卖 7/3)")
    print("=" * 60)

    from config import config
    from model import StockTransformer
    from predict import preprocess_predict_data
    import torch
    import joblib

    feature_cloums_map = {
        '158+39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
            'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0', 'VWAP0',
            'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5', 'STD10', 'STD20',
            'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30',
            'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60',
            'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60', 'QTLD5',
            'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60', 'RSV5', 'RSV10',
            'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60', 'IMIN5', 'IMIN10', 'IMIN20',
            'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60', 'CORR5', 'CORR10', 'CORR20', 'CORR30',
            'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60',
            'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60', 'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5',
            'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60', 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10',
            'SUMD20', 'SUMD30', 'SUMD60', 'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20',
            'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20',
            'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20',
            'VSUMD30', 'VSUMD60', 'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change',
            'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
            'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
            'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread']
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = feature_cloums_map[config['feature_num']]
    seq_len = config['sequence_length']

    raw_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    max_date = raw_df['日期'].max()
    print(f"  数据最新日期: {max_date}")

    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    num_stocks = len(stockid2idx)
    print(f"  股票数: {num_stocks}")

    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device))
    model.to(device)
    model.eval()
    scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))

    processed, _ = preprocess_predict_data(raw_df, stockid2idx)
    processed['instrument_raw'] = processed['instrument'].copy()
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    processed[features] = scaler.transform(processed[features])

    # 预测日期: 数据最新日期
    pred_date = max_date
    pred_date_str = pred_date.strftime('%Y-%m-%d')
    print(f"  预测日期: {pred_date_str}")

    day_seqs, day_codes = [], []
    for sid in stock_ids:
        idx = stockid2idx[sid]
        hist = processed[(processed['instrument_raw'] == idx) & (processed['日期'] <= pred_date)].sort_values('日期').tail(seq_len)
        if len(hist) == seq_len:
            day_seqs.append(hist[features].values.astype(np.float32))
            day_codes.append(sid)

    print(f"  有效股票数 (有完整60日窗口): {len(day_codes)}")

    seq_tensor = torch.FloatTensor(np.array(day_seqs)).unsqueeze(0).to(device)
    stock_idx_list = [stockid2idx[sid] for sid in day_codes]
    stock_idx_tensor = torch.LongTensor(stock_idx_list).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = model(seq_tensor, stock_indices=stock_idx_tensor).squeeze(0).detach().cpu().numpy()

    order = np.argsort(scores)[::-1]
    top5 = [(day_codes[i], scores[i]) for i in order[:5]]

    # 保存结果
    pred_path = os.path.join(output_dir, 'prediction_new.txt')
    with open(pred_path, 'w', encoding='utf-8') as f:
        f.write(f"=== C4_high_lr Retrained Prediction ===\n")
        f.write(f"Predict date: {pred_date_str} | Buy: 2026-06-29 | Sell: 2026-07-03\n")
        f.write(f"Model: C4_high_lr retrained on full data (incl. 6.22-6.26)\n\n")
        f.write(f"--- Top-5 (Recommended Portfolio) ---\n")
        for rank, (code, score) in enumerate(top5, 1):
            f.write(f"  #{rank}: {code} (score: {score:.4f})\n")
        f.write(f"\n--- Top-20 ---\n")
        for rank in range(min(20, len(order))):
            i = order[rank]
            f.write(f"  #{rank+1}: {day_codes[i]} (score: {scores[i]:.4f})\n")

    print(f"\n  === Top-5 推荐组合 ===")
    for rank, (code, score) in enumerate(top5, 1):
        print(f"  #{rank}: {code} (score: {score:.4f})")

    print(f"\n  预测结果已保存到: {pred_path}")
    return top5


# ======================================================================
# MAIN
# ======================================================================
if __name__ == '__main__':
    t0 = time.time()

    # Step 1: 下载新数据
    download_new_data()

    # Step 2: 训练
    output_dir = train_c4_model()

    # Step 3: 预测
    top5 = predict_new(output_dir)

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"全部完成! 总耗时 {total/60:.1f} 分钟")
    print(f"{'='*60}")
