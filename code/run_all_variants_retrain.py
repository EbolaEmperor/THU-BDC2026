#!/usr/bin/env python3
"""
批量重训练所有变体 (C2/C3/C5/C6/C7) 并预测 6/29买入-7/3卖出
数据已包含至 6/26
"""
import sys, os, time, json, copy, shutil
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SRC_DIR = os.path.join(SCRIPT_DIR, 'src')
sys.path.insert(0, SRC_DIR)

import pandas as pd
import numpy as np

# ======================================================================
# 变体配置
# ======================================================================
VARIANTS = {
    'C2_emb64': {
        'stock_emb_dim': 64,
        'learning_rate': 1e-5,
        'dropout': 0.1,
        'd_model': 256, 'nhead': 4, 'dim_feedforward': 512,
        'top5_weight': 2.0, 'pairwise_weight': 1,
    },
    'C3_high_dropout': {
        'stock_emb_dim': 32,
        'learning_rate': 1e-5,
        'dropout': 0.3,
        'd_model': 256, 'nhead': 4, 'dim_feedforward': 512,
        'top5_weight': 2.0, 'pairwise_weight': 1,
    },
    # C4_high_lr 已训练完成, +2.24%
    'C5_multi_seed': {
        'stock_emb_dim': 32,
        'learning_rate': 1e-5,
        'dropout': 0.1,
        'd_model': 256, 'nhead': 4, 'dim_feedforward': 512,
        'top5_weight': 2.0, 'pairwise_weight': 1,
        'multi_seed': True,
        'seeds': [42, 123, 456],
    },
    'C6_wider_model': {
        'stock_emb_dim': 32,
        'learning_rate': 1e-5,
        'dropout': 0.1,
        'd_model': 384, 'nhead': 6, 'dim_feedforward': 768,
        'top5_weight': 2.0, 'pairwise_weight': 1,
    },
    'C7_loss_tune': {
        'stock_emb_dim': 32,
        'learning_rate': 1e-5,
        'dropout': 0.1,
        'd_model': 256, 'nhead': 4, 'dim_feedforward': 512,
        'top5_weight': 5.0, 'pairwise_weight': 2,
    },
}

# 特征列表 (158+39)
FEATURE_COLS = ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
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


def set_base_config(variant_cfg):
    """设置 config 对象"""
    from config import config
    config['sequence_length'] = 60
    config['d_model'] = variant_cfg.get('d_model', 256)
    config['nhead'] = variant_cfg.get('nhead', 4)
    config['num_layers'] = 3
    config['dim_feedforward'] = variant_cfg.get('dim_feedforward', 512)
    config['batch_size'] = 2
    config['num_epochs'] = 30
    config['learning_rate'] = variant_cfg.get('learning_rate', 1e-5)
    config['dropout'] = variant_cfg.get('dropout', 0.1)
    config['feature_num'] = '158+39'
    config['max_grad_norm'] = 5.0
    config['pairwise_weight'] = variant_cfg.get('pairwise_weight', 1)
    config['base_weight'] = 1.0
    config['top5_weight'] = variant_cfg.get('top5_weight', 2.0)
    config['stock_emb_dim'] = variant_cfg.get('stock_emb_dim', 32)
    config['seed'] = variant_cfg.get('seed', 42)
    config['data_path'] = DATA_DIR
    return config


def train_variant(variant_name, variant_cfg):
    """训练一个变体"""
    import torch
    from train import main as train_main

    print(f"\n{'='*60}")
    print(f"训练: {variant_name}")
    print(f"{'='*60}")

    if variant_cfg.get('multi_seed'):
        # C5: 多种子训练
        seeds = variant_cfg['seeds']
        all_top20 = {}  # seed -> top-20 list
        best_val = float('-inf')
        best_seed = None

        for seed in seeds:
            print(f"\n  --- Seed {seed} ---")
            variant_cfg_copy = dict(variant_cfg)
            variant_cfg_copy['seed'] = seed

            output_dir = os.path.join(SRC_DIR, 'model', '60_158+39', f'{variant_name}_seed{seed}')
            set_base_config(variant_cfg_copy)

            from config import config
            config['output_dir'] = output_dir
            os.makedirs(output_dir, exist_ok=True)

            scheduler_cfg = {'type': 'cosine', 'T_max': 30, 'eta_min': 1e-6}

            t0 = time.time()
            train_main(scheduler_cfg=scheduler_cfg)
            elapsed = time.time() - t0
            print(f"  Seed {seed} 完成, 耗时 {elapsed/60:.1f} 分钟")

            # 预测 top-20
            top20, val_score = predict_top20(output_dir, seed_label=f'seed{seed}')
            all_top20[seed] = top20

            if val_score is not None and val_score > best_val:
                best_val = val_score
                best_seed = seed

        # 求交集
        sets = [set(s) for s in all_top20.values()]
        intersection = sets[0]
        for s in sets[1:]:
            intersection = intersection & s

        print(f"\n  Top-20 交集: {len(intersection)} 只股票")
        if len(intersection) >= 5:
            # 从交集中选 top-5 (按第一个seed的分数排序)
            seed0 = seeds[0]
            top5 = [s for s in all_top20[seed0] if s in intersection][:5]
            print(f"  交集足够, 取 top-5: {top5}")
        else:
            # 用 best seed
            print(f"  交集不足, 使用 best seed {best_seed} (val={best_val:.4f})")
            top5 = all_top20[best_seed][:5]

        # 保存最终结果
        final_dir = os.path.join(SRC_DIR, 'model', '60_158+39', f'{variant_name}_final')
        os.makedirs(final_dir, exist_ok=True)
        pred_path = os.path.join(final_dir, 'prediction_retrain.txt')
        with open(pred_path, 'w', encoding='utf-8') as f:
            f.write(f"=== {variant_name} Prediction ===\n")
            f.write(f"Seeds: {seeds} | Intersection size: {len(intersection)}\n")
            f.write(f"Best seed: {best_seed} (val={best_val:.4f})\n\n")
            f.write(f"--- Top-5 ---\n")
            for rank, code in enumerate(top5, 1):
                f.write(f"  #{rank}: {code}\n")

        return top5, final_dir

    else:
        # 单子训练
        output_dir = os.path.join(SRC_DIR, 'model', '60_158+39', f'{variant_name}_retrain')
        set_base_config(variant_cfg)

        from config import config
        config['output_dir'] = output_dir
        os.makedirs(output_dir, exist_ok=True)

        scheduler_cfg = {'type': 'cosine', 'T_max': 30, 'eta_min': 1e-6}

        for k, v in variant_cfg.items():
            if k in ('d_model','nhead','dim_feedforward','learning_rate','dropout','stock_emb_dim','top5_weight','pairwise_weight'):
                print(f"  {k}: {v}")

        t0 = time.time()
        train_main(scheduler_cfg=scheduler_cfg)
        elapsed = time.time() - t0
        print(f"\n  训练完成! 耗时 {elapsed/60:.1f} 分钟")

        top5, _ = predict_top20(output_dir)
        top5_only = top5[:5]

        # 保存 top-5
        pred_path = os.path.join(output_dir, 'prediction_retrain.txt')
        with open(pred_path, 'w', encoding='utf-8') as f:
            f.write(f"=== {variant_name} Prediction ===\n")
            f.write(f"Predict date: 2026-06-26 | Buy: 2026-06-29 | Sell: 2026-07-03\n\n")
            f.write(f"--- Top-5 ---\n")
            for rank, code in enumerate(top5_only, 1):
                f.write(f"  #{rank}: {code}\n")
            f.write(f"\n--- Top-20 ---\n")
            for rank, code in enumerate(top5, 1):
                f.write(f"  #{rank}: {code}\n")

        return top5_only, output_dir


def predict_top20(output_dir, seed_label=''):
    """预测 top-20 股票"""
    import torch
    import joblib
    from config import config
    from model import StockTransformer
    from predict import preprocess_predict_data

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = FEATURE_COLS
    seq_len = config['sequence_length']

    raw_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    max_date = raw_df['日期'].max()

    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    num_stocks = len(stockid2idx)

    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device))
    model.to(device)
    model.eval()
    scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))

    processed, _ = preprocess_predict_data(raw_df, stockid2idx)
    processed['instrument_raw'] = processed['instrument'].copy()
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    processed[features] = scaler.transform(processed[features])

    pred_date = max_date
    day_seqs, day_codes = [], []
    for sid in stock_ids:
        idx = stockid2idx[sid]
        hist = processed[(processed['instrument_raw'] == idx) & (processed['日期'] <= pred_date)].sort_values('日期').tail(seq_len)
        if len(hist) == seq_len:
            day_seqs.append(hist[features].values.astype(np.float32))
            day_codes.append(sid)

    seq_tensor = torch.FloatTensor(np.array(day_seqs)).unsqueeze(0).to(device)
    stock_idx_list = [stockid2idx[sid] for sid in day_codes]
    stock_idx_tensor = torch.LongTensor(stock_idx_list).unsqueeze(0).to(device)

    with torch.no_grad():
        scores = model(seq_tensor, stock_indices=stock_idx_tensor).squeeze(0).detach().cpu().numpy()

    order = np.argsort(scores)[::-1]
    top20 = [day_codes[i] for i in order[:20]]

    # 尝试读取 val score
    val_score = None
    log_path = os.path.join(output_dir, 'train.log')
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if 'best_val' in line.lower() or 'Best val' in line:
                    try:
                        val_score = float(line.split(':')[-1].strip())
                    except:
                        pass

    label = f" ({seed_label})" if seed_label else ""
    print(f"  Top-5{label}: {top20[:5]}")

    return top20, val_score


# ======================================================================
# MAIN
# ======================================================================
if __name__ == '__main__':
    t_total = time.time()
    results = {}  # variant_name -> top-5 list

    # C4_high_lr 已知结果
    results['C4_high_lr'] = ['600039', '600938', '601077', '300251', '600050']

    for name, cfg in VARIANTS.items():
        print(f"\n{'#'*60}")
        print(f"# 开始处理: {name}")
        print(f"{'#'*60}")

        # 跳过已完成
        check_dir = os.path.join(SRC_DIR, 'model', '60_158+39')
        if cfg.get('multi_seed'):
            pred_file = os.path.join(check_dir, f'{name}_final', 'prediction_retrain.txt')
        else:
            pred_file = os.path.join(check_dir, f'{name}_retrain', 'prediction_retrain.txt')

        if os.path.exists(pred_file):
            print(f"  已存在预测结果, 跳过训练: {pred_file}")
            with open(pred_file, 'r', encoding='utf-8') as f:
                content = f.read()
            # 解析 top-5
            top5 = []
            in_top5 = False
            for line in content.split('\n'):
                if '--- Top-5 ---' in line:
                    in_top5 = True
                    continue
                if in_top5 and line.strip().startswith('#'):
                    code = line.split(':')[1].strip().split()[0].strip()
                    top5.append(code)
                    if len(top5) >= 5:
                        break
            if len(top5) == 5:
                results[name] = top5
                print(f"  已有 Top-5: {top5}")
                continue
            else:
                print(f"  解析失败 ({len(top5)} 只), 重新训练")

        try:
            top5, output_dir = train_variant(name, cfg)
            results[name] = top5
            print(f"\n  >>> {name} Top-5: {top5}")
        except Exception as e:
            print(f"\n  >>> {name} 训练失败: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    print(f"\n{'='*60}")
    print(f"所有变体训练完成! 总耗时 {(time.time()-t_total)/60:.1f} 分钟")
    print(f"{'='*60}")
    print(f"\n{'='*60}")
    print("各变体 Top-5 预测汇总:")
    print(f"{'='*60}")
    for name, top5 in results.items():
        print(f"  {name}: {top5}")

    # 保存汇总
    summary_path = os.path.join(SCRIPT_DIR, 'retrain_all_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {summary_path}")
