"""
Master script v2: runs all 5 variant trainings sequentially.
Fixed: no module reloading (causes pickle errors).
Just reset config dict between variants.
"""
import os
import sys
import time
import json
import copy
import traceback
import multiprocessing as mp

mp.set_start_method('spawn', force=True)

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CODE_DIR, 'src')
VARIANTS_DIR = os.path.join(CODE_DIR, 'variants')

os.chdir(SRC_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, VARIANTS_DIR)

MASTER_LOG = os.path.join(CODE_DIR, 'master_train_log.txt')

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(MASTER_LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

BASE_CONFIG = {
    'sequence_length': 60,
    'd_model': 256,
    'nhead': 4,
    'num_layers': 3,
    'dim_feedforward': 512,
    'batch_size': 2,
    'num_epochs': 15,
    'learning_rate': 1e-5,
    'dropout': 0.1,
    'feature_num': '158+39',
    'max_grad_norm': 5.0,
    'pairwise_weight': 1,
    'base_weight': 1.0,
    'top5_weight': 2.0,
    'output_dir': './model/60_158+39',
    'data_path': '../../data',
}

def reset_config():
    from config import config
    # Remove any extra keys added by previous variants
    keys_to_remove = [k for k in config if k not in BASE_CONFIG]
    for k in keys_to_remove:
        del config[k]
    config.update(copy.deepcopy(BASE_CONFIG))
    return config

VARIANTS = [
    {
        'name': 'A_30epoch',
        'overrides': {'num_epochs': 30},
        'scheduler': {'type': 'cosine', 'T_max': 30, 'eta_min': 1e-6},
        'criterion_kwargs': {},
    },
    {
        'name': 'B_loss_fix',
        'overrides': {'num_epochs': 30, 'train_label_mode': 'raw_return'},
        'scheduler': {'type': 'cosine', 'T_max': 30, 'eta_min': 1e-6},
        'criterion_kwargs': {'temperature': 0.3},
    },
    {
        'name': 'C_stock_emb',
        'overrides': {'num_epochs': 30, 'stock_emb_dim': 32},
        'scheduler': {'type': 'cosine', 'T_max': 30, 'eta_min': 1e-6},
        'criterion_kwargs': {},
    },
    {
        'name': 'D_deep_cross',
        'overrides': {'num_epochs': 30, 'cross_layers': 3},
        'scheduler': {'type': 'cosine', 'T_max': 30, 'eta_min': 1e-6},
        'criterion_kwargs': {},
    },
    {
        'name': 'E_combined',
        'overrides': {
            'num_epochs': 30,
            'stock_emb_dim': 32,
            'cross_layers': 3,
            'train_label_mode': 'raw_return',
        },
        'scheduler': {'type': 'cosine', 'T_max': 30, 'eta_min': 1e-6},
        'criterion_kwargs': {'temperature': 0.3},
    },
]


def run_variant(variant):
    import torch
    name = variant['name']

    log(f"{'='*60}")
    log(f"STARTING: {name}")
    log(f"{'='*60}")

    # Reset config (no module reload!)
    config = reset_config()
    for k, v in variant['overrides'].items():
        config[k] = v
    config['output_dir'] = f'./model/60_{config["feature_num"]}/{name}'
    os.makedirs(config['output_dir'], exist_ok=True)

    log(f"Config: epochs={config['num_epochs']}, overrides={variant['overrides']}")

    from train import main as train_main, WeightedRankingLoss

    # Build criterion
    criterion = None
    if variant['criterion_kwargs']:
        kw = variant['criterion_kwargs']
        criterion = WeightedRankingLoss(
            k=5, temperature=kw.get('temperature', 1.0),
            weight_factor=config['top5_weight'],
            pairwise_weight=config['pairwise_weight'],
            base_weight=config.get('base_weight', 1.0),
        )

    start = time.time()
    try:
        best_score = train_main(
            scheduler_cfg=variant['scheduler'],
            criterion_override=criterion,
        )
        elapsed = time.time() - start
        log(f"TRAINING DONE: {name} in {elapsed/60:.1f} min | Best: {best_score:.4f}")

        with open(os.path.join(config['output_dir'], 'DONE'), 'w') as f:
            f.write(f"{name} completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"best_score: {best_score:.6f}\n")

        # Prediction
        log(f"Running prediction for {name}...")
        from predict_helper import run_prediction
        top5 = run_prediction(config, name)
        log(f"Prediction top-5: {[t[0] for t in top5]}")
        log(f"VARIANT {name}: SUCCESS")
        return True

    except Exception as e:
        elapsed = time.time() - start
        log(f"VARIANT {name}: FAILED after {elapsed/60:.1f} min")
        log(f"Error: {e}")
        log(traceback.format_exc())

        # Try to recover CUDA state
        try:
            torch.cuda.empty_cache()
            log("CUDA cache cleared")
        except:
            pass

        return False


def main():
    with open(MASTER_LOG, 'w', encoding='utf-8') as f:
        f.write('')

    log("MASTER TRAINING SCRIPT v2 STARTED")
    log(f"Python: {sys.executable}")

    import torch
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        log("WARNING: No GPU!")

    # Check which variants already completed
    completed = set()
    for v in VARIANTS:
        done_path = os.path.join(SRC_DIR, f"model/60_158+39/{v['name']}/DONE")
        pred_path = os.path.join(SRC_DIR, f"model/60_158+39/{v['name']}/prediction.txt")
        if os.path.exists(done_path) and os.path.exists(pred_path):
            completed.add(v['name'])
            log(f"SKIP {v['name']}: already completed with predictions")

    results = {v['name']: {'success': v['name'] in completed, 'elapsed_min': 0, 'skipped': v['name'] in completed} for v in VARIANTS}
    total_start = time.time()

    for i, variant in enumerate(VARIANTS, 1):
        name = variant['name']
        if name in completed:
            continue

        log(f"\n>>> [{i}/{len(VARIANTS)}] {name}")
        v_start = time.time()
        success = run_variant(variant)
        v_elapsed = time.time() - v_start
        results[name] = {'success': success, 'elapsed_min': round(v_elapsed / 60, 1), 'skipped': False}

    total_elapsed = time.time() - total_start
    log(f"\n{'='*60}")
    log(f"ALL VARIANTS COMPLETE ({total_elapsed/60:.1f} min)")
    log(f"{'='*60}")
    for name, r in results.items():
        if r['skipped']:
            log(f"  {name}: SKIPPED (already done)")
        else:
            status = 'OK' if r['success'] else 'FAILED'
            log(f"  {name}: {status} ({r['elapsed_min']} min)")

    summary_path = os.path.join(CODE_DIR, 'TRAINING_SUMMARY.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'total_min': round(total_elapsed/60, 1),
                   'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)

    with open(os.path.join(CODE_DIR, 'ALL_DONE'), 'w') as f:
        f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == '__main__':
    main()
