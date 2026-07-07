"""
Master script v3: runs 6 new C-based variant trainings sequentially.
All variants based on C_stock_emb (stock identity embedding).
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

MASTER_LOG = os.path.join(CODE_DIR, 'master_v3_log.txt')

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
    'num_epochs': 30,
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
    keys_to_remove = [k for k in config if k not in BASE_CONFIG]
    for k in keys_to_remove:
        del config[k]
    config.update(copy.deepcopy(BASE_CONFIG))
    return config

VARIANTS = [
    {
        'name': 'C2_emb64',
        'branch': 'c2-emb64',
        'overrides': {
            'stock_emb_dim': 64,  # larger embedding (C uses 32)
        },
        'scheduler': {'type': 'cosine', 'T_max': 30, 'eta_min': 1e-6},
        'criterion_kwargs': {},
    },
    {
        'name': 'C3_high_dropout',
        'branch': 'c3-high-dropout',
        'overrides': {
            'stock_emb_dim': 32,
            'dropout': 0.3,  # higher dropout for regularization (C uses 0.1)
        },
        'scheduler': {'type': 'cosine', 'T_max': 30, 'eta_min': 1e-6},
        'criterion_kwargs': {},
    },
    {
        'name': 'C4_high_lr',
        'branch': 'c4-high-lr',
        'overrides': {
            'stock_emb_dim': 32,
            'learning_rate': 5e-5,  # 5x higher LR (C uses 1e-5)
        },
        'scheduler': {'type': 'cosine', 'T_max': 30, 'eta_min': 1e-6},
        'criterion_kwargs': {},
    },
    # C5_multi_seed handled separately (3 passes)
    {
        'name': 'C5_multi_seed',
        'branch': 'c5-multi-seed',
        'overrides': {
            'stock_emb_dim': 32,
        },
        'scheduler': {'type': 'cosine', 'T_max': 30, 'eta_min': 1e-6},
        'criterion_kwargs': {},
        'multi_seed': True,
        'seeds': [42, 123, 456],
    },
    {
        'name': 'C6_wider_model',
        'branch': 'c6-wider-model',
        'overrides': {
            'stock_emb_dim': 32,
            'd_model': 384,      # wider model (C uses 256)
            'nhead': 6,          # more heads (C uses 4)
            'dim_feedforward': 768,  # proportional FFN (C uses 512)
        },
        'scheduler': {'type': 'cosine', 'T_max': 30, 'eta_min': 1e-6},
        'criterion_kwargs': {},
    },
    {
        'name': 'C7_loss_tune',
        'branch': 'c7-loss-tune',
        'overrides': {
            'stock_emb_dim': 32,
            'top5_weight': 5,      # stronger focus on top-5 (C uses 2.0)
            'pairwise_weight': 2,  # stronger pairwise signal (C uses 1)
        },
        'scheduler': {'type': 'cosine', 'T_max': 30, 'eta_min': 1e-6},
        'criterion_kwargs': {},
    },
]


def run_variant(variant):
    import torch
    name = variant['name']

    log(f"{'='*60}")
    log(f"STARTING: {name}")
    log(f"{'='*60}")

    config = reset_config()
    for k, v in variant['overrides'].items():
        config[k] = v

    # Handle multi-seed: use first seed for main training dir
    is_multi_seed = variant.get('multi_seed', False)
    if is_multi_seed:
        seeds = variant['seeds']
        log(f"Multi-seed mode: will train with seeds {seeds}")
        # Train each seed in its own subdirectory
        all_predictions = []
        for seed in seeds:
            seed_name = f"{name}_seed{seed}"
            seed_dir = f'./model/60_{config["feature_num"]}/{name}/{seed_name}'
            config['output_dir'] = seed_dir
            config['seed'] = seed
            os.makedirs(seed_dir, exist_ok=True)
            log(f"[{seed_name}] Training with seed={seed}...")

            from train import main as train_main, WeightedRankingLoss
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
                log(f"[{seed_name}] Done in {elapsed/60:.1f} min | Best: {best_score:.4f}")
            except Exception as e:
                elapsed = time.time() - start
                log(f"[{seed_name}] FAILED after {elapsed/60:.1f} min: {e}")
                log(traceback.format_exc())
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
                continue

        # Run prediction for each seed and collect
        from predict_helper import run_prediction
        for seed in seeds:
            seed_name = f"{name}_seed{seed}"
            seed_dir = f'./model/60_{config["feature_num"]}/{name}/{seed_name}'
            config['output_dir'] = seed_dir
            config['seed'] = seed
            try:
                preds = run_prediction(config, seed_name)
                all_predictions.append(set(p[0] for p in preds[:20]))
                log(f"[{seed_name}] Top-20: {[p[0] for p in preds[:5]]}")
            except Exception as e:
                log(f"[{seed_name}] Prediction failed: {e}")

        # Compute intersection of top-20 predictions
        if all_predictions:
            common = all_predictions[0]
            for p in all_predictions[1:]:
                common = common & p
            log(f"[{name}] Intersection of top-20 across {len(all_predictions)} seeds: {sorted(common)}")

            # Write combined result
            main_dir = f'./model/60_{config["feature_num"]}/{name}'
            os.makedirs(main_dir, exist_ok=True)
            with open(os.path.join(main_dir, 'prediction.txt'), 'w') as f:
                f.write(f"=== {name} Multi-Seed Intersection ===\n")
                f.write(f"Seeds: {seeds}\n\n")
                f.write(f"Stocks appearing in top-20 across ALL seeds ({len(common)} stocks):\n")
                for s in sorted(common):
                    f.write(f"  {s}\n")
            with open(os.path.join(main_dir, 'DONE'), 'w') as f:
                f.write(f"{name} completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"multi_seed: seeds={seeds}, intersection_size={len(common)}\n")
            return True
        return False

    else:
        # Single-seed variant (standard path)
        config['output_dir'] = f'./model/60_{config["feature_num"]}/{name}'
        os.makedirs(config['output_dir'], exist_ok=True)
        log(f"Config: { {k:v for k,v in variant['overrides'].items()} }")

        from train import main as train_main, WeightedRankingLoss
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
            try:
                torch.cuda.empty_cache()
                log("CUDA cache cleared")
            except:
                pass
            return False


def main():
    with open(MASTER_LOG, 'w', encoding='utf-8') as f:
        f.write('')

    log("MASTER TRAINING SCRIPT v3 STARTED (C-based variants)")
    log(f"Python: {sys.executable}")

    import torch
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        log("WARNING: No GPU!")

    # Check completed
    completed = set()
    for v in VARIANTS:
        done_path = os.path.join(SRC_DIR, f"model/60_158+39/{v['name']}/DONE")
        pred_path = os.path.join(SRC_DIR, f"model/60_158+39/{v['name']}/prediction.txt")
        if os.path.exists(done_path) and os.path.exists(pred_path):
            completed.add(v['name'])
            log(f"SKIP {v['name']}: already completed")

    results = {}
    for v in VARIANTS:
        name = v['name']
        results[name] = {'success': name in completed, 'elapsed_min': 0,
                         'skipped': name in completed, 'branch': v['branch']}

    total_start = time.time()

    for i, variant in enumerate(VARIANTS, 1):
        name = variant['name']
        if name in completed:
            continue

        log(f"\n>>> [{i}/{len(VARIANTS)}] {name} (branch: {variant['branch']})")
        v_start = time.time()
        success = run_variant(variant)
        v_elapsed = time.time() - v_start
        results[name] = {'success': success, 'elapsed_min': round(v_elapsed / 60, 1),
                         'skipped': False, 'branch': variant['branch']}

    total_elapsed = time.time() - total_start
    log(f"\n{'='*60}")
    log(f"ALL VARIANTS COMPLETE ({total_elapsed/60:.1f} min)")
    log(f"{'='*60}")
    for name, r in results.items():
        if r['skipped']:
            log(f"  {name} [{r['branch']}]: SKIPPED")
        else:
            status = 'OK' if r['success'] else 'FAILED'
            log(f"  {name} [{r['branch']}]: {status} ({r['elapsed_min']} min)")

    summary_path = os.path.join(CODE_DIR, 'TRAINING_SUMMARY_v3.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'total_min': round(total_elapsed/60, 1),
                   'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)

    with open(os.path.join(CODE_DIR, 'V3_ALL_DONE'), 'w') as f:
        f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == '__main__':
    main()
