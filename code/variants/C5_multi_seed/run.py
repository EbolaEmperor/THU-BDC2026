"""
Variant C5: Multi-Seed Ensemble
Based on C_stock_emb, trains with 3 different seeds (42, 123, 456).
Takes intersection of top-20 predictions to reduce single-seed luck.
"""
import os, sys, time, multiprocessing as mp

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'src')
os.chdir(src_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.dirname(script_dir))

mp.set_start_method('spawn', force=True)
VARIANT_NAME = 'C5_multi_seed'
SEEDS = [42, 123, 456]

def run():
    from config import config
    from train import main as train_main
    from predict_helper import run_prediction
    import torch

    base_output_dir = f'./model/60_{config["feature_num"]}/{VARIANT_NAME}'
    os.makedirs(base_output_dir, exist_ok=True)

    all_top20 = []
    for seed in SEEDS:
        seed_name = f"{VARIANT_NAME}_seed{seed}"
        seed_dir = os.path.join(base_output_dir, seed_name)

        # Reset config
        config['num_epochs'] = 30
        config['stock_emb_dim'] = 32
        config['seed'] = seed
        config['output_dir'] = seed_dir
        os.makedirs(seed_dir, exist_ok=True)

        print(f"[{seed_name}] epochs={config['num_epochs']}, seed={seed}")
        start = time.time()
        try:
            best_score = train_main(
                scheduler_cfg={'type': 'cosine', 'T_max': 30, 'eta_min': 1e-6},
            )
            elapsed = time.time() - start
            print(f"[{seed_name}] Done in {elapsed/60:.1f} min | Best: {best_score:.4f}")
        except Exception as e:
            import traceback; print(f"[{seed_name}] FAILED: {e}"); traceback.print_exc()
            try: torch.cuda.empty_cache()
            except: pass
            continue

        # Prediction
        try:
            preds = run_prediction(config, seed_name)
            top20_ids = set(p[0] for p in preds[:20])
            all_top20.append(top20_ids)
            print(f"[{seed_name}] Top-5: {[p[0] for p in preds[:5]]}")
        except Exception as e:
            print(f"[{seed_name}] Prediction failed: {e}")

    # Intersection
    if all_top20:
        common = all_top20[0]
        for s in all_top20[1:]:
            common = common & s
        print(f"\n[{VARIANT_NAME}] Intersection of top-20 ({len(all_top20)} seeds): {sorted(common)} ({len(common)} stocks)")

        with open(os.path.join(base_output_dir, 'prediction.txt'), 'w') as f:
            f.write(f"=== {VARIANT_NAME} Multi-Seed Intersection ===\n")
            f.write(f"Seeds: {SEEDS}\n\n")
            f.write(f"Stocks in top-20 across ALL seeds ({len(common)} stocks):\n")
            for s in sorted(common):
                f.write(f"  {s}\n")
        with open(os.path.join(base_output_dir, 'DONE'), 'w') as f:
            f.write(f"{VARIANT_NAME} completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"seeds={SEEDS}, intersection_size={len(common)}\n")
    else:
        print(f"[{VARIANT_NAME}] No successful predictions!")

if __name__ == '__main__':
    run()
