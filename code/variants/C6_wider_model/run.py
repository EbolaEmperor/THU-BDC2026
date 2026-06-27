"""
Variant C6: Wider Model (d_model=384, nhead=6, dim_ff=768)
Based on C_stock_emb, increases model capacity by ~50%.
"""
import os, sys, time, multiprocessing as mp

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'src')
os.chdir(src_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.dirname(script_dir))

mp.set_start_method('spawn', force=True)
VARIANT_NAME = 'C6_wider_model'

def run():
    from config import config
    from train import main as train_main

    config['num_epochs'] = 30
    config['stock_emb_dim'] = 32
    config['d_model'] = 384          # 256 -> 384
    config['nhead'] = 6              # 4 -> 6
    config['dim_feedforward'] = 768  # 512 -> 768
    config['output_dir'] = f'./model/60_{config["feature_num"]}/{VARIANT_NAME}'
    os.makedirs(config['output_dir'], exist_ok=True)

    print(f"[{VARIANT_NAME}] epochs={config['num_epochs']}, stock_emb_dim={config['stock_emb_dim']}, d_model={config['d_model']}, nhead={config['nhead']}")
    start = time.time()
    try:
        best_score = train_main(
            scheduler_cfg={'type': 'cosine', 'T_max': config['num_epochs'], 'eta_min': 1e-6},
        )
        elapsed = time.time() - start
        print(f"[{VARIANT_NAME}] Done in {elapsed/60:.1f} min | Best: {best_score:.4f}")

        with open(os.path.join(config['output_dir'], 'DONE'), 'w') as f:
            f.write(f"{VARIANT_NAME} completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\nbest_score: {best_score:.6f}\n")

        from predict_helper import run_prediction
        run_prediction(config, VARIANT_NAME)
    except Exception as e:
        import traceback; print(f"[{VARIANT_NAME}] FAILED: {e}"); traceback.print_exc()

if __name__ == '__main__':
    run()
