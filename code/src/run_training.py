"""
Standalone training script that runs fully autonomously.
Saves results to model directory when complete.
"""
import os
import sys
import time
import multiprocessing as mp

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

mp.set_start_method('spawn', force=True)


def run():
    from train import main
    from config import config
    import json

    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    log_path = os.path.join(output_dir, 'train_log.txt')

    print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: batch_size={config['batch_size']}, epochs={config['num_epochs']}")
    print(f"Log file: {log_path}")

    start_time = time.time()

    try:
        best_score = main()
        elapsed = time.time() - start_time
        msg = f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        msg += f"Total time: {elapsed/60:.1f} minutes\n"
        msg += f"Best score: {best_score:.4f}\n"
        print(msg)

        with open(log_path, 'a') as f:
            f.write(msg)

        # Write a completion marker
        with open(os.path.join(output_dir, 'DONE'), 'w') as f:
            f.write(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"best_score: {best_score:.6f}\n")

    except Exception as e:
        elapsed = time.time() - start_time
        msg = f"\nTraining FAILED at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        msg += f"Time elapsed: {elapsed/60:.1f} minutes\n"
        msg += f"Error: {str(e)}\n"
        import traceback
        msg += traceback.format_exc()
        print(msg)

        with open(log_path, 'a') as f:
            f.write(msg)


if __name__ == '__main__':
    run()
