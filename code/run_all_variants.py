"""
Master launcher: runs all 5 model variants sequentially as detached processes.
Each variant trains independently and saves results to its own directory.

Usage:
    python run_all_variants.py
"""
import os, sys, time, subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
variants_dir = os.path.join(script_dir, 'variants')

VARIANTS = [
    'A_30epoch',
    'B_loss_fix',
    'C_stock_emb',
    'D_deep_cross',
    'E_combined',
]

def run_variant(name):
    """Launch a variant training as a detached process and wait for completion."""
    run_script = os.path.join(variants_dir, name, 'run.py')
    if not os.path.exists(run_script):
        print(f"[MASTER] ERROR: {run_script} not found!")
        return False

    log_file = open(os.path.join(variants_dir, name, 'train_log.txt'), 'w', encoding='utf-8')
    print(f"[MASTER] Starting {name} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    proc = subprocess.Popen(
        [sys.executable, '-u', run_script],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=os.path.join(script_dir, 'src'),
        creationflags=0x00000200  # CREATE_NEW_PROCESS_GROUP
    )
    print(f"[MASTER] {name} PID: {proc.pid}")

    # Wait for completion
    proc.wait()
    log_file.close()

    print(f"[MASTER] {name} finished with exit code {proc.returncode} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return proc.returncode == 0


def main():
    print(f"{'='*60}")
    print(f"MASTER LAUNCHER: {len(VARIANTS)} variants to train")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    results = {}
    start = time.time()

    for i, name in enumerate(VARIANTS):
        print(f"\n--- [{i+1}/{len(VARIANTS)}] {name} ---")
        ok = run_variant(name)
        results[name] = 'OK' if ok else 'FAILED'
        elapsed_total = time.time() - start
        print(f"  Status: {results[name]} | Total elapsed: {elapsed_total/60:.1f} min")

    # Summary
    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"ALL DONE at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {name}: {status}")

    # Write summary
    summary_path = os.path.join(script_dir, 'all_variants_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_time/60:.1f} min\n\n")
        for name, status in results.items():
            f.write(f"{name}: {status}\n")


if __name__ == '__main__':
    main()
