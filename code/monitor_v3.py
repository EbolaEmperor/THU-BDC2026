"""
Monitor v3 training progress. Checks log every 5 minutes.
Writes summary to v3_monitor_summary.txt when done.
"""
import time, os, sys, json

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(CODE_DIR, 'v3_train_stdout.log')
MASTER_LOG = os.path.join(CODE_DIR, 'master_v3_log.txt')
SUMMARY_FILE = os.path.join(CODE_DIR, 'v3_monitor_summary.txt')
ALL_DONE_FLAG = os.path.join(CODE_DIR, 'V3_ALL_DONE')

VARIANTS = ['C2_emb64', 'C3_high_dropout', 'C4_high_lr', 'C5_multi_seed', 'C6_wider_model', 'C7_loss_tune']

def check_status():
    with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    status = {}
    for v in VARIANTS:
        status[v] = {'started': False, 'done': False, 'best_score': None, 'prediction': None}

    for line in lines:
        for v in VARIANTS:
            if f'STARTING: {v}' in line:
                status[v]['started'] = True
            if f'TRAINING DONE: {v}' in line:
                status[v]['done'] = True
                # Extract best score
                if 'Best:' in line:
                    try:
                        status[v]['best_score'] = float(line.split('Best:')[1].strip())
                    except:
                        pass
            if f'Prediction top-5:' in line and v in ''.join(lines[max(0, lines.index(line)-20):lines.index(line)+1]):
                try:
                    preds = line.split('top-5:')[1].strip()
                    status[v]['prediction'] = preds
                except:
                    pass
            if f'VARIANT {v}: SUCCESS' in line:
                status[v]['done'] = True
            if f'VARIANT {v}: FAILED' in line:
                status[v]['done'] = True
                status[v]['failed'] = True

    return status

def write_summary(status):
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write(f"V3 Training Monitor Summary\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        all_done = True
        for v in VARIANTS:
            s = status[v]
            if not s['done']:
                all_done = False
            state = 'DONE' if s['done'] else ('RUNNING' if s['started'] else 'WAITING')
            score = f"{s['best_score']:.4f}" if s['best_score'] else 'N/A'
            f.write(f"{v}: {state} | best_score={score}\n")
            if s.get('prediction'):
                f.write(f"  prediction: {s['prediction']}\n")
        f.write(f"\nAll done: {all_done}\n")
    return all_done

last_lines = 0
while True:
    if os.path.exists(ALL_DONE_FLAG):
        print("All done flag detected. Exiting.")
        break

    status = check_status()
    done = write_summary(status)

    # Print current status
    running = [v for v in VARIANTS if status[v]['started'] and not status[v]['done']]
    completed = [v for v in VARIANTS if status[v]['done']]
    waiting = [v for v in VARIANTS if not status[v]['started']]

    print(f"[{time.strftime('%H:%M:%S')}] Completed: {len(completed)}/6 | Running: {running or 'none'} | Waiting: {len(waiting)}")

    if done:
        print("All 6 variants completed!")
        with open(ALL_DONE_FLAG, 'w') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
        break

    time.sleep(300)  # Check every 5 minutes
