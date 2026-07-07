#!/usr/bin/env python3
"""
evaluate_v3.py — Evaluate THU-BDC2026 variant predictions against actual market returns.

Reads prediction.txt files from git branches, computes portfolio returns using
eval_20260622_20260626.csv data, and produces a ranked comparison table.
"""

import sys
import os
import re
import subprocess
import csv
from typing import Optional, List, Tuple, Dict

# Handle Chinese encoding on Windows
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace')

# ──────────────────────────── Configuration ────────────────────────────

REPO_DIR = r"C:\Users\62770\.qoderworkcn\workspace\mql6v86x74ovudgw\outputs\THU-BDC2026"
EVAL_CSV = os.path.join(REPO_DIR, "data", "eval_20260622_20260626.csv")

BUY_DATE = "2026-06-22"
SELL_DATE = "2026-06-26"
PRED_BASE = "code/src/model/60_158+39"

# Variant definitions: (name, branch, subdir, is_main_branch)
VARIANTS = [
    # Original variants (main branch)
    ("C_stock_emb",  "main", "C_stock_emb",  True),
    ("A_30epoch",    "main", "A_30epoch",    True),
    ("B_loss_fix",   "main", "B_loss_fix",   True),
    ("D_deep_cross", "main", "D_deep_cross", True),
    ("E_combined",   "main", "E_combined",   True),
    # New C2-C7 variants (feature branches)
    ("C2_emb64",        "c2-emb64",        "C2_emb64",        False),
    ("C3_high_dropout", "c3-high-dropout", "C3_high_dropout",  False),
    ("C4_high_lr",      "c4-high-lr",      "C4_high_lr",      False),
    ("C5_multi_seed",   "c5-multi-seed",    "C5_multi_seed",   False),
    ("C6_wider_model",  "c6-wider-model",  "C6_wider_model",  False),
    ("C7_loss_tune",    "c7-loss-tune",     "C7_loss_tune",    False),
    # Fallback for C5 (best single seed)
    ("C5_seed123",      "c5-multi-seed",    "C5_multi_seed/C5_multi_seed_seed123", False),
]


# ──────────────────────────── Helpers ────────────────────────────

def git_show(branch: str, path: str) -> Optional[str]:
    """Read a file from a git branch without checkout."""
    result = subprocess.run(
        ["git", "show", f"{branch}:{path}"],
        capture_output=True, text=True, cwd=REPO_DIR
    )
    if result.returncode != 0:
        return None
    return result.stdout


def parse_top5(text: str) -> List[str]:
    """Extract the top-5 stock codes from a prediction.txt file."""
    stocks = []
    for line in text.splitlines():
        m = re.match(r'\s*#\d+:\s+(\d+)\s+\(score:', line)
        if m:
            stocks.append(m.group(1))
            if len(stocks) == 5:
                break
    return stocks


def parse_val_score(text: str) -> Optional[float]:
    """Try to extract validation score from prediction text."""
    for line in text.splitlines():
        m = re.search(r'score:\s*([\d.]+)', line)
        if m:
            return float(m.group(1))
    return None


def load_eval_data(csv_path: str):
    """
    Load evaluation CSV and return a dict:
      { stock_code_str: { 'buy_open': float, 'sell_open': float } }
    where stock_code_str is the bare 6-digit code from 股票代码 column.
    """
    data = {}  # bare_code -> {buy_open, sell_open}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row["date"].strip()
            bare_code = row["股票代码"].strip()
            open_price = float(row["open"].strip())
            if date == BUY_DATE:
                data.setdefault(bare_code, {})["buy_open"] = open_price
            elif date == SELL_DATE:
                data.setdefault(bare_code, {})["sell_open"] = open_price
    return data


def read_val_score_from_file(branch: str, subdir: str) -> Optional[float]:
    """Read final_score.txt from the variant directory on its branch."""
    path = f"{PRED_BASE}/{subdir}/final_score.txt"
    text = git_show(branch, path)
    if text is None:
        return None
    m = re.search(r'Best final_score:\s*([\d.]+)', text)
    if m:
        return float(m.group(1))
    return None


# ──────────────────────────── Main ────────────────────────────

def main():
    os.chdir(REPO_DIR)

    # 1. Load eval data
    eval_data = load_eval_data(EVAL_CSV)
    total_stocks = len(eval_data)

    # Compute return for every stock
    all_returns = {}
    for code, prices in eval_data.items():
        if "buy_open" in prices and "sell_open" in prices and prices["buy_open"] > 0:
            all_returns[code] = (prices["sell_open"] - prices["buy_open"]) / prices["buy_open"]

    # Random baseline: average return of ALL stocks
    random_baseline = sum(all_returns.values()) / len(all_returns) if all_returns else 0.0

    # Max possible: best 5 stocks by return
    sorted_by_return = sorted(all_returns.items(), key=lambda x: x[1], reverse=True)
    best5 = sorted_by_return[:5]
    max_possible = sum(r for _, r in best5) / 5

    print("=" * 110)
    print("  THU-BDC2026 Variant Evaluation Report")
    print(f"  Buy: {BUY_DATE} open  |  Sell: {SELL_DATE} open")
    print(f"  Total stocks in eval set: {total_stocks}  |  Stocks with valid prices: {len(all_returns)}")
    print("=" * 110)
    print()

    # 2. Gather predictions for each variant
    results = []

    for name, branch, subdir, is_main in VARIANTS:
        pred_path = f"{PRED_BASE}/{subdir}/prediction.txt"
        text = git_show(branch, pred_path)

        if text is None:
            results.append({
                "name": name,
                "val_score": None,
                "top5": [],
                "stock_returns": [],
                "avg_return": None,
                "error": "prediction.txt not found",
            })
            continue

        top5 = parse_top5(text)
        val_score = read_val_score_from_file(branch, subdir)

        # Compute per-stock returns
        stock_returns = []
        missing = []
        for code in top5:
            if code in all_returns:
                stock_returns.append((code, all_returns[code]))
            else:
                stock_returns.append((code, None))
                missing.append(code)

        valid_returns = [r for _, r in stock_returns if r is not None]
        avg_return = sum(valid_returns) / len(valid_returns) if valid_returns else None

        results.append({
            "name": name,
            "val_score": val_score,
            "top5": top5,
            "stock_returns": stock_returns,
            "avg_return": avg_return,
            "missing": missing,
            "error": None,
        })

    # 3. Print detailed per-variant results
    print(f"{'─' * 110}")
    print(f"  {'Variant':<22} {'Val Score':>10}  {'Top-5 Stocks':<40} {'Individual Returns':<30} {'Avg Return':>11}")
    print(f"{'─' * 110}")

    # Sort by avg_return descending (None at end)
    results_sorted = sorted(results, key=lambda r: r["avg_return"] if r["avg_return"] is not None else float('-inf'), reverse=True)

    for r in results_sorted:
        name = r["name"]
        vs = f"{r['val_score']:.4f}" if r["val_score"] is not None else "N/A"

        if r["error"]:
            print(f"  {name:<22} {vs:>10}  {'ERROR: ' + r['error']}")
            continue

        stocks_str = ", ".join(r["top5"])
        returns_parts = []
        for code, ret in r["stock_returns"]:
            if ret is not None:
                returns_parts.append(f"{ret:+.2%}")
            else:
                returns_parts.append("N/A")
        returns_str = ", ".join(returns_parts)

        avg_str = f"{r['avg_return']:+.4%}" if r["avg_return"] is not None else "N/A"
        beat = ""
        if r["avg_return"] is not None:
            beat = " *" if r["avg_return"] > random_baseline else ""

        print(f"  {name:<22} {vs:>10}  {stocks_str:<40} {returns_str:<30} {avg_str:>11}{beat}")

    print(f"{'─' * 110}")
    print(f"  {'[Random Baseline]':<22} {'—':>10}  {'All stocks equal-weight':<40} {'':<30} {random_baseline:>+11.4%}")
    print(f"  {'[Max Possible]':<22} {'—':>10}  {', '.join(c for c, _ in best5):<40} {', '.join(f'{r:+.2%}' for _, r in best5):<30} {max_possible:>+11.4%}")
    print(f"{'─' * 110}")
    print()
    print("  * = beats random baseline")
    print()

    # 4. Summary statistics
    print("=" * 110)
    print("  Summary")
    print("=" * 110)
    print()

    valid_results = [r for r in results if r["avg_return"] is not None and r["name"] != "C5_multi_seed"]
    # Include C5_seed123 but exclude C5_multi_seed (empty intersection)

    best_variant = max(valid_results, key=lambda r: r["avg_return"])
    worst_variant = min(valid_results, key=lambda r: r["avg_return"])

    beating = sum(1 for r in valid_results if r["avg_return"] > random_baseline)
    total_v = len(valid_results)

    print(f"  Variants evaluated:       {total_v}")
    print(f"  Beating random baseline:  {beating} / {total_v}  ({beating/total_v:.0%})")
    print(f"  Random baseline return:   {random_baseline:+.4%}")
    print(f"  Max possible return:      {max_possible:+.4%}")
    print()
    print(f"  Best variant:    {best_variant['name']:<22} {best_variant['avg_return']:+.4%}")
    print(f"  Worst variant:   {worst_variant['name']:<22} {worst_variant['avg_return']:+.4%}")
    print()

    # 5. Ranking table (compact)
    print("=" * 110)
    print("  Ranking by Actual Portfolio Return (descending)")
    print("=" * 110)
    print()
    print(f"  {'#':>2}  {'Variant':<22} {'Val Score':>10}  {'Avg Return':>11}  {'vs Baseline':>12}  {'Beat?':>6}")
    print(f"  {'─'*2}  {'─'*22} {'─'*10}  {'─'*11}  {'─'*12}  {'─'*6}")

    for i, r in enumerate(results_sorted, 1):
        if r["avg_return"] is None:
            print(f"  {i:>2}  {r['name']:<22} {'N/A':>10}  {'N/A':>11}  {'N/A':>12}  {'—':>6}")
            continue
        diff = r["avg_return"] - random_baseline
        beat = "YES" if r["avg_return"] > random_baseline else "no"
        vs = f"{r['val_score']:.4f}" if r["val_score"] is not None else "N/A"
        print(f"  {i:>2}  {r['name']:<22} {vs:>10}  {r['avg_return']:>+11.4%}  {diff:>+12.4%}  {beat:>6}")

    print()
    print(f"  {'—':>2}  {'[Random Baseline]':<22} {'—':>10}  {random_baseline:>+11.4%}  {0.0:>+12.4%}  {'—':>6}")
    print(f"  {'—':>2}  {'[Max Possible]':<22} {'—':>10}  {max_possible:>+11.4%}  {max_possible - random_baseline:>+12.4%}  {'—':>6}")
    print()

    # 6. Correlation: val score vs actual return
    print("=" * 110)
    print("  Val Score vs Actual Return Correlation")
    print("=" * 110)
    print()

    pairs = [(r["val_score"], r["avg_return"], r["name"])
             for r in results if r["val_score"] is not None and r["avg_return"] is not None]

    if len(pairs) >= 3:
        # Simple Pearson correlation
        n = len(pairs)
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
        std_x = (sum((x - mean_x) ** 2 for x in xs) / n) ** 0.5
        std_y = (sum((y - mean_y) ** 2 for y in ys) / n) ** 0.5
        if std_x > 0 and std_y > 0:
            corr = cov_xy / (std_x * std_y)
        else:
            corr = 0.0
        print(f"  Pearson correlation (val_score, actual_return): {corr:.4f}")
        print(f"  (N={n} variants)")
        print()
        for vs, ar, nm in sorted(pairs, key=lambda p: p[0], reverse=True):
            print(f"    {nm:<22}  val={vs:.4f}  return={ar:+.4%}")
    else:
        print("  Not enough data points for correlation.")

    print()
    print("=" * 110)
    print("  Done.")
    print("=" * 110)


if __name__ == "__main__":
    main()
