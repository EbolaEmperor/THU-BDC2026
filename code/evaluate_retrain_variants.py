#!/usr/bin/env python3
"""
评估所有变体重训练后的预测结果
用已知的 6/29开盘 和 7/3开盘 价格计算比赛收益率
"""
import sys, os, json
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace')

# 已知的开盘价 (从 baostock 拉取)
# 格式: stock_code -> (buy_open_629, sell_open_703)
PRICES = {
    # Our C4 stocks
    '600039': (93.6722764900, 97.5703661800),
    '600938': (35.3445925500, 34.6106611200),
    '601077': (9.3117726100, 9.3561848800),
    '300251': (297.8982618500, 321.5104743100),
    '600050': (5.8686391400, 5.9123262800),
    # Brother's stocks
    '688072': (1830.1021440000, 1830.1021440000),
    '603986': (4406.2304350000, 3804.0644631000),
    '600183': (1856.3512695000, 1580.1345203800),
    '300394': (4299.0270810000, 3506.0026680000),
    '002384': (1933.5550950000, 1599.9220590000),
}

# HS300
HS300_PRICES = (4866.0432, 4809.4988)


def calc_return(stocks):
    """计算等权组合的比赛收益率 (0.2 * sum of individual returns)"""
    returns = []
    detail = []
    for code in stocks:
        if code in PRICES:
            buy_open, sell_open = PRICES[code]
            ret = (sell_open - buy_open) / buy_open * 100
            returns.append(ret)
            detail.append((code, ret))
        else:
            detail.append((code, None))
    
    if returns:
        avg_ret = sum(returns) / len(returns)
        # 等权 score = 0.2 * sum
        score = 0.2 * sum(returns)
    else:
        avg_ret = 0
        score = 0
    return avg_ret, score, detail


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(SCRIPT_DIR, 'retrain_all_summary.json')
    
    if not os.path.exists(summary_path):
        print(f"找不到汇总文件: {summary_path}")
        print("请先运行 run_all_variants_retrain.py")
        return
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"{'='*70}")
    print(f"各变体 Top-5 预测收益率对比 (6/29开盘买 → 7/3开盘卖)")
    print(f"{'='*70}")
    
    # HS300 baseline
    hs300_ret = (HS300_PRICES[1] - HS300_PRICES[0]) / HS300_PRICES[0] * 100
    
    all_scores = []
    
    for name, top5 in results.items():
        avg_ret, score, detail = calc_return(top5)
        all_scores.append((name, avg_ret, score, detail))
    
    # 按 score 排序
    all_scores.sort(key=lambda x: x[2], reverse=True)
    
    for rank, (name, avg_ret, score, detail) in enumerate(all_scores, 1):
        print(f"\n{'─'*60}")
        print(f"#{rank} {name}")
        print(f"  平均收益率: {avg_ret:+.2f}%")
        print(f"  比赛 Score: {score:+.2f}%")
        print(f"  股票明细:")
        for code, ret in detail:
            if ret is not None:
                print(f"    {code}: {ret:+.2f}%")
            else:
                print(f"    {code}: 价格未知")
    
    print(f"\n{'='*70}")
    print(f"基准对比:")
    print(f"  沪深300:     {hs300_ret:+.2f}%")
    print(f"  弟弟组合:    ", end="")
    bro = ['688072', '603986', '600183', '300394', '002384']
    _, bro_score, _ = calc_return(bro)
    print(f"{bro_score:+.2f}%")
    
    if all_scores:
        best = all_scores[0]
        c4 = [s for s in all_scores if s[0] == 'C4_high_lr']
        if c4:
            c4_score = c4[0][2]
            print(f"\n  最佳变体: {best[0]} (score: {best[2]:+.2f}%)")
            if best[0] != 'C4_high_lr':
                print(f"  C4_high_lr: {c4_score:+.2f}%")
                print(f"  差距: {best[2] - c4_score:+.2f}%")
            else:
                print(f"  C4_high_lr 就是最佳变体!")
    
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
