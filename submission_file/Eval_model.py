import pandas as pd
import numpy as np

if __name__ == '__main__':
    expected_claims = np.array(pd.read_csv('claims.csv', header=None))[:, 0]
    prices = np.array(pd.read_csv('prices.csv', header=None))[:, 0]

    x_train = pd.read_csv('training.csv')
    y = x_train.pop('claim_amount')

    print(f"A/E: {np.divide(sum(y), sum(expected_claims)):.4f}")
    print(f"RMSE: {np.mean((expected_claims - y) ** 2) ** 0.5:.2f}")
    print(f"Profit rate: {np.divide(np.sum(prices), np.sum(y)):.4f}")
    print('-'*40)
    load_ratios = prices / expected_claims
    print(f'Loading max: {np.max(load_ratios):.2f}')
    print(f'Loading min: {np.min(load_ratios):.2f}')
    print(f'Loading mean: {np.mean(load_ratios):.2f}')
    print(f'Loading median: {np.median(load_ratios):.2f}')
