"""
zscore_demo.py -- Z-score 标准化验证：统一多通道信号标尺
对应周报章节：二.2 包络提取与标准化
"""

import numpy as np


def main():
    np.random.seed(42)
    C = 3
    T = 500
    baseline_len = 100
    eps = 1e-8

    channels = np.zeros((C, T))
    channels[0] = np.random.randn(T) * 0.5 + 10
    channels[1] = np.random.randn(T) * 2.0 - 5
    channels[2] = np.random.randn(T) * 0.1 + 100

    print(f"Simulated {C} channels, {T} samples each")
    print(f"Baseline period: first {baseline_len} samples\n")

    print("Channel statistics before Z-score (baseline period):")
    for i in range(C):
        bl = channels[i, :baseline_len]
        print(f"  Channel {i}: mean={bl.mean():.3f}, std={bl.std():.3f}")

    mu = channels[:, :baseline_len].mean(axis=1, keepdims=True)
    sigma = channels[:, :baseline_len].std(axis=1, keepdims=True)
    z = (channels - mu) / (sigma + eps)

    print(f"\nChannel statistics after Z-score (baseline period, epsilon={eps:.0e}):")
    for i in range(C):
        bl = z[i, :baseline_len]
        print(f"  Channel {i}: mean={bl.mean():.6f}, std={bl.std():.6f}")

    print(f"\nAll channels now have mean~0 and std~1 during baseline.")
    print(f"This ensures no single channel dominates model training due to scale differences.")


if __name__ == "__main__":
    main()
