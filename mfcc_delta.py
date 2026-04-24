"""
mfcc_delta.py -- Delta / Delta-Delta 回归差分计算及 39 维 MFCC 拼接验证
对应周报章节：一.3 MFCC 与动态特征
"""

import numpy as np


def compute_delta(features, window=2):
    T, D = features.shape
    denom = 2 * sum(n * n for n in range(1, window + 1))
    padded = np.pad(features, ((window, window), (0, 0)), mode="edge")
    delta = np.zeros_like(features)
    for t in range(T):
        center = t + window
        for n in range(1, window + 1):
            delta[t] += n * (padded[center + n] - padded[center - n])
        delta[t] /= denom
    return delta


def main():
    np.random.seed(42)
    T, D = 10, 13
    window = 2
    static = np.random.randn(T, D) * 5 + np.array([20, 5, -3, 1, 0.5, 0.2, 0.1, 0, 0, 0, 0, 0, 0])

    print(f"Static MFCC shape: {static.shape} (T={T} frames, D={D} coefficients)")
    print(f"Delta window: N={window} (regression formula)")
    print(f"First 3 frames, first 5 coefficients:")
    for t in range(3):
        print(f"  frame {t}: {static[t, :5].round(3).tolist()}")

    delta = compute_delta(static, window=window)
    delta_delta = compute_delta(delta, window=window)
    full = np.concatenate([static, delta, delta_delta], axis=1)

    print(f"\nDelta shape: {delta.shape}")
    print(f"Delta-Delta shape: {delta_delta.shape}")
    print(f"Full 39-dim feature shape: {full.shape}")

    print(f"\nFull 39-dim feature, frame 0 (first 5 of each section):")
    print(f"  Static:      {full[0, :5].round(4).tolist()}")
    print(f"  Delta:       {full[0, D:D+5].round(4).tolist()}")
    print(f"  Delta-Delta: {full[0, 2*D:2*D+5].round(4).tolist()}")

    t_check = 3
    manual_delta = (
        1 * (static[t_check + 1, 0] - static[t_check - 1, 0]) +
        2 * (static[t_check + 2, 0] - static[t_check - 2, 0])
    ) / (2 * (1 ** 2 + 2 ** 2))
    print("\nVerification with regression delta formula:")
    print("  Delta[t] = sum_n n*(c[t+n]-c[t-n]) / (2*sum_n n^2)")
    print(f"  Check frame t={t_check}, coeff 0")
    print(f"  Computed: {delta[t_check, 0]:.6f}")
    print(f"  Manual:   {manual_delta:.6f}")

    manual_delta2 = (
        1 * (delta[t_check + 1, 0] - delta[t_check - 1, 0]) +
        2 * (delta[t_check + 2, 0] - delta[t_check - 2, 0])
    ) / (2 * (1 ** 2 + 2 ** 2))
    print(f"\nVerification: Delta-Delta at frame t={t_check}, coeff 0")
    print(f"  Computed: {delta_delta[t_check, 0]:.6f}")
    print(f"  Manual:   {manual_delta2:.6f}")

    print(f"\nFinal feature matrix: {full.shape[0]} frames x {full.shape[1]} dimensions = {T} x {D}*3 = {T} x 39")


if __name__ == "__main__":
    main()
