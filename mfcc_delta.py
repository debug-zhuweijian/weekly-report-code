"""
mfcc_delta.py -- Delta / Delta-Delta 计算及 39 维 MFCC 拼接验证
对应周报章节：感知驱动的语音特征表示
"""

import numpy as np


def compute_delta(features):
    T, D = features.shape
    delta = np.zeros_like(features)
    delta[0] = features[1] - features[0]
    delta[-1] = features[-1] - features[-2]
    for t in range(1, T - 1):
        delta[t] = (features[t + 1] - features[t - 1]) / 2.0
    return delta


def main():
    np.random.seed(42)
    T, D = 10, 13
    static = np.random.randn(T, D) * 5 + np.array([20, 5, -3, 1, 0.5, 0.2, 0.1, 0, 0, 0, 0, 0, 0])

    print(f"Static MFCC shape: {static.shape} (T={T} frames, D={D} coefficients)")
    print(f"First 3 frames, first 5 coefficients:")
    for t in range(3):
        print(f"  frame {t}: {static[t, :5].round(3).tolist()}")

    delta = compute_delta(static)
    delta_delta = compute_delta(delta)
    full = np.concatenate([static, delta, delta_delta], axis=1)

    print(f"\nDelta shape: {delta.shape}")
    print(f"Delta-Delta shape: {delta_delta.shape}")
    print(f"Full 39-dim feature shape: {full.shape}")

    print(f"\nFull 39-dim feature, frame 0 (first 5 of each section):")
    print(f"  Static:      {full[0, :5].round(4).tolist()}")
    print(f"  Delta:       {full[0, D:D+5].round(4).tolist()}")
    print(f"  Delta-Delta: {full[0, 2*D:2*D+5].round(4).tolist()}")

    print(f"\nVerification: Delta[1,0] = (static[2,0] - static[0,0]) / 2")
    print(f"  Computed: {delta[1, 0]:.6f}")
    print(f"  Manual:   {(static[2, 0] - static[0, 0]) / 2:.6f}")

    print(f"\nVerification: Delta-Delta[2,0] = (delta[3,0] - delta[1,0]) / 2")
    print(f"  Computed: {delta_delta[2, 0]:.6f}")
    print(f"  Manual:   {(delta[3, 0] - delta[1, 0]) / 2:.6f}")

    print(f"\nFinal feature matrix: {full.shape[0]} frames x {full.shape[1]} dimensions = {T} x {D}*3 = {T} x 39")


if __name__ == "__main__":
    main()
