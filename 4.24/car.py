"""
car_demo.py -- CAR（公共平均参考）去除共模噪声验证
对应周报章节：二.1 预处理管线
"""

import numpy as np


def car(x):
    reference = np.mean(x, axis=0, keepdims=True)
    return x - reference


def main():
    np.random.seed(42)
    T = 1000
    N = 4
    t = np.arange(T) / 200.0
    common_noise = 0.5 * np.sin(2 * np.pi * 50 * t)

    signals = np.zeros((N, T))
    for i in range(N):
        freq = 5 + i * 3
        signals[i] = 0.3 * np.sin(2 * np.pi * freq * t) + common_noise

    print(f"Simulated {N} channels, {T} samples each")
    print(f"Common 50 Hz noise amplitude: 0.5 (added to all channels)\n")

    ref_before = np.mean(signals, axis=0)
    print(f"Common reference before CAR (first 5 values): {ref_before[:5].round(4).tolist()}")

    cleaned = car(signals)
    ref_after = np.mean(cleaned, axis=0)
    print(f"Common reference after CAR  (first 5 values): {ref_after[:5].round(10).tolist()}")

    noise_power_before = np.mean(np.var(signals, axis=1))
    noise_power_after = np.mean(np.var(cleaned, axis=1))
    print(f"\nAverage variance per channel before CAR: {noise_power_before:.6f}")
    print(f"Average variance per channel after CAR:  {noise_power_after:.6f}")
    print(f"CAR removes the common reference (sum of all CAR outputs = 0): {np.abs(np.sum(cleaned[:, 0])):.2e}")


if __name__ == "__main__":
    main()
