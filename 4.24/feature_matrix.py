"""
feature_matrix_demo.py -- 特征矩阵构建验证：模拟完整的预处理链路输出
对应周报章节：二.2 包络提取与标准化
"""

import numpy as np


def main():
    np.random.seed(42)
    fs = 1000
    duration = 2.0
    C = 8
    eps = 1e-8
    t = np.arange(int(fs * duration)) / fs

    signals = np.zeros((C, len(t)))
    for i in range(C):
        freq = 70 + i * 10
        signals[i] = (1 + 0.3 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * freq * t)
        signals[i] += 0.3 * np.random.randn(len(t))

    step_ms = 10
    hop = int(fs * step_ms / 1000)
    n_frames = (len(t) - hop) // hop + 1

    envs = np.zeros((C, len(t)))
    for i in range(C):
        X = np.fft.fft(signals[i])
        h = np.zeros(len(t))
        h[0] = 1
        h[len(t) // 2] = 1
        h[1:len(t) // 2] = 2
        analytic = np.fft.ifft(X * h)
        envs[i] = np.abs(analytic)

    feature_matrix = np.zeros((n_frames, C))
    for f in range(n_frames):
        start = f * hop
        end = min(start + hop, len(t))
        feature_matrix[f] = envs[:, start:end].mean(axis=1)

    baseline = feature_matrix[:10]
    mu = baseline.mean(axis=0, keepdims=True)
    sigma = baseline.std(axis=0, keepdims=True)
    feature_matrix = (feature_matrix - mu) / (sigma + eps)

    print(f"Simulated {C} channels, {fs} Hz, {duration}s")
    print(f"Downsample step: {step_ms} ms (hop={hop} samples)")
    print(f"Feature matrix shape: {feature_matrix.shape} (T={n_frames} frames x C={C} channels)")
    print(f"\nFirst 5 frames (first 4 channels):")
    for f in range(5):
        print(f"  t={f * step_ms:>3}ms: {feature_matrix[f, :4].round(3).tolist()}")

    baseline_z = feature_matrix[:10]
    print(f"\nMatrix statistics (epsilon={eps:.0e} in Z-score):")
    print(f"  Baseline mean: {baseline_z.mean():.6f} (expect ~0)")
    print(f"  Baseline std:  {baseline_z.std():.6f} (expect ~1)")
    print(f"  Global mean:   {feature_matrix.mean():.6f}")
    print(f"  Global std:    {feature_matrix.std():.6f}")
    print("  Note: only the baseline window is forced to zero-mean/unit-std, not the full sequence.")
    print(f"\nThis T x C matrix is the direct input to sequence models (LSTM, CNN, etc.).")


if __name__ == "__main__":
    main()
