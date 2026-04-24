"""
dct_ii_demo.py -- DCT-II 变换与重构验证（正交归一化版）
对应周报章节：一.3 MFCC 与动态特征
"""

import numpy as np


def dct_ii(x):
    N = len(x)
    out = np.zeros(N)
    for k in range(N):
        alpha = np.sqrt(1.0 / N) if k == 0 else np.sqrt(2.0 / N)
        s = 0.0
        for n in range(N):
            s += x[n] * np.cos(np.pi * (n + 0.5) * k / N)
        out[k] = alpha * s
    return out


def idct_iii(c):
    N = len(c)
    out = np.zeros(N)
    for n in range(N):
        s = 0.0
        for k in range(N):
            alpha = np.sqrt(1.0 / N) if k == 0 else np.sqrt(2.0 / N)
            s += alpha * c[k] * np.cos(np.pi * (n + 0.5) * k / N)
        out[n] = s
    return out


def main():
    x = np.array([8.0, 7.0, 6.0, 5.0, 2.0, 1.0, 1.0, 1.0])
    N = len(x)
    c = dct_ii(x)

    print(f"Input signal: {x.tolist()}")
    print(f"DCT-II coefficients:")
    for i, val in enumerate(c):
        print(f"  c[{i}] = {val:>10.6f}")

    print(f"\nPerfect reconstruction check:")
    x_rec = idct_iii(c)
    print(f"  Original:    {[round(v, 6) for v in x]}")
    print(f"  Reconstructed: {[round(v, 6) for v in x_rec]}")
    print(f"  Max abs error: {np.max(np.abs(x - x_rec)):.2e}")

    print(f"\nProgressive reconstruction (keeping first K coefficients):")
    for keep in [1, 2, 3, 4, N]:
        c_trunc = c.copy()
        c_trunc[keep:] = 0
        x_approx = idct_iii(c_trunc)
        err = np.max(np.abs(x - x_approx))
        print(f"  keep={keep}: {[round(v, 4) for v in x_approx]}  max_err={err:.4f}")

    print(f"\nEnergy concentration:")
    total_energy = np.sum(c ** 2)
    for k in [1, 2, 3, 4]:
        pct = np.sum(c[:k] ** 2) / total_energy * 100
        print(f"  First {k} coeffs capture {pct:.1f}% of total energy")


if __name__ == "__main__":
    main()
