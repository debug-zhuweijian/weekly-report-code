"""
conv1d_demo.py -- 1D 卷积滑动窗口验证
对应周报章节：序列建模与卷积基础
"""

import numpy as np


def conv1d(x, kernel, padding=0, stride=1):
    k = len(kernel)
    if padding > 0:
        x = np.concatenate([np.zeros(padding), x, np.zeros(padding)])
    out_len = (len(x) - k) // stride + 1
    out = np.zeros(out_len)
    for i in range(out_len):
        start = i * stride
        out[i] = np.dot(x[start:start + k], kernel)
    return out


def main():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    kernel = np.array([1, 0, -1], dtype=float)

    print(f"Input: {x.astype(int).tolist()}")
    print(f"Kernel: {kernel.astype(int).tolist()} (edge detector)\n")

    y_no_pad = conv1d(x, kernel, padding=0)
    print(f"No padding:    output length = {len(y_no_pad)}")
    print(f"  {y_no_pad.round(2).tolist()}")

    y_padded = conv1d(x, kernel, padding=1)
    print(f"\nPadding=1:     output length = {len(y_padded)}")
    print(f"  {y_padded.round(2).tolist()}")

    y_causal = conv1d(x, kernel, padding=2, stride=1)
    print(f"\nCausal (left pad=2): output length = {len(y_causal)}")
    print(f"  {y_causal.round(2).tolist()}")

    print(f"\nOutput length formula: (T + 2*p - k) / s + 1")
    T = len(x)
    k = len(kernel)
    for p in [0, 1]:
        olen = (T + 2 * p - k) + 1
        print(f"  T={T}, k={k}, p={p}, s=1 => {olen}")


if __name__ == "__main__":
    main()
