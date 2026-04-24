"""
conv1d_demo.py -- 1D 卷积输出长度、对称填充与因果左填充验证
对应周报章节：三.3 卷积基础
"""

import numpy as np


def conv1d(x, kernel, pad_left=0, pad_right=0, stride=1, dilation=1):
    k = len(kernel)
    effective_k = dilation * (k - 1) + 1
    x = np.concatenate([np.zeros(pad_left), x, np.zeros(pad_right)])
    out_len = (len(x) - effective_k) // stride + 1
    out = np.zeros(out_len)
    for i in range(out_len):
        start = i * stride
        window = x[start:start + effective_k:dilation]
        out[i] = np.dot(window, kernel)
    return out


def main():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    kernel = np.array([1, 0, -1], dtype=float)
    stride = 1
    dilation = 1

    print(f"Input: {x.astype(int).tolist()}")
    print(f"Kernel: {kernel.astype(int).tolist()} (edge detector)\n")

    y_no_pad = conv1d(x, kernel, pad_left=0, pad_right=0, stride=stride, dilation=dilation)
    print(f"No padding:    output length = {len(y_no_pad)}")
    print(f"  {y_no_pad.round(2).tolist()}")

    y_same = conv1d(x, kernel, pad_left=1, pad_right=1, stride=stride, dilation=dilation)
    print(f"\nSymmetric pad: output length = {len(y_same)}")
    print(f"  {y_same.round(2).tolist()}")

    y_causal = conv1d(x, kernel, pad_left=2, pad_right=0, stride=stride, dilation=dilation)
    print(f"\nCausal left pad=2: output length = {len(y_causal)}")
    print(f"  {y_causal.round(2).tolist()}")

    print("\nOutput length formula:")
    print("  T_out = floor((T + p_l + p_r - d*(k-1) - 1)/s) + 1")
    T = len(x)
    k = len(kernel)
    configs = [
        ("no padding", 0, 0),
        ("symmetric", 1, 1),
        ("causal", 2, 0),
    ]
    for name, p_l, p_r in configs:
        olen = (T + p_l + p_r - dilation * (k - 1) - 1) // stride + 1
        print(f"  {name:>9}: T={T}, k={k}, p_l={p_l}, p_r={p_r}, d={dilation}, s={stride} => {olen}")


if __name__ == "__main__":
    main()
