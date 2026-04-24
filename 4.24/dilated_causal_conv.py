"""
dilated_causal_conv.py -- 膨胀因果卷积与感受野验证
对应周报章节：三.3 卷积基础
"""

import numpy as np


def dilated_causal_conv1d(x, kernel, dilation=1):
    k = len(kernel)
    pad = (k - 1) * dilation
    x_padded = np.concatenate([np.zeros(pad), x])
    out = np.zeros_like(x)
    for i in range(len(x)):
        total = 0.0
        for j in range(k):
            idx = i + pad - j * dilation
            if 0 <= idx < len(x_padded):
                total += x_padded[idx] * kernel[k - 1 - j]
        out[i] = total
    return out


def receptive_field(kernel_size, dilations):
    return 1 + (kernel_size - 1) * sum(dilations)


def main():
    T = 32
    x = np.zeros(T)
    x[0] = 1.0

    kernel = np.array([0.5, 1.0, 0.5])
    dilations = [1, 2, 4, 8]

    print("Dilated causal convolution: stacking layers with doubling dilation")
    print(f"Kernel size: {len(kernel)}, Input length: {T}\n")

    current = x.copy()
    for layer, dilation in enumerate(dilations):
        current = dilated_causal_conv1d(current, kernel, dilation=dilation)
        nonzero = np.where(np.abs(current) > 1e-10)[0]
        rf = nonzero[-1] - nonzero[0] + 1 if len(nonzero) > 0 else 0
        print(f"Layer {layer + 1} (dilation={dilation:>2}): non-zero range = [{nonzero[0]}..{nonzero[-1]}], effective RF = {rf}")

    print("\nGeneral receptive field formula:")
    print("  RF = 1 + (k-1) * sum(d_l)")
    for L in [3, 5, 10]:
        rf = receptive_field(len(kernel), [2 ** i for i in range(L)])
        print(f"  L={L:>2}, k={len(kernel)} => RF = {rf}")

    rf_10 = receptive_field(3, [2 ** i for i in range(10)])
    print(f"\nSpecial case: k=3 and d_l = 2^l")
    print(f"  RF = 1 + 2*(2^L - 1), so L=10 gives RF = {rf_10}")
    print("=> Efficient long-range dependency without parameter explosion.")


if __name__ == "__main__":
    main()
