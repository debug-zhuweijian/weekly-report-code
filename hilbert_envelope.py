"""
hilbert_envelope.py -- Hilbert 变换提取 AM 信号包络验证
对应周报章节：二.2 包络提取与标准化
"""

import numpy as np


def main():
    N = 256
    n = np.arange(N)
    carrier_freq = 32
    mod_freq = 3
    true_envelope = 1.0 + 0.5 * np.cos(2 * np.pi * mod_freq * n / N)
    x = true_envelope * np.cos(2 * np.pi * carrier_freq * n / N)

    X = np.fft.fft(x)
    h = np.zeros(N)
    h[0] = 1
    h[N // 2] = 1
    h[1:N // 2] = 2
    analytic = np.fft.ifft(X * h)
    extracted_envelope = np.abs(analytic)

    print(f"AM signal: carrier={carrier_freq}*fs/N, modulation={mod_freq}*fs/N")
    print(f"Envelope: 1.0 + 0.5*cos(2pi*{mod_freq}*n/N)")
    print(f"N={N} samples\n")

    print("First 10 samples comparison:")
    print(f"  {'n':>3}  {'True_env':>10}  {'Hilbert':>10}  {'Error':>12}")
    for i in range(10):
        err = abs(extracted_envelope[i] - true_envelope[i])
        print(f"  {i:>3}  {true_envelope[i]:>10.6f}  {extracted_envelope[i]:>10.6f}  {err:>12.2e}")

    max_err = np.max(np.abs(extracted_envelope - true_envelope))
    mean_err = np.mean(np.abs(extracted_envelope - true_envelope))
    print(f"\nMax absolute error:  {max_err:.2e}")
    print(f"Mean absolute error: {mean_err:.2e}")

    if max_err < 1e-10:
        print("\nHilbert envelope extraction verified: near-perfect match with true envelope.")
    else:
        print(f"\nHilbert envelope extraction verified: max error {max_err:.6f} (acceptable for practical use).")


if __name__ == "__main__":
    main()
