"""
dft_basic.py -- DFT 频谱验证：对混合正弦信号做 DFT，打印频谱幅度
对应周报章节：语音信号的时频分析基础
"""

import numpy as np


def main():
    fs = 256
    N = 256
    n = np.arange(N)
    x = 1.0 * np.sin(2 * np.pi * 32 * n / fs) + 0.5 * np.sin(2 * np.pi * 64 * n / fs)

    X = np.fft.fft(x)
    mag = np.abs(X)[:N // 2]
    freqs = np.fft.fftfreq(N, d=1.0 / fs)[:N // 2]

    top_k = 5
    idx = np.argsort(mag)[::-1][:top_k]
    print("Input signal: 1.0*sin(2pi*32Hz) + 0.5*sin(2pi*64Hz)")
    print(f"Sample rate: {fs} Hz, N: {N} points")
    print(f"\nTop {top_k} frequency components:")
    for i in idx:
        print(f"  freq = {freqs[i]:>6.1f} Hz, magnitude = {mag[i]:.6f}")

    peak1 = freqs[np.argmax(mag)]
    print(f"\nStrongest component: {peak1:.1f} Hz (expected 32.0 Hz)")

    mag_64 = mag[np.argmin(np.abs(freqs - 64.0))]
    mag_32 = mag[np.argmax(mag)]
    print(f"Ratio 32Hz/64Hz magnitude: {mag_32 / mag_64:.2f} (expected ~2.00)")

    P = np.abs(X) ** 2
    print(f"\nPower spectrum at 32Hz: {P[np.argmax(mag)]:.2f}")
    print(f"Power spectrum at 64Hz: {P[np.argmin(np.abs(freqs - 64.0))]:.2f}")


if __name__ == "__main__":
    main()
