"""
dft_basic.py -- DFT 频谱验证：输出单边幅度谱与周期图
对应周报章节：一.1 从时域到频域
"""

import numpy as np


def one_sided_spectrum(x, fs):
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    amplitude = 2.0 * np.abs(X) / n
    amplitude[0] /= 2.0
    if n % 2 == 0:
        amplitude[-1] /= 2.0

    periodogram = (np.abs(X) ** 2) / n
    return freqs, amplitude, periodogram


def main():
    fs = 256
    N = 256
    n = np.arange(N)
    x = 1.0 * np.sin(2 * np.pi * 32 * n / fs) + 0.5 * np.sin(2 * np.pi * 64 * n / fs)

    freqs, amplitude, periodogram = one_sided_spectrum(x, fs)

    top_k = 5
    idx = np.argsort(amplitude)[::-1][:top_k]
    print("Input signal: 1.0*sin(2pi*32Hz) + 0.5*sin(2pi*64Hz)")
    print(f"Sample rate: {fs} Hz, N: {N} points")
    print("\nOne-sided amplitude spectrum (amplitude should match sinusoid coefficients):")
    for i in idx:
        print(f"  freq = {freqs[i]:>6.1f} Hz, amplitude = {amplitude[i]:.6f}")

    peak1 = freqs[np.argmax(amplitude)]
    print(f"\nStrongest component: {peak1:.1f} Hz (expected 32.0 Hz)")

    amp_32 = amplitude[np.argmin(np.abs(freqs - 32.0))]
    amp_64 = amplitude[np.argmin(np.abs(freqs - 64.0))]
    print(f"Recovered amplitude at 32Hz: {amp_32:.2f} (expected 1.00)")
    print(f"Recovered amplitude at 64Hz: {amp_64:.2f} (expected 0.50)")
    print(f"Ratio 32Hz/64Hz amplitude: {amp_32 / amp_64:.2f} (expected 2.00)")

    print("\nPeriodogram P[k] = |X[k]|^2 / N:")
    print(f"  P(32Hz) = {periodogram[np.argmin(np.abs(freqs - 32.0))]:.2f}")
    print(f"  P(64Hz) = {periodogram[np.argmin(np.abs(freqs - 64.0))]:.2f}")


if __name__ == "__main__":
    main()
