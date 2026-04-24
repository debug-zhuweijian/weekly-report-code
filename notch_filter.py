"""
notch_filter.py -- 陷波滤波验证：去除 50Hz 工频干扰
对应周报章节：二.1 预处理管线
"""

import numpy as np


def notch_filter(x, freq, fs, r=0.99):
    w0 = 2 * np.pi * freq / fs
    b = np.array([1, -2 * np.cos(w0), 1])
    a = np.array([1, -2 * r * np.cos(w0), r ** 2])
    y = np.zeros_like(x)
    for n in range(2, len(x)):
        y[n] = b[0] * x[n] + b[1] * x[n - 1] + b[2] * x[n - 2]
        y[n] -= a[1] * y[n - 1] + a[2] * y[n - 2]
    return y


def main():
    fs = 500
    T = 2.0
    t = np.arange(int(fs * T)) / fs
    signal = np.sin(2 * np.pi * 10 * t)
    noise_50 = 0.5 * np.sin(2 * np.pi * 50 * t)
    x = signal + noise_50

    print(f"Original signal: 10 Hz sine + 50 Hz interference (amplitude 0.5)")
    print(f"Sample rate: {fs} Hz, Duration: {T}s\n")

    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), d=1.0 / fs)[:len(x) // 2]
    mag_before = np.abs(X)[:len(x) // 2]
    peak_50_before = mag_before[np.argmin(np.abs(freqs - 50))]
    peak_10_before = mag_before[np.argmin(np.abs(freqs - 10))]
    print(f"Before notch: |X(10Hz)| = {peak_10_before:.2f}, |X(50Hz)| = {peak_50_before:.2f}")

    y = notch_filter(x, freq=50, fs=fs, r=0.99)
    Y = np.fft.fft(y)
    mag_after = np.abs(Y)[:len(y) // 2]
    peak_50_after = mag_after[np.argmin(np.abs(freqs - 50))]
    peak_10_after = mag_after[np.argmin(np.abs(freqs - 10))]
    print(f"After notch:  |X(10Hz)| = {peak_10_after:.2f}, |X(50Hz)| = {peak_50_after:.2f}")

    reduction_db = 20 * np.log10(peak_50_after / peak_50_before) if peak_50_after > 0 else -np.inf
    print(f"\n50 Hz reduction: {reduction_db:.1f} dB")
    print(f"10 Hz preservation: {peak_10_after / peak_10_before * 100:.1f}%")


if __name__ == "__main__":
    main()
