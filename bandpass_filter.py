"""
bandpass_filter.py -- 带通滤波验证：提取 70-150 Hz 高伽马频段
对应周报章节：神经信号预处理与特征工程
"""

import numpy as np


def butter_bandpass(x, lowcut, highcut, fs, order=4):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, x)


def main():
    fs = 1000
    T = 1.0
    t = np.arange(int(fs * T)) / fs

    signal_hga = np.sin(2 * np.pi * 100 * t)
    signal_low = 0.5 * np.sin(2 * np.pi * 10 * t)
    signal_high = 0.5 * np.sin(2 * np.pi * 300 * t)
    x = signal_hga + signal_low + signal_high

    print(f"Mixed signal: 100 Hz (HGA) + 10 Hz (low drift) + 300 Hz (high noise)")
    print(f"Bandpass: 70-150 Hz, order=4, fs={fs} Hz\n")

    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), d=1.0 / fs)[:len(x) // 2]
    mag_before = np.abs(X)[:len(x) // 2]

    for f in [10, 100, 300]:
        idx = np.argmin(np.abs(freqs - f))
        print(f"Before filter: |X({f}Hz)| = {mag_before[idx]:.2f}")

    y = butter_bandpass(x, 70, 150, fs, order=4)
    Y = np.fft.fft(y)
    mag_after = np.abs(Y)[:len(y) // 2]

    print()
    for f in [10, 100, 300]:
        idx = np.argmin(np.abs(freqs - f))
        print(f"After filter:  |X({f}Hz)| = {mag_after[idx]:.2f}")

    ratio_10 = mag_after[np.argmin(np.abs(freqs - 10))] / mag_before[np.argmin(np.abs(freqs - 10))]
    ratio_100 = mag_after[np.argmin(np.abs(freqs - 100))] / mag_before[np.argmin(np.abs(freqs - 100))]
    ratio_300 = mag_after[np.argmin(np.abs(freqs - 300))] / mag_before[np.argmin(np.abs(freqs - 300))]

    print(f"\nRetention at 100 Hz (target): {ratio_100 * 100:.1f}%")
    print(f"Suppression at 10 Hz: {20 * np.log10(ratio_10) if ratio_10 > 0 else '-inf':.1f} dB")
    print(f"Suppression at 300 Hz: {20 * np.log10(ratio_300) if ratio_300 > 0 else '-inf':.1f} dB")


if __name__ == "__main__":
    main()
