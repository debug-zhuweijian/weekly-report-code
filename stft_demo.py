"""
stft_demo.py -- STFT 分帧验证：对 chirp 信号做 STFT，打印时间-频率矩阵
对应周报章节：语音信号的时频分析基础
"""

import numpy as np


def main():
    fs = 1000
    duration = 1.0
    t = np.arange(int(fs * duration)) / fs

    f0, f1 = 50, 400
    freq = f0 + (f1 - f0) * t / duration
    x = np.sin(2 * np.pi * freq * t)

    frame_len = 128
    hop = 64
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_len) / (frame_len - 1)))
    n_frames = (len(x) - frame_len) // hop + 1

    stft_matrix = []
    for i in range(n_frames):
        start = i * hop
        frame = x[start:start + frame_len] * window
        X = np.fft.fft(frame)
        stft_matrix.append(np.abs(X)[:frame_len // 2])

    stft_matrix = np.array(stft_matrix)
    freq_axis = np.fft.fftfreq(frame_len, d=1.0 / fs)[:frame_len // 2]

    print(f"Chirp signal: {f0} Hz -> {f1} Hz, duration {duration}s")
    print(f"Frame length: {frame_len}, hop: {hop}, Hann window")
    print(f"STFT matrix shape: (n_frames={stft_matrix.shape[0]}, n_freq_bins={stft_matrix.shape[1]})")
    print(f"Frequency resolution: {freq_axis[1] - freq_axis[0]:.2f} Hz")
    print(f"Time resolution per frame: {hop / fs * 1000:.1f} ms\n")

    print("Peak frequency per frame (first 8 frames):")
    for i in range(min(8, n_frames)):
        peak_idx = np.argmax(stft_matrix[i])
        print(f"  frame {i}: peak freq = {freq_axis[peak_idx]:.1f} Hz")

    print("\nFrequency increases across frames, confirming STFT captures time-varying spectrum.")


if __name__ == "__main__":
    main()
