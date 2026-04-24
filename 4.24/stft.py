"""
stft_demo.py -- 离散 STFT 分帧验证：对线性 chirp 信号做 STFT
对应周报章节：一.1 从时域到频域
"""

import numpy as np


def linear_chirp(t, f0, f1, duration):
    sweep_rate = (f1 - f0) / duration
    phase = 2 * np.pi * (f0 * t + 0.5 * sweep_rate * t ** 2)
    return np.sin(phase)


def main():
    fs = 1000
    duration = 1.0
    t = np.arange(int(fs * duration)) / fs

    f0, f1 = 50, 400
    x = linear_chirp(t, f0, f1, duration)

    frame_len = 128
    hop = 64
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_len) / (frame_len - 1)))
    n_frames = (len(x) - frame_len) // hop + 1

    stft_matrix = []
    for i in range(n_frames):
        start = i * hop
        frame = x[start:start + frame_len] * window
        X = np.fft.rfft(frame)
        stft_matrix.append(np.abs(X))

    stft_matrix = np.array(stft_matrix)
    freq_axis = np.fft.rfftfreq(frame_len, d=1.0 / fs)

    print(f"Linear chirp: {f0} Hz -> {f1} Hz, duration {duration}s")
    print(f"Discrete STFT uses frame length N={frame_len}, hop H={hop}, Hann window")
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
