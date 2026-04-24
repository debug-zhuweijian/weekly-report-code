"""
mel_filterbank.py -- Mel 滤波器构造验证：Hz->Mel 转换 + 三角滤波器权重
对应周报章节：一.2 感知驱动的频率压缩
"""

import numpy as np


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)


def make_mel_filterbank(n_filters, n_fft, fs, fmin=0, fmax=None):
    if fmax is None:
        fmax = fs / 2
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / fs).astype(int)

    filterbank = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(n_filters):
        f_left = bin_points[m]
        f_center = bin_points[m + 1]
        f_right = bin_points[m + 2]
        for k in range(f_left, f_center):
            if f_center != f_left:
                filterbank[m, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                filterbank[m, k] = (f_right - k) / (f_right - f_center)
    return filterbank, hz_points


def main():
    print("Hz -> Mel conversion:")
    test_hz = [300, 1000, 4000, 8000]
    for hz in test_hz:
        print(f"  {hz:>5} Hz -> {hz_to_mel(hz):.1f} mel")

    print("\nMel -> Hz inverse check:")
    for hz in test_hz:
        mel = hz_to_mel(hz)
        recovered = mel_to_hz(mel)
        print(f"  {hz} Hz -> {mel:.1f} mel -> {recovered:.1f} Hz")

    n_filters = 8
    n_fft = 512
    fs = 16000
    fb, hz_pts = make_mel_filterbank(n_filters, n_fft, fs)

    print(f"\nMel filterbank: {n_filters} filters, n_fft={n_fft}, fs={fs}")
    print("Filter center frequencies (Hz):")
    for i in range(n_filters):
        print(f"  Filter {i + 1}: center = {hz_pts[i + 1]:.1f} Hz, peak bin = {np.argmax(fb[i])}")

    print(f"\nFilterbank shape: {fb.shape}")
    print(f"Non-zero elements per filter: {[np.count_nonzero(fb[i]) for i in range(n_filters)]}")


if __name__ == "__main__":
    main()
