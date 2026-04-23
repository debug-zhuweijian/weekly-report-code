# Neural Speech Signal Processing - Weekly Report Code

Weekly report code examples for neural speech signal processing course.

## Code Index

| File | Description | Report Section |
|------|-------------|----------------|
| dft_basic.py | DFT spectrum verification | 1.1 Time-Frequency Analysis |
| stft_demo.py | STFT frame-based verification | 1.1 Time-Frequency Analysis |
| mel_filterbank.py | Mel filterbank construction | 1.2 Perceptual Frequency Compression |
| dct_ii_demo.py | DCT-II transform and reconstruction | 1.3 MFCC & Dynamic Features |
| mfcc_delta.py | Delta / Delta-Delta computation (39-dim MFCC) | 1.3 MFCC & Dynamic Features |
| hilbert_envelope.py | Hilbert transform envelope extraction | 2.2 Envelope & Normalization |
| notch_filter.py | Notch filter for 50Hz powerline noise | 2.1 Preprocessing Pipeline |
| car_demo.py | Common Average Reference verification | 2.1 Preprocessing Pipeline |
| bandpass_filter.py | Bandpass filter for HGA (70-150 Hz) | 2.1 Preprocessing Pipeline |
| zscore_demo.py | Z-score normalization verification | 2.2 Envelope & Normalization |
| feature_matrix_demo.py | Feature matrix construction (simulated pipeline) | 2.2 Envelope & Normalization |
| rnn_gradient.py | RNN hidden state + gradient vanishing | 3.1 RNN → LSTM → BiLSTM |
| lstm_cell.py | LSTM gating mechanism + long-term retention | 3.1 RNN → LSTM → BiLSTM |
| bilstm_context.py | BiLSTM bidirectional context aggregation | 3.1 RNN → LSTM → BiLSTM |
| ctc_alignment.py | CTC path enumeration + collapse + probability sum | 3.2 CTC Alignment |
| conv1d_demo.py | 1D convolution sliding window | 3.3 Convolution Basics |
| dilated_causal_conv.py | Dilated causal convolution + receptive field | 3.3 Convolution Basics |

## Requirements

- Python 3.8+
- numpy
- scipy (for bandpass_filter.py)

## Usage

```bash
python dft_basic.py
python stft_demo.py
# ... each file runs independently
```
