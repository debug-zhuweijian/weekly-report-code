# Neural Speech Signal Processing - Weekly Report Code

Weekly report code examples for neural speech signal processing course.

## Code Index

| File | Description | Report Section |
|------|-------------|----------------|
| dft_basic.py | DFT spectrum verification | Time-Frequency Analysis |
| stft_demo.py | STFT frame-based verification | Time-Frequency Analysis |
| mel_filterbank.py | Mel filterbank construction | Perceptual Feature Representation |
| dct_ii_demo.py | DCT-II transform and reconstruction | Perceptual Feature Representation |
| mfcc_delta.py | Delta / Delta-Delta computation (39-dim MFCC) | Perceptual Feature Representation |
| hilbert_envelope.py | Hilbert transform envelope extraction | Envelope Analysis |
| rnn_gradient.py | RNN hidden state + gradient vanishing | Sequence Modeling |
| lstm_cell.py | LSTM gating mechanism + long-term retention | Sequence Modeling |
| bilstm_context.py | BiLSTM bidirectional context aggregation | Sequence Modeling |
| ctc_alignment.py | CTC path enumeration + collapse + probability sum | Sequence Alignment |

## Requirements

- Python 3.8+
- numpy

## Usage

```bash
python dft_basic.py
python stft_demo.py
# ... each file runs independently
```
