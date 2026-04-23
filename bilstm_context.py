"""
bilstm_context.py -- BiLSTM 前向+后向上下文聚合验证
对应周报章节：序列建模机制的演进
"""

import numpy as np


def main():
    np.random.seed(42)
    seq_len = 5
    input_dim = 3
    hidden_dim = 4

    x = np.random.randn(seq_len, input_dim)
    print(f"Input sequence shape: ({seq_len}, {input_dim})")
    print(f"Input:\n{x.round(3)}\n")

    W_forward = np.random.randn(hidden_dim, input_dim) * 0.5
    W_recur_f = np.eye(hidden_dim) * 0.3
    b_f = np.zeros(hidden_dim)

    forward_h = []
    h = np.zeros(hidden_dim)
    for t in range(seq_len):
        h = np.tanh(W_forward @ x[t] + W_recur_f @ h + b_f)
        forward_h.append(h.copy())
    forward_h = np.array(forward_h)

    W_backward = np.random.randn(hidden_dim, input_dim) * 0.5
    W_recur_b = np.eye(hidden_dim) * 0.3
    b_b = np.zeros(hidden_dim)

    backward_h = []
    h = np.zeros(hidden_dim)
    for t in range(seq_len - 1, -1, -1):
        h = np.tanh(W_backward @ x[t] + W_recur_b @ h + b_b)
        backward_h.insert(0, h.copy())
    backward_h = np.array(backward_h)

    bilstm_output = np.concatenate([forward_h, backward_h], axis=1)

    print(f"Forward hidden states shape:  {forward_h.shape}")
    print(f"Backward hidden states shape: {backward_h.shape}")
    print(f"BiLSTM output shape: {bilstm_output.shape} (seq_len x 2*hidden_dim)\n")

    for t in range(seq_len):
        print(f"  t={t}: forward_h[0]={forward_h[t, 0]:>7.4f}, backward_h[0]={backward_h[t, 0]:>7.4f}, "
              f"concat[0:2]=[{bilstm_output[t, 0]:.4f}, {bilstm_output[t, hidden_dim]:.4f}]")

    print(f"\nBidirectional context capture:")
    print(f"  Forward h[0] at t=0 only sees x[0], no future context")
    print(f"  Backward h[0] at t=0 has processed x[{seq_len - 1}]->x[0], full future context")
    print(f"  => BiLSTM output at each position combines past AND future information")


if __name__ == "__main__":
    main()
