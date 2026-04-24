"""
lstm_cell.py -- LSTM 门控机制 + 长期状态保留验证
对应周报章节：三.1 RNN→LSTM→BiLSTM
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_diag_gate(hidden_dim, h_scale, x_scale):
    return np.concatenate([h_scale * np.eye(hidden_dim), x_scale * np.eye(hidden_dim)], axis=1)


def lstm_step(x_t, h_prev, c_prev, params):
    z = np.concatenate([h_prev, x_t])
    f_t = sigmoid(params["W_f"] @ z + params["b_f"])
    i_t = sigmoid(params["W_i"] @ z + params["b_i"])
    c_tilde = np.tanh(params["W_c"] @ z + params["b_c"])
    c_t = f_t * c_prev + i_t * c_tilde
    o_t = sigmoid(params["W_o"] @ z + params["b_o"])
    h_t = o_t * np.tanh(c_t)
    return h_t, c_t, f_t, i_t, c_tilde, o_t


def main():
    d = 4
    params = {
        "W_f": make_diag_gate(d, h_scale=0.30, x_scale=0.10),
        "W_i": make_diag_gate(d, h_scale=0.10, x_scale=1.20),
        "W_c": make_diag_gate(d, h_scale=0.00, x_scale=1.40),
        "W_o": make_diag_gate(d, h_scale=0.15, x_scale=0.05),
        "b_f": np.ones(d) * 2.8,
        "b_i": np.ones(d) * -0.4,
        "b_c": np.zeros(d),
        "b_o": np.ones(d) * 1.0,
    }

    x_seq = [np.array([1.0, 0.0, 0.0, 0.0]),
             np.array([0.0, 0.0, 0.0, 0.0]),
             np.array([0.0, 0.0, 0.0, 0.0]),
             np.array([0.0, 0.0, 0.0, 0.0]),
             np.array([0.0, 0.0, 0.0, 0.0])]

    h = np.zeros(d)
    c = np.zeros(d)
    c_history = []

    print("Standard LSTM state evolution (all gates depend on [h_{t-1}, x_t]):\n")
    print(f"  {'t':>2}  {'f_t':>8}  {'i_t':>8}  {'o_t':>8}  {'c_tilde':>10}  {'c_t[0]':>10}  {'h_t[0]':>10}")
    print(f"  {'--':>2}  {'----':>8}  {'----':>8}  {'----':>8}  {'------':>10}  {'------':>10}  {'------':>10}")

    for t, x_t in enumerate(x_seq):
        h, c, f_t, i_t, c_tilde, o_t = lstm_step(x_t, h, c, params)
        c_history.append(c.copy())
        print(f"  {t:>2}  {f_t[0]:>8.4f}  {i_t[0]:>8.4f}  {o_t[0]:>8.4f}  {c_tilde[0]:>10.6f}  {c[0]:>10.6f}  {h[0]:>10.6f}")

    print(f"\nAfter 5 steps:")
    retained_pct = c[0] / c_history[0][0] * 100.0
    print(f"  Forget gate remains high (~{params['b_f'][0]:.1f} bias => f_t around {sigmoid(params['b_f'][0]):.4f})")
    print(f"  Cell state right after write: c_0[0] = {c_history[0][0]:.6f}")
    print(f"  Cell state after 5 steps:     c_4[0] = {c[0]:.6f}")
    print(f"  Retained fraction:            {retained_pct:.2f}%")

    rnn_h = np.tanh(1.0)
    for _ in range(len(x_seq) - 1):
        rnn_h = np.tanh(0.5 * rnn_h)
    print(f"  Compare: vanilla RNN hidden state after 5 steps decays to {rnn_h:.6f}")
    print(f"  LSTM cell state after 5 steps retains: {c[0]:.6f}")
    print(f"  => LSTM retains information over multiple time steps via gated cell state.")


if __name__ == "__main__":
    main()
