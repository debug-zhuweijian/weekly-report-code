"""
lstm_cell.py -- LSTM 门控机制 + 长期状态保留验证
对应周报章节：序列建模机制的演进
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    d = 4
    W_f = np.eye(d) * 0.5
    W_i = np.eye(d) * 0.8
    W_c = np.eye(d) * 0.5
    W_o = np.eye(d) * 0.3
    b_f = np.ones(d) * 3.0
    b_i = np.ones(d) * 1.0
    b_c = np.zeros(d)
    b_o = np.ones(d) * 1.0

    x_seq = [np.array([1.0, 0.0, 0.0, 0.0]),
             np.array([0.0, 0.0, 0.0, 0.0]),
             np.array([0.0, 0.0, 0.0, 0.0]),
             np.array([0.0, 0.0, 0.0, 0.0]),
             np.array([0.0, 0.0, 0.0, 0.0])]

    h = np.zeros(d)
    c = np.zeros(d)

    print("LSTM state evolution (input only at t=0, then zeros):\n")
    print(f"  {'t':>2}  {'f_t':>8}  {'i_t':>8}  {'c_tilde':>10}  {'c_t[0]':>10}  {'h_t[0]':>10}")
    print(f"  {'--':>2}  {'----':>8}  {'----':>8}  {'------':>10}  {'------':>10}  {'------':>10}")

    for t, x_t in enumerate(x_seq):
        f_t = sigmoid(W_f @ h + b_f)
        i_t = sigmoid(W_i @ h + x_t + b_i)
        c_tilde = np.tanh(W_c @ x_t + b_c)
        c = f_t * c + i_t * c_tilde
        o_t = sigmoid(W_o @ h + b_o)
        h = o_t * np.tanh(c)

        print(f"  {t:>2}  {f_t[0]:>8.4f}  {i_t[0]:>8.4f}  {c_tilde[0]:>10.6f}  {c[0]:>10.6f}  {h[0]:>10.6f}")

    print(f"\nAfter 5 steps:")
    print(f"  Forget gate f_t stays high (~{sigmoid(b_f[0]):.4f}), preserving cell state")
    print(f"  Cell state c_t[0] = {c[0]:.6f}")

    c_val = c[0]
    print(f"  Compare: RNN hidden state after 5 steps decays to ~0.045")
    print(f"  LSTM cell state after 5 steps retains: {c_val:.6f}")
    print(f"  => LSTM retains information over multiple time steps via gated cell state.")


if __name__ == "__main__":
    main()
