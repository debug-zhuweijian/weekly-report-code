"""
bilstm_context.py -- BiLSTM 前向+后向上下文聚合验证
对应周报章节：三.1 RNN→LSTM→BiLSTM
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def init_lstm_params(rng, input_dim, hidden_dim, scale=0.4):
    total_dim = hidden_dim + input_dim
    return {
        "W_f": rng.normal(scale=scale, size=(hidden_dim, total_dim)),
        "W_i": rng.normal(scale=scale, size=(hidden_dim, total_dim)),
        "W_c": rng.normal(scale=scale, size=(hidden_dim, total_dim)),
        "W_o": rng.normal(scale=scale, size=(hidden_dim, total_dim)),
        "b_f": np.ones(hidden_dim) * 0.8,
        "b_i": np.ones(hidden_dim) * -0.2,
        "b_c": np.zeros(hidden_dim),
        "b_o": np.ones(hidden_dim) * 0.2,
    }


def lstm_pass(x, params, reverse=False):
    hidden_dim = params["b_f"].shape[0]
    h = np.zeros(hidden_dim)
    c = np.zeros(hidden_dim)
    h_list = []
    c_list = []
    gate_list = []

    indices = range(len(x) - 1, -1, -1) if reverse else range(len(x))
    for t in indices:
        z = np.concatenate([h, x[t]])
        f_t = sigmoid(params["W_f"] @ z + params["b_f"])
        i_t = sigmoid(params["W_i"] @ z + params["b_i"])
        c_tilde = np.tanh(params["W_c"] @ z + params["b_c"])
        c = f_t * c + i_t * c_tilde
        o_t = sigmoid(params["W_o"] @ z + params["b_o"])
        h = o_t * np.tanh(c)

        if reverse:
            h_list.insert(0, h.copy())
            c_list.insert(0, c.copy())
            gate_list.insert(0, (f_t.copy(), i_t.copy(), o_t.copy()))
        else:
            h_list.append(h.copy())
            c_list.append(c.copy())
            gate_list.append((f_t.copy(), i_t.copy(), o_t.copy()))

    return np.array(h_list), np.array(c_list), gate_list


def main():
    seq_len = 5
    input_dim = 3
    hidden_dim = 4

    rng_input = np.random.default_rng(42)
    x = rng_input.normal(size=(seq_len, input_dim))
    print(f"Input sequence shape: ({seq_len}, {input_dim})")
    print(f"Input:\n{x.round(3)}\n")

    forward_params = init_lstm_params(np.random.default_rng(42), input_dim, hidden_dim)
    backward_params = init_lstm_params(np.random.default_rng(123), input_dim, hidden_dim)

    forward_h, forward_c, forward_gates = lstm_pass(x, forward_params, reverse=False)
    backward_h, backward_c, backward_gates = lstm_pass(x, backward_params, reverse=True)
    bilstm_output = np.concatenate([forward_h, backward_h], axis=1)

    print(f"Forward hidden states shape:  {forward_h.shape}")
    print(f"Forward cell states shape:    {forward_c.shape}")
    print(f"Backward hidden states shape: {backward_h.shape}")
    print(f"Backward cell states shape:   {backward_c.shape}")
    print(f"BiLSTM output shape: {bilstm_output.shape} (seq_len x 2*hidden_dim)\n")

    for t in range(seq_len):
        print(f"  t={t}: forward_h[0]={forward_h[t, 0]:>7.4f}, backward_h[0]={backward_h[t, 0]:>7.4f}, "
              f"concat[0:2]=[{bilstm_output[t, 0]:.4f}, {bilstm_output[t, hidden_dim]:.4f}]")

    f0, i0, o0 = forward_gates[0]
    bf0, bi0, bo0 = backward_gates[0]
    print("\nSample gate values at original position t=0:")
    print(f"  Forward LSTM:  f={f0[0]:.4f}, i={i0[0]:.4f}, o={o0[0]:.4f}")
    print(f"  Backward LSTM: f={bf0[0]:.4f}, i={bi0[0]:.4f}, o={bo0[0]:.4f}")

    print("\nBidirectional context capture:")
    print("  Forward LSTM state at t=0 only sees x[0], no future context")
    print(f"  Backward LSTM state at t=0 has processed x[{seq_len - 1}]->x[0], so it already encodes future context")
    print("  => BiLSTM output at each position combines past AND future information through two true LSTM cells")


if __name__ == "__main__":
    main()
