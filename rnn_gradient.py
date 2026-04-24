"""
rnn_gradient.py -- RNN 隐状态更新 + 梯度消失现象验证
对应周报章节：三.1 RNN→LSTM→BiLSTM
"""

import numpy as np


def main():
    W_hh = 0.5
    W_xh = 1.0
    x_seq = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    h = 0.0
    hidden_states = []
    for x_t in x_seq:
        h = np.tanh(W_hh * h + W_xh * x_t)
        hidden_states.append(h)
    hidden_states = np.array(hidden_states)

    print("RNN hidden state evolution (single input at t=0, then zeros):")
    print("Index convention: the first written hidden state after x[0] is denoted h_0.\n")
    for t, h_val in enumerate(hidden_states):
        print(f"  h_{t} = {h_val:.6f}")

    local_jacobians = np.abs(W_hh * (1 - hidden_states[1:] ** 2))
    cumulative_grad = np.cumprod(local_jacobians)

    print("\nLocal Jacobian per recurrent transition (|dh_t/dh_{t-1}|, starting from h_1):")
    for t, grad_val in enumerate(local_jacobians, start=1):
        print(f"  h_{t-1} -> h_{t}: |grad| = {grad_val:.6f}")

    print("\nCumulative gradient transmitted from h_0 to later hidden states:")
    for t, grad_val in enumerate(cumulative_grad, start=1):
        print(f"  |dh_{t}/dh_0| = {grad_val:.10f}")

    print(f"\nAfter 9 recurrent transitions, surviving gradient |dh_9/dh_0| = {cumulative_grad[-1]:.2e}")
    if cumulative_grad[-1] < 0.01:
        print("=> Gradient vanishing confirmed: RNN cannot retain information over long sequences.")


if __name__ == "__main__":
    main()
