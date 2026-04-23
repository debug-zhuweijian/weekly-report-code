"""
rnn_gradient.py -- RNN 隐状态更新 + 梯度消失现象验证
对应周报章节：序列建模机制的演进
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

    print("RNN hidden state evolution (single input at t=0, then zeros):")
    for t, h_val in enumerate(hidden_states):
        print(f"  h[{t}] = {h_val:.6f}")

    T = len(x_seq)
    grad_per_step = abs(W_hh * (1 - np.array(hidden_states) ** 2))
    cumulative_grad = np.cumprod(grad_per_step)

    print(f"\nGradient magnitude per step (|dh_t/dh_{{t-1}}|):")
    for t in range(T):
        print(f"  step {t}: |grad| = {grad_per_step[t]:.6f}")

    print(f"\nCumulative gradient |dL/dh_0| (product of per-step gradients):")
    for t in range(T):
        print(f"  after {t + 1} steps: {cumulative_grad[t]:.10f}")

    print(f"\nAfter {T} steps, gradient has decayed by factor: {cumulative_grad[-1]:.2e}")
    if cumulative_grad[-1] < 0.01:
        print("=> Gradient vanishing confirmed: RNN cannot retain information over long sequences.")


if __name__ == "__main__":
    main()
