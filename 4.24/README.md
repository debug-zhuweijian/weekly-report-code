# 4.24 周报代码

## 代码索引

| 文件名 | 验证目标 | 对应周报章节 |
|--------|----------|--------------|
| dft_basic.py | DFT 单边幅度谱与周期图 | 一.1 从时域到频域 |
| stft.py | 离散 STFT 分帧与峰值频率迁移 | 一.1 从时域到频域 |
| mel_filterbank.py | Hz/Mel 互换与三角滤波器组 | 一.2 感知驱动的频率压缩 |
| dct_ii.py | DCT-II 正交归一化与重构 | 一.3 MFCC 与动态特征 |
| mfcc_delta.py | Delta / Delta-Delta 回归差分 | 一.3 MFCC 与动态特征 |
| hilbert_envelope.py | Hilbert 解析信号与包络 | 二.2 包络提取与标准化 |
| notch_filter.py | 二阶 IIR 陷波滤波 | 二.1 预处理管线 |
| car.py | CAR 公共平均参考 | 二.1 预处理管线 |
| bandpass_filter.py | Butterworth 带通滤波 | 二.1 预处理管线 |
| zscore.py | Z-score 标准化 | 二.2 包络提取与标准化 |
| feature_matrix.py | T x C 特征矩阵构建 | 二.2 包络提取与标准化 |
| rnn_gradient.py | RNN 隐状态衰减与梯度消失 | 三.1 RNN→LSTM→BiLSTM |
| lstm_cell.py | LSTM 门控与状态保持 | 三.1 RNN→LSTM→BiLSTM |
| bilstm_context.py | BiLSTM 双向上下文 | 三.1 RNN→LSTM→BiLSTM |
| ctc_alignment.py | CTC 路径枚举与边际概率 | 三.2 时序对齐 |
| conv1d.py | 1D 卷积输出长度与因果填充 | 三.3 卷积基础 |
| dilated_causal_conv.py | 膨胀因果卷积感受野 | 三.3 卷积基础 |

## 运行依赖

- Python 3.8+
- `numpy`
- `scipy`（仅 `bandpass_filter.py` 需要）

## 使用方式

```bash
cd 4.24
python dft_basic.py
python stft.py
python mfcc_delta.py
```

每个脚本都可以独立运行。
