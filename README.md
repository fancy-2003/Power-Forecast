# 电力消耗预测 - 深度学习课程项目

## 项目概述

基于家庭电力消耗数据和天气数据，使用 LSTM、Transformer 和 CNN-Transformer 三种深度学习模型进行短期（90天）和长期（365天）电力消耗预测。

## 项目结构

```
.
├── data_preprocessing.py      # 数据预处理模块
├── experiment_saver.py        # 实验结果保存工具（支持断点续跑）
├── lstm_model.py              # LSTM模型
├── transformer_model.py       # Transformer模型
├── improved_model.py          # CNN-Transformer混合模型
├── main_experiment.py         # 主实验脚本
├── save_dataset.py            # 单独保存数据集脚本
├── household_power_consumption.txt    # 电力数据
├── MENSQ_75_previous-1950-2024.csv  # 天气数据
├── requirements.txt           # 依赖包列表
└── README.md                  # 本文件
```

## 环境配置

```bash
# 创建conda环境
conda create -n power_forecast python=3.10
conda activate power_forecast

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 运行完整实验

```bash
python main_experiment.py
```

实验结果将自动保存到 `experiment_results/时间戳/` 目录下。

### 2. 断点续跑

如果实验中断，直接重新运行即可：

```bash
python main_experiment.py
```

系统会自动跳过已完成的实验，继续运行剩余部分。

### 3. 单独保存数据集

如果只需要生成 train.csv 和 test.csv：

```bash
python save_dataset.py
```

数据将保存到 `dataset/` 目录下。

## 实验结果

运行完成后，结果保存在：

```
experiment_results/
└── 20260618_131500/
    ├── models/              # 保存的模型
    ├── figures/             # 图表
    ├── data/                # 实验数据
    ├── logs/                # 日志
    ├── reports/             # 报告
    └── checkpoints/         # 检查点（断点续跑用）
```

## 数据预处理说明

| 数据类型 | 字段 | 处理方式 |
|---------|------|---------|
| 电力数据 | Global_active_power | 按天取总和 |
| 电力数据 | Global_reactive_power | 按天取总和 |
| 电力数据 | Sub_metering_1, Sub_metering_2 | 按天取总和 |
| 电力数据 | Voltage | 按天取平均 |
| 电力数据 | Global_intensity | 按天取平均 |
| 天气数据 | RR, NBJRR1, NBJRR5, NBJRR10, NBJBROU | 取当天任意一个数据 |

## 模型说明

### LSTM
- 两层LSTM结构
- 适合捕捉时间序列的长期依赖

### Transformer
- 多头注意力机制
- 适合处理长序列预测

### CNN-Transformer（改进模型）
- CNN提取局部特征 + Transformer捕捉全局依赖
- 多尺度卷积核（3, 5, 7）

## 评价指标

- **MSE**（均方误差）
- **MAE**（平均绝对误差）
- 每种模型运行5轮实验，报告平均值±标准差

## 参考文献

[1] Hochreiter S, Schmidhuber J. Long short-term memory[J]. Neural computation, 1997.

[2] Vaswani A, et al. Attention is all you need[C]. NeurIPS, 2017.

## 作者

[范怡鑫]

## 许可证

本项目仅用于学术目的。
