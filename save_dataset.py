"""
单独保存数据集脚本
运行后会生成 train.csv 和 test.csv
"""

import os
import numpy as np
import pandas as pd
import json
from data_preprocessing import DataPreprocessor

def save_dataset(X, y, save_dir='dataset', prefix='', feature_names=None):
    """
    保存数据集为CSV格式

    Args:
        X: 输入数据 (n_samples, seq_len, n_features)
        y: 目标数据 (n_samples, pred_len)
        save_dir: 保存目录
        prefix: 文件名前缀
        feature_names: 特征名称列表（可选）
    """
    os.makedirs(save_dir, exist_ok=True)

    n_samples, seq_len, n_features = X.shape
    pred_len = y.shape[1]

    # 展平X为2D
    X_flat = X.reshape(n_samples, -1)

    # 生成列名
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_features)]

    feature_cols = []
    for t in range(seq_len):
        for fname in feature_names:
            feature_cols.append(f'{fname}_t{t}')

    y_cols = [f'target_t{i}' for i in range(pred_len)]

    # 分割训练集和测试集 (80/20)
    split_idx = int(n_samples * 0.8)

    # 保存训练集
    train_df = pd.DataFrame(X_flat[:split_idx], columns=feature_cols)
    train_df[y_cols] = y[:split_idx]
    train_path = os.path.join(save_dir, f'{prefix}train.csv')
    train_df.to_csv(train_path, index=False)

    # 保存测试集
    test_df = pd.DataFrame(X_flat[split_idx:], columns=feature_cols)
    test_df[y_cols] = y[split_idx:]
    test_path = os.path.join(save_dir, f'{prefix}test.csv')
    test_df.to_csv(test_path, index=False)

    # 保存数据信息
    info = {
        'prefix': prefix,
        'n_train': int(split_idx),
        'n_test': int(n_samples - split_idx),
        'n_total': int(n_samples),
        'sequence_length': int(seq_len),
        'n_features': int(n_features),
        'prediction_length': int(pred_len),
        'feature_names': feature_names,
        'train_path': train_path,
        'test_path': test_path
    }

    info_path = os.path.join(save_dir, f'{prefix}data_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"数据集保存完成: {prefix}")
    print(f"{'='*60}")
    print(f"训练集: {train_path} ({split_idx} 条)")
    print(f"测试集: {test_path} ({n_samples - split_idx} 条)")
    print(f"信息文件: {info_path}")
    print(f"特征数: {n_features}, 序列长度: {seq_len}, 预测长度: {pred_len}")
    print(f"{'='*60}")

    return train_path, test_path


if __name__ == "__main__":
    print("开始处理数据并保存为CSV...")

    # 数据预处理
    preprocessor = DataPreprocessor(
        'household_power_consumption.txt',
        'MENSQ_75_previous-1950-2024.csv'
    )

    # 短期预测数据
    print("\n处理短期预测数据（90天→90天）...")
    X_short, y_short, data_short = preprocessor.preprocess(sequence_length=90, prediction_length=90)

    feature_names = data_short.columns.tolist()
    save_dataset(X_short, y_short, save_dir='dataset', prefix='short_', feature_names=feature_names)

    # 长期预测数据
    print("\n处理长期预测数据（90天→365天）...")
    X_long, y_long, data_long = preprocessor.preprocess(sequence_length=90, prediction_length=365)

    save_dataset(X_long, y_long, save_dir='dataset', prefix='long_', feature_names=feature_names)

    print("\n✅ 所有数据已保存到 dataset/ 目录")
    print("文件列表:")
    for f in sorted(os.listdir('dataset')):
        print(f"  - {f}")
