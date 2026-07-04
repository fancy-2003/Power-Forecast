"""
主实验脚本
数据预处理并保存 train.csv / test.csv
运行 LSTM、Transformer、CNN-Transformer 三种模型
支持断点续跑
自动保存所有结果、图表和模型
生成实验报告所需的所有数据
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# 导入自定义模块
from data_preprocessing import DataPreprocessor, split_data
from experiment_saver import ExperimentSaver

# 导入模型（如果文件存在）
try:
    from lstm_model import LSTMModel, run_lstm_experiments
    LSTM_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入LSTM模型: {e}")
    LSTM_AVAILABLE = False

try:
    from transformer_model import TransformerModel, run_transformer_experiments
    TRANSFORMER_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入Transformer模型: {e}")
    TRANSFORMER_AVAILABLE = False

try:
    from improved_model import ImprovedModel, run_improved_experiments
    IMPROVED_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入CNN-Transformer模型: {e}")
    IMPROVED_AVAILABLE = False


def setup_environment():
    """设置实验环境"""
    print("实验环境设置完成")
    # 设置TensorFlow日志级别
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')


def load_and_preprocess_data():
    """加载和预处理数据"""
    print("开始数据预处理...")

    preprocessor = DataPreprocessor(
        'household_power_consumption.txt',
        'MENSQ_75_previous-1950-2024.csv'
    )

    # 短期预测数据（90天预测90天）- 使用归一化
    print("\n处理短期预测数据...")
    X_short, y_short, data_short = preprocessor.preprocess(
        sequence_length=90, prediction_length=90,
        normalize=True, norm_method='minmax'
    )

    # 保存归一化参数
    preprocessor.save_scalers('scaler_short.json')

    # 长期预测数据（90天预测365天）- 使用归一化
    print("\n处理长期预测数据...")
    # 创建新的预处理器用于长期数据（避免归一化参数冲突）
    preprocessor_long = DataPreprocessor(
        'household_power_consumption.txt',
        'MENSQ_75_previous-1950-2024.csv'
    )
    X_long, y_long, data_long = preprocessor_long.preprocess(
        sequence_length=90, prediction_length=365,
        normalize=True, norm_method='minmax'
    )
    preprocessor_long.save_scalers('scaler_long.json')

    # 分割数据
    X_train_short, y_train_short, X_test_short, y_test_short = split_data(X_short, y_short)
    X_train_long, y_train_long, X_test_long, y_test_long = split_data(X_long, y_long)

    print("\n数据预处理完成！")
    print(f"短期预测 - 训练集: {X_train_short.shape}, 测试集: {X_test_short.shape}")
    print(f"长期预测 - 训练集: {X_train_long.shape}, 测试集: {X_test_long.shape}")

    # 数据质量检查
    print("\n数据质量检查:")
    print(f"  短期 X_train 范围: [{X_train_short.min():.4f}, {X_train_short.max():.4f}]")
    print(f"  短期 y_train 范围: [{y_train_short.min():.4f}, {y_train_short.max():.4f}]")
    print(f"  长期 X_train 范围: [{X_train_long.min():.4f}, {X_train_long.max():.4f}]")
    print(f"  长期 y_train 范围: [{y_train_long.min():.4f}, {y_train_long.max():.4f}]")

    n_features = X_short.shape[2]

    return {
        'X_train_short': X_train_short,
        'y_train_short': y_train_short,
        'X_test_short': X_test_short,
        'y_test_short': y_test_short,
        'X_train_long': X_train_long,
        'y_train_long': y_train_long,
        'X_test_long': X_test_long,
        'y_test_long': y_test_long,
        'n_features': n_features,
        'data_short': data_short,
        'data_long': data_long,
        'preprocessor_short': preprocessor,
        'preprocessor_long': preprocessor_long
    }


def run_single_experiment(model_class, model_name, X_train, y_train, X_test, y_test,
                         n_features, sequence_length, prediction_length,
                         n_experiments=5, saver=None, preprocessor=None):
    """
    运行单个模型的实验（支持断点续跑）

    Args:
        model_class: 模型类
        model_name: 模型名称
        X_train, y_train: 训练数据（已归一化）
        X_test, y_test: 测试数据（已归一化）
        n_features: 特征数
        sequence_length: 序列长度
        prediction_length: 预测长度
        n_experiments: 实验次数
        saver: 保存器
        preprocessor: 预处理器（用于反归一化）
    """
    config_name = f"seq{sequence_length}_pred{prediction_length}"
    print(f"\n{'='*60}")
    print(f"模型: {model_name}, 配置: {config_name}")
    print(f"{'='*60}")

    experiment_results = []
    skipped_count = 0

    for exp in range(n_experiments):
        experiment_id = f"exp_{exp+1}"

        # 检查是否已完成
        if saver and saver.is_experiment_completed(model_name, config_name, experiment_id):
            print(f"  实验 {exp+1}/{n_experiments} 已完成，跳过...")
            skipped_count += 1
            # 从检查点加载结果
            key = f"{model_name}_{config_name}_{experiment_id}"
            if key in saver.completed_experiments:
                saved_result = saver.completed_experiments[key].get('results')
                if saved_result:
                    experiment_results.append(saved_result)
            continue

        print(f"\n  实验 {exp+1}/{n_experiments}")
        print(f"  {'-'*40}")

        try:
            # 创建模型
            model = model_class(
                sequence_length=sequence_length,
                prediction_length=prediction_length,
                n_features=n_features
            )

            # 构建模型
            if model_name == 'LSTM':
                model.build_model(units=128, dropout_rate=0.2, learning_rate=0.001)
            elif model_name == 'Transformer':
                model.build_model(d_model=128, n_heads=8, dff=512,
                                n_encoder_layers=3, dropout_rate=0.1, learning_rate=0.001)
            elif model_name == 'CNN-Transformer':
                model.build_model(d_model=128, n_heads=8, dff=512,
                                n_encoder_layers=3, dropout_rate=0.1, learning_rate=0.001)

            # 分割训练集和验证集
            split_idx = int(len(X_train) * 0.8)
            X_tr = X_train[:split_idx]
            y_tr = y_train[:split_idx]
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]

            # 训练模型
            history = model.train(X_tr, y_tr, X_val, y_val, epochs=100, batch_size=32)

            # 评估模型（在归一化空间）
            metrics_norm = model.evaluate(X_test, y_test)

            # 反归一化后评估
            y_pred_norm = model.predict(X_test)
            if preprocessor:
                y_pred_orig = preprocessor.denormalize_target(y_pred_norm)
                y_test_orig = preprocessor.denormalize_target(y_test)

                from sklearn.metrics import mean_squared_error, mean_absolute_error
                mse_orig = mean_squared_error(y_test_orig.flatten(), y_pred_orig.flatten())
                mae_orig = mean_absolute_error(y_test_orig.flatten(), y_pred_orig.flatten())
                rmse_orig = np.sqrt(mse_orig)

                metrics = {
                    'mse_norm': metrics_norm['mse'],
                    'mae_norm': metrics_norm['mae'],
                    'rmse_norm': metrics_norm['rmse'],
                    'mse': mse_orig,
                    'mae': mae_orig,
                    'rmse': rmse_orig
                }
            else:
                metrics = metrics_norm

            experiment_results.append(metrics)

            print(f"  实验 {exp+1} 结果 (归一化空间): MSE={metrics_norm['mse']:.4f}, MAE={metrics_norm['mae']:.4f}, RMSE={metrics_norm['rmse']:.4f}")
            if 'mse' in metrics and metrics['mse'] != metrics_norm['mse']:
                print(f"  实验 {exp+1} 结果 (原始空间): MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

            # 保存结果到检查点
            if saver:
                saver.mark_experiment_completed(model_name, config_name, experiment_id, metrics)

            # 绘制并保存预测结果（只保存第一次实验的）
            if exp == 0 and saver:
                y_pred_norm = model.predict(X_test)

                # 反归一化用于绘图
                if preprocessor:
                    y_pred_plot = preprocessor.denormalize_target(y_pred_norm)
                    y_test_plot = preprocessor.denormalize_target(y_test)
                else:
                    y_pred_plot = y_pred_norm
                    y_test_plot = y_test

                # 绘制预测对比图
                fig, axes = plt.subplots(2, 5, figsize=(20, 8))
                n_samples = min(10, len(y_test_plot))
                for i in range(n_samples):
                    ax = axes[i//5, i%5]
                    ax.plot(y_test_plot[i], label='True', color='blue', linewidth=2)
                    ax.plot(y_pred_plot[i], label='Predicted', color='red', linestyle='--', linewidth=2)
                    ax.set_title(f'Sample {i+1}')
                    ax.legend()
                    ax.grid(True)
                plt.suptitle(f'{model_name} Predictions ({config_name})', fontsize=16)
                plt.tight_layout()
                saver.save_figure(fig, f'{model_name.lower()}_predictions_{config_name}',
                                subdir=model_name.lower())
                plt.close()

                # 绘制训练历史
                if history and hasattr(history, 'history'):
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    ax1.plot(history.history['loss'], label='Training Loss')
                    if 'val_loss' in history.history:
                        ax1.plot(history.history['val_loss'], label='Validation Loss')
                    ax1.set_title(f'{model_name} Loss')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.legend()
                    ax1.grid(True)

                    ax2.plot(history.history['mae'], label='Training MAE')
                    if 'val_mae' in history.history:
                        ax2.plot(history.history['val_mae'], label='Validation MAE')
                    ax2.set_title(f'{model_name} MAE')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('MAE')
                    ax2.legend()
                    ax2.grid(True)

                    plt.tight_layout()
                    saver.save_figure(fig, f'{model_name.lower()}_training_{config_name}',
                                    subdir=model_name.lower())
                    plt.close()

            # 保存模型（只保存第一次实验的）
            if exp == 0 and saver and hasattr(model, 'model'):
                try:
                    saver.save_model(model.model, f'{model_name.lower()}_{config_name}', framework='keras')
                except Exception as e:
                    print(f"  模型保存失败: {e}")

        except Exception as e:
            print(f"  实验 {exp+1} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    if skipped_count > 0:
        print(f"\n  跳过了 {skipped_count} 个已完成的实验")

    if not experiment_results:
        print(f"  警告: 没有成功的实验结果！")
        return None

    # 计算统计结果
    mse_values = [r['mse'] for r in experiment_results]
    mae_values = [r['mae'] for r in experiment_results]
    rmse_values = [r['rmse'] for r in experiment_results]

    results = {
        'mse_mean': np.mean(mse_values),
        'mse_std': np.std(mse_values),
        'mae_mean': np.mean(mae_values),
        'mae_std': np.std(mae_values),
        'rmse_mean': np.mean(rmse_values),
        'rmse_std': np.std(rmse_values),
        'individual_results': experiment_results,
        'n_successful': len(experiment_results),
        'n_total': n_experiments
    }

    print(f"\n  统计结果:")
    print(f"  MSE: {results['mse_mean']:.4f} ± {results['mse_std']:.4f}")
    print(f"  MAE: {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
    print(f"  RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")

    return results


def run_all_experiments(data_dict, saver=None):
    """
    运行所有实验（支持断点续跑）

    Args:
        data_dict: 数据字典
        saver: 保存器
    """
    all_results = {}

    # 实验配置
    configs = [
        ('short', 90, 90),
        ('long', 90, 365)
    ]

    # LSTM 实验
    if LSTM_AVAILABLE:
        print("\n" + "="*60)
        print("开始 LSTM 模型实验")
        print("="*60)
        lstm_results = {}

        for config_name, seq_len, pred_len in configs:
            X_train = data_dict[f'X_train_{config_name}']
            y_train = data_dict[f'y_train_{config_name}']
            X_test = data_dict[f'X_test_{config_name}']
            y_test = data_dict[f'y_test_{config_name}']
            preprocessor = data_dict[f'preprocessor_{config_name}']

            result = run_single_experiment(
                LSTMModel, 'LSTM',
                X_train, y_train, X_test, y_test,
                data_dict['n_features'], seq_len, pred_len,
                n_experiments=5, saver=saver, preprocessor=preprocessor
            )

            if result:
                lstm_results[f'seq{seq_len}_pred{pred_len}'] = result
                if saver:
                    saver.save_results(result, 'LSTM', f'seq{seq_len}_pred{pred_len}')

        all_results['LSTM'] = lstm_results

    # Transformer 实验
    if TRANSFORMER_AVAILABLE:
        print("\n" + "="*60)
        print("开始 Transformer 模型实验")
        print("="*60)
        transformer_results = {}

        for config_name, seq_len, pred_len in configs:
            X_train = data_dict[f'X_train_{config_name}']
            y_train = data_dict[f'y_train_{config_name}']
            X_test = data_dict[f'X_test_{config_name}']
            y_test = data_dict[f'y_test_{config_name}']
            preprocessor = data_dict[f'preprocessor_{config_name}']

            result = run_single_experiment(
                TransformerModel, 'Transformer',
                X_train, y_train, X_test, y_test,
                data_dict['n_features'], seq_len, pred_len,
                n_experiments=5, saver=saver, preprocessor=preprocessor
            )

            if result:
                transformer_results[f'seq{seq_len}_pred{pred_len}'] = result
                if saver:
                    saver.save_results(result, 'Transformer', f'seq{seq_len}_pred{pred_len}')

        all_results['Transformer'] = transformer_results

    # CNN-Transformer 实验
    if IMPROVED_AVAILABLE:
        print("\n" + "="*60)
        print("开始 CNN-Transformer 模型实验")
        print("="*60)
        improved_results = {}

        for config_name, seq_len, pred_len in configs:
            X_train = data_dict[f'X_train_{config_name}']
            y_train = data_dict[f'y_train_{config_name}']
            X_test = data_dict[f'X_test_{config_name}']
            y_test = data_dict[f'y_test_{config_name}']
            preprocessor = data_dict[f'preprocessor_{config_name}']

            result = run_single_experiment(
                ImprovedModel, 'CNN-Transformer',
                X_train, y_train, X_test, y_test,
                data_dict['n_features'], seq_len, pred_len,
                n_experiments=5, saver=saver, preprocessor=preprocessor
            )

            if result:
                improved_results[f'seq{seq_len}_pred{pred_len}'] = result
                if saver:
                    saver.save_results(result, 'CNN-Transformer', f'seq{seq_len}_pred{pred_len}')

        all_results['CNN-Transformer'] = improved_results

    return all_results


def main():
    """主函数"""
    print("="*60)
    print("机器学习课程项目 - 电力消耗预测 (修复版)")
    print("支持断点续跑 + 数据归一化")
    print("="*60)

    # 设置环境
    setup_environment()

    # 初始化保存器（支持断点续跑）
    saver = ExperimentSaver(base_dir='experiment_results', resume=True)

    try:
        # 加载和预处理数据
        data_dict = load_and_preprocess_data()

        # 保存数据信息
        if saver:
            data_info = {
                'n_features': int(data_dict['n_features']),
                'short_train_shape': [int(x) for x in data_dict['X_train_short'].shape],
                'short_test_shape': [int(x) for x in data_dict['X_test_short'].shape],
                'long_train_shape': [int(x) for x in data_dict['X_train_long'].shape],
                'long_test_shape': [int(x) for x in data_dict['X_test_long'].shape]
            }
            saver.save_data(data_info, 'data_info', format='json')

        # 运行所有实验
        results = run_all_experiments(data_dict, saver)

        # 保存总结报告
        if saver:
            saver.save_summary_report(results)
            summary = saver.get_summary()
            print(f"\n{'='*60}")
            print("实验完成！")
            print(f"结果保存到: {summary['experiment_dir']}")
            print(f"已完成实验: {len(summary['completed_experiments'])} 个")
            print(f"{'='*60}")

        print("\n所有实验完成！")

    except Exception as e:
        print(f"\n实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

        # 保存错误日志
        if saver:
            error_log = f"错误时间: {pd.Timestamp.now()}\n错误信息: {str(e)}\n\n{traceback.format_exc()}"
            saver.save_log(error_log, 'error_log')

        raise


if __name__ == "__main__":
    main()