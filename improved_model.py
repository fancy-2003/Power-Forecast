"""
改进模型实现 - CNN-Transformer混合模型
结合卷积层提取局部特征和Transformer处理长期依赖
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization,
                                   MultiHeadAttention, Conv1D, MaxPooling1D,
                                   Flatten, Concatenate, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import time
import os
from forecast_losses import make_forecast_loss

class ImprovedModel:
    """CNN-Transformer混合模型 - 类名统一为 ImprovedModel"""

    def __init__(self, sequence_length=90, prediction_length=90, n_features=None):
        """
        初始化CNN-Transformer混合模型

        Args:
            sequence_length: 输入序列长度
            prediction_length: 预测序列长度
            n_features: 特征数量
        """
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.n_features = n_features
        self.model = None
        self.history = None

    def build_model(self, d_model=128, n_heads=8, dff=512, n_encoder_layers=3,
                   conv_filters=[64, 128, 256], conv_kernel_sizes=[3, 5, 7],
                   dropout_rate=0.1, learning_rate=0.001):
        """
        构建CNN-Transformer混合模型

        Args:
            d_model: 模型维度
            n_heads: 多头注意力头数
            dff: 前馈网络维度
            n_encoder_layers: 编码器层数
            conv_filters: 卷积层滤波器数量列表
            conv_kernel_sizes: 卷积核大小列表
            dropout_rate: Dropout率
            learning_rate: 学习率

        Returns:
            构建好的模型
        """
        # 输入层
        inputs = Input(shape=(self.sequence_length, self.n_features))

        # 1. CNN分支 - 提取局部特征
        cnn_branch = inputs

        # 多尺度卷积
        conv_outputs = []
        for i, (filters, kernel_size) in enumerate(zip(conv_filters, conv_kernel_sizes)):
            # 卷积 + 批归一化 + ReLU + 池化
            x = Conv1D(filters=filters, kernel_size=kernel_size,
                      padding='same', activation='relu')(cnn_branch)
            x = tf.keras.layers.BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(dropout_rate)(x)
            conv_outputs.append(x)

        # 合并卷积特征
        if len(conv_outputs) > 1:
            cnn_features = Concatenate()(conv_outputs)
        else:
            cnn_features = conv_outputs[0]

        # 全局池化
        cnn_features = GlobalAveragePooling1D()(cnn_features)

        # 2. Transformer分支 - 处理长期依赖
        # 如果 n_features != d_model，先用 Dense 层映射
        if self.n_features != d_model:
            x = Dense(d_model)(inputs)
        else:
            x = inputs

        # 添加位置编码
        pos_encoding = self.get_positional_encoding(self.sequence_length, d_model)
        x = tf.keras.layers.Add()([x, pos_encoding[:, :self.sequence_length, :]])

        # 编码器层
        for _ in range(n_encoder_layers):
            # 多头注意力
            attention = MultiHeadAttention(num_heads=n_heads, key_dim=d_model,
                                          dropout=dropout_rate)(x, x)
            x = LayerNormalization(epsilon=1e-6)(x + attention)

            # 前馈网络
            ffn = Dense(dff, activation='relu')(x)
            ffn = Dropout(dropout_rate)(ffn)
            ffn = Dense(d_model)(ffn)
            ffn = Dropout(dropout_rate)(ffn)
            x = LayerNormalization(epsilon=1e-6)(x + ffn)

        # 全局平均池化
        transformer_features = GlobalAveragePooling1D()(x)

        # 3. 特征融合
        combined_features = Concatenate()([cnn_features, transformer_features])

        # 全连接层
        x = Dense(dff, activation='relu')(combined_features)
        x = Dropout(dropout_rate)(x)
        x = Dense(dff // 2, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

        # 输出层
        outputs = Dense(self.prediction_length, activation='linear')(x)

        # 创建模型
        model = Model(inputs=inputs, outputs=outputs)

        # 编译模型 - 使用梯度裁剪
        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=make_forecast_loss(), metrics=['mae'])

        self.model = model
        print(f"模型构建完成，总参数数: {model.count_params()}")
        return model

    def get_positional_encoding(self, position, d_model):
        """
        生成位置编码

        Args:
            position: 位置数量
            d_model: 模型维度

        Returns:
            位置编码矩阵
        """
        def get_angle(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates

        angle_rads = get_angle(np.arange(position)[:, np.newaxis],
                              np.arange(d_model)[np.newaxis, :],
                              d_model)

        # 对偶数索引使用sin，奇数索引使用cos
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """
        训练模型

        Args:
            X_train: 训练集输入
            y_train: 训练集目标
            X_val: 验证集输入
            y_val: 验证集目标
            epochs: 训练轮数
            batch_size: 批次大小

        Returns:
            训练历史
        """
        print(f"开始训练CNN-Transformer混合模型（序列长度={self.sequence_length}, 预测长度={self.prediction_length}）...")

        # 定义回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)
        ]

        # 训练模型
        start_time = time.time()

        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        training_time = time.time() - start_time
        print(f"模型训练完成，耗时 {training_time:.2f} 秒")
        print(f"最终训练loss: {self.history.history['loss'][-1]:.6f}")
        if 'val_loss' in self.history.history:
            print(f"最终验证loss: {self.history.history['val_loss'][-1]:.6f}")

        return self.history

    def predict(self, X_test):
        """
        预测

        Args:
            X_test: 测试集输入

        Returns:
            预测结果
        """
        return self.model.predict(X_test, verbose=0)

    def evaluate(self, X_test, y_test):
        """
        评估模型

        Args:
            X_test: 测试集输入
            y_test: 测试集目标

        Returns:
            评估指标
        """
        # 预测
        y_pred = self.predict(X_test)

        # 计算指标
        mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
        mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
        rmse = np.sqrt(mse)

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }

    def plot_training_history(self):
        """绘制训练历史"""
        if self.history is None:
            print("没有训练历史可绘制")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # MAE曲线
        ax2.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('cnn_transformer_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_predictions(self, y_test, y_pred, title="CNN-Transformer Predictions"):
        """
        绘制预测结果对比图

        Args:
            y_test: 真实值
            y_pred: 预测值
            title: 图表标题
        """
        plt.figure(figsize=(15, 6))

        # 绘制前10个样本的预测结果
        n_samples = min(10, len(y_test))

        for i in range(n_samples):
            plt.subplot(2, 5, i+1)
            plt.plot(y_test[i], label='True', color='blue', linewidth=2)
            plt.plot(y_pred[i], label='Predicted', color='red', linestyle='--', linewidth=2)
            plt.title(f'Sample {i+1}')
            plt.xlabel('Time Step')
            plt.ylabel('Power')
            plt.legend()
            plt.grid(True)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(f'cnn_transformer_predictions_{self.prediction_length}days.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, filepath):
        """保存模型"""
        self.model.save(filepath)
        print(f"模型已保存到 {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"模型已从 {filepath} 加载")

def run_improved_experiments(X_train, y_train, X_test, y_test, n_features,
                            sequence_lengths=[90], prediction_lengths=[90, 365],
                            n_experiments=5):
    """
    运行CNN-Transformer混合模型实验

    Args:
        X_train: 训练集输入
        y_train: 训练集目标
        X_test: 测试集输入
        y_test: 测试集目标
        n_features: 特征数量
        sequence_lengths: 序列长度列表
        prediction_lengths: 预测长度列表
        n_experiments: 实验次数

    Returns:
        实验结果
    """
    results = {}

    for seq_len in sequence_lengths:
        for pred_len in prediction_lengths:
            print(f"\n{'='*50}")
            print(f"实验配置: 序列长度={seq_len}, 预测长度={pred_len}")
            print(f"{'='*50}")

            experiment_results = []

            for exp in range(n_experiments):
                print(f"\n实验 {exp+1}/{n_experiments}")

                # 创建模型
                model = ImprovedModel(sequence_length=seq_len, prediction_length=pred_len, n_features=n_features)
                model.build_model(d_model=128, n_heads=8, dff=512, n_encoder_layers=3,
                                conv_filters=[64, 128, 256], conv_kernel_sizes=[3, 5, 7],
                                dropout_rate=0.1, learning_rate=0.001)

                # 分割训练集和验证集
                split_idx = int(len(X_train) * 0.8)
                X_tr = X_train[:split_idx]
                y_tr = y_train[:split_idx]
                X_val = X_train[split_idx:]
                y_val = y_train[split_idx:]

                # 训练模型
                history = model.train(X_tr, y_tr, X_val, y_val, epochs=100, batch_size=32)

                # 评估模型
                metrics = model.evaluate(X_test, y_test)
                experiment_results.append(metrics)

                print(f"实验 {exp+1} 结果: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

                # 绘制预测结果
                if exp == 0:  # 只绘制第一次实验的结果
                    y_pred = model.predict(X_test)
                    model.plot_predictions(y_test, y_pred,
                                        f"CNN-Transformer Predictions (Seq={seq_len}, Pred={pred_len})")

            # 计算统计结果
            mse_values = [r['mse'] for r in experiment_results]
            mae_values = [r['mae'] for r in experiment_results]
            rmse_values = [r['rmse'] for r in experiment_results]

            results[f"seq{seq_len}_pred{pred_len}"] = {
                'mse_mean': np.mean(mse_values),
                'mse_std': np.std(mse_values),
                'mae_mean': np.mean(mae_values),
                'mae_std': np.std(mae_values),
                'rmse_mean': np.mean(rmse_values),
                'rmse_std': np.std(rmse_values),
                'individual_results': experiment_results
            }

            print(f"\n实验配置 {seq_len}-{pred_len} 的统计结果:")
            print(f"MSE: {np.mean(mse_values):.4f} ± {np.std(mse_values):.4f}")
            print(f"MAE: {np.mean(mae_values):.4f} ± {np.std(mae_values):.4f}")
            print(f"RMSE: {np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}")

    return results

if __name__ == "__main__":
    print("CNN-Transformer混合模型模块已加载")
