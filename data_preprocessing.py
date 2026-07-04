"""
数据预处理模块
处理家庭电力消耗数据和天气数据，准备用于模型训练
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import os
warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self, power_data_path, weather_data_path):
        """
        初始化数据预处理器

        Args:
            power_data_path: 电力数据文件路径
            weather_data_path: 天气数据文件路径
        """
        self.power_data_path = power_data_path
        self.weather_data_path = weather_data_path
        self.power_data = None
        self.weather_data = None
        self.processed_data = None

        # 归一化参数
        self.feature_scaler = None  # 用于X的归一化
        self.target_scaler = None   # 用于y的归一化
        self.feature_cols = None
        self.target_col = 'Global_active_power'

    def _sync_target_scaler_from_features(self):
        """Use the target column's feature scaler for y denormalization."""
        if self.feature_scaler is None or not self.feature_cols:
            return

        if self.target_col not in self.feature_cols:
            raise ValueError(f"目标列 {self.target_col} 不在特征列中")

        target_idx = self.feature_cols.index(self.target_col)
        method = self.feature_scaler.get('method')

        if method == 'minmax':
            self.target_scaler = {
                'method': 'minmax',
                'min': self.feature_scaler['min'][target_idx],
                'max': self.feature_scaler['max'][target_idx],
                'range': self.feature_scaler['range'][target_idx]
            }
        elif method == 'standard':
            self.target_scaler = {
                'method': 'standard',
                'mean': self.feature_scaler['mean'][target_idx],
                'std': self.feature_scaler['std'][target_idx]
            }
        else:
            raise ValueError(f"不支持的归一化方法: {method}")

    def load_power_data(self):
        """加载并预处理电力数据"""
        print("加载电力数据...")

        # 先读取前几行查看格式
        try:
            sample = pd.read_csv(self.power_data_path, sep=';', nrows=5)
            print("电力数据前5行预览：")
            print(sample.head())
            print("列名：", sample.columns.tolist())
        except Exception as e:
            print(f"预览文件失败: {e}")

        # 先读取原始数据，再手动合并日期时间列
        df = pd.read_csv(self.power_data_path, sep=';',
                        na_values=['?', 'nan', 'NaN', 'NAN'])

        print(f"原始列名: {df.columns.tolist()}")

        # 检查 Date 和 Time 列是否存在
        if 'Date' not in df.columns or 'Time' not in df.columns:
            date_col = [c for c in df.columns if 'date' in c.lower()]
            time_col = [c for c in df.columns if 'time' in c.lower()]
            if date_col and time_col:
                print(f"使用列: {date_col[0]} 和 {time_col[0]}")
                df = df.rename(columns={date_col[0]: 'Date', time_col[0]: 'Time'})
            else:
                raise ValueError(f"找不到 Date 和 Time 列。可用列: {df.columns.tolist()}")

        # 手动合并日期和时间列
        datetime_formats = ['%d/%m/%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S',
                           '%d/%m/%Y %H:%M', '%Y/%m/%d %H:%M:%S']

        df['datetime'] = pd.NaT
        for fmt in datetime_formats:
            mask = df['datetime'].isna()
            if mask.any():
                try:
                    df.loc[mask, 'datetime'] = pd.to_datetime(
                        df.loc[mask, 'Date'].astype(str) + ' ' + df.loc[mask, 'Time'].astype(str),
                        format=fmt,
                        errors='coerce'
                    )
                except:
                    pass

        # 如果所有格式都失败，使用自动推断
        if df['datetime'].isna().all():
            df['datetime'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                errors='coerce'
            )

        # 删除无法解析日期的行
        df = df.dropna(subset=['datetime'])

        if len(df) == 0:
            raise ValueError("无法解析任何日期时间，请检查文件格式")

        # 设置日期时间为索引
        df = df.set_index('datetime')

        # 删除原始的 Date 和 Time 列
        df = df.drop(columns=['Date', 'Time'], errors='ignore')

        # 转换数值列
        numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        self.power_data = df

        print(f"电力数据加载完成，共 {len(self.power_data)} 条记录")
        print(f"电力数据日期范围: {self.power_data.index.min()} 至 {self.power_data.index.max()}")

        # 数据质量检查
        print("\n电力数据质量检查:")
        for col in numeric_cols:
            if col in df.columns:
                valid = df[col].notna().sum()
                print(f"  {col}: 有效值 {valid}/{len(df)} ({valid/len(df)*100:.1f}%)")

        return self.power_data

    def load_weather_data(self):
        """加载并预处理天气数据"""
        print("加载天气数据...")

        # 天气数据使用 AAAAMM 列作为日期
        df = pd.read_csv(self.weather_data_path, sep=';',
                        na_values=['?', 'nan', 'NaN', 'NAN'])

        print(f"天气数据列名: {df.columns.tolist()[:10]}...")  # 只显示前10个列名
        print(f"天气数据共 {len(df)} 条记录")

        # 检查是否有 NUM_POSTE 列（气象站编号）
        if 'NUM_POSTE' not in df.columns:
            raise ValueError("天气数据中没有 NUM_POSTE 列（气象站编号）")

        # 将 AAAAMM 转换为日期
        df['DATE'] = pd.to_datetime(
            df['AAAAMM'].astype(str),
            format='%Y%m',
            errors='coerce'
        )
        df = df.dropna(subset=['DATE'])

        # 获取电力数据的时间范围
        power_start = self.power_data.index.min()
        power_end = self.power_data.index.max()
        print(f"电力数据时间范围: {power_start.date()} 至 {power_end.date()}")

        # 遍历所有气象站，找到与电力数据时间范围有重叠的站点
        unique_stations = df['NUM_POSTE'].unique()
        print(f"共有 {len(unique_stations)} 个气象站")

        # 检查每个气象站的数据完整性（在电力数据时间段内）
        station_info = []
        for station in unique_stations:
            station_data = df[df['NUM_POSTE'] == station].copy()
            station_start = station_data['DATE'].min()
            station_end = station_data['DATE'].max()

            # 计算与电力数据的重叠
            overlap_start = max(station_start, power_start)
            overlap_end = min(station_end, power_end)

            if overlap_start <= overlap_end:
                # 检查在重叠期间，天气特征列的缺失情况
                overlap_data = station_data[
                    (station_data['DATE'] >= overlap_start) &
                    (station_data['DATE'] <= overlap_end)
                ]

                # 检查关键特征列的缺失率
                weather_cols = ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
                available_cols = [c for c in weather_cols if c in overlap_data.columns]

                if available_cols:
                    missing_rate = overlap_data[available_cols].isna().mean().mean()
                    valid_months = len(overlap_data.dropna(subset=available_cols))
                else:
                    missing_rate = 1.0
                    valid_months = 0

                station_info.append({
                    'station': station,
                    'start': station_start,
                    'end': station_end,
                    'valid_months': valid_months,
                    'missing_rate': missing_rate
                })

        # 打印前10个气象站的信息
        print("\n气象站数据完整性（前10个）：")
        station_info_sorted = sorted(station_info, key=lambda x: x['valid_months'], reverse=True)
        for info in station_info_sorted[:10]:
            print(f"  站点 {info['station']}: {info['start'].date()} 至 {info['end'].date()}, "
                  f"有效月份 {info['valid_months']}, 缺失率 {info['missing_rate']:.2%}")

        print("\n使用所有气象站数据按月份取平均值，以获得完整的时间覆盖...")

        # 选择需要的天气列
        weather_cols = ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
        available_cols = [col for col in weather_cols if col in df.columns]

        if not available_cols:
            raise ValueError("天气数据中没有可用的特征列")

        print(f"使用的天气特征: {available_cols}")

        # 按月份分组，对所有气象站取平均值
        agg_dict = {col: 'mean' for col in available_cols}
        df_monthly = df.groupby('DATE').agg(agg_dict).reset_index()

        print(f"合并后共 {len(df_monthly)} 个月的数据")
        print(f"合并后日期范围: {df_monthly['DATE'].min()} 至 {df_monthly['DATE'].max()}")

        # 检查电力数据时间段内的缺失情况
        power_period_data = df_monthly[
            (df_monthly['DATE'] >= power_start) &
            (df_monthly['DATE'] <= power_end)
        ]
        print(f"电力数据时间段内（{power_start.date()} 至 {power_end.date()}）有 {len(power_period_data)} 个月的数据")

        # 设置日期为索引
        df_monthly = df_monthly.set_index('DATE')

        # 检查并处理重复的日期索引
        duplicate_dates = df_monthly.index.duplicated().sum()
        if duplicate_dates > 0:
            print(f"发现 {duplicate_dates} 个重复日期，进行去重处理...")
            df_monthly = df_monthly[~df_monthly.index.duplicated(keep='first')]

        # 转换天气数值列
        for col in available_cols:
            if col in df_monthly.columns:
                df_monthly[col] = pd.to_numeric(df_monthly[col], errors='coerce')

        self.weather_data = df_monthly

        print(f"天气数据加载完成，共 {len(self.weather_data)} 条记录")
        print(f"天气数据日期范围: {self.weather_data.index.min()} 至 {self.weather_data.index.max()}")
        return self.weather_data

    def aggregate_daily_power(self):
        """将电力数据按天聚合"""
        print("按天聚合电力数据...")

        # 按天聚合：global_active_power、global_reactive_power、sub_metering_1、sub_metering_2 按天取总和
        # voltage、global_intensity 按天取平均
        daily_power = self.power_data.resample('D').agg({
            'Global_active_power': 'sum',
            'Global_reactive_power': 'sum',
            'Voltage': 'mean',
            'Global_intensity': 'mean',
            'Sub_metering_1': 'sum',
            'Sub_metering_2': 'sum',
            'Sub_metering_3': 'sum'
        })

        # 计算剩余能耗
        daily_power['Sub_metering_remainder'] = (
            daily_power['Global_active_power'] * 1000 / 60 -
            (daily_power['Sub_metering_1'] +
             daily_power['Sub_metering_2'] +
             daily_power['Sub_metering_3'])
        )

        # 删除包含NaN的行
        daily_power = daily_power.dropna()

        print(f"电力数据按天聚合完成，共 {len(daily_power)} 天")
        print(f"电力数据日期范围: {daily_power.index.min()} 至 {daily_power.index.max()}")

        # 打印基本统计信息
        print("\n电力数据统计信息:")
        print(daily_power[['Global_active_power', 'Global_reactive_power', 'Voltage']].describe())

        return daily_power

    def prepare_weather_features(self):
        """准备天气特征（将月级数据转换为日级）"""
        print("准备天气特征...")

        # 选择需要的天气列：RR、NBJRR1、NBJRR5、NBJRR10、NBJBROU
        weather_cols = ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
        available_cols = [col for col in weather_cols if col in self.weather_data.columns]

        if not available_cols:
            print("警告：没有可用的天气数据列")
            return pd.DataFrame()

        # 获取月级天气数据
        monthly_weather = self.weather_data[available_cols].copy()

        # 确保索引是唯一的（去重）
        if monthly_weather.index.duplicated().any():
            print("天气数据索引仍有重复，进行去重...")
            monthly_weather = monthly_weather[~monthly_weather.index.duplicated(keep='first')]

        # 将月级数据重采样为日级（向前填充，即该月内每天使用相同的月值）
        daily_weather = monthly_weather.resample('D').ffill()

        # 删除NaN值
        daily_weather = daily_weather.dropna()

        print(f"天气特征准备完成，共 {len(daily_weather)} 天")
        if len(daily_weather) > 0:
            print(f"天气数据日期范围: {daily_weather.index.min()} 至 {daily_weather.index.max()}")
            print("\n天气数据统计信息:")
            print(daily_weather.describe())
        print(f"使用的天气特征: {available_cols}")
        return daily_weather

    def merge_datasets(self, daily_power, daily_weather):
        """合并电力数据和天气数据"""
        print("合并电力数据和天气数据...")

        # 合并数据集
        merged_data = pd.merge(daily_power, daily_weather, left_index=True, right_index=True, how='inner')

        # 确保数据按时间排序
        merged_data = merged_data.sort_index()

        print(f"数据合并完成，共 {len(merged_data)} 天的有效数据")
        if len(merged_data) > 0:
            print(f"合并后日期范围: {merged_data.index.min()} 至 {merged_data.index.max()}")

            # 打印合并后的统计信息
            print("\n合并后数据统计信息:")
            print(merged_data.describe())

        if len(merged_data) == 0:
            print("警告：合并后数据为空！请检查电力数据和天气数据的日期范围是否有重叠。")
            print(f"电力数据范围: {daily_power.index.min()} 至 {daily_power.index.max()}")
            if len(daily_weather) > 0:
                print(f"天气数据范围: {daily_weather.index.min()} 至 {daily_weather.index.max()}")

        return merged_data

    def add_time_features(self, data):
        """Add calendar features that help the model learn periodic patterns."""
        if not isinstance(data.index, pd.DatetimeIndex):
            return data

        data = data.copy()
        day_of_week = data.index.dayofweek.to_numpy()
        month = data.index.month.to_numpy()
        day_of_year = data.index.dayofyear.to_numpy()

        data['dayofweek_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        data['dayofweek_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        data['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
        data['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
        data['dayofyear_sin'] = np.sin(2 * np.pi * (day_of_year - 1) / 366)
        data['dayofyear_cos'] = np.cos(2 * np.pi * (day_of_year - 1) / 366)

        return data

    def normalize_data(self, data, method='minmax'):
        """
        对数据进行归一化/标准化

        Args:
            data: 要归一化的数据 (numpy array)
            method: 'minmax' 或 'standard'

        Returns:
            normalized_data: 归一化后的数据
        """
        if method == 'minmax':
            # Min-Max 归一化到 [0, 1]
            self.feature_scaler = {
                'method': 'minmax',
                'min': np.min(data, axis=0),
                'max': np.max(data, axis=0),
                'range': np.max(data, axis=0) - np.min(data, axis=0)
            }
            # 防止除零
            range_vals = self.feature_scaler['range'].copy()
            range_vals[range_vals == 0] = 1.0
            normalized = (data - self.feature_scaler['min']) / range_vals

        elif method == 'standard':
            # Z-score 标准化
            self.feature_scaler = {
                'method': 'standard',
                'mean': np.mean(data, axis=0),
                'std': np.std(data, axis=0)
            }
            # 防止除零
            std_vals = self.feature_scaler['std'].copy()
            std_vals[std_vals == 0] = 1.0
            normalized = (data - self.feature_scaler['mean']) / std_vals
        else:
            raise ValueError(f"不支持的归一化方法: {method}")

        return normalized

    def denormalize_data(self, data, scaler=None):
        """
        反归一化数据

        Args:
            data: 归一化后的数据
            scaler: 归一化参数（如果为None则使用self.feature_scaler）

        Returns:
            original_data: 反归一化后的数据
        """
        scaler = scaler or self.feature_scaler
        if scaler is None:
            raise ValueError("没有可用的归一化参数，请先进行归一化")

        if scaler['method'] == 'minmax':
            range_vals = scaler['range'].copy()
            range_vals[range_vals == 0] = 1.0
            return data * range_vals + scaler['min']
        elif scaler['method'] == 'standard':
            std_vals = scaler['std'].copy()
            std_vals[std_vals == 0] = 1.0
            return data * std_vals + scaler['mean']
        else:
            raise ValueError(f"不支持的归一化方法: {scaler['method']}")

    def normalize_target(self, y_data, method='minmax'):
        """
        单独对目标变量进行归一化

        Args:
            y_data: 目标数据 (n_samples, pred_len) 或 (n_samples,)
            method: 'minmax' 或 'standard'

        Returns:
            normalized_y: 归一化后的目标数据
        """
        if method == 'minmax':
            self.target_scaler = {
                'method': 'minmax',
                'min': np.min(y_data),
                'max': np.max(y_data),
                'range': np.max(y_data) - np.min(y_data)
            }
            range_val = self.target_scaler['range']
            if range_val == 0:
                range_val = 1.0
            normalized = (y_data - self.target_scaler['min']) / range_val

        elif method == 'standard':
            self.target_scaler = {
                'method': 'standard',
                'mean': np.mean(y_data),
                'std': np.std(y_data)
            }
            std_val = self.target_scaler['std']
            if std_val == 0:
                std_val = 1.0
            normalized = (y_data - self.target_scaler['mean']) / std_val
        else:
            raise ValueError(f"不支持的归一化方法: {method}")

        return normalized

    def denormalize_target(self, y_data):
        """
        反归一化目标变量

        Args:
            y_data: 归一化后的目标数据

        Returns:
            original_y: 反归一化后的目标数据
        """
        if self.target_scaler is None:
            self._sync_target_scaler_from_features()

        if self.target_scaler is None:
            raise ValueError("没有可用的目标归一化参数")

        if self.target_scaler['method'] == 'minmax':
            range_val = self.target_scaler['range']
            if range_val == 0:
                range_val = 1.0
            return y_data * range_val + self.target_scaler['min']
        elif self.target_scaler['method'] == 'standard':
            std_val = self.target_scaler['std']
            if std_val == 0:
                std_val = 1.0
            return y_data * std_val + self.target_scaler['mean']

    def create_sequences(self, data, sequence_length=90, prediction_length=90, normalize=True, norm_method='minmax'):
        """
        创建时间序列数据

        Args:
            data: 合并后的数据
            sequence_length: 输入序列长度
            prediction_length: 预测序列长度
            normalize: 是否进行归一化
            norm_method: 归一化方法 ('minmax' 或 'standard')

        Returns:
            X: 输入序列 (已归一化)
            y: 目标序列 (已归一化)
            feature_data_norm: 归一化后的完整特征数据
        """
        print(f"创建时间序列数据（序列长度={sequence_length}，预测长度={prediction_length}）...")

        data = self.add_time_features(data)

        # 选择特征列
        feature_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
                       'Sub_metering_3', 'Sub_metering_remainder']

        # 添加天气特征：RR、NBJRR1、NBJRR5、NBJRR10、NBJBROU
        weather_cols = ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
        available_weather = [col for col in weather_cols if col in data.columns]
        feature_cols.extend(available_weather)

        time_cols = ['dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos',
                     'dayofyear_sin', 'dayofyear_cos']
        available_time = [col for col in time_cols if col in data.columns]
        feature_cols.extend(available_time)

        # 只保留实际存在的列
        feature_cols = [col for col in feature_cols if col in data.columns]
        self.feature_cols = feature_cols

        print(f"使用的特征列: {feature_cols}")
        print(f"目标列: {self.target_col}")

        # 提取特征数据
        feature_data = data[feature_cols].values

        # 检查数据
        print(f"\n特征数据统计:")
        for i, col in enumerate(feature_cols):
            col_data = feature_data[:, i]
            print(f"  {col}: min={col_data.min():.4f}, max={col_data.max():.4f}, mean={col_data.mean():.4f}, std={col_data.std():.4f}")

        # 归一化
        if normalize:
            print(f"\n使用 {norm_method} 方法进行归一化...")
            feature_data_norm = self.normalize_data(feature_data, method=norm_method)
            self._sync_target_scaler_from_features()
            print("特征数据归一化完成")
        else:
            feature_data_norm = feature_data
            self.target_scaler = {
                'method': 'minmax',
                'min': 0.0,
                'max': 1.0,
                'range': 1.0
            }
            print("\n跳过归一化（不推荐）")

        # 创建序列
        X, y = [], []

        for i in range(len(feature_data_norm) - sequence_length - prediction_length + 1):
            X.append(feature_data_norm[i:i+sequence_length])
            # 目标变量是 Global_active_power 的原始值（第0列）
            y.append(feature_data_norm[i+sequence_length:i+sequence_length+prediction_length, 0])

        X = np.array(X)
        y = np.array(y)

        print(f"\n序列创建完成，共 {len(X)} 个样本")
        if len(X) > 0:
            print(f"输入形状: {X.shape}, 输出形状: {y.shape}")
            print(f"X 范围: [{X.min():.4f}, {X.max():.4f}]")
            print(f"y 范围: [{y.min():.4f}, {y.max():.4f}]")
        else:
            print("警告：没有创建任何序列样本！数据量可能不足以支持当前的序列长度和预测长度。")

        return X, y, feature_data_norm

    def preprocess(self, sequence_length=90, prediction_length=90, normalize=True, norm_method='minmax'):
        """
        完整的数据预处理流程

        Args:
            sequence_length: 输入序列长度
            prediction_length: 预测序列长度
            normalize: 是否进行归一化
            norm_method: 归一化方法

        Returns:
            X: 输入序列
            y: 目标序列
            data: 处理后的完整数据
        """
        print("\n" + "="*50)
        print("开始数据预处理...")
        print("="*50)

        # 加载数据
        self.load_power_data()
        self.load_weather_data()

        # 聚合和合并数据
        daily_power = self.aggregate_daily_power()
        daily_weather = self.prepare_weather_features()
        merged_data = self.merge_datasets(daily_power, daily_weather)

        if len(merged_data) == 0:
            raise ValueError("数据合并后为空，无法创建序列。请检查数据日期范围。")

        # 创建序列
        X, y, _ = self.create_sequences(merged_data, sequence_length, prediction_length,
                                       normalize=normalize, norm_method=norm_method)

        # 保存处理后的数据
        self.processed_data = merged_data

        print("\n" + "="*50)
        print("数据预处理完成！")
        print("="*50)
        return X, y, merged_data

    def save_scalers(self, filepath):
        """保存归一化参数到文件"""
        scaler_data = {
            'feature_scaler': {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.feature_scaler.items()
            } if self.feature_scaler else None,
            'target_scaler': {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.target_scaler.items()
            } if self.target_scaler else None,
            'feature_cols': self.feature_cols,
            'target_col': self.target_col
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(scaler_data, f, indent=2, ensure_ascii=False)
        print(f"归一化参数已保存到: {filepath}")

    def load_scalers(self, filepath):
        """从文件加载归一化参数"""
        with open(filepath, 'r', encoding='utf-8') as f:
            scaler_data = json.load(f)

        if scaler_data['feature_scaler']:
            self.feature_scaler = {
                k: (np.array(v) if isinstance(v, list) else v)
                for k, v in scaler_data['feature_scaler'].items()
            }
        if scaler_data['target_scaler']:
            self.target_scaler = {
                k: (np.array(v) if isinstance(v, list) else v)
                for k, v in scaler_data['target_scaler'].items()
            }
        self.feature_cols = scaler_data.get('feature_cols')
        self.target_col = scaler_data.get('target_col', 'Global_active_power')
        print(f"归一化参数已从 {filepath} 加载")


def split_data(X, y, train_ratio=0.8):
    """
    分割训练集和测试集

    Args:
        X: 输入序列
        y: 目标序列
        train_ratio: 训练集比例

    Returns:
        X_train, y_train, X_test, y_test
    """
    split_idx = int(len(X) * train_ratio)

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    return X_train, y_train, X_test, y_test


def split_data_with_save(X, y, train_ratio=0.8, save_dir=None, prefix=''):
    """
    分割训练集和测试集，并保存为CSV文件

    Args:
        X: 输入序列 (n_samples, seq_len, n_features)
        y: 目标序列 (n_samples, pred_len)
        train_ratio: 训练集比例
        save_dir: 保存目录（None则不保存）
        prefix: 文件名前缀（如 'short_' 或 'long_'）

    Returns:
        X_train, y_train, X_test, y_test
    """
    import json
    split_idx = int(len(X) * train_ratio)

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    # 保存为CSV
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)

        # 将3D数组转换为2D（每个时间步作为一列）
        n_train, seq_len, n_features = X_train.shape
        n_test = X_test.shape[0]

        # 保存X_train: 展平为 (n_samples, seq_len * n_features)
        X_train_flat = X_train.reshape(n_train, -1)
        X_test_flat = X_test.reshape(n_test, -1)

        # 生成列名
        feature_cols = []
        for t in range(seq_len):
            for f in range(n_features):
                feature_cols.append(f'X_t{t}_f{f}')

        y_cols = [f'y_t{i}' for i in range(y_train.shape[1])]

        # 创建DataFrame并保存
        train_df = pd.DataFrame(X_train_flat, columns=feature_cols)
        train_df[y_cols] = y_train
        train_path = os.path.join(save_dir, f'{prefix}train.csv')
        train_df.to_csv(train_path, index=False)

        test_df = pd.DataFrame(X_test_flat, columns=feature_cols)
        test_df[y_cols] = y_test
        test_path = os.path.join(save_dir, f'{prefix}test.csv')
        test_df.to_csv(test_path, index=False)

        # 保存数据信息
        info = {
            'n_train': int(n_train),
            'n_test': int(n_test),
            'sequence_length': int(seq_len),
            'n_features': int(n_features),
            'prediction_length': int(y_train.shape[1]),
            'train_ratio': float(train_ratio),
            'feature_cols': feature_cols,
            'y_cols': y_cols
        }
        info_path = os.path.join(save_dir, f'{prefix}data_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        print(f"\n  数据已保存:")
        print(f"    - {train_path}: {n_train} 条")
        print(f"    - {test_path}: {n_test} 条")
        print(f"    - {info_path}: 数据信息")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # 数据预处理测试
    preprocessor = DataPreprocessor(
        'household_power_consumption.txt',
        'MENSQ_75_previous-1950-2024.csv'
    )

    # 短期预测数据（预测90天）
    X_short, y_short, data_short = preprocessor.preprocess(sequence_length=90, prediction_length=90)

    # 长期预测数据（预测365天）
    X_long, y_long, data_long = preprocessor.preprocess(sequence_length=90, prediction_length=365)

    # 分割数据
    X_train_short, y_train_short, X_test_short, y_test_short = split_data(X_short, y_short)
    X_train_long, y_train_long, X_test_long, y_test_long = split_data(X_long, y_long)

    print("\n数据预处理完成！")
    print(f"短期预测 - 训练集: {X_train_short.shape}, 测试集: {X_test_short.shape}")
    print(f"长期预测 - 训练集: {X_train_long.shape}, 测试集: {X_test_long.shape}")
