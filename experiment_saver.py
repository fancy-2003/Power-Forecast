"""
实验结果保存工具模块
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import hashlib

class ExperimentSaver:
    """实验结果保存器"""

    def __init__(self, base_dir='experiment_results', experiment_name=None, resume=True):
        """
        初始化保存器

        Args:
            base_dir: 结果保存的根目录
            experiment_name: 实验名称（默认使用时间戳）
            resume: 是否启用断点续跑（True=查找已有实验继续，False=创建新实验）
        """
        self.base_dir = base_dir
        self.resume = resume
        self.experiment_name = experiment_name or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = os.path.join(base_dir, self.experiment_name)

        # 创建子目录
        self.dirs = {
            'models': os.path.join(self.exp_dir, 'models'),
            'figures': os.path.join(self.exp_dir, 'figures'),
            'data': os.path.join(self.exp_dir, 'data'),
            'logs': os.path.join(self.exp_dir, 'logs'),
            'reports': os.path.join(self.exp_dir, 'reports'),
            'checkpoints': os.path.join(self.exp_dir, 'checkpoints')
        }

        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        # 检查点文件路径
        self.checkpoint_file = os.path.join(self.dirs['checkpoints'], 'experiment_checkpoint.json')

        # 加载已有检查点（如果 resume=True）
        self.completed_experiments = self._load_checkpoint()

        print(f"实验结果将保存到: {self.exp_dir}")
        if self.completed_experiments:
            print(f"已完成的实验: {list(self.completed_experiments.keys())}")

    def _load_checkpoint(self):
        """加载检查点，获取已完成的实验列表"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_checkpoint(self):
        """保存检查点"""
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(self.completed_experiments, f, indent=2, ensure_ascii=False, default=str)

    def is_experiment_completed(self, model_name, config_name, experiment_id):
        """
        检查某个实验是否已经完成

        Args:
            model_name: 模型名称（如 'LSTM', 'Transformer'）
            config_name: 配置名称（如 'seq90_pred90'）
            experiment_id: 实验编号（如 'exp_1'）

        Returns:
            bool: 是否已完成
        """
        key = f"{model_name}_{config_name}_{experiment_id}"
        return key in self.completed_experiments

    def mark_experiment_completed(self, model_name, config_name, experiment_id, results=None):
        """
        标记实验为已完成

        Args:
            model_name: 模型名称
            config_name: 配置名称
            experiment_id: 实验编号
            results: 实验结果（可选）
        """
        key = f"{model_name}_{config_name}_{experiment_id}"
        self.completed_experiments[key] = {
            'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': results
        }
        self._save_checkpoint()

    def save_figure(self, fig, name, subdir=''):
        """
        保存图表

        Args:
            fig: matplotlib figure 对象
            name: 文件名（不含扩展名）
            subdir: 子目录
        """
        save_dir = os.path.join(self.dirs['figures'], subdir) if subdir else self.dirs['figures']
        os.makedirs(save_dir, exist_ok=True)

        png_path = os.path.join(save_dir, f'{name}.png')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"  [图表保存] {png_path}")
        return png_path

    def save_model(self, model, name, framework='keras'):
        """
        保存模型

        Args:
            model: 模型对象
            name: 模型名称
            framework: 框架类型 ('keras', 'pytorch', 'sklearn')
        """
        save_dir = os.path.join(self.dirs['models'], name)
        os.makedirs(save_dir, exist_ok=True)

        if framework == 'keras':
            model_path = os.path.join(save_dir, 'model.h5')
            model.save(model_path)
        elif framework == 'pytorch':
            import torch
            model_path = os.path.join(save_dir, 'model.pt')
            torch.save(model.state_dict(), model_path)
        elif framework == 'sklearn':
            model_path = os.path.join(save_dir, 'model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        print(f"  [模型保存] {model_path}")
        return model_path

    def save_data(self, data, name, format='auto'):
        """
        保存数据

        Args:
            data: 数据对象
            name: 文件名（不含扩展名）
            format: 保存格式 ('csv', 'npy', 'json', 'pkl', 'auto')
        """
        save_dir = self.dirs['data']

        if format == 'auto':
            if isinstance(data, pd.DataFrame):
                format = 'csv'
            elif isinstance(data, np.ndarray):
                format = 'npy'
            elif isinstance(data, dict):
                format = 'json'
            else:
                format = 'pkl'

        if format == 'csv':
            path = os.path.join(save_dir, f'{name}.csv')
            data.to_csv(path, index=True)
        elif format == 'npy':
            path = os.path.join(save_dir, f'{name}.npy')
            np.save(path, data)
        elif format == 'json':
            path = os.path.join(save_dir, f'{name}.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        elif format == 'pkl':
            path = os.path.join(save_dir, f'{name}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(data, f)

        print(f"  [数据保存] {path}")
        return path

    def save_results(self, results, model_name, config_name):
        """
        保存实验结果

        Args:
            results: 结果字典
            model_name: 模型名称
            config_name: 配置名称
        """
        json_path = os.path.join(self.dirs['data'], f'{model_name}_{config_name}_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        if 'individual_results' in results:
            df = pd.DataFrame(results['individual_results'])
            csv_path = os.path.join(self.dirs['data'], f'{model_name}_{config_name}_results.csv')
            df.to_csv(csv_path, index=False)

        print(f"  [结果保存] {json_path}")
        return json_path

    def save_log(self, content, name='experiment_log'):
        """保存日志"""
        path = os.path.join(self.dirs['logs'], f'{name}.txt')
        with open(path, 'w', encoding='utf-8') as f:
            if isinstance(content, list):
                f.write('\n'.join(content))
            else:
                f.write(str(content))
        return path

    def save_summary_report(self, all_results):
        """
        生成并保存实验总结报告

        Args:
            all_results: 所有实验结果的字典
        """
        report = []
        report.append("=" * 80)
        report.append("电力消耗预测实验报告")
        report.append(f"实验时间: {self.experiment_name}")
        report.append("=" * 80)
        report.append("")

        for model_name, model_results in all_results.items():
            report.append(f"\n{'='*80}")
            report.append(f"模型: {model_name}")
            report.append(f"{'='*80}")

            for config_name, results in model_results.items():
                report.append(f"\n配置: {config_name}")
                report.append("-" * 40)

                if 'mse_mean' in results:
                    report.append(f"MSE: {results['mse_mean']:.6f} ± {results.get('mse_std', 0):.6f}")
                    report.append(f"MAE: {results['mae_mean']:.6f} ± {results.get('mae_std', 0):.6f}")
                    report.append(f"RMSE: {results['rmse_mean']:.6f} ± {results.get('rmse_std', 0):.6f}")

                if 'training_time' in results:
                    report.append(f"训练时间: {results['training_time']:.2f} 秒")

        report_text = '\n'.join(report)

        txt_path = os.path.join(self.dirs['reports'], 'experiment_summary.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        md_path = os.path.join(self.dirs['reports'], 'experiment_summary.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\n总结报告已保存:")
        print(f"  - {txt_path}")
        print(f"  - {md_path}")

        return txt_path

    def get_summary(self):
        """获取保存摘要"""
        summary = {
            'experiment_dir': self.exp_dir,
            'experiment_name': self.experiment_name,
            'completed_experiments': list(self.completed_experiments.keys()),
            'saved_files': {}
        }

        for dir_name, dir_path in self.dirs.items():
            if os.path.exists(dir_path):
                files = os.listdir(dir_path)
                summary['saved_files'][dir_name] = files

        return summary


# 全局保存器实例
_saver = None

def get_saver(base_dir='experiment_results', experiment_name=None, resume=True):
    """获取全局保存器实例"""
    global _saver
    if _saver is None:
        _saver = ExperimentSaver(base_dir, experiment_name, resume)
    return _saver


def reset_saver(base_dir='experiment_results', experiment_name=None):
    """重置保存器（创建新的实验目录）"""
    global _saver
    _saver = ExperimentSaver(base_dir, experiment_name, resume=False)
    return _saver


if __name__ == "__main__":
    saver = ExperimentSaver()
    print(f"\n保存器初始化完成")
    print(f"实验目录: {saver.exp_dir}")