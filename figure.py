import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
from datetime import datetime
import seaborn as sns
from sklearn.metrics import r2_score, explained_variance_score

# 设置中文字体和风格
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
sns.set_style("whitegrid")
sns.set_palette("husl")


class ModelComparator:
    def __init__(self, data_file, stgcn_model_path, gwn_model_path, n_his=12, n_pred=3, batch_size=32):
        """初始化模型比较器"""
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.n_his = n_his
        self.n_pred = n_pred
        self.batch_size = batch_size

        # 验证文件路径是否存在
        self._validate_paths(data_file, stgcn_model_path, gwn_model_path)

        # 创建结果保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"comparison_results_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)

        # 加载数据和模型
        self._load_data(data_file)
        self._load_models(stgcn_model_path, gwn_model_path)

        print(f"模型比较器初始化完成，结果将保存到: {self.result_dir}")
        print(f"使用设备: {self.device}")

    def _validate_paths(self, data_file, stgcn_model_path, gwn_model_path):
        """验证文件路径是否存在"""
        for path, name in [(data_file, "数据文件"), (stgcn_model_path, "STGCN模型文件"),
                           (gwn_model_path, "GWN模型文件")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name}不存在: {path}")

    def _load_data(self, data_file):
        """加载数据"""
        try:
            # 动态导入数据预处理器
            from datasets.WaterDataset import EpanetUnifiedPreprocessor

            preprocessor = EpanetUnifiedPreprocessor(data_file)

            # 获取STGCN数据
            stgcn_data = preprocessor.get_model_specific_data('STGCN', self.n_his, self.n_pred, self.batch_size)
            self.stgcn_test_loader = stgcn_data['test_loader']
            self.num_nodes = stgcn_data['num_nodes']
            self.gso = stgcn_data['gso']

            # 获取GWN数据
            gwn_data = preprocessor.get_model_specific_data('GWN', self.n_his, self.n_pred, self.batch_size)
            self.gwn_test_loader = gwn_data['test_loader']

            # 保存标准化参数
            self.pressure_mean = stgcn_data.get('pressure_mean', 0)
            self.pressure_std = stgcn_data.get('pressure_std', 1)

            print(f"数据加载完成: {self.num_nodes}个节点, 历史步长{self.n_his}, 预测步长{self.n_pred}")

        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            raise

    def _load_models(self, stgcn_model_path, gwn_model_path):
        """加载模型"""
        try:
            # 动态导入模型
            from model.STGCN import STGCNChebGraphConv as STGCN
            from model.GWN import gwnet

            # 加载STGCN模型
            self.stgcn_model = self._load_stgcn_model(stgcn_model_path)

            # 加载GWN模型
            self.gwn_model = self._load_gwn_model(gwn_model_path)

            print("所有模型加载完成")

        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise

    def _load_stgcn_model(self, model_path):
        """加载STGCN模型"""
        from model.STGCN import STGCNChebGraphConv

        # 创建模型结构（假设blocks和args是按照项目中的标准设置）
        blocks = [[1], [64, 16, 64], [64, 32]]
        args = type('Args', (), {'Kt': 3, 'stblock_num': 2, 'Ko': self.n_his - 2 * (3 - 1)})

        model = STGCNChebGraphConv(args, blocks, self.num_nodes).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def _load_gwn_model(self, model_path):
        """加载GWN模型"""
        from model.GWN import gwnet

        model = gwnet(
            device=self.device,
            num_nodes=self.num_nodes,
            in_dim=1,
            out_dim=self.n_pred,
            supports=None,
            gcn_bool=True,
            addaptadj=True,
            dropout=0.3
        ).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def evaluate_model(self, model, test_loader, model_name="未知模型", is_gwn=False):
        """评估模型性能"""
        print(f"开始评估{model_name}模型...")

        model.eval()
        all_preds = []
        all_targets = []
        total_time = 0

        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(self.device).float(), y.to(self.device).float()

                # 测量推理时间
                start_time = time.time()
                output = model(x)
                end_time = time.time()
                total_time += (end_time - start_time)

                # 调整GWN输出维度
                if is_gwn:
                    # 确保输出维度符合[batch, channels, time, nodes]格式
                    if output.dim() == 4:
                        output = output.permute(0, 3, 1, 2)  # 从[B, N, T, C]到[B, C, T, N]

                # 保存预测值和真实值
                all_preds.append(output.cpu().numpy())
                all_targets.append(y.cpu().numpy())

                # 打印进度
                if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                    print(f"  {model_name}进度: {i + 1}/{len(test_loader)}")

        # 合并结果
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # 反归一化
        all_preds = all_preds * self.pressure_std + self.pressure_mean
        all_targets = all_targets * self.pressure_std + self.pressure_mean

        # 计算评估指标
        mse = np.mean((all_preds - all_targets) ** 2)
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(mse)

        # 计算R²评分
        try:
            # 展平数组以适应sklearn的r2_score
            y_true_flat = all_targets.reshape(-1)
            y_pred_flat = all_preds.reshape(-1)
            r2 = r2_score(y_true_flat, y_pred_flat)
            explained_var = explained_variance_score(y_true_flat, y_pred_flat)
        except Exception as e:
            print(f"计算高级指标时出错: {str(e)}")
            r2 = 1 - (np.sum((all_targets - all_preds) ** 2) / (
                        np.sum((all_targets - np.mean(all_targets)) ** 2) + 1e-8))
            explained_var = 1 - (np.var(all_targets - all_preds) / (np.var(all_targets) + 1e-8))

        # 计算平均推理时间
        num_samples = len(test_loader.dataset)
        avg_inference_time = (total_time / num_samples) * 1000  # 转换为毫秒

        results = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'explained_variance': explained_var,
            'avg_inference_time': avg_inference_time,
            'predictions': all_preds,
            'targets': all_targets,
            'inference_time_total': total_time,
            'num_samples': num_samples
        }

        print(f"{model_name}模型评估完成:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²: {r2:.6f}")
        print(f"  可解释方差: {explained_var:.6f}")
        print(f"  平均推理时间: {avg_inference_time:.4f}毫秒")

        return results

    def compare_models(self):
        """对比两个模型的性能"""
        print("=" * 70)
        print("开始模型性能对比")
        print("=" * 70)

        # 评估STGCN模型
        stgcn_results = self.evaluate_model(self.stgcn_model, self.stgcn_test_loader, "STGCN")

        # 评估GWN模型
        gwn_results = self.evaluate_model(self.gwn_model, self.gwn_test_loader, "GWN", is_gwn=True)

        # 打印对比结果
        self._print_comparison_results(stgcn_results, gwn_results)

        # 保存结果到文件
        self._save_results_to_file(stgcn_results, gwn_results)

        # 可视化对比结果
        self._visualize_comparison(stgcn_results, gwn_results)

        return stgcn_results, gwn_results

    def _print_comparison_results(self, stgcn_results, gwn_results):
        """打印对比结果"""
        print("=" * 70)
        print("模型性能对比")
        print("=" * 70)
        print(f"{'指标':<20}{'STGCN':<15}{'GWN':<15}{'更好的模型'}")
        print("=" * 70)

        # 对于误差指标，值越小越好
        metrics = [
            ('MSE', 'mse', False),
            ('MAE', 'mae', False),
            ('RMSE', 'rmse', False),
            ('R² 评分', 'r2', True),
            ('可解释方差', 'explained_variance', True),
            ('推理时间(ms)', 'avg_inference_time', False)
        ]

        for name, key, higher_better in metrics:
            stgcn_val = stgcn_results[key]
            gwn_val = gwn_results[key]

            if higher_better:
                better_model = 'STGCN' if stgcn_val > gwn_val else 'GWN'
            else:
                better_model = 'STGCN' if stgcn_val < gwn_val else 'GWN'

            print(f"{name:<20}{stgcn_val:<15.6f}{gwn_val:<15.6f}{better_model}")
        print("=" * 70)

    def _save_results_to_file(self, stgcn_results, gwn_results):
        """保存结果到文件"""
        result_file = os.path.join(self.result_dir, "comparison_results.txt")

        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("模型性能对比结果\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"节点数: {self.num_nodes}\n")
            f.write(f"历史步长: {self.n_his}\n")
            f.write(f"预测步长: {self.n_pred}\n")
            f.write(f"使用设备: {self.device}\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'指标':<20}{'STGCN':<15}{'GWN':<15}\n")
            f.write("=" * 60 + "\n")

            metrics = [
                ('MSE', 'mse'),
                ('MAE', 'mae'),
                ('RMSE', 'rmse'),
                ('R² 评分', 'r2'),
                ('可解释方差', 'explained_variance'),
                ('推理时间(ms)', 'avg_inference_time'),
                ('总推理时间(s)', 'inference_time_total'),
                ('样本数量', 'num_samples')
            ]

            for name, key in metrics:
                f.write(f"{name:<20}{stgcn_results.get(key, 0):<15.6f}{gwn_results.get(key, 0):<15.6f}\n")

        print(f"对比结果已保存到: {result_file}")

    def _visualize_comparison(self, stgcn_results, gwn_results):
        """可视化对比结果"""
        print("开始生成可视化图表...")

        # 1. 指标对比柱状图
        self._plot_metric_comparison(stgcn_results, gwn_results)

        # 2. 预测值与真实值对比图（多个节点）
        self._plot_prediction_comparison(stgcn_results, gwn_results)

        # 3. 误差分布直方图
        self._plot_error_distribution(stgcn_results, gwn_results)

        # 4. 散点图：预测值 vs 真实值
        self._plot_scatter_comparison(stgcn_results, gwn_results)

        # 5. 时间序列对比图
        self._plot_time_series_comparison(stgcn_results, gwn_results)

        print(f"所有对比图表已保存到 {self.result_dir} 目录中")

    def _plot_metric_comparison(self, stgcn_results, gwn_results):
        """绘制指标对比柱状图"""
        metrics = ['mse', 'mae', 'rmse', 'r2', 'explained_variance']
        metric_names = ['MSE', 'MAE', 'RMSE', 'R²', '可解释方差']

        stgcn_values = [stgcn_results[m] for m in metrics]
        gwn_values = [gwn_results[m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        # 创建两个子图，分别展示误差指标和性能指标
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 误差指标 (MSE, MAE, RMSE)
        rects1 = axes[0].bar(x[:3] - width / 2, stgcn_values[:3], width, label='STGCN')
        rects2 = axes[0].bar(x[:3] + width / 2, gwn_values[:3], width, label='GWN')
        axes[0].set_ylabel('误差值')
        axes[0].set_title('模型误差指标对比')
        axes[0].set_xticks(x[:3])
        axes[0].set_xticklabels(metric_names[:3])
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # 为误差指标添加数值标签
        for rects in [rects1, rects2]:
            for rect in rects[:3]:
                height = rect.get_height()
                axes[0].annotate('%.4f' % height,
                                 xy=(rect.get_x() + rect.get_width() / 2, height),
                                 xytext=(0, 3),  # 3 points vertical offset
                                 textcoords="offset points",
                                 ha='center', va='bottom')

        # 性能指标 (R², 可解释方差)
        rects1 = axes[1].bar(x[3:] - width / 2, stgcn_values[3:], width, label='STGCN')
        rects2 = axes[1].bar(x[3:] + width / 2, gwn_values[3:], width, label='GWN')
        axes[1].set_ylabel('性能值')
        axes[1].set_title('模型性能指标对比')
        axes[1].set_xticks(x[3:])
        axes[1].set_xticklabels(metric_names[3:])
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.7)

        # 为性能指标添加数值标签
        for rects in [rects1, rects2]:
            for rect in rects:
                height = rect.get_height()
                axes[1].annotate('%.4f' % height,
                                 xy=(rect.get_x() + rect.get_width() / 2, height),
                                 xytext=(0, 3),  # 3 points vertical offset
                                 textcoords="offset points",
                                 ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'metric_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_prediction_comparison(self, stgcn_results, gwn_results):
        """绘制预测值与真实值对比图"""
        # 随机选择3个节点进行可视化
        num_nodes = stgcn_results['predictions'].shape[-1]
        selected_nodes = np.random.choice(num_nodes, min(3, num_nodes), replace=False)

        # 选择第一个样本进行可视化
        sample_idx = 0

        # 获取数据形状信息
        stgcn_shape = stgcn_results['predictions'].shape
        gwn_shape = gwn_results['predictions'].shape

        fig, axes = plt.subplots(len(selected_nodes), 1, figsize=(12, 5 * len(selected_nodes)))
        if len(selected_nodes) == 1:
            axes = [axes]

        for i, node_idx in enumerate(selected_nodes):
            # 获取真实值（根据实际形状调整索引）
            if len(stgcn_shape) == 4:
                # 假设形状为 [B, C, T, N]
                true_values = stgcn_results['targets'][sample_idx, 0, :, node_idx]
                stgcn_preds = stgcn_results['predictions'][sample_idx, 0, :, node_idx]
                # 确保GWN的维度匹配
                if len(gwn_shape) == 4:
                    gwn_preds = gwn_results['predictions'][sample_idx, 0, :, node_idx]
                else:
                    gwn_preds = gwn_results['predictions'][sample_idx, :, node_idx]
            else:
                # 处理其他可能的形状
                true_values = stgcn_results['targets'][sample_idx, :, node_idx]
                stgcn_preds = stgcn_results['predictions'][sample_idx, :, node_idx]
                gwn_preds = gwn_results['predictions'][sample_idx, :, node_idx]

            # 创建时间点
            time_steps = np.arange(len(true_values))

            # 绘制图表
            axes[i].plot(time_steps, true_values, 'b-', label='真实值', linewidth=2)
            axes[i].plot(time_steps, stgcn_preds, 'g--', label='STGCN预测', linewidth=2)
            axes[i].plot(time_steps, gwn_preds, 'r-.', label='GWN预测', linewidth=2)
            axes[i].set_title(f'节点 {node_idx} 的预测值与真实值对比', fontsize=14)
            axes[i].set_xlabel('时间步', fontsize=12)
            axes[i].set_ylabel('压力值', fontsize=12)
            axes[i].legend(fontsize=12)
            axes[i].grid(True, linestyle='--', alpha=0.7)

            # 添加误差信息
            stgcn_mae = np.mean(np.abs(stgcn_preds - true_values))
            gwn_mae = np.mean(np.abs(gwn_preds - true_values))
            axes[i].text(0.02, 0.95, f'STGCN MAE: {stgcn_mae:.4f}\nGWN MAE: {gwn_mae:.4f}',
                         transform=axes[i].transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'prediction_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_error_distribution(self, stgcn_results, gwn_results):
        """绘制误差分布直方图"""
        # 计算绝对误差
        stgcn_errors = np.abs(stgcn_results['predictions'] - stgcn_results['targets']).flatten()
        gwn_errors = np.abs(gwn_results['predictions'] - gwn_results['targets']).flatten()

        # 限制误差范围，移除异常值
        max_error = np.percentile(np.concatenate([stgcn_errors, gwn_errors]), 95)
        stgcn_errors = stgcn_errors[stgcn_errors <= max_error]
        gwn_errors = gwn_errors[gwn_errors <= max_error]

        # 创建直方图
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.hist(stgcn_errors, bins=50, alpha=0.6, label='STGCN误差', color='green', edgecolor='black')
        ax.hist(gwn_errors, bins=50, alpha=0.6, label='GWN误差', color='red', edgecolor='black')

        ax.set_xlabel('绝对误差', fontsize=12)
        ax.set_ylabel('频率', fontsize=12)
        ax.set_title('模型预测误差分布对比', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        # 添加统计信息
        stgcn_mean = np.mean(stgcn_errors)
        gwn_mean = np.mean(gwn_errors)
        stgcn_std = np.std(stgcn_errors)
        gwn_std = np.std(gwn_errors)

        stats_text = (f'STGCN: 均值={stgcn_mean:.4f}, 标准差={stgcn_std:.4f}\n'
                      f'GWN: 均值={gwn_mean:.4f}, 标准差={gwn_std:.4f}')

        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_scatter_comparison(self, stgcn_results, gwn_results):
        """绘制预测值 vs 真实值的散点图"""
        # 展平数据
        stgcn_preds_flat = stgcn_results['predictions'].flatten()
        gwn_preds_flat = gwn_results['predictions'].flatten()
        targets_flat = stgcn_results['targets'].flatten()  # 两个模型的targets应该是相同的

        # 限制数据范围以提高可视化效果
        max_val = np.percentile(targets_flat, 99)
        min_val = np.percentile(targets_flat, 1)

        mask_stgcn = (stgcn_preds_flat >= min_val) & (stgcn_preds_flat <= max_val) & \
                     (targets_flat >= min_val) & (targets_flat <= max_val)
        mask_gwn = (gwn_preds_flat >= min_val) & (gwn_preds_flat <= max_val) & \
                   (targets_flat >= min_val) & (targets_flat <= max_val)

        # 创建散点图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # STGCN散点图
        axes[0].scatter(targets_flat[mask_stgcn], stgcn_preds_flat[mask_stgcn], alpha=0.3, s=10, color='green')
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2)  # 理想线
        axes[0].set_xlabel('真实值', fontsize=12)
        axes[0].set_ylabel('STGCN预测值', fontsize=12)
        axes[0].set_title('STGCN: 预测值 vs 真实值', fontsize=14)
        axes[0].grid(True, linestyle='--', alpha=0.7)
        axes[0].set_aspect('equal')

        # GWN散点图
        axes[1].scatter(targets_flat[mask_gwn], gwn_preds_flat[mask_gwn], alpha=0.3, s=10, color='red')
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2)  # 理想线
        axes[1].set_xlabel('真实值', fontsize=12)
        axes[1].set_ylabel('GWN预测值', fontsize=12)
        axes[1].set_title('GWN: 预测值 vs 真实值', fontsize=14)
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].set_aspect('equal')

        # 添加R²评分
        stgcn_r2 = r2_score(targets_flat, stgcn_preds_flat)
        gwn_r2 = r2_score(targets_flat, gwn_preds_flat)

        axes[0].text(0.05, 0.95, f'R² = {stgcn_r2:.4f}', transform=axes[0].transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].text(0.05, 0.95, f'R² = {gwn_r2:.4f}', transform=axes[1].transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'scatter_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_time_series_comparison(self, stgcn_results, gwn_results):
        """绘制多步预测时间序列对比图"""
        # 选择一个节点进行可视化
        num_nodes = stgcn_results['predictions'].shape[-1]
        node_idx = np.random.choice(num_nodes)

        # 获取第一个样本的预测结果
        sample_idx = 0

        # 获取真实值和预测值
        if len(stgcn_results['targets'].shape) == 4:
            # 假设形状为 [B, C, T, N]
            true_values = stgcn_results['targets'][sample_idx, 0, :, node_idx]
            stgcn_preds = stgcn_results['predictions'][sample_idx, 0, :, node_idx]
            # 确保GWN的维度匹配
            if len(gwn_results['predictions'].shape) == 4:
                gwn_preds = gwn_results['predictions'][sample_idx, 0, :, node_idx]
            else:
                gwn_preds = gwn_results['predictions'][sample_idx, :, node_idx]
        else:
            # 处理其他可能的形状
            true_values = stgcn_results['targets'][sample_idx, :, node_idx]
            stgcn_preds = stgcn_results['predictions'][sample_idx, :, node_idx]
            gwn_preds = gwn_results['predictions'][sample_idx, :, node_idx]

        # 创建图表
        fig, ax = plt.subplots(figsize=(14, 6))

        # 为每个预测步长绘制条形图
        bar_width = 0.25
        x = np.arange(len(true_values))

        ax.bar(x - bar_width, true_values, bar_width, label='真实值', color='blue', alpha=0.7)
        ax.bar(x, stgcn_preds, bar_width, label='STGCN预测', color='green', alpha=0.7)
        ax.bar(x + bar_width, gwn_preds, bar_width, label='GWN预测', color='red', alpha=0.7)

        ax.set_xlabel('预测时间步', fontsize=12)
        ax.set_ylabel('压力值', fontsize=12)
        ax.set_title(f'节点 {node_idx} 的多步预测对比', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'步长 {i + 1}' for i in range(len(true_values))])
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')

        # 添加每个时间步的误差
        for i in range(len(true_values)):
            stgcn_error = abs(stgcn_preds[i] - true_values[i])
            gwn_error = abs(gwn_preds[i] - true_values[i])

            max_height = max(true_values[i], stgcn_preds[i], gwn_preds[i])
            ax.text(i, max_height * 1.05, f'STGCN: {stgcn_error:.3f}', ha='center', fontsize=8)
            ax.text(i + bar_width, max_height * 1.05, f'GWN: {gwn_error:.3f}', ha='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'time_series_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='比较STGCN和GWN模型的性能')
    parser.add_argument('--data_file', type=str, default='./data/d-town.inp',
                        help='EPANET输入文件路径')
    parser.add_argument('--stgcn_model_path', type=str, default='./models/stgcn_best.pth',
                        help='STGCN模型文件路径')
    parser.add_argument('--gwn_model_path', type=str, default='./models/gwn_best.pth',
                        help='GWN模型文件路径')
    parser.add_argument('--n_his', type=int, default=12,
                        help='历史时间步长')
    parser.add_argument('--n_pred', type=int, default=3,
                        help='预测时间步长')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    try:
        # 创建对比器并运行对比
        print("正在初始化模型比较器...")
        comparator = ModelComparator(
            data_file=args.data_file,
            stgcn_model_path=args.stgcn_model_path,
            gwn_model_path=args.gwn_model_path,
            n_his=args.n_his,
            n_pred=args.n_pred,
            batch_size=args.batch_size
        )

        # 执行对比
        print("\n开始执行模型对比...")
        stgcn_results, gwn_results = comparator.compare_models()

        print("\n模型对比完成！")

    except Exception as e:
        print(f"执行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()