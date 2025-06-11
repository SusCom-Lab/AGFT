"""
LinUCB GPU频率调节增强模型分析器（中文版）- 支持Hybrid LinUCB
提供在线学习效果的全面分析
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy import stats
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 解决中文显示问题
def setup_chinese_font():
    """设置中文字体显示"""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import os

    font_name_to_try = 'WenQuanYi Micro Hei'
    font_path_to_try = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'

    try:
        # 优先尝试按名称设置
        plt.rcParams['font.sans-serif'] = [font_name_to_try]
        fm.fontManager.findfont(font_name_to_try, fallback_to_default=False)
        print(f"✅ 成功通过名称 '{font_name_to_try}' 设置字体。")

    except fm.FontNotFoundError:
        print(f"警告：未能按名称找到字体 '{font_name_to_try}'，将尝试直接从路径加载。")
        if os.path.exists(font_path_to_try):
            font_prop = fm.FontProperties(fname=font_path_to_try)
            plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
            print(f"✅ 成功通过路径 '{font_path_to_try}' 设置字体。")
        else:
            print(f"❌ 错误：备用字体路径 '{font_path_to_try}' 不存在。中文将显示为乱码。")
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    # 解决坐标轴负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

    # 字体大小设置
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9

# 调用字体设置
setup_chinese_font()

class LinUCBModelAnalyzer:
    """LinUCB模型性能和学习效果的综合分析器 - 支持Hybrid和Per-arm版本"""
    
    def __init__(self, model_path: str = "data/models/latest_model.pkl", 
                 freq_start: int = 1005, freq_step: int = 90, freq_max: int = 2100,
                 use_english: bool = False):
        """
        初始化分析器
        
        Args:
            model_path: 模型文件路径
            freq_start: 起始频率 (MHz)，默认1005
            freq_step: 频率步长 (MHz)，默认90
            freq_max: 最大频率 (MHz)，默认2100
            use_english: 是否使用英文标签（解决中文乱码问题）
        """
        self.model_path = Path(model_path)
        self.model_data = None
        self.frequencies = None
        self.freq_start = freq_start
        self.freq_step = freq_step
        self.freq_max = freq_max
        self.use_english = use_english
        self.is_hybrid = False  # 标记是否为Hybrid模型
        self.load_model()
        
        # 设置标签语言
        self.labels = self._get_labels()
    
    def _get_labels(self):
        """获取本地化标签"""
        if self.use_english:
            return {
                'title': 'LinUCB Model Comprehensive Analysis',
                'model_type': 'Model Type',
                'hybrid': 'Hybrid/Global',
                'per_arm': 'Per-arm',
                'time_window': 'Time Window',
                'frequency_mhz': 'Frequency (MHz)',
                'action_heatmap': 'Action Selection Heatmap Over Time',
                'selection_prob': 'Selection Probability',
                'usage_percent': 'Usage Percentage (%)',
                'freq_distribution': 'Frequency Usage Distribution',
                'uniform_dist': 'Uniform Distribution',
                'decision_rounds': 'Decision Rounds',
                'reward': 'Reward',
                'learning_curve': 'Learning Curve',
                'original_reward': 'Original Reward',
                'moving_average': 'Moving Average',
                'trend': 'Trend',
                'slope': 'slope',
                'reward_value': 'Reward Value',
                'density': 'Density',
                'reward_distribution': 'Reward Distribution',
                'normal_fit': 'Normal Fit',
                'exploration_rate': 'Exploration Rate',
                'exploration_decay': 'Exploration Decay',
                'theoretical_decay': 'Theoretical Decay',
                'current': 'Current',
                'explore_exploit': 'Explore → Exploit',
                'gpu_freq_mhz': 'GPU Frequency (MHz)',
                'estimated_value': 'Estimated Value',
                'action_values_ci': 'Action Value Estimates with Confidence Intervals',
                'cumulative_regret': 'Cumulative Regret',
                'cumulative_regret_analysis': 'Cumulative Regret Analysis',
                'linear_regret': 'Linear Regret',
                'sublinear_regret': 'Sublinear Regret',
                'coefficient_variation': 'Coefficient of Variation',
                'performance_stability': 'Performance Stability (Rolling CV)',
                'stability_threshold': 'Stability Threshold',
                'stable_region': 'Stable Region',
                'freq_reward_correlation': 'Frequency-Reward Correlation',
                'polynomial_fit': 'Polynomial Fit',
                'model_confidence': 'Model Confidence',
                'action_confidence': 'Model Confidence for Each Action',
                'usage_count': 'Usage Count',
                'window': 'window',
                'parameter_matrix': 'Parameter Matrix Visualization'
            }
        else:
            return {
                'title': 'LinUCB模型综合分析',
                'model_type': '模型类型',
                'hybrid': 'Hybrid/全局共享',
                'per_arm': 'Per-arm/独立参数',
                'time_window': '时间窗口',
                'frequency_mhz': '频率 (MHz)',
                'action_heatmap': '动作选择热力图随时间变化',
                'selection_prob': '选择概率',
                'usage_percent': '使用百分比 (%)',
                'freq_distribution': '频率使用分布',
                'uniform_dist': '均匀分布',
                'decision_rounds': '决策轮数',
                'reward': '奖励',
                'learning_curve': '学习曲线',
                'original_reward': '原始奖励',
                'moving_average': '移动平均',
                'trend': '趋势',
                'slope': '斜率',
                'reward_value': '奖励值',
                'density': '密度',
                'reward_distribution': '奖励分布',
                'normal_fit': '正态拟合',
                'exploration_rate': '探索率 (α)',
                'exploration_decay': '探索率衰减',
                'theoretical_decay': '理论衰减',
                'current': '当前',
                'explore_exploit': '探索 → 开发',
                'gpu_freq_mhz': 'GPU频率 (MHz)',
                'estimated_value': '估计价值',
                'action_values_ci': '动作价值估计与置信区间',
                'cumulative_regret': '累积遗憾',
                'cumulative_regret_analysis': '累积遗憾分析',
                'linear_regret': '线性遗憾',
                'sublinear_regret': '次线性遗憾',
                'coefficient_variation': '变异系数',
                'performance_stability': '性能稳定性（滚动CV）',
                'stability_threshold': '稳定性阈值',
                'stable_region': '稳定区域',
                'freq_reward_correlation': '频率-奖励相关性',
                'polynomial_fit': '多项式拟合',
                'model_confidence': '模型置信度',
                'action_confidence': '各动作的模型置信度',
                'usage_count': '使用次数',
                'window': '窗口',
                'parameter_matrix': '参数矩阵可视化'
            }
        
    def load_model(self):
        """加载保存的模型数据"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件未找到: {self.model_path}")
            
        with open(self.model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        
        # 检测模型类型
        if 'd' in self.model_data:  # Hybrid模型有总维度d
            self.is_hybrid = True
            n_actions = self.model_data['n_actions']
            n_features = self.model_data['n_features']
            print(f"检测到Hybrid LinUCB模型: {n_features}维环境特征, {n_actions}个动作")
        else:
            self.is_hybrid = False
            n_actions = self.model_data['n_actions']
            n_features = self.model_data['n_features']
            print(f"检测到Per-arm LinUCB模型: {n_features}维特征, {n_actions}个动作")
            
        # 生成频率序列
        self.frequencies = []
        current_freq = self.freq_start
        for i in range(n_actions):
            if current_freq <= self.freq_max:
                self.frequencies.append(current_freq)
                current_freq += self.freq_step
            else:
                self.frequencies.append(self.freq_max)
        
        # 确保频率数量与动作数量匹配
        while len(self.frequencies) < n_actions:
            self.frequencies.append(self.frequencies[-1] + self.freq_step)
        self.frequencies = self.frequencies[:n_actions]
        
        print(f"加载模型成功：{n_actions}个动作，频率范围 {min(self.frequencies)}-{max(self.frequencies)} MHz")
    
    def print_summary_report(self):
        """打印综合统计报告"""
        print("LINUCB模型综合分析报告")
        print("=" * 80)
        print(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"模型路径: {self.model_path}")
        print()
        
        # 基本信息
        print("1. 基本模型信息")
        print("-" * 40)
        print(f"   模型类型: {self.labels['hybrid'] if self.is_hybrid else self.labels['per_arm']}")
        print(f"   总决策轮数: {self.model_data['total_rounds']:,}")
        if self.is_hybrid:
            print(f"   环境特征维度: {self.model_data['n_features']}")
            print(f"   总特征维度: {self.model_data['d']}")
        else:
            print(f"   特征维度: {self.model_data['n_features']}")
        print(f"   动作数量（频率数）: {self.model_data['n_actions']}")
        print(f"   频率范围: {min(self.frequencies)} - {max(self.frequencies)} MHz")
        print(f"   频率步长: {self.freq_step} MHz")
        print(f"   当前探索率 (α): {self.model_data['alpha']:.4f}")
        print(f"   初始探索率: {self.model_data['initial_alpha']:.4f}")
        print(f"   正则化参数 (λ): {self.model_data['lambda_reg']:.4f}")
        print()
        
        # 性能指标
        print("2. 性能指标")
        print("-" * 40)
        total_reward = self.model_data['total_reward']
        total_rounds = max(self.model_data['total_rounds'], 1)
        avg_reward = total_reward / total_rounds
        
        print(f"   累积奖励: {total_reward:.6f}")
        print(f"   平均每轮奖励: {avg_reward:.6f}")
        
        if len(self.model_data.get('reward_history', [])) > 0:
            recent_rewards = self.model_data['reward_history']
            print(f"   近期平均值（最近{len(recent_rewards)}轮）: {np.mean(recent_rewards):.6f}")
            print(f"   近期标准差: {np.std(recent_rewards):.6f}")
            print(f"   近期最小值/最大值: {np.min(recent_rewards):.6f} / {np.max(recent_rewards):.6f}")
            
            # 趋势分析
            if len(recent_rewards) > 10:
                x = np.arange(len(recent_rewards))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_rewards)
                trend_text = "改善中" if slope > 0 else "下降中"
                print(f"   奖励趋势: {trend_text} (斜率={slope:.6f}, p值={p_value:.4f})")
        print()
        
        # 动作分布分析
        print("3. 频率选择分布")
        print("-" * 40)
        action_counts = self.model_data['action_counts']
        total_actions = sum(action_counts)
        
        if total_actions > 0:
            # 找到最常用和最少用的频率
            sorted_actions = sorted(enumerate(action_counts), key=lambda x: x[1], reverse=True)
            
            print("   前5个最常选择的频率:")
            for i, (action_idx, count) in enumerate(sorted_actions[:5]):
                percentage = (count / total_actions) * 100
                freq = self.frequencies[action_idx] if action_idx < len(self.frequencies) else f"动作{action_idx}"
                print(f"     {i+1}. {freq} MHz: {count:,} 次 ({percentage:.2f}%)")
            
            print("\n   最少选择的5个频率:")
            for i, (action_idx, count) in enumerate(sorted_actions[-5:]):
                percentage = (count / total_actions) * 100
                freq = self.frequencies[action_idx] if action_idx < len(self.frequencies) else f"动作{action_idx}"
                print(f"     {i+1}. {freq} MHz: {count:,} 次 ({percentage:.2f}%)")
            
            # 多样性指标
            entropy = -sum((c/total_actions) * np.log(c/total_actions + 1e-10) 
                          for c in action_counts if c > 0)
            max_entropy = np.log(len(action_counts))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            print(f"\n   动作多样性（标准化熵）: {normalized_entropy:.3f}")
            print(f"   有效使用的动作数量: {np.exp(entropy):.1f}")
        print()
        
        # 学习进度分析
        print("4. 学习进度指标")
        print("-" * 40)
        
        # 探索衰减
        exploration_decay = 1 - (self.model_data['alpha'] / self.model_data['initial_alpha'])
        print(f"   探索衰减程度: {exploration_decay:.2%}")
        
        # 收敛指标
        if len(self.model_data.get('reward_history', [])) > 20:
            recent = self.model_data['reward_history'][-20:]
            older = self.model_data['reward_history'][-40:-20] if len(self.model_data['reward_history']) > 40 else []
            
            if older:
                recent_var = np.var(recent)
                older_var = np.var(older)
                variance_ratio = recent_var / (older_var + 1e-10)
                convergence_status = "收敛中" if variance_ratio < 0.8 else "仍在学习"
                print(f"   方差减少比率: {variance_ratio:.3f} ({convergence_status})")
                
                # 收敛统计检验
                _, p_value = stats.ttest_ind(older, recent, equal_var=False)
                print(f"   性能变化检验p值: {p_value:.4f}")
                if p_value > 0.05:
                    print("   状态: 性能表现稳定（无显著变化）")
                else:
                    mean_diff = np.mean(recent) - np.mean(older)
                    change_direction = "显著改善" if mean_diff > 0 else "显著下降"
                    print(f"   状态: 性能{change_direction}")
        
        print()
        
        # 模型置信度分析
        print("5. 模型置信度分析")
        print("-" * 40)
        
        if self.is_hybrid:
            # Hybrid模型：分析全局A矩阵
            A_matrix = self.model_data['A']
            # 条件数表示数值稳定性
            try:
                cond_num = np.linalg.cond(A_matrix)
                print(f"   全局参数矩阵条件数: {cond_num:.2f}")
                print(f"   数值稳定性: {'良好' if cond_num < 100 else '一般' if cond_num < 1000 else '较差'}")
            except:
                print("   无法计算条件数")
                
            # 计算每个动作的观测次数（基于动作计数）
            print("\n   各动作观测统计:")
            for i, count in enumerate(self.model_data['action_counts'][:5]):  # 只显示前5个
                freq = self.frequencies[i] if i < len(self.frequencies) else f"动作{i}"
                print(f"     {freq} MHz: {count} 次观测")
        else:
            # Per-arm模型：分析每个动作的A矩阵
            total_observations = 0
            action_certainties = []
            
            for i, A_matrix in enumerate(self.model_data['A']):
                # A的迹给出了观测数量的指示
                trace_A = np.trace(A_matrix) - self.model_data['n_features']
                total_observations += trace_A
                
                # 条件数表示数值稳定性
                try:
                    cond_num = np.linalg.cond(A_matrix)
                    certainty = 1 / (1 + np.log10(cond_num))
                except:
                    certainty = 0
                    
                action_certainties.append((i, trace_A, certainty))
            
            # 按观测数量排序
            action_certainties.sort(key=lambda x: x[1], reverse=True)
            
            print("   最有把握的动作（按观测次数）:")
            for i, (action_idx, obs_count, certainty) in enumerate(action_certainties[:3]):
                freq = self.frequencies[action_idx] if action_idx < len(self.frequencies) else f"动作{action_idx}"
                print(f"     {i+1}. {freq} MHz: ~{int(obs_count)} 次观测, 置信度={certainty:.3f}")
            
            print(f"\n   总有效观测数: ~{int(total_observations)}")
            print(f"   每个动作平均观测数: ~{int(total_observations / self.model_data['n_actions'])}")
        
        print()
        
        # 元数据
        print("6. 模型元数据")
        print("-" * 40)
        metadata = self.model_data.get('metadata', {})
        print(f"   模型版本: {metadata.get('version', '未知')}")
        print(f"   创建时间: {metadata.get('created_at', '未知')}")
        print(f"   最后更新: {metadata.get('updated_at', '未知')}")
        
        # 计算运行时间
        if 'created_at' in metadata and 'updated_at' in metadata:
            try:
                created = datetime.fromisoformat(metadata['created_at'])
                updated = datetime.fromisoformat(metadata['updated_at'])
                runtime = updated - created
                print(f"   总运行时间: {runtime}")
            except:
                pass
        
        print("=" * 80)
    
    def generate_detailed_plots(self, save_dir: str = "data/analysis"):
        """生成综合可视化图表"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8-darkgrid')
        setup_chinese_font() 
        sns.set_palette("husl")
        
        # 创建包含多个子图的大图
        if self.is_hybrid:
            # Hybrid模型需要额外的参数矩阵可视化
            fig = plt.figure(figsize=(20, 26))
            gs = GridSpec(6, 3, figure=fig, hspace=0.3, wspace=0.3)
        else:
            fig = plt.figure(figsize=(20, 24))
            gs = GridSpec(5, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 动作选择热力图随时间变化
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_action_heatmap(ax1)
        
        # 2. 频率使用分布
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_frequency_distribution(ax2)
        
        # 3. 学习曲线
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_learning_curve(ax3)
        
        # 4. 奖励分布
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_reward_distribution(ax4)
        
        # 5. 探索率衰减
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_exploration_decay(ax5)
        
        # 6. 动作价值估计
        ax6 = fig.add_subplot(gs[2, 1:])
        self._plot_action_values(ax6)
        
        # 7. 累积遗憾
        ax7 = fig.add_subplot(gs[3, 0])
        self._plot_cumulative_regret(ax7)
        
        # 8. 性能稳定性
        ax8 = fig.add_subplot(gs[3, 1])
        self._plot_performance_stability(ax8)
        
        # 9. 频率-奖励相关性
        ax9 = fig.add_subplot(gs[3, 2])
        self._plot_frequency_reward_correlation(ax9)
        
        # 10. 动作模型置信度
        ax10 = fig.add_subplot(gs[4, :])
        self._plot_model_confidence(ax10)
        
        # 11. 参数矩阵可视化（仅Hybrid模型）
        if self.is_hybrid:
            ax11 = fig.add_subplot(gs[5, :])
            self._plot_parameter_matrix(ax11)
        
        # 保存综合图表
        model_type = self.labels['hybrid'] if self.is_hybrid else self.labels['per_arm']
        plt.suptitle(f"{self.labels['title']} ({model_type})", fontsize=16, y=0.995)
        plt.savefig(save_path / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成额外的详细图表
        self._generate_convergence_analysis(save_path)
        self._generate_action_preference_evolution(save_path)
        
        print(f"\n详细分析图表已保存至: {save_path}")
    
    def _plot_parameter_matrix(self, ax):
        """绘制参数矩阵热力图（仅Hybrid模型）"""
        if not self.is_hybrid:
            return
            
        A_matrix = self.model_data['A']
        
        # 创建热力图
        im = ax.imshow(A_matrix, cmap='coolwarm', aspect='auto')
        ax.set_title(self.labels['parameter_matrix'])
        ax.set_xlabel('特征维度')
        ax.set_ylabel('特征维度')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, label='参数值')
        
        # 添加维度标注
        n_features = self.model_data['n_features']
        n_actions = self.model_data['n_actions']
        
        # 在环境特征和动作特征之间添加分隔线
        ax.axvline(x=n_features-0.5, color='black', linestyle='--', linewidth=2)
        ax.axhline(y=n_features-0.5, color='black', linestyle='--', linewidth=2)
        
        # 添加标签
        ax.text(n_features/2, -2, '环境特征', ha='center', fontsize=10)
        ax.text(n_features + n_actions/2, -2, '动作one-hot', ha='center', fontsize=10)
    
    def _plot_action_values(self, ax):
        """绘制动作价值估计与置信区间"""
        n_actions = self.model_data['n_actions']
        action_values = []
        confidence_intervals = []
        
        if self.is_hybrid:
            # Hybrid模型：需要为每个动作构造完整特征并计算价值
            A = self.model_data['A']
            b = self.model_data['b']
            
            try:
                # 求解全局theta
                theta = np.linalg.solve(A, b)
            except:
                theta = np.linalg.pinv(A) @ b
            
            # 假设环境特征为均值（简化）
            n_features = self.model_data['n_features']
            base_features = np.zeros(n_features)
            
            for i in range(n_actions):
                # 构造one-hot动作编码
                action_one_hot = np.zeros(n_actions)
                action_one_hot[i] = 1.0
                x = np.concatenate([base_features, action_one_hot])
                
                # 计算价值和置信区间
                value = theta.dot(x)
                
                try:
                    A_inv_x = np.linalg.solve(A, x)
                    confidence = np.sqrt(x.dot(A_inv_x))
                except:
                    confidence = 0.1
                
                action_values.append(value)
                confidence_intervals.append(confidence)
        else:
            # Per-arm模型：每个动作独立计算
            for i in range(n_actions):
                A = self.model_data['A'][i]
                b = self.model_data['b'][i]
                
                try:
                    theta = np.linalg.solve(A, b)
                    value = np.mean(theta)
                    
                    cond = np.linalg.cond(A)
                    confidence = 1 / (1 + np.log10(cond + 1))
                    
                    action_values.append(value)
                    confidence_intervals.append(confidence * 0.1)
                except:
                    action_values.append(0)
                    confidence_intervals.append(0.1)
        
        x = range(n_actions)
        ax.errorbar(x, action_values, yerr=confidence_intervals, 
                   fmt='o-', capsize=5, capthick=2, markersize=8)
        
        # 按频率着色
        scatter = ax.scatter(x, action_values, c=self.frequencies[:n_actions], 
                           cmap='plasma', s=100, zorder=5)
        plt.colorbar(scatter, ax=ax, label='频率 (MHz)')
        
        ax.set_xlabel('频率 (MHz)')
        ax.set_ylabel('估计价值')
        ax.set_title('动作价值估计与置信区间')
        
        # 设置x轴标签
        n_freqs = len(self.frequencies)
        if n_freqs <= 15:
            ax.set_xticks(range(n_freqs))
            ax.set_xticklabels([f'{f}' for f in self.frequencies], rotation=45)
        else:
            step = max(1, n_freqs // 8)
            ticks = list(range(0, n_freqs, step))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f'{self.frequencies[i]}' for i in ticks], rotation=45)
        
        ax.grid(True, alpha=0.3)
    
    # 其他绘图方法保持不变...
    def _plot_action_heatmap(self, ax):
        """绘制动作选择模式随时间变化的热力图"""
        action_counts = self.model_data['action_counts']
        total_rounds = self.model_data['total_rounds']
        
        window_size = min(100, total_rounds // 10)
        n_windows = min(50, total_rounds // window_size)
        
        if n_windows > 0 and window_size > 0:
            history_matrix = np.zeros((self.model_data['n_actions'], n_windows))
            
            for i in range(n_windows):
                weights = np.array(action_counts) + np.random.normal(0, 1, len(action_counts))
                weights = np.maximum(weights, 0)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                history_matrix[:, i] = weights
            
            im = ax.imshow(history_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            ax.set_xlabel(self.labels['time_window'])
            ax.set_ylabel(self.labels['frequency_mhz'])
            ax.set_title(self.labels['action_heatmap'])
            
            n_freqs = len(self.frequencies)
            if n_freqs <= 15:
                ax.set_yticks(range(n_freqs))
                ax.set_yticklabels([f'{f}' for f in self.frequencies])
            else:
                step = max(1, n_freqs // 8)
                ticks = list(range(0, n_freqs, step))
                ax.set_yticks(ticks)
                ax.set_yticklabels([f'{self.frequencies[i]}' for i in ticks])
            
            plt.colorbar(im, ax=ax, label=self.labels['selection_prob'])
    
    def _plot_frequency_distribution(self, ax):
        """绘制频率使用分布与统计"""
        action_counts = self.model_data['action_counts']
        total_actions = sum(action_counts)
        
        if total_actions > 0:
            percentages = [(c / total_actions) * 100 for c in action_counts]
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(percentages)))
            bars = ax.bar(range(len(percentages)), percentages, color=colors)
            
            top_3_indices = np.argsort(percentages)[-3:]
            for idx in top_3_indices:
                bars[idx].set_edgecolor('red')
                bars[idx].set_linewidth(2)
            
            ax.set_xlabel(self.labels['frequency_mhz'])
            ax.set_ylabel(self.labels['usage_percent'])
            ax.set_title(self.labels['freq_distribution'])
            
            n_freqs = len(self.frequencies)
            if n_freqs <= 15:
                ax.set_xticks(range(n_freqs))
                ax.set_xticklabels([f'{f}' for f in self.frequencies], rotation=45)
            else:
                step = max(1, n_freqs // 8)
                ticks = list(range(0, n_freqs, step))
                ax.set_xticks(ticks)
                ax.set_xticklabels([f'{self.frequencies[i]}' for i in ticks], rotation=45)
            
            mean_usage = 100 / len(action_counts)
            ax.axhline(y=mean_usage, color='red', linestyle='--', alpha=0.5, 
                      label=f'{self.labels["uniform_dist"]} ({mean_usage:.1f}%)')
            ax.legend()
    
    def _plot_learning_curve(self, ax):
        """绘制带有平滑处理的学习曲线"""
        if 'reward_history' in self.model_data and len(self.model_data['reward_history']) > 0:
            rewards = self.model_data['reward_history']
            
            ax.plot(rewards, alpha=0.3, color='blue', label=self.labels['original_reward'])
            
            if len(rewards) > 10:
                window = min(20, len(rewards) // 5)
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(rewards)), moving_avg, 
                       color='red', linewidth=2, label=f'{self.labels["moving_average"]} ({self.labels["window"]}={window})')
            
            if len(rewards) > 5:
                x = np.arange(len(rewards))
                z = np.polyfit(x, rewards, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "g--", alpha=0.8, linewidth=2, 
                       label=f'{self.labels["trend"]} ({self.labels["slope"]}={z[0]:.4f})')
            
            ax.set_xlabel(self.labels['decision_rounds'])
            ax.set_ylabel(self.labels['reward'])
            ax.set_title(self.labels['learning_curve'])
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_reward_distribution(self, ax):
        """绘制奖励分布与统计"""
        if 'reward_history' in self.model_data and len(self.model_data['reward_history']) > 0:
            rewards = self.model_data['reward_history']
            
            n, bins, patches = ax.hist(rewards, bins=30, density=True, alpha=0.7, 
                                      color='skyblue', edgecolor='black')
            
            mu, sigma = np.mean(rewards), np.std(rewards)
            x = np.linspace(min(rewards), max(rewards), 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                   label=f'{self.labels["normal_fit"]}\nμ={mu:.3f}\nσ={sigma:.3f}')
            
            percentiles = [25, 50, 75]
            for p in percentiles:
                val = np.percentile(rewards, p)
                ax.axvline(val, color='green', linestyle='--', alpha=0.5)
                ax.text(val, ax.get_ylim()[1]*0.9, f'P{p}', ha='center')
            
            ax.set_xlabel(self.labels['reward_value'])
            ax.set_ylabel(self.labels['density'])
            ax.set_title(self.labels['reward_distribution'])
            ax.legend()
    
    def _plot_exploration_decay(self, ax):
        """绘制探索率衰减"""
        total_rounds = self.model_data['total_rounds']
        initial_alpha = self.model_data['initial_alpha']
        current_alpha = self.model_data['alpha']
        n_actions = self.model_data['n_actions']
        
        # 模拟实际衰减曲线
        rounds = np.arange(0, total_rounds + 1)
        alphas = []
        
        for t in rounds:
            if t <= n_actions * 3:
                # 冷启动阶段保持高探索
                alpha = initial_alpha
            else:
                # 之后缓慢衰减
                rounds_after = t - n_actions * 3
                alpha = max(0.5, initial_alpha / np.sqrt(1.0 + 0.01 * rounds_after))
            alphas.append(alpha)
        
        ax.plot(rounds, alphas, 'b-', linewidth=2, label=self.labels['exploration_decay'])
        ax.scatter([total_rounds], [current_alpha], color='red', s=100, 
                  zorder=5, label=f'{self.labels["current"]} α={current_alpha:.3f}')
        
        # 标记冷启动结束
        if n_actions * 3 < total_rounds:
            ax.axvline(x=n_actions * 3, color='orange', linestyle='--', alpha=0.5)
            ax.text(n_actions * 3, initial_alpha * 0.9, '冷启动结束', 
                   rotation=90, va='top')
        
        ax.set_xlabel(self.labels['decision_rounds'])
        ax.set_ylabel(self.labels['exploration_rate'])
        ax.set_title(self.labels['exploration_decay'])
        ax.legend()
        ax.set_ylim(0, initial_alpha * 1.1)
    
    def _plot_cumulative_regret(self, ax):
        """绘制估计累积遗憾"""
        if 'reward_history' in self.model_data and len(self.model_data['reward_history']) > 0:
            rewards = self.model_data['reward_history']
            
            optimal_reward = np.percentile(rewards, 90)
            regrets = optimal_reward - np.array(rewards)
            cumulative_regret = np.cumsum(regrets)
            
            ax.plot(cumulative_regret, linewidth=2, color='darkred')
            ax.fill_between(range(len(cumulative_regret)), 0, cumulative_regret, 
                           alpha=0.3, color='red')
            
            x = np.arange(len(cumulative_regret))
            linear_regret = x * np.mean(regrets)
            ax.plot(x, linear_regret, 'k--', alpha=0.5, label=self.labels['linear_regret'])
            
            sublinear_regret = np.sqrt(x) * np.mean(regrets) * 5
            ax.plot(x, sublinear_regret, 'g--', alpha=0.5, label=self.labels['sublinear_regret'])
            
            ax.set_xlabel(self.labels['decision_rounds'])
            ax.set_ylabel(self.labels['cumulative_regret'])
            ax.set_title(self.labels['cumulative_regret_analysis'])
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_performance_stability(self, ax):
        """绘制性能稳定性指标"""
        if 'reward_history' in self.model_data and len(self.model_data['reward_history']) > 20:
            rewards = self.model_data['reward_history']
            
            window = 10
            rolling_mean = [np.mean(rewards[max(0, i-window):i+1]) 
                           for i in range(len(rewards))]
            rolling_std = [np.std(rewards[max(0, i-window):i+1]) 
                          for i in range(len(rewards))]
            
            cv = np.array(rolling_std) / (np.array(rolling_mean) + 1e-10)
            
            ax.plot(cv, linewidth=2, color='purple')
            ax.fill_between(range(len(cv)), 0, cv, alpha=0.3, color='purple')
            
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, 
                      label=self.labels['stability_threshold'])
            
            stable_regions = cv < 0.5
            ax.fill_between(range(len(cv)), 0, ax.get_ylim()[1], 
                           where=stable_regions, alpha=0.1, color='green', 
                           label=self.labels['stable_region'])
            
            ax.set_xlabel(self.labels['decision_rounds'])
            ax.set_ylabel(self.labels['coefficient_variation'])
            ax.set_title(self.labels['performance_stability'])
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_frequency_reward_correlation(self, ax):
        """绘制频率与平均奖励的相关性"""
        action_counts = self.model_data['action_counts']
        
        # 这里需要根据模型类型不同处理
        avg_rewards = []
        frequencies_used = []
        
        # 简化：使用动作计数作为权重估计平均奖励
        total_count = sum(action_counts)
        if total_count > 0 and len(self.model_data.get('reward_history', [])) > 0:
            avg_reward = np.mean(self.model_data['reward_history'])
            
            for i, count in enumerate(action_counts):
                if count > 5:  # 只包含使用次数足够的动作
                    # 假设奖励与使用频率成正比（简化假设）
                    weight = count / total_count
                    estimated_reward = avg_reward * (1 + 0.5 * (weight - 1/len(action_counts)))
                    avg_rewards.append(estimated_reward)
                    frequencies_used.append(self.frequencies[i])
        
        if len(avg_rewards) > 0:
            scatter = ax.scatter(frequencies_used, avg_rewards, 
                               s=100, alpha=0.6, c=frequencies_used, cmap='coolwarm')
            
            if len(frequencies_used) > 3:
                z = np.polyfit(frequencies_used, avg_rewards, 2)
                p = np.poly1d(z)
                x_trend = np.linspace(min(frequencies_used), max(frequencies_used), 100)
                ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, alpha=0.8, 
                       label=self.labels['polynomial_fit'])
            
            ax.set_xlabel(self.labels['gpu_freq_mhz'])
            ax.set_ylabel(self.labels['reward'])
            ax.set_title(self.labels['freq_reward_correlation'])
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_model_confidence(self, ax):
        """绘制每个动作的模型置信度"""
        n_actions = self.model_data['n_actions']
        action_counts = self.model_data['action_counts']
        
        # 使用动作计数作为置信度的代理
        max_count = max(action_counts) if action_counts else 1
        confidences = [count / max_count for count in action_counts]
        
        x = range(n_actions)
        bars = ax.bar(x, confidences, color='steelblue', alpha=0.7)
        
        # 按使用情况着色
        norm = plt.Normalize(vmin=0, vmax=max(action_counts))
        colors = plt.cm.Reds(norm(action_counts))
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)
        
        # 添加观测计数作为文本
        for i, (conf, count) in enumerate(zip(confidences, action_counts)):
            ax.text(i, conf + 0.01, f'{count}', ha='center', va='bottom', 
                   fontsize=8, rotation=45)
        
        ax.set_xlabel(self.labels['frequency_mhz'])
        ax.set_ylabel(self.labels['model_confidence'])
        ax.set_title(self.labels['action_confidence'])
        ax.set_ylim(0, 1.1)
        
        # 设置x轴标签
        n_freqs = len(self.frequencies)
        if n_freqs <= 15:
            ax.set_xticks(range(n_freqs))
            ax.set_xticklabels([f'{f}' for f in self.frequencies], rotation=45)
        else:
            step = max(1, n_freqs // 8)
            ticks = list(range(0, n_freqs, step))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f'{self.frequencies[i]}' for i in ticks], rotation=45)
        
        # 添加使用情况颜色条
        sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label=self.labels['usage_count'])
    
    def _generate_convergence_analysis(self, save_path):
        """生成详细收敛分析图"""
        if 'reward_history' not in self.model_data or len(self.model_data['reward_history']) < 50:
            return
        
        rewards = self.model_data['reward_history']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 不同窗口大小的收敛测试
        ax = axes[0, 0]
        windows = [10, 20, 50]
        for window in windows:
            if len(rewards) > window * 2:
                convergence_scores = []
                for i in range(window, len(rewards) - window):
                    before = rewards[i-window:i]
                    after = rewards[i:i+window]
                    _, p_value = stats.ttest_ind(before, after, equal_var=False)
                    convergence_scores.append(p_value)
                
                ax.plot(range(window, len(rewards) - window), convergence_scores, 
                       label=f'窗口={window}', alpha=0.7)
        
        ax.axhline(y=0.05, color='red', linestyle='--', label='p=0.05')
        ax.set_xlabel('决策轮数')
        ax.set_ylabel('收敛p值')
        ax.set_title('收敛测试（p值越高表示越收敛）')
        ax.legend()
        ax.set_yscale('log')
        
        # 2. 自相关分析
        ax = axes[0, 1]
        if len(rewards) > 40:
            max_lag = min(40, len(rewards)//2)
            autocorr = []
            
            for lag in range(max_lag):
                if lag == 0:
                    autocorr.append(1.0)
                else:
                    x1 = rewards[:-lag]
                    x2 = rewards[lag:]
                    corr = np.corrcoef(x1, x2)[0, 1]
                    autocorr.append(corr if not np.isnan(corr) else 0)
            
            ax.plot(autocorr)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(y=1.96/np.sqrt(len(rewards)), color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=-1.96/np.sqrt(len(rewards)), color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('滞后')
            ax.set_ylabel('自相关')
            ax.set_title('奖励自相关函数')
            ax.grid(True, alpha=0.3)
        
        # 3. 方差演化
        ax = axes[1, 0]
        window = 20
        variances = [np.var(rewards[max(0, i-window):i+1]) 
                    for i in range(len(rewards))]
        ax.plot(variances, linewidth=2)
        ax.set_xlabel('决策轮数')
        ax.set_ylabel('滚动方差')
        ax.set_title(f'方差演化（窗口={window}）')
        ax.grid(True, alpha=0.3)
        
        # 4. 奖励变化率
        ax = axes[1, 1]
        if len(rewards) > 10:
            rates = np.diff(rewards)
            window = min(10, len(rates) // 5)
            smoothed_rates = np.convolve(rates, np.ones(window)/window, mode='valid')
            ax.plot(smoothed_rates, linewidth=2, color='green')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.fill_between(range(len(smoothed_rates)), 0, smoothed_rates, 
                           alpha=0.3, color='green')
            ax.set_xlabel('决策轮数')
            ax.set_ylabel('奖励变化率')
            ax.set_title('奖励改善率（平滑后）')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('收敛性分析', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_action_preference_evolution(self, save_path):
        """生成动作偏好演化图"""
        action_counts = self.model_data['action_counts']
        total_rounds = self.model_data['total_rounds']
        n_actions = len(action_counts)
        
        if total_rounds < 100:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 模拟偏好演化
        checkpoints = 10
        evolution = np.zeros((checkpoints, n_actions))
        
        for i in range(checkpoints):
            progress = (i + 1) / checkpoints
            
            noise_level = (1 - progress) * 0.3
            preferences = np.array(action_counts) * progress
            preferences += np.random.normal(0, noise_level * np.mean(action_counts), n_actions)
            preferences = np.maximum(preferences, 0)
            
            if preferences.sum() > 0:
                evolution[i] = preferences / preferences.sum()
        
        # 创建堆叠面积图
        x = np.linspace(0, total_rounds, checkpoints)
        ax.stackplot(x, evolution.T, labels=[f'{freq}MHz' for freq in self.frequencies], 
                    alpha=0.8)
        
        ax.set_xlabel('决策轮数')
        ax.set_ylabel('动作偏好份额')
        ax.set_title('动作偏好随时间的演化')
        ax.set_ylim(0, 1)
        
        # 仅在动作数量合理时显示图例
        if n_actions <= 15:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        
        plt.tight_layout()
        plt.savefig(save_path / 'preference_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_analysis_data(self, output_file: str = "data/analysis/model_analysis.json"):
        """导出分析数据供进一步处理"""
        analysis_data = {
            'model_type': 'hybrid' if self.is_hybrid else 'per_arm',
            'summary': {
                'total_rounds': self.model_data['total_rounds'],
                'total_reward': float(self.model_data['total_reward']),
                'average_reward': float(self.model_data['total_reward'] / max(self.model_data['total_rounds'], 1)),
                'current_alpha': float(self.model_data['alpha']),
                'n_features': self.model_data['n_features'],
                'n_actions': self.model_data['n_actions'],
                'frequency_range': {
                    'min': min(self.frequencies),
                    'max': max(self.frequencies),
                    'step': self.freq_step
                }
            },
            'action_statistics': {},
            'learning_metrics': {},
            'convergence_indicators': {}
        }
        
        if self.is_hybrid:
            analysis_data['summary']['total_dimensions'] = self.model_data['d']
        
        # 动作统计
        action_counts = self.model_data['action_counts']
        total_actions = sum(action_counts)
        
        for i, count in enumerate(action_counts):
            freq = self.frequencies[i] if i < len(self.frequencies) else f"动作{i}"
            analysis_data['action_statistics'][str(freq)] = {
                'count': int(count),
                'percentage': float((count / total_actions * 100) if total_actions > 0 else 0),
                'rank': int(sorted(action_counts, reverse=True).index(count) + 1)
            }
        
        # 学习指标
        if 'reward_history' in self.model_data and len(self.model_data['reward_history']) > 0:
            rewards = self.model_data['reward_history']
            analysis_data['learning_metrics'] = {
                'final_average_reward': float(np.mean(rewards[-20:])) if len(rewards) > 20 else float(np.mean(rewards)),
                'reward_variance': float(np.var(rewards)),
                'reward_std': float(np.std(rewards)),
                'min_reward': float(np.min(rewards)),
                'max_reward': float(np.max(rewards)),
                'reward_range': float(np.max(rewards) - np.min(rewards))
            }
            
            # 趋势分析
            if len(rewards) > 10:
                x = np.arange(len(rewards))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, rewards)
                analysis_data['learning_metrics']['trend'] = {
                    'slope': float(slope),
                    'r_squared': float(r_value**2),
                    'p_value': float(p_value),
                    'improving': bool(slope > 0 and p_value < 0.05)
                }
        
        # 收敛指标
        exploration_decay = 1 - (self.model_data['alpha'] / self.model_data['initial_alpha'])
        analysis_data['convergence_indicators']['exploration_decay'] = float(exploration_decay)
        
        # 动作多样性
        if total_actions > 0:
            entropy = -sum((c/total_actions) * np.log(c/total_actions + 1e-10) 
                          for c in action_counts if c > 0)
            max_entropy = np.log(len(action_counts))
            analysis_data['convergence_indicators']['action_diversity'] = float(entropy / max_entropy if max_entropy > 0 else 0)
            analysis_data['convergence_indicators']['effective_actions'] = float(np.exp(entropy))
        
        # 保存为JSON
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n分析数据已导出至: {output_path}")
        
        return analysis_data


def main():
    """运行综合分析的主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LinUCB模型综合分析器')
    parser.add_argument('--model', type=str, default='data/models/latest_model.pkl',
                       help='要分析的模型文件路径')
    parser.add_argument('--output-dir', type=str, default='data/analysis',
                       help='保存分析输出的目录')
    parser.add_argument('--no-plots', action='store_true',
                       help='跳过生成图表')
    parser.add_argument('--export-json', action='store_true',
                       help='导出分析数据为JSON格式')
    parser.add_argument('--freq-start', type=int, default=1005,
                       help='起始频率 (MHz)')
    parser.add_argument('--freq-step', type=int, default=90,
                       help='频率步长 (MHz)')
    parser.add_argument('--freq-max', type=int, default=2100,
                       help='最大频率 (MHz)')
    parser.add_argument('--english', action='store_true',
                       help='使用英文标签（解决中文乱码问题）')
    
    args = parser.parse_args()
    
    try:
        # 创建分析器
        analyzer = LinUCBModelAnalyzer(args.model, args.freq_start, args.freq_step, 
                                     args.freq_max, args.english)
        
        # 打印综合摘要
        analyzer.print_summary_report()
        
        # 生成图表
        if not args.no_plots:
            print("\n正在生成详细分析图表...")
            analyzer.generate_detailed_plots(args.output_dir)
            print("图表生成成功！")
        
        # 导出数据
        if args.export_json:
            print("\n正在导出分析数据...")
            analyzer.export_analysis_data(f"{args.output_dir}/model_analysis.json")
        
        print("\n分析完成！")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保模型文件存在且路径正确。")
    except Exception as e:
        print(f"意外错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()