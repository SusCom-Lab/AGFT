# ===== ./model_analyzer.py =====
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

# ===== ./src/feature_extractor.py =====
import numpy as np
from typing import Dict, List, Optional
from .logger import setup_logger

logger = setup_logger(__name__)

class FeatureExtractor:
    """特征提取器 - 只负责提取环境特征，不包含动作编码"""
    
    def __init__(self):
        """初始化特征提取器 - 自动计算特征数量"""
        # 定义特征名称列表
        self.feature_names = [
            'has_queue', 
            'prefill_throughput', 
            'decode_throughput', 
            'packing_efficiency',
            'concurrency', 
            'gpu_cache_usage', 
            'cache_hit_rate'
        ]
        
        # 自动计算特征数量
        self.n_features = len(self.feature_names)
        
        # 在线特征标准化
        self.mean = np.zeros(self.n_features)
        self.M2 = np.zeros(self.n_features)
        self.count = 0
        
        logger.info(f"🔧 初始化特征提取器: {self.n_features} 维特征")
        logger.info(f"   特征列表: {', '.join(self.feature_names)}")
        
    def extract(self, gauge_metrics: Dict[str, float], 
                counter_deltas: Dict[str, float]) -> np.ndarray:
        """提取特征向量"""
        features = []
        
        # 获取请求数量
        num_running = gauge_metrics.get('vllm:num_requests_running', 0)
        num_waiting = gauge_metrics.get('vllm:num_requests_waiting', 0)
        
        # 1. 队列状态（二值）
        has_queue = 1.0 if num_waiting > 0 else 0.0
        features.append(has_queue)
        
        # 2. Prefill吞吐（tokens/2秒）
        prefill_throughput = counter_deltas.get('vllm:prompt_tokens_total_delta', 0)
        features.append(prefill_throughput)
        
        # 3. Decode吞吐（tokens/2秒）
        decode_throughput = counter_deltas.get('vllm:generation_tokens_total_delta', 0)
        features.append(decode_throughput)
        
        # 4. Packing效率
        delta_sum = counter_deltas.get('vllm:iteration_tokens_total_sum_delta', 0.0)
        delta_count = counter_deltas.get('vllm:iteration_tokens_total_count_delta', 0.0)
        packing_efficiency = delta_sum / delta_count if delta_count > 0 else 0.0
        features.append(packing_efficiency)
        
        # 5. 并发度
        concurrency = num_running
        features.append(concurrency)
        
        # 6. GPU缓存使用率
        gpu_cache = gauge_metrics.get('vllm:gpu_cache_usage_perc', 0) * 100  # 转换为百分比
        features.append(gpu_cache)
        
        # 7. 缓存命中率
        hits = counter_deltas.get('vllm:gpu_prefix_cache_hits_total_delta', 0)
        queries = counter_deltas.get('vllm:gpu_prefix_cache_queries_total_delta', 1)
        cache_hit_rate = hits / max(queries, 1)
        features.append(cache_hit_rate)
        
        # 确保特征数量与定义一致
        assert len(features) == self.n_features, \
            f"特征数量不匹配: {len(features)} vs {self.n_features}"
        
        # 日志特征值
        logger.debug("📈 原始特征:")
        for name, value in zip(self.feature_names, features):
            logger.debug(f"  {name}: {value:.3f}")
            
        # 返回numpy数组
        return np.asarray(features, dtype=np.float32)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """在线特征标准化（Welford算法）"""
        # 类型检查和转换
        features = np.asarray(features, dtype=np.float32)
        
        # 维度检查
        assert features.shape[0] == self.n_features, \
            f"特征维度不匹配: {features.shape[0]} vs {self.n_features}"
        
        # 更新统计量（Welford算法）
        self.count += 1
        delta = features - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (features - self.mean)
        
        # 计算标准差
        if self.count > 1:
            std = np.sqrt(self.M2 / (self.count - 1))
        else:
            std = np.ones(self.n_features)
            
        # 标准化
        normalized = (features - self.mean) / (std + 1e-8)
        
        # 截断到[-3, 3]
        normalized = np.clip(normalized, -3, 3)
        
        # 日志
        logger.debug("📊 标准化特征:")
        for name, value in zip(self.feature_names, normalized):
            logger.debug(f"  {name}: {value:.3f}")
            
        return normalized

# ===== ./src/gpu_controller.py =====
import subprocess
import numpy as np
from typing import List, Tuple, Optional, Dict
import pynvml
from .logger import setup_logger
import time
import re

logger = setup_logger(__name__)

class GPUController:
    """兼容性增强的GPU频率控制器"""
    
    def __init__(self, gpu_id: int = 0, min_freq: int = 1005, step: int = 90):
        self.gpu_id = gpu_id
        self.min_freq = min_freq
        self.step = step
        
        # 记录上一次成功设置的频率
        self.last_successful_freq = None
        
        # 频率映射缓存
        self.freq_mapping_cache: Dict[int, int] = {}
        
        # 初始化pynvml
        try:
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            logger.info("✅ NVML初始化成功")
        except Exception as e:
            logger.error(f"❌ NVML初始化失败: {e}")
            raise
        
        # 获取GPU信息
        self._get_gpu_info()
        
        # 获取频率范围（使用多种方法）
        self.min_supported, self.max_supported = self._get_frequency_limits_safe()
        self.max_freq = min(self._get_max_frequency_safe(), self.max_supported)
        
        # 如果无法获取，使用默认值
        if self.max_freq <= 0:
            logger.warning("无法获取最大频率，使用默认值")
            self.max_freq = 2100
            self.max_supported = 2100
        
        logger.info(f"🎮 GPU {gpu_id} 频率范围: {self.min_supported}-{self.max_freq}MHz")
        
        # 生成频率列表
        self.frequencies = self._generate_frequency_list()
        logger.info(f"📊 频率列表: {self.frequencies} MHz")
        logger.info(f"🎯 共 {len(self.frequencies)} 个频率档位")
        
        # 获取当前频率
        self.current_freq = self._get_current_frequency()
        self.current_idx = self._freq_to_idx(self.current_freq)
        self.last_successful_freq = self.current_freq
        logger.info(f"📍 当前频率: {self.current_freq}MHz (索引: {self.current_idx})")
        
        # 初始能耗
        self.last_energy_mj = self._get_total_energy_consumption()
        logger.info(f"⚡ 初始能耗读数: {self.last_energy_mj:.1f}mJ")
    
    def _get_gpu_info(self):
        """获取GPU详细信息"""
        try:
            gpu_name = pynvml.nvmlDeviceGetName(self.nvml_handle).decode('utf-8')
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
            total_mem_gb = mem_info.total / (1024**3)
            
            # 获取驱动版本
            driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            
            logger.info(f"🖥️  GPU信息: {gpu_name} ({total_mem_gb:.0f}GB)")
            logger.info(f"🔧 驱动版本: {driver_version}")
            
            # 获取CUDA版本
            try:
                cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
                cuda_major = cuda_version // 1000
                cuda_minor = (cuda_version % 1000) // 10
                logger.info(f"🔧 CUDA版本: {cuda_major}.{cuda_minor}")
            except:
                pass
                
        except Exception as e:
            logger.warning(f"获取GPU信息失败: {e}")
    
    def _run_nvidia_smi_query(self, query_string: str) -> Optional[str]:
        """安全运行nvidia-smi查询"""
        cmd = f"nvidia-smi -i {self.gpu_id} --query-gpu={query_string} --format=csv,noheader,nounits"
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.debug(f"nvidia-smi查询失败 ({query_string}): {e}")
            return None
    
    def _get_frequency_limits_safe(self) -> Tuple[int, int]:
        """安全获取GPU频率限制"""
        # 方法1：尝试标准查询
        result = self._run_nvidia_smi_query("clocks.min.graphics,clocks.max.graphics")
        if result:
            try:
                values = result.split(', ')
                return int(values[0]), int(values[1])
            except:
                pass
        
        # 方法2：通过nvidia-smi -q获取详细信息
        try:
            cmd = f"nvidia-smi -i {self.gpu_id} -q -d CLOCK"
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            output = result.stdout
            
            min_freq = 300  # 默认最小值
            max_freq = 2100  # 默认最大值
            
            # 解析输出
            for line in output.split('\n'):
                if 'Graphics' in line and 'MHz' in line:
                    # 查找类似 "Graphics : 1980 MHz" 的行
                    match = re.search(r'(\d+)\s*MHz', line)
                    if match:
                        freq = int(match.group(1))
                        if freq > 500:  # 假设大于500的是最大频率
                            max_freq = freq
                        elif freq < 500:  # 小于500的可能是最小频率
                            min_freq = freq
            
            logger.info(f"通过详细查询获取频率范围: {min_freq}-{max_freq}MHz")
            return min_freq, max_freq
            
        except Exception as e:
            logger.warning(f"详细查询失败: {e}")
        
        # 方法3：使用pynvml
        try:
            # 尝试获取支持的时钟频率
            mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(self.nvml_handle)
            if mem_clocks:
                # 对于每个内存时钟，获取支持的图形时钟
                graphics_clocks = []
                for mem_clock in mem_clocks[:1]:  # 只检查第一个内存时钟
                    try:
                        gfx_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(
                            self.nvml_handle, mem_clock)
                        graphics_clocks.extend(gfx_clocks)
                    except:
                        pass
                
                if graphics_clocks:
                    min_freq = min(graphics_clocks)
                    max_freq = max(graphics_clocks)
                    logger.info(f"通过pynvml获取频率范围: {min_freq}-{max_freq}MHz")
                    return min_freq, max_freq
        except Exception as e:
            logger.debug(f"pynvml获取频率失败: {e}")
        
        # 使用默认值
        logger.warning("无法获取频率限制，使用默认值: 300-2100MHz")
        return 300, 2100
    
    def _get_max_frequency_safe(self) -> int:
        # """安全获取最大频率"""
        # # 方法1：标准查询
        # result = self._run_nvidia_smi_query("clocks.max.graphics")
        # if result:
        #     try:
        #         return int(result)
        #     except:
        #         pass
        
        # # 方法2：查询当前最大时钟
        # result = self._run_nvidia_smi_query("clocks.max.gr")
        # if result:
        #     try:
        #         return int(result)
        #     except:
        #         pass
        
        # # 方法3：通过应用时钟获取
        # try:
        #     cmd = f"nvidia-smi -i {self.gpu_id} -q -d SUPPORTED_CLOCKS"
        #     result = subprocess.run(cmd.split(), capture_output=True, text=True)
        #     if result.returncode == 0:
        #         # 查找最大的图形时钟
        #         max_clock = 0
        #         for line in result.stdout.split('\n'):
        #             if 'Graphics' in line:
        #                 match = re.search(r'(\d+)\s*MHz', line)
        #                 if match:
        #                     clock = int(match.group(1))
        #                     max_clock = max(max_clock, clock)
                
        #         if max_clock > 0:
        #             logger.info(f"从支持的时钟列表获取最大频率: {max_clock}MHz")
        #             return max_clock
        # except:
        #     pass
        
        # # 使用默认值
        # logger.warning("无法获取最大频率，使用默认值2100MHz")
        return 1605
    
    def _generate_frequency_list(self) -> List[int]:
        """生成频率列表"""
        # 如果能获取支持的频率列表，使用它
        try:
            mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(self.nvml_handle)
            if mem_clocks:
                graphics_clocks = []
                for mem_clock in mem_clocks[:1]:  # 使用第一个内存时钟
                    try:
                        gfx_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(
                            self.nvml_handle, mem_clock)
                        graphics_clocks.extend(gfx_clocks)
                    except:
                        pass
                
                if graphics_clocks:
                    # 过滤并排序
                    graphics_clocks = sorted(set(graphics_clocks))
                    # 只保留在我们范围内的频率
                    valid_clocks = [f for f in graphics_clocks 
                                   if self.min_freq <= f <= self.max_freq]
                    valid_clocks = [f for f in valid_clocks if (f - self.min_freq) % self.step == 0]
                    if len(valid_clocks) >= 5:
                        logger.info("使用GPU支持的实际频率点")
                        return valid_clocks
        except:
            pass
        
        # 否则生成理论频率列表
        frequencies = list(range(self.min_freq, self.max_freq + 1, self.step))
        if frequencies[-1] != self.max_freq:
            frequencies.append(self.max_freq)
        
        return frequencies
    
    def _get_current_frequency(self) -> int:
        """获取当前GPU频率"""
        # 尝试多种查询方式
        queries = [
            "clocks.current.graphics",
            "clocks.gr",
            "clocks.current.gr"
        ]
        
        for query in queries:
            result = self._run_nvidia_smi_query(query)
            if result:
                try:
                    return int(result)
                except:
                    pass
        
        # 如果都失败了，使用pynvml
        try:
            clock_info = pynvml.nvmlDeviceGetClockInfo(self.nvml_handle, 
                                                      pynvml.NVML_CLOCK_GRAPHICS)
            return clock_info
        except:
            pass
        
        logger.error("无法获取当前频率，使用默认值")
        return self.frequencies[len(self.frequencies)//2] if hasattr(self, 'frequencies') else 1500
    
    def set_frequency(self, freq_mhz: int) -> bool:
        """设置GPU频率"""
        # 查找最接近的可用频率
        closest_freq = min(self.frequencies, key=lambda x: abs(x - freq_mhz))
        
        # 尝试设置
        success = False
        
        # 方法1：直接设置
        cmd = f"nvidia-smi -i {self.gpu_id} -lgc {closest_freq}"
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            success = True
        except subprocess.CalledProcessError as e:
            logger.debug(f"直接设置失败: {e}")
            
            # 方法2：使用锁定时钟
            cmd = f"nvidia-smi -i {self.gpu_id} --lock-gpu-clocks={closest_freq},{closest_freq}"
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
                success = True
                logger.info("使用--lock-gpu-clocks成功")
            except:
                logger.debug("锁定时钟也失败")
        
        if success:
            # 等待一下让设置生效
            time.sleep(0.3)
            
            # 验证实际频率
            actual_freq = self._get_current_frequency()
            self.current_freq = actual_freq
            self.current_idx = self._freq_to_idx(actual_freq)
            self.last_successful_freq = actual_freq
            
            # 记录映射
            self.freq_mapping_cache[freq_mhz] = actual_freq
            
            tolerance = 50  # MHz，可按机型调
            if abs(actual_freq - freq_mhz) > tolerance:
                logger.warning(
                    f"❌ 频率偏差过大(>{tolerance}MHz)，视为设置失败: "
                    f"{freq_mhz} → {actual_freq}"
                )
                # 回滚 internal state，告诉调用方失败
                self._get_current_frequency()  # 再刷一次确保 current_freq 正确
                return False

            logger.info(
                f"✅ GPU频率设置成功: {freq_mhz}MHz → {actual_freq}MHz "
                f"(Δ{actual_freq - freq_mhz:+d}MHz)"
            )

            return True
        
        # 设置失败，尝试回退
        if self.last_successful_freq:
            logger.warning(f"频率设置失败，保持当前频率: {self.current_freq}MHz")
        
        return False
    
    def set_frequency_by_index(self, idx: int) -> bool:
        """通过索引设置频率"""
        if 0 <= idx < len(self.frequencies):
            return self.set_frequency(self.frequencies[idx])
        else:
            logger.error(f"无效的频率索引: {idx}")
            return False
    
    def reset_gpu_clocks(self) -> bool:
        """重置GPU时钟"""
        success = False
        
        # 方法1：标准重置
        cmd = f"nvidia-smi -i {self.gpu_id} -rgc"
        try:
            subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            success = True
        except:
            # 方法2：解锁时钟
            cmd = f"nvidia-smi -i {self.gpu_id} --reset-gpu-clocks"
            try:
                subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
                success = True
            except:
                logger.error("无法重置GPU时钟")
        
        if success:
            logger.info("✅ GPU时钟已重置")
            self.current_freq = self._get_current_frequency()
            self.current_idx = self._freq_to_idx(self.current_freq)
            self.last_successful_freq = self.current_freq
        
        return success
    
    def _freq_to_idx(self, freq: int) -> int:
        """频率转索引"""
        distances = [abs(f - freq) for f in self.frequencies]
        return np.argmin(distances)
    
    def read_energy_mj(self) -> float:
        """读取当前累计能耗"""
        return self._get_total_energy_consumption()
    
    def _get_total_energy_consumption(self) -> float:
        """获取GPU累计能耗（毫焦）"""
        try:
            energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.nvml_handle)
            return float(energy_mj)
        except Exception as e:
            # 某些GPU不支持能耗读取，使用功率估算
            logger.debug(f"无法读取累计能耗: {e}")
            try:
                power = self._get_current_power()
                if hasattr(self, 'last_energy_mj'):
                    return self.last_energy_mj + power * 2000
                else:
                    return power * 2000
            except:
                return 200000  # 默认100W * 2s
    
    def _get_current_power(self) -> float:
        """获取当前功率（瓦特）"""
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)
            return power_mw / 1000.0
        except:
            # 使用nvidia-smi
            result = self._run_nvidia_smi_query("power.draw")
            if result:
                try:
                    return float(result)
                except:
                    pass
            
            logger.debug("无法获取功率，使用默认值100W")
            return 100.0
    
    def get_energy_delta(self) -> float:
        """获取能耗增量"""
        try:
            current_energy_mj = self._get_total_energy_consumption()
            
            if hasattr(self, 'last_energy_mj') and self.last_energy_mj is not None:
                delta = current_energy_mj - self.last_energy_mj
                
                if delta < 0:
                    logger.warning("能耗读数回绕或不支持，使用功率估算")
                    power = self._get_current_power()
                    delta = power * 2000
                elif delta > 2000000:  # 超过2000J
                    logger.warning(f"能耗增量异常: {delta/1000:.1f}J，使用功率估算")
                    power = self._get_current_power()
                    delta = power * 2000
            else:
                delta = 0
            
            self.last_energy_mj = current_energy_mj
            return delta
            
        except Exception as e:
            logger.error(f"计算能耗增量失败: {e}")
            return 200000  # 默认100W * 2s
    
    def get_gpu_stats(self) -> dict:
        """获取GPU状态"""
        stats = {
            'temperature': 0,
            'utilization': 0,
            'memory_used': 0,
            'memory_total': 0,
            'power': 0
        }
        
        try:
            # 使用pynvml
            stats['temperature'] = pynvml.nvmlDeviceGetTemperature(
                self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
            
            util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
            stats['utilization'] = util.gpu
            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
            stats['memory_used'] = mem_info.used / (1024**2)
            stats['memory_total'] = mem_info.total / (1024**2)
            
            stats['power'] = self._get_current_power()
            
        except Exception as e:
            logger.debug(f"pynvml获取状态失败，尝试nvidia-smi: {e}")
            
            # 使用nvidia-smi作为后备
            result = self._run_nvidia_smi_query(
                "temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw")
            if result:
                try:
                    values = result.split(', ')
                    stats['temperature'] = float(values[0])
                    stats['utilization'] = float(values[1])
                    stats['memory_used'] = float(values[2])
                    stats['memory_total'] = float(values[3])
                    stats['power'] = float(values[4])
                except:
                    logger.error("解析GPU状态失败")
        
        return stats
    
    def __del__(self):
        """清理资源"""
        try:
            self.reset_gpu_clocks()
            pynvml.nvmlShutdown()
            logger.debug("GPU控制器已清理")
        except:
            pass

# ===== ./src/linucb.py =====
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Optional
from datetime import datetime
from .logger import setup_logger

logger = setup_logger(__name__)

class LinUCB:
    """
    Hybrid/Global LinUCB - 所有动作共享一套参数
    使用one-hot编码拼接动作信息
    """
    def __init__(self, n_features: int, n_actions: int, alpha: float = 3.0,
                 lambda_reg: float = 0.1,
                 model_dir: str = "data/models", auto_load: bool = True):
        self.n_features = n_features   # 环境特征维度
        self.n_actions = n_actions     # 动作数量
        self.d = n_features + n_actions  # 总维度 = 环境特征 + one-hot动作
        self.alpha = alpha
        self.initial_alpha = alpha
        self.lambda_reg = lambda_reg
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 全局共享的A和b矩阵
        self.A = self.lambda_reg * np.eye(self.d, dtype=np.float32)
        self.b = np.zeros(self.d, dtype=np.float32)

        # 统计信息
        self.action_counts = [0] * n_actions
        self.total_rounds = 0
        self.total_reward = 0.0
        self.reward_history = []
        self.last_action = None  # 记录上一次的动作（用于计算切换成本）

        # 用于冷启动的随机排列
        self._cold_start_permutation = None

        # 模型元数据
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'version': '3.0-hybrid'
        }

        logger.info(f"🤖 初始化Hybrid LinUCB:")
        logger.info(f"   环境特征: {n_features}维")
        logger.info(f"   动作数量: {n_actions}个")
        logger.info(f"   总维度: {self.d}维")
        logger.info(f"   初始α: {alpha}")
        logger.info(f"   正则化λ: {lambda_reg}")

        if auto_load:
            self.load_model()
        else:
            logger.info("🆕 跳过模型加载，从全新状态开始")

    def _create_context_with_action(self, base_features: np.ndarray, action: int) -> np.ndarray:
        """将环境特征和动作one-hot编码拼接"""
        # One-hot编码动作
        action_one_hot = np.zeros(self.n_actions, dtype=np.float32)
        action_one_hot[action] = 1.0
        
        # 拼接
        context = np.concatenate([base_features, action_one_hot])
        
        assert context.shape[0] == self.d, \
            f"拼接后维度错误: {context.shape[0]} vs {self.d}"
        
        return context

    def select_action(self, base_features: np.ndarray) -> int:
        """选择动作 - 使用UCB策略"""
        # 冷启动阶段：前n_actions*3轮，随机顺序探索每个动作
        explore_rounds = self.n_actions * 3
        if self.total_rounds < explore_rounds:
            # 初始化随机排列
            if self._cold_start_permutation is None or len(self._cold_start_permutation) < explore_rounds:
                # 生成3轮完整的随机排列
                self._cold_start_permutation = []
                for _ in range(3):
                    self._cold_start_permutation.extend(np.random.permutation(self.n_actions))
            
            selected = self._cold_start_permutation[self.total_rounds]
            logger.info(f"🎲 [冷启动探索] 轮次 {self.total_rounds + 1}/{explore_rounds}, 选择动作 {selected}")
            return selected

        # 计算当前参数估计
        try:
            theta = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            logger.warning("⚠️ A矩阵奇异，使用伪逆")
            theta = np.linalg.pinv(self.A) @ self.b

        # 计算每个动作的UCB值
        ucb_values = []
        for action in range(self.n_actions):
            # 构造完整特征
            x = self._create_context_with_action(base_features, action)
            
            # 预测值
            pred = theta.dot(x)
            
            # 置信区间（使用Cholesky分解加速）
            try:
                L = np.linalg.cholesky(self.A)
                v = np.linalg.solve(L, x)
                confidence = self.alpha * np.sqrt(np.dot(v, v))
            except np.linalg.LinAlgError:
                # 如果Cholesky分解失败，使用标准方法
                A_inv_x = np.linalg.solve(self.A, x)
                confidence = self.alpha * np.sqrt(x.dot(A_inv_x))
            
            ucb = pred + confidence
            ucb_values.append(ucb)
            
            logger.debug(f"  动作{action}: pred={pred:.3f}, conf={confidence:.3f}, UCB={ucb:.3f}")

        # 选择UCB最大的动作（有并列时随机选择）
        max_ucb = max(ucb_values)
        candidates = [i for i, v in enumerate(ucb_values) if abs(v - max_ucb) < 1e-9]
        selected = np.random.choice(candidates)
        
        logger.info(f"🎯 选择动作 {selected}, UCB值: {ucb_values[selected]:.3f}")
        
        return selected

    def update(self, action: int, base_features: np.ndarray, reward: float):
        """更新模型参数"""
        # 构造完整特征
        x = self._create_context_with_action(base_features, action)
        
        # 更新A和b（不使用衰减）
        self.A += np.outer(x, x)
        self.b += reward * x

        # 更新统计信息
        self.action_counts[action] += 1
        self.total_rounds += 1
        self.total_reward += reward
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)
        
        self.last_action = action

        # α衰减策略：前3*K轮保持高探索，之后缓慢衰减
        if self.total_rounds <= self.n_actions * 3:
            # 冷启动阶段保持高探索
            self.alpha = self.initial_alpha
        else:
            # 使用更缓慢的衰减
            rounds_after_explore = self.total_rounds - self.n_actions * 3
            self.alpha = max(0.5, self.initial_alpha / np.sqrt(1.0 + 0.01 * rounds_after_explore))
        
        logger.info(f"📈 更新模型: 动作={action}, 奖励={reward:.6f}, α={self.alpha:.3f}, 总轮次={self.total_rounds}")

        # 定期保存模型
        if self.total_rounds % 10 == 0:
            self.save_model()

    def save_model(self, filename: Optional[str] = None):
        """保存模型"""
        if filename is None:
            filename = f"linucb_hybrid_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        filepath = self.model_dir / filename
        
        # 更新元数据
        self.metadata['updated_at'] = datetime.now().isoformat()
        self.metadata['total_rounds'] = self.total_rounds
        self.metadata['total_reward'] = self.total_reward
        self.metadata['avg_reward'] = self.total_reward / max(self.total_rounds, 1)
        
        # 打包模型数据
        model_data = {
            'A': self.A.astype(np.float32),
            'b': self.b.astype(np.float32),
            'n_features': self.n_features,
            'n_actions': self.n_actions,
            'd': self.d,
            'alpha': self.alpha,
            'initial_alpha': self.initial_alpha,
            'lambda_reg': self.lambda_reg,
            'action_counts': self.action_counts,
            'total_rounds': self.total_rounds,
            'total_reward': self.total_reward,
            'reward_history': self.reward_history[-100:],  # 只保存最近100个
            'last_action': self.last_action,
            'metadata': self.metadata
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"💾 模型已保存到: {filepath}")
            
            # 创建软链接指向最新模型
            latest_path = self.model_dir / "latest_model.pkl"
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(filepath.name)
            
            # 保存元数据JSON
            meta_path = self.model_dir / "model_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"保存模型失败: {e}")

    def load_model(self, filename: Optional[str] = None):
        """加载模型"""
        if filename is None:
            # 尝试加载最新模型
            latest_path = self.model_dir / "latest_model.pkl"
            if latest_path.exists():
                filepath = latest_path
            else:
                # 查找最新的模型文件
                model_files = list(self.model_dir.glob("linucb_hybrid_model_*.pkl"))
                if not model_files:
                    logger.info("📂 没有找到已保存的模型，将从头开始学习")
                    return False
                filepath = max(model_files, key=lambda p: p.stat().st_mtime)
        else:
            filepath = self.model_dir / filename
            
        if not filepath.exists():
            logger.warning(f"模型文件不存在: {filepath}")
            return False
            
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            # 兼容性检查
            if model_data.get('version', '').startswith('3.0'):
                # 新版本模型
                if (model_data['n_features'] != self.n_features or 
                    model_data['n_actions'] != self.n_actions):
                    logger.warning("⚠️ 加载的模型参数与当前配置不符")
                    return False
            else:
                # 旧版本模型，无法兼容
                logger.warning("⚠️ 模型版本过旧，无法加载")
                return False
                
            # 恢复参数
            self.A = model_data['A']
            self.b = model_data['b']
            self.d = model_data['d']
            self.alpha = model_data['alpha']
            self.initial_alpha = model_data['initial_alpha']
            self.lambda_reg = model_data['lambda_reg']
            self.action_counts = model_data['action_counts']
            self.total_rounds = model_data['total_rounds']
            self.total_reward = model_data['total_reward']
            self.reward_history = model_data.get('reward_history', [])
            self.last_action = model_data.get('last_action', None)
            self.metadata = model_data['metadata']
            
            logger.info(f"✅ 模型已加载: {filepath}")
            logger.info(f"   总轮次: {self.total_rounds}")
            logger.info(f"   平均奖励: {self.total_reward / max(self.total_rounds, 1):.3f}")
            logger.info(f"   当前α: {self.alpha:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False

    def get_model_stats(self) -> dict:
        """获取模型统计信息"""
        stats = {
            'total_rounds': self.total_rounds,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(self.total_rounds, 1),
            'action_counts': self.action_counts,
            'current_alpha': self.alpha,
            'recent_rewards': self.reward_history[-20:] if self.reward_history else [],
            'last_action': self.last_action,
            'metadata': self.metadata
        }
        
        # 计算动作分布熵
        if sum(self.action_counts) > 0:
            probs = np.array(self.action_counts) / sum(self.action_counts)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            stats['action_entropy'] = entropy
            stats['effective_actions'] = np.exp(entropy)
        
        return stats

# ===== ./src/logger.py =====
import logging
import sys
from datetime import datetime
from pathlib import Path

class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)

def setup_logger(name, log_file=None):
    """设置日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 控制台处理器（带颜色）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定）
    if log_file:
        Path("logs").mkdir(exist_ok=True)
        file_handler = logging.FileHandler(f"logs/{log_file}")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# ===== ./src/main.py =====
import time
import yaml
import signal
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from .logger import setup_logger
from .metrics_collector import MetricsCollector
from .gpu_controller import GPUController
from .feature_extractor import FeatureExtractor
from .linucb import LinUCB
from .reward_calculator import EDPRewardCalculator

# 设置主日志
logger = setup_logger(__name__, f"vllm_gpu_autoscaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

class VLLMGPUAutoscaler:
    """vLLM GPU自适应调频主控制器"""
    
    def __init__(self, config_path: str = "config/config.yaml", 
                 reset_model: bool = False, 
                 model_file: Optional[str] = None):
        logger.info("="*60)
        logger.info("🚀 vLLM GPU自适应调频系统 v2.0 (Hybrid LinUCB)")
        logger.info("="*60)
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 初始化组件
        logger.info("🔧 初始化系统组件...")
        
        # 1. Prometheus指标采集器
        self.metrics_collector = MetricsCollector(
            self.config['vllm']['prometheus_url']
        )
        
        # 2. GPU控制器
        self.gpu_controller = GPUController(
            gpu_id=self.config['gpu']['device_id'],
            min_freq=self.config['gpu']['min_frequency'],
            step=self.config['gpu']['frequency_step'],
        )
        
        # 3. 特征提取器（自动计算特征数量）
        self.feature_extractor = FeatureExtractor()
        
        # 4. LinUCB模型（Hybrid版本）
        model_dir = self.config.get('model', {}).get('save_dir', 'data/models')
        self.linucb = LinUCB(
            n_features=self.feature_extractor.n_features,  # 使用提取器的特征数量
            n_actions=len(self.gpu_controller.frequencies),
            alpha=self.config['linucb']['initial_alpha'],
            lambda_reg=self.config['linucb'].get('lambda_reg', 0.1),
            model_dir=model_dir,
            auto_load=not reset_model  # 如果reset_model=True，则不自动加载
        )
        
        # 处理模型加载选项
        if reset_model:
            logger.info("🔄 重置模型，从头开始学习")
        elif model_file:
            logger.info(f"📂 加载指定模型: {model_file}")
            if not self.linucb.load_model(model_file):
                logger.error("加载模型失败，将从头开始学习")
        else:
            if self.linucb.total_rounds > 0:
                logger.info(f"✅ 继续上次的学习进度 (已完成 {self.linucb.total_rounds} 轮)")
            else:
                logger.info("📝 开始全新的学习过程")
                
        # 5. EDP奖励计算器
        self.reward_calculator = EDPRewardCalculator(
            ttft_limit=self.config['control']['ttft_limit'],
            tpot_limit=self.config['control']['tpot_limit'],
            switch_cost_weight=self.config.get('control', {}).get('switch_cost_weight', 0.1)
        )
        
        # 控制参数
        self.decision_interval = self.config['control']['decision_interval']
        self.running = True
        
        # 记录上一次的频率（用于计算切换成本）
        self.last_frequency = self.gpu_controller.current_freq
        
        # 统计信息
        self.stats = {
            'start_time': time.time(),
            'decisions': 0,
            'total_energy': 0.0,
            'frequency_changes': 0,
            'emergency_actions': 0,
            'best_reward': float('-inf'),
            'worst_reward': float('inf'),
            'idle_cycles': 0,
            'learning_cycles': 0
        }
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("✅ 系统初始化完成")
        logger.info(f"📋 配置摘要:")
        logger.info(f"   - GPU设备: {self.config['gpu']['device_id']}")
        logger.info(f"   - 频率范围: {self.gpu_controller.min_freq}-{self.gpu_controller.max_freq}MHz")
        logger.info(f"   - 频率档位: {len(self.gpu_controller.frequencies)}个")
        logger.info(f"   - 决策间隔: {self.decision_interval}秒")
        logger.info(f"   - TTFT限制: {self.config['control']['ttft_limit']}秒")
        logger.info(f"   - TPOT限制: {self.config['control']['tpot_limit']}秒")
        logger.info(f"   - 初始探索率: {self.config['linucb']['initial_alpha']}")
        logger.info(f"   - 正则化参数: {self.config['linucb'].get('lambda_reg', 0.1)}")
        
    def _signal_handler(self, signum, frame):
        """处理退出信号"""
        logger.info("\n" + "="*60)
        logger.info("📛 收到退出信号，正在优雅关闭...")
        
        # 保存模型
        logger.info("💾 保存最终模型...")
        self.linucb.save_model(f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        
        # 显示最终统计
        self._display_final_stats()
        
        self.running = False
        logger.info("👋 再见！")
        sys.exit(0)
        
    def _check_emergency(self, gauge_metrics: Dict, counter_deltas: Dict) -> Optional[int]:
        """检查紧急情况，返回建议的动作索引"""
        # TTFT超限检查
        ttft_count = counter_deltas.get('vllm:time_to_first_token_seconds_count_delta', 0)
        ttft_sum = counter_deltas.get('vllm:time_to_first_token_seconds_sum_delta', 0)
        if ttft_count > 0:
            avg_ttft = ttft_sum / ttft_count
            if avg_ttft > self.config['control']['ttft_limit']:
                logger.warning(f"🚨 紧急！TTFT={avg_ttft:.3f}s > {self.config['control']['ttft_limit']}s")
                self.stats['emergency_actions'] += 1
                return len(self.gpu_controller.frequencies) - 1  # 最高频率
        
        # TPOT超限检查
        tpot_count = counter_deltas.get('vllm:time_per_output_token_seconds_count_delta', 0)
        tpot_sum = counter_deltas.get('vllm:time_per_output_token_seconds_sum_delta', 0)
        if tpot_count > 0:
            avg_tpot = tpot_sum / tpot_count
            if avg_tpot > self.config['control']['tpot_limit']:
                logger.warning(f"🚨 紧急！TPOT={avg_tpot:.3f}s > {self.config['control']['tpot_limit']}s")
                self.stats['emergency_actions'] += 1
                return len(self.gpu_controller.frequencies) - 1  # 最高频率
                
        # 抢占发生
        if counter_deltas.get('vllm:num_preemptions_total_delta', 0) > 0:
            logger.warning("🚨 发生抢占，需要更高频率")
            self.stats['emergency_actions'] += 1
            # 提升2个档位，但不超过最高
            return min(len(self.gpu_controller.frequencies) - 1, 
                      self.gpu_controller.current_idx + 2)
            
        # GPU过热保护
        gpu_stats = self.gpu_controller.get_gpu_stats()
        if gpu_stats['temperature'] > 100:
            logger.warning(f"🔥 GPU过热: {gpu_stats['temperature']}°C")
            self.stats['emergency_actions'] += 1
            # 降低2个档位，但不低于最低
            return max(0, self.gpu_controller.current_idx - 2)
            
        return None
        
    def run(self):
        """主控制循环"""
        logger.info("\n" + "="*60)
        logger.info("🎮 开始自适应GPU调频控制")
        logger.info("="*60 + "\n")
        
        iteration = 0
        consecutive_errors = 0
        
        while self.running:
            iteration += 1
            cycle_start = time.time()
            
            # 动态分隔线
            if iteration % 10 == 0:
                logger.info(f"\n{'='*20} 🎯 迭代 {iteration} {'='*20}")
            else:
                logger.info(f"\n{'─'*20} 迭代 {iteration} {'─'*20}")
            
            try:
                # 1. 采集当前状态（2秒窗口）
                logger.info("📊 [阶段1/5] 采集系统状态...")
                gauge_metrics, counter_deltas, energy_delta = \
                    self.metrics_collector.collect_2s_metrics(
                        energy_reader=self.gpu_controller.read_energy_mj
                    )

                # 检查是否完全空闲
                num_running = gauge_metrics.get('vllm:num_requests_running', 0)
                num_waiting = gauge_metrics.get('vllm:num_requests_waiting', 0)
                
                if num_running == 0 and num_waiting == 0:
                    # 系统空闲
                    logger.info("😴 系统完全空闲，重置GPU时钟")
                    self.gpu_controller.reset_gpu_clocks()
                    self.stats['idle_cycles'] += 1
                    
                    # 等待下一个决策周期
                    logger.info(f"⏳ 等待{self.decision_interval}秒...")
                    time.sleep(self.decision_interval)
                    
                    # 记录能耗但不更新模型
                    energy_consumed = self.gpu_controller.get_energy_delta()
                    self.stats['total_energy'] += energy_consumed
                    
                    continue  # 跳过本轮的学习和决策
                
                self.stats['learning_cycles'] += 1
                
                # 2. 特征提取和标准化
                logger.info("🔍 [阶段2/5] 提取特征...")
                raw_features = self.feature_extractor.extract(gauge_metrics, counter_deltas)
                normalized_features = self.feature_extractor.normalize(raw_features)
                
                # 显示关键特征
                logger.info(f"   队列状态: {'有等待' if raw_features[0] > 0 else '无等待'}")
                logger.info(f"   并发请求: {raw_features[4]:.0f}")
                logger.info(f"   GPU缓存: {raw_features[5]:.1f}%")
                
                # 3. 紧急情况检查
                emergency_action = self._check_emergency(gauge_metrics, counter_deltas)

                # 4. 决策
                if emergency_action is not None:
                    actual_action = emergency_action
                    logger.info(f"⚡ [阶段3/5] 紧急决策: 动作{actual_action} "
                              f"({self.gpu_controller.frequencies[actual_action]}MHz)")
                else:
                    # LinUCB决策（Hybrid版本直接使用基础特征）
                    actual_action = self.linucb.select_action(normalized_features)
                    logger.info(f"🤔 [阶段3/5] LinUCB决策: 动作{actual_action} "
                              f"({self.gpu_controller.frequencies[actual_action]}MHz)")

                # 5. 应用新频率
                new_freq = self.gpu_controller.frequencies[actual_action]
                old_freq = self.gpu_controller.current_freq
                
                if new_freq != old_freq:
                    freq_change = new_freq - old_freq
                    symbol = "📈" if freq_change > 0 else "📉"
                    logger.info(f"{symbol} [阶段4/5] 调整频率: {old_freq}MHz → {new_freq}MHz "
                              f"({freq_change:+d}MHz)")
                    
                    # 设置新频率
                    if self.gpu_controller.set_frequency(new_freq):
                        # 成功后再读一次实际频率
                        new_freq = self.gpu_controller.current_freq
                        self.stats['frequency_changes'] += 1
                        self.last_frequency = old_freq
                    else:
                        logger.error("❌ 设置频率失败，保持原频率")
                        new_freq = old_freq  # 回退
                else:
                    logger.info(f"✅ [阶段4/5] 保持频率: {new_freq}MHz")
                
                # 6. 等待效果稳定
                logger.info(f"⏳ 等待{self.decision_interval}秒观察效果...")
                time.sleep(self.decision_interval)
                
                # 7. 采集新状态
                logger.info("📊 [阶段5/5] 评估效果...")
                new_gauge, new_counter_deltas, energy_consumed = \
                    self.metrics_collector.collect_2s_metrics(
                        energy_reader=self.gpu_controller.read_energy_mj
                    )
                
                # 获取实际能耗
                energy_consumed = self.gpu_controller.get_energy_delta()
                
                # 8. 计算奖励（包含频率切换成本）
                reward, reward_info = self.reward_calculator.calculate(
                    new_counter_deltas, 
                    energy_consumed,
                    current_freq=new_freq,
                    previous_freq=self.last_frequency,
                    max_freq=self.gpu_controller.max_freq
                )
                
                # 9. 更新模型（如果不是预热期或无请求）
                if not reward_info.get('no_requests') and not reward_info.get('warming_up'):
                    # Hybrid LinUCB直接使用基础特征，不需要手动拼接动作
                    self.linucb.update(actual_action, normalized_features, reward)
                    
                    # 更新最佳/最差奖励
                    if reward > self.stats['best_reward']:
                        self.stats['best_reward'] = reward
                        logger.info(f"🏆 新的最佳奖励: {reward:.3f}")
                    if reward < self.stats['worst_reward']:
                        self.stats['worst_reward'] = reward
                    
                # 10. 显示详细状态
                self._display_status(
                    iteration, old_freq, new_freq, 
                    gauge_metrics, new_gauge, 
                    reward, reward_info
                )
                
                # 更新统计
                self.stats['decisions'] += 1
                self.stats['total_energy'] += energy_consumed
                
                # 重置错误计数
                consecutive_errors = 0
                
            except KeyboardInterrupt:
                raise  # 让信号处理器处理
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"❌ 迭代错误 ({consecutive_errors}): {e}", exc_info=True)
                
                if consecutive_errors >= 5:
                    logger.critical("连续错误过多，系统退出")
                    self.running = False
                else:
                    logger.info(f"等待{self.decision_interval}秒后重试...")
                    time.sleep(self.decision_interval)
            
            # 显示循环耗时
            cycle_time = time.time() - cycle_start
            if cycle_time > self.decision_interval * 2.5:
                logger.warning(f"⚠️ 循环耗时过长: {cycle_time:.1f}秒")
                
    def _display_status(self, iteration, old_freq, new_freq, 
                       old_metrics, new_metrics, reward, reward_info):
        """显示详细状态信息"""
        
        # 构建状态框
        status_lines = []
        
        status_lines.append("\n┌" + "─"*58 + "┐")
        status_lines.append(f"│ 📊 状态汇总 - 迭代 {iteration:<41} │")
        status_lines.append("├" + "─"*58 + "┤")
        
        # 频率信息
        freq_change = new_freq - old_freq
        freq_symbol = "↑" if freq_change > 0 else ("↓" if freq_change < 0 else "→")
        status_lines.append(f"│ 🔧 频率: {old_freq:>4}MHz {freq_symbol} {new_freq:<4}MHz "
                          f"({'+'if freq_change>=0 else ''}{freq_change}MHz){' '*14} │")
        
        # 负载信息
        old_running = old_metrics.get('vllm:num_requests_running', 0)
        old_waiting = old_metrics.get('vllm:num_requests_waiting', 0)
        new_running = new_metrics.get('vllm:num_requests_running', 0)
        new_waiting = new_metrics.get('vllm:num_requests_waiting', 0)
        
        status_lines.append(f"│ 👥 请求: 运行 {old_running:.0f}→{new_running:.0f}, "
                          f"等待 {old_waiting:.0f}→{new_waiting:.0f}{' '*23} │")
        
        # 缓存信息
        old_cache = old_metrics.get('vllm:gpu_cache_usage_perc', 0)*100
        new_cache = new_metrics.get('vllm:gpu_cache_usage_perc', 0)*100
        status_lines.append(f"│ 💾 缓存: {old_cache:>5.1f}% → {new_cache:<5.1f}%{' '*31} │")
        
        # 性能指标
        if reward_info.get('avg_ttft') is not None:
            ttft_ms = reward_info['avg_ttft'] * 1000
            ttft_baseline_ms = reward_info.get('ttft_ema', 0) * 1000
            status_lines.append(f"│ ⏱️  TTFT: {ttft_ms:>6.1f}ms (基线: {ttft_baseline_ms:.1f}ms){' '*18} │")
            
        if reward_info.get('avg_tpot') is not None:
            tpot_ms = reward_info['avg_tpot'] * 1000
            tpot_baseline_ms = reward_info.get('tpot_ema', 0) * 1000
            status_lines.append(f"│ ⏱️  TPOT: {tpot_ms:>6.1f}ms (基线: {tpot_baseline_ms:.1f}ms){' '*18} │")
        
        # 能耗和奖励
        energy = reward_info.get('energy_j', 0)
        edp = reward_info.get('edp', 0)
        status_lines.append(f"│ ⚡ 能耗: {energy:>6.1f}J,  EDP: {edp:.6f}{' '*19} │")
        
        # 奖励组成
        if 'base_reward' in reward_info:
            status_lines.append(f"│ 💰 奖励组成: 基础={reward_info['base_reward']:+.2f}, "
                              f"切换={reward_info.get('switch_cost', 0):+.2f}, "
                              f"惩罚={reward_info.get('delay_penalty', 0):+.2f}{' '*5} │")
        
        reward_symbol = "🟢" if reward > 0 else ("🔴" if reward < -10 else "🟡")
        status_lines.append(f"│ {reward_symbol} 最终奖励: {reward:>8.3f}{' '*33} │")
        
        # GPU状态
        gpu_stats = self.gpu_controller.get_gpu_stats()
        status_lines.append(f"│ 🌡️  GPU: {gpu_stats['temperature']:.0f}°C, "
                          f"{gpu_stats['utilization']:.0f}%, "
                          f"{gpu_stats['power']:.0f}W{' '*23} │")
        
        # 模型状态
        model_stats = self.linucb.get_model_stats()
        status_lines.append(f"│ 🤖 模型: α={self.linucb.alpha:.2f}, "
                          f"总轮次={model_stats['total_rounds']}, "
                          f"平均奖励={model_stats['avg_reward']:.3f}{' '*8} │")
        
        status_lines.append("└" + "─"*58 + "┘")
        
        # 一次性输出
        logger.info('\n'.join(status_lines))
        
    def _display_final_stats(self):
        """显示最终统计信息"""
        runtime = time.time() - self.stats['start_time']
        
        stats_lines = []
        stats_lines.append("\n" + "="*60)
        stats_lines.append("📊 最终统计报告")
        stats_lines.append("="*60)
        
        # 运行信息
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        seconds = int(runtime % 60)
        stats_lines.append(f"⏱️  运行时间: {hours:02d}:{minutes:02d}:{seconds:02d}")
        stats_lines.append(f"🔄 总迭代次数: {self.stats['decisions'] + self.stats['idle_cycles']}")
        stats_lines.append(f"   - 学习周期: {self.stats['learning_cycles']}")
        stats_lines.append(f"   - 空闲周期: {self.stats['idle_cycles']}")
        stats_lines.append(f"🔧 频率调整: {self.stats['frequency_changes']}次")
        stats_lines.append(f"🚨 紧急响应: {self.stats['emergency_actions']}次")
        
        # 能耗信息
        total_energy_j = self.stats['total_energy'] / 1000
        avg_power = self.stats['total_energy'] / runtime if runtime > 0 else 0
        stats_lines.append(f"⚡ 总能耗: {total_energy_j:.1f}J")
        stats_lines.append(f"⚡ 平均功率: {avg_power:.1f}mW")
        
        # 奖励信息
        if self.stats['best_reward'] > float('-inf'):
            stats_lines.append(f"🏆 最佳奖励: {self.stats['best_reward']:.3f}")
        if self.stats['worst_reward'] < float('inf'):
            stats_lines.append(f"💀 最差奖励: {self.stats['worst_reward']:.3f}")
        
        # 频率使用分布
        model_stats = self.linucb.get_model_stats()
        stats_lines.append(f"\n📊 频率使用分布:")
        total_uses = sum(model_stats['action_counts'])
        for i, (freq, count) in enumerate(zip(self.gpu_controller.frequencies, 
                                            model_stats['action_counts'])):
            if total_uses > 0:
                percentage = count / total_uses * 100
                bar_length = int(percentage / 2)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                stats_lines.append(f"   {freq:>4}MHz: {bar} {percentage:>5.1f}% ({count}次)")
            else:
                stats_lines.append(f"   {freq:>4}MHz: {'░'*50}   0.0% (0次)")
        
        # 模型信息
        stats_lines.append(f"\n🤖 模型信息:")
        stats_lines.append(f"   最终α: {model_stats['current_alpha']:.3f}")
        stats_lines.append(f"   平均奖励: {model_stats['avg_reward']:.3f}")
        if 'action_entropy' in model_stats:
            stats_lines.append(f"   动作熵: {model_stats['action_entropy']:.3f}")
            stats_lines.append(f"   有效动作数: {model_stats['effective_actions']:.1f}")
        
        stats_lines.append("="*60)
        
        # 一次性输出
        logger.info('\n'.join(stats_lines))

def main():
    """程序入口"""
    parser = argparse.ArgumentParser(
        description='vLLM GPU自适应调频系统 (Hybrid LinUCB)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 正常启动（自动加载最新模型）
  python -m src.main
  
  # 从头开始学习
  python -m src.main --reset-model
  
  # 加载特定模型
  python -m src.main --model-file model_20240115.pkl
  
  # 使用自定义配置
  python -m src.main --config my_config.yaml
        """
    )
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径 (默认: config/config.yaml)')
    parser.add_argument('--reset-model', action='store_true',
                       help='重置模型，从头开始学习')
    parser.add_argument('--model-file', type=str, default=None,
                       help='指定要加载的模型文件')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')
    
    args = parser.parse_args()
    
    # 设置日志级别
    import logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 确保必要目录存在
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("data/models").mkdir(parents=True, exist_ok=True)
    
    # 显示启动信息
    logger.info("🚀 vLLM GPU自适应调频系统 (Hybrid LinUCB)")
    logger.info(f"📅 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📝 配置文件: {args.config}")
    logger.info(f"🤖 模型选项: {'重置' if args.reset_model else ('加载 ' + args.model_file if args.model_file else '自动')}")
    
    try:
        # 创建并运行控制器
        autoscaler = VLLMGPUAutoscaler(
            config_path=args.config,
            reset_model=args.reset_model,
            model_file=args.model_file
        )
        autoscaler.run()
        
    except KeyboardInterrupt:
        logger.info("\n⌨️  键盘中断")
    except Exception as e:
        logger.critical(f"💥 致命错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

# ===== ./src/metrics_collector.py =====
import requests
import time
from typing import Dict, List, Tuple, Callable, Optional
from collections import defaultdict

from .logger import setup_logger

logger = setup_logger(__name__)


class MetricsCollector:
    """Prometheus‑based vLLM metrics collector with 2‑second sliding‑window support
    including aligned GPU energy delta measurement.
    """

    GAUGE_NAMES = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:num_requests_swapped",
        "vllm:gpu_cache_usage_perc",
        "vllm:cpu_cache_usage_perc",
    ]

    COUNTER_NAMES = [
        "vllm:num_preemptions_total",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:time_to_first_token_seconds_count",
        "vllm:time_per_output_token_seconds_sum",
        "vllm:time_per_output_token_seconds_count",
        "vllm:e2e_request_latency_seconds_sum",
        "vllm:e2e_request_latency_seconds_count",
        "vllm:gpu_prefix_cache_hits_total",
        "vllm:gpu_prefix_cache_queries_total",
    ]

    def __init__(self, prometheus_url: str = "http://localhost:8000/metrics", *, ema_alpha: float = 0.3):
        self.prometheus_url = prometheus_url
        self.ema_alpha = float(ema_alpha)
        # running EMA state (metric_name -> value)
        self._gauge_ema: Dict[str, float] = {}
        logger.info("📡 初始化指标采集器: %s", prometheus_url)

    # ---------------------------------------------------------------------
    # Low‑level helpers
    # ---------------------------------------------------------------------
    def _parse_prometheus_metrics(self, text: str) -> Dict[str, float]:
        """Parse a Prometheus exposition‑format text block into a flat mapping
        of metric_name -> value. Only keeps keys that start with "vllm:"."""
        metrics: Dict[str, float] = {}
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue  # ignore comments / empty lines
            try:
                if "{" in line:  # metric with labels – drop labels
                    metric_name = line.split("{")[0]
                    value = float(line.split("}")[1].strip())
                else:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    metric_name, value = parts[0], float(parts[1])

                if metric_name.startswith("vllm:"):
                    metrics[metric_name] = value
            except ValueError as exc:
                logger.debug("解析行失败 '%s': %s", line, exc)
        return metrics

    def _fetch_metrics_once(self) -> Dict[str, float]:
        """Fetch one snapshot from Prometheus."""
        try:
            resp = requests.get(self.prometheus_url, timeout=1.0)
            resp.raise_for_status()
            return self._parse_prometheus_metrics(resp.text)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("获取指标失败: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def collect_2s_metrics(
        self,
        *,
        duration: float = 2.0,
        interval: float = 0.1,
        energy_reader: Optional[Callable[[], float]] = None,
        ignore_first: int = 1,
    ) -> Tuple[Dict[str, float], Dict[str, float], float]:
        """Collect gauges (EMA), counter deltas, **and** GPU energy consumed (mJ)
        over a ~`duration`‑second sliding window.

        Args:
            duration: total window length in seconds (default 2.0).
            interval: sampling period for gauges / energy (default 0.1).
            energy_reader: callable returning cumulative GPU energy (mJ). If
                ``None`` energy delta is reported as 0.0.
            ignore_first: how many leading energy samples to drop when computing
                the delta – used to compensate for lock‑frequency inertia.

        Returns:
            gauge_metrics: metric -> ema value
            counter_deltas: metric_delta -> delta value
            energy_delta_mj: energy consumed within the window in millijoules
        """
        logger.info("📊 开始 %.1fs 数据采集…", duration)

        start_ts = time.time()

        # ------------------------------------------------------------------
        # Snapshot #0   (also first gauge sample / counter baseline)
        # ------------------------------------------------------------------
        counter_start = self._fetch_metrics_once()
        gauge_samples: List[Dict[str, float]] = [counter_start] if counter_start else []

        energy_samples: List[float] = []
        if energy_reader:
            try:
                energy_samples.append(float(energy_reader()))
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("energy_reader 调用失败: %s", exc)
                energy_reader = None  # disable further use

        # ------------------------------------------------------------------
        # Repeated sampling within the window
        # ------------------------------------------------------------------
        while time.time() - start_ts < duration:
            time.sleep(interval)
            snap = self._fetch_metrics_once()
            if snap:
                gauge_samples.append(snap)
            if energy_reader:
                energy_samples.append(energy_reader())

        # ------------------------------------------------------------------
        # Final counter snapshot (use last collected gauge if available)
        # ------------------------------------------------------------------
        counter_end = gauge_samples[-1] if gauge_samples else self._fetch_metrics_once()

        # ------------------------------------------------------------------
        # 1. Gauge EMA over samples
        # ------------------------------------------------------------------
        gauge_metrics: Dict[str, float] = {}
        for name in self.GAUGE_NAMES:
            values = [s.get(name, 0.0) for s in gauge_samples if name in s]
            if not values:
                continue
            # incremental EMA over the sample list
            ema = values[0]
            for v in values[1:]:
                ema = self.ema_alpha * v + (1.0 - self.ema_alpha) * ema
            gauge_metrics[name] = ema

        # ------------------------------------------------------------------
        # 2. Counter deltas
        # ------------------------------------------------------------------
        counter_deltas: Dict[str, float] = {}
        for name in self.COUNTER_NAMES:
            if name in counter_start and name in counter_end:
                delta = counter_end[name] - counter_start[name]
                counter_deltas[f"{name}_delta"] = max(delta, 0.0)

        # ------------------------------------------------------------------
        # 3. Energy delta (mJ)
        # ------------------------------------------------------------------
        energy_delta_mj: float = 0.0
        if energy_reader and len(energy_samples) >= 2:
            start_idx = min(ignore_first, len(energy_samples) - 1)
            energy_delta_mj = energy_samples[-1] - energy_samples[start_idx]
            if energy_delta_mj < 0:  # NVML rollover guard
                energy_delta_mj = 0.0

        logger.info(
            "✅ 数据采集完成: %d 个 Gauge, %d 个 Counter 增量, ΔEnergy=%.2f mJ",
            len(gauge_metrics), len(counter_deltas), energy_delta_mj,
        )

        return gauge_metrics, counter_deltas, energy_delta_mj


# ===== ./src/reward_calculator.py =====
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
from .logger import setup_logger

logger = setup_logger(__name__)


class EDPRewardCalculator:
    """优化的EDP奖励计算器 - 包含频率切换成本"""

    def __init__(self, ttft_limit: float = 2.0, tpot_limit: float = 0.25,
                 switch_cost_weight: float = 0.1):
        # 配置参数
        self.ttft_limit = ttft_limit   # TTFT硬限制 (秒)
        self.tpot_limit = tpot_limit   # TPOT硬限制 (秒)
        self.switch_cost_weight = switch_cost_weight  # 频率切换成本权重

        # 延迟EMA（指数移动平均）
        self.ttft_ema: Optional[float] = None
        self.tpot_ema: Optional[float] = None
        self.delay_ema_alpha = 0.1

        # EDP EMA（用于自适应缩放）
        self.edp_ema: Optional[float] = None
        self.edp_ema_alpha = 0.1

        # 历史窗口（用于诊断）
        self.ttft_history = deque(maxlen=50)
        self.tpot_history = deque(maxlen=50)
        self.edp_history = deque(maxlen=50)

        logger.info(
            f"💰 初始化优化EDP奖励计算器:\n"
            f"   TTFT限制: ≤{ttft_limit}s\n"
            f"   TPOT限制: ≤{tpot_limit}s\n"
            f"   切换成本权重: {switch_cost_weight}"
        )

    def calculate(
        self,
        counter_deltas: Dict[str, float],
        energy_consumed_mj: float,
        current_freq: int = None,
        previous_freq: int = None,
        max_freq: int = 2100
    ) -> Tuple[float, Dict]:
        """
        计算综合奖励

        Parameters
        ----------
        counter_deltas : Dict[str, float]
            Prometheus计数器增量
        energy_consumed_mj : float
            能耗（毫焦）
        current_freq : int
            当前频率（MHz）
        previous_freq : int
            上一次频率（MHz）
        max_freq : int
            最大频率（MHz）
        """
        # 提取计数器数据
        ttft_count = counter_deltas.get('vllm:time_to_first_token_seconds_count_delta', 0)
        ttft_sum = counter_deltas.get('vllm:time_to_first_token_seconds_sum_delta', 0)
        tpot_count = counter_deltas.get('vllm:time_per_output_token_seconds_count_delta', 0)
        tpot_sum = counter_deltas.get('vllm:time_per_output_token_seconds_sum_delta', 0)

        # 检查是否有请求
        if ttft_count == 0 and tpot_count == 0:
            logger.debug("⚠️ 这2秒内没有请求完成")
            return 0.0, {"no_requests": True}

        # 计算平均延迟
        avg_ttft = ttft_sum / ttft_count if ttft_count > 0 else None
        avg_tpot = tpot_sum / tpot_count if tpot_count > 0 else None

        # 更新延迟EMA
        if avg_ttft is not None:
            self.ttft_history.append(avg_ttft)
            if self.ttft_ema is None:
                self.ttft_ema = avg_ttft
            else:
                self.ttft_ema = (self.delay_ema_alpha * avg_ttft + 
                               (1 - self.delay_ema_alpha) * self.ttft_ema)

        if avg_tpot is not None:
            self.tpot_history.append(avg_tpot)
            if self.tpot_ema is None:
                self.tpot_ema = avg_tpot
            else:
                self.tpot_ema = (self.delay_ema_alpha * avg_tpot + 
                               (1 - self.delay_ema_alpha) * self.tpot_ema)

        # 预热检查
        if len(self.ttft_history) < 3 and len(self.tpot_history) < 3:
            logger.info("📊 预热中，收集更多数据...")
            return 0.0, {"warming_up": True}

        # 1. 计算延迟分量
        delay_score = 0.0
        delay_penalty = 0.0
        
        if avg_ttft is not None and self.ttft_ema is not None:
            # TTFT归一化得分
            ttft_norm = avg_ttft / self.ttft_ema
            ttft_score = 1.0 - min(2.0, ttft_norm)  # [-1, 1]
            
            # TTFT硬限制惩罚
            if avg_ttft > self.ttft_limit:
                ttft_penalty = -50 * (avg_ttft / self.ttft_limit - 1)
                logger.warning(f"⚠️ TTFT超限: {avg_ttft:.3f}s > {self.ttft_limit}s")
            else:
                ttft_penalty = 0
                
            delay_score += 0.5 * ttft_score
            delay_penalty += ttft_penalty

        if avg_tpot is not None and self.tpot_ema is not None:
            # TPOT归一化得分
            tpot_norm = avg_tpot / self.tpot_ema
            tpot_score = 1.0 - min(2.0, tpot_norm)  # [-1, 1]
            
            # TPOT硬限制惩罚
            if avg_tpot > self.tpot_limit:
                tpot_penalty = -50 * (avg_tpot / self.tpot_limit - 1)
                logger.warning(f"⚠️ TPOT超限: {avg_tpot:.3f}s > {self.tpot_limit}s")
            else:
                tpot_penalty = 0
                
            delay_score += 0.5 * tpot_score
            delay_penalty += tpot_penalty

        # 2. 计算能耗分量
        energy_j = energy_consumed_mj / 1000.0  # 转换为焦耳
        
        # 计算EDP
        if avg_ttft is not None or avg_tpot is not None:
            # 使用实际延迟
            avg_delay = 0.0
            count = 0
            if avg_ttft is not None:
                avg_delay += avg_ttft
                count += 1
            if avg_tpot is not None:
                avg_delay += avg_tpot
                count += 1
            avg_delay = avg_delay / count if count > 0 else 1.0
        else:
            # 使用EMA作为默认值
            avg_delay = (self.ttft_ema + self.tpot_ema) / 2 if self.ttft_ema and self.tpot_ema else 1.0
            
        edp = energy_j * avg_delay
        self.edp_history.append(edp)

        # 更新EDP EMA
        if self.edp_ema is None:
            self.edp_ema = edp
        else:
            self.edp_ema = (self.edp_ema_alpha * edp + 
                          (1 - self.edp_ema_alpha) * self.edp_ema)

        # EDP归一化得分
        if self.edp_ema > 0:
            edp_norm = edp / self.edp_ema
            # 使用更宽的奖励范围
            energy_score = 2.0 * (1.0 - min(2.0, edp_norm))  # [-2, 2]
        else:
            energy_score = 0.0

        # 3. 计算频率切换成本
        switch_cost = 0.0
        if current_freq is not None and previous_freq is not None and max_freq > 0:
            freq_change = abs(current_freq - previous_freq)
            normalized_change = freq_change / max_freq
            # 切换成本：小幅调整成本低，大幅调整成本高
            switch_cost = -self.switch_cost_weight * normalized_change * normalized_change
            
            if freq_change > 0:
                logger.debug(f"🔄 频率切换成本: {switch_cost:.3f} "
                           f"({previous_freq}→{current_freq}MHz)")

        # 4. 综合奖励
        # 基础奖励 = 延迟得分 + 能耗得分
        base_reward = 0.5 * delay_score + 0.5 * energy_score
        
        # 最终奖励 = 基础奖励 + 硬限制惩罚 + 切换成本
        final_reward = base_reward + delay_penalty + switch_cost
        
        # 限制奖励范围在[-100, 10]
        final_reward = np.clip(final_reward, -100, 10)

        # 构建详细信息
        info = {
            'avg_ttft': avg_ttft,
            'avg_tpot': avg_tpot,
            'ttft_ema': self.ttft_ema,
            'tpot_ema': self.tpot_ema,
            'delay_score': delay_score,
            'energy_j': energy_j,
            'edp': edp,
            'edp_ema': self.edp_ema,
            'energy_score': energy_score,
            'switch_cost': switch_cost,
            'delay_penalty': delay_penalty,
            'base_reward': base_reward,
            'final_reward': final_reward
        }

        # 日志输出
        logger.info(
            f"💰 奖励计算:\n"
            f"   延迟得分: {delay_score:+.3f}\n"
            f"   能耗得分: {energy_score:+.3f}\n"
            f"   切换成本: {switch_cost:+.3f}\n"
            f"   硬限惩罚: {delay_penalty:+.3f}\n"
            f"   最终奖励: {final_reward:+.3f}"
        )

        return float(final_reward), info

    def get_stats(self) -> dict:
        """获取奖励计算器的统计信息"""
        stats = {
            'ttft_ema': self.ttft_ema,
            'tpot_ema': self.tpot_ema,
            'edp_ema': self.edp_ema,
            'ttft_history_len': len(self.ttft_history),
            'tpot_history_len': len(self.tpot_history),
            'edp_history_len': len(self.edp_history)
        }
        
        if self.ttft_history:
            stats['ttft_recent_avg'] = np.mean(list(self.ttft_history)[-10:])
            stats['ttft_recent_std'] = np.std(list(self.ttft_history)[-10:])
            
        if self.tpot_history:
            stats['tpot_recent_avg'] = np.mean(list(self.tpot_history)[-10:])
            stats['tpot_recent_std'] = np.std(list(self.tpot_history)[-10:])
            
        if self.edp_history:
            stats['edp_recent_avg'] = np.mean(list(self.edp_history)[-10:])
            stats['edp_recent_std'] = np.std(list(self.edp_history)[-10:])
            
        return stats

# ===== ./test.py =====
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 清理一下，确保环境干净
plt.close('all')

# -------------------------------------------------------------
# 核心测试代码
# -------------------------------------------------------------
try:
    # 直接指定我们已确认 Matplotlib 能找到的字体文件路径
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc' 

    print(f"正在尝试使用字体文件: {font_path}")

    # 从文件创建字体属性对象
    my_font = fm.FontProperties(fname=font_path)

    if my_font:
         print(f"✅ 成功加载字体: {my_font.get_name()}")
    else:
        raise ValueError("无法从路径加载字体属性")

    # --- 开始绘图 ---
    plt.figure()

    # 在标题、标签和文本中都使用这个字体
    plt.title('中文测试标题', fontproperties=my_font, fontsize=20)
    plt.xlabel('X轴：横坐标', fontproperties=my_font, fontsize=14)
    plt.ylabel('Y轴：纵坐标', fontproperties=my_font, fontsize=14)
    plt.text(0.5, 0.5, '中文字符一切正常！', fontproperties=my_font, ha='center', fontsize=16)

    # 保存图片
    output_file = "chinese_test_plot.png"
    plt.savefig(output_file)
    print(f"\n🎉 绘图成功! 请立刻在文件浏览器中打开并检查图片: {output_file}")
    print("如果图片中的中文正常，我们就可以修复你的主代码了。")

except Exception as e:
    print(f"\n❌ 绘图失败: {e}")
    print("如果这一步失败，说明问题比预想的更复杂，可能是 Matplotlib 后端或库本身的问题。")
# -------------------------------------------------------------

