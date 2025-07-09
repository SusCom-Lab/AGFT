#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM GPU Autoscaler 数据分析工具 - 针对新JSON日志格式重构
Refactored Data Analysis Tool for New JSON Log Format

此工具专门针对新的JSON结构化日志格式设计，提供全面的在线学习效果评估。
"""

import os
import re
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from glob import glob
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib environment
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.makedirs('/tmp/matplotlib_cache', exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import Counter, defaultdict

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return super().default(obj)

# Setup matplotlib with font configuration
def setup_display():
    """Setup matplotlib with font configuration"""
    try:
        user_font_path = '/home/ldaphome/colin/.local/share/fonts/NotoSansCJKsc-Regular.otf'
        if os.path.exists(user_font_path):
            fm.fontManager.addfont(user_font_path)
            plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'sans-serif']
        else:
            plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    except:
        plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.max_open_warning'] = 0
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 100
    
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')

setup_display()

class VLLMAutoscalerAnalyzer:
    """vLLM GPU Autoscaler 数据分析器 - 新JSON格式"""
    
    def __init__(self, log_dir: str = "logs", output_dir: str = "data/analysis"):
        self.log_dir = log_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔍 初始化数据分析器 (JSON格式)...")
        print(f"   日志目录: {log_dir}")
        print(f"   输出目录: {output_dir}")
    
    def find_latest_log(self, gpu_model: str = None) -> Optional[str]:
        """查找最新的日志文件"""
        if gpu_model:
            gpu_log_dir = os.path.join(self.log_dir, gpu_model)
            if os.path.exists(gpu_log_dir):
                log_files = sorted(glob(os.path.join(gpu_log_dir, "*.log")))
            else:
                print(f"⚠️ GPU目录不存在: {gpu_log_dir}")
                return None
        else:
            log_files = []
            gpu_dirs = [d for d in os.listdir(self.log_dir) 
                       if os.path.isdir(os.path.join(self.log_dir, d)) and not d.startswith('.')]
            
            for gpu_dir in gpu_dirs:
                gpu_log_dir = os.path.join(self.log_dir, gpu_dir)
                gpu_logs = glob(os.path.join(gpu_log_dir, "*.log"))
                log_files.extend(gpu_logs)
        
        if not log_files:
            print("❌ 未找到日志文件")
            return None
        
        return max(log_files, key=os.path.getmtime)
    
    def parse_log(self, log_file_path: str) -> pd.DataFrame:
        """解析新JSON格式的日志文件"""
        print(f"📖 解析JSON格式日志文件: {os.path.basename(log_file_path)}")
        
        if not os.path.exists(log_file_path):
            print(f"❌ 日志文件不存在: {log_file_path}")
            return pd.DataFrame()
        
        parsed_data = []
        system_info = {}
        mode_switch_events = []  # 记录模式切换事件
        
        # JSON数据提取模式
        json_pattern = r'📋 JSON数据: ({.*})'
        
        # 系统信息提取模式
        info_patterns = {
            'gpu_model': r'🔧 GPU型号: (.+)',
            'algorithm': r'算法: ([^(]+)',
            'feature_dims': r'特征维度: (\d+)维',
            'convergence_window': r'收敛窗口: (\d+)轮'
        }
        
        # 新增模式切换检测模式
        mode_switch_patterns = {
            'convergence_detected': r'🎯 模型收敛检测: 主导动作(\d+)MHz占比([\d.]+)% >= (\d+)%',
            'switch_to_exploitation': r'🔄 从学习模式切换到利用模式',
            'performance_degradation': r'⚠️ 性能退化检测: 最近50轮EDP平均值\(([\d.]+)\)',
            'switch_to_learning': r'🔄 (.+)，从利用模式切换回学习模式',
            'adaptive_sampler_skip': r'🔒 利用模式下跳过自适应采样器更新',
            'exploitation_to_learning': r'⚠️.*从利用模式切换回学习模式',
            'convergence_unstable': r'⚠️ 收敛状态不稳定.*主导动作占比降至([\d.]+)%'
        }
        
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取系统信息
            for key, pattern in info_patterns.items():
                match = re.search(pattern, content)
                if match:
                    if key == 'feature_dims' or key == 'convergence_window':
                        system_info[key] = int(match.group(1))
                    else:
                        system_info[key] = match.group(1).strip()
            
            # 提取模式切换事件
            for event_type, pattern in mode_switch_patterns.items():
                for line in content.split('\n'):
                    if re.search(pattern, line):
                        # 提取时间戳
                        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                        if timestamp_match:
                            event_info = {
                                'timestamp': pd.to_datetime(timestamp_match.group(1)),
                                'event_type': event_type,
                                'description': line.strip()
                            }
                            mode_switch_events.append(event_info)
            
            print(f"   检测到 {len(mode_switch_events)} 个模式切换事件")
            
            # 提取所有JSON数据
            json_matches = re.findall(json_pattern, content)
            print(f"   找到 {len(json_matches)} 条JSON记录")
            
            for json_str in json_matches:
                try:
                    # 解析JSON数据
                    json_data = json.loads(json_str)
                    
                    # 构造扁平化的数据条目
                    data_entry = {
                        # 基础信息
                        'timestamp': pd.to_datetime(json_data.get('timestamp')),
                        'round': json_data.get('round'),
                        
                        # 决策信息
                        'selected_frequency': json_data.get('decision', {}).get('selected_frequency'),
                        'selection_count': json_data.get('decision', {}).get('selection_count'),
                        'mode': json_data.get('decision', {}).get('mode'),
                        'available_frequencies': json_data.get('decision', {}).get('available_frequencies', []),
                        'n_available_frequencies': len(json_data.get('decision', {}).get('available_frequencies', [])),
                        
                        # 特征向量 (7维)
                        'has_queue': json_data.get('features', {}).get('has_queue'),
                        'prefill_throughput': json_data.get('features', {}).get('prefill_throughput'),
                        'decode_throughput': json_data.get('features', {}).get('decode_throughput'),
                        'packing_efficiency': json_data.get('features', {}).get('packing_efficiency'),
                        'concurrency': json_data.get('features', {}).get('concurrency'),
                        'gpu_cache_usage': json_data.get('features', {}).get('gpu_cache_usage'),
                        'cache_hit_rate': json_data.get('features', {}).get('cache_hit_rate'),
                        
                        # 性能指标
                        'avg_ttft': json_data.get('performance', {}).get('avg_ttft'),
                        'avg_tpot': json_data.get('performance', {}).get('avg_tpot'),
                        'ttft_count': json_data.get('performance', {}).get('ttft_count'),
                        'tpot_count': json_data.get('performance', {}).get('tpot_count'),
                        'energy_delta_mj': json_data.get('performance', {}).get('energy_delta_mj'),
                        'energy_delta_j': json_data.get('performance', {}).get('energy_delta_j'),
                        'edp_value': json_data.get('performance', {}).get('edp_value'),
                        
                        # 奖励信息
                        'current_reward': json_data.get('reward', {}).get('current'),
                        'average_reward': json_data.get('reward', {}).get('average'),
                        'recent_average_reward': json_data.get('reward', {}).get('recent_average'),
                        
                        # 系统状态
                        'gpu_utilization': json_data.get('system', {}).get('gpu_utilization'),
                        'gpu_memory_mb': json_data.get('system', {}).get('gpu_memory_mb'),
                        'queue_size': json_data.get('system', {}).get('queue_size'),
                        
                        # 模型状态
                        'total_rounds': json_data.get('model', {}).get('total_rounds'),
                        'n_arms': json_data.get('model', {}).get('n_arms'),
                        'converged': json_data.get('model', {}).get('converged'),
                        'exploitation_mode': json_data.get('model', {}).get('exploitation_mode'),
                        
                        # 添加系统信息
                        **system_info
                    }
                    
                    # 计算衍生指标
                    # 处理新的模式字段
                    mode = json_data.get('decision', {}).get('mode', 'exploration')
                    if mode == 'exploitation' or data_entry['exploitation_mode']:
                        data_entry['phase'] = 'EXPLOITATION'
                    else:
                        data_entry['phase'] = 'LEARNING'
                    data_entry['energy_efficiency'] = 1.0 / max(data_entry['edp_value'], 1e-6) if data_entry['edp_value'] else 0
                    data_entry['total_latency'] = (data_entry['avg_ttft'] or 0) + (data_entry['avg_tpot'] or 0)
                    data_entry['throughput_ratio'] = (data_entry['decode_throughput'] or 0) / max(data_entry['prefill_throughput'] or 1, 1e-6)
                    
                    parsed_data.append(data_entry)
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON解析失败: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️ 数据处理失败: {e}")
                    continue
        
        except Exception as e:
            print(f"❌ 解析日志文件时出错: {e}")
            return pd.DataFrame()
        
        if not parsed_data:
            print("⚠️ 未找到有效的JSON数据")
            return pd.DataFrame()
        
        df = pd.DataFrame(parsed_data)
        
        # 存储模式切换事件信息
        self.mode_switch_events = mode_switch_events
        
        # 数据清理和排序
        df = df.sort_values('round').reset_index(drop=True)
        
        print(f"   解析完成: {len(df)} 条记录")
        print(f"   时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
        print(f"   轮次范围: {df['round'].min()} - {df['round'].max()}")
        print(f"   系统信息: {system_info}")
        
        return df
    
    def analyze_learning_effectiveness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析在线学习效果 - 核心分析方法"""
        print("🧠 分析在线学习效果...")
        
        if df.empty:
            return {}
        
        analysis = {}
        
        # 1. 基础统计
        analysis['basic_stats'] = {
            'total_rounds': len(df),
            'learning_rounds': len(df[df['phase'] == 'LEARNING']),
            'exploitation_rounds': len(df[df['phase'] == 'EXPLOITATION']),
            'unique_frequencies_tried': df['selected_frequency'].nunique(),
            'total_frequencies_available': df['n_available_frequencies'].iloc[-1] if not df.empty else 0,
            'algorithm': df['algorithm'].iloc[0] if 'algorithm' in df.columns else 'Unknown',
            'gpu_model': df['gpu_model'].iloc[0] if 'gpu_model' in df.columns else 'Unknown'
        }
        
        # 2. 收敛分析
        analysis['convergence_analysis'] = self._analyze_convergence(df)
        
        # 3. 探索-利用权衡分析
        analysis['exploration_exploitation'] = self._analyze_exploration_exploitation(df)
        
        # 4. 奖励学习分析
        analysis['reward_learning'] = self._analyze_reward_learning(df)
        
        # 5. 频率偏好学习分析
        analysis['frequency_preference'] = self._analyze_frequency_preference(df)
        
        # 6. 特征重要性分析
        analysis['feature_analysis'] = self._analyze_feature_patterns(df)
        
        # 7. 性能优化效果分析
        analysis['performance_optimization'] = self._analyze_performance_optimization(df)
        
        # 8. 能效分析
        analysis['energy_efficiency'] = self._analyze_energy_efficiency(df)
        
        # 9. 自适应能力分析
        analysis['adaptability'] = self._analyze_adaptability(df)
        
        # 10. 在线学习质量评估
        analysis['learning_quality'] = self._evaluate_learning_quality(df)
        
        # 11. 模式切换事件分析
        analysis['mode_switching'] = self._analyze_mode_switching_events(df)
        
        return analysis
    
    def _analyze_convergence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析收敛行为"""
        convergence = {}
        
        # 收敛检测
        learning_df = df[df['phase'] == 'LEARNING']
        exploitation_df = df[df['phase'] == 'EXPLOITATION']
        
        convergence['convergence_achieved'] = len(exploitation_df) > 0
        convergence['rounds_to_convergence'] = len(learning_df) if convergence['convergence_achieved'] else None
        convergence['convergence_ratio'] = len(exploitation_df) / len(df) * 100 if len(df) > 0 else 0
        
        # 收敛稳定性分析
        if len(exploitation_df) > 0:
            # 利用阶段的频率稳定性
            exploit_freqs = exploitation_df['selected_frequency'].value_counts()
            convergence['dominant_frequency'] = exploit_freqs.index[0] if not exploit_freqs.empty else None
            convergence['frequency_stability'] = exploit_freqs.iloc[0] / len(exploitation_df) * 100 if not exploit_freqs.empty else 0
            
            # 奖励稳定性
            exploit_rewards = exploitation_df['current_reward']
            convergence['reward_stability'] = {
                'mean': exploit_rewards.mean(),
                'std': exploit_rewards.std(),
                'coefficient_of_variation': exploit_rewards.std() / max(abs(exploit_rewards.mean()), 1e-6)
            }
        
        return convergence
    
    def _analyze_exploration_exploitation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析探索-利用权衡"""
        exp_exp = {}
        
        # 频率选择多样性分析
        freq_counts = df['selected_frequency'].value_counts()
        total_selections = len(df)
        
        exp_exp['frequency_diversity'] = {
            'unique_frequencies': len(freq_counts),
            'gini_coefficient': self._calculate_gini_coefficient(freq_counts.values),
            'entropy': self._calculate_entropy(freq_counts.values),
            'most_selected_frequency': freq_counts.index[0] if not freq_counts.empty else None,
            'most_selected_ratio': freq_counts.iloc[0] / total_selections * 100 if not freq_counts.empty else 0
        }
        
        # 探索模式分析
        learning_df = df[df['phase'] == 'LEARNING']
        if not learning_df.empty:
            learning_freq_counts = learning_df['selected_frequency'].value_counts()
            exp_exp['exploration_pattern'] = {
                'frequencies_explored': len(learning_freq_counts),
                'exploration_uniformity': 1.0 - self._calculate_gini_coefficient(learning_freq_counts.values),
                'average_explorations_per_frequency': learning_freq_counts.mean()
            }
        
        # 利用模式分析
        exploitation_df = df[df['phase'] == 'EXPLOITATION']
        if not exploitation_df.empty:
            exploit_freq_counts = exploitation_df['selected_frequency'].value_counts()
            exp_exp['exploitation_pattern'] = {
                'exploitation_concentration': exploit_freq_counts.iloc[0] / len(exploitation_df) * 100 if not exploit_freq_counts.empty else 0,
                'frequencies_used_in_exploitation': len(exploit_freq_counts)
            }
        
        return exp_exp
    
    def _analyze_reward_learning(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析奖励学习效果"""
        reward_learning = {}
        
        if 'current_reward' in df.columns:
            rewards = df['current_reward'].dropna()
            
            # 奖励趋势分析
            reward_learning['trend_analysis'] = {
                'initial_reward': rewards.iloc[0] if not rewards.empty else None,
                'final_reward': rewards.iloc[-1] if not rewards.empty else None,
                'reward_improvement': rewards.iloc[-1] - rewards.iloc[0] if len(rewards) > 0 else 0,
                'average_reward': rewards.mean(),
                'reward_volatility': rewards.std()
            }
            
            # 学习速度分析
            if len(rewards) > 10:
                # 计算移动平均来评估学习速度
                window_size = min(20, len(rewards) // 4)
                moving_avg = rewards.rolling(window=window_size).mean()
                reward_learning['learning_speed'] = {
                    'window_size': window_size,
                    'early_avg': moving_avg.iloc[window_size-1] if len(moving_avg) >= window_size else None,
                    'late_avg': moving_avg.iloc[-1] if not moving_avg.empty else None,
                    'improvement_rate': (moving_avg.iloc[-1] - moving_avg.iloc[window_size-1]) / window_size if len(moving_avg) >= window_size else 0
                }
        
        return reward_learning
    
    def _analyze_frequency_preference(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析频率偏好学习"""
        freq_pref = {}
        
        # 频率选择统计
        freq_counts = df['selected_frequency'].value_counts()
        freq_rewards = df.groupby('selected_frequency')['current_reward'].agg(['count', 'mean', 'std']).fillna(0)
        
        freq_pref['selection_statistics'] = {
            'frequency_counts': freq_counts.to_dict(),
            'frequency_rewards': freq_rewards.to_dict('index')
        }
        
        # 最优频率识别
        best_freq_by_reward = freq_rewards['mean'].idxmax() if not freq_rewards.empty else None
        most_selected_freq = freq_counts.index[0] if not freq_counts.empty else None
        
        freq_pref['optimal_frequency_analysis'] = {
            'best_frequency_by_reward': best_freq_by_reward,
            'best_average_reward': freq_rewards.loc[best_freq_by_reward, 'mean'] if best_freq_by_reward else None,
            'most_selected_frequency': most_selected_freq,
            'most_selected_count': freq_counts.iloc[0] if not freq_counts.empty else None,
            'optimal_vs_preferred_match': best_freq_by_reward == most_selected_freq
        }
        
        # EDP (能效) 分析
        if 'edp_value' in df.columns:
            freq_edp = df.groupby('selected_frequency')['edp_value'].agg(['count', 'mean', 'std']).fillna(0)
            best_freq_by_edp = freq_edp['mean'].idxmin() if not freq_edp.empty else None  # EDP越小越好
            
            freq_pref['energy_efficiency_analysis'] = {
                'best_frequency_by_edp': best_freq_by_edp,
                'best_edp_value': freq_edp.loc[best_freq_by_edp, 'mean'] if best_freq_by_edp else None,
                'frequency_edp_stats': freq_edp.to_dict('index')
            }
        
        return freq_pref
    
    def _analyze_feature_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析特征模式和重要性"""
        feature_analysis = {}
        
        # 7维特征分析
        feature_cols = ['has_queue', 'prefill_throughput', 'decode_throughput', 
                       'packing_efficiency', 'concurrency', 'gpu_cache_usage', 'cache_hit_rate']
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        if available_features:
            feature_stats = df[available_features].describe()
            feature_analysis['feature_statistics'] = feature_stats.to_dict()
            
            # 特征相关性分析
            feature_correlation = df[available_features + ['current_reward']].corr()['current_reward'].drop('current_reward')
            feature_analysis['feature_reward_correlation'] = feature_correlation.to_dict()
            
            # 特征变化范围分析
            feature_ranges = {}
            for feature in available_features:
                feature_data = df[feature].dropna()
                if not feature_data.empty:
                    feature_ranges[feature] = {
                        'min': feature_data.min(),
                        'max': feature_data.max(),
                        'range': feature_data.max() - feature_data.min(),
                        'std': feature_data.std()
                    }
            
            feature_analysis['feature_ranges'] = feature_ranges
        
        return feature_analysis
    
    def _analyze_performance_optimization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析性能优化效果"""
        perf_opt = {}
        
        # TTFT (Time To First Token) 优化
        if 'avg_ttft' in df.columns:
            ttft_data = df['avg_ttft'].dropna()
            if not ttft_data.empty:
                perf_opt['ttft_optimization'] = {
                    'initial_ttft': ttft_data.iloc[0],
                    'final_ttft': ttft_data.iloc[-1],
                    'ttft_improvement': ttft_data.iloc[0] - ttft_data.iloc[-1],  # 减少是改善
                    'ttft_improvement_ratio': (ttft_data.iloc[0] - ttft_data.iloc[-1]) / ttft_data.iloc[0] * 100 if ttft_data.iloc[0] != 0 else 0,
                    'average_ttft': ttft_data.mean(),
                    'ttft_volatility': ttft_data.std()
                }
        
        # TPOT (Time Per Output Token) 优化
        if 'avg_tpot' in df.columns:
            tpot_data = df['avg_tpot'].dropna()
            if not tpot_data.empty:
                perf_opt['tpot_optimization'] = {
                    'initial_tpot': tpot_data.iloc[0],
                    'final_tpot': tpot_data.iloc[-1],
                    'tpot_improvement': tpot_data.iloc[0] - tpot_data.iloc[-1],  # 减少是改善
                    'tpot_improvement_ratio': (tpot_data.iloc[0] - tpot_data.iloc[-1]) / tpot_data.iloc[0] * 100 if tpot_data.iloc[0] != 0 else 0,
                    'average_tpot': tpot_data.mean(),
                    'tpot_volatility': tpot_data.std()
                }
        
        # 总延迟优化
        if 'total_latency' in df.columns:
            latency_data = df['total_latency'].dropna()
            if not latency_data.empty:
                perf_opt['latency_optimization'] = {
                    'initial_latency': latency_data.iloc[0],
                    'final_latency': latency_data.iloc[-1],
                    'latency_improvement': latency_data.iloc[0] - latency_data.iloc[-1],
                    'latency_improvement_ratio': (latency_data.iloc[0] - latency_data.iloc[-1]) / latency_data.iloc[0] * 100 if latency_data.iloc[0] != 0 else 0
                }
        
        return perf_opt
    
    def _analyze_energy_efficiency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析能效优化"""
        energy_eff = {}
        
        # EDP (Energy-Delay Product) 分析
        if 'edp_value' in df.columns:
            edp_data = df['edp_value'].dropna()
            if not edp_data.empty:
                energy_eff['edp_optimization'] = {
                    'initial_edp': edp_data.iloc[0],
                    'final_edp': edp_data.iloc[-1],
                    'edp_improvement': edp_data.iloc[0] - edp_data.iloc[-1],  # EDP减少是改善
                    'edp_improvement_ratio': (edp_data.iloc[0] - edp_data.iloc[-1]) / edp_data.iloc[0] * 100 if edp_data.iloc[0] != 0 else 0,
                    'average_edp': edp_data.mean(),
                    'best_edp': edp_data.min(),
                    'worst_edp': edp_data.max(),
                    'edp_volatility': edp_data.std()
                }
        
        # 能耗分析
        if 'energy_delta_j' in df.columns:
            energy_data = df['energy_delta_j'].dropna()
            if not energy_data.empty:
                energy_eff['energy_consumption'] = {
                    'average_energy_per_round': energy_data.mean(),
                    'total_energy_consumed': energy_data.sum(),
                    'energy_efficiency_trend': self._calculate_trend(energy_data),
                    'min_energy': energy_data.min(),
                    'max_energy': energy_data.max()
                }
        
        # 能效指标
        if 'energy_efficiency' in df.columns:
            eff_data = df['energy_efficiency'].dropna()
            if not eff_data.empty:
                energy_eff['efficiency_metrics'] = {
                    'initial_efficiency': eff_data.iloc[0],
                    'final_efficiency': eff_data.iloc[-1],
                    'efficiency_improvement': eff_data.iloc[-1] - eff_data.iloc[0],
                    'efficiency_improvement_ratio': (eff_data.iloc[-1] - eff_data.iloc[0]) / eff_data.iloc[0] * 100 if eff_data.iloc[0] != 0 else 0,
                    'average_efficiency': eff_data.mean()
                }
        
        return energy_eff
    
    def _analyze_adaptability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析自适应能力"""
        adaptability = {}
        
        # 工作负载变化适应性
        feature_cols = ['has_queue', 'prefill_throughput', 'decode_throughput', 
                       'packing_efficiency', 'concurrency', 'gpu_cache_usage', 'cache_hit_rate']
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        if available_features:
            # 计算特征变化幅度
            feature_changes = {}
            for feature in available_features:
                feature_data = df[feature].dropna()
                if len(feature_data) > 1:
                    # 计算相邻值之间的变化
                    changes = abs(feature_data.diff()).dropna()
                    feature_changes[feature] = {
                        'average_change': changes.mean(),
                        'max_change': changes.max(),
                        'change_frequency': len(changes[changes > 0]) / len(changes) * 100 if len(changes) > 0 else 0
                    }
            
            adaptability['workload_adaptability'] = feature_changes
        
        # 频率调整适应性
        freq_changes = df['selected_frequency'].diff().dropna()
        if not freq_changes.empty:
            adaptability['frequency_adaptability'] = {
                'total_frequency_changes': len(freq_changes[freq_changes != 0]),
                'frequency_change_rate': len(freq_changes[freq_changes != 0]) / len(df) * 100 if len(df) > 0 else 0,
                'average_frequency_change': abs(freq_changes[freq_changes != 0]).mean() if len(freq_changes[freq_changes != 0]) > 0 else 0
            }
        
        return adaptability
    
    def _evaluate_learning_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评估在线学习质量"""
        quality = {}
        
        # 学习效率评估
        learning_rounds = len(df[df['phase'] == 'LEARNING'])
        total_rounds = len(df)
        
        quality['learning_efficiency'] = {
            'learning_rounds': learning_rounds,
            'total_rounds': total_rounds,
            'learning_efficiency_ratio': learning_rounds / total_rounds * 100 if total_rounds > 0 else 0,
            'convergence_speed': 'Fast' if learning_rounds < total_rounds * 0.3 else ('Medium' if learning_rounds < total_rounds * 0.7 else 'Slow')
        }
        
        # 决策质量评估
        if 'current_reward' in df.columns:
            rewards = df['current_reward'].dropna()
            if not rewards.empty:
                positive_rewards = len(rewards[rewards > 0])
                quality['decision_quality'] = {
                    'positive_reward_ratio': positive_rewards / len(rewards) * 100,
                    'reward_consistency': 1.0 - (rewards.std() / max(abs(rewards.mean()), 1e-6)),
                    'learning_stability': 'Stable' if rewards.std() < 0.1 else 'Volatile'
                }
        
        # 算法性能评估
        if 'edp_value' in df.columns:
            edp_data = df['edp_value'].dropna()
            if len(edp_data) > 1:
                edp_trend = self._calculate_trend(edp_data)  # 负趋势表示EDP在减少（改善）
                quality['algorithm_performance'] = {
                    'edp_trend': edp_trend,
                    'optimization_effectiveness': 'Excellent' if edp_trend < -0.1 else ('Good' if edp_trend < 0 else 'Poor'),
                    'performance_improvement': 'Yes' if edp_trend < 0 else 'No'
                }
        
        return quality
    
    def _calculate_gini_coefficient(self, values):
        """计算基尼系数"""
        if len(values) == 0:
            return 0
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
    
    def _calculate_entropy(self, values):
        """计算信息熵"""
        if len(values) == 0:
            return 0
        total = sum(values)
        probabilities = [v / total for v in values if v > 0]
        return -sum(p * np.log2(p) for p in probabilities)
    
    def _calculate_trend(self, data):
        """计算数据趋势 (线性回归斜率)"""
        if len(data) < 2:
            return 0
        x = np.arange(len(data))
        return np.polyfit(x, data, 1)[0]
    
    def create_comprehensive_visualizations(self, df: pd.DataFrame, analysis: Dict[str, Any]):
        """创建全面的可视化图表"""
        print("📈 生成全面可视化图表...")
        
        if df.empty:
            print("⚠️ 没有数据可视化")
            return
        
        # 创建主要分析图表
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. 奖励学习曲线
        ax1 = fig.add_subplot(gs[0, 0:2])
        self._plot_reward_learning_curve(df, ax1)
        
        # 2. 频率选择模式
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_frequency_selection_pattern(df, ax2)
        
        # 3. EDP优化趋势
        ax3 = fig.add_subplot(gs[1, 0:2])
        self._plot_edp_optimization(df, ax3)
        
        # 4. 探索-利用阶段分布
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_exploration_exploitation_phases(df, ax4)
        
        # 5. 特征重要性热力图
        ax5 = fig.add_subplot(gs[1, 3])
        self._plot_feature_importance_heatmap(df, analysis, ax5)
        
        # 6. 性能指标对比
        ax6 = fig.add_subplot(gs[2, 0:2])
        self._plot_performance_metrics(df, ax6)
        
        # 7. 频率-奖励关系
        ax7 = fig.add_subplot(gs[2, 2:])
        self._plot_frequency_reward_relationship(df, ax7)
        
        # 8. 能效分析
        ax8 = fig.add_subplot(gs[3, 0:2])
        self._plot_energy_efficiency_analysis(df, ax8)
        
        # 9. 学习质量评估
        ax9 = fig.add_subplot(gs[3, 2:])
        self._plot_learning_quality_assessment(df, analysis, ax9)
        
        plt.suptitle(f'vLLM GPU Autoscaler - Online Learning Performance Analysis\nGPU: {analysis.get("basic_stats", {}).get("gpu_model", "Unknown")} | Algorithm: {analysis.get("basic_stats", {}).get("algorithm", "Unknown")}', 
                    fontsize=16, fontweight='bold')
        
        plt.savefig(f"{self.output_dir}/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        print(f"   保存: {self.output_dir}/comprehensive_analysis.png")
        
        # 创建收敛分析专门图表
        self._create_convergence_analysis_plot(df, analysis)
        
        plt.show()
    
    def _plot_reward_learning_curve(self, df: pd.DataFrame, ax):
        """绘制奖励学习曲线"""
        if 'current_reward' not in df.columns:
            ax.text(0.5, 0.5, 'No Reward Data Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Reward Learning Curve')
            return
        
        rounds = df['round']
        rewards = df['current_reward']
        avg_rewards = df['average_reward']
        
        # 绘制原始奖励
        ax.plot(rounds, rewards, alpha=0.4, color='lightblue', linewidth=1, label='Current Reward')
        
        # 绘制平均奖励
        ax.plot(rounds, avg_rewards, color='red', linewidth=2, label='Average Reward')
        
        # 绘制移动平均
        if len(rewards) > 10:
            window = min(20, len(rewards) // 5)
            moving_avg = rewards.rolling(window=window, center=True).mean()
            ax.plot(rounds, moving_avg, color='darkblue', linewidth=2, label=f'{window}-Round Moving Avg')
        
        # 标记学习和利用阶段
        learning_mask = df['phase'] == 'LEARNING'
        if learning_mask.any():
            ax.axvspan(rounds[learning_mask].min(), rounds[learning_mask].max(), 
                      alpha=0.2, color='orange', label='Learning Phase')
        
        exploitation_mask = df['phase'] == 'EXPLOITATION'
        if exploitation_mask.any():
            ax.axvspan(rounds[exploitation_mask].min(), rounds[exploitation_mask].max(), 
                      alpha=0.2, color='green', label='Exploitation Phase')
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Learning Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_frequency_selection_pattern(self, df: pd.DataFrame, ax):
        """绘制频率选择模式"""
        freq_counts = df['selected_frequency'].value_counts().sort_index()
        
        # 根据阶段设置颜色
        phase_colors = []
        for freq in freq_counts.index:
            freq_data = df[df['selected_frequency'] == freq]
            if not freq_data.empty:
                phase = freq_data['phase'].iloc[0]
                phase_colors.append('lightcoral' if phase == 'LEARNING' else 'lightblue')
            else:
                phase_colors.append('lightcoral')  # 默认学习阶段颜色
        
        bars = ax.bar(freq_counts.index, freq_counts.values, color=phase_colors, alpha=0.7)
        
        # 添加数值标签
        for bar, count in zip(bars, freq_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   str(count), ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Selection Count')
        ax.set_title('Frequency Selection Pattern')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightcoral', alpha=0.7, label='Learning Phase'),
                          Patch(facecolor='lightblue', alpha=0.7, label='Exploitation Phase')]
        ax.legend(handles=legend_elements)
        
        ax.grid(True, alpha=0.3)
    
    def _plot_edp_optimization(self, df: pd.DataFrame, ax):
        """绘制EDP优化趋势"""
        if 'edp_value' not in df.columns:
            ax.text(0.5, 0.5, 'No EDP Data Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('EDP Optimization Trend')
            return
        
        rounds = df['round']
        edp_values = df['edp_value']
        
        # 绘制EDP值
        ax.plot(rounds, edp_values, color='purple', linewidth=1, alpha=0.6, label='EDP Value')
        
        # 绘制移动平均
        if len(edp_values) > 5:
            window = min(10, len(edp_values) // 3)
            moving_avg = edp_values.rolling(window=window, center=True).mean()
            ax.plot(rounds, moving_avg, color='darkred', linewidth=2, label=f'{window}-Round Moving Avg')
        
        # 标记最优点
        min_edp_idx = edp_values.idxmin()
        min_edp_round = rounds[min_edp_idx]
        min_edp_value = edp_values[min_edp_idx]
        ax.scatter(min_edp_round, min_edp_value, color='red', s=100, zorder=5, label=f'Best EDP: {min_edp_value:.2f}')
        
        # 添加趋势线
        if len(edp_values) > 2:
            z = np.polyfit(rounds, edp_values, 1)
            p = np.poly1d(z)
            ax.plot(rounds, p(rounds), "--", color='orange', alpha=0.8, label=f'Trend (slope: {z[0]:.4f})')
        
        ax.set_xlabel('Round')
        ax.set_ylabel('EDP Value')
        ax.set_title('EDP Optimization Trend')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_exploration_exploitation_phases(self, df: pd.DataFrame, ax):
        """绘制探索-利用阶段分布"""
        phase_counts = df['phase'].value_counts()
        
        # 动态创建标签和颜色
        color_map = {'LEARNING': 'lightcoral', 'EXPLOITATION': 'lightblue'}
        label_map = {'LEARNING': 'Learning', 'EXPLOITATION': 'Exploitation'}
        
        colors = [color_map.get(phase, 'gray') for phase in phase_counts.index]
        labels = [label_map.get(phase, phase) for phase in phase_counts.index]
        
        if len(phase_counts) > 0:
            wedges, texts, autotexts = ax.pie(phase_counts.values, labels=labels, autopct='%1.1f%%', 
                                             colors=colors, startangle=90)
        else:
            ax.text(0.5, 0.5, 'No Phase Data Available', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Exploration vs Exploitation')
    
    def _plot_feature_importance_heatmap(self, df: pd.DataFrame, analysis: Dict[str, Any], ax):
        """绘制特征重要性热力图"""
        feature_corr = analysis.get('feature_analysis', {}).get('feature_reward_correlation', {})
        
        if not feature_corr:
            ax.text(0.5, 0.5, 'No Feature Correlation Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance')
            return
        
        features = list(feature_corr.keys())
        correlations = list(feature_corr.values())
        
        # 创建热力图数据
        corr_matrix = np.array(correlations).reshape(-1, 1)
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # 设置刻度
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
        ax.set_xticks([0])
        ax.set_xticklabels(['Reward Correlation'])
        
        # 添加数值标签
        for i, corr in enumerate(correlations):
            ax.text(0, i, f'{corr:.3f}', ha='center', va='center', 
                   color='white' if abs(corr) > 0.5 else 'black', fontweight='bold')
        
        ax.set_title('Feature-Reward Correlation')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_performance_metrics(self, df: pd.DataFrame, ax):
        """绘制性能指标对比"""
        if 'avg_ttft' not in df.columns or 'avg_tpot' not in df.columns:
            ax.text(0.5, 0.5, 'No Performance Data Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance Metrics')
            return
        
        rounds = df['round']
        ttft = df['avg_ttft']
        tpot = df['avg_tpot']
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(rounds, ttft, color='blue', linewidth=2, label='TTFT (s)')
        line2 = ax2.plot(rounds, tpot, color='red', linewidth=2, label='TPOT (s)')
        
        ax.set_xlabel('Round')
        ax.set_ylabel('TTFT (seconds)', color='blue')
        ax2.set_ylabel('TPOT (seconds)', color='red')
        ax.set_title('Performance Metrics Trend')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_frequency_reward_relationship(self, df: pd.DataFrame, ax):
        """绘制频率-奖励关系"""
        freq_reward_stats = df.groupby('selected_frequency')['current_reward'].agg(['mean', 'std', 'count']).reset_index()
        
        # 气泡图：x=频率, y=平均奖励, 大小=选择次数
        scatter = ax.scatter(freq_reward_stats['selected_frequency'], freq_reward_stats['mean'], 
                           s=freq_reward_stats['count']*20, alpha=0.6, c=freq_reward_stats['mean'], 
                           cmap='RdYlBu_r')
        
        # 添加误差条
        ax.errorbar(freq_reward_stats['selected_frequency'], freq_reward_stats['mean'], 
                   yerr=freq_reward_stats['std'], fmt='none', color='black', alpha=0.5)
        
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Average Reward')
        ax.set_title('Frequency-Reward Relationship')
        
        # 添加最佳频率标记
        best_freq_idx = freq_reward_stats['mean'].idxmax()
        best_freq = freq_reward_stats.loc[best_freq_idx, 'selected_frequency']
        best_reward = freq_reward_stats.loc[best_freq_idx, 'mean']
        ax.annotate(f'Best: {best_freq}MHz\n({best_reward:.3f})', 
                   xy=(best_freq, best_reward), xytext=(10, 10), 
                   textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.colorbar(scatter, ax=ax, label='Average Reward')
        ax.grid(True, alpha=0.3)
    
    def _plot_energy_efficiency_analysis(self, df: pd.DataFrame, ax):
        """绘制能效分析"""
        if 'energy_delta_j' not in df.columns:
            ax.text(0.5, 0.5, 'No Energy Data Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Energy Efficiency Analysis')
            return
        
        rounds = df['round']
        energy = df['energy_delta_j']
        efficiency = df['energy_efficiency']
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(rounds, energy, color='orange', linewidth=2, alpha=0.7, label='Energy Consumption (J)')
        line2 = ax2.plot(rounds, efficiency, color='green', linewidth=2, label='Energy Efficiency (1/EDP)')
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Energy Consumption (J)', color='orange')
        ax2.set_ylabel('Energy Efficiency', color='green')
        ax.set_title('Energy Efficiency Analysis')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_learning_quality_assessment(self, df: pd.DataFrame, analysis: Dict[str, Any], ax):
        """绘制学习质量评估"""
        quality_metrics = {}
        
        # 收集质量指标
        convergence = analysis.get('convergence_analysis', {})
        learning_quality = analysis.get('learning_quality', {})
        
        if convergence.get('convergence_achieved'):
            quality_metrics['Convergence'] = 100
        else:
            quality_metrics['Convergence'] = 0
        
        learning_eff = learning_quality.get('learning_efficiency', {})
        if 'learning_efficiency_ratio' in learning_eff:
            quality_metrics['Learning Efficiency'] = 100 - learning_eff['learning_efficiency_ratio']  # 越少越好
        
        decision_qual = learning_quality.get('decision_quality', {})
        if 'positive_reward_ratio' in decision_qual:
            quality_metrics['Decision Quality'] = decision_qual['positive_reward_ratio']
        
        algo_perf = learning_quality.get('algorithm_performance', {})
        if 'edp_trend' in algo_perf:
            # 将EDP趋势转换为性能分数
            edp_trend = algo_perf['edp_trend']
            if edp_trend < -0.1:
                quality_metrics['Performance'] = 100
            elif edp_trend < 0:
                quality_metrics['Performance'] = 70
            else:
                quality_metrics['Performance'] = 30
        
        if not quality_metrics:
            ax.text(0.5, 0.5, 'No Quality Data Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Quality Assessment')
            return
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(quality_metrics), endpoint=False)
        values = list(quality_metrics.values())
        
        # 闭合图形
        angles = np.concatenate((angles, [angles[0]]))
        values = np.concatenate((values, [values[0]]))
        
        ax.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(quality_metrics.keys())
        ax.set_ylim(0, 100)
        ax.set_title('Learning Quality Assessment\n(Score: 0-100)')
        ax.grid(True)
        
        # 添加分数标签
        for angle, value, label in zip(angles[:-1], values[:-1], quality_metrics.keys()):
            ax.annotate(f'{value:.1f}', xy=(angle, value), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
    
    def _create_convergence_analysis_plot(self, df: pd.DataFrame, analysis: Dict[str, Any]):
        """创建收敛分析专门图表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')
        
        # 1. 动作选择稳定性
        ax = axes[0, 0]
        if not df.empty:
            window_size = 20
            rolling_mode = df['selected_frequency'].rolling(window=window_size).apply(lambda x: x.mode()[0] if not x.empty else 0)
            stability = (df['selected_frequency'] == rolling_mode).rolling(window=window_size).mean() * 100
            
            ax.plot(df['round'], stability, color='blue', linewidth=2)
            ax.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='Convergence Threshold (60%)')
            ax.set_xlabel('Round')
            ax.set_ylabel('Action Stability (%)')
            ax.set_title('Action Selection Stability')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. 奖励方差分析
        ax = axes[0, 1]
        if 'current_reward' in df.columns:
            window_size = 15
            rolling_var = df['current_reward'].rolling(window=window_size).var()
            
            ax.plot(df['round'], rolling_var, color='green', linewidth=2)
            ax.set_xlabel('Round')
            ax.set_ylabel('Reward Variance')
            ax.set_title('Reward Variance Trend')
            ax.grid(True, alpha=0.3)
        
        # 3. 探索频率衰减
        ax = axes[1, 0]
        exploration_ratio = []
        window = 20
        for i in range(len(df)):
            start_idx = max(0, i - window + 1)
            recent_frequencies = df['selected_frequency'].iloc[start_idx:i+1]
            unique_count = recent_frequencies.nunique()
            total_available = df['n_available_frequencies'].iloc[i]
            exploration_ratio.append(unique_count / total_available * 100 if total_available > 0 else 0)
        
        ax.plot(df['round'], exploration_ratio, color='orange', linewidth=2)
        ax.set_xlabel('Round')
        ax.set_ylabel('Exploration Ratio (%)')
        ax.set_title('Exploration Frequency Decay')
        ax.grid(True, alpha=0.3)
        
        # 4. 收敛状态总结
        ax = axes[1, 1]
        convergence = analysis.get('convergence_analysis', {})
        
        # 创建收敛指标总结表
        convergence_data = []
        if convergence.get('convergence_achieved'):
            convergence_data.append(['Convergence Status', '✅ Achieved'])
            convergence_data.append(['Rounds to Convergence', f"{convergence.get('rounds_to_convergence', 'N/A')}"])
            convergence_data.append(['Dominant Frequency', f"{convergence.get('dominant_frequency', 'N/A')} MHz"])
            convergence_data.append(['Frequency Stability', f"{convergence.get('frequency_stability', 0):.1f}%"])
        else:
            convergence_data.append(['Convergence Status', '❌ Not Achieved'])
            convergence_data.append(['Learning Progress', f"{analysis.get('basic_stats', {}).get('learning_rounds', 0)} rounds"])
        
        # 显示表格
        table = ax.table(cellText=convergence_data, cellLoc='left', loc='center', colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Convergence Summary')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/convergence_analysis.png", dpi=300, bbox_inches='tight')
        print(f"   保存: {self.output_dir}/convergence_analysis.png")
    
    def generate_analysis_report(self, analysis: Dict[str, Any]) -> str:
        """生成分析报告"""
        print("📝 生成分析报告...")
        
        report = []
        report.append("# vLLM GPU Autoscaler 在线学习效果分析报告")
        report.append("=" * 50)
        report.append("")
        
        # 基础信息
        basic_stats = analysis.get('basic_stats', {})
        report.append("## 基础统计信息")
        report.append(f"- 总轮次: {basic_stats.get('total_rounds', 0)}")
        report.append(f"- 学习轮次: {basic_stats.get('learning_rounds', 0)}")
        report.append(f"- 利用轮次: {basic_stats.get('exploitation_rounds', 0)}")
        report.append(f"- 尝试频率数: {basic_stats.get('unique_frequencies_tried', 0)}")
        report.append(f"- 可用频率数: {basic_stats.get('total_frequencies_available', 0)}")
        report.append(f"- 算法: {basic_stats.get('algorithm', 'Unknown')}")
        report.append(f"- GPU型号: {basic_stats.get('gpu_model', 'Unknown')}")
        report.append("")
        
        # 收敛分析
        convergence = analysis.get('convergence_analysis', {})
        report.append("## 收敛性分析")
        if convergence.get('convergence_achieved'):
            report.append("✅ **模型已收敛**")
            report.append(f"- 收敛轮次: {convergence.get('rounds_to_convergence', 'N/A')}")
            report.append(f"- 收敛比例: {convergence.get('convergence_ratio', 0):.1f}%")
            report.append(f"- 主导频率: {convergence.get('dominant_frequency', 'N/A')} MHz")
            report.append(f"- 频率稳定性: {convergence.get('frequency_stability', 0):.1f}%")
        else:
            report.append("❌ **模型未收敛**")
            report.append("- 建议增加训练轮次或调整算法参数")
        report.append("")
        
        # 模式切换事件分析
        mode_switching = analysis.get('mode_switching', {})
        report.append("## 模式切换事件分析")
        
        if mode_switching.get('has_mode_switches'):
            report.append(f"- 切换事件总数: {mode_switching.get('events_count', 0)}")
            report.append(f"- 收敛事件: {mode_switching.get('convergence_events', 0)}")
            report.append(f"- 性能退化事件: {mode_switching.get('degradation_events', 0)}")
            report.append(f"- 学习-利用循环: {mode_switching.get('learning_exploitation_cycles', 0)}")
            
            if 'avg_switch_interval_seconds' in mode_switching:
                report.append(f"- 平均切换间隔: {mode_switching['avg_switch_interval_seconds']:.1f}秒")
            
            report.append(f"- 收敛行为: {mode_switching.get('convergence_message', 'N/A')}")
            report.append(f"- 适应行为: {mode_switching.get('adaptation_message', 'N/A')}")
        else:
            report.append("- 状态: 未检测到模式切换事件")
        report.append("")
        
        # 学习质量评估
        learning_quality = analysis.get('learning_quality', {})
        report.append("## 学习质量评估")
        
        learning_eff = learning_quality.get('learning_efficiency', {})
        if learning_eff:
            report.append(f"- 学习效率: {learning_eff.get('convergence_speed', 'Unknown')}")
            report.append(f"- 学习效率比例: {learning_eff.get('learning_efficiency_ratio', 0):.1f}%")
        
        decision_qual = learning_quality.get('decision_quality', {})
        if decision_qual:
            report.append(f"- 决策质量: 正奖励比例 {decision_qual.get('positive_reward_ratio', 0):.1f}%")
            report.append(f"- 学习稳定性: {decision_qual.get('learning_stability', 'Unknown')}")
        
        algo_perf = learning_quality.get('algorithm_performance', {})
        if algo_perf:
            report.append(f"- 算法性能: {algo_perf.get('optimization_effectiveness', 'Unknown')}")
            report.append(f"- 性能改善: {algo_perf.get('performance_improvement', 'Unknown')}")
        report.append("")
        
        # 探索-利用分析
        exp_exp = analysis.get('exploration_exploitation', {})
        report.append("## 探索-利用权衡分析")
        
        freq_diversity = exp_exp.get('frequency_diversity', {})
        if freq_diversity:
            report.append(f"- 频率多样性: {freq_diversity.get('unique_frequencies', 0)} 个不同频率")
            report.append(f"- 选择熵: {freq_diversity.get('entropy', 0):.3f}")
            report.append(f"- 最常选择频率: {freq_diversity.get('most_selected_frequency', 'N/A')} MHz ({freq_diversity.get('most_selected_ratio', 0):.1f}%)")
        
        exploration_pattern = exp_exp.get('exploration_pattern', {})
        if exploration_pattern:
            report.append(f"- 探索阶段频率数: {exploration_pattern.get('frequencies_explored', 0)}")
            report.append(f"- 探索均匀性: {exploration_pattern.get('exploration_uniformity', 0):.3f}")
        
        exploitation_pattern = exp_exp.get('exploitation_pattern', {})
        if exploitation_pattern:
            report.append(f"- 利用阶段集中度: {exploitation_pattern.get('exploitation_concentration', 0):.1f}%")
        report.append("")
        
        # 性能优化效果
        perf_opt = analysis.get('performance_optimization', {})
        report.append("## 性能优化效果")
        
        ttft_opt = perf_opt.get('ttft_optimization', {})
        if ttft_opt:
            report.append(f"- TTFT优化: {ttft_opt.get('ttft_improvement', 0):.4f}s ({ttft_opt.get('ttft_improvement_ratio', 0):.1f}%)")
        
        tpot_opt = perf_opt.get('tpot_optimization', {})
        if tpot_opt:
            report.append(f"- TPOT优化: {tpot_opt.get('tpot_improvement', 0):.4f}s ({tpot_opt.get('tpot_improvement_ratio', 0):.1f}%)")
        
        latency_opt = perf_opt.get('latency_optimization', {})
        if latency_opt:
            report.append(f"- 总延迟优化: {latency_opt.get('latency_improvement', 0):.4f}s ({latency_opt.get('latency_improvement_ratio', 0):.1f}%)")
        report.append("")
        
        # 能效分析
        energy_eff = analysis.get('energy_efficiency', {})
        report.append("## 能效优化分析")
        
        edp_opt = energy_eff.get('edp_optimization', {})
        if edp_opt:
            report.append(f"- EDP优化: {edp_opt.get('edp_improvement', 0):.3f} ({edp_opt.get('edp_improvement_ratio', 0):.1f}%)")
            report.append(f"- 最佳EDP: {edp_opt.get('best_edp', 0):.3f}")
            report.append(f"- 平均EDP: {edp_opt.get('average_edp', 0):.3f}")
        
        efficiency_metrics = energy_eff.get('efficiency_metrics', {})
        if efficiency_metrics:
            report.append(f"- 能效提升: {efficiency_metrics.get('efficiency_improvement_ratio', 0):.1f}%")
        report.append("")
        
        # 频率偏好分析
        freq_pref = analysis.get('frequency_preference', {})
        report.append("## 频率偏好学习")
        
        optimal_freq = freq_pref.get('optimal_frequency_analysis', {})
        if optimal_freq:
            report.append(f"- 最优频率(奖励): {optimal_freq.get('best_frequency_by_reward', 'N/A')} MHz")
            report.append(f"- 最常选择频率: {optimal_freq.get('most_selected_frequency', 'N/A')} MHz")
            report.append(f"- 偏好匹配: {'✅' if optimal_freq.get('optimal_vs_preferred_match') else '❌'}")
        
        energy_analysis = freq_pref.get('energy_efficiency_analysis', {})
        if energy_analysis:
            report.append(f"- 最优频率(EDP): {energy_analysis.get('best_frequency_by_edp', 'N/A')} MHz")
            report.append(f"- 最佳EDP值: {energy_analysis.get('best_edp_value', 0):.3f}")
        report.append("")
        
        # 总结建议
        report.append("## 总结与建议")
        
        if convergence.get('convergence_achieved'):
            report.append("✅ 模型训练成功，已达到收敛状态")
        else:
            report.append("⚠️ 模型尚未收敛，建议：")
            report.append("  - 增加训练轮次")
            report.append("  - 调整探索参数")
            report.append("  - 检查奖励函数设计")
        
        if algo_perf.get('optimization_effectiveness') == 'Excellent':
            report.append("✅ 算法性能优秀，EDP持续改善")
        elif algo_perf.get('optimization_effectiveness') == 'Good':
            report.append("👍 算法性能良好，有一定改善效果")
        else:
            report.append("⚠️ 算法性能有待提升，建议优化参数")
        
        if optimal_freq.get('optimal_vs_preferred_match'):
            report.append("✅ 学习效果良好，最优频率与偏好频率一致")
        else:
            report.append("⚠️ 可能存在探索不充分，建议增加探索")
        
        report_text = "\n".join(report)
        
        # 保存报告
        report_path = f"{self.output_dir}/analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"   报告已保存: {report_path}")
        return report_text
    
    def save_analysis_json(self, analysis: Dict[str, Any]):
        """保存分析结果为JSON"""
        json_path = f"{self.output_dir}/analysis_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"   JSON分析结果已保存: {json_path}")
    
    def run_comprehensive_analysis(self, log_file_path: str = None, gpu_model: str = None):
        """运行全面分析"""
        print("🚀 开始全面在线学习效果分析...")
        
        # 查找日志文件
        if not log_file_path:
            log_file_path = self.find_latest_log(gpu_model)
            if not log_file_path:
                print("❌ 未找到日志文件")
                return
        
        # 解析日志
        df = self.parse_log(log_file_path)
        if df.empty:
            print("❌ 日志解析失败或无数据")
            return
        
        # 进行全面分析
        analysis = self.analyze_learning_effectiveness(df)
        
        # 生成可视化
        self.create_comprehensive_visualizations(df, analysis)
        
        # 生成报告
        report = self.generate_analysis_report(analysis)
        
        # 保存JSON结果
        self.save_analysis_json(analysis)
        
        print("=" * 60)
        print("📊 **分析完成！主要发现：**")
        print("=" * 60)
        
        basic_stats = analysis.get('basic_stats', {})
        convergence = analysis.get('convergence_analysis', {})
        learning_quality = analysis.get('learning_quality', {})
        
        print(f"🎯 **基础统计**: {basic_stats.get('total_rounds', 0)} 轮, {basic_stats.get('unique_frequencies_tried', 0)} 频率")
        
        if convergence.get('convergence_achieved'):
            print(f"✅ **收敛状态**: 已收敛 ({convergence.get('rounds_to_convergence', 'N/A')} 轮)")
            print(f"🎲 **主导频率**: {convergence.get('dominant_frequency', 'N/A')} MHz ({convergence.get('frequency_stability', 0):.1f}% 稳定性)")
        else:
            print("❌ **收敛状态**: 未收敛")
        
        algo_perf = learning_quality.get('algorithm_performance', {})
        if algo_perf:
            print(f"⚡ **算法性能**: {algo_perf.get('optimization_effectiveness', 'Unknown')}")
        
        # 性能改善总结
        perf_opt = analysis.get('performance_optimization', {})
        energy_eff = analysis.get('energy_efficiency', {})
        
        ttft_improvement = perf_opt.get('ttft_optimization', {}).get('ttft_improvement_ratio', 0)
        tpot_improvement = perf_opt.get('tpot_optimization', {}).get('tpot_improvement_ratio', 0)
        edp_improvement = energy_eff.get('edp_optimization', {}).get('edp_improvement_ratio', 0)
        
        print(f"📈 **性能改善**: TTFT {ttft_improvement:+.1f}%, TPOT {tpot_improvement:+.1f}%, EDP {edp_improvement:+.1f}%")
        
        print("=" * 60)
        print(f"📁 **输出文件**:")
        print(f"   - 综合分析图: {self.output_dir}/comprehensive_analysis.png")
        print(f"   - 收敛分析图: {self.output_dir}/convergence_analysis.png") 
        print(f"   - 分析报告: {self.output_dir}/analysis_report.txt")
        print(f"   - JSON数据: {self.output_dir}/analysis_report.json")
        print("=" * 60)
    
    def _analyze_mode_switching_events(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析模式切换事件"""
        print("🔄 分析模式切换事件...")
        
        events_analysis = {}
        
        # 检查是否有模式切换事件记录
        if not hasattr(self, 'mode_switch_events') or not self.mode_switch_events:
            events_analysis['has_mode_switches'] = False
            events_analysis['events_count'] = 0
            events_analysis['convergence_events'] = 0
            events_analysis['degradation_events'] = 0
            events_analysis['message'] = '未检测到模式切换事件'
            return events_analysis
        
        events = self.mode_switch_events
        events_analysis['has_mode_switches'] = True
        events_analysis['events_count'] = len(events)
        
        # 按事件类型分类统计
        event_types = {}
        for event in events:
            event_type = event['event_type']
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)
        
        events_analysis['event_types'] = event_types
        events_analysis['convergence_events'] = len(event_types.get('convergence_detected', []))
        events_analysis['degradation_events'] = len(event_types.get('performance_degradation', []))
        events_analysis['exploitation_switches'] = len(event_types.get('switch_to_exploitation', []))
        events_analysis['learning_switches'] = len(event_types.get('switch_to_learning', []))
        
        # 分析模式切换频率
        if len(events) > 1:
            time_deltas = []
            for i in range(1, len(events)):
                delta = events[i]['timestamp'] - events[i-1]['timestamp']
                time_deltas.append(delta.total_seconds())
            
            if time_deltas:
                events_analysis['avg_switch_interval_seconds'] = np.mean(time_deltas)
                events_analysis['min_switch_interval_seconds'] = min(time_deltas)
                events_analysis['max_switch_interval_seconds'] = max(time_deltas)
        
        # 检查学习-利用循环模式
        cycle_count = 0
        last_mode = None
        for event in events:
            if event['event_type'] in ['switch_to_exploitation', 'switch_to_learning']:
                current_mode = 'exploitation' if 'exploitation' in event['event_type'] else 'learning'
                if last_mode and last_mode != current_mode:
                    cycle_count += 1
                last_mode = current_mode
        
        events_analysis['learning_exploitation_cycles'] = cycle_count // 2  # 完整循环数
        
        # 总结模式切换效果
        if events_analysis['convergence_events'] > 0:
            events_analysis['convergence_behavior'] = 'good'
            events_analysis['convergence_message'] = f'检测到 {events_analysis["convergence_events"]} 次收敛事件'
        else:
            events_analysis['convergence_behavior'] = 'none'
            events_analysis['convergence_message'] = '未检测到收敛事件'
        
        if events_analysis['degradation_events'] > 0:
            events_analysis['adaptation_behavior'] = 'reactive'
            events_analysis['adaptation_message'] = f'检测到 {events_analysis["degradation_events"]} 次性能退化，系统主动切换回学习模式'
        else:
            events_analysis['adaptation_behavior'] = 'stable'
            events_analysis['adaptation_message'] = '性能稳定，无退化检测'
        
        return events_analysis


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='vLLM GPU Autoscaler 在线学习效果分析')
    parser.add_argument('--log-file', type=str, help='指定日志文件路径')
    parser.add_argument('--gpu-model', type=str, help='指定GPU型号目录 (如: A800_80GB_PCIe)')
    parser.add_argument('--output-dir', type=str, default='data/analysis', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建分析器并运行分析
    analyzer = VLLMAutoscalerAnalyzer(output_dir=args.output_dir)
    analyzer.run_comprehensive_analysis(log_file_path=args.log_file, gpu_model=args.gpu_model)


if __name__ == "__main__":
    main()