#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM GPU Autoscaler EDP专项分析工具
EDP-Focused Analysis Tool - 专注于长期原始EDP变化趋势分析
"""

import os
import re
import json
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
import seaborn as sns
from collections import Counter, defaultdict

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['figure.dpi'] = 100

class EDPAnalyzer:
    """EDP专项分析器 - 专注于原始EDP值的长期变化趋势"""
    
    def __init__(self, log_pattern: str = "logs/**/*.log", max_rounds: Optional[int] = None):
        self.log_pattern = log_pattern
        self.max_rounds = max_rounds
        self.data = []
        self.idle_periods = []
        self.alpha_history = []
        self.pruning_history = []
        print("🔍 vLLM GPU调频器EDP专项分析器已初始化")
        print(f"📂 日志模式: {log_pattern}")
        if max_rounds:
            print(f"🔢 分析范围: 前{max_rounds}轮数据")
    
    def find_latest_log(self) -> str:
        """查找最新的日志文件"""
        log_files = glob(self.log_pattern, recursive=True)
        if not log_files:
            print(f"⚠️ 未找到匹配的日志文件: {self.log_pattern}")
            return None
        
        # 按修改时间排序，最新的在前
        latest_log = max(log_files, key=lambda x: os.path.getmtime(x))
        mtime = datetime.fromtimestamp(os.path.getmtime(latest_log))
        size_mb = os.path.getsize(latest_log) / (1024 * 1024)
        
        print(f"📄 找到最新日志文件:")
        print(f"   {latest_log} ({size_mb:.1f}MB, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        
        return latest_log
    
    def parse_log_file(self, log_file: str):
        """解析日志文件，专注提取EDP相关数据"""
        print(f"📖 解析日志文件: {log_file}")
        
        json_count = 0
        idle_count = 0
        alpha_count = 0
        pruning_count = 0
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # 解析JSON格式的决策数据
                    if '📋 JSON数据:' in line:
                        try:
                            json_str = line.split('📋 JSON数据: ', 1)[1]
                            data = json.loads(json_str)
                            data['log_file'] = log_file
                            data['line_num'] = line_num
                            
                            # 确保EDP值存在且有效
                            edp_val = data.get('performance', {}).get('edp_value')
                            if edp_val is not None and edp_val > 0:
                                # 检查是否超过最大轮次限制
                                round_num = data.get('round', 0)
                                if self.max_rounds is None or round_num <= self.max_rounds:
                                    self.data.append(data)
                                    json_count += 1
                        except json.JSONDecodeError:
                            pass
                    
                    # 解析休息模式信息
                    elif '😴 检测到持续空闲' in line or '🏃 检测到运行任务' in line:
                        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                        if timestamp_match:
                            timestamp = timestamp_match.group(1)
                            event_type = 'enter_idle' if '😴' in line else 'exit_idle'
                            self.idle_periods.append({
                                'timestamp': timestamp,
                                'event': event_type,
                                'line': line
                            })
                            idle_count += 1
                    
                    # 解析alpha衰减信息
                    elif '当前α=' in line:
                        try:
                            alpha_match = re.search(r'当前α=(\d+\.\d+)', line)
                            if alpha_match:
                                current_alpha = float(alpha_match.group(1))
                                self.alpha_history.append({
                                    'alpha': current_alpha,
                                    'line': line
                                })
                                alpha_count += 1
                        except ValueError:
                            pass
                    
                    # 解析智能修剪信息
                    elif '🗂️ [智能修剪]' in line:
                        self.pruning_history.append({
                            'line': line
                        })
                        pruning_count += 1
        
        except Exception as e:
            print(f"❌ 文件读取错误: {e}")
            return
        
        max_rounds_info = f" (限制前{self.max_rounds}轮)" if self.max_rounds else ""
        print(f"   ✅ 解析完成: {json_count}条有效EDP数据{max_rounds_info}, {idle_count}条休息记录, {alpha_count}条α记录, {pruning_count}条修剪记录")
    
    def analyze_edp_trends(self):
        """分析EDP趋势"""
        if not self.data:
            print("❌ 没有数据可分析")
            return None
        
        # 提取EDP数据
        edp_data = []
        for d in self.data:
            round_num = d['round']
            edp_value = d['performance']['edp_value']
            frequency = d['decision']['selected_frequency_compat']
            energy = d['performance']['energy_delta_j']
            avg_ttft = d['performance']['avg_ttft']
            avg_tpot = d['performance']['avg_tpot']
            reward = d['reward']['current']
            
            edp_data.append({
                'round': round_num,
                'edp': edp_value,
                'frequency': frequency,
                'energy': energy,
                'ttft': avg_ttft,
                'tpot': avg_tpot,
                'delay': avg_ttft + avg_tpot,  # 总延迟
                'reward': reward
            })
        
        df = pd.DataFrame(edp_data)
        
        # 计算EDP趋势统计
        analysis_result = {
            'total_rounds': len(df),
            'edp_stats': {
                'min_edp': df['edp'].min(),
                'max_edp': df['edp'].max(),
                'mean_edp': df['edp'].mean(),
                'std_edp': df['edp'].std(),
                'median_edp': df['edp'].median(),
                'p25_edp': df['edp'].quantile(0.25),
                'p75_edp': df['edp'].quantile(0.75)
            },
            'trend_analysis': self._analyze_edp_trend(df),
            'frequency_performance': self._analyze_frequency_performance(df),
            'data': df,
            'improvement_analysis': self._calculate_improvement(df)
        }
        
        return analysis_result
    
    def _analyze_edp_trend(self, df):
        """分析EDP趋势变化"""
        if len(df) < 10:
            return {'trend': 'insufficient_data'}
        
        # 计算移动平均
        window_sizes = [10, 20, 50]
        trends = {}
        
        for window in window_sizes:
            if len(df) >= window:
                ma = df['edp'].rolling(window=window, center=True).mean()
                # 计算趋势斜率（最后部分vs开始部分）
                start_avg = ma.iloc[:window].mean()
                end_avg = ma.iloc[-window:].mean()
                
                if start_avg > 0:
                    improvement_pct = (start_avg - end_avg) / start_avg * 100
                    trends[f'ma_{window}'] = {
                        'improvement_percent': improvement_pct,
                        'start_avg': start_avg,
                        'end_avg': end_avg,
                        'trend': 'improving' if improvement_pct > 5 else 'stable' if abs(improvement_pct) <= 5 else 'degrading'
                    }
        
        return trends
    
    def _analyze_frequency_performance(self, df):
        """分析不同频率的EDP表现"""
        freq_performance = df.groupby('frequency').agg({
            'edp': ['count', 'mean', 'std', 'min'],
            'energy': 'mean',
            'delay': 'mean',
            'reward': 'mean'
        }).round(4)
        
        # 展平列名
        freq_performance.columns = ['_'.join(col).strip() for col in freq_performance.columns]
        freq_performance = freq_performance.reset_index()
        
        # 按平均EDP排序（越小越好）
        freq_performance = freq_performance.sort_values('edp_mean')
        
        return freq_performance
    
    def _calculate_improvement(self, df):
        """计算整体改进效果"""
        if len(df) < 20:
            return {'status': 'insufficient_data'}
        
        # 分割前后期数据
        split_point = len(df) // 2
        early_period = df.iloc[:split_point]
        late_period = df.iloc[split_point:]
        
        early_edp = early_period['edp'].mean()
        late_edp = late_period['edp'].mean()
        
        if early_edp > 0:
            improvement_pct = (early_edp - late_edp) / early_edp * 100
            
            return {
                'early_avg_edp': early_edp,
                'late_avg_edp': late_edp,
                'improvement_percent': improvement_pct,
                'status': 'improving' if improvement_pct > 5 else 'stable' if abs(improvement_pct) <= 5 else 'degrading',
                'early_rounds': len(early_period),
                'late_rounds': len(late_period)
            }
        
        return {'status': 'calculation_error'}
    
    def create_edp_visualizations(self, analysis_result: dict):
        """生成EDP专项可视化图表 - 图表标签纯英文"""
        if not analysis_result:
            return
        
        df = analysis_result['data']
        
        # 创建综合图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        title_suffix = f" (First {self.max_rounds} Rounds)" if self.max_rounds else ""
        fig.suptitle(f'vLLM GPU Auto-Scaler: Long-term EDP Analysis{title_suffix}', fontsize=16, fontweight='bold')
        
        # 1. 原始EDP变化趋势
        rounds = df['round'].values
        edp_values = df['edp'].values
        
        axes[0, 0].plot(rounds, edp_values, 'b-', alpha=0.6, linewidth=1, label='Raw EDP')
        
        # 添加移动平均线
        for window in [10, 20, 50]:
            if len(df) >= window:
                ma = df['edp'].rolling(window=window, center=True).mean()
                axes[0, 0].plot(rounds, ma, linewidth=2, label=f'MA-{window}')
        
        axes[0, 0].set_title('Raw EDP Evolution Over Time')
        axes[0, 0].set_xlabel('Decision Round')
        axes[0, 0].set_ylabel('EDP (Energy × Delay)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加改进区间标注
        improvement = analysis_result['improvement_analysis']
        if improvement['status'] == 'improving':
            axes[0, 0].text(0.7, 0.9, f'Improvement: {improvement["improvement_percent"]:.1f}%', 
                           transform=axes[0, 0].transAxes, bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor="lightgreen", alpha=0.7), fontsize=10)
        
        # 2. EDP vs 频率散点图
        freq_perf = analysis_result['frequency_performance']
        frequencies = freq_perf['frequency'].values
        avg_edp = freq_perf['edp_mean'].values
        selection_counts = freq_perf['edp_count'].values
        
        scatter = axes[0, 1].scatter(frequencies, avg_edp, s=selection_counts*10, 
                                   alpha=0.6, c=avg_edp, cmap='RdYlBu_r')
        axes[0, 1].set_title('Frequency vs Average EDP')
        axes[0, 1].set_xlabel('Frequency (MHz)')
        axes[0, 1].set_ylabel('Average EDP')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=axes[0, 1])
        cbar.set_label('Average EDP')
        
        # 3. EDP分布直方图
        axes[0, 2].hist(edp_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 2].axvline(np.mean(edp_values), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(edp_values):.3f}')
        axes[0, 2].axvline(np.median(edp_values), color='orange', linestyle='--', 
                          label=f'Median: {np.median(edp_values):.3f}')
        axes[0, 2].set_title('EDP Distribution')
        axes[0, 2].set_xlabel('EDP Value')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 能耗和延迟的双轴图
        ax_energy = axes[1, 0]
        ax_delay = ax_energy.twinx()
        
        line1 = ax_energy.plot(rounds, df['energy'], 'g-', alpha=0.7, label='Energy (J)')
        line2 = ax_delay.plot(rounds, df['delay']*1000, 'r-', alpha=0.7, label='Delay (ms)')
        
        ax_energy.set_xlabel('Decision Round')
        ax_energy.set_ylabel('Energy Consumption (J)', color='g')
        ax_delay.set_ylabel('Total Delay (ms)', color='r')
        ax_energy.set_title('Energy vs Delay Trade-off')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_energy.legend(lines, labels, loc='upper right')
        ax_energy.grid(True, alpha=0.3)
        
        # 5. 频率使用热图
        freq_counts = df.groupby(['round', 'frequency']).size().unstack(fill_value=0)
        if len(freq_counts.columns) > 1:
            # 只显示最常用的频率
            top_freqs = df['frequency'].value_counts().head(10).index
            freq_counts_top = freq_counts[top_freqs]
            
            sns.heatmap(freq_counts_top.T, cmap='YlOrRd', ax=axes[1, 1], 
                       cbar_kws={'label': 'Selection Count'})
            axes[1, 1].set_title('Frequency Usage Pattern')
            axes[1, 1].set_xlabel('Decision Round')
            axes[1, 1].set_ylabel('Frequency (MHz)')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient frequency variation', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Frequency Usage Pattern')
        
        # 6. EDP改进趋势
        if len(df) >= 50:
            # 计算滚动最佳EDP
            rolling_min_edp = df['edp'].expanding().min()
            axes[1, 2].plot(rounds, edp_values, 'b-', alpha=0.3, label='Raw EDP')
            axes[1, 2].plot(rounds, rolling_min_edp, 'r-', linewidth=2, label='Best EDP So Far')
            axes[1, 2].set_title('EDP Improvement Progress')
            axes[1, 2].set_xlabel('Decision Round')
            axes[1, 2].set_ylabel('EDP Value')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
            # 计算改进率
            if len(rolling_min_edp) > 10:
                initial_best = rolling_min_edp.iloc[9]  # 前10轮的最佳
                final_best = rolling_min_edp.iloc[-1]
                if initial_best > 0:
                    total_improvement = (initial_best - final_best) / initial_best * 100
                    axes[1, 2].text(0.7, 0.1, f'Total Improvement: {total_improvement:.1f}%',
                                   transform=axes[1, 2].transAxes, 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        else:
            axes[1, 2].text(0.5, 0.5, 'Insufficient data for trend analysis', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('EDP Improvement Progress')
        
        plt.tight_layout()
        
        # 保存图表
        suffix = f"_first{self.max_rounds}rounds" if self.max_rounds else ""
        output_file = f"data/analysis/edp_comprehensive_analysis{suffix}.png"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n📊 EDP专项可视化图表已保存: {output_file}")
        
        return output_file
    
    def create_comprehensive_report(self, analysis_result: dict):
        """生成EDP专项分析报告"""
        if not analysis_result:
            return
        
        print("\n" + "="*80)
        print("📊 vLLM GPU自动调频器 - EDP专项分析报告")
        print("="*80)
        
        # 基本统计
        print(f"\n🎯 基本统计:")
        total_rounds_info = f"总决策轮次: {analysis_result['total_rounds']}"
        if self.max_rounds:
            total_rounds_info += f" (分析前{self.max_rounds}轮)"
        print(f"   {total_rounds_info}")
        
        # EDP统计
        edp_stats = analysis_result['edp_stats']
        print(f"\n⚡ EDP统计 (Energy-Delay Product):")
        print(f"   最小值: {edp_stats['min_edp']:.4f}")
        print(f"   最大值: {edp_stats['max_edp']:.4f}")
        print(f"   平均值: {edp_stats['mean_edp']:.4f} ± {edp_stats['std_edp']:.4f}")
        print(f"   中位数: {edp_stats['median_edp']:.4f}")
        print(f"   25%分位: {edp_stats['p25_edp']:.4f}")
        print(f"   75%分位: {edp_stats['p75_edp']:.4f}")
        
        # 趋势分析
        trends = analysis_result['trend_analysis']
        print(f"\n📈 EDP趋势分析:")
        for ma_name, trend_data in trends.items():
            if isinstance(trend_data, dict) and 'improvement_percent' in trend_data:
                window = ma_name.split('_')[1]
                improvement = trend_data['improvement_percent']
                status = trend_data['trend']
                print(f"   {window}轮移动平均: {improvement:+.2f}% ({status})")
        
        # 整体改进分析
        improvement = analysis_result['improvement_analysis']
        if improvement['status'] != 'insufficient_data':
            print(f"\n🚀 整体改进效果:")
            print(f"   前期平均EDP: {improvement['early_avg_edp']:.4f}")
            print(f"   后期平均EDP: {improvement['late_avg_edp']:.4f}")
            print(f"   改进程度: {improvement['improvement_percent']:+.2f}% ({improvement['status']})")
        
        # 频率性能排名
        freq_perf = analysis_result['frequency_performance']
        print(f"\n🏆 频率性能排名 (按平均EDP排序, 越小越好):")
        print(f"   {'频率(MHz)':<10} {'选择次数':<8} {'平均EDP':<10} {'平均能耗(J)':<12} {'平均延迟(s)':<12}")
        print(f"   {'-'*60}")
        
        for _, row in freq_perf.head(10).iterrows():
            freq = int(row['frequency'])
            count = int(row['edp_count'])
            avg_edp = row['edp_mean']
            avg_energy = row['energy_mean']
            avg_delay = row['delay_mean']
            print(f"   {freq:<10} {count:<8} {avg_edp:<10.4f} {avg_energy:<12.2f} {avg_delay:<12.4f}")
        
        return analysis_result
    
    def run_analysis(self, target_log: str = None):
        """运行EDP专项分析"""
        rounds_info = f" (前{self.max_rounds}轮)" if self.max_rounds else ""
        print(f"🚀 开始EDP专项分析{rounds_info}...")
        
        # 查找日志文件
        if target_log:
            log_file = target_log
        else:
            log_file = self.find_latest_log()
        
        if not log_file:
            return None
        
        # 解析日志
        self.parse_log_file(log_file)
        
        # 分析EDP趋势
        analysis_result = self.analyze_edp_trends()
        if not analysis_result:
            return None
        
        # 生成报告
        self.create_comprehensive_report(analysis_result)
        
        # 生成可视化
        self.create_edp_visualizations(analysis_result)
        
        # 保存分析结果
        suffix = f"_first{self.max_rounds}rounds" if self.max_rounds else ""
        output_file = f"data/analysis/edp_analysis_report{suffix}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            # 转换DataFrame为字典以便JSON序列化
            result_copy = analysis_result.copy()
            result_copy['data'] = result_copy['data'].to_dict('records')
            result_copy['frequency_performance'] = result_copy['frequency_performance'].to_dict('records')
            json.dump(result_copy, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 EDP分析结果已保存: {output_file}")
        print("✅ EDP专项分析完成!")
        
        return analysis_result

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM GPU调频器EDP专项分析工具")
    parser.add_argument("--log", help="指定要分析的日志文件路径")
    parser.add_argument("--pattern", default="logs/**/*.log", help="日志文件搜索模式")
    parser.add_argument("--max-rounds", type=int, help="指定分析前n轮数据，不指定则分析全部数据")
    
    args = parser.parse_args()
    
    # 创建EDP分析器
    analyzer = EDPAnalyzer(args.pattern, args.max_rounds)
    
    # 运行分析
    result = analyzer.run_analysis(args.log)
    
    if result:
        suffix = f"_first{args.max_rounds}rounds" if args.max_rounds else ""
        print(f"\n🎉 EDP专项分析报告已生成完成!")
        print(f"📊 查看EDP可视化图表: data/analysis/edp_comprehensive_analysis{suffix}.png")
        print(f"📄 查看详细数据: data/analysis/edp_analysis_report{suffix}.json")

if __name__ == "__main__":
    main()