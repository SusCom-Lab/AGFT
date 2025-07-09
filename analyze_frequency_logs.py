#!/usr/bin/env python3
"""
日志频率选择信息分析器
用于查找和分析现有日志文件中的频率选择模式
"""

import os
import re
import glob
from datetime import datetime
from collections import defaultdict, Counter
import json

def find_frequency_patterns():
    """查找所有日志文件中的频率相关模式"""
    
    # 定义需要搜索的模式
    patterns = {
        'decision': r'🤔.*LinUCB决策:?\s*(\d+)MHz',  # LinUCB决策
        'adjust_up': r'📈.*调整频率:?\s*(\d+)MHz\s*→\s*(\d+)MHz',  # 向上调整频率
        'adjust_down': r'📉.*调整频率:?\s*(\d+)MHz\s*→\s*(\d+)MHz',  # 向下调整频率
        'keep_freq': r'✅.*保持频率:?\s*(\d+)MHz',  # 保持频率
        'set_error': r'❌\s*设置GPU频率\s*(\d+)MHz\s*失败',  # 设置频率失败
        'freq_sync': r'智能频率空间已同步:?\s*(\d+)个频率\s*\((\d+)-(\d+)MHz\)',  # 频率同步
        'freq_range': r'频率范围:?\s*(\d+)-(\d+)MHz',  # 频率范围
        'round_info': r'轮次\s*(\d+):.*奖励=([\d\.-]+)',  # 轮次信息
    }
    
    log_dirs = glob.glob('/home/ldaphome/colin/workplace/vllm_mab/logs/*/')
    results = {}
    
    for log_dir in log_dirs:
        gpu_type = os.path.basename(log_dir.rstrip('/'))
        results[gpu_type] = {
            'files': [],
            'patterns': defaultdict(list),
            'frequency_stats': Counter(),
            'timestamps': []
        }
        
        log_files = glob.glob(os.path.join(log_dir, '*.log'))
        
        for log_file in sorted(log_files):
            file_info = {
                'filename': os.path.basename(log_file),
                'path': log_file,
                'patterns_found': defaultdict(list),
                'size': os.path.getsize(log_file)
            }
            
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        # 提取时间戳
                        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                        if timestamp_match:
                            timestamp_str = timestamp_match.group(1)
                            try:
                                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                                results[gpu_type]['timestamps'].append(timestamp)
                            except:
                                pass
                        
                        # 检查每个模式
                        for pattern_name, pattern_regex in patterns.items():
                            matches = re.findall(pattern_regex, line, re.IGNORECASE)
                            if matches:
                                match_info = {
                                    'line_num': line_num,
                                    'line': line.strip(),
                                    'matches': matches,
                                    'timestamp': timestamp_str if timestamp_match else None
                                }
                                file_info['patterns_found'][pattern_name].append(match_info)
                                results[gpu_type]['patterns'][pattern_name].append(match_info)
                                
                                # 统计频率使用情况
                                if pattern_name in ['decision', 'keep_freq', 'set_error']:
                                    if matches and isinstance(matches[0], str):
                                        freq = int(matches[0])
                                        results[gpu_type]['frequency_stats'][freq] += 1
                                elif pattern_name in ['adjust_up', 'adjust_down']:
                                    if matches and len(matches[0]) >= 2:
                                        freq_to = int(matches[0][1])
                                        results[gpu_type]['frequency_stats'][freq_to] += 1
            
            except Exception as e:
                file_info['error'] = str(e)
            
            results[gpu_type]['files'].append(file_info)
    
    return results

def print_analysis_summary(results):
    """打印分析摘要"""
    print("="*80)
    print("📊 日志频率选择信息分析报告")
    print("="*80)
    
    for gpu_type, data in results.items():
        print(f"\n🎮 GPU类型: {gpu_type}")
        print(f"📁 日志文件数: {len(data['files'])}")
        
        # 打印文件概览
        total_size = sum(f.get('size', 0) for f in data['files'])
        print(f"📦 总文件大小: {total_size/1024/1024:.2f} MB")
        
        # 时间范围
        if data['timestamps']:
            min_time = min(data['timestamps'])
            max_time = max(data['timestamps'])
            print(f"⏰ 时间范围: {min_time} ~ {max_time}")
        
        # 模式统计
        print(f"\n📋 发现的日志模式:")
        for pattern_name, matches in data['patterns'].items():
            if matches:
                print(f"   • {pattern_name}: {len(matches)} 条记录")
        
        # 频率使用统计
        if data['frequency_stats']:
            print(f"\n🔢 频率使用统计 (Top 10):")
            for freq, count in data['frequency_stats'].most_common(10):
                print(f"   • {freq}MHz: {count} 次")
        
        print("-" * 60)

def export_detailed_results(results, output_file='frequency_analysis_results.json'):
    """导出详细结果到JSON文件"""
    # 转换datetime对象为字符串以便JSON序列化
    serializable_results = {}
    for gpu_type, data in results.items():
        serializable_results[gpu_type] = {
            'files': data['files'],
            'patterns': dict(data['patterns']),
            'frequency_stats': dict(data['frequency_stats']),
            'timestamps': [ts.isoformat() for ts in data['timestamps']]
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细结果已导出到: {output_file}")

def show_sample_patterns(results, limit=5):
    """显示各种模式的示例"""
    print("\n📋 各类日志模式示例:")
    print("="*80)
    
    for gpu_type, data in results.items():
        if not any(data['patterns'].values()):
            continue
            
        print(f"\n🎮 GPU: {gpu_type}")
        
        for pattern_name, matches in data['patterns'].items():
            if matches:
                print(f"\n   📌 {pattern_name} 模式示例:")
                for i, match in enumerate(matches[:limit]):
                    timestamp = match.get('timestamp', '未知时间')
                    line = match['line'][:100] + '...' if len(match['line']) > 100 else match['line']
                    print(f"      [{timestamp}] {line}")
                    
                if len(matches) > limit:
                    print(f"      ... 还有 {len(matches) - limit} 条记录")

if __name__ == "__main__":
    print("🔍 开始分析日志文件中的频率信息...")
    
    # 执行分析
    results = find_frequency_patterns()
    
    # 打印摘要
    print_analysis_summary(results)
    
    # 显示模式示例
    show_sample_patterns(results)
    
    # 导出详细结果
    export_detailed_results(results)
    
    print("\n✅ 分析完成!")