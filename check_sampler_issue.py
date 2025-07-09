#!/usr/bin/env python3
"""
检查自适应采样器问题
"""

import re
from pathlib import Path

def check_debug_logs(log_path):
    """检查DEBUG级别的日志"""
    print("=== 检查DEBUG级别日志 ===")
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 搜索DEBUG日志
    debug_lines = []
    for i, line in enumerate(content.split('\n')):
        if 'DEBUG' in line and ('adaptive' in line.lower() or '自适应' in line or '细化' in line):
            debug_lines.append((i+1, line))
    
    if debug_lines:
        print(f"找到 {len(debug_lines)} 条相关DEBUG日志:")
        for line_num, line in debug_lines[:10]:  # 显示前10条
            print(f"  {line_num}: {line}")
    else:
        print("未找到相关DEBUG日志")
    
    return debug_lines

def check_gpu_controller_logs(log_path):
    """检查GPU控制器相关日志"""
    print("\n=== 检查GPU控制器日志 ===")
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 搜索GPU控制器初始化相关日志
    patterns = [
        r'GPU.*控制器',
        r'自适应.*采样器.*初始化',
        r'失败.*采样器',
        r'错误.*采样器',
        r'AdaptiveFrequencySampler',
        r'adaptive.*sampler.*failed',
        r'频率.*生成.*方案'
    ]
    
    found_any = False
    for pattern in patterns:
        matches = []
        for i, line in enumerate(content.split('\n')):
            if re.search(pattern, line, re.IGNORECASE):
                matches.append((i+1, line))
        
        if matches:
            found_any = True
            print(f"模式 '{pattern}' 匹配:")
            for line_num, line in matches[:5]:
                print(f"  {line_num}: {line.strip()}")
    
    if not found_any:
        print("未找到GPU控制器或自适应采样器相关日志")

def check_initialization_sequence(log_path):
    """检查初始化序列"""
    print("\n=== 检查初始化序列 ===")
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 查找初始化相关的前50行日志
    print("前50行初始化日志:")
    for i, line in enumerate(lines[:50]):
        if any(keyword in line for keyword in ['初始化', '启动', 'init', '配置', '组件']):
            print(f"  {i+1:2d}: {line.strip()}")

def analyze_round_45_context():
    """分析第45轮的上下文"""
    print("\n=== 第45轮上下文分析 ===")
    
    # 检查adaptive_frequency_sampler.py中的refinement_count逻辑
    sampler_file = Path("src/adaptive_frequency_sampler.py")
    if sampler_file.exists():
        with open(sampler_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找refinement_count相关代码
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'refinement_count' in line:
                context_start = max(0, i-2)
                context_end = min(len(lines), i+3)
                print(f"第{i+1}行附近的refinement_count逻辑:")
                for j in range(context_start, context_end):
                    marker = ">>> " if j == i else "    "
                    print(f"{marker}{j+1:3d}: {lines[j]}")
                print()
    
    # 分析如果refinement_count独立计数，第45轮时实际的计数可能是多少
    print("分析: 如果refinement_count独立计数")
    print("- 每轮调用_update_adaptive_sampler时，refinement_count += 1")
    print("- 第45轮时，refinement_count应该是45")  
    print("- 45 % 45 = 0，应该触发细化")
    print("- 但如果有条件跳过调用，计数可能不同")

def check_update_calls():
    """检查update调用的条件"""
    print("\n=== 检查update调用条件 ===")
    
    main_file = Path("src/main.py")
    if main_file.exists():
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找_update_adaptive_sampler调用的条件
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '_update_adaptive_sampler' in line:
                # 显示前后10行上下文
                context_start = max(0, i-10)
                context_end = min(len(lines), i+5)
                print(f"第{i+1}行_update_adaptive_sampler调用上下文:")
                for j in range(context_start, context_end):
                    marker = ">>> " if j == i else "    "
                    print(f"{marker}{j+1:3d}: {lines[j]}")
                print()

def main():
    log_path = "logs/RTX_A6000/vllm_gpu_autoscaler_20250702_144110.log"
    
    if not Path(log_path).exists():
        print(f"错误: 日志文件不存在: {log_path}")
        return
    
    # 逐步检查各种可能的问题
    check_debug_logs(log_path)
    check_gpu_controller_logs(log_path)
    check_initialization_sequence(log_path)
    analyze_round_45_context()
    check_update_calls()
    
    print("\n=== 问题诊断 ===")
    print("基于分析，可能的问题:")
    print("1. 自适应采样器初始化失败，使用了后备方案")
    print("2. DEBUG级别的细化检查日志被过滤")
    print("3. update_frequency_reward方法内部有异常但被捕获")
    print("4. refinement_count计数与预期不符")
    
    print("\n建议验证:")
    print("1. 检查GPU控制器初始化时是否有错误")
    print("2. 临时降低日志级别为DEBUG重新运行")
    print("3. 在代码中添加更详细的日志输出")
    print("4. 验证self.adaptive_sampler是否为None")

if __name__ == "__main__":
    main()