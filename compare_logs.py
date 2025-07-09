#!/usr/bin/env python3
"""
Log Comparison Script for vLLM Multi-Armed Bandit GPU Autoscaler

This script compares two log files:
1. Default frequency (data collection only mode)
2. Normal frequency adjustment mode

Generates three comparison plots:
- EDP (Energy-Delay Product) vs Decision Rounds
- TPOT (Time Per Output Token) vs Decision Rounds  
- Power Consumption vs Decision Rounds
"""

import re
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import seaborn as sns
from datetime import datetime

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class LogAnalyzer:
    """Analyzer for vLLM MAB log files"""
    
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.data = []
        
    def parse_log(self) -> List[Dict]:
        """Parse log file and extract performance metrics"""
        print(f"📖 Parsing log file: {self.log_path}")
        
        if not self.log_path.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_path}")
            
        with open(self.log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find all JSON data entries - improved pattern
        json_pattern = r'📋 JSON数据: ({.*?})\s*$'
        json_matches = re.findall(json_pattern, content, re.MULTILINE)
        
        parsed_data = []
        for i, match in enumerate(json_matches):
            try:
                # Clean up the JSON string
                clean_json = match.strip()
                # Handle potential floating point precision issues
                data = json.loads(clean_json)
                
                # Extract key metrics
                round_num = data.get('round', 0)
                frequency = data.get('decision', {}).get('selected_frequency', 0)
                
                performance = data.get('performance', {})
                edp_value = performance.get('edp_value', 0)
                avg_tpot = performance.get('avg_tpot', 0)
                energy_delta_j = performance.get('energy_delta_j', 0)
                
                reward = data.get('reward', {}).get('current', 0)
                
                parsed_data.append({
                    'round': round_num,
                    'frequency': frequency,
                    'edp': edp_value,
                    'tpot': avg_tpot,
                    'power_j': energy_delta_j,
                    'reward': reward
                })
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Failed to parse JSON entry {i+1}: {e}")
                # Print first 100 chars for debugging
                if len(clean_json) > 100:
                    print(f"   JSON preview: {clean_json[:100]}...")
                else:
                    print(f"   JSON content: {clean_json}")
                continue
                
        print(f"✅ Parsed {len(parsed_data)} data points from {self.log_path.name}")
        self.data = parsed_data
        return parsed_data
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for the parsed data"""
        if not self.data:
            return {}
            
        edp_values = [d['edp'] for d in self.data if d.get('edp') is not None and d['edp'] > 0]
        tpot_values = [d['tpot'] for d in self.data if d.get('tpot') is not None and d['tpot'] > 0]
        power_values = [d['power_j'] for d in self.data if d.get('power_j') is not None and d['power_j'] > 0]
        
        return {
            'total_rounds': len(self.data),
            'avg_edp': np.mean(edp_values) if edp_values else 0,
            'avg_tpot': np.mean(tpot_values) if tpot_values else 0,
            'avg_power': np.mean(power_values) if power_values else 0,
            'min_edp': np.min(edp_values) if edp_values else 0,
            'min_tpot': np.min(tpot_values) if tpot_values else 0,
            'frequency_range': (
                min(d['frequency'] for d in self.data),
                max(d['frequency'] for d in self.data)
            ) if self.data else (0, 0)
        }

def create_comparison_plots(default_analyzer: LogAnalyzer, adaptive_analyzer: LogAnalyzer, 
                          output_dir: str = "comparison_plots"):
    """Create three comparison plots for EDP, TPOT, and Power"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Prepare data
    default_data = default_analyzer.data
    adaptive_data = adaptive_analyzer.data
    
    if not default_data or not adaptive_data:
        print("❌ No data available for plotting")
        return
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('vLLM Multi-Armed Bandit: Default vs Adaptive Frequency Control', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: EDP Comparison
    ax1 = axes[0]
    default_rounds = [d['round'] for d in default_data if d.get('edp') is not None and d['edp'] > 0]
    default_edp = [d['edp'] for d in default_data if d.get('edp') is not None and d['edp'] > 0]
    adaptive_rounds = [d['round'] for d in adaptive_data if d.get('edp') is not None and d['edp'] > 0]
    adaptive_edp = [d['edp'] for d in adaptive_data if d.get('edp') is not None and d['edp'] > 0]
    
    ax1.plot(default_rounds, default_edp, 'o-', alpha=0.7, label='Default Frequency', 
             color='#ff7f0e', markersize=3, linewidth=1.5)
    ax1.plot(adaptive_rounds, adaptive_edp, 's-', alpha=0.7, label='Adaptive Frequency', 
             color='#2ca02c', markersize=3, linewidth=1.5)
    
    ax1.set_xlabel('Decision Rounds')
    ax1.set_ylabel('EDP (Energy-Delay Product)')
    ax1.set_title('EDP Comparison: Default vs Adaptive Frequency Control')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add rolling average lines
    if len(default_edp) >= 10:
        default_edp_smooth = np.convolve(default_edp, np.ones(10)/10, mode='valid')
        default_rounds_smooth = default_rounds[9:]
        ax1.plot(default_rounds_smooth, default_edp_smooth, '--', alpha=0.8, 
                color='#ff7f0e', linewidth=2, label='Default (10-round avg)')
        
    if len(adaptive_edp) >= 10:
        adaptive_edp_smooth = np.convolve(adaptive_edp, np.ones(10)/10, mode='valid')
        adaptive_rounds_smooth = adaptive_rounds[9:]
        ax1.plot(adaptive_rounds_smooth, adaptive_edp_smooth, '--', alpha=0.8, 
                color='#2ca02c', linewidth=2, label='Adaptive (10-round avg)')
    
    # Plot 2: TPOT Comparison
    ax2 = axes[1]
    default_tpot = [d['tpot'] for d in default_data if d.get('tpot') is not None and d['tpot'] > 0]
    adaptive_tpot = [d['tpot'] for d in adaptive_data if d.get('tpot') is not None and d['tpot'] > 0]
    default_rounds_tpot = [d['round'] for d in default_data if d.get('tpot') is not None and d['tpot'] > 0]
    adaptive_rounds_tpot = [d['round'] for d in adaptive_data if d.get('tpot') is not None and d['tpot'] > 0]
    
    ax2.plot(default_rounds_tpot, default_tpot, 'o-', alpha=0.7, label='Default Frequency', 
             color='#ff7f0e', markersize=3, linewidth=1.5)
    ax2.plot(adaptive_rounds_tpot, adaptive_tpot, 's-', alpha=0.7, label='Adaptive Frequency', 
             color='#2ca02c', markersize=3, linewidth=1.5)
    
    ax2.set_xlabel('Decision Rounds')
    ax2.set_ylabel('TPOT (Time Per Output Token, seconds)')
    ax2.set_title('TPOT Comparison: Default vs Adaptive Frequency Control')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Power Consumption Comparison
    ax3 = axes[2]
    default_power = [d['power_j'] for d in default_data if d.get('power_j') is not None and d['power_j'] > 0]
    adaptive_power = [d['power_j'] for d in adaptive_data if d.get('power_j') is not None and d['power_j'] > 0]
    default_rounds_power = [d['round'] for d in default_data if d.get('power_j') is not None and d['power_j'] > 0]
    adaptive_rounds_power = [d['round'] for d in adaptive_data if d.get('power_j') is not None and d['power_j'] > 0]
    
    ax3.plot(default_rounds_power, default_power, 'o-', alpha=0.7, label='Default Frequency', 
             color='#ff7f0e', markersize=3, linewidth=1.5)
    ax3.plot(adaptive_rounds_power, adaptive_power, 's-', alpha=0.7, label='Adaptive Frequency', 
             color='#2ca02c', markersize=3, linewidth=1.5)
    
    ax3.set_xlabel('Decision Rounds')
    ax3.set_ylabel('Power Consumption (Joules)')
    ax3.set_title('Power Consumption Comparison: Default vs Adaptive Frequency Control')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = output_path / f"comparison_plots_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plots saved to: {plot_filename}")
    
    # Save individual plots
    individual_dir = output_path / "individual"
    individual_dir.mkdir(exist_ok=True)
    
    # Save EDP plot
    fig1, ax = plt.subplots(figsize=(10, 6))
    ax.plot(default_rounds, default_edp, 'o-', alpha=0.7, label='Default Frequency', 
            color='#ff7f0e', markersize=3, linewidth=1.5)
    ax.plot(adaptive_rounds, adaptive_edp, 's-', alpha=0.7, label='Adaptive Frequency', 
            color='#2ca02c', markersize=3, linewidth=1.5)
    ax.set_xlabel('Decision Rounds')
    ax.set_ylabel('EDP (Energy-Delay Product)')
    ax.set_title('EDP Comparison: Default vs Adaptive Frequency Control')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(individual_dir / f"edp_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save TPOT plot
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(default_rounds_tpot, default_tpot, 'o-', alpha=0.7, label='Default Frequency', 
            color='#ff7f0e', markersize=3, linewidth=1.5)
    ax.plot(adaptive_rounds_tpot, adaptive_tpot, 's-', alpha=0.7, label='Adaptive Frequency', 
            color='#2ca02c', markersize=3, linewidth=1.5)
    ax.set_xlabel('Decision Rounds')
    ax.set_ylabel('TPOT (Time Per Output Token, seconds)')
    ax.set_title('TPOT Comparison: Default vs Adaptive Frequency Control')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(individual_dir / f"tpot_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save Power plot
    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.plot(default_rounds_power, default_power, 'o-', alpha=0.7, label='Default Frequency', 
            color='#ff7f0e', markersize=3, linewidth=1.5)
    ax.plot(adaptive_rounds_power, adaptive_power, 's-', alpha=0.7, label='Adaptive Frequency', 
            color='#2ca02c', markersize=3, linewidth=1.5)
    ax.set_xlabel('Decision Rounds')
    ax.set_ylabel('Power Consumption (Joules)')
    ax.set_title('Power Consumption Comparison: Default vs Adaptive Frequency Control')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(individual_dir / f"power_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Individual plots saved to: {individual_dir}")
    
    # Show the combined plot
    plt.show()

def print_summary_comparison(default_analyzer: LogAnalyzer, adaptive_analyzer: LogAnalyzer):
    """Print summary statistics comparison"""
    default_stats = default_analyzer.get_summary_stats()
    adaptive_stats = adaptive_analyzer.get_summary_stats()
    
    print("\n" + "="*60)
    print("📊 SUMMARY STATISTICS COMPARISON")
    print("="*60)
    
    # Check if we have valid data
    if not default_stats or not adaptive_stats:
        print("❌ No valid statistics available for comparison")
        return
    
    print(f"{'Metric':<25} {'Default':<15} {'Adaptive':<15} {'Improvement':<15}")
    print("-" * 70)
    
    # EDP comparison
    if default_stats.get('avg_edp', 0) > 0 and adaptive_stats.get('avg_edp', 0) > 0:
        edp_improvement = ((default_stats['avg_edp'] - adaptive_stats['avg_edp']) / default_stats['avg_edp']) * 100
        print(f"{'Average EDP':<25} {default_stats['avg_edp']:<15.2f} {adaptive_stats['avg_edp']:<15.2f} {edp_improvement:<14.1f}%")
    
    # TPOT comparison
    if default_stats.get('avg_tpot', 0) > 0 and adaptive_stats.get('avg_tpot', 0) > 0:
        tpot_improvement = ((default_stats['avg_tpot'] - adaptive_stats['avg_tpot']) / default_stats['avg_tpot']) * 100
        print(f"{'Average TPOT':<25} {default_stats['avg_tpot']:<15.3f} {adaptive_stats['avg_tpot']:<15.3f} {tpot_improvement:<14.1f}%")
    
    # Power comparison
    if default_stats.get('avg_power', 0) > 0 and adaptive_stats.get('avg_power', 0) > 0:
        power_improvement = ((default_stats['avg_power'] - adaptive_stats['avg_power']) / default_stats['avg_power']) * 100
        print(f"{'Average Power (J)':<25} {default_stats['avg_power']:<15.1f} {adaptive_stats['avg_power']:<15.1f} {power_improvement:<14.1f}%")
    
    print("-" * 70)
    print(f"{'Total Rounds':<25} {default_stats.get('total_rounds', 0):<15} {adaptive_stats.get('total_rounds', 0):<15}")
    print(f"{'Min EDP':<25} {default_stats.get('min_edp', 0):<15.2f} {adaptive_stats.get('min_edp', 0):<15.2f}")
    freq_range_default = default_stats.get('frequency_range', (0, 0))
    freq_range_adaptive = adaptive_stats.get('frequency_range', (0, 0))
    print(f"{'Frequency Range':<25} {freq_range_default[0]}-{freq_range_default[1]:<10} {freq_range_adaptive[0]}-{freq_range_adaptive[1]}")

def main():
    parser = argparse.ArgumentParser(description='Compare vLLM MAB log files')
    parser.add_argument('default_log', help='Path to default frequency log file')
    parser.add_argument('adaptive_log', help='Path to adaptive frequency log file')
    parser.add_argument('--output-dir', default='comparison_plots', 
                       help='Output directory for plots (default: comparison_plots)')
    parser.add_argument('--no-plot', action='store_true', 
                       help='Skip generating plots, only show statistics')
    
    args = parser.parse_args()
    
    try:
        # Analyze both log files
        print("🚀 Starting log comparison analysis...")
        
        default_analyzer = LogAnalyzer(args.default_log)
        default_analyzer.parse_log()
        
        adaptive_analyzer = LogAnalyzer(args.adaptive_log)
        adaptive_analyzer.parse_log()
        
        # Print summary comparison
        print_summary_comparison(default_analyzer, adaptive_analyzer)
        
        if not args.no_plot:
            # Generate comparison plots
            create_comparison_plots(default_analyzer, adaptive_analyzer, args.output_dir)
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())