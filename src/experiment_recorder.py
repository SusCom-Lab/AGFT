"""
实验数据记录器 - 用于保存每轮的实验数据到文件
支持JSON Lines格式，便于后续分析
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class ExperimentRecorder:
    """实验数据记录器类"""
    
    def __init__(self, experiment_name: Optional[str] = None, base_dir: str = "experiment"):
        """
        初始化实验记录器
        
        Args:
            experiment_name: 实验名称，如果为None则自动生成
            base_dir: 基础目录，默认为experiment
        """
        self.base_dir = Path(base_dir)
        self.experiment_start_time = datetime.now()
        
        # 生成实验名称和目录
        if experiment_name is None:
            timestamp = self.experiment_start_time.strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"vllm_mab_exp_{timestamp}"
        else:
            self.experiment_name = experiment_name
            
        # 创建实验目录
        self.experiment_dir = self.base_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据文件路径
        self.rounds_file = self.experiment_dir / "rounds_data.jsonl"
        self.metadata_file = self.experiment_dir / "experiment_metadata.json"
        
        # 初始化元数据
        self.metadata = {
            'experiment_name': self.experiment_name,
            'start_time': self.experiment_start_time.isoformat(),
            'start_timestamp': time.time(),
            'end_time': None,
            'end_timestamp': None,
            'total_rounds': 0,
            'gpu_model': None,
            'config_used': None,
            'convergence_round': None,
            'best_edp_achieved': None,
            'best_frequency': None,
            'total_energy_consumed_j': 0.0,
            'final_performance_summary': {}
        }
        
        print(f"📊 实验数据记录器已初始化")
        print(f"   实验名称: {self.experiment_name}")
        print(f"   数据目录: {self.experiment_dir}")
        print(f"   数据文件: {self.rounds_file}")
    
    def save_config_snapshot(self, config_path: str):
        """保存配置文件快照"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            config_file = self.experiment_dir / "config_snapshot.yaml"
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            self.metadata['config_used'] = config_data
            print(f"✅ 配置文件快照已保存: {config_file}")
            
        except Exception as e:
            print(f"⚠️ 保存配置文件快照失败: {e}")
    
    def record_round_data(self, round_data: Dict[str, Any]):
        """
        记录单轮数据到JSONL文件
        
        Args:
            round_data: 包含本轮所有数据的字典
        """
        try:
            # 确保能耗单位转换为焦耳
            if 'energy_mj' in round_data:
                round_data['energy_j'] = round_data['energy_mj'] / 1000.0
                del round_data['energy_mj']  # 删除毫焦耳版本
            
            # 添加时间戳信息
            if 'timestamp' not in round_data:
                round_data['timestamp'] = time.time()
            if 'datetime' not in round_data:
                round_data['datetime'] = datetime.now().isoformat()
            
            # 写入JSONL文件
            with open(self.rounds_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(round_data, ensure_ascii=False) + '\n')
            
            # 更新元数据
            self.metadata['total_rounds'] = round_data.get('round', self.metadata['total_rounds'])
            
            # 跟踪最佳EDP
            if 'edp_raw' in round_data:
                current_edp = round_data['edp_raw']
                if self.metadata['best_edp_achieved'] is None or current_edp < self.metadata['best_edp_achieved']:
                    self.metadata['best_edp_achieved'] = current_edp
                    self.metadata['best_frequency'] = round_data.get('gpu_frequency_mhz')
            
            # 跟踪收敛轮次
            if round_data.get('learning_phase') == 'EXPLOITATION' and self.metadata['convergence_round'] is None:
                self.metadata['convergence_round'] = round_data.get('round')
            
            # 累计能耗
            if 'energy_j' in round_data:
                self.metadata['total_energy_consumed_j'] += round_data['energy_j']
                
        except Exception as e:
            print(f"❌ 记录轮次数据失败: {e}")
    
    def set_gpu_model(self, gpu_model: str):
        """设置GPU型号信息"""
        self.metadata['gpu_model'] = gpu_model
    
    def finalize_experiment(self):
        """结束实验，保存最终元数据"""
        try:
            self.metadata['end_time'] = datetime.now().isoformat()
            self.metadata['end_timestamp'] = time.time()
            
            # 计算实验总时长
            duration = self.metadata['end_timestamp'] - self.metadata['start_timestamp']
            self.metadata['duration_seconds'] = duration
            self.metadata['duration_hours'] = duration / 3600.0
            
            # 计算平均功率
            if self.metadata['total_energy_consumed_j'] > 0 and duration > 0:
                self.metadata['average_power_w'] = self.metadata['total_energy_consumed_j'] / duration
            
            # 保存元数据
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 实验已结束，元数据已保存")
            print(f"   总轮次: {self.metadata['total_rounds']}")
            print(f"   实验时长: {self.metadata.get('duration_hours', 0):.2f} 小时")
            print(f"   总能耗: {self.metadata['total_energy_consumed_j']:.3f} 焦耳")
            print(f"   最佳EDP: {self.metadata['best_edp_achieved']}")
            print(f"   数据文件: {self.rounds_file}")
            
        except Exception as e:
            print(f"❌ 保存实验元数据失败: {e}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """获取实验摘要信息"""
        return {
            'experiment_name': self.experiment_name,
            'experiment_dir': str(self.experiment_dir),
            'rounds_recorded': self.metadata['total_rounds'],
            'start_time': self.metadata['start_time'],
            'best_edp': self.metadata['best_edp_achieved'],
            'best_frequency': self.metadata['best_frequency'],
            'convergence_round': self.metadata['convergence_round'],
            'total_energy_j': self.metadata['total_energy_consumed_j']
        }


def create_round_data_dict(
    round_num: int,
    gpu_frequency: int,
    energy_j: float,
    ttft: float,
    tpot: float,
    e2e: float,
    total_throughput: float,
    prefill_throughput: float,
    decode_throughput: float,
    edp_raw: float,
    edp_normalized: float,
    edp_baseline: float,
    active_requests: int,
    cache_usage: float,
    completed_requests: int,
    learning_phase: str,
    reward: float,
    alpha: float,
    ucb_confidence: float,
    action_method: str,
    available_freqs: int,
    pruned_freqs: int,
    freq_exploration_count: int,
    slo_violation: bool,
    ttft_limit: float,
    tpot_limit: float,
    features: Dict[str, float]
) -> Dict[str, Any]:
    """
    创建标准的轮次数据字典
    
    Args:
        各种实验数据参数
    
    Returns:
        格式化的轮次数据字典
    """
    return {
        # 基本信息
        'round': round_num,
        'timestamp': time.time(),
        'datetime': datetime.now().isoformat(),
        'gpu_frequency_mhz': gpu_frequency,
        
        # 能耗和延迟
        'energy_j': energy_j,
        'ttft_avg': ttft,
        'tpot_avg': tpot,
        'e2e_avg': e2e,
        
        # 吞吐量指标
        'total_throughput_tps': total_throughput,
        'prefill_throughput_tps': prefill_throughput,
        'decode_throughput_tps': decode_throughput,
        
        # EDP核心指标
        'edp_raw': edp_raw,
        'edp_normalized': edp_normalized,
        'edp_baseline': edp_baseline,
        'edp_improvement_pct': (edp_baseline - edp_raw) / edp_baseline * 100 if edp_baseline > 0 else 0,
        
        # 系统状态
        'active_requests': active_requests,
        'cache_usage_pct': cache_usage,
        'completed_requests_delta': completed_requests,
        
        # 学习算法状态
        'learning_phase': learning_phase,
        'reward': reward,
        'alpha_value': alpha,
        'ucb_confidence': ucb_confidence,
        'action_selection_method': action_method,
        
        # 频率管理
        'available_frequencies_count': available_freqs,
        'pruned_frequencies_count': pruned_freqs,
        'frequency_exploration_count': freq_exploration_count,
        
        # 性能约束
        'slo_violation': slo_violation,
        'ttft_limit': ttft_limit,
        'tpot_limit': tpot_limit,
        
        # 上下文特征
        'has_queue': features.get('has_queue', 0),
        'prefill_throughput_feature': features.get('prefill_throughput', 0),
        'decode_throughput_feature': features.get('decode_throughput', 0),
        'packing_efficiency': features.get('packing_efficiency', 0),
        'concurrency': features.get('concurrency', 0),
        'gpu_cache_usage': features.get('gpu_cache_usage', 0),
        'cache_hit_rate': features.get('cache_hit_rate', 0),
    }