import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
from .logger import setup_logger

logger = setup_logger(__name__)


class EDPRewardCalculator:
    """简化的EDP奖励计算器 - 统一使用EDP优化"""

    def __init__(self, ttft_limit: float = 2.0, tpot_limit: float = 0.2, 
                 ignore_slo: bool = False):
        # 配置参数
        self.ttft_limit = ttft_limit   # TTFT硬限制 (秒)
        self.tpot_limit = tpot_limit   # TPOT硬限制 (秒)
        self.ignore_slo = ignore_slo   # 无视SLO模式

        # 自适应基线 - 近期p50作为归一化分母
        self.edp_p50_history = deque(maxlen=100)
        self.edp_history = deque(maxlen=50)
        

        logger.info(
            f"💰 初始化简化EDP奖励计算器:\n"
            f"   TTFT限制: ≤{ttft_limit}s {'(忽略)' if ignore_slo else ''}\n"
            f"   TPOT限制: ≤{tpot_limit}s {'(忽略)' if ignore_slo else ''}\n"
            f"   无视SLO模式: {'启用' if ignore_slo else '关闭'}"
        )

    def calculate(
        self,
        counter_deltas: Dict[str, float],
        energy_consumed_mj: float
    ) -> Tuple[float, Dict]:
        """
        计算基于EDP的奖励

        Parameters
        ----------
        counter_deltas : Dict[str, float]
            Prometheus计数器增量
        energy_consumed_mj : float
            能耗（毫焦）
        """
        # 提取TPOT数据
        tpot_count = counter_deltas.get('vllm:time_per_output_token_seconds_count_delta', 0)
        tpot_sum = counter_deltas.get('vllm:time_per_output_token_seconds_sum_delta', 0)

        # 检查是否有请求
        if tpot_count == 0:
            logger.debug("⚠️ 这个周期内没有请求完成")
            return 0.0, {"no_requests": True}

        # 计算平均TPOT延迟
        avg_tpot = tpot_sum / tpot_count

        # 移除预热检查 - 直接开始计算，避免初始0奖励问题

        # 计算EDP
        energy_j = energy_consumed_mj / 1000.0  # 转换为焦耳
        edp = energy_j * avg_tpot
        self.edp_history.append(edp)
        self.edp_p50_history.append(edp)
        
        # 计算EDP得分 - 使用自适应p50基线
        def _get_p50_baseline(history_deque):
            if len(history_deque) < 10:  
                return None
            values = list(history_deque)
            return np.percentile(values, 50)
        
        edp_p50 = _get_p50_baseline(self.edp_p50_history)
        
        if edp_p50 and edp_p50 > 0:
            edp_norm = edp / edp_p50
            # EDP越小越好，所以得分是 1 - normalized_edp
            edp_score = 2.0 * (1.0 - min(2.0, edp_norm))  # [-2, 2]
        else:
            # 初始情况：给予中性分数
            edp_score = 0.01
            logger.debug(f"使用初始基线，EDP得分: {edp_score:.3f}")

        # 最终奖励计算（纯EDP优化）
        final_reward = edp_score

        # 构建详细信息
        info = {
            'avg_tpot': avg_tpot,
            'energy_j': energy_j,
            'edp': edp,
            'edp_p50': edp_p50,
            'edp_score': edp_score,
            'final_reward': final_reward
        }

        # 日志输出
        logger.info(
            f"💰 EDP奖励计算:\n"
            f"   TPOT: {avg_tpot:.3f}s, 能耗: {energy_j:.4f}J\n"
            f"   EDP: {edp:.4f}, 最终奖励: {final_reward:+.3f}"
        )

        return float(final_reward), info

    def calculate_reward(self, post_metrics: dict, energy_delta: float) -> float:
        """适配器方法，用于与main.py的调用方式兼容"""
        # 从vLLM指标中提取计数器增量
        counter_deltas = {}
        
        # 从post_metrics中提取需要的信息
        # 这是一个简化版本，假设metrics_collector已经计算了delta
        ttft_sum_key = "vllm:time_to_first_token_seconds_sum_delta"
        ttft_count_key = "vllm:time_to_first_token_seconds_count_delta"
        tpot_sum_key = "vllm:time_per_output_token_seconds_sum_delta"
        tpot_count_key = "vllm:time_per_output_token_seconds_count_delta"
        
        # 如果metrics已经包含delta，直接使用
        if ttft_sum_key in post_metrics:
            counter_deltas[ttft_sum_key] = post_metrics[ttft_sum_key]
        if ttft_count_key in post_metrics:
            counter_deltas[ttft_count_key] = post_metrics[ttft_count_key]
        if tpot_sum_key in post_metrics:
            counter_deltas[tpot_sum_key] = post_metrics[tpot_sum_key]
        if tpot_count_key in post_metrics:
            counter_deltas[tpot_count_key] = post_metrics[tpot_count_key]
        
        
        
        # 调用原始的calculate方法
        reward, _ = self.calculate(
            counter_deltas=counter_deltas,
            energy_consumed_mj=energy_delta
        )
        
        return reward

    def check_slo_violation(self, metrics: dict, frequency: int) -> bool:
        """检查是否发生SLO违规"""
        if self.ignore_slo:
            return False
            
        # 从metrics中提取TTFT和TPOT数据
        ttft_count = metrics.get('vllm:time_to_first_token_seconds_count_delta', 0)
        ttft_sum = metrics.get('vllm:time_to_first_token_seconds_sum_delta', 0)
        tpot_count = metrics.get('vllm:time_per_output_token_seconds_count_delta', 0)
        tpot_sum = metrics.get('vllm:time_per_output_token_seconds_sum_delta', 0)
        
        # 检查是否有足够的数据
        if ttft_count == 0 and tpot_count == 0:
            return False
            
        slo_violated = False
        
        # 检查TTFT限制
        if ttft_count > 0:
            avg_ttft = ttft_sum / ttft_count
            if avg_ttft > self.ttft_limit:
                logger.warning(f"⚠️ TTFT SLO违规: {avg_ttft:.3f}s > {self.ttft_limit}s @ {frequency}MHz")
                slo_violated = True
        
        # 检查TPOT限制
        if tpot_count > 0:
            avg_tpot = tpot_sum / tpot_count
            if avg_tpot > self.tpot_limit:
                logger.warning(f"⚠️ TPOT SLO违规: {avg_tpot:.3f}s > {self.tpot_limit}s @ {frequency}MHz")
                slo_violated = True
                
        return slo_violated

    def get_stats(self) -> dict:
        """获取奖励计算器的统计信息"""
        stats = {
            'edp_history_len': len(self.edp_history)
        }
        
        if self.edp_history:
            stats['edp_recent_avg'] = np.mean(list(self.edp_history)[-10:])
            stats['edp_recent_std'] = np.std(list(self.edp_history)[-10:])
            
        return stats

        
    
    
        
    
    
