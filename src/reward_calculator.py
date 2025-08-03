import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
from .logger import setup_logger

logger = setup_logger(__name__)


class EDPRewardCalculator:
    """简化的EDP奖励计算器 - 统一使用EDP优化"""

    def __init__(self, ttft_limit: float = 2.0, tpot_limit: float = 0.2, 
                 ignore_slo: bool = False, sampling_duration: float = 0.8, 
                 baseline_measurements: int = 5):
        # 配置参数
        self.ttft_limit = ttft_limit   # TTFT硬限制 (秒)
        self.tpot_limit = tpot_limit   # TPOT硬限制 (秒)
        self.ignore_slo = ignore_slo   # 无视SLO模式
        self.sampling_duration = sampling_duration  # 采样窗口时长

        # 采样窗口EDP基线 - 支持多次测量
        self.baseline_edp = None  # 不锁频采样窗口EDP基线
        self.baseline_collected = False
        self.baseline_measurements = []  # 存储多次基线测量
        self.baseline_target_count = baseline_measurements  # 目标测量次数，从配置文件读取
        self.edp_history = deque(maxlen=50)
        

        logger.info(
            f"💰 初始化简化EDP奖励计算器:\n"
            f"   TTFT限制: ≤{ttft_limit}s {'(忽略)' if ignore_slo else ''}\n"
            f"   TPOT限制: ≤{tpot_limit}s {'(忽略)' if ignore_slo else ''}\n"
            f"   无视SLO模式: {'启用' if ignore_slo else '关闭'}\n"
            f"   EDP基线: 不锁频{sampling_duration}秒采样 (测量{self.baseline_target_count}次取平均)"
        )

    def set_baseline_edp(self, baseline_edp: float):
        """设置不锁频采样窗口的EDP基线"""
        self.baseline_edp = baseline_edp
        self.baseline_collected = True
        logger.info(
            f"🎯 设置EDP基线:\n"
            f"   不锁频{self.sampling_duration}秒采样EDP: {baseline_edp:.6f} J·s\n"
            f"   基线状态: 已收集"
        )

    def calculate(
        self,
        counter_deltas: Dict[str, float],
        energy_consumed_j: float,
        is_baseline_collection: bool = False
    ) -> Tuple[float, Dict]:
        """
        计算基于EDP的奖励

        Parameters
        ----------
        counter_deltas : Dict[str, float]
            Prometheus计数器增量
        energy_consumed_j : float
            能耗（焦耳）
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
        energy_j = energy_consumed_j  # 已经是焦耳
        edp = energy_j * avg_tpot
        
        # 只有当TPOT > 0时才记录有效的EDP
        if avg_tpot > 0:
            self.edp_history.append(edp)
        else:
            # 没有请求完成时，不记录EDP（避免0值污染历史数据）
            logger.debug(f"⏭️ TPOT=0，跳过EDP记录（避免0值污染）")
        
        # 如果是基线收集，收集多次测量（只记录有效的EDP）
        if is_baseline_collection and avg_tpot > 0:
            self.baseline_measurements.append(edp)
            current_count = len(self.baseline_measurements)
            
            if current_count < self.baseline_target_count:
                # 还需要更多测量
                logger.info(f"🎯 基线测量 {current_count}/{self.baseline_target_count}: EDP={edp:.6f} J·s")
                return 0.0, {
                    "baseline_collection": True,
                    "baseline_edp": edp,
                    "edp_raw": edp,
                    "reward_type": "baseline_measuring",
                    "measurements_count": current_count,
                    "target_count": self.baseline_target_count
                }
            else:
                # 完成所有测量，计算平均值
                import numpy as np
                avg_baseline = np.mean(self.baseline_measurements)*0.9
                self.set_baseline_edp(avg_baseline)
                
                logger.info(f"🎯 基线收集完成 {self.baseline_target_count}次测量:")
                for i, measurement in enumerate(self.baseline_measurements, 1):
                    logger.info(f"   测量{i}: {measurement:.6f} J·s")
                logger.info(f"   平均基线: {avg_baseline:.6f} J·s")
                
                return 0.0, {
                    "baseline_collection": True,
                    "baseline_edp": avg_baseline,
                    "edp_raw": edp,
                    "reward_type": "baseline_completed",
                    "measurements": list(self.baseline_measurements),
                    "baseline_std": np.std(self.baseline_measurements)
                }
        
        # 计算EDP得分 - 使用基线
        if self.baseline_edp and self.baseline_edp > 0:
            # 使用基线计算奖励
            edp_ratio = edp / self.baseline_edp  # EDP比率：<1为改进，>1为变差
            # 将比率转换为-2到+2的奖励范围
            # ratio=0.5 -> reward=+2 (50%改进)
            # ratio=1.0 -> reward=0 (无变化) 
            # ratio=2.0 -> reward=-2 (性能变差一倍)
            final_reward = 2.0 * (1.0 - min(2.0, edp_ratio))  # 范围[-2, +2]
            
            # 计算百分比改进
            edp_improvement_pct = (self.baseline_edp - edp) / self.baseline_edp * 100
            
            logger.debug(
                f"🔄 EDP得分 (基线): "
                f"EDP={edp:.6f}, 基线={self.baseline_edp:.6f}, 改进={edp_improvement_pct:+.1f}%, 奖励={final_reward:+.3f}"
            )
        else:
            # 没有基线，等待基线收集
            if not self.baseline_collected:
                logger.debug(f"⏳ 等待基线收集: EDP={edp:.6f} J·s")
                final_reward = 0.0  # 中性奖励
            else:
                # 基线异常，使用默认值
                logger.warning(f"⚠️ 基线异常: baseline_edp={self.baseline_edp}")
                final_reward = 0.0  # 中性奖励

        # 构建详细信息
        info = {
            'avg_tpot': avg_tpot,
            'energy_j': energy_j,
            'edp_raw': edp,
            'edp_baseline': self.baseline_edp if self.baseline_edp else 0,
            'edp_improvement_pct': edp_improvement_pct if 'edp_improvement_pct' in locals() else 0,
            'final_reward': final_reward,
            'reward_type': 'baseline' if self.baseline_collected else 'waiting_baseline'
        }

        # 日志输出
        if self.baseline_edp:
            logger.info(
                f"💰 EDP奖励计算:\n"
                f"   TPOT: {avg_tpot:.3f}s, 能耗: {energy_j:.4f}J\n"
                f"   EDP: {edp:.6f}, 基线: {self.baseline_edp:.6f}\n"
                f"   改进: {info['edp_improvement_pct']:+.1f}%, 奖励: {final_reward:+.3f}"
            )
        else:
            logger.info(
                f"💰 EDP奖励计算:\n"
                f"   TPOT: {avg_tpot:.3f}s, 能耗: {energy_j:.4f}J\n"
                f"   EDP: {edp:.6f}, 奖励: {final_reward:+.3f} (等待基线)"
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
            energy_consumed_j=energy_delta
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

        
    
    
        
    
    
