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