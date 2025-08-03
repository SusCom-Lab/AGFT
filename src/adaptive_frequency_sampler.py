import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict, deque
import logging
from dataclasses import dataclass, field
from enum import Enum
from .logger import setup_logger

logger = setup_logger(__name__)


class SamplingMode(Enum):
    """采样模式枚举"""
    SLO_AWARE = "slo_aware"      # SLO约束模式  
    EDP_OPTIMAL = "edp_optimal"  # EDP优化模式


@dataclass
class FrequencyZone:
    """频率区域定义"""
    min_freq: int
    max_freq: int
    step_size: int  # 必须是15的倍数
    zone_type: str  # 'safe', 'violation', 'high_reward', 'low_reward'
    
    def generate_frequencies(self) -> List[int]:
        """生成该区域的频率点"""
        frequencies = []
        freq = self.min_freq
        while freq <= self.max_freq:
            frequencies.append(freq)
            freq += self.step_size
        return frequencies


@dataclass 
class AdaptiveSamplingConfig:
    """自适应采样配置"""
    min_freq: int = 210  # 默认值，应该由GPU检测覆盖
    max_freq: int = 2100  # 默认值，应该由GPU检测覆盖
    
    # SLO模式配置
    slo_coarse_step: int = 90     # SLO模式粗搜步长
    slo_fine_step: int = 15       # SLO模式细搜步长，固定使用
    
    # EDP模式配置
    edp_initial_step: int = 90    # EDP模式初始步长
    edp_fine_step: int = 15       # EDP模式细搜步长，固定使用
    reward_threshold: float = 0.5 # 高奖励区域阈值（平衡识别度和精确性）
    
    # 最优频率搜索范围配置
    optimal_search_range: int = 150  # 最优频率周围的搜索范围（MHz）
    
    # 学习器成熟度配置
    learner_maturity_threshold: int = 100  # 学习器成熟度门槛（决策轮次）
    
    # 细化控制配置
    refinement_start_threshold: int = 50  # 开始细化的最小轮次阈值


class AdaptiveFrequencySampler:
    """
    自适应频率采样器
    
    核心功能：
    1. 双模式频率采样（SLO约束 vs EDP优化）
    2. 动态频率空间调整
    3. 奖励驱动的区域细化
    4. SLO感知的边界检测
    """
    
    def __init__(self, config: AdaptiveSamplingConfig):
        self.config = config
        self.current_mode = SamplingMode.SLO_AWARE
        self.current_frequencies = []
        
        # SLO状态跟踪
        self.slo_violation_boundary = None  # SLO违反边界频率
        self.slo_violation_history = deque(maxlen=20)
        
        # 奖励分析
        self.frequency_rewards = defaultdict(list)  # 频率->奖励历史
        
        # 频率区域管理
        self.frequency_zones = []
        self.refinement_count = 0
        
        logger.info(f"🚀 初始化自适应频率采样器 - 搜索范围: ±{config.optimal_search_range}MHz, 步长: {config.edp_fine_step}MHz")
    
    def set_mode(self, mode: SamplingMode, force_regenerate: bool = True):
        """设置采样模式"""
        if self.current_mode != mode or force_regenerate:
            old_mode = self.current_mode
            self.current_mode = mode
            logger.info(f"🔄 采样模式切换: {old_mode.value} -> {mode.value}")
            
            if force_regenerate:
                self._regenerate_frequency_space()
    
    def get_initial_frequencies(self) -> List[int]:
        """获取初始频率列表（第一阶段粗搜）"""
        if self.current_mode == SamplingMode.SLO_AWARE:
            return self._generate_slo_initial_frequencies()
        else:
            return self._generate_edp_initial_frequencies()
    
    def update_slo_boundary(self, violation_action, gpu_controller=None) -> bool:
        """
        更新SLO违反边界 - 支持组合频率的智能边界设置
        
        Args:
            violation_action: 发现违反SLO的动作 (可能是频率或(核心频率, 显存频率)元组)
            gpu_controller: GPU控制器实例，用于检查组合频率模式
            
        Returns:
            bool: 是否需要重新生成频率空间
        """
        # 检查是否是组合频率模式
        is_memory_combo_mode = (gpu_controller and 
                               hasattr(gpu_controller, 'enable_memory_frequency_control') and
                               gpu_controller.enable_memory_frequency_control and
                               hasattr(gpu_controller, 'memory_frequency_supported') and
                               gpu_controller.memory_frequency_supported)
        
        if is_memory_combo_mode and isinstance(violation_action, tuple):
            # 组合频率模式：violation_action = (core_freq, memory_freq)
            core_freq, memory_freq = violation_action
            logger.warning(f"⚠️ 组合频率SLO违规: {core_freq}MHz核心+{memory_freq}MHz显存")
            
            # 对于组合频率，我们不设置核心频率边界，而是设置显存频率边界
            # 这允许相同的核心频率搭配更高的显存频率
            old_memory_boundary = getattr(self, 'slo_memory_violation_boundary', None)
            self.slo_memory_violation_boundary = memory_freq
            
            logger.info(f"⚠️ 更新显存频率SLO边界: {old_memory_boundary}MHz -> {memory_freq}MHz")
            logger.info(f"📌 保留核心频率{core_freq}MHz，过滤显存频率≤{memory_freq}MHz")
            
            # 对于组合频率违规，我们不更新核心频率边界
            # 只在GPU控制器中处理显存频率过滤
            if hasattr(gpu_controller, 'update_memory_slo_boundary'):
                return gpu_controller.update_memory_slo_boundary(memory_freq, core_freq)
            else:
                logger.warning("⚠️ GPU控制器不支持显存频率SLO边界更新")
                return False
        else:
            # 传统单一核心频率模式
            violation_freq = violation_action if isinstance(violation_action, int) else violation_action[0]
            
            if self.slo_violation_boundary != violation_freq:
                old_boundary = self.slo_violation_boundary
                self.slo_violation_boundary = violation_freq
                self.slo_violation_history.append(violation_freq)
                
                logger.info(f"⚠️ 更新核心频率SLO边界: {old_boundary}MHz -> {violation_freq}MHz")
                
                if self.current_mode == SamplingMode.SLO_AWARE:
                    # 在初期遍历阶段，仅进行级联修剪违规频率，不跳转到最高频率
                    cascade_pruned = self._cascade_prune_violated_frequencies()
                    if cascade_pruned:
                        logger.info(f"✂️ 初期遍历级联修剪完成，继续探索剩余安全频率")
                        return True
                    else:
                        # 如果级联修剪失败，回退到常规重新生成
                        self._regenerate_slo_frequencies()
                        return True
            return False
    
    def _cascade_prune_violated_frequencies(self) -> bool:
        """
        初期遍历阶段的级联修剪：简单移除违规频率及以下频率
        
        核心思想：
        - 保持初期遍历的简洁性，不跳转到高频范围
        - 仅级联移除SLO违规边界及以下的频率
        - 继续当前的高频到低频探索策略
        
        Returns:
            bool: 是否成功进行了级联修剪
        """
        if not self.slo_violation_boundary:
            logger.warning("⚠️ 未设置SLO违规边界，无法进行级联修剪")
            return False
        
        # 获取当前频率列表
        original_freqs = self.current_frequencies.copy()
        if not original_freqs:
            logger.warning("⚠️ 当前频率列表为空")
            return False
        
        # 级联修剪：移除违规边界及以下的所有频率
        safe_freqs = [freq for freq in original_freqs if freq > self.slo_violation_boundary]
        
        if not safe_freqs:
            logger.warning(f"⚠️ 级联修剪后无安全频率（边界: >{self.slo_violation_boundary}MHz）")
            return False
        
        # 统计修剪结果
        pruned_freqs = [freq for freq in original_freqs if freq <= self.slo_violation_boundary]
        
        # 更新频率列表
        self.current_frequencies = safe_freqs
        
        logger.info(f"✂️ 初期遍历级联修剪:")
        logger.info(f"   📊 修剪统计: {len(original_freqs)} -> {len(safe_freqs)}个频率")
        logger.info(f"   ⛔ 违规边界: ≤{self.slo_violation_boundary}MHz")
        logger.info(f"   🗑️ 移除频率: {pruned_freqs}")
        logger.info(f"   ✅ 保留频率: {safe_freqs}")
        logger.info(f"   🎯 策略优势: 保持初期遍历简洁性，避免跳转高频重新开始")
        
        return True
    
    def _emergency_slo_refinement(self, gpu_controller=None) -> bool:
        """
        紧急SLO细化：一旦发现SLO违规，立即细化到安全频率范围
        以当前可用最大频率为上界，-300MHz为下界，60MHz步长
        
        Returns:
            bool: 是否成功进行了紧急细化
        """
        try:
            # 使用统一方法获取真正可用的频率（排除修剪的、失败的频率）
            # 注意：强制禁用SLO过滤，因为我们要在所有理论可用频率中找最大值
            available_freqs = self.get_available_frequencies_unified(
                linucb_model=None,  # 在紧急情况下暂不需要
                gpu_controller=gpu_controller,  # 传递GPU控制器以获取失败频率
                force_slo_filter=False  # 强制禁用SLO过滤
            )
            
            if not available_freqs:
                logger.warning("⚠️ 没有可用频率进行紧急细化")
                return False
            
            # 通过外部接口获取实际可设置的频率上界
            actual_max_freq = None
            if hasattr(self, '_gpu_controller') and self._gpu_controller:
                try:
                    # 获取GPU控制器的实际最大频率
                    actual_max_freq = getattr(self._gpu_controller, 'max_freq', None)
                except:
                    pass
            
            # 确定紧急细化范围：使用实际可设置的最大频率
            if actual_max_freq:
                max_settable_freq = min(max(available_freqs), actual_max_freq)
                logger.debug(f"🔧 使用实际可设置最大频率: {max_settable_freq}MHz (理论:{max(available_freqs)}MHz, 硬件:{actual_max_freq}MHz)")
            else:
                max_settable_freq = max(available_freqs)
                logger.debug(f"🔧 使用理论最大频率: {max_settable_freq}MHz")
            
            emergency_lower_bound = max_settable_freq - 450  # 实际上界-450MHz为下界
            emergency_upper_bound = max_settable_freq
            
            # 确保下界不低于配置的最小频率和SLO边界
            emergency_lower_bound = max(
                emergency_lower_bound,
                self.config.min_freq,
                self.slo_violation_boundary if self.slo_violation_boundary else 0
            )
            
            # 检查范围有效性
            if emergency_lower_bound >= emergency_upper_bound:
                logger.warning(f"⚠️ 紧急细化范围无效: [{emergency_lower_bound}-{emergency_upper_bound}]MHz")
                return False
            
            # 生成紧急细化频率列表（90MHz步长）
            emergency_frequencies = list(range(emergency_lower_bound, emergency_upper_bound + 1, 90))
            
            # 过滤掉违规频率
            emergency_frequencies = [freq for freq in emergency_frequencies 
                                   if freq > self.slo_violation_boundary]
            
            if not emergency_frequencies:
                logger.warning(f"⚠️ 紧急细化后没有安全频率")
                return False
            
            # 更新频率列表
            old_count = len(self.current_frequencies)
            self.current_frequencies = emergency_frequencies
            
            logger.info(f"🚨 紧急SLO细化: {old_count} -> {len(emergency_frequencies)}个频率")
            logger.info(f"🔒 紧急安全范围: [{emergency_lower_bound}-{emergency_upper_bound}]MHz")
            logger.info(f"⛔ SLO违规边界: >{self.slo_violation_boundary}MHz")
            logger.info(f"⚙️ 紧急步长: 15MHz")
            logger.info(f"📋 紧急频率: {emergency_frequencies[:5]}{'...' if len(emergency_frequencies) > 5 else ''}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 紧急SLO细化失败: {e}")
            return False
    
    def update_reward_feedback(self, frequency: int, reward: float):
        """更新频率奖励反馈"""
        self.frequency_rewards[frequency].append(reward)
        
        # 限制历史长度
        if len(self.frequency_rewards[frequency]) > 10:
            self.frequency_rewards[frequency] = self.frequency_rewards[frequency][-10:]
    
    def refine_frequency_space(self, min_refinement_interval: int = 20, 
                             linucb_model=None, current_context=None, gpu_controller=None) -> bool:
        """
        基于奖励反馈细化频率空间
        
        Args:
            min_refinement_interval: 最小细化间隔
            linucb_model: LinUCB模型实例（用于智能预测）
            current_context: 当前上下文特征
            gpu_controller: GPU控制器实例（用于获取失败频率）
            
        Returns:
            bool: 是否进行了细化
        """
        self.refinement_count += 1
        logger.debug(f"🔍 频率空间细化检查 - 轮次: {self.refinement_count}, 间隔要求: {min_refinement_interval}, 开始阈值: {self.config.refinement_start_threshold}")
        
        # 1. 检查是否达到开始细化的最小轮次阈值
        if self.refinement_count < self.config.refinement_start_threshold:
            logger.debug(f"📊 未达到细化开始阈值，跳过 (轮次 {self.refinement_count} < {self.config.refinement_start_threshold})")
            return False
        
        # 2. 计算从开始阈值之后的有效轮次
        effective_rounds = self.refinement_count - self.config.refinement_start_threshold
        
        # 3. 检查是否到达细化间隔
        # 第一次细化：正好在开始阈值轮次 (effective_rounds == 0)
        # 后续细化：每隔 min_refinement_interval 轮次
        if effective_rounds > 0 and effective_rounds % min_refinement_interval != 0:
            logger.info(f"📊 未达到细化间隔，跳过 (有效轮次 {effective_rounds} 不是 {min_refinement_interval} 的倍数)")
            return False
        
        logger.info(f"🎯 触发频率空间细化 - 模式: {self.current_mode.value}, 轮次: {self.refinement_count}")
        
        if self.current_mode == SamplingMode.EDP_OPTIMAL:
            logger.info(f"📊 EDP模式细化调试信息:")
            logger.info(f"   LinUCB模型: {linucb_model is not None}")
            if linucb_model:
                logger.info(f"   模型轮次: {getattr(linucb_model, 'total_rounds', 'N/A')}")
                logger.info(f"   是否有EDP历史: {hasattr(linucb_model, 'edp_history') and bool(getattr(linucb_model, 'edp_history', []))}")
            logger.info(f"   频率奖励历史: {len(self.frequency_rewards)} 个频率")
            result = self._refine_edp_frequencies(linucb_model, current_context, gpu_controller)
            logger.info(f"📈 EDP模式细化结果: {'成功' if result else '失败/跳过'}")
            return result
        else:
            logger.info(f"📊 SLO模式细化调试信息:")
            logger.info(f"   SLO边界: {self.slo_violation_boundary}")
            result = self._refine_slo_frequencies(linucb_model, current_context, gpu_controller)
            logger.info(f"📈 SLO模式细化结果: {'成功' if result else '失败/跳过'}")
            return result
    
    def get_current_frequencies(self) -> List[int]:
        """获取当前频率列表"""
        if not self.current_frequencies:
            self.current_frequencies = self.get_initial_frequencies()
        return self.current_frequencies.copy()
    
    def get_filtered_frequencies(self, linucb_model=None) -> List[int]:
        """获取经过修剪过滤的当前频率列表"""
        current_freqs = self.get_current_frequencies()
        
        if linucb_model and hasattr(linucb_model, 'pruned_frequencies'):
            pruned_frequencies = linucb_model.pruned_frequencies
            if pruned_frequencies:
                original_count = len(current_freqs)
                current_freqs = [freq for freq in current_freqs if freq not in pruned_frequencies]
                filtered_count = len(current_freqs)
                if original_count != filtered_count:
                    logger.debug(f"📝 应用频率修剪过滤: {original_count} -> {filtered_count}个可用频率")
        
        return current_freqs
    
    def get_available_frequencies_unified(self, linucb_model=None, gpu_controller=None, 
                                         force_slo_filter: bool = None) -> List[int]:
        """
        统一的可用频率获取方法 - 替代分散的过滤逻辑
        
        Args:
            linucb_model: LinUCB模型（用于获取修剪频率）
            gpu_controller: GPU控制器（用于获取失败频率）
            force_slo_filter: 强制启用/禁用SLO过滤，None则根据当前模式自动判断
            
        Returns:
            经过所有必要过滤的可用频率列表
        """
        # 1. 获取基础频率列表
        base_frequencies = self.get_current_frequencies()
        if not base_frequencies:
            logger.warning("⚠️ 基础频率列表为空")
            return []
        
        # 2. 确定是否需要SLO过滤
        if force_slo_filter is None:
            apply_slo_filter = (self.current_mode == SamplingMode.SLO_AWARE and 
                              self.slo_violation_boundary is not None)
        else:
            apply_slo_filter = force_slo_filter and self.slo_violation_boundary is not None
        
        # 3. 应用统一过滤
        return self._filter_valid_frequencies(
            base_frequencies, linucb_model, gpu_controller, apply_slo_filter
        )
    
    def _filter_valid_frequencies(self, frequencies: List[int], linucb_model=None, gpu_controller=None, 
                                 apply_slo_filter: bool = False) -> List[int]:
        """
        核心过滤方法：排除已修剪、设置失败和SLO违规的频率
        
        Args:
            frequencies: 待过滤的频率列表
            linucb_model: LinUCB模型（用于获取修剪频率）
            gpu_controller: GPU控制器（用于获取失败频率）
            apply_slo_filter: 是否应用SLO边界过滤
        """
        if not frequencies:
            return []
        
        valid_freqs = frequencies.copy()
        original_count = len(valid_freqs)
        filter_stats = {"original": original_count, "pruned": 0, "failed": 0, "slo": 0}
        
        # 过滤器1: 排除已修剪的频率
        if linucb_model and hasattr(linucb_model, 'pruned_frequencies'):
            pruned_frequencies = linucb_model.pruned_frequencies
            if pruned_frequencies:
                before_count = len(valid_freqs)
                valid_freqs = [freq for freq in valid_freqs if freq not in pruned_frequencies]
                filter_stats["pruned"] = before_count - len(valid_freqs)
        
        # 过滤器2: 排除设置失败的频率
        if gpu_controller and hasattr(gpu_controller, 'failed_frequencies'):
            failed_frequencies = gpu_controller.failed_frequencies
            if failed_frequencies:
                before_count = len(valid_freqs)
                valid_freqs = [freq for freq in valid_freqs if freq not in failed_frequencies]
                filter_stats["failed"] = before_count - len(valid_freqs)
                # 添加调试信息
                if filter_stats["failed"] > 0:
                    removed_failed = [f for f in frequencies if f in failed_frequencies]
                    logger.debug(f"🚫 过滤失败频率: {removed_failed} (GPU控制器报告: {sorted(list(failed_frequencies))})")
        
        # 过滤器3: 排除SLO违规频率
        if apply_slo_filter and self.slo_violation_boundary is not None:
            before_count = len(valid_freqs)
            valid_freqs = [freq for freq in valid_freqs if freq > self.slo_violation_boundary]
            filter_stats["slo"] = before_count - len(valid_freqs)
        
        # 统一日志输出
        total_filtered = filter_stats["pruned"] + filter_stats["failed"] + filter_stats["slo"]
        if total_filtered > 0:
            filter_details = []
            if filter_stats["pruned"] > 0:
                filter_details.append(f"修剪{filter_stats['pruned']}个")
            if filter_stats["failed"] > 0:
                filter_details.append(f"失败{filter_stats['failed']}个")
            if filter_stats["slo"] > 0:
                filter_details.append(f"SLO违规{filter_stats['slo']}个")
            
            logger.debug(f"📊 频率过滤: {original_count} -> {len(valid_freqs)}个 "
                        f"({', '.join(filter_details)})")
        
        # 安全检查：如果所有频率都被过滤，返回空列表（系统将重置GPU频率）
        if len(valid_freqs) == 0:
            if len(frequencies) > 0:
                logger.info(f"📊 所有频率都被过滤，系统将重置GPU频率到默认状态")
            return []
        
        return valid_freqs
    
    def _generate_slo_initial_frequencies(self) -> List[int]:
        """生成SLO模式的初始频率（高频往低频粗搜）"""
        frequencies = []
        freq = self.config.max_freq
        
        while freq >= self.config.min_freq:
            frequencies.append(freq)
            freq -= self.config.slo_coarse_step
        
        # 确保包含最小频率
        if self.config.min_freq not in frequencies:
            frequencies.append(self.config.min_freq)
        
        frequencies = sorted(frequencies)
        logger.info(f"🔍 SLO初始粗搜: {len(frequencies)}个频点, 搜索顺序: {list(reversed(frequencies))}")
        return frequencies
    
    def _generate_edp_initial_frequencies(self) -> List[int]:
        """生成EDP模式的初始频率（从小到大遍历）"""
        frequencies = list(range(
            self.config.min_freq, 
            self.config.max_freq + 1, 
            self.config.edp_initial_step
        ))
        
        # 确保包含最大频率
        if self.config.max_freq not in frequencies:
            frequencies.append(self.config.max_freq)
        
        frequencies = sorted(frequencies)
        logger.info(f"🔍 EDP初始粗搜: {len(frequencies)}个频点, 搜索顺序: 从小到大 {frequencies}")
        return frequencies
    
    def _regenerate_slo_frequencies(self):
        """基于SLO边界修剪频率空间 - 使用通用过滤方法"""
        if self.slo_violation_boundary is None:
            return
        
        # 使用通用过滤方法，启用SLO过滤
        original_count = len(self.current_frequencies)
        safe_freqs = self._filter_valid_frequencies(
            self.current_frequencies, 
            apply_slo_filter=True
        )
        
        if not safe_freqs:
            logger.warning(f"⚠️ SLO边界过高({self.slo_violation_boundary}MHz)，所有频率都被过滤")
            return
        
        self.current_frequencies = safe_freqs
        filtered_count = original_count - len(safe_freqs)
        
        logger.info(f"🛡️ SLO频率修剪: 边界>{self.slo_violation_boundary}MHz, "
                   f"过滤{filtered_count}个违规频率, "
                   f"保留{len(safe_freqs)}个安全频率")
    
    def _refine_edp_frequencies(self, linucb_model=None, current_context=None, gpu_controller=None) -> bool:
        """
        基于学习器成熟度的混合细化策略
        
        Args:
            linucb_model: LinUCB模型实例
            current_context: 当前上下文特征向量
            gpu_controller: GPU控制器实例
        """
        # 从配置中获取学习器成熟度门槛
        MATURITY_THRESHOLD = self.config.learner_maturity_threshold
        
        # 检查模型是否成熟
        if linucb_model is None or linucb_model.total_rounds < MATURITY_THRESHOLD:
            # --- 策略一：模型不成熟时，使用基于历史观测的统计细化 ---
            logger.debug(f"🧠 模型不成熟 (轮次 {linucb_model.total_rounds if linucb_model else 0} < {MATURITY_THRESHOLD})，使用基于历史观测的统计细化...")
            return self._refine_based_on_observed_median_rewards(linucb_model, gpu_controller)
        else:
            # --- 策略二：模型成熟后，完全信任并咨询模型预测 ---
            logger.debug(f"🧠 模型已成熟 (轮次 {linucb_model.total_rounds} >= {MATURITY_THRESHOLD})，使用基于模型预测的智能细化...")
            return self._refine_based_on_ucb_edp_hybrid(linucb_model, current_context, gpu_controller)
    
    def _refine_based_on_observed_median_rewards(self, linucb_model=None, gpu_controller=None) -> bool:
        """
        统一的EDP导向频率细化逻辑：
        1. 识别当前负载情况（基于最近EDP）
        2. 找到相同负载下历史EDP最优频率
        3. 围绕最优频率±范围细化，固定15MHz步长
        """
        # 检查是否有足够的历史数据进行EDP分析
        if not linucb_model or not hasattr(linucb_model, 'edp_history') or not linucb_model.edp_history:
            logger.info("📊 无EDP历史数据，跳过负载感知细化")
            logger.info(f"   LinUCB模型存在: {linucb_model is not None}")
            if linucb_model:
                logger.info(f"   有edp_history属性: {hasattr(linucb_model, 'edp_history')}")
                logger.info(f"   EDP历史长度: {len(getattr(linucb_model, 'edp_history', []))}")
            return False
        
        # 获取所有探索过的频率，排除已修剪和设置失败的频率
        all_explored_frequencies = list(self.frequency_rewards.keys())
        explored_frequencies = self._filter_valid_frequencies(all_explored_frequencies, linucb_model, gpu_controller)
        
        logger.info(f"📊 频率探索状况:")
        logger.info(f"   所有探索频率: {len(all_explored_frequencies)} 个")
        logger.info(f"   有效探索频率: {len(explored_frequencies)} 个")
        
        if not explored_frequencies:
            logger.info("📊 没有可用的探索频率，跳过细化")
            return False
        
        # Step 1: 使用负载感知方法找到当前负载下的最优频率
        best_edp_freq = self._find_load_normalized_best_freq(linucb_model, explored_frequencies)
        logger.info(f"📊 负载感知最优频率: {best_edp_freq}")
        
        if best_edp_freq is None:
            # 如果负载感知失败，回退到整体历史EDP最佳频率
            best_edp_freq = self._find_overall_best_edp_freq(linucb_model, explored_frequencies)
            logger.info(f"📊 回退到整体最优频率: {best_edp_freq}")
            
        if best_edp_freq is None:
            logger.info("📊 无法确定最优频率，跳过细化")
            logger.info(f"   探索频率列表: {explored_frequencies[:10]}...")  # 只显示前10个
            return False
        
        # Step 2: 围绕最优频率进行范围细化
        search_range = self.config.optimal_search_range
        freq_min = max(self.config.min_freq, best_edp_freq - search_range)
        freq_max = min(self.config.max_freq, best_edp_freq + search_range)
        
        # Step 3: 生成频率列表，固定15MHz步长
        new_frequencies = list(range(freq_min, freq_max + 1, self.config.edp_fine_step))
        
        # 确保最佳频率在列表中
        if best_edp_freq not in new_frequencies:
            new_frequencies.append(best_edp_freq)
            new_frequencies = sorted(new_frequencies)
        
        # Step 4: 排除已修剪的频率
        new_frequencies = self._filter_valid_frequencies(new_frequencies, linucb_model, gpu_controller)
        
        # 更新频率列表
        old_count = len(self.current_frequencies)
        self.current_frequencies = new_frequencies
        # 不重置计数器，让它持续累积以支持间隔检查
        
        logger.info(f"🎯 EDP导向细化: {old_count} -> {len(new_frequencies)}个频率")
        logger.info(f"📍 最优频率: {best_edp_freq}MHz")
        logger.info(f"📏 搜索范围: ±{search_range}MHz = [{freq_min}-{freq_max}]MHz")
        logger.info(f"⚙️  固定步长: {self.config.edp_fine_step}MHz")
        
        return True
    
    def _refine_slo_frequencies(self, linucb_model=None, current_context=None, gpu_controller=None) -> bool:
        """
        SLO模式下的混合细化策略：基于学习器成熟度选择细化方法
        
        Args:
            linucb_model: LinUCB模型实例
            current_context: 当前上下文特征向量
        """
        # 如果没有SLO边界，使用EDP导向的全域细化策略
        if not self.slo_violation_boundary:
            logger.info("📊 SLO模式但无违反边界，使用EDP导向全域细化策略")
            return self._refine_edp_frequencies(linucb_model, current_context, gpu_controller)
        
        # 从配置中获取学习器成熟度门槛（与EDP模式保持一致）
        MATURITY_THRESHOLD = self.config.learner_maturity_threshold
        
        # 检查模型是否成熟
        if linucb_model is None or linucb_model.total_rounds < MATURITY_THRESHOLD:
            # --- 策略一：模型不成熟时，使用基于历史观测的安全区统计细化 ---
            logger.debug(f"🛡️ SLO模式-模型不成熟 (轮次 {linucb_model.total_rounds if linucb_model else 0} < {MATURITY_THRESHOLD})，使用基于历史观测的安全区统计细化...")
            return self._refine_slo_based_on_observed_median(linucb_model, gpu_controller)
        else:
            # --- 策略二：模型成熟后，使用基于模型预测的安全区智能细化 ---
            logger.debug(f"🛡️ SLO模式-模型已成熟 (轮次 {linucb_model.total_rounds} >= {MATURITY_THRESHOLD})，使用基于模型预测的安全区智能细化...")
            return self._refine_based_on_ucb_edp_hybrid(linucb_model, current_context, gpu_controller)
    
    def _refine_slo_based_on_observed_median(self, linucb_model=None, gpu_controller=None) -> bool:
        """
        SLO模式下的EDP导向安全区细化（简化版本）
        围绕安全区内最优频率进行细化，固定15MHz步长
        """

        
        # 检查是否有足够的历史数据进行EDP分析
        if not linucb_model or not hasattr(linucb_model, 'edp_history') or not linucb_model.edp_history:
            logger.debug("📊 无EDP历史数据，跳过SLO负载感知细化")
            return False
        
        # 定义安全区：频率必须大于等于SLO违规边界
        safe_zone_min = self.slo_violation_boundary
        safe_zone_max = self.config.max_freq
        
        # 获取安全区内的探索频率
        all_explored_frequencies = list(self.frequency_rewards.keys())
        safe_explored_freqs = self._filter_valid_frequencies(all_explored_frequencies, linucb_model, gpu_controller, apply_slo_filter=True)
        
        if not safe_explored_freqs:
            logger.debug("📊 安全区内没有可用的探索频率，跳过细化")
            return False
        
        # 找到安全区内的最优频率
        best_edp_freq = self._find_load_normalized_best_freq(linucb_model, safe_explored_freqs)
        
        if best_edp_freq is None:
            # 回退到安全区内整体历史EDP最佳频率
            best_edp_freq = self._find_overall_best_edp_freq(linucb_model, safe_explored_freqs)
            
        if best_edp_freq is None:
            logger.debug("📊 无法确定安全区内最优频率，跳过细化")
            return False
        
        # 围绕安全区内最优频率进行范围细化
        search_range = self.config.optimal_search_range
        freq_min = max(safe_zone_min, best_edp_freq - search_range)
        freq_max = min(safe_zone_max, best_edp_freq + search_range)
        
        # 生成频率列表，固定15MHz步长
        new_frequencies = list(range(freq_min, freq_max + 1, self.config.slo_fine_step))
        
        # 确保最佳频率在列表中
        if best_edp_freq not in new_frequencies:
            new_frequencies.append(best_edp_freq)
            new_frequencies = sorted(new_frequencies)
        
        # 确保所有频率都在安全区内
        new_frequencies = self._filter_valid_frequencies(new_frequencies, apply_slo_filter=True)
        
        if not new_frequencies:
            logger.warning("⚠️ 细化后没有安全频率，保持原有配置")
            return False
        
        # 更新频率列表
        old_count = len(self.current_frequencies)
        self.current_frequencies = new_frequencies
        # 不重置计数器，让它持续累积以支持间隔检查
        
        logger.info(f"🛡️ SLO-EDP导向细化: {old_count} -> {len(new_frequencies)}个频率")
        logger.info(f"🔒 安全边界: ≥{safe_zone_min}MHz")
        logger.info(f"📍 最优频率: {best_edp_freq}MHz")
        logger.info(f"📏 搜索范围: ±{search_range}MHz = [{freq_min}-{freq_max}]MHz")
        logger.info(f"⚙️  固定步长: {self.config.slo_fine_step}MHz")
        
        return True
    
    def _refine_based_on_ucb_edp_hybrid(self, linucb_model, current_context, gpu_controller=None) -> bool:
        """
        基于UCB+EDP混合策略的智能细化（适用于SLO和EDP模式）
        1. 获取UCB值（预测奖励+置信区间）
        2. 筛选高潜力候选池（UCB值 > 最大UCB * 0.85）
        3. 用历史EDP做最终裁决，选择EDP最佳的候选作为锚点
        4. 围绕可靠锚点±150MHz细化
        
        Args:
            linucb_model: 成熟的LinUCB模型实例
            current_context: 当前上下文特征向量
            gpu_controller: GPU控制器实例
        """
        # 如果没有提供模型或上下文，跳过细化
        if linucb_model is None or current_context is None:
            logger.warning("⚠️ 缺少LinUCB模型或上下文，跳过细化")
            return False
        
        # 获取所有探索过的频率，根据当前模式应用相应的过滤策略
        all_explored_frequencies = list(self.frequency_rewards.keys())
        # SLO模式: 应用SLO过滤，EDP模式: 不应用SLO过滤
        apply_slo_filter = (self.current_mode == SamplingMode.SLO_AWARE)
        valid_frequencies = self._filter_valid_frequencies(all_explored_frequencies, linucb_model, gpu_controller, apply_slo_filter=apply_slo_filter)
        explored_freqs = valid_frequencies
        
        if not explored_freqs:
            mode_desc = "安全区内" if apply_slo_filter else "有效区内"
            logger.debug(f"📊 {mode_desc}没有探索过的频率，跳过细化")
            return False
        
        # Step 1: 获取所有有效频率的UCB值（潜力与不确定性）
        ucb_values = {}
        mode_desc = "安全" if apply_slo_filter else "有效"
        logger.debug(f"🎯 使用LinUCB获取 {len(explored_freqs)} 个{mode_desc}频率的UCB值...")
        
        # 优化：一次性获取所有UCB值，而不是逐个频率计算
        if hasattr(linucb_model, 'get_ucb_values'):
            # 使用批量UCB计算方法（更高效）
            all_ucb_values = linucb_model.get_ucb_values(current_context)
            # 只保留我们需要的频率
            ucb_values = {freq: all_ucb_values.get(freq, 0.0) for freq in explored_freqs if freq in all_ucb_values}
       
        
        # 为批量获取的结果添加调试日志
        if hasattr(linucb_model, 'get_ucb_values') and ucb_values:
            for freq, ucb_value in ucb_values.items():
                logger.debug(f"  {mode_desc}频率 {freq}MHz: UCB = {ucb_value:.4f}")
        
        if not ucb_values:
            logger.warning("⚠️ 无法获取任何UCB值，跳过细化")
            return False
        
        # Step 2: 筛选高潜力候选池（相对选择，支持负值）
        max_ucb = max(ucb_values.values())
        
        # 如果最大UCB是正数，使用比例阈值；如果是负数，使用绝对差阈值
        if max_ucb > 0:
            ucb_threshold = max_ucb * 0.8  # 正数情况：保持原逻辑
        else:
            # 负数情况：选择与最大值差距不超过0.3的候选
            ucb_threshold = max_ucb - 0.3
        
        high_potential_candidates = {freq: ucb for freq, ucb in ucb_values.items() if ucb >= ucb_threshold}
        
        # 确保至少有一个候选（最大UCB值的频率）
        if not high_potential_candidates:
            best_freq = max(ucb_values.keys(), key=lambda f: ucb_values[f])
            high_potential_candidates = {best_freq: ucb_values[best_freq]}
            logger.info(f"⚠️ 无高潜力候选，强制选择最佳UCB频率: {best_freq}MHz (UCB={ucb_values[best_freq]:.4f})")
        
        logger.info(f"🔍 UCB筛选: 最大UCB={max_ucb:.4f}, 阈值={ucb_threshold:.4f}, "
                   f"高潜力候选={len(high_potential_candidates)}个")
        
        # Step 3: 用历史EDP做最终裁决
        best_anchor_freq = self._select_edp_best_anchor(high_potential_candidates, linucb_model)
        
        if best_anchor_freq is None:
            logger.warning("⚠️ 无法确定可靠的EDP锚点，跳过细化")
            return False
        
        # Step 4: 围绕可靠锚点进行细化（±150MHz）
        search_range = self.config.optimal_search_range  # 150MHz
        
        # 确定可用区域边界
        zone_min = self.config.min_freq
        zone_max = self.config.max_freq
        if hasattr(gpu_controller, 'get_available_frequencies_unified'):
            unified_freqs = gpu_controller.get_available_frequencies_unified()
            if unified_freqs:
                zone_min = min(unified_freqs)
                zone_max = max(unified_freqs)
        
        # 围绕锚点±150MHz生成频率列表
        freq_min = max(zone_min, best_anchor_freq - search_range)
        freq_max = min(zone_max, best_anchor_freq + search_range)
        
        # 根据当前模式选择合适的步长
        step_size = self.config.slo_fine_step if apply_slo_filter else self.config.edp_fine_step
        
        # 生成频率列表
        new_frequencies = list(range(freq_min, freq_max + 1, step_size))
        
        # 确保锚点频率在列表中
        if best_anchor_freq not in new_frequencies:
            new_frequencies.append(best_anchor_freq)
            new_frequencies = sorted(new_frequencies)
        
        # 应用相应的过滤策略
        new_frequencies = self._filter_valid_frequencies(new_frequencies, apply_slo_filter=apply_slo_filter)
        
        if not new_frequencies:
            filter_desc = "安全" if apply_slo_filter else "有效"
            logger.warning(f"⚠️ 细化后没有{filter_desc}频率，保持原有配置")
            return False
        
        old_count = len(self.current_frequencies)
        self.current_frequencies = new_frequencies
        # 不重置计数器，让它持续累积以支持间隔检查
        
        mode_name = "SLO-aware" if apply_slo_filter else "EDP-optimal" 
        logger.info(f"🎯 UCB+EDP混合策略细化({mode_name}): {old_count} -> {len(new_frequencies)}个频率")
        logger.info(f"🔒 可用边界: [{zone_min}-{zone_max}]MHz")
        logger.info(f"📍 EDP最佳锚点: {best_anchor_freq}MHz")
        logger.info(f"📏 搜索范围: ±{search_range}MHz = [{freq_min}-{freq_max}]MHz")
        logger.info(f"⚙️ 步长: {step_size}MHz")
        
        # 输出高潜力候选前5名用于调试
        top_candidates = sorted(high_potential_candidates.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.debug(f"🎯 高潜力候选TOP5: {top_candidates}")
        
        return True
    
    def _select_edp_best_anchor(self, high_potential_candidates: Dict[int, float], linucb_model=None) -> Optional[int]:
        """
        从高潜力候选池中选择历史EDP最佳的频率作为锚点
        
        Args:
            high_potential_candidates: 高潜力候选频率字典 {频率: UCB值}
            linucb_model: LinUCB模型实例，用于获取EDP历史数据
            
        Returns:
            EDP最佳的锚点频率，如果无法确定则返回None
        """
        if not high_potential_candidates:
            logger.warning("⚠️ 高潜力候选池为空，无法选择EDP锚点")
            return None
        
        # 样本数量门槛：确保有足够的样本才能可靠地计算平均EDP
        MIN_SAMPLES_FOR_EDP = 4
        
        # 收集每个候选频率的历史EDP数据
        candidate_edp_stats = {}
        
        # 如果有LinUCB模型，尝试使用其EDP历史数据
        if linucb_model and hasattr(linucb_model, 'edp_history') and hasattr(linucb_model, 'action_history'):
            edp_history = linucb_model.edp_history
            action_history = linucb_model.action_history
            
            # 按核心频率分组EDP数据（处理组合频率）
            freq_edp_data = {}
            for action, edp in zip(action_history, edp_history):
                # 提取核心频率
                core_freq = action[0] if isinstance(action, tuple) else action
                if core_freq not in freq_edp_data:
                    freq_edp_data[core_freq] = []
                freq_edp_data[core_freq].append(edp)
            
            # 分析每个候选频率的EDP表现
            for freq in high_potential_candidates.keys():
                if freq in freq_edp_data and len(freq_edp_data[freq]) >= MIN_SAMPLES_FOR_EDP:
                    edp_values = freq_edp_data[freq]
                    mean_edp = np.mean(edp_values)
                    count = len(edp_values)
                    
                    candidate_edp_stats[freq] = {
                        'mean_edp': mean_edp,  # 较低的EDP = 更好
                        'count': count,
                        'ucb': high_potential_candidates[freq]
                    }
                    
                    logger.debug(f"  ✅ 候选 {freq}MHz: 平均EDP={mean_edp:.4f}, 样本数={count}, UCB={high_potential_candidates[freq]:.4f}")
                elif freq in freq_edp_data:
                    logger.debug(f"  ⚠️ 候选 {freq}MHz: 样本数不足({len(freq_edp_data[freq])} < {MIN_SAMPLES_FOR_EDP})，跳过EDP比较")
        
        if not candidate_edp_stats:
            logger.warning(f"⚠️ 没有候选频率有足够EDP样本数(>={MIN_SAMPLES_FOR_EDP})，使用UCB最高者")
            return max(high_potential_candidates.keys(), key=lambda f: high_potential_candidates[f])
        
        # 选择历史平均EDP最低的频率（EDP越低越好）
        best_freq = min(candidate_edp_stats.keys(), key=lambda f: candidate_edp_stats[f]['mean_edp'])
        best_stats = candidate_edp_stats[best_freq]
        
        logger.info(f"🎯 EDP锚点选择: {best_freq}MHz (平均EDP={best_stats['mean_edp']:.4f}, "
                   f"样本数={best_stats['count']}, UCB={best_stats['ucb']:.4f})")
        
        # 输出所有候选的EDP排名用于调试
        edp_ranking = sorted(candidate_edp_stats.items(), 
                           key=lambda x: x[1]['mean_edp'])[:3]  # EDP越低越好，所以升序排列
        logger.debug(f"📊 EDP排名TOP3(最佳): {[(f, s['mean_edp']) for f, s in edp_ranking]}")
        
        return best_freq
    
    def _find_optimal_step(self, freq_range: int, target_count: int) -> int:
        """找到最优步长（必须是15的倍数）"""
        if target_count <= 1:
            return freq_range
        
        ideal_step = freq_range // (target_count - 1)
        
        # 找到最接近的15的倍数
        for step in self.config.valid_step_sizes:
            if step >= ideal_step:
                return step
        
        return self.config.valid_step_sizes[-1]  # 返回最大步长
    
    def _regenerate_frequency_space(self):
        """重新生成频率空间"""
        if self.current_mode == SamplingMode.SLO_AWARE:
            if self.slo_violation_boundary:
                self._regenerate_slo_frequencies()
            else:
                self.current_frequencies = self._generate_slo_initial_frequencies()
        else:
            self.current_frequencies = self._generate_edp_initial_frequencies()
    
    def _find_load_normalized_best_freq(self, linucb_model, explored_frequencies, current_freq=None, current_edp=None) -> Optional[int]:
        """
        频率归一化的负载感知推荐 - 基于当前频率+EDP判断负载档位，推荐同档位最优频率
        
        核心逻辑：
        1. 当前频率 + 当前EDP → 判断在当前频率下属于哪个负载档位
        2. 查找所有频率在各自相同负载档位下的表现
        3. 推荐在该负载档位下表现最佳的频率
        
        Args:
            linucb_model: LinUCB模型，包含EDP历史记录
            explored_frequencies: 已探索的频率列表
            current_freq: 当前频率，如果为None则使用最近的频率
            current_edp: 当前EDP，如果为None则使用最近的EDP
            
        Returns:
            针对当前负载档位的推荐最优频率
        """
        if not hasattr(linucb_model, 'edp_history') or not linucb_model.edp_history:
            logger.debug("📊 无EDP历史数据，无法进行负载感知推荐")
            return None
            
        if not hasattr(linucb_model, 'action_history') or len(linucb_model.action_history) != len(linucb_model.edp_history):
            action_len = len(getattr(linucb_model, 'action_history', []))
            edp_len = len(getattr(linucb_model, 'edp_history', []))
            logger.debug(f"📊 动作历史与EDP历史不匹配 (action:{action_len}, edp:{edp_len})")
            
            # 🔧 自动修复历史长度不匹配问题
            if action_len > edp_len:
                missing_count = action_len - edp_len
                logger.debug(f"🔧 补齐edp_history缺失的{missing_count}个值")
                linucb_model.edp_history.extend([0.0] * missing_count)
            elif edp_len > action_len:
                logger.debug(f"🔧 截断edp_history多余的{edp_len - action_len}个值")
                linucb_model.edp_history = linucb_model.edp_history[:action_len]
            
            # 验证修复结果
            if len(linucb_model.action_history) != len(linucb_model.edp_history):
                logger.debug("❌ 自动修复失败，无法进行负载感知分析")
                return None
        
        # 1. 确定当前状态（频率+EDP）
        if current_freq is None:
            last_action = linucb_model.action_history[-1] if linucb_model.action_history else None
            # 提取核心频率
            current_freq = last_action[0] if isinstance(last_action, tuple) else last_action
        if current_edp is None:
            current_edp = linucb_model.edp_history[-1] if linucb_model.edp_history else None
            
        if current_freq is None or current_edp is None:
            logger.debug("📊 无法获取当前频率或EDP，无法进行负载感知分析")
            return None
        
        # 2. 为每个频率建立负载分类标准
        freq_load_data = {}
        min_samples = 6
        
        # 收集每个频率的EDP数据和负载分类
        freq_edp_data = defaultdict(list)
        for action, edp in zip(linucb_model.action_history, linucb_model.edp_history):
            # 提取核心频率
            core_freq = action[0] if isinstance(action, tuple) else action
            if core_freq in explored_frequencies:
                freq_edp_data[core_freq].append(edp)
        
        for freq, edps in freq_edp_data.items():
            if len(edps) < min_samples:
                continue
                
            edps_array = np.array(edps)
            
            # 计算该频率的负载分档阈值
            p33 = np.percentile(edps_array, 33)
            p67 = np.percentile(edps_array, 67)
            
            # 分档该频率的历史EDP数据
            low_load_edps = edps_array[edps_array <= p33]
            medium_load_edps = edps_array[(edps_array > p33) & (edps_array < p67)]
            high_load_edps = edps_array[edps_array >= p67]
            
            # 计算各档位的平均表现
            load_performance = {}
            if len(low_load_edps) >= 2:
                load_performance['low'] = np.mean(low_load_edps)
            if len(medium_load_edps) >= 2:
                load_performance['medium'] = np.mean(medium_load_edps)
            if len(high_load_edps) >= 2:
                load_performance['high'] = np.mean(high_load_edps)
            
            freq_load_data[freq] = {
                'p33': p33,
                'p67': p67,
                'performance': load_performance,
                'sample_count': len(edps)
            }
        
        # 3. 判断当前EDP在当前频率下属于哪个负载档位
        if current_freq not in freq_load_data:
            logger.debug(f"📊 当前频率{current_freq}MHz缺乏历史数据，无法进行负载档位判断")
            return None
        
        current_freq_data = freq_load_data[current_freq]
        current_p33 = current_freq_data['p33']
        current_p67 = current_freq_data['p67']
        
        if current_edp <= current_p33:
            current_load_level = 'low'
        elif current_edp >= current_p67:
            current_load_level = 'high'
        else:
            current_load_level = 'medium'
        
        logger.info(f"🎯 当前状态分析: {current_freq}MHz + EDP={current_edp:.4f} → {current_load_level}负载档位 "
                   f"(该频率阈值[{current_p33:.3f}, {current_p67:.3f}])")
        
        # 4. 在相同负载档位下，找到所有频率中表现最佳的
        same_load_candidates = {}
        for freq, data in freq_load_data.items():
            if current_load_level in data['performance']:
                same_load_candidates[freq] = data['performance'][current_load_level]
        
        if not same_load_candidates:
            logger.debug(f"📊 在{current_load_level}负载档位下没有找到候选频率")
            return None
        
        # 选择在该负载档位下EDP最小（表现最佳）的频率
        best_freq = min(same_load_candidates.keys(), key=lambda f: same_load_candidates[f])
        best_performance = same_load_candidates[best_freq]
        
        logger.info(f"🎯 负载档位推荐: {current_load_level}负载下推荐{best_freq}MHz "
                   f"(该频率在{current_load_level}负载下平均EDP={best_performance:.4f})")
        
        # 显示该负载档位下的所有候选频率
        sorted_candidates = sorted(same_load_candidates.items(), key=lambda x: x[1])
        logger.debug(f"📊 {current_load_level}负载档位下所有候选:")
        for i, (freq, performance) in enumerate(sorted_candidates, 1):
            status = "👑" if freq == best_freq else f"{i}."
            logger.debug(f"   {status} {freq}MHz: {current_load_level}负载EDP={performance:.4f}")
        
        return best_freq

    def _find_overall_best_edp_freq(self, linucb_model, explored_frequencies) -> Optional[int]:
        """
        简单的回退方法：找到历史整体EDP表现最佳的频率
        
        Args:
            linucb_model: LinUCB模型，包含EDP历史记录
            explored_frequencies: 已探索的频率列表
            
        Returns:
            整体历史EDP最佳的频率
        """
        logger.info("🔍 开始整体最优频率查找...")
        
        if not hasattr(linucb_model, 'edp_history') or not linucb_model.edp_history:
            logger.info("❌ LinUCB模型缺少EDP历史数据")
            return None
            
        if not hasattr(linucb_model, 'action_history') or len(linucb_model.action_history) != len(linucb_model.edp_history):
            action_len = len(getattr(linucb_model, 'action_history', []))
            edp_len = len(getattr(linucb_model, 'edp_history', []))
            logger.warning("❌ LinUCB模型的action_history与edp_history长度不匹配")
            logger.warning(f"   action_history长度: {action_len}")
            logger.warning(f"   edp_history长度: {edp_len}")
            
            # 🔧 自动修复历史长度不匹配问题
            if action_len > edp_len:
                # action_history更长，用默认EDP值补齐edp_history
                missing_count = action_len - edp_len
                logger.info(f"🔧 自动修复: 为edp_history补齐{missing_count}个默认值(0.0)")
                linucb_model.edp_history.extend([0.0] * missing_count)
            elif edp_len > action_len:
                # edp_history更长，截断到与action_history相同长度
                logger.info(f"🔧 自动修复: 截断edp_history的{edp_len - action_len}个多余值")
                linucb_model.edp_history = linucb_model.edp_history[:action_len]
            
            # 再次验证修复结果
            if len(linucb_model.action_history) != len(linucb_model.edp_history):
                logger.error("❌ 自动修复失败，无法继续频率细化")
                return None
            else:
                logger.info("✅ 历史长度不匹配问题已修复，继续频率细化")
        
        # 计算每个频率的平均EDP（EDP越小越好）
        freq_edp_history = defaultdict(list)
        for action, edp in zip(linucb_model.action_history, linucb_model.edp_history):
            # 处理组合频率：提取核心频率进行比较
            core_freq = action[0] if isinstance(action, tuple) else action
            if core_freq in explored_frequencies:
                freq_edp_history[core_freq].append(edp)
        
        logger.info(f"📊 频率EDP统计:")
        logger.info(f"   总历史记录: {len(linucb_model.edp_history)}")
        logger.info(f"   涉及频率数: {len(freq_edp_history)}")
        
        if not freq_edp_history:
            logger.info("❌ 没有找到任何有效的频率EDP记录")
            return None
        
        # 显示每个频率的样本数
        for freq, edp_list in freq_edp_history.items():
            logger.info(f"   {freq}MHz: {len(edp_list)}个样本, 平均EDP: {np.mean(edp_list):.4f}")
        
        # 找到平均EDP最小的频率
        freq_avg_edp = {}
        insufficient_samples = []
        for freq, edp_list in freq_edp_history.items():
            if len(edp_list) >= 4:  # 至少需要4个样本
                freq_avg_edp[freq] = np.mean(edp_list)
            else:
                insufficient_samples.append((freq, len(edp_list)))
        
        logger.info(f"📊 样本数检查结果:")
        logger.info(f"   满足条件的频率(>=3样本): {len(freq_avg_edp)}")
        logger.info(f"   样本不足的频率: {insufficient_samples}")
        
        if not freq_avg_edp:
            logger.info("❌ 没有频率满足最小样本数要求(4个)")
            logger.info("💡 建议: 降低样本数要求或继续训练更多轮次")
            return None
        
        best_freq = min(freq_avg_edp.keys(), key=lambda f: freq_avg_edp[f])
        best_edp = freq_avg_edp[best_freq]
        
        logger.info(f"🎯 整体历史EDP最佳频率: {best_freq}MHz (平均EDP: {best_edp:.3f})")
        
        return best_freq

    def get_sampling_statistics(self) -> Dict[str, Union[int, float, str]]:
        """获取采样统计信息"""
        stats = {
            'mode': self.current_mode.value,
            'total_actions': len(self.current_frequencies),
            'frequency_range': f"{min(self.current_frequencies)}-{max(self.current_frequencies)}MHz",
            'refinement_count': self.refinement_count,
            'slo_boundary': self.slo_violation_boundary,
            'zones': len(self.frequency_zones)
        }
        
        if self.current_mode == SamplingMode.SLO_AWARE and self.slo_violation_boundary:
            safe_count = len([f for f in self.current_frequencies 
                            if f >= self.slo_violation_boundary])
            stats['safe_ratio'] = safe_count / len(self.current_frequencies)
        
        # 高奖励区域统计已简化（reward_zones不再使用）
        
        return stats
    
    def reset(self):
        """重置采样器状态"""
        self.current_frequencies = []
        self.slo_violation_boundary = None
        self.slo_violation_history.clear()
        self.frequency_rewards.clear()
        self.frequency_zones = []
        self.refinement_count = 0
        
        logger.info("🔄 自适应频率采样器已重置")


def create_default_sampler(min_freq: int = 210, 
                          max_freq: int = 2100,
                          optimal_search_range: int = 150,
                          learner_maturity_threshold: int = 100,
                          refinement_start_threshold: int = 50) -> AdaptiveFrequencySampler:
    """创建默认配置的自适应采样器"""
    config = AdaptiveSamplingConfig(
        min_freq=min_freq,
        max_freq=max_freq,
        optimal_search_range=optimal_search_range,
        learner_maturity_threshold=learner_maturity_threshold,
        refinement_start_threshold=refinement_start_threshold
    )
    return AdaptiveFrequencySampler(config)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建采样器
    sampler = create_default_sampler()
    
    print("🧪 测试自适应频率采样器")
    print("="*50)
    
    # 测试SLO模式
    sampler.set_mode(SamplingMode.SLO_AWARE)
    slo_freqs = sampler.get_current_frequencies()
    print(f"SLO模式初始频率: {len(slo_freqs)}个")
    print(f"频率范围: {min(slo_freqs)}-{max(slo_freqs)}MHz")
    
    # 模拟SLO违反发现
    sampler.update_slo_boundary(780)
    slo_refined = sampler.get_current_frequencies()
    print(f"SLO边界细化后: {len(slo_refined)}个")
    
    print("\n" + "="*50)
    
    # 测试EDP模式
    sampler.set_mode(SamplingMode.EDP_OPTIMAL)
    edp_freqs = sampler.get_current_frequencies()
    print(f"EDP模式初始频率: {len(edp_freqs)}个")
    
    # 模拟奖励反馈
    for freq in edp_freqs:
        if 600 <= freq <= 900:  # 模拟高奖励区域
            sampler.update_reward_feedback(freq, 0.9)
        else:
            sampler.update_reward_feedback(freq, 0.3)
    
    # 触发细化
    sampler.refinement_count = 50
    sampler.refine_frequency_space()
    edp_refined = sampler.get_current_frequencies()
    print(f"EDP奖励细化后: {len(edp_refined)}个")
    
    # 统计信息
    print(f"\n📊 统计信息: {sampler.get_sampling_statistics()}")