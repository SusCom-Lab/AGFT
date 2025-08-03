import numpy as np
import pickle
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
try:
    from .logger import setup_logger
except ImportError:
    # 处理直接运行时的导入问题
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from logger import setup_logger

logger = setup_logger(__name__)

class ContextualLinUCB:
    """
    标准Contextual LinUCB多臂老虎机
    
    频率作为动作(arms)，工作负载特征作为上下文(context)
    每个频率维护独立的线性模型：reward = context^T * theta_freq
    
    这是标准的contextual bandit，频率不作为特征输入！
    """
    
    def __init__(self, 
                 n_features: int = 7,
                 alpha: float = 1.0, 
                 lambda_reg: float = 1.0,
                 alpha_decay_rate: float = 0.01,
                 min_alpha: float = 0.1,
                 enable_action_pruning: bool = True,
                 pruning_check_interval: int = 100,
                 pruning_threshold: float = 1.0,
                 min_exploration_for_pruning: int = 5,
                 pruning_maturity_threshold: int = 100,
                 cascade_pruning_threshold: int = 600,
                 gpu_max_freq: int = None,  # GPU硬件支持的最大频率
                 # 极端频率即时修剪参数
                 enable_extreme_pruning: bool = True,
                 extreme_pruning_threshold: float = -1.5,
                 extreme_pruning_min_samples: int = 3,
                 extreme_pruning_max_rounds: int = 20,
                 model_dir: str = "data/models", 
                 auto_load: bool = True):
        
        self.n_features = n_features  # 上下文特征维度 (只包含工作负载特征)
        self.alpha = alpha           # UCB探索参数
        self.initial_alpha = alpha   # 保存初始alpha值
        self.lambda_reg = lambda_reg # 正则化参数
        self.alpha_decay_rate = alpha_decay_rate  # alpha衰减率
        self.min_alpha = min_alpha   # 最小alpha值
        
        # 智能动作修剪参数
        self.enable_action_pruning = enable_action_pruning
        self.pruning_check_interval = pruning_check_interval
        self.pruning_threshold = pruning_threshold
        self.min_exploration_for_pruning = min_exploration_for_pruning
        self.pruning_maturity_threshold = pruning_maturity_threshold
        self.cascade_pruning_threshold = cascade_pruning_threshold  # 固定阈值，用于备用
        self.gpu_max_freq = gpu_max_freq  # GPU硬件支持的最大频率
        self.adaptive_cascade_pruning = True  # 启用自适应级联修剪
        
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 动作空间管理
        self.available_frequencies = []  # 当前可用频率列表 (核心频率或组合频率)
        
        # 显存+核心频率组合优化支持（向后兼容）
        self.memory_optimization_enabled = False  # 是否启用显存频率优化
        self.available_memory_frequencies = []    # 可用显存频率列表
        
        # 精细的组合频率禁用管理
        self.globally_disabled_memory_frequencies = set()  # 全局禁用的显存频率（设置失败）
        self.core_memory_disabled_combinations = {}  # 核心频率特定的禁用显存频率 {core_freq: set(disabled_memory_freqs)}
        self.disabled_core_frequencies = set()  # 因显存频率问题被禁用的核心频率
        
        # 核心频率依赖的SLO边界传播管理
        self.core_specific_memory_slo_boundaries = {}  # 每个核心频率的显存频率SLO边界 {core_freq: min_allowed_memory_freq}
        
        # 智能修剪状态
        self.pruned_frequencies = set()  # 被修剪的频率集合
        self.last_pruning_check = 0      # 上次修剪检查的轮次
        self.pruning_history = []        # 修剪历史记录
        
        # 极端频率即时修剪参数
        self.enable_extreme_pruning = enable_extreme_pruning  # 启用极端频率即时修剪
        self.extreme_pruning_threshold = extreme_pruning_threshold  # 极端差频率的奖励阈值
        self.extreme_pruning_min_samples = extreme_pruning_min_samples  # 判断极端频率的最小样本数
        self.extreme_pruning_max_rounds = extreme_pruning_max_rounds  # 在前N轮内进行极端修剪
        
        # 每个频率(动作)维护独立的线性模型
        # arm_models[freq] = {'A': A_matrix, 'b': b_vector, 'theta': theta_vector}
        self.arm_models = {}  
        
        # 统计信息
        self.total_rounds = 0
        self.total_reward = 0.0
        self.reward_history = []
        self.action_history = []  # 记录每次选择的频率
        self.context_history = []  # 记录每次的上下文
        self.edp_history = []  # 记录每次的原始EDP值
        
        # 每个频率的选择次数和累积奖励
        self.arm_counts = {}     # freq -> count
        self.arm_rewards = {}    # freq -> total_reward
        
        # 收敛状态
        self.exploitation_mode = False
        self.is_converged = False
        
        # 学习阶段顺序遍历控制
        self.learning_phase_complete = False  # 是否完成学习阶段遍历
        self.learning_frequency_index = 0    # 当前学习遍历的频率索引
        
        # 模型元数据
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'version': '5.0-contextual-bandit',
            'algorithm': 'contextual_linucb',
            'frequency_as_action': True,  # 标记：频率作为动作，不是特征
        }
        
        logger.info(f"🎯 初始化Contextual LinUCB:")
        logger.info(f"   上下文特征: {n_features}维 (仅工作负载特征)")
        logger.info(f"   频率作为动作: 是 (每个频率独立建模)")
        logger.info(f"   UCB参数α: {alpha}")
        logger.info(f"   正则化λ: {lambda_reg}")
        
        # 极端修剪配置信息
        if self.enable_extreme_pruning:
            logger.info(f"🚨 极端频率修剪已启用 (阈值: {self.extreme_pruning_threshold}, 前{self.extreme_pruning_max_rounds}轮检查)")
        else:
            logger.info("⚠️  极端频率修剪已禁用")
        
        if auto_load:
            self.load_model()
        else:
            logger.info("🆕 跳过模型加载，从零开始")
    
    def enable_memory_frequency_optimization(self, memory_frequencies: List[int]):
        """
        启用显存+核心频率组合优化（向后兼容）
        
        Args:
            memory_frequencies: 支持的显存频率列表
        """
        self.memory_optimization_enabled = True
        self.available_memory_frequencies = memory_frequencies
        
        logger.info(f"🔧 启用显存+核心频率组合优化")
        logger.info(f"   支持的显存频率: {memory_frequencies}")
        logger.info(f"   动作空间将扩展为 (核心频率, 显存频率) 组合")
        
        # 如果已有核心频率数据，重新构建组合动作空间
        if self.available_frequencies:
            self._rebuild_action_space()
    
    def _rebuild_action_space(self):
        """重建动作空间（从核心频率转换为组合频率或反之）"""
        if not self.memory_optimization_enabled:
            # 禁用显存频率优化时，确保动作空间只包含核心频率
            return
            
        # 构建新的组合动作空间
        old_arm_models = self.arm_models.copy()
        old_arm_counts = self.arm_counts.copy() 
        old_arm_rewards = self.arm_rewards.copy()
        
        # 清空原有模型
        self.arm_models = {}
        self.arm_counts = {}
        self.arm_rewards = {}
        
        # 迁移核心频率数据到组合频率
        for core_freq in self.available_frequencies:
            if core_freq in old_arm_models:
                # 为每个显存频率创建组合动作
                for mem_freq in self.available_memory_frequencies:
                    action = (core_freq, mem_freq)
                    # 复制原核心频率的模型参数
                    self.arm_models[action] = old_arm_models[core_freq].copy()
                    self.arm_counts[action] = old_arm_counts.get(core_freq, 0)
                    self.arm_rewards[action] = old_arm_rewards.get(core_freq, 0.0)
        
        logger.info(f"🔄 动作空间重建完成：{len(old_arm_models)}个核心频率 → {len(self.arm_models)}个组合动作")
    
    def _init_arm_model(self, action):
        """为新动作初始化线性模型（支持核心频率或组合频率）"""
        if action not in self.arm_models:
            # 使用稍大的初始正则化确保数值稳定性
            initial_reg = max(self.lambda_reg, 5.0)  # 至少5.0的正则化，提高数值稳定性
            
            self.arm_models[action] = {
                'A': initial_reg * np.eye(self.n_features, dtype=np.float64),  # 使用double精度
                'b': np.zeros(self.n_features, dtype=np.float64),
                'theta': np.zeros(self.n_features, dtype=np.float64)
            }
            self.arm_counts[action] = 0
            self.arm_rewards[action] = 0.0
            
            # 动作描述（向后兼容）
            if self.memory_optimization_enabled and isinstance(action, tuple):
                action_desc = f"{action[0]}MHz核心+{action[1]}MHz显存"
            else:
                action_desc = f"{action}MHz核心"
            
            logger.debug(f"🆕 初始化动作 {action_desc} 的线性模型 (正则化={initial_reg})")
    
    def _get_current_alpha(self) -> float:
        """计算当前的alpha值（应用衰减）"""
        if self.total_rounds == 0:
            return self.alpha
        
        # 指数衰减：alpha = initial_alpha * exp(-decay_rate * rounds)
        decayed_alpha = self.initial_alpha * np.exp(-self.alpha_decay_rate * self.total_rounds)
        
        # 确保不低于最小值
        current_alpha = max(decayed_alpha, self.min_alpha)
        
        return current_alpha
    
    def _get_representative_context(self) -> Optional[np.ndarray]:
        """获取代表性上下文（用于修剪检查）"""
        if not self.context_history:
            return None
        
        # 使用最近的上下文历史（最多50个）计算平均值
        recent_contexts = self.context_history[-50:]
        if len(recent_contexts) == 0:
            return None
        
        # 计算平均上下文
        representative_context = np.mean(recent_contexts, axis=0)
        
        logger.debug(f"🔍 计算代表性上下文: 基于最近{len(recent_contexts)}个上下文")
        
        return representative_context
    
    def _should_perform_pruning_check(self) -> bool:
        """判断是否应该执行修剪检查"""
        if not self.enable_action_pruning:
            logger.debug(f"🗂️ 修剪检查: 未启用 (enable_action_pruning=False)")
            return False
        
        # 检查是否达到成熟度门槛
        if self.total_rounds < self.pruning_maturity_threshold:
            logger.debug(f"🗂️ 修剪检查: 未达成熟度 ({self.total_rounds} < {self.pruning_maturity_threshold})")
            return False
        
        # 检查是否到了修剪检查间隔
        rounds_since_last_check = self.total_rounds - self.last_pruning_check
        if rounds_since_last_check < self.pruning_check_interval:
            logger.debug(f"🗂️ 修剪检查: 间隔不足 ({rounds_since_last_check} < {self.pruning_check_interval})")
            return False
        
        # 检查是否有足够的频率（至少保留2个）
        # 注意：available_frequencies 已经通过统一过滤方法处理，不包含已修剪频率
        available_count = len(self.available_frequencies)
        if available_count <= 2:
            logger.debug(f"🗂️ 修剪检查: 可用频率太少 ({available_count} <= 2)")
            return False
        
        logger.info(f"🗂️ 修剪检查: 满足条件 (轮次{self.total_rounds}, 距上次{rounds_since_last_check}轮, 可用{available_count}个)")
        return True
    
    def _get_adaptive_cascade_threshold(self) -> int:
        """计算自适应的级联修剪阈值"""
        if not self.adaptive_cascade_pruning or self.gpu_max_freq is None:
            return self.cascade_pruning_threshold
        
        # 自适应阈值：GPU硬件最大频率的一半
        adaptive_threshold = self.gpu_max_freq // 2
        
        logger.debug(f"🔄 自适应级联修剪阈值: {adaptive_threshold}MHz (GPU最大频率{self.gpu_max_freq}MHz的一半)")
        
        return adaptive_threshold
    
    def _perform_cascade_pruning(self, trigger_freq: int) -> list:
        """执行级联修剪逻辑，返回被级联修剪的频率列表"""
        cascade_pruned = []
        cascade_threshold = self._get_adaptive_cascade_threshold()
        
        if trigger_freq < cascade_threshold:
            # 找到需要级联修剪的频率
            # 注意：available_frequencies 应该已经不包含已修剪频率
            available_for_evaluation = self.available_frequencies.copy()
            
            # 安全检查：确保不包含已修剪频率
            if self.pruned_frequencies:
                available_for_evaluation = [f for f in available_for_evaluation if f not in self.pruned_frequencies]
            
            for potential_freq in available_for_evaluation:
                if potential_freq <= trigger_freq and potential_freq != trigger_freq:
                    self.pruned_frequencies.add(potential_freq)
                    cascade_pruned.append(potential_freq)
                    
                    # 计算级联修剪频率的EDP信息（如果有的话）
                    avg_edp = None
                    edp_samples = 0
                    if hasattr(self, 'edp_history') and self.edp_history:
                        freq_edp_values = []
                        for action, edp in zip(self.action_history, self.edp_history):
                            # 处理组合频率情况，只比较核心频率
                            action_freq = action[0] if isinstance(action, tuple) else action
                            if action_freq == potential_freq and edp > 0:
                                freq_edp_values.append(edp)
                        if freq_edp_values:
                            avg_edp = np.mean(freq_edp_values)
                            edp_samples = len(freq_edp_values)
                    
                    # 为级联修剪的频率添加修剪记录
                    cascade_record = {
                        'round': self.total_rounds,
                        'frequency': potential_freq,
                        'historical_avg_reward': self.arm_rewards.get(potential_freq, 0.0) / max(self.arm_counts.get(potential_freq, 1), 1),
                        'historical_avg_edp': avg_edp,
                        'edp_samples': edp_samples,
                        'exploration_count': self.arm_counts.get(potential_freq, 0),
                        'threshold': cascade_threshold,
                        'reason': f'级联修剪: {trigger_freq}MHz<{cascade_threshold}被修剪，连带修剪所有≤{trigger_freq}MHz频率',
                        'pruning_type': 'cascade',
                        'cascade_trigger': trigger_freq
                    }
                    self.pruning_history.append(cascade_record)
        
        return cascade_pruned

    def _check_extreme_frequency_pruning(self, freq: int):
        """检查并执行极端频率即时修剪"""
        # 只在该频率未被修剪时才检查
        if freq in self.pruned_frequencies:
            return
        
        # 检查该频率是否有足够的样本
        count = self.arm_counts[freq]
        if count < self.extreme_pruning_min_samples:
            return
        
        # 计算该频率的平均奖励
        avg_reward = self.arm_rewards[freq] / count
        
        # 如果平均奖励极端糟糕，立即修剪
        if avg_reward <= self.extreme_pruning_threshold:
            logger.warning(f"⚡ 极端频率即时修剪: {freq}MHz (平均奖励: {avg_reward:.3f} <= {self.extreme_pruning_threshold})")
            
            # 执行修剪
            self.pruned_frequencies.add(freq)
            
            # 记录主要修剪历史
            pruning_record = {
                'round': self.total_rounds,
                'frequency': freq,
                'historical_avg_reward': avg_reward,
                'exploration_count': count,
                'threshold': self.extreme_pruning_threshold,
                'reason': f'极端频率即时修剪: 平均奖励{avg_reward:.3f}极端糟糕',
                'pruning_type': 'extreme_immediate'
            }
            self.pruning_history.append(pruning_record)
            
            # 检查是否需要级联修剪
            cascade_pruned = self._perform_cascade_pruning(freq)
            
            if cascade_pruned:
                logger.warning(f"📉 级联修剪: 同时移除 {sorted(cascade_pruned)} MHz (≤{freq}MHz)")
            
            # 更新可用频率列表
            self.available_frequencies = [f for f in self.available_frequencies 
                                        if f not in self.pruned_frequencies]
            
            logger.info(f"🚫 即时修剪完成: 剩余 {len(self.available_frequencies)} 个可用频率")
    
    def _perform_action_pruning(self):
        """执行智能动作修剪 - 基于EDP统计学动态阈值 (自适应的永恒标尺)"""
        if not self._should_perform_pruning_check():
            return
        
        # 更新上次修剪检查时间
        self.last_pruning_check = self.total_rounds
        
        # 注意：available_frequencies 应该已经通过统一过滤，不包含已修剪频率
        # 但为了安全起见，这里仍然进行检查
        available_for_evaluation = self.available_frequencies.copy()
        
        # 安全检查：移除可能意外包含的已修剪频率
        if self.pruned_frequencies:
            before_count = len(available_for_evaluation)
            available_for_evaluation = [f for f in available_for_evaluation if f not in self.pruned_frequencies]
            if len(available_for_evaluation) != before_count:
                logger.warning(f"⚠️ available_frequencies包含已修剪频率，已清理 ({before_count} -> {len(available_for_evaluation)})")
        
        if len(available_for_evaluation) <= 2:
            logger.debug("📊 可用频率太少，跳过修剪")
            return
        
        # 检查是否有EDP历史数据
        if not hasattr(self, 'edp_history') or not self.edp_history:
            logger.debug("📊 无EDP历史数据，跳过基于EDP的修剪")
            return
        
        # 计算所有频率的历史平均EDP
        historical_edp_data = {}
        for freq in available_for_evaluation:
            exploration_count = self.arm_counts.get(freq, 0)
            if exploration_count >= self.min_exploration_for_pruning:
                # 收集该频率的所有EDP记录
                freq_edp_values = []
                for action, edp in zip(self.action_history, self.edp_history):
                    # 处理组合频率情况，只比较核心频率
                    action_freq = action[0] if isinstance(action, tuple) else action
                    if action_freq == freq and edp > 0:  # 只考虑有效的EDP值
                        freq_edp_values.append(edp)
                
                if len(freq_edp_values) >= self.min_exploration_for_pruning:
                    avg_edp = np.mean(freq_edp_values)
                    historical_edp_data[freq] = {
                        'avg_edp': avg_edp,
                        'exploration_count': len(freq_edp_values),
                        'edp_values': freq_edp_values
                    }
        
        if len(historical_edp_data) < 2:
            logger.debug(f"📊 有充分EDP历史的频率太少({len(historical_edp_data)}个)，跳过修剪")
            return
        
        # 找到EDP表现最好的频率 (EDP越小越好)
        best_freq = min(historical_edp_data.keys(), key=lambda f: historical_edp_data[f]['avg_edp'])
        best_avg_edp = historical_edp_data[best_freq]['avg_edp']
        
        # 计算基于标准差的动态EDP阈值
        all_avg_edp_values = [data['avg_edp'] for data in historical_edp_data.values()]
        edp_std = np.std(all_avg_edp_values) if len(all_avg_edp_values) > 1 else 0.1
        
        # 动态阈值：配置的倍数 * EDP标准差
        # 使用现有的 pruning_threshold 配置作为标准差倍数
        edp_threshold = self.pruning_threshold * max(edp_std, 0.05)  # 最小阈值0.05J·s
        
        logger.debug(f"📊 EDP统计: 平均值范围{min(all_avg_edp_values):.3f}-{max(all_avg_edp_values):.3f}J·s, "
                    f"标准差{edp_std:.3f}J·s, 动态阈值{edp_threshold:.3f}J·s")
        
        # 执行基于EDP的修剪逻辑
        newly_pruned = []
        
        for freq, data in historical_edp_data.items():
            if freq == best_freq:
                continue  # 不修剪EDP表现最优的频率
            
            # 检查修剪条件
            edp_gap = data['avg_edp'] - best_avg_edp  # EDP差距（正值表示更差）
            exploration_count = data['exploration_count']
            
            # 修剪条件：
            # 1. 历史平均EDP远高于最优（能效差）
            # 2. 已经被充分探索
            # 3. 不是最后剩余的几个频率
            if (edp_gap > edp_threshold and 
                exploration_count >= self.min_exploration_for_pruning and
                len(available_for_evaluation) - len(newly_pruned) > 2):  # 至少保留2个频率
                
                self.pruned_frequencies.add(freq)
                newly_pruned.append(freq)
                
                # 记录EDP修剪历史
                pruning_record = {
                    'round': self.total_rounds,
                    'frequency': freq,
                    'historical_avg_edp': data['avg_edp'],
                    'best_historical_edp': best_avg_edp,
                    'best_frequency': best_freq,
                    'edp_gap': edp_gap,
                    'exploration_count': exploration_count,
                    'edp_threshold': edp_threshold,
                    'edp_std': edp_std,
                    'edp_samples': len(data['edp_values']),
                    'std_multiplier': self.pruning_threshold,
                    'reason': f'历史平均EDP比最优高{edp_gap:.3f}J·s > {edp_threshold:.3f}J·s({self.pruning_threshold:.1f}×标准差)',
                    'permanently_banned': True,
                    'pruning_method': 'EDP_dynamic_threshold'
                }
                self.pruning_history.append(pruning_record)
        
        # 执行级联修剪（为所有新修剪的频率）
        all_cascade_pruned = []
        for freq in newly_pruned:
            cascade_pruned = self._perform_cascade_pruning(freq)
            all_cascade_pruned.extend(cascade_pruned)
        
        # 输出修剪结果
        if newly_pruned or all_cascade_pruned:
            total_pruned = len(newly_pruned) + len(all_cascade_pruned)
            logger.info(f"🗂️ [EDP智能修剪] 轮次{self.total_rounds}: 修剪{total_pruned}个频率")
            logger.info(f"   EDP最优频率: {best_freq}MHz (平均EDP: {best_avg_edp:.3f}J·s)")
            logger.info(f"   动态修剪阈值: {edp_threshold:.3f}J·s ({self.pruning_threshold:.1f}×标准差{edp_std:.3f})")
            
            # 显示基于EDP主动修剪的频率
            for freq in newly_pruned:
                record = next(r for r in self.pruning_history if r['frequency'] == freq and r['round'] == self.total_rounds and 'cascade_trigger' not in r)
                logger.info(f"   🎯 EDP修剪: {freq}MHz (平均EDP: {record['historical_avg_edp']:.3f}J·s, "
                          f"差距: +{record['edp_gap']:.3f}J·s, 样本: {record['edp_samples']}个)")
            
            # 显示级联修剪的频率
            if all_cascade_pruned:
                cascade_threshold = self._get_adaptive_cascade_threshold()
                logger.info(f"   🔗 级联修剪(<{cascade_threshold}MHz触发): {sorted(all_cascade_pruned)}MHz")
                for freq in sorted(all_cascade_pruned):
                    cascade_record = next(r for r in self.pruning_history if r['frequency'] == freq and r['round'] == self.total_rounds and 'cascade_trigger' in r)
                    # 获取级联修剪频率的EDP信息
                    if freq in historical_edp_data:
                        edp_info = f"平均EDP: {historical_edp_data[freq]['avg_edp']:.3f}J·s"
                    else:
                        edp_info = f"探索: {cascade_record['exploration_count']}次"
                    logger.info(f"      ↳ {freq}MHz ({edp_info})")
            
            # 注意: available_frequencies 的同步由外部调用方负责
            # 这里只负责维护 pruned_frequencies 状态
            active_count = len([f for f in self.available_frequencies if f not in self.pruned_frequencies])
            logger.info(f"   剩余活跃频率: {active_count}个 (需外部同步available_frequencies)")
        else:
            logger.debug(f"📊 EDP修剪检查完成，无频率需要修剪 (EDP最优: {best_freq}MHz, 平均EDP: {best_avg_edp:.3f}J·s, "
                        f"动态阈值: {edp_threshold:.3f}J·s)")
    
    def _update_theta(self, action):
        """更新动作的参数向量 theta = A^-1 * b（支持核心频率或组合频率）"""
        model = self.arm_models[action]
        try:
            # 使用更稳定的求解方法
            model['theta'] = np.linalg.solve(model['A'], model['b']).astype(np.float32)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，添加更大的正则化
            # 动作描述（向后兼容）
            if self.memory_optimization_enabled and isinstance(action, tuple):
                action_desc = f"{action[0]}MHz核心+{action[1]}MHz显存"
            else:
                action_desc = f"{action}MHz核心"
            
            logger.warning(f"动作 {action_desc} 的A矩阵奇异，增加正则化")
            model['A'] += 0.1 * np.eye(self.n_features)
            model['theta'] = np.linalg.solve(model['A'], model['b']).astype(np.float32)
    
    def update_action_space(self, frequencies: List[int]):
        """更新可用频率列表 - 过滤掉被禁用的频率"""
        if not frequencies:
            logger.warning("⚠️ 收到空的频率列表，保持当前动作空间不变")
            return
        
        # 过滤掉被禁用的核心频率
        filtered_frequencies = [freq for freq in frequencies if freq not in self.disabled_core_frequencies]
        
        if not filtered_frequencies:
            logger.warning("⚠️ 过滤后没有可用频率，保持当前动作空间不变")
            return
        
        if len(filtered_frequencies) != len(frequencies):
            logger.info(f"🚫 过滤禁用频率: {len(frequencies)} -> {len(filtered_frequencies)} 个频率")
            logger.debug(f"   禁用的频率: {sorted(set(frequencies) - set(filtered_frequencies))}")
        
        old_freqs = set(self.available_frequencies)
        new_freqs = set(filtered_frequencies)
        
        # 为新频率初始化模型
        added_freqs = new_freqs - old_freqs
        for freq in added_freqs:
            self._init_arm_model(freq)
            logger.debug(f"➕ 添加新频率: {freq}MHz")
        
        # 移除的频率（可能是被调用方过滤掉的）
        removed_freqs = old_freqs - new_freqs
        if removed_freqs:
            logger.debug(f"➖ 移除频率: {sorted(removed_freqs)}")
        
        # 更新频率列表
        sorted_frequencies = sorted(filtered_frequencies)
        logger.info(f"🎯 动作空间更新: {sorted_frequencies} (共{len(filtered_frequencies)}个频率)")
        
        self.available_frequencies = filtered_frequencies.copy()
        logger.debug(f"🔄 频率空间状态: 可用{len(filtered_frequencies)}个, 已修剪{len(self.pruned_frequencies)}个")
    
    def add_actual_frequency(self, actual_freq: int):
        """
        动态添加实际使用的频率到动作空间
        当目标频率设置失败时，添加实际采用的频率以扩展动作空间
        
        Args:
            actual_freq: 实际设置成功的频率（MHz）
        """
        if actual_freq in self.available_frequencies:
            logger.debug(f"🔄 频率 {actual_freq}MHz 已存在于动作空间中")
            return
        
        if actual_freq in self.disabled_core_frequencies:
            logger.warning(f"⚠️ 频率 {actual_freq}MHz 已被禁用，不能添加到动作空间")
            return
        
        if actual_freq in self.pruned_frequencies:
            logger.warning(f"⚠️ 频率 {actual_freq}MHz 已被修剪，不能添加到动作空间")
            return
        
        # 添加到可用频率列表
        self.available_frequencies.append(actual_freq)
        self.available_frequencies.sort()  # 保持排序
        
        # 初始化新频率的模型
        self._init_arm_model(actual_freq)
        
        logger.info(f"✅ 动态添加实际频率 {actual_freq}MHz 到动作空间 (总计{len(self.available_frequencies)}个频率)")
        logger.debug(f"🔄 更新后的动作空间: {sorted(self.available_frequencies)}")
    
    def select_action(self, context: np.ndarray, available_frequencies: List[int]):
        """
        使用LinUCB算法选择最优动作 - 支持核心频率或组合频率优化
        
        Args:
            context: 工作负载上下文特征 (n_features维)
            available_frequencies: 当前可用核心频率列表
        
        Returns:
            selected_action: 选择的动作 (频率MHz 或 (核心频率MHz, 显存频率MHz) 元组)
        """
        if not available_frequencies:
            raise ValueError("可用频率列表为空")
        
        # 构建动作空间 - 使用新的有效动作过滤逻辑
        if self.memory_optimization_enabled:
            # 组合频率模式：构建有效的 (核心频率, 显存频率) 组合
            available_actions = []
            for core_freq in available_frequencies:
                if core_freq in self.disabled_core_frequencies:
                    continue  # 跳过被禁用的核心频率
                    
                for mem_freq in self.available_memory_frequencies:
                    action = (core_freq, mem_freq)
                    if self.is_action_allowed(action):
                        available_actions.append(action)
        else:
            # 核心频率模式：只使用未被禁用的核心频率
            available_actions = [freq for freq in available_frequencies 
                               if freq not in self.disabled_core_frequencies]
        
        if not available_actions:
            logger.error("❌ 所有动作都被禁用了！")
            # 应急情况：使用第一个可用的核心频率
            if available_frequencies:
                if self.memory_optimization_enabled and self.available_memory_frequencies:
                    emergency_action = (available_frequencies[0], self.available_memory_frequencies[0])
                else:
                    emergency_action = available_frequencies[0]
                logger.warning(f"🚨 使用应急动作: {emergency_action}")
                available_actions = [emergency_action]
            else:
                raise ValueError("没有任何可用的动作！")
        
        # 确保所有动作都有模型
        for action in available_actions:
            self._init_arm_model(action)
        
        # 学习阶段：从大到小顺序遍历所有动作
        if not self.learning_phase_complete:
            # 初始化学习动作列表（只在第一次时设置）
            if not hasattr(self, '_learning_action_list'):
                if self.memory_optimization_enabled:
                    # 组合频率模式：按核心频率从高到低，每个核心频率搭配所有显存频率
                    self._learning_action_list = []
                    for core_freq in sorted(available_frequencies, reverse=True):
                        for mem_freq in sorted(self.available_memory_frequencies, reverse=True):
                            self._learning_action_list.append((core_freq, mem_freq))
                else:
                    # 核心频率模式：按频率从高到低
                    self._learning_action_list = sorted(available_frequencies, reverse=True)
                
                logger.info(f"📚 初始化学习阶段动作列表: {len(self._learning_action_list)}个动作")
                
            # 重命名索引变量以反映新的动作概念
            if not hasattr(self, 'learning_action_index'):
                self.learning_action_index = getattr(self, 'learning_frequency_index', 0)
            
            # 跳过已经失败的动作，继续下一个可用动作
            while self.learning_action_index < len(self._learning_action_list):
                candidate_action = self._learning_action_list[self.learning_action_index]
                
                # 检查候选动作是否仍然可用
                if candidate_action in available_actions:
                    self.learning_action_index += 1
                    
                    # 动作描述（向后兼容）
                    if self.memory_optimization_enabled and isinstance(candidate_action, tuple):
                        action_desc = f"{candidate_action[0]}MHz核心+{candidate_action[1]}MHz显存"
                    else:
                        action_desc = f"{candidate_action}MHz核心"
                    
                    logger.info(f"📚 [学习阶段遍历] 选择动作 {action_desc} ({self.learning_action_index}/{len(self._learning_action_list)}) - 从高到低")
                    return candidate_action
                else:
                    # 跳过失败或不可用的动作
                    self.learning_action_index += 1
                    
                    # 动作描述（向后兼容）
                    if self.memory_optimization_enabled and isinstance(candidate_action, tuple):
                        action_desc = f"{candidate_action[0]}MHz核心+{candidate_action[1]}MHz显存"
                    else:
                        action_desc = f"{candidate_action}MHz核心"
                    
                    logger.info(f"⏭️ [学习阶段遍历] 跳过不可用动作 {action_desc} ({self.learning_action_index}/{len(self._learning_action_list)})")
                    continue
            
            # 完成学习阶段遍历
            self.learning_phase_complete = True
            logger.info(f"✅ 学习阶段遍历完成，开始LinUCB算法选择")
        
        # 正常LinUCB选择逻辑
        current_alpha = self._get_current_alpha()
        fallback_confidence = current_alpha * 10.0
        
        # 计算每个动作的UCB值
        ucb_values = {}
        predictions = {}
        confidence_widths = {}
        
        for action in available_actions:
            model = self.arm_models[action]
            
            # 预测奖励: theta^T * context
            predicted_reward = np.dot(model['theta'], context)
            
            # 计算置信区间宽度: alpha * sqrt(context^T * A^-1 * context)
            try:
                A_inv_context = np.linalg.solve(model['A'], context)
                quadratic_form = np.dot(context, A_inv_context)
                
                # 检查数值稳定性
                if quadratic_form < 0:
                    if abs(quadratic_form) < 1e-10:
                        quadratic_form = 1e-10
                        
                        # 动作描述（向后兼容）
                        if self.memory_optimization_enabled and isinstance(action, tuple):
                            action_desc = f"{action[0]}MHz核心+{action[1]}MHz显存"
                        else:
                            action_desc = f"{action}MHz核心"
                        logger.debug(f"动作 {action_desc} 二次型数值误差修正")
                    else:
                        # 动作描述（向后兼容）
                        if self.memory_optimization_enabled and isinstance(action, tuple):
                            action_desc = f"{action[0]}MHz核心+{action[1]}MHz显存"
                        else:
                            action_desc = f"{action}MHz核心"
                        logger.warning(f"动作 {action_desc} A矩阵不稳定，二次型={quadratic_form:.3e}")
                        confidence_width = fallback_confidence
                        quadratic_form = None
                elif np.isnan(quadratic_form) or np.isinf(quadratic_form):
                    # 动作描述（向后兼容）
                    if self.memory_optimization_enabled and isinstance(action, tuple):
                        action_desc = f"{action[0]}MHz核心+{action[1]}MHz显存"
                    else:
                        action_desc = f"{action}MHz核心"
                    logger.warning(f"动作 {action_desc} 二次型计算异常: {quadratic_form}")
                    confidence_width = fallback_confidence
                    quadratic_form = None
                
                if quadratic_form is not None:
                    confidence_width = current_alpha * np.sqrt(quadratic_form)
                    
            except np.linalg.LinAlgError as e:
                confidence_width = fallback_confidence
                # 动作描述（向后兼容）
                if self.memory_optimization_enabled and isinstance(action, tuple):
                    action_desc = f"{action[0]}MHz核心+{action[1]}MHz显存"
                else:
                    action_desc = f"{action}MHz核心"
                logger.warning(f"动作 {action_desc} A矩阵求解失败 ({e})")
            
            # UCB值 = 预测奖励 + 置信区间
            ucb_value = predicted_reward + confidence_width
            
            ucb_values[action] = ucb_value
            predictions[action] = predicted_reward
            confidence_widths[action] = confidence_width
            
            # 动作描述（向后兼容）
            if self.memory_optimization_enabled and isinstance(action, tuple):
                action_desc = f"{action[0]}MHz核心+{action[1]}MHz显存"
            else:
                action_desc = f"{action}MHz核心"
            
            logger.debug(f"  动作{action_desc}: 预测={predicted_reward:.3f}, "
                        f"置信区间={confidence_width:.3f}, UCB={ucb_value:.3f}")
        
        # 选择UCB值最大的动作
        selected_action = max(ucb_values.keys(), key=lambda a: ucb_values[a])
        
        # 动作描述（向后兼容）
        if self.memory_optimization_enabled and isinstance(selected_action, tuple):
            action_desc = f"{selected_action[0]}MHz核心+{selected_action[1]}MHz显存"
        else:
            action_desc = f"{selected_action}MHz核心"
        
        logger.info(f"🎯 [Contextual LinUCB] 选择动作 {action_desc}, "
                   f"预测奖励={predictions[selected_action]:.3f}, "
                   f"置信区间={confidence_widths[selected_action]:.3f}, "
                   f"UCB={ucb_values[selected_action]:.3f}, "
                   f"选择次数={self.arm_counts[selected_action] + 1}, "
                   f"当前α={current_alpha:.3f}")
        
        return selected_action
    
    def update(self, context: np.ndarray, action, reward: float, edp_value: Optional[float] = None):
        """
        更新模型参数（支持核心频率或组合频率动作）
        
        Args:
            context: 上下文特征
            action: 选择的动作 (频率MHz 或 (核心频率MHz, 显存频率MHz) 元组)
            reward: 观察到的奖励
            edp_value: 原始EDP值 (可选，用于性能退化检测)
        """
        # 确保动作有模型
        self._init_arm_model(action)
        
        # 更新该动作的线性模型
        model = self.arm_models[action]
        
        # 更新 A = A + context * context^T
        outer_product = np.outer(context, context)
        model['A'] += outer_product
        
        
        # 更新 b = b + reward * context  
        model['b'] += reward * context
        
        # 重新计算 theta = A^-1 * b
        self._update_theta(action)
        
        # 更新统计信息
        self.total_rounds += 1
        self.total_reward += reward
        self.reward_history.append(reward)
        self.action_history.append(action)
        self.context_history.append(context.copy())
        # 🔧 修复历史数组长度不匹配问题：确保edp_history与action_history长度一致
        if edp_value is not None:
            self.edp_history.append(edp_value)
        else:
            # 如果没有EDP值，使用默认值占位，确保数组长度一致
            self.edp_history.append(0.0)  # 使用0.0作为默认EDP占位值
            logger.debug(f"⚠️ 轮次{self.total_rounds}: EDP值为None，使用0.0占位")
        self.arm_rewards[action] += reward
        self.arm_counts[action] += 1  # 增加该动作的选择次数
        
        # 更新元数据
        self.metadata['updated_at'] = datetime.now().isoformat()
        self.metadata['total_rounds'] = self.total_rounds
        self.metadata['avg_reward'] = self.total_reward / max(self.total_rounds, 1)
        
        # 动作描述（向后兼容）
        if self.memory_optimization_enabled and isinstance(action, tuple):
            action_desc = f"{action[0]}MHz核心+{action[1]}MHz显存"
        else:
            action_desc = f"{action}MHz核心"
        
        logger.debug(f"📈 更新动作 {action_desc} 模型, 奖励={reward:.3f}, "
                    f"累积奖励={self.arm_rewards[action]:.3f}")
        
        # 检查是否需要进行极端频率即时修剪
        if (self.enable_extreme_pruning and 
            self.total_rounds <= self.extreme_pruning_max_rounds):
            self._check_extreme_frequency_pruning(action)
    
    def get_model_stats(self) -> dict:
        """获取模型统计信息"""
        # 计算每个频率的平均奖励
        avg_rewards = {}
        for freq in self.arm_models.keys():
            if self.arm_counts[freq] > 0:
                avg_rewards[freq] = self.arm_rewards[freq] / self.arm_counts[freq]
            else:
                avg_rewards[freq] = 0.0
        
        # 最近奖励
        recent_rewards = self.reward_history[-50:] if self.reward_history else []
        
        current_alpha = self._get_current_alpha()
        stats = {
            'total_rounds': self.total_rounds,
            'avg_reward': self.total_reward / max(self.total_rounds, 1),
            'recent_avg_reward': np.mean(recent_rewards) if recent_rewards else 0.0,
            'n_arms': len(self.arm_models),
            'arm_counts': dict(self.arm_counts),
            'arm_avg_rewards': avg_rewards,
            'exploration_strategy': 'LinUCB',
            'algorithm': 'contextual_bandit',
            'current_alpha': current_alpha,
            'initial_alpha': self.initial_alpha,
            'alpha_decay_rate': self.alpha_decay_rate,
            'min_alpha': self.min_alpha,
            'converged': self.is_converged,
            'metadata': self.metadata.copy(),
            # 智能修剪统计
            'action_pruning_enabled': self.enable_action_pruning,
            'pruned_frequencies_count': len(self.pruned_frequencies),
            'pruned_frequencies': sorted(list(self.pruned_frequencies)),
            'pruning_operations_count': len(self.pruning_history),
            'active_frequencies_count': len([f for f in self.available_frequencies if f not in self.pruned_frequencies]),
            # 级联修剪统计
            'cascade_pruning_count': len([r for r in self.pruning_history if 'cascade_trigger' in r]),
            'direct_pruning_count': len([r for r in self.pruning_history if 'cascade_trigger' not in r])
        }
        
        return stats
    
    def save_model(self, filename: str = None):
        """保存模型"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"contextual_linucb_model_{timestamp}.pkl"
        
        filepath = self.model_dir / filename
        
        # 准备保存数据
        model_data = {
            'arm_models': self.arm_models,
            'available_frequencies': self.available_frequencies,
            'total_rounds': self.total_rounds,
            'total_reward': self.total_reward,
            'reward_history': self.reward_history,
            'action_history': self.action_history,
            'context_history': self.context_history,
            'edp_history': self.edp_history,
            'arm_counts': self.arm_counts,
            'arm_rewards': self.arm_rewards,
            'exploitation_mode': self.exploitation_mode,
            'is_converged': self.is_converged,
            'metadata': self.metadata,
            # 超参数
            'n_features': self.n_features,
            'alpha': self.alpha,
            'lambda_reg': self.lambda_reg,
            # 智能修剪状态
            'pruned_frequencies': list(self.pruned_frequencies),
            'last_pruning_check': self.last_pruning_check,
            'pruning_history': self.pruning_history,
            # 修剪配置参数
            'enable_action_pruning': self.enable_action_pruning,
            'pruning_check_interval': self.pruning_check_interval,
            'pruning_threshold': self.pruning_threshold,
            'min_exploration_for_pruning': self.min_exploration_for_pruning,
            'pruning_maturity_threshold': self.pruning_maturity_threshold,
            # 学习阶段状态
            'learning_phase_complete': self.learning_phase_complete,
            'learning_frequency_index': self.learning_frequency_index,
            '_learning_frequency_list': getattr(self, '_learning_frequency_list', None)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # 清理旧模型文件，只保留1个最新的
        self._cleanup_old_models(current_file=filepath.name)
        
        # 创建最新模型链接
        latest_path = self.model_dir / "latest_contextual_model.pkl"
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(filepath.name)
        
        logger.info(f"💾 Contextual LinUCB模型已保存: {filepath}")
    
    def _cleanup_old_models(self, current_file: str):
        """清理旧模型文件，只保留1个最新的"""
        try:
            # 查找所有contextual_linucb_model文件
            model_files = list(self.model_dir.glob("contextual_linucb_model_*.pkl"))
            
            if len(model_files) <= 1:
                return  # 没有旧文件需要清理
            
            # 按修改时间排序，最新的在前
            model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            # 删除除当前文件外的所有文件
            deleted_count = 0
            for model_file in model_files:
                if model_file.name != current_file:
                    try:
                        model_file.unlink()
                        deleted_count += 1
                        logger.debug(f"🗑️ 删除旧模型文件: {model_file.name}")
                    except Exception as e:
                        logger.warning(f"⚠️ 删除旧模型文件失败 {model_file.name}: {e}")
            
            if deleted_count > 0:
                logger.info(f"🧹 清理完成，删除了 {deleted_count} 个旧模型文件")
                
        except Exception as e:
            logger.warning(f"⚠️ 模型文件清理失败: {e}")
    
    def load_model(self, filename: str = None):
        """加载模型"""
        if filename is None:
            # 尝试加载最新模型
            latest_path = self.model_dir / "latest_contextual_model.pkl"
            if latest_path.exists():
                filepath = latest_path
            else:
                # 查找最新的模型文件
                model_files = list(self.model_dir.glob("contextual_linucb_model_*.pkl"))
                if not model_files:
                    logger.info("📁 未找到已有模型文件，从零开始")
                    return
                filepath = max(model_files, key=lambda p: p.stat().st_mtime)
        else:
            filepath = self.model_dir / filename
        
        if not filepath.exists():
            logger.warning(f"⚠️ 模型文件不存在: {filepath}")
            return
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # 验证模型兼容性
            if model_data.get('n_features') != self.n_features:
                logger.warning("⚠️ 模型特征维度不兼容，从零开始")
                return
            
            # 加载数据
            self.arm_models = model_data.get('arm_models', {})
            self.available_frequencies = model_data.get('available_frequencies', [])
            self.total_rounds = model_data.get('total_rounds', 0)
            self.total_reward = model_data.get('total_reward', 0.0)
            self.reward_history = model_data.get('reward_history', [])
            self.action_history = model_data.get('action_history', [])
            self.context_history = model_data.get('context_history', [])
            self.edp_history = model_data.get('edp_history', [])
            self.arm_counts = model_data.get('arm_counts', {})
            self.arm_rewards = model_data.get('arm_rewards', {})
            self.exploitation_mode = model_data.get('exploitation_mode', False)
            self.is_converged = model_data.get('is_converged', False)
            self.metadata = model_data.get('metadata', self.metadata)
            # 加载修剪状态（向后兼容）
            self.pruned_frequencies = set(model_data.get('pruned_frequencies', []))
            self.last_pruning_check = model_data.get('last_pruning_check', 0)
            self.pruning_history = model_data.get('pruning_history', [])
            # 加载学习阶段状态（向后兼容）
            self.learning_phase_complete = model_data.get('learning_phase_complete', False)
            self.learning_frequency_index = model_data.get('learning_frequency_index', 0)
            saved_learning_list = model_data.get('_learning_frequency_list', None)
            if saved_learning_list is not None:
                self._learning_frequency_list = saved_learning_list
            
            logger.info(f"✅ Contextual LinUCB模型已加载: {filepath}")
            logger.info(f"   轮次: {self.total_rounds}, 平均奖励: {self.total_reward/max(self.total_rounds,1):.3f}")
            logger.info(f"   频率数量: {len(self.arm_models)}")
            if self.enable_action_pruning and self.pruned_frequencies:
                logger.info(f"   已修剪频率: {len(self.pruned_frequencies)}个 {sorted(list(self.pruned_frequencies))}")
                logger.info(f"   修剪历史: {len(self.pruning_history)}次修剪操作")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            logger.info("🆕 将从零开始")
    
    # 兼容性接口 (与原LinUCB保持一致)
    @property 
    def round_count(self) -> int:
        return self.total_rounds
    
    @property
    def phase(self) -> str:
        return "EXPLOITATION" if self.exploitation_mode else "LEARNING"
    
    @property
    def converged(self) -> bool:
        return self.is_converged
    
    @property
    def n_actions(self) -> int:
        return len(self.available_frequencies)
    
    def set_exploitation_mode(self, mode: bool):
        """设置利用模式"""
        self.exploitation_mode = mode
        logger.info(f"🔄 切换到{'利用' if mode else '学习'}模式")
    
    def select_action_exploitation(self, context: np.ndarray, 
                                  available_frequencies: List[int]) -> int:
        """利用模式选择 (贪心选择最佳预测)"""
        if not available_frequencies:
            raise ValueError("可用频率列表为空")
        
        best_freq = None
        best_prediction = float('-inf')
        
        for freq in available_frequencies:
            self._init_arm_model(freq)
            model = self.arm_models[freq]
            prediction = np.dot(model['theta'], context)
            
            if prediction > best_prediction:
                best_prediction = prediction
                best_freq = freq
        
        logger.info(f"🎯 [利用模式] 选择频率 {best_freq}MHz, 预测奖励={best_prediction:.3f}")
        return best_freq
    
    def predict_reward(self, context: np.ndarray, frequency: int) -> float:
        """
        预测指定频率在给定上下文下的期望奖励（不含探索噪声）
        
        Args:
            context: 上下文特征向量
            frequency: 目标频率
            
        Returns:
            float: 预测的期望奖励
        """
        # 确保频率模型已初始化
        self._init_arm_model(frequency)
        
        if frequency not in self.arm_models:
            logger.warning(f"频率 {frequency}MHz 初始化失败，返回0")
            return 0.0
        
        model = self.arm_models[frequency]
        
        try:
            # 计算期望奖励 E[r] = context^T * theta
            expected_reward = np.dot(context, model['theta'])
            logger.debug(f"预测频率 {frequency}MHz 期望奖励: {expected_reward:.4f}")
            return float(expected_reward)
            
        except Exception as e:
            logger.warning(f"预测频率 {frequency}MHz 奖励失败: {e}")
            return 0.0
    
    def get_ucb_values(self, context: np.ndarray) -> Dict[int, float]:
        """获取所有频率的UCB值（使用衰减后的alpha）"""
        ucb_values = {}
        
        # 使用衰减后的alpha值
        current_alpha = self._get_current_alpha()
        
        for freq in self.available_frequencies:
            self._init_arm_model(freq)
            model = self.arm_models[freq]
            
            predicted_reward = np.dot(model['theta'], context)
            
            try:
                A_inv = np.linalg.inv(model['A'])
                confidence_width = current_alpha * np.sqrt(
                    np.dot(context, np.dot(A_inv, context))
                )
            except np.linalg.LinAlgError:
                confidence_width = current_alpha * 10.0
            
            ucb_values[freq] = predicted_reward + confidence_width
        
        return ucb_values
    
    def get_confidence_intervals(self, context: np.ndarray) -> Dict[int, dict]:
        """获取置信区间"""
        intervals = {}
        
        for freq in self.available_frequencies:
            self._init_arm_model(freq)
            model = self.arm_models[freq]
            
            mean = np.dot(model['theta'], context)
            
            try:
                A_inv = np.linalg.inv(model['A'])
                current_alpha = self._get_current_alpha()
                width = current_alpha * np.sqrt(np.dot(context, np.dot(A_inv, context)))
            except np.linalg.LinAlgError:
                current_alpha = self._get_current_alpha()
                width = current_alpha * 10.0
            
            intervals[freq] = {
                'mean': mean,
                'lower': mean - width,
                'upper': mean + width,
                'width': 2 * width
            }
        
        return intervals
    
    def update_memory_frequencies(self, new_memory_frequencies: List[int]):
        """
        更新可用显存频率列表，并重新生成学习动作列表
        
        Args:
            new_memory_frequencies: 新的可用显存频率列表
        """
        if not self.memory_optimization_enabled:
            logger.debug("📝 显存频率优化未启用，跳过显存频率更新")
            return
        
        old_count = len(self.available_memory_frequencies)
        self.available_memory_frequencies = new_memory_frequencies.copy()
        new_count = len(self.available_memory_frequencies)
        
        logger.info(f"🔄 更新显存频率列表: {old_count} -> {new_count} 个频率")
        logger.debug(f"   新显存频率: {sorted(new_memory_frequencies)}")
        
        # 重新生成学习动作列表（如果在学习阶段）
        if hasattr(self, '_learning_frequency_list') and not self.learning_phase_complete:
            self._regenerate_learning_action_list()
            logger.info("📚 已重新生成学习动作列表以反映显存频率变化")
    
    def _regenerate_learning_action_list(self):
        """
        重新生成学习动作列表，考虑当前可用的核心频率和显存频率，以及所有禁用规则
        """
        if self.memory_optimization_enabled and self.available_memory_frequencies:
            # 组合频率模式：生成所有有效的 (核心频率, 显存频率) 组合
            self._learning_frequency_list = []
            for core_freq in self.available_frequencies:
                if core_freq in self.disabled_core_frequencies:
                    continue
                    
                for mem_freq in self.available_memory_frequencies:
                    action = (core_freq, mem_freq)
                    if self.is_action_allowed(action):
                        self._learning_frequency_list.append(action)
            
            logger.debug(f"🔄 重新生成组合频率学习列表: {len(self._learning_frequency_list)} 个有效组合")
            logger.debug(f"   核心频率: {len(self.available_frequencies)} 个")
            logger.debug(f"   显存频率: {len(self.available_memory_frequencies)} 个")
            logger.debug(f"   禁用核心频率: {len(self.disabled_core_frequencies)} 个")
            logger.debug(f"   全局禁用显存频率: {len(self.globally_disabled_memory_frequencies)} 个")
        else:
            # 仅核心频率模式
            self._learning_frequency_list = [freq for freq in self.available_frequencies 
                                           if freq not in self.disabled_core_frequencies]
            logger.debug(f"🔄 重新生成核心频率学习列表: {len(self._learning_frequency_list)} 个有效频率")
        
        # 重置学习索引到合理位置
        if self.learning_frequency_index >= len(self._learning_frequency_list):
            self.learning_frequency_index = 0
            logger.debug("📍 学习索引重置为0（超出新列表范围）")
    
    def disable_memory_frequency_globally(self, memory_freq: int, reason: str = "设置失败"):
        """
        全局禁用显存频率 - 所有核心频率都不再使用这个显存频率
        
        Args:
            memory_freq: 要禁用的显存频率
            reason: 禁用原因
        """
        if not self.memory_optimization_enabled:
            return
        
        self.globally_disabled_memory_frequencies.add(memory_freq)
        
        # 从可用显存频率列表中移除
        if memory_freq in self.available_memory_frequencies:
            self.available_memory_frequencies.remove(memory_freq)
        
        logger.warning(f"🚫 全局禁用显存频率 {memory_freq}MHz ({reason})")
        logger.info(f"   剩余可用显存频率: {sorted(self.available_memory_frequencies)}")
        
        # 重新生成学习动作列表
        if hasattr(self, '_learning_frequency_list') and not self.learning_phase_complete:
            self._regenerate_learning_action_list()
    
    def disable_core_memory_combination(self, core_freq: int, memory_freq: int, include_lower: bool = True):
        """
        禁用特定核心频率下的显存频率组合
        
        Args:
            core_freq: 核心频率
            memory_freq: 要禁用的显存频率边界
            include_lower: 是否包含更低的显存频率
        """
        if not self.memory_optimization_enabled:
            return
        
        if core_freq not in self.core_memory_disabled_combinations:
            self.core_memory_disabled_combinations[core_freq] = set()
        
        # 禁用指定的显存频率
        disabled_freqs = {memory_freq}
        
        # 如果包含更低频率，也禁用所有更低的显存频率
        if include_lower:
            for mem_freq in self.available_memory_frequencies:
                if mem_freq <= memory_freq:
                    disabled_freqs.add(mem_freq)
        
        self.core_memory_disabled_combinations[core_freq].update(disabled_freqs)
        
        logger.warning(f"🚫 禁用核心频率 {core_freq}MHz 下的显存频率组合: {sorted(disabled_freqs)}")
        
        # 重新生成学习动作列表
        if hasattr(self, '_learning_frequency_list') and not self.learning_phase_complete:
            self._regenerate_learning_action_list()
    
    def disable_core_frequency_for_memory_issues(self, core_freq: int, reason: str):
        """
        因显存频率问题禁用核心频率
        
        Args:
            core_freq: 要禁用的核心频率
            reason: 禁用原因
        """
        self.disabled_core_frequencies.add(core_freq)
        
        # 从可用核心频率列表中移除
        if core_freq in self.available_frequencies:
            self.available_frequencies.remove(core_freq)
        
        logger.warning(f"🚫 禁用核心频率 {core_freq}MHz ({reason})")
        logger.info(f"   剩余可用核心频率: {len(self.available_frequencies)} 个")
        
        # 重新生成学习动作列表
        if hasattr(self, '_learning_frequency_list') and not self.learning_phase_complete:
            self._regenerate_learning_action_list()
    
    def disable_core_frequencies_below_threshold(self, threshold_freq: int, reason: str):
        """
        禁用阈值频率及以下的所有核心频率
        
        Args:
            threshold_freq: 阈值频率
            reason: 禁用原因
        """
        disabled_freqs = []
        for freq in self.available_frequencies.copy():
            if freq <= threshold_freq:
                self.disabled_core_frequencies.add(freq)
                self.available_frequencies.remove(freq)
                disabled_freqs.append(freq)
        
        logger.warning(f"🚫 禁用核心频率 ≤{threshold_freq}MHz: {sorted(disabled_freqs)} ({reason})")
        logger.info(f"   剩余可用核心频率: {len(self.available_frequencies)} 个")
        
        # 重新生成学习动作列表
        if hasattr(self, '_learning_frequency_list') and not self.learning_phase_complete:
            self._regenerate_learning_action_list()
    
    def propagate_memory_slo_boundary(self, violating_core_freq: int, violating_memory_freq: int):
        """
        传播显存频率SLO边界到更小的核心频率
        
        当大核心频率+小显存频率组合违规时，传播约束到更小的核心频率：
        - 所有小于等于violating_core_freq的核心频率都不能使用小于等于violating_memory_freq的显存频率
        - 大于violating_core_freq的核心频率不受影响
        
        Args:
            violating_core_freq: 违规的核心频率
            violating_memory_freq: 违规的显存频率
        """
        if not self.memory_optimization_enabled:
            return
            
        # 计算需要设置边界的核心频率（小于等于违规核心频率）
        affected_core_freqs = [freq for freq in self.available_frequencies 
                              if freq <= violating_core_freq]
        
        # 对每个受影响的核心频率设置或更新SLO边界
        for core_freq in affected_core_freqs:
            # 获取当前边界
            current_boundary = self.core_specific_memory_slo_boundaries.get(core_freq, 0)
            
            # 设置更严格的边界（更高的最小显存频率）
            new_boundary = max(current_boundary, violating_memory_freq + 1)
            
            # 只有在边界确实改变时才设置
            if new_boundary > current_boundary:
                self.core_specific_memory_slo_boundaries[core_freq] = new_boundary
                
                # 禁用该核心频率下所有小于等于violating_memory_freq的显存频率
                self.disable_core_memory_combination(core_freq, violating_memory_freq, include_lower=True)
        
        logger.warning(f"🔄 SLO边界传播: {violating_core_freq}MHz+{violating_memory_freq}MHz违规")
        logger.info(f"   影响核心频率: {sorted(affected_core_freqs)} (设置显存频率边界 >{violating_memory_freq}MHz)")
        logger.info(f"   不影响核心频率: {sorted([f for f in self.available_frequencies if f > violating_core_freq])}")
        
        # 重新生成学习动作列表
        if hasattr(self, '_learning_frequency_list') and not self.learning_phase_complete:
            self._regenerate_learning_action_list()
    
    def is_action_allowed(self, action) -> bool:
        """
        检查动作是否被允许（未被禁用）
        
        Args:
            action: 动作 (核心频率 或 (核心频率, 显存频率) 元组)
            
        Returns:
            bool: 是否允许此动作
        """
        if not self.memory_optimization_enabled:
            # 仅核心频率模式
            return action not in self.disabled_core_frequencies
        
        if isinstance(action, tuple):
            core_freq, memory_freq = action
            
            # 检查核心频率是否被禁用
            if core_freq in self.disabled_core_frequencies:
                return False
            
            # 检查显存频率是否被全局禁用
            if memory_freq in self.globally_disabled_memory_frequencies:
                return False
            
            # 检查特定核心频率下的显存频率组合是否被禁用
            if (core_freq in self.core_memory_disabled_combinations and 
                memory_freq in self.core_memory_disabled_combinations[core_freq]):
                return False
            
            # 检查核心频率特定的SLO边界
            if core_freq in self.core_specific_memory_slo_boundaries:
                min_allowed_memory_freq = self.core_specific_memory_slo_boundaries[core_freq]
                if memory_freq < min_allowed_memory_freq:
                    return False
            
            return True
        else:
            # 单一核心频率动作
            return action not in self.disabled_core_frequencies
    
    def get_valid_actions(self) -> List:
        """
        获取所有有效（未被禁用）的动作列表
        
        Returns:
            List: 有效动作列表
        """
        if not self.memory_optimization_enabled:
            # 仅核心频率模式
            return [freq for freq in self.available_frequencies 
                   if freq not in self.disabled_core_frequencies]
        
        valid_actions = []
        for core_freq in self.available_frequencies:
            if core_freq in self.disabled_core_frequencies:
                continue
                
            for memory_freq in self.available_memory_frequencies:
                action = (core_freq, memory_freq)
                if self.is_action_allowed(action):
                    valid_actions.append(action)
        
        return valid_actions