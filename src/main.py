import time
import yaml
import signal
import sys
import argparse
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from collections import Counter
import pynvml

from .logger import setup_logger
from .metrics_collector import MetricsCollector
from .gpu_controller import GPUController
from .feature_extractor import FeatureExtractor
from .contextual_bandit import ContextualLinUCB
from .reward_calculator import EDPRewardCalculator
from .experiment_recorder import ExperimentRecorder, create_round_data_dict


def get_gpu_model_for_logging(gpu_id: int = 0) -> str:
    """获取GPU型号用于日志目录分类"""
    try:
        pynvml.nvmlInit()
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        gpu_name = pynvml.nvmlDeviceGetName(nvml_handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode('utf-8')
        
        # 清理GPU名称，移除特殊字符，用于目录名
        cleaned_name = gpu_name.replace("NVIDIA ", "").replace("GeForce ", "")
        cleaned_name = cleaned_name.replace(" ", "_").replace("-", "_")
        # 移除常见的后缀信息
        for suffix in ["_SXM2", "_32GB", "_16GB", "_8GB", "_PCIE", "_SMX2"]:
            if suffix in cleaned_name:
                cleaned_name = cleaned_name.split(suffix)[0]
                break
        
        return cleaned_name[:20]  # 限制长度避免路径过长
        
    except Exception as e:
        print(f"⚠️ 无法获取GPU型号信息: {e}")
        return "Unknown_GPU"


def create_gpu_classified_log_dir(config_path: str = "config/config.yaml") -> tuple[Path, str]:
    """根据配置文件中的GPU ID创建分类的日志目录"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        gpu_id = config.get('gpu', {}).get('device_id', 0)
    except Exception:
        gpu_id = 0
    
    gpu_model = get_gpu_model_for_logging(gpu_id)
    log_dir = Path("logs") / gpu_model
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"vllm_gpu_autoscaler_{timestamp}.log"
    
    return log_dir / log_file, gpu_model


class VLLMGPUAutoscaler:
    """vLLM GPU自动调频器 - 纯Contextual LinUCB版本"""
    
    def __init__(self, config_path: str = "config/config.yaml", reset_model: bool = False):
        """初始化自动调频器"""
        self.config_path = config_path
        self.reset_model = reset_model
        self.running = False
        
        # 加载配置
        self.config = self._load_config()
        
        # 检查数据收集模式
        self.data_collection_mode = self.config.get('control', {}).get('data_collection_mode', False)
        
        # 设置日志
        log_file_path, gpu_model = create_gpu_classified_log_dir(config_path)
        logging_config = self.config.get('logging', {})
        
        # 获取日志级别
        console_level_str = logging_config.get('console_level', 'INFO').upper()
        file_level_str = logging_config.get('file_level', 'DEBUG').upper()
        
        # 转换为logging级别
        import logging
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        console_level = level_map.get(console_level_str, logging.INFO)
        file_level = level_map.get(file_level_str, logging.DEBUG)
        
        self.logger = setup_logger("VLLMAutoscaler", str(log_file_path), 
                                  console_level=console_level, file_level=file_level)
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 vLLM GPU自动调频器启动")
        self.logger.info(f"📁 配置文件: {config_path}")
        self.logger.info(f"🔧 GPU型号: {gpu_model}")
        self.logger.info(f"📝 日志文件: {log_file_path}")
        self.logger.info(f"📊 日志级别: 控制台={console_level_str}, 文件={file_level_str}")
        self.logger.info(f"🔍 详细轮次记录: {'启用' if logging_config.get('detailed_round_logging', True) else '禁用'}")
        self.logger.info(f"📊 运行模式: {'数据收集模式 (不调频)' if self.data_collection_mode else '正常学习模式'}")
        self.logger.info("=" * 80)
        
        # 初始化核心组件
        self._init_components()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("✅ vLLM GPU自动调频器初始化完成")
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            sys.exit(1)
    
    def _init_components(self):
        """初始化系统组件"""
        # 初始化指标收集器
        prometheus_url = self.config['vllm']['prometheus_url']
        metrics_config = self.config.get('metrics', {})
        # 只传递MetricsCollector需要的参数
        metrics_collector_args = {
            'ema_alpha': metrics_config.get('ema_alpha', 0.4),
            'sampling_duration': metrics_config.get('sampling_duration', 0.8),
            'sampling_interval': metrics_config.get('sampling_interval', 0.01)
        }
        self.metrics_collector = MetricsCollector(prometheus_url, **metrics_collector_args)
        
        # 初始化GPU控制器
        gpu_config = self.config.get('gpu', {})
        # 映射参数名称以匹配GPUController期望的参数
        gpu_controller_args = {}
        if 'device_id' in gpu_config:
            gpu_controller_args['gpu_id'] = gpu_config['device_id']
        if 'frequency_step' in gpu_config:
            gpu_controller_args['step'] = gpu_config['frequency_step']
        if 'min_frequency' in gpu_config:
            gpu_controller_args['min_freq'] = gpu_config['min_frequency']
        # 直接映射的参数 (移除max_action_count限制)
        for key in ['auto_step', 'enable_memory_frequency_control', 
                   'memory_auto_detect', 'memory_frequencies']:
            if key in gpu_config:
                gpu_controller_args[key] = gpu_config[key]
        # 从control配置中获取ignore_slo参数和成熟度阈值
        control_config = self.config.get('control', {})
        if 'ignore_slo' in control_config:
            gpu_controller_args['ignore_slo'] = control_config['ignore_slo']
        if 'adaptive_update_interval' in control_config:
            gpu_controller_args['adaptive_update_interval'] = control_config['adaptive_update_interval']
        if 'learner_maturity_threshold' in control_config:
            gpu_controller_args['learner_maturity_threshold'] = control_config['learner_maturity_threshold']
        if 'refinement_start_threshold' in control_config:
            gpu_controller_args['refinement_start_threshold'] = control_config['refinement_start_threshold']
        # 从adaptive_sampling配置中获取reward_threshold参数
        adaptive_config = self.config.get('adaptive_sampling', {})
        if 'reward_threshold' in adaptive_config:
            gpu_controller_args['reward_threshold'] = adaptive_config['reward_threshold']
        
        # 添加实际频率回调函数
        gpu_controller_args['actual_frequency_callback'] = self._on_actual_frequency_available
        
        self.gpu_controller = GPUController(**gpu_controller_args)
        
        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor()
        
        # 初始化奖励计算器
        control_config = self.config.get('control', {})
        # 获取metrics配置用于baseline
        metrics_config = self.config.get('metrics', {})
        sampling_duration = metrics_config.get('sampling_duration', 0.8)
        
        self.reward_calculator = EDPRewardCalculator(
            ttft_limit=control_config.get('ttft_limit', 2.0),
            tpot_limit=control_config.get('tpot_limit', 0.25),
            ignore_slo=control_config.get('ignore_slo', True),
            sampling_duration=sampling_duration,
            baseline_measurements=metrics_config.get('baseline_measurements', 3)
        )
        
        # 基线状态由reward_calculator管理
        
        # 初始化Contextual LinUCB模型 (不再支持神经网络)
        linucb_config = self.config.get('linucb', {})
        self.model = ContextualLinUCB(
            n_features=7,  # 仅工作负载特征，不包含频率
            alpha=linucb_config.get('initial_alpha', 0.1),  # 大幅降低alpha减少过度探索
            lambda_reg=linucb_config.get('lambda_reg', 5.0),  # 保持正则化提高数值稳定性
            alpha_decay_rate=linucb_config.get('alpha_decay_rate', 0.01),  # alpha衰减率
            min_alpha=linucb_config.get('min_alpha', 0.1),  # 最小alpha值
            # 智能动作修剪参数
            enable_action_pruning=linucb_config.get('enable_action_pruning', True),
            pruning_check_interval=linucb_config.get('pruning_check_interval', 100),
            pruning_threshold=linucb_config.get('pruning_threshold', 1.0),
            min_exploration_for_pruning=linucb_config.get('min_exploration_for_pruning', 5),
            pruning_maturity_threshold=linucb_config.get('pruning_maturity_threshold', 100),
            cascade_pruning_threshold=linucb_config.get('cascade_pruning_threshold', 600),
            gpu_max_freq=self.gpu_controller.max_freq,  # 传入GPU硬件最大频率
            # 极端频率即时修剪参数
            enable_extreme_pruning=linucb_config.get('enable_extreme_pruning', True),
            extreme_pruning_threshold=linucb_config.get('extreme_pruning_threshold', -1.5),
            extreme_pruning_min_samples=linucb_config.get('extreme_pruning_min_samples', 3),
            extreme_pruning_max_rounds=linucb_config.get('extreme_pruning_max_rounds', 20),
            auto_load=not self.reset_model
        )
        
        # 启用显存+核心频率组合优化（如果支持）
        if (self.gpu_controller.enable_memory_frequency_control and 
            self.gpu_controller.memory_frequency_supported and 
            hasattr(self.gpu_controller, 'memory_frequencies')):
            
            memory_freqs = self.gpu_controller.memory_frequencies
            self.model.enable_memory_frequency_optimization(memory_freqs)
            
            # 设置LinUCB模型回调，用于精细的频率禁用管理
            self.gpu_controller.set_linucb_callback(self.model)
            
            self.logger.info(f"🔧 显存+核心频率组合优化已启用")
            self.logger.info(f"   显存频率: {memory_freqs}")
            total_combinations = len(self.gpu_controller.frequencies) * len(memory_freqs)
            self.logger.info(f"   总动作空间: {total_combinations} 个组合")
            self.logger.info(f"   精细禁用管理: 已启用")
        else:
            self.logger.info(f"🔧 仅核心频率优化模式")
        
        # 控制参数
        self.convergence_window = control_config.get('convergence_window', 100)
        self.performance_degradation_threshold = control_config.get('performance_degradation_threshold', 1.2)
        self.convergence_top_k = control_config.get('convergence_top_k', 3)
        self.convergence_threshold = control_config.get('convergence_threshold', 0.70)
        
        # 休息模式状态
        self.idle_mode = False              # 当前是否处于休息模式
        self.idle_confirmation_count = 0    # 连续空闲检测计数
        self.last_idle_reset = False        # 是否已经执行过频率重置
        
        self.logger.info("🔧 系统组件初始化完成:")
        self.logger.info(f"   算法: Contextual LinUCB (频率作为动作)")
        self.logger.info(f"   特征维度: 7维 (仅工作负载特征)")
        self.logger.info(f"   收敛窗口: {self.convergence_window}轮")
        self.logger.info(f"   收敛策略: 前{self.convergence_top_k}个动作联合占比 >= {self.convergence_threshold:.0%}")
        
        # 显示当前模型状态
        initial_mode = "利用(已收敛)" if self.model.exploitation_mode else "学习(探索中)"
        self.logger.info(f"   初始模式: {initial_mode}")
        if hasattr(self.model, 'total_rounds') and self.model.total_rounds > 0:
            self.logger.info(f"   已训练轮次: {self.model.total_rounds}")
            self.logger.info(f"   已知频率: {len(self.model.available_frequencies)}个")
        
        # 初始化实验数据记录器
        self.experiment_recorder = ExperimentRecorder()
        self.experiment_recorder.save_config_snapshot(self.config_path)
        
        # 获取GPU型号信息并保存到实验记录器
        gpu_model = get_gpu_model_for_logging(gpu_config.get('device_id', 0))
        self.experiment_recorder.set_gpu_model(gpu_model)
    
    def _on_actual_frequency_available(self, actual_freq: int):
        """
        实际频率可用回调函数
        当GPU频率设置失败但有实际频率可用时，将其添加到动作空间
        
        Args:
            actual_freq: 实际设置成功的频率（MHz）
        """
        try:
            # 检查配置是否启用了自适应恢复
            control_config = self.config.get('control', {})
            if not control_config.get('auto_add_actual_frequency', True):
                self.logger.debug(f"🔄 自动添加实际频率功能已禁用，忽略 {actual_freq}MHz")
                return
            
            # 检查是否有模型存在
            if not hasattr(self, 'model') or self.model is None:
                self.logger.warning(f"⚠️ 模型不存在，无法添加实际频率 {actual_freq}MHz")
                return
            
            # 检查动作空间大小，只有当空间很小时才添加
            min_action_size = control_config.get('min_action_space_size', 1)
            current_action_count = len(self.model.available_frequencies)
            
            if current_action_count > min_action_size:
                self.logger.debug(f"🔄 动作空间足够大({current_action_count}>{min_action_size})，不添加实际频率 {actual_freq}MHz")
                return
            
            # 添加实际频率到模型的动作空间
            self.model.add_actual_frequency(actual_freq)
            
            self.logger.info(f"🚀 动作空间自适应恢复: 成功添加实际频率 {actual_freq}MHz")
            
        except Exception as e:
            self.logger.error(f"❌ 添加实际频率失败: {e}")
            import traceback
            self.logger.debug(f"错误详情: {traceback.format_exc()}")
    
    def _signal_handler(self, signum, _frame):
        """信号处理器"""
        self.logger.info(f"🛑 接收到信号 {signum}，开始优雅关闭...")
        self.running = False
        
        # 结束实验并保存数据
        if hasattr(self, 'experiment_recorder'):
            self.experiment_recorder.finalize_experiment()
    
    def _get_current_state(self) -> tuple[np.ndarray, list, dict]:
        """获取当前系统状态"""
        # 收集指标（包含能耗数据）
        metrics = self.metrics_collector.collect_metrics(
            energy_reader=self.gpu_controller.read_energy_j
        )
        if not metrics:
            raise RuntimeError("无法收集vLLM指标")
        
        # 提取特征 (仅工作负载特征，不包含频率)
        features = self.feature_extractor.extract_features(metrics)
        if features is None or len(features) != 7:
            raise RuntimeError(f"特征提取失败，期望7维，实际{len(features) if features is not None else 0}维")
        
        # 获取可用频率（排除修剪和失败的频率）
        available_frequencies = self.gpu_controller.get_available_frequencies(self.model)
        if not available_frequencies:
            self.logger.warning("⚠️ 无可用GPU频率，重置GPU频率到默认状态")
            # 重置GPU频率到默认状态，不再锁频
            if self.gpu_controller.reset_gpu_clocks():
                self.logger.info("✅ GPU频率已重置，系统继续运行（不锁频模式）")
                # 返回一个空的频率列表，表示不进行频率决策
                return features, [], metrics
            else:
                raise RuntimeError("GPU频率重置失败，无法继续运行")
        
        return features, available_frequencies, metrics
    
    def _make_decision(self, features: np.ndarray, available_frequencies: list) -> int:
        """做出频率决策"""
        # 如果没有可用频率，返回None表示不进行频率决策
        if not available_frequencies:
            self.logger.debug("📊 无可用频率，跳过频率决策（保持重置状态）")
            return None
            
        # 更新模型的动作空间
        self.model.update_action_space(available_frequencies)
        
        # 选择频率 (频率作为动作，不作为特征)
        if self.model.exploitation_mode:
            selected_freq = self.model.select_action_exploitation(features, available_frequencies)
        else:
            selected_freq = self.model.select_action(features, available_frequencies)
        
        return selected_freq
    
    def _execute_action_and_measure(self, action, is_baseline: bool = False) -> tuple[float, dict, Optional[float]]:
        """执行动作并测量奖励（支持核心频率或组合频率）"""
        # 如果action为None，表示系统处于重置状态，不设置频率
        if action is None:
            self.logger.debug("📊 频率已重置，跳过频率设置，直接测量性能")
            # 等待决策间隔
            time.sleep(0.3)
            
            # 收集执行后的指标
            post_metrics = self.metrics_collector.collect_metrics(
                energy_reader=self.gpu_controller.read_energy_j
            )
            if not post_metrics:
                self.logger.warning("⚠️ post_metrics为空，跳过自适应采样器更新")
                return 0.0, {}, None
                
            # 计算奖励（使用当前的实际频率）
            current_freq = self.gpu_controller._get_current_frequency()
            self.logger.info(f"📊 重置模式下当前GPU频率: {current_freq}MHz")
            
            # 获取能耗增量
            energy_delta = post_metrics.get('energy_delta_j', 0.0)
            reward = self.reward_calculator.calculate_reward(post_metrics, energy_delta)
            return reward, post_metrics, None
            
        # 设置GPU频率（组合频率或核心频率）
        if self.model.memory_optimization_enabled and isinstance(action, tuple):
            # 组合频率模式：设置核心频率和显存频率
            core_freq, memory_freq = action
            success = self.gpu_controller.set_frequency(core_freq)
            if success and hasattr(self.gpu_controller, 'set_memory_frequency'):
                success = success and self.gpu_controller.set_memory_frequency(memory_freq)
            
            if not success:
                action_desc = f"{core_freq}MHz核心+{memory_freq}MHz显存"
                self.logger.error(f"❌ 设置组合频率 {action_desc} 失败")
                return -1.0, {}, None  # 惩罚失败的频率设置
        else:
            # 核心频率模式：只设置核心频率
            if not self.gpu_controller.set_frequency(action):
                self.logger.error(f"❌ 设置GPU核心频率 {action}MHz 失败")
                return -1.0, {}, None  # 惩罚失败的频率设置
        
        # 等待决策间隔
        time.sleep(0.3)
        
        # 收集执行后的指标（包含能耗数据）
        post_metrics = self.metrics_collector.collect_metrics(
            energy_reader=self.gpu_controller.read_energy_j
        )
        if not post_metrics:
            self.logger.warning("⚠️ 无法收集执行后指标")
            return -0.5, {}, None
        
        # 从指标中获取能耗增量（焦耳）
        energy_delta = post_metrics.get('energy_delta_j', 0.0)
        
        # 计算奖励和EDP值
        # 从vLLM指标中提取计数器增量
        counter_deltas = {}
        ttft_sum_key = "vllm:time_to_first_token_seconds_sum_delta"
        ttft_count_key = "vllm:time_to_first_token_seconds_count_delta"
        tpot_sum_key = "vllm:time_per_output_token_seconds_sum_delta"
        tpot_count_key = "vllm:time_per_output_token_seconds_count_delta"
        
        for key in [ttft_sum_key, ttft_count_key, tpot_sum_key, tpot_count_key]:
            if key in post_metrics:
                counter_deltas[key] = post_metrics[key]
        
        # 调用calculate方法获取详细信息
        reward, info = self.reward_calculator.calculate(
            counter_deltas=counter_deltas,
            energy_consumed_j=energy_delta,
            is_baseline_collection=is_baseline
        )
        
        # 提取EDP值
        edp_value = info.get('edp_raw', None)
        
        return reward, post_metrics, edp_value
    
    def _collect_baseline_metrics(self) -> tuple[float, dict, Optional[float]]:
        """专门用于基线收集的指标收集方法 - 不改变GPU频率，但检查任务状态"""
        try:
            # 等待系统稳定
            time.sleep(0.3)
            
            # 收集指标（包含能耗数据）
            post_metrics = self.metrics_collector.collect_metrics(
                energy_reader=self.gpu_controller.read_energy_j
            )
            if not post_metrics:
                self.logger.warning("⚠️ 基线收集时无法收集指标")
                return 0.0, {}, None
            
            # 基线收集期间检查任务状态 - 如果没有任务，跳过这次收集
            running_requests = post_metrics.get('vllm:num_requests_running', 0)
            has_queue = post_metrics.get('vllm:num_requests_waiting', 0) > 0
            
            if running_requests == 0 and not has_queue:
                self.logger.info("⏸️ 基线收集期间检测到无任务，跳过此次收集")
                return 0.0, {}, None
            
            # 计算奖励（基线收集模式）
            reward, info = self.reward_calculator.calculate(
                counter_deltas=post_metrics,
                energy_consumed_j=post_metrics.get('energy_delta_j', 0.0),
                is_baseline_collection=True
            )
            
            # 提取EDP值用于记录
            edp_value = info.get('edp_raw', None)
            
            return reward, post_metrics, edp_value
            
        except Exception as e:
            self.logger.error(f"❌ 基线指标收集失败: {e}")
            return 0.0, {}, None
    
    def _check_convergence(self) -> bool:
        """检查模型是否收敛 - 基于前几个最优动作的联合稳定性"""
        if self.model.total_rounds < self.convergence_window:
            return False
        
        # 获取最近的动作历史
        recent_actions = self.model.action_history[-self.convergence_window:]
        if len(recent_actions) < self.convergence_window:
            return False
        
        # 统计每个动作的选择次数
        action_counts = Counter(recent_actions)
        
        # 从配置中获取收敛参数
        control_config = self.config.get('control', {})
        top_k = control_config.get('convergence_top_k', 3)  # 考虑前k个最优动作
        convergence_threshold = control_config.get('convergence_threshold', 0.70)  # 收敛阈值
        
        # 获取被选择次数最多的前k个动作
        most_common_actions = action_counts.most_common(top_k)
        
        if not most_common_actions:
            return False
        
        # 计算前k个动作的总占比
        top_k_total_count = sum(count for _, count in most_common_actions)
        top_k_combined_ratio = top_k_total_count / len(recent_actions)
        is_converged = top_k_combined_ratio >= convergence_threshold
        
        # 记录收敛状态变化
        if is_converged and not self.model.exploitation_mode:
            # 构建收敛信息字符串
            top_actions_info = ", ".join([f"{action}MHz({count}次,{count/len(recent_actions):.1%})" 
                                        for action, count in most_common_actions])
            
            self.logger.info(f"🎯 模型收敛检测: 前{len(most_common_actions)}个优势动作联合占比{top_k_combined_ratio:.1%} >= {convergence_threshold:.0%}")
            self.logger.info(f"📊 稳定状态集: {top_actions_info}")
            self.logger.info(f"🔄 从学习模式切换到利用模式")
            self.model.set_exploitation_mode(True)
            self.model.is_converged = True
            self.model.save_model("final_contextual_model.pkl")
            
            # 记录收敛时的统计信息
            self.logger.info(f"📊 收敛统计: 总轮次={self.model.total_rounds}, 动作空间={len(self.model.available_frequencies)}个频率")
            
        elif not is_converged and self.model.exploitation_mode:
            # 如果之前收敛但现在不稳定了，可能需要重新学习
            self.logger.warning(f"⚠️ 收敛状态不稳定: 前{top_k}个动作联合占比降至{top_k_combined_ratio:.1%} < {convergence_threshold:.0%}")
        
        return is_converged
    
    def _check_performance_degradation(self) -> bool:
        """检查性能是否退化 - 基于原始EDP值"""
        if self.model.total_rounds < 100:
            return False
        
        # 使用EDP历史而不是奖励历史
        edp_history = self.model.edp_history
        if len(edp_history) < 100:
            return False
        
        recent_50_edp = edp_history[-50:]
        previous_50_edp = edp_history[-100:-50]
        
        if not previous_50_edp or not recent_50_edp:
            return False
        
        recent_avg_edp = np.mean(recent_50_edp)
        previous_avg_edp = np.mean(previous_50_edp)
        
        # EDP越小越好，所以如果最近的平均EDP比之前的高出配置阈值，就认为性能退化
        if previous_avg_edp > 0:
            edp_ratio = recent_avg_edp / previous_avg_edp
            # 修正阈值逻辑：只有当EDP恶化超过配置的百分比时才触发
            degradation_threshold = 1.0 + self.performance_degradation_threshold  # 0.65 -> 1.65
            if edp_ratio > degradation_threshold:
                self.logger.warning(f"⚠️ 性能退化检测: 最近50轮EDP平均值({recent_avg_edp:.4f}) "
                                  f"比前50轮({previous_avg_edp:.4f})恶化了{(edp_ratio-1)*100:.1f}% > {self.performance_degradation_threshold*100:.0f}%")
                self.logger.warning(f"🔄 性能退化触发条件: EDP恶化比例 {edp_ratio:.3f} > 阈值 {degradation_threshold:.3f}")
                return True
            else:
                # 记录轻微恶化但未触发的情况
                if edp_ratio > 1.05:  # 恶化超过5%时记录info级别日志
                    self.logger.info(f"📊 轻微性能变化: EDP恶化{(edp_ratio-1)*100:.1f}% (未达到{self.performance_degradation_threshold*100:.0f}%触发阈值)")
        
        return False
    
    def _check_idle_mode(self, metrics: dict) -> bool:
        """检查是否应该进入/退出休息模式 - 基于正在运行的请求数"""
        # 获取正在运行的请求数
        running_requests = metrics.get('vllm:num_requests_running', 0)
        
        # 空闲检测：没有正在运行的请求
        is_currently_idle = (running_requests == 0)
        
        if is_currently_idle:
            self.idle_confirmation_count += 1
            
            # 连续3次检测到空闲才进入休息模式（避免短暂空闲的误判）
            if not self.idle_mode and self.idle_confirmation_count >= 3:
                self.idle_mode = True
                self.last_idle_reset = False  # 重置频率重置标志
                self.logger.info("😴 检测到持续空闲（无运行任务），进入休息模式")
                return True
        else:
            # 有运行任务时重置计数并退出休息模式
            if self.idle_mode:
                self.idle_mode = False
                self.idle_confirmation_count = 0
                self.logger.info(f"🏃 检测到运行任务({running_requests}个)，退出休息模式，恢复学习")
            else:
                self.idle_confirmation_count = 0
        
        return self.idle_mode
    
    def _handle_idle_mode(self):
        """处理休息模式：重置频率到安全值（只执行一次）"""
        if self.idle_mode and not self.last_idle_reset:
            # 重置GPU频率到安全的空闲频率（210MHz）
            idle_frequency = 210  # MHz，安全的空闲频率
            if self.gpu_controller.set_frequency(idle_frequency):
                self.logger.info(f"🔧 休息模式：GPU频率已重置为 {idle_frequency}MHz")
                self.last_idle_reset = True
            else:
                self.logger.warning(f"⚠️ 休息模式：GPU频率重置失败")
    
    def _handle_data_collection_mode(self, features: np.ndarray, pre_metrics: dict, post_metrics: dict):
        """处理数据收集模式：仅收集数据不调频，但收集完整的能量和EDP数据"""
        # 获取当前GPU频率
        current_freq = self.gpu_controller.current_freq
        
        # 记录数据收集轮次
        data_round = getattr(self, '_data_collection_round', 0) + 1
        setattr(self, '_data_collection_round', data_round)
        
        self.logger.info(f"📊 数据收集模式 - 轮次 {data_round}: GPU频率 {current_freq}MHz (无调频)")
        
        # 计算EDP数据（与学习模式相同的计算方式）
        edp_value = 0.0
        energy_j = 0.0
        
        try:
            # 计算能量消耗（直接使用焦耳）
            energy_j = post_metrics.get('energy_delta_j', 0.0) or 0.0
            
            # 计算延迟指标
            ttft_avg = 0.0
            tpot_avg = 0.0
            
            if post_metrics.get('vllm:time_to_first_token_seconds_count_delta', 0) > 0:
                ttft_avg = (post_metrics.get('vllm:time_to_first_token_seconds_sum_delta', 0.0) / 
                           post_metrics.get('vllm:time_to_first_token_seconds_count_delta', 1))
            
            if post_metrics.get('vllm:time_per_output_token_seconds_count_delta', 0) > 0:
                tpot_avg = (post_metrics.get('vllm:time_per_output_token_seconds_sum_delta', 0.0) / 
                           post_metrics.get('vllm:time_per_output_token_seconds_count_delta', 1))
            
            # 计算EDP (Energy-Delay Product)
            if energy_j > 0 and (ttft_avg > 0 or tpot_avg > 0):
                total_latency = ttft_avg + tpot_avg
                edp_value = energy_j * total_latency
                
        except Exception as e:
            self.logger.warning(f"⚠️ 数据收集模式计算EDP失败: {e}")
        
        # 记录完整数据到实验文件
        if hasattr(self, 'experiment_recorder'):
            try:
                # 使用与学习模式相同的数据记录方式
                self._record_experiment_data(
                    round_num=data_round,
                    features=features,
                    selected_action=current_freq,
                    reward=0.0,  # 数据收集模式无奖励
                    edp_value=edp_value,
                    pre_metrics=pre_metrics,
                    post_metrics=post_metrics
                )
            except Exception as e:
                self.logger.error(f"❌ 数据收集模式记录实验数据失败: {e}")
        
        print(f"📊 数据收集轮次: {data_round} (频率: {current_freq}MHz, EDP: {edp_value:.6f}J·s, 能量: {energy_j:.6f}J)")
    
    def _log_round_details(self, round_num: int, features: np.ndarray, action, 
                          reward: float, edp_value: Optional[float], 
                          _pre_metrics: dict, post_metrics: dict):
        """记录每一轮的详细信息到日志文件"""
        
        # 检查是否启用详细日志
        logging_config = self.config.get('logging', {})
        if not logging_config.get('detailed_round_logging', True):
            return
        
        # 提取关键指标
        energy_delta = post_metrics.get('energy_delta_j', 0.0)
        
        # vLLM指标
        ttft_count = post_metrics.get('vllm:time_to_first_token_seconds_count_delta', 0)
        ttft_sum = post_metrics.get('vllm:time_to_first_token_seconds_sum_delta', 0.0)
        tpot_count = post_metrics.get('vllm:time_per_output_token_seconds_count_delta', 0)
        tpot_sum = post_metrics.get('vllm:time_per_output_token_seconds_sum_delta', 0.0)
        
        # 计算平均值
        avg_ttft = ttft_sum / ttft_count if ttft_count > 0 else 0.0
        avg_tpot = tpot_sum / tpot_count if tpot_count > 0 else 0.0
        
        # GPU使用率和内存等其他指标
        gpu_util = post_metrics.get('gpu_utilization', 0.0)
        gpu_memory = post_metrics.get('gpu_memory_used', 0.0)
        queue_size = post_metrics.get('vllm:num_requests_waiting', 0)
        
        # 模型状态
        model_stats = self.model.get_model_stats()
        
        # 获取动作选择统计（向后兼容）
        available_freqs = self.gpu_controller.get_available_frequencies(self.model)
        action_selection_count = getattr(self.model, 'arm_counts', {}).get(action, 0)
        
        # 动作描述（向后兼容）
        if self.model.memory_optimization_enabled and isinstance(action, tuple):
            action_desc = f"{action[0]}MHz核心+{action[1]}MHz显存"
            freq_for_compat = action[0]  # 用于兼容性的核心频率
        else:
            action_desc = f"{action}MHz核心"
            freq_for_compat = action
        
        # 获取正确的特征名称
        feature_names = self.feature_extractor.feature_names
        
        # 记录完整信息
        self.logger.info(
            f"🔍 轮次 {round_num} 详细记录:\n"
            f"📊 决策信息:\n"
            f"   选择动作: {action_desc} (选择次数: {action_selection_count})\n"
            f"   可用频率: {len(available_freqs)}个 [{min(available_freqs)}-{max(available_freqs)}MHz]\n"
            f"   模式: {'利用' if self.model.exploitation_mode else '学习'}\n"
            f"🎯 特征向量 (7维):\n"
            f"   [0]{feature_names[0]}: {features[0]:.3f}\n"
            f"   [1]{feature_names[1]}: {features[1]:.3f}\n"
            f"   [2]{feature_names[2]}: {features[2]:.3f}\n"
            f"   [3]{feature_names[3]}: {features[3]:.3f}\n"
            f"   [4]{feature_names[4]}: {features[4]:.3f}\n"
            f"   [5]{feature_names[5]}: {features[5]:.3f}\n"
            f"   [6]{feature_names[6]}: {features[6]:.3f}\n"
            f"⚡ 性能指标:\n"
            f"   TTFT: {avg_ttft:.4f}s (计数: {ttft_count})\n"
            f"   TPOT: {avg_tpot:.4f}s (计数: {tpot_count})\n"
            f"   能耗增量: {energy_delta:.2f}mJ ({energy_delta/1000:.5f}J)\n"
            f"   EDP值: {f'{edp_value:.6f}' if edp_value is not None else 'N/A'}\n"
            f"💰 奖励信息:\n"
            f"   当前奖励: {reward:+.4f}\n"
            f"   平均奖励: {model_stats.get('avg_reward', 0.0):.4f}\n"
            f"   最近平均: {model_stats.get('recent_avg_reward', 0.0):.4f}\n"
            f"🖥️ 系统状态:\n"
            f"   GPU利用率: {gpu_util:.1f}%\n"
            f"   GPU内存: {gpu_memory:.1f}MB\n"
            f"   等待队列: {queue_size}个请求\n"
            f"📈 模型统计:\n"
            f"   总轮次: {model_stats.get('total_rounds', 0)}\n"
            f"   动作空间: {model_stats.get('n_arms', 0)}个频率\n"
            f"   探索/利用: {'收敛' if model_stats.get('converged', False) else '学习中'}\n"
            f"{'='*80}"
        )
        
        # 同时记录JSON格式的结构化数据，便于后续分析
        round_data = {
            'timestamp': datetime.now().isoformat(),
            'round': round_num,
            'decision': {
                'selected_action': action_desc,
                'selected_frequency_compat': freq_for_compat,  # 向后兼容
                'selection_count': action_selection_count,
                'available_frequencies': available_freqs,
                'mode': 'exploitation' if self.model.exploitation_mode else 'exploration'
            },
            'features': {
                feature_names[0]: float(features[0]),
                feature_names[1]: float(features[1]),
                feature_names[2]: float(features[2]),
                feature_names[3]: float(features[3]),
                feature_names[4]: float(features[4]),
                feature_names[5]: float(features[5]),
                feature_names[6]: float(features[6])
            },
            'performance': {
                'avg_ttft': avg_ttft,
                'avg_tpot': avg_tpot,
                'ttft_count': ttft_count,
                'tpot_count': tpot_count,
                'energy_delta_j': energy_delta,
                'energy_delta_mj': energy_delta * 1000,
                'edp_value': edp_value
            },
            'reward': {
                'current': reward,
                'average': model_stats.get('avg_reward', 0.0),
                'recent_average': model_stats.get('recent_avg_reward', 0.0)
            },
            'system': {
                'gpu_utilization': gpu_util,
                'gpu_memory_mb': gpu_memory,
                'queue_size': queue_size
            },
            'model': {
                'total_rounds': model_stats.get('total_rounds', 0),
                'n_arms': model_stats.get('n_arms', 0),
                'converged': model_stats.get('converged', False),
                'exploitation_mode': self.model.exploitation_mode
            }
        }
        
        # 记录JSON格式数据
        self.logger.info(f"📋 JSON数据: {json.dumps(round_data, ensure_ascii=False)}")
    
    def _record_experiment_data(self, round_num: int, features: np.ndarray, selected_action, 
                              reward: float, edp_value: float, pre_metrics: dict, post_metrics: dict):
        """记录实验数据到文件"""
        try:
            # 获取必要的数据
            feature_names = self.feature_extractor.feature_names
            
            # 处理动作信息（支持组合频率）
            if isinstance(selected_action, tuple):
                gpu_frequency = selected_action[0]  # 只记录核心频率
            else:
                gpu_frequency = selected_action
            
            # 计算延迟指标（使用delta版本）
            ttft_sum_key = 'vllm:time_to_first_token_seconds_sum_delta'
            ttft_count_key = 'vllm:time_to_first_token_seconds_count_delta'
            tpot_sum_key = 'vllm:time_per_output_token_seconds_sum_delta'
            tpot_count_key = 'vllm:time_per_output_token_seconds_count_delta'
            e2e_sum_key = 'vllm:e2e_request_latency_seconds_sum_delta'
            e2e_count_key = 'vllm:e2e_request_latency_seconds_count_delta'
            
            # 计算平均延迟
            ttft_avg = 0.0
            tpot_avg = 0.0
            e2e_avg = 0.0
            
            if (ttft_sum_key in post_metrics and ttft_count_key in post_metrics and 
                post_metrics[ttft_count_key] and post_metrics[ttft_count_key] > 0):
                ttft_avg = post_metrics[ttft_sum_key] / post_metrics[ttft_count_key]
            
            if (tpot_sum_key in post_metrics and tpot_count_key in post_metrics and 
                post_metrics[tpot_count_key] and post_metrics[tpot_count_key] > 0):
                tpot_avg = post_metrics[tpot_sum_key] / post_metrics[tpot_count_key]
                
            if (e2e_sum_key in post_metrics and e2e_count_key in post_metrics and 
                post_metrics[e2e_count_key] and post_metrics[e2e_count_key] > 0):
                e2e_avg = post_metrics[e2e_sum_key] / post_metrics[e2e_count_key]
            
            # 计算吞吐量（使用delta版本或计算差值）
            # 优先使用delta版本，确保不是None
            prompt_delta = post_metrics.get('vllm:prompt_tokens_total_delta', 0) or 0
            generation_delta = post_metrics.get('vllm:generation_tokens_total_delta', 0) or 0
            
            # 如果没有delta版本，手动计算
            if prompt_delta == 0 and generation_delta == 0:
                prompt_tokens_current = post_metrics.get('vllm:prompt_tokens_total', 0) or 0
                generation_tokens_current = post_metrics.get('vllm:generation_tokens_total', 0) or 0
                prompt_tokens_previous = pre_metrics.get('vllm:prompt_tokens_total', 0) or 0
                generation_tokens_previous = pre_metrics.get('vllm:generation_tokens_total', 0) or 0
                
                prompt_delta = prompt_tokens_current - prompt_tokens_previous
                generation_delta = generation_tokens_current - generation_tokens_previous
            
            # 计算采样期间的时间差（使用配置的采样窗口）
            metrics_config = self.config.get('metrics', {})
            sampling_duration = metrics_config.get('sampling_duration', 1.5) or 1.5  # 确保不是None
            
            total_delta = prompt_delta + generation_delta
            
            prefill_throughput = prompt_delta / sampling_duration if sampling_duration and sampling_duration > 0 else 0
            decode_throughput = generation_delta / sampling_duration if sampling_duration and sampling_duration > 0 else 0
            total_throughput = total_delta / sampling_duration if sampling_duration and sampling_duration > 0 else 0
            
            # 获取能耗（已经是焦耳）
            energy_j = post_metrics.get('energy_delta_j', 0.0) or 0.0
            
            # 计算EDP信息
            edp_raw = energy_j * tpot_avg if tpot_avg and tpot_avg > 0 else 0  # 使用TPOT而不是E2E
            edp_normalized = reward  # reward就是归一化的EDP
            edp_baseline = getattr(self.reward_calculator, 'baseline_edp', 0) or 0
            
            # 获取系统状态（确保None安全）
            running_requests = post_metrics.get('vllm:num_requests_running', 0) or 0
            waiting_requests = post_metrics.get('vllm:num_requests_waiting', 0) or 0
            active_requests = running_requests + waiting_requests
            cache_usage = (post_metrics.get('vllm:gpu_cache_usage_perc', 0) or 0) * 100  # 转换为百分比
            
            current_success = post_metrics.get('vllm:request_success_total', 0) or 0
            previous_success = pre_metrics.get('vllm:request_success_total', 0) or 0
            completed_requests = current_success - previous_success
            
            # 获取学习算法状态
            if self.data_collection_mode:
                learning_phase = 'DATA_COLLECTION'
                alpha_value = 0.0
            else:
                learning_phase = 'EXPLOITATION' if self.model.exploitation_mode else 'EXPLORATION'
                alpha_value = getattr(self.model, 'alpha', 0)
            
            # 获取UCB置信度（如果有的话）
            ucb_confidence = 0.0
            if not self.data_collection_mode and hasattr(self.model, '_last_ucb_values') and self.model._last_ucb_values:
                action_key = selected_action if not isinstance(selected_action, tuple) else selected_action[0]
                ucb_confidence = self.model._last_ucb_values.get(action_key, 0.0)
            
            # 获取动作选择方法
            if self.data_collection_mode:
                action_method = 'DATA_COLLECTION'
            else:
                action_method = 'GREEDY' if self.model.exploitation_mode else 'UCB'
            
            # 获取频率管理信息
            if self.data_collection_mode:
                available_freqs = 0
                pruned_freqs = 0
            else:
                available_freqs = len(getattr(self.model, 'available_frequencies', []))
                pruned_freqs = len(getattr(self.model, 'pruned_frequencies', set()))
            
            # 获取当前频率的探索次数
            freq_exploration_count = 0
            if not self.data_collection_mode and hasattr(self.model, 'action_counts'):
                freq_exploration_count = self.model.action_counts.get(selected_action, 0)
            
            # 检查SLO违规
            control_config = self.config.get('control', {})
            ttft_limit = control_config.get('ttft_limit', 2.0)
            tpot_limit = control_config.get('tpot_limit', 0.3)
            slo_violation = ttft_avg > ttft_limit or tpot_avg > tpot_limit
            
            # 构建特征字典
            features_dict = {}
            for i, name in enumerate(feature_names):
                if i < len(features):
                    features_dict[name] = float(features[i])
            
            # 创建轮次数据
            round_data = create_round_data_dict(
                round_num=round_num,
                gpu_frequency=gpu_frequency,
                energy_j=energy_j,
                ttft=ttft_avg,
                tpot=tpot_avg,
                e2e=e2e_avg,
                total_throughput=total_throughput,
                prefill_throughput=prefill_throughput,
                decode_throughput=decode_throughput,
                edp_raw=edp_raw,
                edp_normalized=edp_normalized,
                edp_baseline=edp_baseline,
                active_requests=int(active_requests),
                cache_usage=cache_usage,
                completed_requests=int(completed_requests),
                learning_phase=learning_phase,
                reward=reward,
                alpha=alpha_value,
                ucb_confidence=ucb_confidence,
                action_method=action_method,
                available_freqs=available_freqs,
                pruned_freqs=pruned_freqs,
                freq_exploration_count=freq_exploration_count,
                slo_violation=slo_violation,
                ttft_limit=ttft_limit,
                tpot_limit=tpot_limit,
                features=features_dict
            )
            
            # 记录到文件
            self.experiment_recorder.record_round_data(round_data)
            
        except Exception as e:
            self.logger.error(f"❌ 记录实验数据失败: {e}")
    
    def _update_adaptive_sampler(self, action, reward: float, post_metrics: dict):
        """更新自适应采样器的反馈信息 - 仅在学习模式下执行，支持组合频率"""
        # 提取核心频率用于日志显示
        core_freq = action[0] if isinstance(action, tuple) else action
        action_desc = f"{action[0]}MHz核心+{action[1]}MHz显存" if isinstance(action, tuple) else f"{action}MHz核心"
        self.logger.debug(f"🔄 更新自适应采样器: 轮次{getattr(self.model, 'total_rounds', 'N/A')}, {action_desc}")
        
        if not post_metrics:
            self.logger.warning("⚠️ post_metrics为空，跳过自适应采样器更新")
            return
        
        # ✅ 关键改进：仅在学习模式下执行自适应采样
        if self.model.exploitation_mode:
            # 利用模式下不进行频率空间探索，但检查是否有严重SLO违规
            control_config = self.config.get('control', {})
            ignore_slo = control_config.get('ignore_slo', True)
            
            if not ignore_slo:
                # 在利用模式下也需要正确处理组合频率的SLO检查
                check_freq = action[0] if isinstance(action, tuple) else action
                slo_violated = self.reward_calculator.check_slo_violation(post_metrics, check_freq)
                if slo_violated:
                    self.logger.warning("⚠️ 利用模式下检测到SLO违规，可能环境发生重大变化")
                    self.logger.info("🔄 从利用模式切换回学习模式以重新适应环境")
                    self.model.set_exploitation_mode(False)
                    self.model.is_converged = False
                    # 继续执行下面的自适应采样逻辑
                else:
                    self.logger.debug("🔒 利用模式下跳过自适应采样器更新（已收敛，性能正常）")
                    return
            else:
                self.logger.debug("🔒 利用模式下跳过自适应采样器更新（已收敛，不再探索）")
                return
            
        # 1. 更新频率奖励反馈（传递LinUCB模型和当前上下文用于智能细化）
        # 获取当前上下文特征
        try:
            current_features, _, _ = self._get_current_state()
        except:
            current_features = None
        
        frequency_space_updated = self.gpu_controller.update_frequency_reward(
            core_freq, reward, 
            linucb_model=self.model, 
            current_context=current_features
        )
        
        # 2. 检查SLO违规（只在非ignore_slo模式下）
        control_config = self.config.get('control', {})
        ignore_slo = control_config.get('ignore_slo', True)
        
        if not ignore_slo:
            # 检查SLO违规时需要传递完整的动作信息（支持组合频率）
            check_freq = action[0] if isinstance(action, tuple) else action
            slo_violated = self.reward_calculator.check_slo_violation(post_metrics, check_freq)
            if slo_violated:
                # 传递完整的动作给SLO违规处理器
                slo_space_updated = self.gpu_controller.update_slo_violation(action)
                if slo_space_updated:
                    frequency_space_updated = True
        
        # 3. 如果频率空间发生了更新，重新设置模型的动作空间
        if frequency_space_updated:
            new_frequencies = self.gpu_controller.get_available_frequencies(self.model)
            self.model.update_action_space(new_frequencies)
            
            # 如果启用了显存频率优化，也需要同步显存频率列表
            if (self.model.memory_optimization_enabled and 
                self.gpu_controller.enable_memory_frequency_control and 
                self.gpu_controller.memory_frequency_supported):
                
                new_memory_frequencies = self.gpu_controller.get_available_memory_frequencies()
                if hasattr(self.model, 'update_memory_frequencies'):
                    self.model.update_memory_frequencies(new_memory_frequencies)
                else:
                    # 直接更新显存频率列表
                    self.model.available_memory_frequencies = new_memory_frequencies
                    self.logger.info(f"💾 同步显存频率列表: {len(new_memory_frequencies)}个频率")
            
            self.logger.info(f"🎯 学习模式下自适应采样器触发频率空间更新: {len(new_frequencies)}个频率")
            self.logger.info("📊 发现新的频率区域，继续探索学习")
    
    
    def run(self):
        """运行主控制循环"""
        self.logger.info("🎮 开始主控制循环...")
        self.running = True
        
        
        # 运行统计  
        start_time = time.time()
        self.logger.info(f"🚀 开始时间: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            while self.running:
                try:
                    # 获取当前状态
                    features, available_frequencies, metrics = self._get_current_state()
                    
                    # Baseline收集：只有在有任务时才开始基线收集
                    current_round = getattr(self.model, 'total_rounds', 0) or 0
                    baseline_target = self.reward_calculator.baseline_target_count
                    if not self.reward_calculator.baseline_collected and current_round < baseline_target:
                        # 检查是否有任务 - 只有在有任务时才进行基线收集
                        running_requests = metrics.get('vllm:num_requests_running', 0)
                        has_queue = metrics.get('vllm:num_requests_waiting', 0) > 0
                        
                        if running_requests == 0 and not has_queue:
                            # 没有任务，等待任务开始
                            self.logger.info("⏳ 等待任务开始以进行基线收集...")
                            self.logger.info(f"   当前状态: 运行任务={running_requests}, 等待队列={has_queue}")
                            self.logger.info("   说明: 系统将等待检测到任务后才开始基线EDP收集")
                            time.sleep(2)  # 等待任务到来
                            continue
                        
                        measurement_num = len(self.reward_calculator.baseline_measurements) + 1
                        self.logger.info(f"🎯 检测到任务，开始基线EDP收集轮次 {measurement_num}/{baseline_target}...")
                        self.logger.info(f"   当前状态: 运行任务={running_requests}, 等待队列={has_queue}")
                        # 基线收集使用安全的默认频率，不尝试改变频率
                        default_freq = self.gpu_controller.current_freq
                        self.logger.info(f"🎯 基线收集使用当前频率 {default_freq}MHz（不改变频率）")
                        
                        # 直接收集指标而不设置频率
                        reward, post_metrics, edp_value = self._collect_baseline_metrics()
                        
                        # 只有在收集到有效指标时才更新模型
                        if post_metrics and reward != 0.0:
                            # 更新模型以增加轮次计数
                            self.model.update(features, default_freq, reward, edp_value)
                            
                            # 记录baseline收集
                            current_round = self.model.total_rounds
                            if self.reward_calculator.baseline_collected:
                                self.logger.info(f"✅ Baseline EDP收集全部完成 (轮次: {current_round}, 频率: {default_freq}MHz)")
                            else:
                                self.logger.info(f"🔄 Baseline EDP测量进行中 {measurement_num}/{baseline_target} (轮次: {current_round})")
                            
                            # 记录详细信息
                            self._log_round_details(current_round, features, default_freq, 
                                                  reward, edp_value, metrics, post_metrics)
                            self._record_experiment_data(current_round, features, default_freq, 
                                                       reward, edp_value, metrics, post_metrics)
                        else:
                            # 没有收集到有效指标，不增加轮次计数
                            self.logger.info("⏸️ 基线收集期间无任务，暂停此轮收集")
                            time.sleep(1.0)  # 短暂等待后重试
                        
                        # 继续下一轮
                        time.sleep(1.8)  # 基线收集后短暂休息
                        continue
                    
                    # 检查是否应该进入休息模式
                    is_idle = self._check_idle_mode(metrics)
                    
                    if is_idle:
                        # 进入休息模式：重置频率，等待，不增加决策轮次
                        self._handle_idle_mode()
                        time.sleep(2)  # 休息模式下等待2秒
                        continue  # 跳过学习，不增加轮次
                    
                    # 数据收集模式：仅收集数据不调频，但收集完整的能量和EDP数据
                    if self.data_collection_mode:
                        # 数据收集模式下也需要双重指标收集来计算能量和EDP
                        pre_metrics = metrics.copy()
                        
                        # 等待硬件稳定（与学习模式相同）
                        time.sleep(0.3)
                        
                        # 收集执行后的指标
                        post_metrics = self.metrics_collector.collect_metrics(
                            energy_reader=self.gpu_controller.read_energy_j
                        )
                        
                        # 计算delta指标
                        for key in post_metrics:
                            if key in pre_metrics and (key.endswith('_total') or key.endswith('_sum') or key.endswith('_count')):
                                delta_key = key + '_delta'
                                post_metrics[delta_key] = post_metrics[key] - pre_metrics[key]
                        
                        # 处理数据收集模式
                        self._handle_data_collection_mode(features, pre_metrics, post_metrics)
                        
                        time.sleep(1.5)  # 保持与决策间隔相似的节奏
                        continue
                    
                    # 正常学习模式：做出决策
                    selected_action = self._make_decision(features, available_frequencies)
                    
                    # 执行动作并测量奖励
                    reward, post_metrics, edp_value = self._execute_action_and_measure(selected_action, is_baseline=False)
                    
                    # 更新模型（只有在有有效动作时才更新）
                    if selected_action is not None:
                        self.model.update(features, selected_action, reward, edp_value)
                    else:
                        self.logger.info("📊 重置模式，跳过模型更新，继续监控系统状态")
                    
                    # 显示当前轮次和记录数据（只有在有效动作时）
                    if selected_action is not None:
                        current_round = getattr(self.model, 'total_rounds', 0) or 0
                        print(f"🎯 当前决策轮次: {current_round}")
                        
                        # 记录详细的每轮信息
                        self._log_round_details(self.model.total_rounds, features, selected_action, 
                                              reward, edp_value, metrics, post_metrics)
                        
                        # 记录实验数据到文件
                        self._record_experiment_data(self.model.total_rounds, features, selected_action, 
                                                    reward, edp_value, metrics, post_metrics)
                    else:
                        # 重置模式下只显示状态
                        current_freq = self.gpu_controller._get_current_frequency()
                        print(f"🔄 GPU重置模式运行中 (当前频率: {current_freq}MHz)")
                    
                    # 集成自适应采样器反馈 (仅在有有效动作时)
                    if selected_action is not None:
                        # 修复：传递完整的动作信息支持组合频率SLO处理
                        self._update_adaptive_sampler(selected_action, reward, post_metrics)
                    
                    # 执行智能动作修剪检查（仅在有有效动作时）
                    if selected_action is not None and hasattr(self.model, '_perform_action_pruning'):
                        # 记录修剪前的状态
                        pruned_before = len(getattr(self.model, 'pruned_frequencies', set()))
                        
                        # 执行修剪（内部会检查条件）
                        self.model._perform_action_pruning()
                        
                        # 检查是否有新的修剪发生
                        pruned_after = len(getattr(self.model, 'pruned_frequencies', set()))
                        if pruned_after > pruned_before:
                            # 有新修剪发生，需要同步可用频率列表
                            self.logger.debug(f"🔄 检测到新修剪({pruned_after - pruned_before}个)，同步频率列表")
                            # 使用统一方法获取最新的可用频率
                            if hasattr(self.gpu_controller, 'adaptive_sampler'):
                                updated_frequencies = self.gpu_controller.adaptive_sampler.get_available_frequencies_unified(
                                    linucb_model=self.model,
                                    gpu_controller=self.gpu_controller
                                )
                                # 更新模型的动作空间
                                self.model.update_action_space(updated_frequencies)
                        
                        # 每20轮显示一次修剪状态
                        if self.model.total_rounds % 20 == 0:
                            pruning_stats = self.model.get_model_stats()
                            self.logger.info(f"📊 智能修剪状态 - 轮次{self.model.total_rounds}: "
                                           f"已修剪{pruning_stats.get('pruned_frequencies_count', 0)}个频率, "
                                           f"活跃频率{pruning_stats.get('active_frequencies_count', 0)}个")
                    
                    # 定期保存模型
                    if self.model.total_rounds % 50 == 0:
                        self.model.save_model()
                        
                        # 输出统计信息
                        stats = self.model.get_model_stats()
                        mode_str = "利用(已收敛)" if self.model.exploitation_mode else "学习(探索中)"
                        self.logger.info(f"📊 轮次 {self.model.total_rounds}: "
                                       f"奖励={reward:.3f}, "
                                       f"平均奖励={stats['avg_reward']:.3f}, "
                                       f"模式={mode_str}, "
                                       f"频率空间={stats.get('n_arms', 0)}个")
                    
                    # 检查收敛 (学习→利用模式切换)
                    self._check_convergence()
                    
                    # 检查性能退化 (利用→学习模式切换)
                    if self.model.exploitation_mode and self._check_performance_degradation():
                        self.logger.warning("🔄 检测到性能退化，从利用模式切换回学习模式")
                        self.model.set_exploitation_mode(False)
                        self.model.is_converged = False
                        self.logger.info("📊 重新进入学习阶段，将重新探索频率空间")
                    
                except Exception as e:
                    self.logger.error(f"❌ 控制循环异常: {e}")
                    time.sleep(5)  # 错误后等待5秒
                    
        except KeyboardInterrupt:
            self.logger.info("🛑 用户中断程序")
        
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """清理资源"""
        self.logger.info("🧹 开始清理资源...")
        
        # 结束实验并保存数据
        if hasattr(self, 'experiment_recorder'):
            self.experiment_recorder.finalize_experiment()
        
        # 保存最终模型
        if hasattr(self, 'model'):
            self.model.save_model()
            
            # 输出最终统计
            stats = self.model.get_model_stats()
            self.logger.info("📈 最终统计:")
            self.logger.info(f"   总轮次: {stats['total_rounds']}")
            self.logger.info(f"   平均奖励: {stats['avg_reward']:.3f}")
            self.logger.info(f"   最近平均奖励: {stats['recent_avg_reward']:.3f}")
            self.logger.info(f"   频率数量: {stats['n_arms']}")
        
        # 重置GPU频率到安全值
        if hasattr(self, 'gpu_controller'):
            self.gpu_controller.reset_gpu_clocks()
        
        self.logger.info("✅ 资源清理完成")
        self.logger.info("👋 vLLM GPU自动调频器已退出")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="vLLM GPU自动调频器 - Contextual LinUCB版本")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--reset-model", action="store_true", help="重置模型，从零开始")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    print("🚀 启动vLLM GPU自动调频器 (Contextual LinUCB)")
    print(f"📁 配置文件: {args.config}")
    print(f"🔄 重置模型: {'是' if args.reset_model else '否'}")
    print(f"📊 日志级别: {args.log_level}")
    print("=" * 60)
    
    try:
        # 创建并运行自动调频器
        autoscaler = VLLMGPUAutoscaler(
            config_path=args.config,
            reset_model=args.reset_model
        )
        autoscaler.run()
        
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()