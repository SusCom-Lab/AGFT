import subprocess
import numpy as np
from typing import List, Tuple, Optional, Dict
import pynvml
from .logger import setup_logger
import time
import re
import threading
import atexit
import traceback

logger = setup_logger(__name__)

try:
    from .adaptive_frequency_sampler import (
        AdaptiveFrequencySampler, 
        AdaptiveSamplingConfig, 
        SamplingMode,
        create_default_sampler
    )
    logger.debug("✅ 自适应频率采样器模块导入成功")
except ImportError as e:
    logger.error(f"❌ 自适应频率采样器模块导入失败: {e}")
    AdaptiveFrequencySampler = None
    AdaptiveSamplingConfig = None
    SamplingMode = None
    create_default_sampler = None
except Exception as e:
    logger.error(f"❌ 自适应频率采样器模块导入异常: {e}")
    AdaptiveFrequencySampler = None
    AdaptiveSamplingConfig = None
    SamplingMode = None
    create_default_sampler = None

class GPUController:
    """兼容性增强的GPU频率控制器 - 支持连续频率优化"""
    
    def __init__(self, gpu_id: int = 0, min_freq: int = 210, step: int = 15, 
                 auto_step: bool = True, ignore_slo: bool = True,
                 adaptive_update_interval: int = 20, enable_memory_frequency_control: bool = False,
                 memory_auto_detect: bool = True, memory_frequencies: List[int] = None,
                 reward_threshold: float = 0.5, learner_maturity_threshold: int = 100,
                 refinement_start_threshold: int = 50, actual_frequency_callback=None):
        self.gpu_id = gpu_id
        self.step = step
        self.auto_step = auto_step
        self.ignore_slo = ignore_slo
        self.adaptive_update_interval = adaptive_update_interval
        self.reward_threshold = reward_threshold
        self.learner_maturity_threshold = learner_maturity_threshold
        self.refinement_start_threshold = refinement_start_threshold
        
        # 动作空间自适应恢复回调函数
        self.actual_frequency_callback = actual_frequency_callback
        
        # 显存频率控制参数
        self.enable_memory_frequency_control = enable_memory_frequency_control
        self.memory_auto_detect = memory_auto_detect
        self.memory_frequencies = memory_frequencies or []
        self.memory_frequency_supported = False
        self.current_memory_freq = None
        
        # 线程安全锁
        self._lock = threading.Lock()
        
        
        
        # 失败频率记录 - 用于动态移除失败的频率
        self.failed_frequencies: set = set()  # 失败的核心频率
        self.failed_memory_frequencies: set = set()  # 失败的显存频率
        self._failed_frequencies_lock = threading.Lock()  # 失败频率集合的专用锁
        
        # GPU时钟重置状态标记
        self.is_clock_reset = False
        
        # 自适应频率采样器 - 智能频率空间管理
        self.adaptive_sampler = None
        
        # 初始化pynvml
        try:
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            # 注册清理函数
            atexit.register(self._cleanup)
            logger.info("✅ NVML初始化成功")
        except Exception as e:
            logger.error(f"❌ NVML初始化失败: {e}")
            raise
        
        # 获取GPU信息
        self._get_gpu_info()
        
        # 获取频率范围
        self.min_freq, self.max_freq = self._get_frequency_limits_safe()
        
        logger.info(f"🎮 GPU {gpu_id} 频率范围: {self.min_freq}-{self.max_freq}MHz")
        
        # 检测和初始化显存频率控制
        if self.enable_memory_frequency_control:
            self._detect_memory_frequency_support()
            self._initialize_memory_frequencies()
        
        # 初始化自适应频率采样器
        logger.info("🚀 GPU控制器：开始初始化自适应频率采样器...")
        self._initialize_adaptive_sampler()
        logger.info(f"🔍 初始化后状态检查: self.adaptive_sampler = {self.adaptive_sampler is not None}")
        
        # 生成智能频率列表
        logger.info("🚀 GPU控制器：开始生成智能频率列表...")
        self.frequencies = self._get_adaptive_frequency_list()
        logger.info(f"📊 最终智能频率列表: {self.frequencies}")
        if len(self.frequencies) > 10:
            logger.info(f"📊 显示频率列表: {self.frequencies[:5]}...{self.frequencies[-5:]} MHz")
        else:
            logger.info(f"📊 完整频率列表: {self.frequencies} MHz")
        logger.info(f"🎯 共 {len(self.frequencies)} 个频率档位")
        
        # 显示动作空间信息
        if self.enable_memory_frequency_control and self.memory_frequency_supported:
            total_actions = len(self.frequencies) * len(self.memory_frequencies)
            logger.info(f"💾 显存频率控制已启用: {len(self.memory_frequencies)} 个档位")
            logger.info(f"🎯 总动作空间: {total_actions} 个组合 ({len(self.frequencies)}核心 × {len(self.memory_frequencies)}显存) - 无限制")
        else:
            logger.info(f"🎯 动作空间: {len(self.frequencies)} 个核心频率 (仅核心频率控制) - 无限制")
        
        # 获取当前频率
        self.current_freq = self._get_current_frequency()
        self.current_idx = self._freq_to_idx(self.current_freq)
        logger.info(f"📍 当前频率: {self.current_freq}MHz (索引: {self.current_idx})")
        
        # 初始能耗和时间戳
        self.last_energy_j = self._get_total_energy_consumption()
        self.last_energy_timestamp = time.time()
        logger.info(f"⚡ 初始能耗读数: {self.last_energy_j:.3f}J")
    
    def _get_gpu_info(self):
        """获取GPU详细信息"""
        try:
            # 新版本pynvml可能返回字符串或bytes
            gpu_name = pynvml.nvmlDeviceGetName(self.nvml_handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')
            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
            total_mem_gb = mem_info.total / (1024**3)
            
            # 获取驱动版本
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode('utf-8')
            
            logger.info(f"🖥️  GPU信息: {gpu_name} ({total_mem_gb:.0f}GB)")
            logger.info(f"🔧 驱动版本: {driver_version}")
            
            # 获取CUDA版本
            try:
                cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
                cuda_major = cuda_version // 1000
                cuda_minor = (cuda_version % 1000) // 10
                logger.info(f"🔧 CUDA版本: {cuda_major}.{cuda_minor}")
            except:
                pass
                
        except Exception as e:
            logger.warning(f"获取GPU信息失败: {e}")
    
    def _run_nvidia_smi_query(self, query_string: str, timeout: int = 5) -> Optional[str]:
        """安全运行nvidia-smi查询 - 添加超时保护"""
        cmd = f"nvidia-smi -i {self.gpu_id} --query-gpu={query_string} --format=csv,noheader,nounits"
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, 
                                  check=True, timeout=timeout)
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.warning(f"nvidia-smi查询超时 ({query_string}): {timeout}秒")
            return None
        except subprocess.CalledProcessError as e:
            logger.debug(f"nvidia-smi查询失败 ({query_string}): {e}")
            return None
    
    def _get_frequency_limits_safe(self) -> Tuple[int, int]:
        """通过NVML原生频率检测获取GPU频率限制"""
        # 优先尝试从NVML原生频率点获取真实范围
        if self.auto_step:
            try:
                mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(self.nvml_handle)
                if mem_clocks:
                    all_graphics_clocks = []
                    for mem_clock in mem_clocks:
                        try:
                            gfx_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(
                                self.nvml_handle, mem_clock)
                            all_graphics_clocks.extend(gfx_clocks)
                        except:
                            pass
                    
                    if all_graphics_clocks:
                        unique_clocks = sorted(set(all_graphics_clocks))
                        min_freq, max_freq = unique_clocks[0], unique_clocks[-1]
                        logger.info(f"✅ 通过NVML原生频率检测: {min_freq}-{max_freq}MHz ({len(unique_clocks)}个频点)")
                        return min_freq, max_freq
            except Exception as e:
                logger.debug(f"NVML原生频率检测失败: {e}")
        
        # 备用方案：nvidia-smi查询
        max_freq = 2100  # 默认值
        result = self._run_nvidia_smi_query("clocks.max.graphics")
        if result:
            try:
                max_freq = int(result.strip())
                logger.info(f"✅ nvidia-smi查询最大频率: {max_freq}MHz")
            except:
                logger.warning("解析最大频率失败，使用默认值2100MHz")
        else:
            logger.warning("无法查询最大频率，使用默认值2100MHz")
        
        min_freq = 210  # 通用默认最小频率
        logger.info(f"📊 GPU频率范围: {min_freq}-{max_freq}MHz")
        return min_freq, max_freq
    
    def _detect_memory_frequency_support(self) -> bool:
        """检测GPU是否支持显存频率控制"""
        if not self.memory_auto_detect:
            self.memory_frequency_supported = True
            logger.info("🔧 跳过显存频率支持检测（手动配置）")
            return True
        
        logger.info("🔍 检测GPU显存频率控制支持...")
        
        try:
            # 方法1：尝试通过NVML获取支持的显存频率
            mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(self.nvml_handle)
            if mem_clocks and len(mem_clocks) > 1:
                logger.info(f"✅ NVML检测到 {len(mem_clocks)} 个显存频率档位")
                self.memory_frequency_supported = True
                return True
        except Exception as e:
            logger.debug(f"NVML显存频率检测失败: {e}")
        
        try:
            # 方法2：尝试通过nvidia-smi测试显存频率锁定
            current_mem_freq = self._get_current_memory_frequency()
            if current_mem_freq > 0:
                # 尝试设置相同的显存频率（应该不会有副作用）
                cmd = f"nvidia-smi -i {self.gpu_id} -lmc {current_mem_freq},{current_mem_freq}"
                result = subprocess.run(cmd.split(), capture_output=True, text=True, 
                                      check=True, timeout=5)
                
                if result.returncode == 0:
                    logger.info("✅ nvidia-smi显存频率控制测试成功")
                    self.memory_frequency_supported = True
                    # 重置显存频率到默认状态
                    subprocess.run(f"nvidia-smi -i {self.gpu_id} -rmc".split(), 
                                 capture_output=True, timeout=5)
                    return True
        except Exception as e:
            logger.debug(f"nvidia-smi显存频率测试失败: {e}")
        
        # 方法3：检查nvidia-smi帮助是否包含显存频率选项
        try:
            result = subprocess.run(["nvidia-smi", "--help"], capture_output=True, text=True)
            if "-lmc" in result.stdout and "--lock-memory-clocks" in result.stdout:
                logger.info("✅ nvidia-smi支持显存频率控制命令")
                self.memory_frequency_supported = True
                return True
        except:
            pass
        
        logger.warning("❌ GPU不支持显存频率控制或检测失败")
        self.memory_frequency_supported = False
        return False
    
    def _initialize_memory_frequencies(self):
        """初始化显存频率列表"""
        if not self.memory_frequency_supported:
            logger.warning("显存频率控制不支持，跳过初始化")
            return
        
        # 如果手动指定了显存频率列表，直接使用
        if self.memory_frequencies:
            logger.info(f"💾 使用手动配置的显存频率: {self.memory_frequencies}")
            return
        
        # 自动检测显存频率
        logger.info("🔍 自动检测可用显存频率...")
        
        try:
            # 通过NVML获取支持的显存频率
            mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(self.nvml_handle)
            if mem_clocks:
                # 对频率进行排序和去重
                unique_mem_freqs = sorted(set(mem_clocks), reverse=True)
                
                # 限制显存频率数量以控制动作空间大小
                max_memory_actions = min(8, len(unique_mem_freqs))  # 最多8个显存频率
                if len(unique_mem_freqs) > max_memory_actions:
                    # 均匀采样选择代表性频率
                    indices = np.linspace(0, len(unique_mem_freqs)-1, max_memory_actions, dtype=int)
                    self.memory_frequencies = [unique_mem_freqs[i] for i in indices]
                else:
                    self.memory_frequencies = unique_mem_freqs
                
                logger.info(f"✅ 检测到显存频率: {self.memory_frequencies}")
                
                # 获取当前显存频率
                self.current_memory_freq = self._get_current_memory_frequency()
                logger.info(f"📍 当前显存频率: {self.current_memory_freq}MHz")
                return
                
        except Exception as e:
            logger.warning(f"自动检测显存频率失败: {e}")
        
        # 使用默认的显存频率列表作为后备
        default_mem_freqs = [8001, 7601, 5001, 810, 405]  # 基于之前的检测结果
        current_mem = self._get_current_memory_frequency()
        
        # 过滤出当前GPU支持的频率
        supported_freqs = []
        for freq in default_mem_freqs:
            if abs(freq - current_mem) < 100:  # 如果当前频率接近某个默认值，说明支持
                supported_freqs.append(freq)
        
        if not supported_freqs:
            supported_freqs = [current_mem] if current_mem > 0 else [810]  # 至少包含当前频率
        
        self.memory_frequencies = supported_freqs
        self.current_memory_freq = current_mem
        logger.info(f"🔧 使用默认显存频率列表: {self.memory_frequencies}")
        
    def _initialize_adaptive_sampler(self):
        """初始化自适应频率采样器 - 使用动态检测的频率范围"""
        logger.debug("🚀 初始化自适应频率采样器...")
        
        if AdaptiveFrequencySampler is None or AdaptiveSamplingConfig is None or SamplingMode is None:
            logger.error("❌ 关键类未导入，无法初始化自适应采样器")
            self.adaptive_sampler = None
            return
        
        try:
            config = AdaptiveSamplingConfig(
                min_freq=self.min_freq,
                max_freq=self.max_freq,
                reward_threshold=self.reward_threshold,
                learner_maturity_threshold=self.learner_maturity_threshold,
                refinement_start_threshold=self.refinement_start_threshold
            )
            
            mode = SamplingMode.EDP_OPTIMAL if self.ignore_slo else SamplingMode.SLO_AWARE
        except Exception as e:
            logger.error(f"❌ 采样模式确定失败: {e}")
            logger.error(f"   错误堆栈: {traceback.format_exc()}")
            self.adaptive_sampler = None
            return
        
        try:
            self.adaptive_sampler = AdaptiveFrequencySampler(config)
            self.adaptive_sampler.set_mode(mode)
            logger.info(f"✅ 自适应采样器初始化成功 (模式: {mode.value}, 频率范围: {self.min_freq}-{self.max_freq}MHz)")
        except Exception as e:
            logger.error(f"❌ 自适应采样器初始化失败: {e}")
            self.adaptive_sampler = None
    
    def _get_adaptive_frequency_list(self) -> List[int]:
        """获取智能频率列表"""
        if self.adaptive_sampler:
            try:
                frequencies = self.adaptive_sampler.get_current_frequencies()
                logger.debug(f"✅ 自适应采样器生成 {len(frequencies)} 个频率点")
                return frequencies
            except Exception as e:
                logger.error(f"❌ 自适应采样器获取频率失败: {e}")
        
        # 后备方案：使用简化的频率列表生成
        logger.info("🔧 使用后备频率生成方案")
        fallback_frequencies = self._generate_frequency_list()
        return fallback_frequencies
    
    def get_available_frequencies(self, linucb_model=None) -> List[int]:
        """获取当前可用的频率列表（排除失败和修剪的频率）- 统一入口"""
        if self.adaptive_sampler:
            try:
                # 使用统一的频率获取方法，自动应用所有必要的过滤
                return self.adaptive_sampler.get_available_frequencies_unified(
                    linucb_model=linucb_model, 
                    gpu_controller=self
                )
            except Exception as e:
                logger.warning(f"⚠️ 获取过滤频率失败: {e}，使用基础频率列表")
        
        # 后备方案：使用当前频率列表并排除失败频率
        frequencies = [f for f in self.frequencies if f not in self.failed_frequencies]
        return frequencies
    
    def get_available_memory_frequencies(self) -> List[int]:
        """获取当前可用的显存频率列表（仅排除设置失败的频率）"""
        if not self.enable_memory_frequency_control or not self.memory_frequency_supported:
            return []
        
        # 只排除设置失败的显存频率，不应用全局SLO过滤
        # SLO边界应该在组合验证时应用，而不是全局过滤
        with self._failed_frequencies_lock:
            valid_memory_freqs = [f for f in self.memory_frequencies 
                                 if f not in self.failed_memory_frequencies]
        
        if not valid_memory_freqs:
            logger.error("❌ 所有显存频率都失败了，使用原始最高频率作为最后手段")
            sorted_freqs = sorted(self.memory_frequencies, reverse=True)
            valid_memory_freqs = sorted_freqs[:1]
        
        logger.debug(f"📊 可用显存频率: {valid_memory_freqs}")
        return valid_memory_freqs
    
    def _notify_global_memory_frequency_disabled(self, memory_freq: int):
        """通知关联的LinUCB模型全局禁用显存频率"""
        # 这个方法会被主控制器设置的回调函数覆盖
        pass
    
    def _notify_core_frequency_disabled(self, core_freq: int, reason: str):
        """通知关联的LinUCB模型禁用核心频率"""
        # 这个方法会被主控制器设置的回调函数覆盖
        pass
    
    def _notify_core_frequencies_disabled_below_threshold(self, threshold_freq: int, reason: str):
        """通知关联的LinUCB模型禁用阈值频率及以下的所有核心频率"""
        # 这个方法会被主控制器设置的回调函数覆盖
        pass
    
    def _notify_core_memory_combination_disabled(self, core_freq: int, memory_freq: int, include_lower: bool = True):
        """通知关联的LinUCB模型禁用核心频率-显存频率组合"""
        # 这个方法会被主控制器设置的回调函数覆盖
        pass
    
    def _notify_memory_slo_boundary_propagated(self, violating_core_freq: int, violating_memory_freq: int):
        """通知关联的LinUCB模型传播显存频率SLO边界"""
        # 这个方法会被主控制器设置的回调函数覆盖
        pass
    
    def set_linucb_callback(self, linucb_model):
        """设置LinUCB模型回调，用于处理频率禁用通知"""
        def notify_global_memory_disabled(memory_freq):
            if hasattr(linucb_model, 'disable_memory_frequency_globally'):
                linucb_model.disable_memory_frequency_globally(memory_freq, "设置失败")
        
        def notify_core_disabled(core_freq, reason):
            if hasattr(linucb_model, 'disable_core_frequency_for_memory_issues'):
                linucb_model.disable_core_frequency_for_memory_issues(core_freq, reason)
        
        def notify_core_frequencies_below_threshold_disabled(threshold_freq, reason):
            if hasattr(linucb_model, 'disable_core_frequencies_below_threshold'):
                linucb_model.disable_core_frequencies_below_threshold(threshold_freq, reason)
        
        def notify_combination_disabled(core_freq, memory_freq, include_lower):
            if hasattr(linucb_model, 'disable_core_memory_combination'):
                linucb_model.disable_core_memory_combination(core_freq, memory_freq, include_lower)
        
        def notify_memory_slo_boundary_propagated(violating_core_freq, violating_memory_freq):
            if hasattr(linucb_model, 'propagate_memory_slo_boundary'):
                linucb_model.propagate_memory_slo_boundary(violating_core_freq, violating_memory_freq)
        
        self._notify_global_memory_frequency_disabled = notify_global_memory_disabled
        self._notify_core_frequency_disabled = notify_core_disabled
        self._notify_core_frequencies_disabled_below_threshold = notify_core_frequencies_below_threshold_disabled
        self._notify_core_memory_combination_disabled = notify_combination_disabled
        self._notify_memory_slo_boundary_propagated = notify_memory_slo_boundary_propagated
    
    def update_adaptive_sampler_frequencies(self, new_min_freq: int = None, new_max_freq: int = None):
        """更新自适应采样器的频率范围"""
        if not self.adaptive_sampler:
            return
            
        # 更新频率范围
        if new_min_freq is not None:
            self.min_freq = new_min_freq
        if new_max_freq is not None:
            self.max_freq = new_max_freq
            
        # 重新初始化采样器
        old_frequencies = self.frequencies.copy() if hasattr(self, 'frequencies') else []
        self._initialize_adaptive_sampler()
        self.frequencies = self._get_adaptive_frequency_list()
        
        logger.info(f"🔄 更新频率范围: {self.min_freq}-{self.max_freq}MHz")
        logger.info(f"   频率列表: {len(old_frequencies)} -> {len(self.frequencies)} 个点")
    
    def _generate_frequency_list(self) -> List[int]:
        """简化的后备频率列表生成 - 仅在自适应采样器失败时使用"""
        
        # 使用粗粒度等间距采样作为可靠的后备方案
        fallback_step = 90  # 90MHz步长，确保足够的覆盖范围
        
        frequencies = list(range(self.min_freq, self.max_freq + 1, fallback_step))
        
        # 确保包含最大频率
        if self.max_freq not in frequencies:
            frequencies.append(self.max_freq)
        
        logger.info(f"🔧 使用简化后备方案，共{len(frequencies)}个动作 (步长: {fallback_step}MHz, 无限制)")
        return frequencies
    
    def _get_current_frequency(self) -> int:
        """获取当前GPU频率"""
        # 尝试多种查询方式
        queries = [
            "clocks.current.graphics",
            "clocks.gr",
            "clocks.current.gr"
        ]
        
        for query in queries:
            result = self._run_nvidia_smi_query(query)
            if result:
                try:
                    freq = int(result)
                    logger.debug(f"通过 {query} 获取当前频率: {freq}MHz")
                    return freq
                except Exception as e:
                    logger.debug(f"解析 {query} 结果失败: {result}, 错误: {e}")
        
        
        
        # 使用缓存的频率作为最后手段
        if hasattr(self, 'current_freq') and self.current_freq:
            logger.warning(f"所有查询方式失败，使用缓存频率: {self.current_freq}MHz")
            return self.current_freq
        
        fallback_freq = self.frequencies[len(self.frequencies)//2] if hasattr(self, 'frequencies') else 1500
        logger.error(f"无法获取当前频率，使用默认值: {fallback_freq}MHz")
        return fallback_freq
    
    def _get_current_memory_frequency(self) -> int:
        """获取当前GPU显存频率"""
        # 尝试多种查询方式
        queries = [
            "clocks.current.memory",
            "clocks.mem",
            "clocks.current.mem"
        ]
        
        for query in queries:
            result = self._run_nvidia_smi_query(query)
            if result:
                try:
                    return int(result)
                except:
                    pass
        
        # 使用pynvml作为后备
        try:
            clock_info = pynvml.nvmlDeviceGetClockInfo(self.nvml_handle, 
                                                      pynvml.NVML_CLOCK_MEM)
            return clock_info
        except:
            pass
        
        logger.warning("无法获取当前显存频率")
        return 0
    
    def set_frequency(self, freq_mhz: int, max_retries: int = 3) -> bool:
        """设置GPU频率 - 添加重试机制和线程安全"""
        pre_set_freq = self._get_current_frequency()
        
        # 查找最接近的可用频率
        closest_freq = freq_mhz
        
        # 指数退避重试
        for attempt in range(max_retries):
            success = False
            
            # 方法1：直接设置
            cmd = f"nvidia-smi -i {self.gpu_id} -lgc {closest_freq}"
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True, 
                                      check=True, timeout=10)
                success = True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.debug(f"直接设置失败 (尝试{attempt+1}/{max_retries}): {e}")
                
                # 方法2：使用锁定时钟
                cmd = f"nvidia-smi -i {self.gpu_id} --lock-gpu-clocks={closest_freq},{closest_freq}"
                try:
                    result = subprocess.run(cmd.split(), capture_output=True, text=True, 
                                          check=True, timeout=10)
                    success = True
                    logger.debug("使用--lock-gpu-clocks成功")
                except Exception as e2:
                    logger.debug(f"锁定时钟也失败 (尝试{attempt+1}/{max_retries}): {e2}")
            
            if success:
                # 等待一下让设置生效
                time.sleep(1)
                
                # 验证实际频率
                actual_freq = self._get_current_frequency()
                self.current_freq = actual_freq
                # 只有当频率列表不为空时才更新索引
                if self.frequencies:
                    self.current_idx = self._freq_to_idx(actual_freq)
                else:
                    self.current_idx = -1  # 重置模式
                
                tolerance = 5  # MHz
                # 检查实际频率与我们试图设置的频率的偏差
                actual_deviation = abs(actual_freq - closest_freq)
                if actual_deviation > tolerance:
                    logger.warning(
                        f"❌ GPU频率设置偏差过大(>{tolerance}MHz): "
                        f"尝试设置{closest_freq}MHz，实际{actual_freq}MHz (偏差{actual_deviation}MHz)"
                    )
                    # 标记试图设置的频率为失败频率
                    with self._failed_frequencies_lock:
                        self.failed_frequencies.add(closest_freq)
                    logger.info(f"🚫 已将频率 {closest_freq}MHz 标记为失败，将从动作空间移除")
                    
                    # 通知主控制器添加实际频率到动作空间
                    if self.actual_frequency_callback and actual_freq != closest_freq:
                        try:
                            logger.info(f"🔄 通知主控制器将实际频率 {actual_freq}MHz 添加到动作空间")
                            self.actual_frequency_callback(actual_freq)
                        except Exception as e:
                            logger.error(f"❌ 调用实际频率回调函数失败: {e}")
                    
                    # 回滚 internal state，告诉调用方失败
                    self._get_current_frequency()  # 再刷一次确保 current_freq 正确
                    return False

                # 显示频率设置结果
                freq_change = actual_freq - pre_set_freq
                if freq_change != 0:
                    change_symbol = "📈" if freq_change > 0 else "📉"
                    logger.info(
                        f"✅ GPU频率设置成功: {pre_set_freq}MHz → {actual_freq}MHz "
                        f"({change_symbol} {freq_change:+d}MHz)"
                    )
                else:
                    logger.info(f"✅ GPU频率保持: {actual_freq}MHz")

                # 设置了特定频率后，时钟不再处于重置状态
                self.is_clock_reset = False
                return True
            
            # 如果不是最后一次尝试，等待指数退避时间
            if attempt < max_retries - 1:
                wait_time = 0.1 * (2 ** attempt)  # 0.1s, 0.2s, 0.4s
                logger.warning(f"⏳ 频率设置失败，{wait_time:.1f}秒后重试...")
                time.sleep(wait_time)
        
        # 所有重试都失败
        logger.error(f"❌ GPU频率设置失败，已重试{max_retries}次: {freq_mhz}MHz")
        
        # 标记此频率为失败频率 - 线程安全
        with self._failed_frequencies_lock:
            self.failed_frequencies.add(freq_mhz)
        logger.info(f"🚫 已将频率 {freq_mhz}MHz 标记为失败，将从动作空间移除")
        
        
        return False
    
    def set_frequency_by_index(self, idx: int) -> bool:
        """通过索引设置频率"""
        if 0 <= idx < len(self.frequencies):
            return self.set_frequency(self.frequencies[idx])
        else:
            logger.error(f"无效的频率索引: {idx}")
            return False
    
    def set_memory_frequency(self, mem_freq_mhz: int, max_retries: int = 3) -> bool:
        """设置GPU显存频率"""
        if not self.enable_memory_frequency_control or not self.memory_frequency_supported:
            logger.debug("显存频率控制未启用或不支持")
            return True  # 返回True避免影响主控制流程
        
        # 记录设置前的显存频率 - 使用缓存值
        pre_set_mem_freq = getattr(self, 'current_memory_freq', None)
        if pre_set_mem_freq is None:
            pre_set_mem_freq = self._get_current_memory_frequency()
        
        with self._lock:
            # 查找最接近的可用显存频率
            closest_mem_freq = min(self.memory_frequencies, key=lambda x: abs(x - mem_freq_mhz))
            
            # 检查目标频率与最接近频率的偏差
            freq_diff = abs(mem_freq_mhz - closest_mem_freq)
            if freq_diff > 50:
                logger.warning(f"⚠️ 目标显存频率{mem_freq_mhz}MHz不在列表中，使用最接近的{closest_mem_freq}MHz")
        
        # 指数退避重试
        for attempt in range(max_retries):
            success = False
            
            # 使用nvidia-smi锁定显存频率
            cmd = f"nvidia-smi -i {self.gpu_id} -lmc {closest_mem_freq},{closest_mem_freq}"
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True, 
                                      check=True, timeout=10)
                success = True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.debug(f"显存频率设置失败 (尝试{attempt+1}/{max_retries}): {e}")
            
            if success:
                # 等待设置生效
                time.sleep(0.7)
                
                # 验证实际显存频率
                actual_mem_freq = self._get_current_memory_frequency()
                self.current_memory_freq = actual_mem_freq
                
                # 检查设置是否成功
                tolerance = 50  # MHz容差
                actual_deviation = abs(actual_mem_freq - closest_mem_freq)
                if actual_deviation > tolerance:
                    logger.warning(
                        f"❌ 显存频率设置偏差过大: 尝试{closest_mem_freq}MHz，实际{actual_mem_freq}MHz"
                    )
                    continue  # 重试
                
                # 显示设置结果
                mem_freq_change = actual_mem_freq - pre_set_mem_freq
                if abs(mem_freq_change) > 50:
                    change_symbol = "📈" if mem_freq_change > 0 else "📉"
                    logger.info(
                        f"✅ 显存频率设置成功: {pre_set_mem_freq}MHz → {actual_mem_freq}MHz "
                        f"({change_symbol} {mem_freq_change:+d}MHz)"
                    )
                else:
                    logger.info(f"✅ 显存频率保持: {actual_mem_freq}MHz")
                
                return True
            
            # 重试等待
            if attempt < max_retries - 1:
                wait_time = 0.1 * (2 ** attempt)
                logger.warning(f"⏳ 显存频率设置失败，{wait_time:.1f}秒后重试...")
                time.sleep(wait_time)
        
        logger.error(f"❌ 显存频率设置失败，已重试{max_retries}次: {mem_freq_mhz}MHz")
        
        # 将失败的显存频率添加到失败列表
        with self._failed_frequencies_lock:
            self.failed_memory_frequencies.add(closest_mem_freq)
        logger.warning(f"🚫 显存频率 {closest_mem_freq}MHz 已标记为失败，后续将被排除")
        
        # 通知LinUCB模型全局禁用这个显存频率
        self._notify_global_memory_frequency_disabled(closest_mem_freq)
        
        return False
    
    def set_dual_frequency(self, core_freq_mhz: int, memory_freq_mhz: int = None) -> bool:
        """同时设置核心频率和显存频率"""
        success_core = True
        success_memory = True
        
        # 设置核心频率
        success_core = self.set_frequency(core_freq_mhz)
        
        # 如果启用了显存频率控制且指定了显存频率，则设置显存频率
        if (self.enable_memory_frequency_control and 
            self.memory_frequency_supported and 
            memory_freq_mhz is not None):
            success_memory = self.set_memory_frequency(memory_freq_mhz)
        
        return success_core and success_memory
    
    def set_dual_frequency_by_action(self, action_idx: int) -> bool:
        """通过动作索引设置双频率"""
        if not self.enable_memory_frequency_control or not self.memory_frequency_supported:
            # 仅核心频率控制模式
            return self.set_frequency_by_index(action_idx)
        
        # 双频率控制模式：解码动作索引
        num_core_freqs = len(self.frequencies)
        num_memory_freqs = len(self.memory_frequencies)
        
        if action_idx >= num_core_freqs * num_memory_freqs:
            logger.error(f"无效的动作索引: {action_idx}")
            return False
        
        # 解码：action_idx = core_idx * num_memory_freqs + memory_idx
        core_idx = action_idx // num_memory_freqs
        memory_idx = action_idx % num_memory_freqs
        
        core_freq = self.frequencies[core_idx]
        memory_freq = self.memory_frequencies[memory_idx]
        
        return self.set_dual_frequency(core_freq, memory_freq)
    
    def get_total_action_count(self) -> int:
        """获取总动作数量"""
        if self.enable_memory_frequency_control and self.memory_frequency_supported:
            return len(self.frequencies) * len(self.memory_frequencies)
        else:
            return len(self.frequencies)
    
    def action_to_frequencies(self, action_idx: int) -> Tuple[int, Optional[int]]:
        """将动作索引转换为频率对"""
        if not self.enable_memory_frequency_control or not self.memory_frequency_supported:
            # 仅核心频率模式
            return self.frequencies[action_idx], None
        
        # 双频率模式
        num_memory_freqs = len(self.memory_frequencies)
        core_idx = action_idx // num_memory_freqs
        memory_idx = action_idx % num_memory_freqs
        
        return self.frequencies[core_idx], self.memory_frequencies[memory_idx]
    
    def frequencies_to_action(self, core_freq: int, memory_freq: int = None) -> int:
        """将频率对转换为动作索引"""
        # 找到最接近的核心频率索引
        core_idx = self._freq_to_idx(core_freq)
        # 如果处于重置模式（频率列表为空），返回-1
        if core_idx == -1:
            return -1
        
        if not self.enable_memory_frequency_control or not self.memory_frequency_supported or memory_freq is None:
            return core_idx
        
        # 找到最接近的显存频率索引
        if not self.memory_frequencies:
            logger.error(f"❌ 显存频率列表为空，无法转换显存频率 {memory_freq}MHz 到索引")
            return core_idx  # 回退到仅核心频率模式
        memory_distances = [abs(f - memory_freq) for f in self.memory_frequencies]
        memory_idx = np.argmin(memory_distances)
        
        # 编码为动作索引
        return core_idx * len(self.memory_frequencies) + memory_idx
    
    def reset_gpu_clocks(self) -> bool:
        """重置GPU时钟 - 智能避免重复重置"""
        # 如果已经重置过，直接返回成功
        if self.is_clock_reset:
            logger.debug("🔄 GPU时钟已处于重置状态，跳过重复重置")
            return True
        
        success = False
        
        # 方法1：标准重置
        cmd = f"nvidia-smi -i {self.gpu_id} -rgc"
        try:
            subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            success = True
        except:
            # 方法2：解锁时钟
            cmd = f"nvidia-smi -i {self.gpu_id} --reset-gpu-clocks"
            try:
                subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
                success = True
            except:
                logger.error("无法重置GPU时钟")
        
        # 重置显存频率（如果启用了显存频率控制）
        memory_success = True
        if success and self.enable_memory_frequency_control and self.memory_frequency_supported:
            cmd = f"nvidia-smi -i {self.gpu_id} -rmc"
            try:
                subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
                logger.info("✅ 显存频率已重置")
            except Exception as e:
                logger.warning(f"显存频率重置失败: {e}")
                memory_success = False
        
        if success:
            logger.info("✅ GPU时钟已重置")
            self.current_freq = self._get_current_frequency()
            # 只有当频率列表不为空时才更新索引
            if self.frequencies:
                self.current_idx = self._freq_to_idx(self.current_freq)
            else:
                self.current_idx = -1  # 表示重置模式，无对应索引
                logger.debug(f"📊 重置模式：频率{self.current_freq}MHz，无对应索引")
            
            # 更新显存频率状态
            if self.enable_memory_frequency_control and self.memory_frequency_supported:
                self.current_memory_freq = self._get_current_memory_frequency()
            
            self.is_clock_reset = True  # 标记为已重置
        
        return success
    
    def _freq_to_idx(self, freq: int) -> int:
        """频率转索引"""
        if not self.frequencies:
            logger.debug(f"📊 频率列表为空，频率 {freq}MHz 无对应索引（重置模式）")
            return -1  # 返回-1表示重置模式
        distances = [abs(f - freq) for f in self.frequencies]
        return np.argmin(distances)
    
    def read_energy_j(self) -> float:
        """读取当前累计能耗 - 线程安全版本"""
        with self._lock:
            return self._get_total_energy_consumption()
    
    
    def _get_total_energy_consumption(self) -> float:
        """获取GPU累计能耗（焦耳）"""
        try:
            energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.nvml_handle)
            return float(energy_mj) / 1000.0  # 转换毫焦为焦耳
        except Exception as e:
            # 某些GPU不支持能耗读取，使用功率估算
            logger.debug(f"无法读取累计能耗: {e}")
            try:
                power = self._get_current_power()
                current_time = time.time()
                if hasattr(self, 'last_energy_timestamp'):
                    time_delta = current_time - self.last_energy_timestamp
                    energy_delta = power * time_delta  # W * s = J
                    if hasattr(self, 'last_energy_j'):
                        return self.last_energy_j + energy_delta
                    else:
                        return energy_delta
                else:
                    return power  # 默认1W * 1s = 1J
            except:
                return 100  # 默认100W * 1s = 100J
    
    def _get_current_power(self) -> float:
        """获取当前功率（瓦特）"""
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)
            return power_mw / 1000.0
        except:
            # 使用nvidia-smi
            result = self._run_nvidia_smi_query("power.draw")
            if result:
                try:
                    return float(result)
                except:
                    pass
            
            logger.debug("无法获取功率，使用默认值100W")
            return 100.0
    
    def get_gpu_stats(self) -> dict:
        """获取GPU状态"""
        stats = {
            'temperature': 0,
            'utilization': 0,
            'memory_used': 0,
            'memory_total': 0,
            'power': 0
        }
        
        try:
            # 使用pynvml
            stats['temperature'] = pynvml.nvmlDeviceGetTemperature(
                self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
            
            util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
            stats['utilization'] = util.gpu
            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
            stats['memory_used'] = mem_info.used / (1024**2)
            stats['memory_total'] = mem_info.total / (1024**2)
            
            stats['power'] = self._get_current_power()
            
        except Exception as e:
            logger.debug(f"pynvml获取状态失败，尝试nvidia-smi: {e}")
            
            # 使用nvidia-smi作为后备
            result = self._run_nvidia_smi_query(
                "temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw")
            if result:
                try:
                    values = result.split(', ')
                    stats['temperature'] = float(values[0])
                    stats['utilization'] = float(values[1])
                    stats['memory_used'] = float(values[2])
                    stats['memory_total'] = float(values[3])
                    stats['power'] = float(values[4])
                except:
                    logger.error("解析GPU状态失败")
        
        return stats
    
    
    def is_frequency_failed(self, freq: int) -> bool:
        """检查频率是否被标记为失败"""
        with self._failed_frequencies_lock:
            return freq in self.failed_frequencies
    
    def reset_failed_frequencies(self):
        """重置失败频率列表"""
        with self._failed_frequencies_lock:
            old_count = len(self.failed_frequencies)
            self.failed_frequencies.clear()
        logger.info(f"🔄 已重置失败频率列表，移除了 {old_count} 个失败频率")
    
    def update_frequency_reward(self, frequency: int, reward: float, 
                              linucb_model=None, current_context=None) -> bool:
        """
        更新频率奖励反馈到自适应采样器
        
        Args:
            frequency: 使用的频率
            reward: 获得的奖励值
            linucb_model: LinUCB模型实例（用于智能细化）
            current_context: 当前上下文特征向量
            
        Returns:
            bool: 是否触发了频率空间细化
        """
        logger.debug(f"🔄 更新频率奖励反馈: {frequency}MHz, 奖励: {reward:.4f}")
        
        if not self.adaptive_sampler:
            logger.error("❌ 自适应采样器未初始化，跳过奖励更新")
            return False
        
        try:
            
            # 向自适应采样器反馈奖励
            self.adaptive_sampler.update_reward_feedback(frequency, reward)
            
            # 检查是否需要细化频率空间（传递LinUCB模型和上下文）
            logger.debug(f"🔍 检查频率空间细化需求 (间隔: {self.adaptive_update_interval})")
            
            try:
                refined = self.adaptive_sampler.refine_frequency_space(
                    min_refinement_interval=self.adaptive_update_interval,
                    linucb_model=linucb_model,
                    current_context=current_context,
                    gpu_controller=self
                )
            except Exception as e:
                logger.error(f"❌ 频率空间细化失败: {e}")
                refined = False
            
            if refined:
                # 更新频率列表
                old_count = len(self.frequencies)
                self.frequencies = self.adaptive_sampler.get_current_frequencies()
                
                # 移除可能的失败频率
                original_count = len(self.frequencies)
                self.frequencies = [f for f in self.frequencies if f not in self.failed_frequencies]
                filtered_count = len(self.frequencies)
                
                logger.info(f"🎯 频率空间细化成功: {old_count} -> {filtered_count}个频率")
                if original_count != filtered_count:
                    logger.info(f"   失败频率过滤: {original_count} -> {filtered_count}个可用频率")
                logger.info(f"   当前频率列表: {self.frequencies}")
                return True
            else:
                logger.debug("📊 频率空间未触发细化")
                
        except Exception as e:
            logger.error(f"❌ 更新频率奖励失败: {e}")
            logger.debug(f"   错误堆栈: {traceback.format_exc()}")
        
        return False
    
    def update_slo_violation(self, violation_action) -> bool:
        """
        报告SLO违规动作到自适应采样器 - 支持组合频率
        
        Args:
            violation_action: 导致SLO违规的动作 (频率或(核心频率, 显存频率)元组)
            
        Returns:
            bool: 是否触发了频率空间更新
        """
        if not self.adaptive_sampler:
            logger.debug("自适应采样器未初始化，跳过SLO违规更新")
            return False
        
        try:
            # 向自适应采样器报告SLO边界（支持组合频率）
            updated = self.adaptive_sampler.update_slo_boundary(violation_action, gpu_controller=self)
            
            if updated:
                # 获取更新后的频率列表
                old_count = len(self.frequencies)
                self.frequencies = self.adaptive_sampler.get_current_frequencies()
                
                # 移除可能的失败频率
                self.frequencies = [f for f in self.frequencies if f not in self.failed_frequencies]
                
                if isinstance(violation_action, tuple):
                    core_freq, memory_freq = violation_action
                    logger.info(f"⚠️ 组合频率SLO违规更新: {core_freq}MHz核心+{memory_freq}MHz显存")
                else:
                    logger.info(f"⚠️ 核心频率SLO违规更新: {violation_action}MHz")
                logger.info(f"   频率空间调整: {old_count} -> {len(self.frequencies)}个频率")
                return True
                
        except Exception as e:
            logger.error(f"更新SLO违规失败: {e}")
        
        return False
    
    def update_memory_slo_boundary(self, memory_freq_boundary: int, core_freq: int) -> bool:
        """
        根据新逻辑处理显存频率SLO违规：
        1. 如果违规显存频率是最大可用显存频率 → 禁用核心频率
        2. 如果不是最大可用显存频率 → 只禁用当前核心频率下的该显存频率及更低频率
        
        Args:
            memory_freq_boundary: 显存频率SLO违规边界
            core_freq: 违规时的核心频率
            
        Returns:
            bool: 是否成功更新了频率空间
        """
        if not self.enable_memory_frequency_control or not self.memory_frequency_supported:
            logger.warning("⚠️ 显存频率控制未启用，无法设置显存SLO边界")
            return False
        
        logger.info(f"🔧 分析显存频率SLO违规: {core_freq}MHz+{memory_freq_boundary}MHz")
        
        # 获取当前可用的显存频率（排除已失败的）
        available_memory_freqs = [f for f in self.memory_frequencies 
                                 if f not in self.failed_memory_frequencies]
        
        if not available_memory_freqs:
            logger.error("❌ 没有可用的显存频率")
            return False
            
        max_available_memory_freq = max(available_memory_freqs)
        
        logger.info(f"📊 可用显存频率: {sorted(available_memory_freqs)}")
        logger.info(f"📊 最大可用显存频率: {max_available_memory_freq}MHz")
        logger.info(f"📊 违规显存频率: {memory_freq_boundary}MHz")
        
        if memory_freq_boundary == max_available_memory_freq:
            # 情况1: 违规显存频率是最大可用显存频率 → 禁用核心频率及更低的核心频率
            logger.warning(f"🚫 违规显存频率{memory_freq_boundary}MHz是最大可用显存频率")
            logger.warning(f"🚫 禁用核心频率{core_freq}MHz及更低频率（无更高显存频率可用，更低核心频率必然也违规）")
            self._notify_core_frequencies_disabled_below_threshold(core_freq, f"显存频率{memory_freq_boundary}MHz SLO违规且无更高频率可用")
            return True
        else:
            # 情况2: 违规显存频率不是最大可用 → 禁用组合并传播SLO边界
            logger.info(f"🔧 违规显存频率{memory_freq_boundary}MHz不是最大可用频率")
            logger.info(f"🔧 执行SLO边界传播: 影响核心频率≤{core_freq}MHz的显存频率边界")
            
            # 传播SLO边界到更小的核心频率
            self._notify_memory_slo_boundary_propagated(core_freq, memory_freq_boundary)
            return True
    
    def get_adaptive_sampling_stats(self) -> dict:
        """获取自适应采样器统计信息"""
        if not self.adaptive_sampler:
            return {"status": "disabled"}
        
        try:
            stats = self.adaptive_sampler.get_sampling_statistics()
            stats["failed_frequencies"] = list(self.failed_frequencies)
            stats["total_frequencies"] = len(self.frequencies)
            stats["available_frequencies"] = len(self.get_available_frequencies())
            return stats
        except Exception as e:
            logger.error(f"获取采样器统计失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def _cleanup(self):
        """清理资源 - 用于atexit"""
        try:
            self.reset_gpu_clocks()
            pynvml.nvmlShutdown()
            logger.debug("GPU控制器已清理")
        except:
            pass
    
    def __del__(self):
        """清理资源"""
        self._cleanup()