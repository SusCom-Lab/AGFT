import subprocess
import numpy as np
from typing import List, Tuple, Optional, Dict
import pynvml
from .logger import setup_logger
import time
import re

logger = setup_logger(__name__)

class GPUController:
    """兼容性增强的GPU频率控制器"""
    
    def __init__(self, gpu_id: int = 0, min_freq: int = 1005, step: int = 90):
        self.gpu_id = gpu_id
        self.min_freq = min_freq
        self.step = step
        
        # 记录上一次成功设置的频率
        self.last_successful_freq = None
        
        # 频率映射缓存
        self.freq_mapping_cache: Dict[int, int] = {}
        
        # 初始化pynvml
        try:
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            logger.info("✅ NVML初始化成功")
        except Exception as e:
            logger.error(f"❌ NVML初始化失败: {e}")
            raise
        
        # 获取GPU信息
        self._get_gpu_info()
        
        # 获取频率范围（使用多种方法）
        self.min_supported, self.max_supported = self._get_frequency_limits_safe()
        self.max_freq = min(self._get_max_frequency_safe(), self.max_supported)
        
        # 如果无法获取，使用默认值
        if self.max_freq <= 0:
            logger.warning("无法获取最大频率，使用默认值")
            self.max_freq = 2100
            self.max_supported = 2100
        
        logger.info(f"🎮 GPU {gpu_id} 频率范围: {self.min_supported}-{self.max_freq}MHz")
        
        # 生成频率列表
        self.frequencies = self._generate_frequency_list()
        logger.info(f"📊 频率列表: {self.frequencies} MHz")
        logger.info(f"🎯 共 {len(self.frequencies)} 个频率档位")
        
        # 获取当前频率
        self.current_freq = self._get_current_frequency()
        self.current_idx = self._freq_to_idx(self.current_freq)
        self.last_successful_freq = self.current_freq
        logger.info(f"📍 当前频率: {self.current_freq}MHz (索引: {self.current_idx})")
        
        # 初始能耗
        self.last_energy_mj = self._get_total_energy_consumption()
        logger.info(f"⚡ 初始能耗读数: {self.last_energy_mj:.1f}mJ")
    
    def _get_gpu_info(self):
        """获取GPU详细信息"""
        try:
            gpu_name = pynvml.nvmlDeviceGetName(self.nvml_handle).decode('utf-8')
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
            total_mem_gb = mem_info.total / (1024**3)
            
            # 获取驱动版本
            driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            
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
    
    def _run_nvidia_smi_query(self, query_string: str) -> Optional[str]:
        """安全运行nvidia-smi查询"""
        cmd = f"nvidia-smi -i {self.gpu_id} --query-gpu={query_string} --format=csv,noheader,nounits"
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.debug(f"nvidia-smi查询失败 ({query_string}): {e}")
            return None
    
    def _get_frequency_limits_safe(self) -> Tuple[int, int]:
        """安全获取GPU频率限制"""
        # 方法1：尝试标准查询
        result = self._run_nvidia_smi_query("clocks.min.graphics,clocks.max.graphics")
        if result:
            try:
                values = result.split(', ')
                return int(values[0]), int(values[1])
            except:
                pass
        
        # 方法2：通过nvidia-smi -q获取详细信息
        try:
            cmd = f"nvidia-smi -i {self.gpu_id} -q -d CLOCK"
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            output = result.stdout
            
            min_freq = 300  # 默认最小值
            max_freq = 2100  # 默认最大值
            
            # 解析输出
            for line in output.split('\n'):
                if 'Graphics' in line and 'MHz' in line:
                    # 查找类似 "Graphics : 1980 MHz" 的行
                    match = re.search(r'(\d+)\s*MHz', line)
                    if match:
                        freq = int(match.group(1))
                        if freq > 500:  # 假设大于500的是最大频率
                            max_freq = freq
                        elif freq < 500:  # 小于500的可能是最小频率
                            min_freq = freq
            
            logger.info(f"通过详细查询获取频率范围: {min_freq}-{max_freq}MHz")
            return min_freq, max_freq
            
        except Exception as e:
            logger.warning(f"详细查询失败: {e}")
        
        # 方法3：使用pynvml
        try:
            # 尝试获取支持的时钟频率
            mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(self.nvml_handle)
            if mem_clocks:
                # 对于每个内存时钟，获取支持的图形时钟
                graphics_clocks = []
                for mem_clock in mem_clocks[:1]:  # 只检查第一个内存时钟
                    try:
                        gfx_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(
                            self.nvml_handle, mem_clock)
                        graphics_clocks.extend(gfx_clocks)
                    except:
                        pass
                
                if graphics_clocks:
                    min_freq = min(graphics_clocks)
                    max_freq = max(graphics_clocks)
                    logger.info(f"通过pynvml获取频率范围: {min_freq}-{max_freq}MHz")
                    return min_freq, max_freq
        except Exception as e:
            logger.debug(f"pynvml获取频率失败: {e}")
        
        # 使用默认值
        logger.warning("无法获取频率限制，使用默认值: 300-2100MHz")
        return 300, 2100
    
    def _get_max_frequency_safe(self) -> int:
        # """安全获取最大频率"""
        # # 方法1：标准查询
        # result = self._run_nvidia_smi_query("clocks.max.graphics")
        # if result:
        #     try:
        #         return int(result)
        #     except:
        #         pass
        
        # # 方法2：查询当前最大时钟
        # result = self._run_nvidia_smi_query("clocks.max.gr")
        # if result:
        #     try:
        #         return int(result)
        #     except:
        #         pass
        
        # # 方法3：通过应用时钟获取
        # try:
        #     cmd = f"nvidia-smi -i {self.gpu_id} -q -d SUPPORTED_CLOCKS"
        #     result = subprocess.run(cmd.split(), capture_output=True, text=True)
        #     if result.returncode == 0:
        #         # 查找最大的图形时钟
        #         max_clock = 0
        #         for line in result.stdout.split('\n'):
        #             if 'Graphics' in line:
        #                 match = re.search(r'(\d+)\s*MHz', line)
        #                 if match:
        #                     clock = int(match.group(1))
        #                     max_clock = max(max_clock, clock)
                
        #         if max_clock > 0:
        #             logger.info(f"从支持的时钟列表获取最大频率: {max_clock}MHz")
        #             return max_clock
        # except:
        #     pass
        
        # # 使用默认值
        # logger.warning("无法获取最大频率，使用默认值2100MHz")
        return 1605
    
    def _generate_frequency_list(self) -> List[int]:
        """生成频率列表"""
        # 如果能获取支持的频率列表，使用它
        try:
            mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(self.nvml_handle)
            if mem_clocks:
                graphics_clocks = []
                for mem_clock in mem_clocks[:1]:  # 使用第一个内存时钟
                    try:
                        gfx_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(
                            self.nvml_handle, mem_clock)
                        graphics_clocks.extend(gfx_clocks)
                    except:
                        pass
                
                if graphics_clocks:
                    # 过滤并排序
                    graphics_clocks = sorted(set(graphics_clocks))
                    # 只保留在我们范围内的频率
                    valid_clocks = [f for f in graphics_clocks 
                                   if self.min_freq <= f <= self.max_freq]
                    valid_clocks = [f for f in valid_clocks if (f - self.min_freq) % self.step == 0]
                    if len(valid_clocks) >= 5:
                        logger.info("使用GPU支持的实际频率点")
                        return valid_clocks
        except:
            pass
        
        # 否则生成理论频率列表
        frequencies = list(range(self.min_freq, self.max_freq + 1, self.step))
        if frequencies[-1] != self.max_freq:
            frequencies.append(self.max_freq)
        
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
                    return int(result)
                except:
                    pass
        
        # 如果都失败了，使用pynvml
        try:
            clock_info = pynvml.nvmlDeviceGetClockInfo(self.nvml_handle, 
                                                      pynvml.NVML_CLOCK_GRAPHICS)
            return clock_info
        except:
            pass
        
        logger.error("无法获取当前频率，使用默认值")
        return self.frequencies[len(self.frequencies)//2] if hasattr(self, 'frequencies') else 1500
    
    def set_frequency(self, freq_mhz: int) -> bool:
        """设置GPU频率"""
        # 查找最接近的可用频率
        closest_freq = min(self.frequencies, key=lambda x: abs(x - freq_mhz))
        
        # 尝试设置
        success = False
        
        # 方法1：直接设置
        cmd = f"nvidia-smi -i {self.gpu_id} -lgc {closest_freq}"
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            success = True
        except subprocess.CalledProcessError as e:
            logger.debug(f"直接设置失败: {e}")
            
            # 方法2：使用锁定时钟
            cmd = f"nvidia-smi -i {self.gpu_id} --lock-gpu-clocks={closest_freq},{closest_freq}"
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
                success = True
                logger.info("使用--lock-gpu-clocks成功")
            except:
                logger.debug("锁定时钟也失败")
        
        if success:
            # 等待一下让设置生效
            time.sleep(0.3)
            
            # 验证实际频率
            actual_freq = self._get_current_frequency()
            self.current_freq = actual_freq
            self.current_idx = self._freq_to_idx(actual_freq)
            self.last_successful_freq = actual_freq
            
            # 记录映射
            self.freq_mapping_cache[freq_mhz] = actual_freq
            
            tolerance = 50  # MHz，可按机型调
            if abs(actual_freq - freq_mhz) > tolerance:
                logger.warning(
                    f"❌ 频率偏差过大(>{tolerance}MHz)，视为设置失败: "
                    f"{freq_mhz} → {actual_freq}"
                )
                # 回滚 internal state，告诉调用方失败
                self._get_current_frequency()  # 再刷一次确保 current_freq 正确
                return False

            logger.info(
                f"✅ GPU频率设置成功: {freq_mhz}MHz → {actual_freq}MHz "
                f"(Δ{actual_freq - freq_mhz:+d}MHz)"
            )

            return True
        
        # 设置失败，尝试回退
        if self.last_successful_freq:
            logger.warning(f"频率设置失败，保持当前频率: {self.current_freq}MHz")
        
        return False
    
    def set_frequency_by_index(self, idx: int) -> bool:
        """通过索引设置频率"""
        if 0 <= idx < len(self.frequencies):
            return self.set_frequency(self.frequencies[idx])
        else:
            logger.error(f"无效的频率索引: {idx}")
            return False
    
    def reset_gpu_clocks(self) -> bool:
        """重置GPU时钟"""
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
        
        if success:
            logger.info("✅ GPU时钟已重置")
            self.current_freq = self._get_current_frequency()
            self.current_idx = self._freq_to_idx(self.current_freq)
            self.last_successful_freq = self.current_freq
        
        return success
    
    def _freq_to_idx(self, freq: int) -> int:
        """频率转索引"""
        distances = [abs(f - freq) for f in self.frequencies]
        return np.argmin(distances)
    
    def read_energy_mj(self) -> float:
        """读取当前累计能耗"""
        return self._get_total_energy_consumption()
    
    def _get_total_energy_consumption(self) -> float:
        """获取GPU累计能耗（毫焦）"""
        try:
            energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.nvml_handle)
            return float(energy_mj)
        except Exception as e:
            # 某些GPU不支持能耗读取，使用功率估算
            logger.debug(f"无法读取累计能耗: {e}")
            try:
                power = self._get_current_power()
                if hasattr(self, 'last_energy_mj'):
                    return self.last_energy_mj + power * 2000
                else:
                    return power * 2000
            except:
                return 200000  # 默认100W * 2s
    
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
    
    def get_energy_delta(self) -> float:
        """获取能耗增量"""
        try:
            current_energy_mj = self._get_total_energy_consumption()
            
            if hasattr(self, 'last_energy_mj') and self.last_energy_mj is not None:
                delta = current_energy_mj - self.last_energy_mj
                
                if delta < 0:
                    logger.warning("能耗读数回绕或不支持，使用功率估算")
                    power = self._get_current_power()
                    delta = power * 2000
                elif delta > 2000000:  # 超过2000J
                    logger.warning(f"能耗增量异常: {delta/1000:.1f}J，使用功率估算")
                    power = self._get_current_power()
                    delta = power * 2000
            else:
                delta = 0
            
            self.last_energy_mj = current_energy_mj
            return delta
            
        except Exception as e:
            logger.error(f"计算能耗增量失败: {e}")
            return 200000  # 默认100W * 2s
    
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
    
    def __del__(self):
        """清理资源"""
        try:
            self.reset_gpu_clocks()
            pynvml.nvmlShutdown()
            logger.debug("GPU控制器已清理")
        except:
            pass