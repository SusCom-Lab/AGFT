import requests
import time
from typing import Dict, List, Tuple, Callable, Optional
from collections import defaultdict

from .logger import setup_logger

logger = setup_logger(__name__)


class MetricsCollector:
    """Prometheus‑based vLLM metrics collector - 优化采样窗口与决策间隔同步
    including aligned GPU energy delta measurement.
    使用requests.Session()保持连接复用，避免TIME_WAIT积累。
    """

    GAUGE_NAMES = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:gpu_cache_usage_perc",
        "vllm:cpu_cache_usage_perc",
    ]

    COUNTER_NAMES = [
        "vllm:num_preemptions_total",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:time_to_first_token_seconds_count",
        "vllm:time_per_output_token_seconds_sum",
        "vllm:time_per_output_token_seconds_count",
        "vllm:e2e_request_latency_seconds_sum",
        "vllm:e2e_request_latency_seconds_count",
        "vllm:gpu_prefix_cache_hits_total",
        "vllm:gpu_prefix_cache_queries_total",
        "vllm:iteration_tokens_total_sum",
        "vllm:iteration_tokens_total_count",
    ]

    def __init__(self, prometheus_url: str = "http://127.0.0.1:8000/metrics", 
                 *, ema_alpha: float = 0.3, 
                 sampling_duration: float = 2.0,
                 sampling_interval: float = 0.2):
        self.prometheus_url = prometheus_url
        self.ema_alpha = float(ema_alpha)
        self.sampling_duration = sampling_duration
        self.sampling_interval = sampling_interval
        
        # 使用Session保持连接复用，避免TIME_WAIT积累
        self._session = requests.Session()
        self._session.headers.update({'Connection': 'keep-alive'})
        
        # running EMA state (metric_name -> value)
        self._gauge_ema: Dict[str, float] = {}
        
        logger.info("📡 初始化优化指标采集器:")
        logger.info(f"   Prometheus URL: {prometheus_url}")
        logger.info(f"   标准采样: {sampling_duration}s窗口, {sampling_interval}s间隔")
        logger.info(f"   EMA α: {ema_alpha}")
        logger.info("   🔗 使用Session连接复用，减少TIME_WAIT")

    # ---------------------------------------------------------------------
    # Low‑level helpers
    # ---------------------------------------------------------------------
    def _parse_prometheus_metrics(self, text: str) -> Dict[str, float]:
        """Parse a Prometheus exposition‑format text block into a flat mapping
        of metric_name -> value. Only keeps keys that start with "vllm:"."""
        metrics: Dict[str, float] = {}
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue  # ignore comments / empty lines
            try:
                if "{" in line:  # metric with labels – drop labels
                    metric_name = line.split("{")[0]
                    value = float(line.split("}")[1].strip())
                else:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    metric_name, value = parts[0], float(parts[1])

                if metric_name.startswith("vllm:"):
                    metrics[metric_name] = value
            except ValueError as exc:
                logger.debug("解析行失败 '%s': %s", line, exc)
        return metrics

    def _fetch_metrics_once(self) -> Dict[str, float]:
        """Fetch one snapshot from Prometheus using session for connection reuse."""
        try:
            resp = self._session.get(self.prometheus_url, timeout=1.0)
            resp.raise_for_status()
            return self._parse_prometheus_metrics(resp.text)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("获取指标失败: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def collect_optimized_metrics(
        self,
        *,
        duration: Optional[float] = None,
        interval: Optional[float] = None,
        energy_reader: Optional[Callable[[], float]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float], float]:
        """收集优化的指标数据 - 基于vLLM服务特性的科学采样
        
        Args:
            duration: 采样窗口长度(秒)，若为None则使用sampling_duration
            interval: 采样周期(秒)，若为None则使用sampling_interval
            energy_reader: GPU能耗读取函数，返回累计能耗(mJ)
            
        Returns:
            gauge_metrics: 仪表指标 -> EMA平滑值
            counter_deltas: 计数器增量 -> 首尾相减总增量
            energy_delta_mj: 窗口内GPU能耗增量(毫焦)
        """
        # 使用配置的采样参数
        if duration is None:
            duration = self.sampling_duration
        if interval is None:
            interval = self.sampling_interval
            
        logger.info("📊 开始优化数据采集 (窗口=%.1fs)…", duration)

        start_ts = time.time()

        # ------------------------------------------------------------------
        # Initialize sampling
        # ------------------------------------------------------------------
        gauge_samples: List[Dict[str, float]] = []

        # 改进的能耗测量：使用瞬时功率采样
        power_samples: List[float] = []  # 只保存功率值
        if energy_reader:
            try:
                # 获取GPU控制器实例来读取瞬时功率
                gpu_controller = getattr(energy_reader, '__self__', None)
                if gpu_controller and hasattr(gpu_controller, '_get_current_power'):
                    initial_power = float(gpu_controller._get_current_power())
                    power_samples.append(initial_power)
                else:
                    logger.warning("无法获取功率读取接口")
                    energy_reader = None
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("功率读取初始化失败: %s", exc)
                energy_reader = None  # disable further use

        # ------------------------------------------------------------------
        # Repeated sampling within the window
        # ------------------------------------------------------------------
        while time.time() - start_ts < duration:
            time.sleep(interval)
            current_ts = time.time()
            snap = self._fetch_metrics_once()
            if snap:
                gauge_samples.append(snap)
            if energy_reader:
                try:
                    # 重新获取GPU控制器实例
                    gpu_controller = getattr(energy_reader, '__self__', None)
                    if gpu_controller and hasattr(gpu_controller, '_get_current_power'):
                        current_power = float(gpu_controller._get_current_power())
                        power_samples.append(current_power)
                except Exception as exc:
                    logger.debug("采样中功率读取失败: %s", exc)

        # 确保采样窗口结束时的精确测量
        if energy_reader and len(power_samples) > 0:
            try:
                # 重新获取GPU控制器实例
                gpu_controller = getattr(energy_reader, '__self__', None)
                if gpu_controller and hasattr(gpu_controller, '_get_current_power'):
                    final_power = float(gpu_controller._get_current_power())
                    power_samples.append(final_power)
            except Exception as exc:
                logger.debug("结束时功率读取失败: %s", exc)

        # ------------------------------------------------------------------
        # 1. Gauge EMA over samples
        # ------------------------------------------------------------------
        gauge_metrics: Dict[str, float] = {}
        for name in self.GAUGE_NAMES:
            values = [s.get(name, 0.0) for s in gauge_samples if name in s]
            if not values:
                continue
            # incremental EMA over the sample list
            ema = values[0]
            for v in values[1:]:
                ema = self.ema_alpha * v + (1.0 - self.ema_alpha) * ema
            gauge_metrics[name] = ema

        # ------------------------------------------------------------------
        # 2. Counter deltas - simple first/last difference (like Energy)
        # ------------------------------------------------------------------
        counter_deltas: Dict[str, float] = {}
        for name in self.COUNTER_NAMES:
            # 从所有样本中提取该计数器的值
            counter_values = [s.get(name, 0.0) for s in gauge_samples if name in s]
            if len(counter_values) < 2:
                continue
                
            # 简单的首尾相减，避免采样间隔不均匀的影响
            delta = counter_values[-1] - counter_values[0]
            if delta >= 0:  # 过滤掉计数器重置的异常值
                counter_deltas[f"{name}_delta"] = delta

        # ------------------------------------------------------------------
        # 3. 功率积分能耗测量 - 使用配置的采样窗口时间
        # ------------------------------------------------------------------
        energy_delta_j: float = 0.0
        if energy_reader and len(power_samples) >= 1:
            logger.info(f"📊 功率采样统计: 总采样点={len(power_samples)}, 配置窗口={duration:.3f}s")
            
            # 使用平均功率×采样窗口时间计算能耗
            energy_delta_j = self._calculate_energy_from_power_integration(
                power_samples, duration
            )

        logger.info(
            "✅ 优化数据采集完成: %d个Gauge, %d个Counter增量, ΔEnergy=%.3fJ (配置窗口=%.3fs)",
            len(gauge_metrics), len(counter_deltas), energy_delta_j, duration,
        )

        return gauge_metrics, counter_deltas, energy_delta_j
    
    
    def collect_metrics(self, energy_reader: Optional[Callable[[], float]] = None) -> Dict[str, float]:
        """标准指标收集方法 - 兼容性适配器
        
        Args:
            energy_reader: GPU能耗读取函数，可选
        
        Returns:
            合并的指标字典，包括gauge和counter增量以及能耗数据
        """
        try:
            gauge_metrics, counter_deltas, energy_delta = self.collect_optimized_metrics(
                energy_reader=energy_reader
            )
            
            # 合并所有指标到一个字典中
            combined_metrics = {}
            combined_metrics.update(gauge_metrics)
            combined_metrics.update(counter_deltas)
            combined_metrics['energy_delta_j'] = energy_delta
            
            return combined_metrics
            
        except Exception as exc:
            logger.error("标准指标收集失败: %s", exc)
            return {}
    
    def _extract_unique_energy_updates(self, timestamps: List[float], energies: List[float]) -> Tuple[List[float], List[float]]:
        """提取真实的能耗更新点，去除重复值"""
        if len(energies) < 2:
            return timestamps, energies
        
        # 只保留能耗发生变化的采样点
        unique_timestamps = [timestamps[0]]  # 总是保留第一个点
        unique_energies = [energies[0]]
        
        for i in range(1, len(energies)):
            # 只保留能耗值发生变化的采样点（大于0.1mJ的变化）
            if abs(energies[i] - energies[i-1]) > 0.1:
                unique_timestamps.append(timestamps[i])
                unique_energies.append(energies[i])
        
        # 确保包含最后一个点（用于计算总时间窗口）
        if len(unique_energies) == 1 or abs(energies[-1] - unique_energies[-1]) > 0.1:
            unique_timestamps.append(timestamps[-1])
            unique_energies.append(energies[-1])
        
        logger.info(f"🧹 能耗去重: 原始{len(energies)}个样本 -> 有效更新{len(unique_energies)}个样本")
        
        return unique_timestamps, unique_energies
    
    def _calculate_energy_from_power_integration(self, power_samples: List[float], target_duration: float) -> float:
        """使用平均功率×采样窗口时间计算能耗 - 与基线测量方法统一"""
        if len(power_samples) < 1:
            return 0.0
        
        import numpy as np
        
        # 计算平均功率
        avg_power = np.mean(power_samples)
        
        # 使用平均功率 × 采样窗口时间计算能耗（直接返回焦耳）
        energy_joules = avg_power * target_duration  # W × s = J
        
        logger.info(
            f"⚡ 平均功率能耗计算: 平均功率={avg_power:.1f}W, "
            f"采样窗口={target_duration:.3f}s, 能耗={energy_joules:.3f}J"
        )
        
        return max(0.0, energy_joules)  # 确保非负值
    
    def _calculate_energy_rate_linear_regression(self, timestamps: List[float], energies: List[float]) -> float:
        """使用线性回归计算能耗率 (J/s)"""
        import numpy as np
        
        # 转换为numpy数组
        t = np.array(timestamps)
        e = np.array(energies)
        
        # 时间归一化（从0开始）
        t_norm = t - t[0]
        
        # 使用numpy的线性回归
        # 斜率 = 协方差(t,e) / 方差(t)
        n = len(t_norm)
        if n < 2:
            return 0.0
            
        # 简单线性回归公式
        sum_t = np.sum(t_norm)
        sum_e = np.sum(e)
        sum_te = np.sum(t_norm * e)
        sum_t2 = np.sum(t_norm * t_norm)
        
        denominator = n * sum_t2 - sum_t * sum_t
        if abs(denominator) < 1e-10:
            # 避免除零，回退到简单斜率
            return (e[-1] - e[0]) / (t_norm[-1] - t_norm[0]) if t_norm[-1] > t_norm[0] else 0.0
        
        slope = (n * sum_te - sum_t * sum_e) / denominator
        return slope
    
    def _calculate_r_squared(self, timestamps: List[float], energies: List[float], energy_rate: float) -> float:
        """计算线性回归的R²值"""
        import numpy as np
        
        if len(energies) < 2:
            return 0.0
            
        # 转换为numpy数组
        t = np.array(timestamps)
        e = np.array(energies)
        
        # 时间归一化
        t_norm = t - t[0]
        
        # 计算截距
        intercept = np.mean(e) - energy_rate * np.mean(t_norm)
        
        # 预测值
        e_pred = intercept + energy_rate * t_norm
        
        # 计算R²
        ss_res = np.sum((e - e_pred) ** 2)  # 残差平方和
        ss_tot = np.sum((e - np.mean(e)) ** 2)  # 总平方和
        
        if abs(ss_tot) < 1e-10:
            return 1.0  # 如果数据完全无变化，认为拟合完美
        
        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, r_squared)  # 确保R²不为负
