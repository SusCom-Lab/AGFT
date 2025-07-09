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

        energy_samples: List[float] = []
        if energy_reader:
            try:
                energy_samples.append(float(energy_reader()))
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("energy_reader 调用失败: %s", exc)
                energy_reader = None  # disable further use

        # ------------------------------------------------------------------
        # Repeated sampling within the window
        # ------------------------------------------------------------------
        while time.time() - start_ts < duration:
            time.sleep(interval)
            snap = self._fetch_metrics_once()
            if snap:
                gauge_samples.append(snap)
            if energy_reader:
                energy_samples.append(energy_reader())

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
        # 3. Energy delta (mJ) - simple first/last difference
        # ------------------------------------------------------------------
        energy_delta_mj: float = 0.0
        if energy_reader and len(energy_samples) >= 2:
            energy_delta_mj = energy_samples[-1] - energy_samples[0]
            if energy_delta_mj < 0:  # NVML rollover guard
                energy_delta_mj = 0.0

        logger.info(
            "✅ 优化数据采集完成: %d个Gauge, %d个Counter增量, ΔEnergy=%.1fmJ",
            len(gauge_metrics), len(counter_deltas), energy_delta_mj,
        )

        return gauge_metrics, counter_deltas, energy_delta_mj
    
    
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
            combined_metrics['energy_delta_mj'] = energy_delta
            
            return combined_metrics
            
        except Exception as exc:
            logger.error("标准指标收集失败: %s", exc)
            return {}
