import requests
import time
from typing import Dict, List, Tuple, Callable, Optional
from collections import defaultdict

from .logger import setup_logger

logger = setup_logger(__name__)


class MetricsCollector:
    """Prometheus‑based vLLM metrics collector with 2‑second sliding‑window support
    including aligned GPU energy delta measurement.
    """

    GAUGE_NAMES = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:num_requests_swapped",
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
    ]

    def __init__(self, prometheus_url: str = "http://localhost:8000/metrics", *, ema_alpha: float = 0.3):
        self.prometheus_url = prometheus_url
        self.ema_alpha = float(ema_alpha)
        # running EMA state (metric_name -> value)
        self._gauge_ema: Dict[str, float] = {}
        logger.info("📡 初始化指标采集器: %s", prometheus_url)

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
        """Fetch one snapshot from Prometheus."""
        try:
            resp = requests.get(self.prometheus_url, timeout=1.0)
            resp.raise_for_status()
            return self._parse_prometheus_metrics(resp.text)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("获取指标失败: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def collect_2s_metrics(
        self,
        *,
        duration: float = 2.0,
        interval: float = 0.1,
        energy_reader: Optional[Callable[[], float]] = None,
        ignore_first: int = 1,
    ) -> Tuple[Dict[str, float], Dict[str, float], float]:
        """Collect gauges (EMA), counter deltas, **and** GPU energy consumed (mJ)
        over a ~`duration`‑second sliding window.

        Args:
            duration: total window length in seconds (default 2.0).
            interval: sampling period for gauges / energy (default 0.1).
            energy_reader: callable returning cumulative GPU energy (mJ). If
                ``None`` energy delta is reported as 0.0.
            ignore_first: how many leading energy samples to drop when computing
                the delta – used to compensate for lock‑frequency inertia.

        Returns:
            gauge_metrics: metric -> ema value
            counter_deltas: metric_delta -> delta value
            energy_delta_mj: energy consumed within the window in millijoules
        """
        logger.info("📊 开始 %.1fs 数据采集…", duration)

        start_ts = time.time()

        # ------------------------------------------------------------------
        # Snapshot #0   (also first gauge sample / counter baseline)
        # ------------------------------------------------------------------
        counter_start = self._fetch_metrics_once()
        gauge_samples: List[Dict[str, float]] = [counter_start] if counter_start else []

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
        # Final counter snapshot (use last collected gauge if available)
        # ------------------------------------------------------------------
        counter_end = gauge_samples[-1] if gauge_samples else self._fetch_metrics_once()

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
        # 2. Counter deltas
        # ------------------------------------------------------------------
        counter_deltas: Dict[str, float] = {}
        for name in self.COUNTER_NAMES:
            if name in counter_start and name in counter_end:
                delta = counter_end[name] - counter_start[name]
                counter_deltas[f"{name}_delta"] = max(delta, 0.0)

        # ------------------------------------------------------------------
        # 3. Energy delta (mJ)
        # ------------------------------------------------------------------
        energy_delta_mj: float = 0.0
        if energy_reader and len(energy_samples) >= 2:
            start_idx = min(ignore_first, len(energy_samples) - 1)
            energy_delta_mj = energy_samples[-1] - energy_samples[start_idx]
            if energy_delta_mj < 0:  # NVML rollover guard
                energy_delta_mj = 0.0

        logger.info(
            "✅ 数据采集完成: %d 个 Gauge, %d 个 Counter 增量, ΔEnergy=%.2f mJ",
            len(gauge_metrics), len(counter_deltas), energy_delta_mj,
        )

        return gauge_metrics, counter_deltas, energy_delta_mj
