import numpy as np
from typing import Dict, List, Optional
from .logger import setup_logger

logger = setup_logger(__name__)

class FeatureExtractor:
    """特征提取器 - 增强版，包含GPU硬件特征"""
    
    def __init__(self, gpu_controller=None):
        """初始化特征提取器 - 支持GPU硬件特征"""
        # 定义基础特征名称列表
        self.feature_names = [
            'has_queue', 
            'prefill_throughput', 
            'decode_throughput', 
            'packing_efficiency',
            'concurrency', 
            'gpu_cache_usage', 
            'cache_hit_rate'
            # 注意：频率不作为环境特征，而是在LinUCB中作为连续动作特征处理
        ]
        
        # 自动计算特征数量
        self.n_features = len(self.feature_names)
        
        # GPU控制器引用（用于获取硬件特征）
        self.gpu_controller = gpu_controller
        
        # 在线特征标准化
        self.mean = np.zeros(self.n_features)
        self.M2 = np.zeros(self.n_features)
        self.count = 0
        
        logger.info(f"🔧 初始化增强特征提取器: {self.n_features} 维特征")
        logger.info(f"   环境特征: {', '.join(self.feature_names)} (7维)")
        logger.info(f"   GPU控制器: {'已连接' if gpu_controller else '未连接'}")
        
    def extract(self, gauge_metrics: Dict[str, float], 
                counter_deltas: Dict[str, float]) -> np.ndarray:
        """提取特征向量"""
        features = []
        
        # 获取请求数量
        num_running = gauge_metrics.get('vllm:num_requests_running', 0)
        num_waiting = gauge_metrics.get('vllm:num_requests_waiting', 0)
        
        # 1. 队列状态（二值）
        has_queue = 1.0 if num_waiting > 0 else 0.0
        features.append(has_queue)
        
        # 2. Prefill吞吐（tokens/2秒）
        prefill_delta = counter_deltas.get('vllm:prompt_tokens_total_delta', 0)
        prefill_throughput = max(prefill_delta, 1e-3)
        features.append(prefill_throughput)
        
        # 3. Decode吞吐（tokens/2秒）
        decode_delta = counter_deltas.get('vllm:generation_tokens_total_delta', 0)
        decode_throughput = max(decode_delta, 1e-3)
        features.append(decode_throughput)

        
        # 4. Packing效率
        delta_sum = counter_deltas.get('vllm:iteration_tokens_total_sum_delta', 0.0)
        delta_count = counter_deltas.get('vllm:iteration_tokens_total_count_delta', 0.0)
        packing_efficiency = delta_sum / delta_count if delta_count > 0 else 0.0
        features.append(packing_efficiency)
        
        # 5. 并发度
        concurrency = num_running
        features.append(concurrency)
        
        # 6. GPU缓存使用率
        gpu_cache = gauge_metrics.get('vllm:gpu_cache_usage_perc', 0) * 100  # 转换为百分比
        features.append(gpu_cache)
        
        # 7. 缓存命中率
        hits = counter_deltas.get('vllm:gpu_prefix_cache_hits_total_delta', 0)
        queries = counter_deltas.get('vllm:gpu_prefix_cache_queries_total_delta', 1)
        cache_hit_rate = hits / max(queries, 1) *100
        features.append(cache_hit_rate)
        
        # 确保特征数量与定义一致
        assert len(features) == self.n_features, \
            f"特征数量不匹配: {len(features)} vs {self.n_features}"
        
        # 日志特征值
        logger.debug("📈 原始特征:")
        for name, value in zip(self.feature_names, features):
            logger.debug(f"  {name}: {value:.3f}")
            
        # 返回numpy数组
        return np.asarray(features, dtype=np.float32)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """在线特征标准化（Welford算法）"""
        # 类型检查和转换
        features = np.asarray(features, dtype=np.float32)
        
        # 维度检查
        assert features.shape[0] == self.n_features, \
            f"特征维度不匹配: {features.shape[0]} vs {self.n_features}"
        
        # 更新统计量（Welford算法）
        self.count += 1
        delta = features - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (features - self.mean)
        
        # 计算标准差
        if self.count > 1:
            std = np.sqrt(self.M2 / (self.count - 1))
        else:
            std = np.ones(self.n_features)
            
        # 标准化
        normalized = (features - self.mean) / (std + 1e-8)
        
        # 截断到[-3, 3]
        normalized = np.clip(normalized, -3, 3)
        
        # 日志
        logger.debug("📊 标准化特征:")
        for name, value in zip(self.feature_names, normalized):
            logger.debug(f"  {name}: {value:.3f}")
            
        return normalized
    
    def extract_features(self, metrics: Dict[str, float]) -> np.ndarray:
        """适配器方法，用于与main.py的调用方式兼容"""
        # 将合并的metrics分离为gauge和counter
        gauge_metrics = {}
        counter_deltas = {}
        
        # 分离不同类型的指标
        for key, value in metrics.items():
            if key.endswith('_delta'):
                counter_deltas[key] = value
            elif key.startswith('vllm:'):
                gauge_metrics[key] = value
        
        # 调用原始的extract方法获取原始特征
        raw_features = self.extract(gauge_metrics, counter_deltas)
        
        # 应用在线标准化
        normalized_features = self.normalize(raw_features)
        
        return normalized_features