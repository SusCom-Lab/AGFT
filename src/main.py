import time
import yaml
import signal
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from .logger import setup_logger
from .metrics_collector import MetricsCollector
from .gpu_controller import GPUController
from .feature_extractor import FeatureExtractor
from .linucb import LinUCB
from .reward_calculator import EDPRewardCalculator

# 设置主日志
logger = setup_logger(__name__, f"vllm_gpu_autoscaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

class VLLMGPUAutoscaler:
    """vLLM GPU自适应调频主控制器"""
    
    def __init__(self, config_path: str = "config/config.yaml", 
                 reset_model: bool = False, 
                 model_file: Optional[str] = None,
                 no_learn: bool = False):
        logger.info("="*60)
        logger.info("🚀 vLLM GPU自适应调频系统 v2.0 (Hybrid LinUCB)")
        logger.info("="*60)
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 初始化组件
        logger.info("🔧 初始化系统组件...")
        
        # 1. Prometheus指标采集器
        self.metrics_collector = MetricsCollector(
            self.config['vllm']['prometheus_url']
        )
        
        # 2. GPU控制器
        self.gpu_controller = GPUController(
            gpu_id=self.config['gpu']['device_id'],
            min_freq=self.config['gpu']['min_frequency'],
            step=self.config['gpu']['frequency_step'],
        )
        
        # 3. 特征提取器（自动计算特征数量）
        self.feature_extractor = FeatureExtractor()

        self.no_learn = no_learn
        if self.no_learn:
            logger.warning("🚫 仅推断模式开启：LinUCB 将不会更新！")

   

        
        # 4. LinUCB模型（Hybrid版本）
        model_dir = self.config.get('model', {}).get('save_dir', 'data/models')
        self.linucb = LinUCB(
            n_features=self.feature_extractor.n_features,  # 使用提取器的特征数量
            n_actions=len(self.gpu_controller.frequencies),
            alpha=self.config['linucb']['initial_alpha'],
            lambda_reg=self.config['linucb'].get('lambda_reg', 0.1),
            model_dir=model_dir,
            auto_load=not reset_model  # 如果reset_model=True，则不自动加载
        )
        
        # 处理模型加载选项
        if reset_model:
            logger.info("🔄 重置模型，从头开始学习")
        elif model_file:
            logger.info(f"📂 加载指定模型: {model_file}")
            if not self.linucb.load_model(model_file):
                logger.error("加载模型失败，将从头开始学习")
        else:
            if self.linucb.total_rounds > 0:
                logger.info(f"✅ 继续上次的学习进度 (已完成 {self.linucb.total_rounds} 轮)")
            else:
                logger.info("📝 开始全新的学习过程")
                
        # 5. EDP奖励计算器
        self.reward_calculator = EDPRewardCalculator(
            ttft_limit=self.config['control']['ttft_limit'],
            tpot_limit=self.config['control']['tpot_limit'],
            switch_cost_weight=self.config.get('control', {}).get('switch_cost_weight', 0.1)
        )
        
        # 控制参数
        self.decision_interval = self.config['control']['decision_interval']
        self.running = True
        
        # 记录上一次的频率（用于计算切换成本）
        self.last_frequency = self.gpu_controller.current_freq
        
        # 统计信息
        self.stats = {
            'start_time': time.time(),
            'decisions': 0,
            'total_energy': 0.0,
            'frequency_changes': 0,
            'emergency_actions': 0,
            'best_reward': float('-inf'),
            'worst_reward': float('inf'),
            'idle_cycles': 0,
            'learning_cycles': 0
        }
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("✅ 系统初始化完成")
        logger.info(f"📋 配置摘要:")
        logger.info(f"   - GPU设备: {self.config['gpu']['device_id']}")
        logger.info(f"   - 频率范围: {self.gpu_controller.min_freq}-{self.gpu_controller.max_freq}MHz")
        logger.info(f"   - 频率档位: {len(self.gpu_controller.frequencies)}个")
        logger.info(f"   - 决策间隔: {self.decision_interval}秒")
        logger.info(f"   - TTFT限制: {self.config['control']['ttft_limit']}秒")
        logger.info(f"   - TPOT限制: {self.config['control']['tpot_limit']}秒")
        logger.info(f"   - 初始探索率: {self.config['linucb']['initial_alpha']}")
        logger.info(f"   - 正则化参数: {self.config['linucb'].get('lambda_reg', 0.1)}")
        
    def _signal_handler(self, signum, frame):
        """处理退出信号"""
        logger.info("\n" + "="*60)
        logger.info("📛 收到退出信号，正在优雅关闭...")
        
        # 保存模型
        logger.info("💾 保存最终模型...")
        self.linucb.save_model(f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        
        # 显示最终统计
        self._display_final_stats()
        
        self.running = False
        logger.info("👋 再见！")
        sys.exit(0)
        
    def _check_emergency(self, gauge_metrics: Dict, counter_deltas: Dict) -> Optional[int]:
        """检查紧急情况，返回建议的动作索引"""
        # TTFT超限检查
        ttft_count = counter_deltas.get('vllm:time_to_first_token_seconds_count_delta', 0)
        ttft_sum = counter_deltas.get('vllm:time_to_first_token_seconds_sum_delta', 0)
        if ttft_count > 0:
            avg_ttft = ttft_sum / ttft_count
            if avg_ttft > self.config['control']['ttft_limit']:
                logger.warning(f"🚨 紧急！TTFT={avg_ttft:.3f}s > {self.config['control']['ttft_limit']}s")
                self.stats['emergency_actions'] += 1
                return len(self.gpu_controller.frequencies) - 1  # 最高频率
        
        # TPOT超限检查
        tpot_count = counter_deltas.get('vllm:time_per_output_token_seconds_count_delta', 0)
        tpot_sum = counter_deltas.get('vllm:time_per_output_token_seconds_sum_delta', 0)
        if tpot_count > 0:
            avg_tpot = tpot_sum / tpot_count
            if avg_tpot > self.config['control']['tpot_limit']:
                logger.warning(f"🚨 紧急！TPOT={avg_tpot:.3f}s > {self.config['control']['tpot_limit']}s")
                self.stats['emergency_actions'] += 1
                return len(self.gpu_controller.frequencies) - 1  # 最高频率
                
        # 抢占发生
        if counter_deltas.get('vllm:num_preemptions_total_delta', 0) > 0:
            logger.warning("🚨 发生抢占，需要更高频率")
            self.stats['emergency_actions'] += 1
            # 提升2个档位，但不超过最高
            return min(len(self.gpu_controller.frequencies) - 1, 
                      self.gpu_controller.current_idx + 2)
            
        # GPU过热保护
        gpu_stats = self.gpu_controller.get_gpu_stats()
        if gpu_stats['temperature'] > 100:
            logger.warning(f"🔥 GPU过热: {gpu_stats['temperature']}°C")
            self.stats['emergency_actions'] += 1
            # 降低2个档位，但不低于最低
            return max(0, self.gpu_controller.current_idx - 2)
            
        return None
        
    def run(self):
        """主控制循环"""
        logger.info("\n" + "="*60)
        logger.info("🎮 开始自适应GPU调频控制")
        logger.info("="*60 + "\n")
        
        iteration = 0
        consecutive_errors = 0
        
        while self.running:
            iteration += 1
            cycle_start = time.time()
            
            # 动态分隔线
            if iteration % 10 == 0:
                logger.info(f"\n{'='*20} 🎯 迭代 {iteration} {'='*20}")
            else:
                logger.info(f"\n{'─'*20} 迭代 {iteration} {'─'*20}")
            
            try:
                # 1. 采集当前状态（2秒窗口）
                logger.info("📊 [阶段1/5] 采集系统状态...")
                gauge_metrics, counter_deltas, energy_delta = \
                    self.metrics_collector.collect_2s_metrics(
                        energy_reader=self.gpu_controller.read_energy_mj
                    )

                # 检查是否完全空闲
                num_running = gauge_metrics.get('vllm:num_requests_running', 0)
                num_waiting = gauge_metrics.get('vllm:num_requests_waiting', 0)
                
                if num_running == 0 and num_waiting == 0:
                    # 系统空闲
                    logger.info("😴 系统完全空闲，重置GPU时钟")
                    self.gpu_controller.reset_gpu_clocks()
                    self.stats['idle_cycles'] += 1
                    
                    # 等待下一个决策周期
                    logger.info(f"⏳ 等待{self.decision_interval}秒...")
                    time.sleep(self.decision_interval)
                    
                    # 记录能耗但不更新模型
                    energy_consumed = self.gpu_controller.get_energy_delta()
                    self.stats['total_energy'] += energy_consumed
                    
                    continue  # 跳过本轮的学习和决策
                
                self.stats['learning_cycles'] += 1
                
                # 2. 特征提取和标准化
                logger.info("🔍 [阶段2/5] 提取特征...")
                raw_features = self.feature_extractor.extract(gauge_metrics, counter_deltas)
                normalized_features = self.feature_extractor.normalize(raw_features)
                
                # 显示关键特征
                logger.info(f"   队列状态: {'有等待' if raw_features[0] > 0 else '无等待'}")
                logger.info(f"   并发请求: {raw_features[4]:.0f}")
                logger.info(f"   GPU缓存: {raw_features[5]:.1f}%")
                
                # 3. 紧急情况检查
                emergency_action = self._check_emergency(gauge_metrics, counter_deltas)

                # 4. 决策
                if emergency_action is not None:
                    actual_action = emergency_action
                    logger.info(f"⚡ [阶段3/5] 紧急决策: 动作{actual_action} "
                              f"({self.gpu_controller.frequencies[actual_action]}MHz)")
                else:
                    # LinUCB决策（Hybrid版本直接使用基础特征）
                    actual_action = self.linucb.select_action(normalized_features)
                    logger.info(f"🤔 [阶段3/5] LinUCB决策: 动作{actual_action} "
                              f"({self.gpu_controller.frequencies[actual_action]}MHz)")

                # 5. 应用新频率
                new_freq = self.gpu_controller.frequencies[actual_action]
                old_freq = self.gpu_controller.current_freq
                
                if new_freq != old_freq:
                    freq_change = new_freq - old_freq
                    symbol = "📈" if freq_change > 0 else "📉"
                    logger.info(f"{symbol} [阶段4/5] 调整频率: {old_freq}MHz → {new_freq}MHz "
                              f"({freq_change:+d}MHz)")
                    
                    # 设置新频率
                    if self.gpu_controller.set_frequency(new_freq):
                        # 成功后再读一次实际频率
                        new_freq = self.gpu_controller.current_freq
                        self.stats['frequency_changes'] += 1
                        self.last_frequency = old_freq
                    else:
                        logger.error("❌ 设置频率失败，保持原频率")
                        new_freq = old_freq  # 回退
                else:
                    logger.info(f"✅ [阶段4/5] 保持频率: {new_freq}MHz")
                
                # 6. 等待效果稳定
                logger.info(f"⏳ 等待{self.decision_interval}秒观察效果...")
                time.sleep(self.decision_interval)
                
                # 7. 采集新状态
                logger.info("📊 [阶段5/5] 评估效果...")
                new_gauge, new_counter_deltas, energy_consumed = \
                    self.metrics_collector.collect_2s_metrics(
                        energy_reader=self.gpu_controller.read_energy_mj
                    )
                
                # 获取实际能耗
                energy_consumed = self.gpu_controller.get_energy_delta()
                
                # 8. 计算奖励（包含频率切换成本）
                reward, reward_info = self.reward_calculator.calculate(
                    new_counter_deltas, 
                    energy_consumed,
                    current_freq=new_freq,
                    previous_freq=self.last_frequency,
                    max_freq=self.gpu_controller.max_freq
                )
                
                # 9. 更新模型（如果不是预热期或无请求）
                if (not self.no_learn                       # ← 新增判断
                    and not reward_info.get('no_requests')
                    and not reward_info.get('warming_up')):
                    self.linucb.update(actual_action, normalized_features, reward)

                    
                    # 更新最佳/最差奖励
                    if reward > self.stats['best_reward']:
                        self.stats['best_reward'] = reward
                        logger.info(f"🏆 新的最佳奖励: {reward:.3f}")
                    if reward < self.stats['worst_reward']:
                        self.stats['worst_reward'] = reward
                    
                # 10. 显示详细状态
                self._display_status(
                    iteration, old_freq, new_freq, 
                    gauge_metrics, new_gauge, 
                    reward, reward_info
                )
                
                # 更新统计
                self.stats['decisions'] += 1
                self.stats['total_energy'] += energy_consumed
                
                # 重置错误计数
                consecutive_errors = 0
                
            except KeyboardInterrupt:
                raise  # 让信号处理器处理
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"❌ 迭代错误 ({consecutive_errors}): {e}", exc_info=True)
                
                if consecutive_errors >= 5:
                    logger.critical("连续错误过多，系统退出")
                    self.running = False
                else:
                    logger.info(f"等待{self.decision_interval}秒后重试...")
                    time.sleep(self.decision_interval)
            
            # 显示循环耗时
            cycle_time = time.time() - cycle_start
            if cycle_time > self.decision_interval * 2.5:
                logger.warning(f"⚠️ 循环耗时过长: {cycle_time:.1f}秒")
                
    def _display_status(self, iteration, old_freq, new_freq, 
                       old_metrics, new_metrics, reward, reward_info):
        """显示详细状态信息"""
        
        # 构建状态框
        status_lines = []
        
        status_lines.append("\n┌" + "─"*58 + "┐")
        status_lines.append(f"│ 📊 状态汇总 - 迭代 {iteration:<41} │")
        status_lines.append("├" + "─"*58 + "┤")
        
        # 频率信息
        freq_change = new_freq - old_freq
        freq_symbol = "↑" if freq_change > 0 else ("↓" if freq_change < 0 else "→")
        status_lines.append(f"│ 🔧 频率: {old_freq:>4}MHz {freq_symbol} {new_freq:<4}MHz "
                          f"({'+'if freq_change>=0 else ''}{freq_change}MHz){' '*14} │")
        
        # 负载信息
        old_running = old_metrics.get('vllm:num_requests_running', 0)
        old_waiting = old_metrics.get('vllm:num_requests_waiting', 0)
        new_running = new_metrics.get('vllm:num_requests_running', 0)
        new_waiting = new_metrics.get('vllm:num_requests_waiting', 0)
        
        status_lines.append(f"│ 👥 请求: 运行 {old_running:.0f}→{new_running:.0f}, "
                          f"等待 {old_waiting:.0f}→{new_waiting:.0f}{' '*23} │")
        
        # 缓存信息
        old_cache = old_metrics.get('vllm:gpu_cache_usage_perc', 0)*100
        new_cache = new_metrics.get('vllm:gpu_cache_usage_perc', 0)*100
        status_lines.append(f"│ 💾 缓存: {old_cache:>5.1f}% → {new_cache:<5.1f}%{' '*31} │")
        
        # 性能指标
        if reward_info.get('avg_ttft') is not None:
            ttft_ms = reward_info['avg_ttft'] * 1000
            ttft_baseline_ms = reward_info.get('ttft_ema', 0) * 1000
            status_lines.append(f"│ ⏱️  TTFT: {ttft_ms:>6.1f}ms (基线: {ttft_baseline_ms:.1f}ms){' '*18} │")
            
        if reward_info.get('avg_tpot') is not None:
            tpot_ms = reward_info['avg_tpot'] * 1000
            tpot_baseline_ms = reward_info.get('tpot_ema', 0) * 1000
            status_lines.append(f"│ ⏱️  TPOT: {tpot_ms:>6.1f}ms (基线: {tpot_baseline_ms:.1f}ms){' '*18} │")
        
        # 能耗和奖励
        energy = reward_info.get('energy_j', 0)
        edp = reward_info.get('edp', 0)
        status_lines.append(f"│ ⚡ 能耗: {energy:>6.1f}J,  EDP: {edp:.6f}{' '*19} │")
        
        # 奖励组成
        if 'base_reward' in reward_info:
            status_lines.append(f"│ 💰 奖励组成: 基础={reward_info['base_reward']:+.2f}, "
                              f"切换={reward_info.get('switch_cost', 0):+.2f}, "
                              f"惩罚={reward_info.get('delay_penalty', 0):+.2f}{' '*5} │")
        
        reward_symbol = "🟢" if reward > 0 else ("🔴" if reward < -10 else "🟡")
        status_lines.append(f"│ {reward_symbol} 最终奖励: {reward:>8.3f}{' '*33} │")
        
        # GPU状态
        gpu_stats = self.gpu_controller.get_gpu_stats()
        status_lines.append(f"│ 🌡️  GPU: {gpu_stats['temperature']:.0f}°C, "
                          f"{gpu_stats['utilization']:.0f}%, "
                          f"{gpu_stats['power']:.0f}W{' '*23} │")
        
        # 模型状态

        model_stats = self.linucb.get_model_stats()
        status_lines.append(f"│ 🤖 模型: α={self.linucb.alpha:.2f}, "
                          f"总轮次={model_stats['total_rounds']}, "
                          f"平均奖励={model_stats['avg_reward']:.3f}{' '*8} │")
        
        status_lines.append("└" + "─"*58 + "┘")
        
        # 一次性输出
        logger.info('\n'.join(status_lines))
        
    def _display_final_stats(self):
        """显示最终统计信息"""
        runtime = time.time() - self.stats['start_time']
        
        stats_lines = []
        stats_lines.append("\n" + "="*60)
        stats_lines.append("📊 最终统计报告")
        stats_lines.append("="*60)
        
        # 运行信息
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        seconds = int(runtime % 60)
        stats_lines.append(f"⏱️  运行时间: {hours:02d}:{minutes:02d}:{seconds:02d}")
        stats_lines.append(f"🔄 总迭代次数: {self.stats['decisions'] + self.stats['idle_cycles']}")
        stats_lines.append(f"   - 学习周期: {self.stats['learning_cycles']}")
        stats_lines.append(f"   - 空闲周期: {self.stats['idle_cycles']}")
        stats_lines.append(f"🔧 频率调整: {self.stats['frequency_changes']}次")
        stats_lines.append(f"🚨 紧急响应: {self.stats['emergency_actions']}次")
        
        # 能耗信息
        total_energy_j = self.stats['total_energy'] / 1000
        avg_power = self.stats['total_energy'] / runtime if runtime > 0 else 0
        stats_lines.append(f"⚡ 总能耗: {total_energy_j:.1f}J")
        stats_lines.append(f"⚡ 平均功率: {avg_power:.1f}mW")
        
        # 奖励信息
        if self.stats['best_reward'] > float('-inf'):
            stats_lines.append(f"🏆 最佳奖励: {self.stats['best_reward']:.3f}")
        if self.stats['worst_reward'] < float('inf'):
            stats_lines.append(f"💀 最差奖励: {self.stats['worst_reward']:.3f}")
        
        # 频率使用分布
        model_stats = self.linucb.get_model_stats()
        stats_lines.append(f"\n📊 频率使用分布:")
        total_uses = sum(model_stats['action_counts'])
        for i, (freq, count) in enumerate(zip(self.gpu_controller.frequencies, 
                                            model_stats['action_counts'])):
            if total_uses > 0:
                percentage = count / total_uses * 100
                bar_length = int(percentage / 2)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                stats_lines.append(f"   {freq:>4}MHz: {bar} {percentage:>5.1f}% ({count}次)")
            else:
                stats_lines.append(f"   {freq:>4}MHz: {'░'*50}   0.0% (0次)")
        
        # 模型信息
        stats_lines.append(f"\n🤖 模型信息:")
        stats_lines.append(f"   最终α: {model_stats['current_alpha']:.3f}")
        stats_lines.append(f"   平均奖励: {model_stats['avg_reward']:.3f}")
        if 'action_entropy' in model_stats:
            stats_lines.append(f"   动作熵: {model_stats['action_entropy']:.3f}")
            stats_lines.append(f"   有效动作数: {model_stats['effective_actions']:.1f}")
        
        stats_lines.append("="*60)
        
        # 一次性输出
        logger.info('\n'.join(stats_lines))

def main():
    """程序入口"""
    parser = argparse.ArgumentParser(
        description='vLLM GPU自适应调频系统 (Hybrid LinUCB)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 正常启动（自动加载最新模型）
  python -m src.main
  
  # 从头开始学习
  python -m src.main --reset-model
  
  # 加载特定模型
  python -m src.main --model-file model_20240115.pkl
  
  # 使用自定义配置
  python -m src.main --config my_config.yaml
        """
    )
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径 (默认: config/config.yaml)')
    parser.add_argument('--reset-model', action='store_true',
                       help='重置模型，从头开始学习')
    parser.add_argument('--model-file', type=str, default=None,
                       help='指定要加载的模型文件')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')
    parser.add_argument(
    '--no-learn',            # 或 --inference-only
    action='store_true',
    help='仅加载模型推断，不进行在线学习'
                    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    import logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 确保必要目录存在
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("data/models").mkdir(parents=True, exist_ok=True)
    
    # 显示启动信息
    logger.info("🚀 vLLM GPU自适应调频系统 (Hybrid LinUCB)")
    logger.info(f"📅 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📝 配置文件: {args.config}")
    logger.info(f"🤖 模型选项: {'重置' if args.reset_model else ('加载 ' + args.model_file if args.model_file else '自动')}")
    
    try:
        # 创建并运行控制器
        autoscaler = VLLMGPUAutoscaler(
            config_path=args.config,
            reset_model=args.reset_model,
            model_file=args.model_file,
            no_learn=args.no_learn
        )
        autoscaler.run()
        
    except KeyboardInterrupt:
        logger.info("\n⌨️  键盘中断")
    except Exception as e:
        logger.critical(f"💥 致命错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()