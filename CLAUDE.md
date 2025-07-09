# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **vLLM Multi-Armed Bandit GPU Autoscaler** - an intelligent GPU frequency optimization system that uses advanced reinforcement learning (Contextual LinUCB) to automatically adjust NVIDIA GPU clock frequencies in real-time. The system employs a **Contextual Bandit architecture** where frequency acts as the arm/action, and workload features serve as context, to minimize the Energy-Delay Product (EDP) while maintaining quality of service constraints for vLLM (large language model inference server).

### Core Innovation (Current State - v5.0-contextual-bandit)
- **Pure Contextual LinUCB Multi-Armed Bandit**: Standard contextual bandit where frequencies are arms (actions), workload features are context (7-dimensional)
- **Real-Time Energy Optimization**: Direct GPU power measurement with fixed 0.3s decision cycles
- **Adaptive Frequency Discovery**: Intelligent exploration with dual-mode adaptive frequency space management (SLO-aware and EDP-optimal)
- **Smart Action Pruning**: Advanced frequency pruning with cascade protection and extreme frequency instant pruning
- **Mixed Maturity-Based Refinement**: Dual strategy based on learner maturity (statistical vs predictive refinement)
- **Optional Memory Frequency Optimization**: Support for combined core and memory frequency optimization when hardware supports it

## Current Architecture (v5.0-contextual-bandit)

The system consists of 8 main components working in a control loop:

1. **Main Controller** (`src/main.py`) - Orchestrates the entire system with GPU model classification, idle mode detection, convergence management, and optional memory frequency optimization
2. **Contextual LinUCB Model** (`src/contextual_bandit.py`) - Pure contextual bandit where frequencies are arms, workload features are context (7-dimensional) with advanced action pruning including extreme frequency instant pruning
3. **GPU Controller** (`src/gpu_controller.py`) - Advanced GPU frequency control with adaptive sampling integration and optional memory frequency support
4. **Adaptive Frequency Sampler** (`src/adaptive_frequency_sampler.py`) - Dual-mode intelligent frequency space management (SLO-aware and EDP-optimal) with mixed maturity-based refinement (statistical vs predictive)
5. **Feature Extractor** (`src/feature_extractor.py`) - Extracts 7-dimensional workload contextual features from vLLM metrics with online normalization using Welford's algorithm
6. **Metrics Collector** (`src/metrics_collector.py`) - Optimized Prometheus metrics collection with session reuse and EMA smoothing for gauge metrics
7. **Reward Calculator** (`src/reward_calculator.py`) - EDP-focused reward calculation with adaptive baseline using percentile normalization
8. **Logger** (`src/logger.py`) - GPU-classified logging system with detailed round logging and JSON structured data

## Key Technical Features (Current Implementation v5.0)

- **Pure Contextual Bandit**: Frequencies as actions/arms, workload features as context (7-dimensional, no frequency features)
- **Enhanced Action Pruning**: Automatic removal of poor-performing frequencies with adaptive cascade pruning (gpu_max_freq // 2 threshold) and extreme frequency instant pruning (first 20 rounds)
- **Dual-Mode Adaptive Sampling**: SLO-aware mode (high-to-low frequency search) and EDP-optimal mode (reward-driven refinement)
- **Mixed Maturity-Based Refinement**: Statistical refinement for immature models (<100 rounds), predictive refinement for mature models (≥100 rounds) with load-aware frequency recommendation
- **Adaptive Alpha Decay**: Exploration parameter decreases over time with configurable minimum (default: 0.1)
- **Optional Memory Frequency Optimization**: Combined core and memory frequency optimization when hardware supports it
- **GPU-Classified Logging**: Automatic log organization by GPU model with detailed round logging and JSON structured data
- **Fixed Timing System**: Hardcoded 0.3s decision intervals for consistent energy measurement
- **Online Feature Normalization**: Welford's algorithm for stable feature standardization

### Adaptive Control System (Current v5.0)
- **Dual-mode adaptive sampling**: SLO-aware mode (high-to-low frequency search) and EDP-optimal mode (reward-driven refinement)
- **Automatic frequency detection**: No hardcoded frequency parameters, intelligently detects GPU capabilities (both core and memory frequencies)
- **Real-time learning**: Adapts GPU frequency every 0.3 seconds based on current workload (fixed timing)
- **Intelligent two-phase control**: Automatic switching between learning and exploitation modes with top-k stability convergence
- **Dynamic action space**: Runtime frequency list updates based on performance feedback with pruning protection
- **SLO boundary detection**: Automatic constraint awareness and frequency space adjustment
- **Performance monitoring**: EDP-based degradation detection with automatic return to learning mode
- **Idle mode detection**: Automatic GPU frequency reset during no-load periods (210MHz safe frequency)
- **Memory frequency optimization**: Optional combined core+memory frequency optimization for maximum energy efficiency

### Current Infrastructure Features
- **Model persistence**: Automatic saving/loading with contextual bandit format, pruning state, and memory frequency optimization state
- **Energy monitoring**: Direct GPU power consumption measurement with fixed 0.3s intervals for consistent timing
- **GPU-classified logging**: Logs automatically organized by GPU model with detailed round logging and JSON structured data
- **Robust GPU control**: Multi-method frequency setting with failure recovery, state tracking, and optional memory frequency control
- **High-frequency monitoring**: Sub-second metrics collection and decision cycles (fixed 0.3s intervals)
- **Connection pooling**: Uses requests.Session() for efficient Prometheus connection reuse
- **Error handling and recovery**: Automatic fallback mechanisms for frequency setting failures and energy counter rollover protection
- **Thread-safe operations**: Locks for frequency access and energy reading to ensure data consistency
- **Optimized performance**: Batch UCB calculations and memory-efficient circular buffers for history management

## Current System Operational Modes (v5.0)

### Adaptive Frequency Sampling Modes

**SLO-Aware Mode** (`ignore_slo: false`):
- High-to-low frequency search strategy with safety prioritization
- SLO boundary detection and safe zone generation
- Mixed refinement: statistical (immature) vs predictive (mature) learning
- Automatic safe frequency space adjustment based on violation detection
- Supports both core-only and combined core+memory frequency optimization

**EDP-Optimal Mode** (`ignore_slo: true`):
- Full domain coarse search followed by reward-driven refinement
- Focuses purely on Energy-Delay Product optimization with mixed strategies
- High-reward zone identification with maturity-based refinement approaches
- Ignores SLO constraints for maximum performance optimization
- Optimal for research environments and maximum energy efficiency scenarios

### Current Learning vs Exploitation Phases

**Phase 1: Learning Mode (Exploration)**:
- Contextual LinUCB exploration with UCB confidence bounds
- Adaptive alpha decay with configurable minimum threshold (0.1-10.0)
- Smart action pruning during exploration to remove poor frequencies (both standard and extreme pruning)
- Mixed refinement strategies based on learner maturity threshold (100 rounds)
- Optional memory frequency exploration when hardware supports it
- Triggers: Fresh start, no existing model, performance degradation, or significant SLO violations

**Phase 2: Exploitation Mode (Convergence)**:
- Greedy selection using trained contextual model predictions
- Minimal exploration, focuses on top-k stable frequency patterns (default: top-3 with 60% threshold)
- Automatic model saving as "final_contextual_model" or "contextual_linucb_model"
- Supports both core-only and combined frequency exploitation
- Triggers: Top-k action stability convergence detection (configurable threshold)

## Running the System

### Prerequisites
```bash
# Install Python dependencies
pip install numpy pynvml requests pyyaml matplotlib seaborn scipy

# Ensure nvidia-smi is available and GPU permissions are set
nvidia-smi

# Test NVML Python bindings
python -c "import pynvml; pynvml.nvmlInit(); print('NVML OK')"
```

### Basic Usage (Current v5.0)
```bash
# Run the main contextual autoscaler (recommended module import method)
python -m src.main

# Run with specific options (reset model starts fresh)
python -m src.main --reset-model --log-level DEBUG

# Run with custom configuration
python -m src.main --config config/custom_config.yaml

# Enable memory frequency optimization (if hardware supports it)
# Edit config.yaml: enable_memory_frequency_control: true

# Load specific contextual model for inference only (auto-loading supported)
# Note: v5.0 uses auto-loading of latest contextual model files with memory frequency state
```

### Critical Configuration Parameters

Configuration file: `config/config.yaml`

**Critical Configuration Parameters (v5.0)**:
- Decision interval is hardcoded to 0.3s in main loop (`time.sleep(0.3)`)
- Energy measurements are automatically aligned with this timing
- No manual decision_interval configuration required

**Adaptive Sampling Control (v5.0)**:
- `control.ignore_slo`: Controls sampling mode (true=EDP-optimal, false=SLO-aware)
- `control.adaptive_update_interval`: Frequency space refinement interval (default: 10 decisions)
- `control.convergence_window`: Learning phase duration and convergence history window (default: 100)
- `control.learner_maturity_threshold`: Switches refinement strategy (default: 100)
- `control.convergence_top_k`: Top-k actions for stability detection (default: 3)
- `control.convergence_threshold`: Stability threshold for convergence (default: 0.6)
- `control.long_interval`: Long-period heartbeat interval for mixed trigger mechanism (default: 10 seconds)
- `control.change_detection_threshold`: Event-driven trigger threshold (default: 0.9)
- `control.convergence_p_value_threshold`: P-value threshold for convergence detection (default: 0.05)
- `control.performance_degradation_threshold`: Performance degradation threshold (default: 0.3)
- `control.refinement_start_threshold`: Minimum rounds before refinement starts (default: 50)

**Adaptive Frequency Sampling Parameters (v5.0)**:
- `adaptive_sampling.reward_threshold`: High-reward zone threshold for EDP mode refinement (default: 0.5)
- `adaptive_sampling.slo_safe_ratio`: Safe zone action ratio in SLO mode (default: 0.7)
- `adaptive_sampling.slo_coarse_step`: Coarse search step size in SLO mode (default: 90MHz)
- `adaptive_sampling.slo_fine_step`: Fine search step size in SLO mode (default: 15MHz)
- `adaptive_sampling.edp_initial_step`: Initial step size in EDP mode (default: 90MHz)
- `adaptive_sampling.edp_fine_step`: Fine step size in EDP mode (default: 15MHz)

**GPU Frequency Management (v5.0)**:
- `gpu.auto_step`: Enables automatic GPU frequency detection (recommended: true)
- `gpu.frequency_step`: Base frequency step size in MHz (default: 15)
- `gpu.enable_memory_frequency_control`: Memory frequency control support (default: false)
- `gpu.memory_auto_detect`: Automatic memory frequency detection (default: true)
- `gpu.memory_frequencies`: Manual memory frequency list (empty for auto-detection)

**Contextual LinUCB Configuration (Current Algorithm)**:
- `model.type`: Fixed to "contextual_linucb" (only supported algorithm in v5.0)
- `linucb.initial_alpha`: UCB confidence parameter (default: 10.0, increased exploration)
- `linucb.alpha_decay_rate`: Alpha decay rate over time (default: 0.02)
- `linucb.min_alpha`: Minimum alpha threshold (default: 0.1)
- `linucb.lambda_reg`: Regularization parameter (default: 1.0)
- `linucb.use_continuous`: Enable continuous parameterization (default: true)
- `linucb.forgetting_factor`: Historical data decay (default: 0.01, light forgetting - retains 99% of history per update)
- `linucb.convergence_method`: Convergence detection method (default: "page_hinkley", options: t_test, page_hinkley, cusum)

**Smart Action Pruning (v5.0 Enhanced Feature)**:
- `linucb.enable_action_pruning`: Enable intelligent frequency pruning (default: true)
- `linucb.pruning_check_interval`: Pruning evaluation interval (default: 20)
- `linucb.pruning_threshold`: Historical reward gap threshold (default: 3.0)
- `linucb.min_exploration_for_pruning`: Minimum exploration before pruning (default: 6)
- `linucb.pruning_maturity_threshold`: Maturity requirement for pruning (default: 30)
- `linucb.cascade_pruning_threshold`: Frequency threshold for cascade pruning (default: 800MHz, fallback only - system uses adaptive threshold = gpu_max_freq // 2)

**Extreme Frequency Instant Pruning (v5.0 New Feature)**:
- `linucb.enable_extreme_pruning`: Enable extreme frequency instant pruning (default: true)
- `linucb.extreme_pruning_threshold`: Extreme frequency reward threshold (default: -1.5)
- `linucb.extreme_pruning_min_samples`: Minimum samples for extreme pruning (default: 3)
- `linucb.extreme_pruning_max_rounds`: Maximum rounds for extreme pruning (default: 50)

**Metrics Collection Configuration (v5.0)**:
- `metrics.sampling_duration`: Standard collection window duration (default: 1.5 seconds)
- `metrics.sampling_interval`: Sub-sampling interval (default: 0.01 seconds)
- `metrics.ema_alpha`: EMA smoothing coefficient for gauge metrics (default: 0.4)

**Model Persistence Configuration (v5.0)**:
- `model.save_dir`: Model save directory (default: "data/models")
- `model.save_interval`: Model save frequency in rounds (default: 50)
- `model.keep_last_n`: Number of recent models to retain (default: 3)

**Advanced Logging Configuration (v5.0)**:
- `logging.console_level`: Console log level (default: INFO)
- `logging.file_level`: File log level (default: DEBUG)
- `logging.detailed_round_logging`: Enable detailed round information logging (default: true)
- `logging.console_simple`: Use simplified console output filtering (default: true)

## Development Commands

### Core System Operations (v5.0)
```bash
# Run main contextual system with different modes
python -m src.main                          # Normal operation with auto-loaded contextual model
python -m src.main --reset-model             # Fresh start, ignore existing contextual models
python -m src.main --log-level DEBUG        # Enable debug logging for detailed information

# Test individual components (use module import to avoid path issues)
python -m src.gpu_controller                 # Test GPU control with adaptive sampling and memory frequency support
python -m src.metrics_collector              # Test vLLM metrics collection
python -m src.feature_extractor              # Test contextual feature extraction (7-dim)
python -c "from src.adaptive_frequency_sampler import create_default_sampler; s=create_default_sampler(); print('Sampler OK')"
python -c "from src.contextual_bandit import ContextualLinUCB; m=ContextualLinUCB(7); print('Contextual LinUCB OK')"

# Test memory frequency support
python -c "from src.gpu_controller import GPUController; g=GPUController(enable_memory_frequency_control=True); print('Memory freq support:', g.memory_frequency_supported)"

# Analysis and debugging
python analysis.py                           # Comprehensive performance analysis with EDP focus
python analysis_edp_focused.py              # EDP-focused analysis
```

### Testing and Validation
```bash
# Component-specific tests
python test_action_pruning.py               # Test action pruning functionality
python test_adaptive_integration.py         # Test adaptive sampling integration
python test_alpha_decay.py                  # Test alpha decay mechanism
python test_convergence.py                  # Test convergence detection
python test_mixed_refinement.py             # Test mixed refinement strategies
python test_memory_frequency_fix.py         # Test memory frequency functionality

# GPU-specific validation
python test_gpu_frequency.py                # Test GPU frequency control
python test_memory_frequency_sweep.py       # Test memory frequency sweep
python test_fixes.py                        # Test various system fixes

# Logging and debugging tests
python test_simplified_logs.py              # Test simplified logging
python test_no_redundant_logs.py            # Test redundant log filtering
```

### Data Management
```bash
# Cleanup operations (interactive with safety prompts)
./cleanup.sh                                 # Interactive cleanup
./cleanup.sh --force                         # Automated cleanup
./cleanup.sh --logs-only                     # Clean only logs
./cleanup.sh --data-only                     # Clean only data

# Code analysis and merging
./merge_for_gpt.sh                          # Create merged code for analysis
cat merged_for_gpt.txt                      # View merged output
```

### Monitoring and Debugging
```bash
# Real-time monitoring
tail -f logs/*/vllm_gpu_autoscaler_*.log     # Latest logs
watch -n 1 'nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits'  # GPU frequency

# Model status checking
python -c "
import json
with open('data/models/model_metadata.json') as f:
    meta = json.load(f)
    print(f'Phase: {meta.get(\"phase\", \"unknown\")}')
    print(f'Converged: {meta.get(\"converged\", \"unknown\")}')
"

# vLLM connectivity test
curl http://localhost:8001/metrics | head -20
```

## Current Implementation Details (v5.0)

### Smart Action Pruning (Enhanced Critical Feature)
The system includes intelligent frequency pruning to optimize action space:
- **Historical performance pruning**: Removes consistently poor-performing frequencies with enhanced thresholds
- **Adaptive cascade pruning**: When low frequencies are pruned, all lower frequencies are also removed. Uses adaptive threshold based on GPU max frequency (gpu_max_freq // 2) instead of fixed 800MHz threshold
- **Extreme frequency instant pruning**: Immediately removes extremely poor frequencies in early rounds (first 50 rounds)
- **Maturity-based pruning**: Standard pruning activates after sufficient exploration (default: 30 rounds)
- **Exploration protection**: Frequencies need minimum exploration before pruning eligibility (6 samples)
- **State persistence**: Pruning state and history persist across model saves/loads
- **Memory frequency integration**: Supports pruning in combined core+memory frequency optimization

**Key files**: `src/contextual_bandit.py` (pruning logic), `src/main.py` (pruning integration)

### Energy Measurement Timing (Fixed in v5.0)
Energy measurement timing is hardcoded and consistent:
- Decision interval fixed at 0.3 seconds in main control loop
- Energy measurements automatically aligned with this timing
- No manual configuration of timing parameters required
- Eliminates timing inconsistency issues from previous versions

### Contextual Bandit Architecture (v5.0)
- **Pure contextual design**: Frequencies are arms/actions, workload features are context
- **7-dimensional context**: No frequency features, only workload characterization
- **Independent arm models**: Each frequency maintains separate linear model
- **Adaptive alpha decay**: Exploration decreases over time with minimum threshold
- **Mixed refinement strategies**: Statistical vs predictive based on learner maturity
- **Optional memory frequency optimization**: Combined core+memory frequency actions when hardware supports it

### Current Model Architecture (v5.0-contextual-bandit)
- Standard contextual LinUCB with frequencies as actions
- 7-dimensional workload context features only
- Dynamic action space with enhanced intelligent pruning (standard + extreme)
- Forgetting factors for non-stationary adaptation (light: 0.01)
- Top-k stability convergence detection (default: top-3 with 60% threshold)
- Optional combined core+memory frequency action space
- Model version: "5.0-contextual-bandit" with enhanced action pruning

### GPU Control Robustness (Enhanced v5.0)
- **Adaptive sampling integration**: GPU controller manages frequency space via dual-mode adaptive sampler
- **Failure frequency tracking**: Automatically removes problematic frequencies (both core and memory)
- **Multi-method frequency setting**: Falls back through different nvidia-smi approaches
- **Memory frequency support**: Optional memory frequency control alongside core frequency
- **Idle mode detection**: Automatic frequency reset during no-load periods (210MHz safe frequency)
- **Hardware capability detection**: Automatic detection of memory frequency control support

### Current Feature Engineering (7-dimensional context)
Key extracted contextual features from vLLM metrics (no frequency features):
1. **Has Queue**: Binary indicator of waiting requests (workload presence)
2. **Prefill Throughput**: Prompt tokens processed per sampling period (input processing speed)
3. **Decode Throughput**: Generation tokens produced per sampling period (output generation speed)
4. **Packing Efficiency**: Average tokens per batch iteration (batching efficiency)
5. **Concurrency**: Number of currently running requests (system load)
6. **GPU Cache Usage**: GPU memory cache utilization percentage (memory pressure)
7. **Cache Hit Rate**: Prefix cache hit percentage (caching effectiveness)

**Note**: Frequency is NOT included as a feature - it serves as the action/arm in the contextual bandit

## Troubleshooting

### Common Issues

**GPU Permission Errors**:
```bash
# Test basic GPU access
nvidia-smi
# Test programmatic access
python -c "import pynvml; pynvml.nvmlInit(); print('NVML Access OK')"
# Check frequency control permissions
nvidia-smi -i 0 -lgc 1000  # Test frequency setting
```

**vLLM Connection Issues**:
```bash
# Verify vLLM is running and accessible
curl http://localhost:8001/metrics
# Check for required metrics
curl http://localhost:8001/metrics | grep -E "(queue|memory|request)"
```

**Energy Measurement Problems**:
- Decision interval is fixed at 0.3s (hardcoded) - no configuration needed
- Verify NVML energy API support: `nvidia-smi -q -d POWER`
- Monitor for negative energy deltas (indicates GPU energy counter reset)

**Model Loading/Saving Issues**:
```bash
# Check data directory permissions
ls -la data/models/
# Verify contextual model file integrity
python -c "import pickle; pickle.load(open('data/models/contextual_linucb_model_*.pkl', 'rb')); print('Contextual Model OK')"
# Check for latest model files
ls -la data/models/contextual_linucb_model_*.pkl | tail -5
```

**Memory Frequency Issues**:
```bash
# Test memory frequency support detection
python -c "from src.gpu_controller import GPUController; g=GPUController(enable_memory_frequency_control=True); print('Memory support:', g.memory_frequency_supported)"
# Check memory frequency capabilities
nvidia-smi -q -d SUPPORTED_CLOCKS | grep -A 10 "Supported Memory"
```

**Hardcoded Values and Thresholds**:
- **Emergency lower bound**: `max_settable_freq - 450MHz` for SLO violation recovery
- **Fallback frequency**: `1500MHz` when frequency setting fails
- **Frequency tolerance**: `5MHz` for frequency setting verification
- **Default memory frequencies**: `[8001, 7601, 5001, 810, 405]MHz` when auto-detection fails
- **Load percentiles**: `p33 percentile` used for load classification in adaptive sampling

### Advanced Debugging

**Contextual LinUCB Model Debug**:
```python
# Check contextual model statistics
from src.contextual_bandit import ContextualLinUCB
model = ContextualLinUCB(7, auto_load=True)
model_stats = model.get_model_stats()
print(f"Model stats: {model_stats}")
print(f"Available frequencies: {len(model.available_frequencies)}")
print(f"Pruned frequencies: {len(getattr(model, 'pruned_frequencies', set()))}")
```

**Action Pruning Debug**:
```python
# Check action pruning status
from src.contextual_bandit import ContextualLinUCB
model = ContextualLinUCB(7, auto_load=True)
if hasattr(model, '_get_pruning_candidates'):
    candidates = model._get_pruning_candidates()
    print(f"Pruning candidates: {candidates}")
```

**Adaptive Sampling Debug**:
```python
# Check adaptive sampler statistics
from src.gpu_controller import GPUController
gpu = GPUController()
if hasattr(gpu, 'adaptive_sampler') and gpu.adaptive_sampler:
    print(f"Sampling mode: {gpu.adaptive_sampler.current_mode}")
    print(f"Current frequencies: {len(gpu.adaptive_sampler.current_frequencies)}")
```

**Memory Frequency Debug**:
```python
# Test memory frequency functionality
from src.gpu_controller import GPUController
gpu = GPUController(enable_memory_frequency_control=True)
print(f"Memory frequency supported: {gpu.memory_frequency_supported}")
if gpu.memory_frequency_supported:
    print(f"Available memory frequencies: {gpu.memory_frequencies}")
    print(f"Current memory frequency: {gpu.current_memory_freq}")
```

**Energy Timing Debug**:
```python
# Monitor energy measurement timing (fixed 0.3s intervals)
import time
from src.gpu_controller import GPUController
gpu = GPUController()
start = time.time()
energy = gpu.read_energy_mj()
elapsed = time.time() - start
print(f"Energy: {energy}mJ, Time: {elapsed:.3f}s")
```


## Performance Optimization

### LinUCB Tuning Guide

**For Stable Performance** (Conservative):
```yaml
linucb:
  initial_alpha: 5.0            # Balanced confidence bound
  forgetting_factor: 0.01       # Light historical decay
  alpha_decay_rate: 0.01        # Slow alpha decay
  min_alpha: 0.5                # Higher minimum alpha
  pruning_maturity_threshold: 50 # Later pruning activation
```

**For Aggressive Optimization** (Experimental):
```yaml
linucb:
  initial_alpha: 15.0           # High exploration
  forgetting_factor: 0.02       # Moderate forgetting
  alpha_decay_rate: 0.03        # Faster alpha decay
  min_alpha: 0.1                # Lower minimum alpha
  enable_extreme_pruning: true  # Enable extreme pruning
  extreme_pruning_max_rounds: 30 # Earlier extreme pruning
```

**For Fast Convergence** (Production):
```yaml
linucb:
  initial_alpha: 2.0            # Lower exploration
  forgetting_factor: 0.005      # Very light forgetting
  alpha_decay_rate: 0.02        # Standard decay
  min_alpha: 0.1                # Standard minimum
  pruning_maturity_threshold: 20 # Earlier pruning
  convergence_threshold: 0.7    # Higher convergence threshold
```

### GPU-Specific Optimization

**High-End GPUs** (RTX 4090, A100, H100):
```yaml
gpu:
  frequency_step: 15            # Fine-grained control
  enable_memory_frequency_control: true  # Enable if supported
  memory_auto_detect: true
control:
  convergence_window: 150       # Longer learning phase
  adaptive_update_interval: 10  # More frequent refinement
  learner_maturity_threshold: 150
```

**Mid-Range GPUs** (RTX 3080, RTX 4080):
```yaml
gpu:
  frequency_step: 15            # Standard fine control
  enable_memory_frequency_control: false  # Core-only for stability
control:
  convergence_window: 100       # Standard learning phase
  adaptive_update_interval: 10  # Standard refinement
  learner_maturity_threshold: 100
```

**Data Center GPUs** (Tesla V100, A6000, A800):
```yaml
gpu:
  frequency_step: 15            # Precision for efficiency
  enable_memory_frequency_control: true  # Maximum optimization
  memory_auto_detect: true
control:
  ignore_slo: false             # Respect SLO constraints
  ttft_limit: 1.5               # Stricter latency requirements
  tpot_limit: 0.2
  adaptive_update_interval: 10
```

### Workload-Specific Tuning

**Batch Inference** (High Throughput):
```yaml
control:
  ignore_slo: true              # Focus on EDP only
metrics:
  sampling_duration: 3          # Longer measurement windows
  ema_alpha: 0.2                # Slower smoothing
linucb:
  initial_alpha: 4.0            # More exploration for throughput
```

**Interactive Chat** (Low Latency):
```yaml
control:
  ignore_slo: false
  ttft_limit: 1.0               # Aggressive latency targets
  tpot_limit: 0.15
metrics:
  sampling_duration: 1.0        # Faster response detection
linucb:
  initial_alpha: 1.0            # Conservative exploration
```

**Research/Development** (Maximum Exploration):
```yaml
linucb:
  initial_alpha: 6.0            # Maximum exploration
  min_exploration_count: 10     # Thorough exploration
control:
  convergence_window: 200       # Extended learning
  convergence_p_value_threshold: 0.01  # Stricter convergence
```

## Extension Points

### Adding New Sampling Strategies
Extend `adaptive_frequency_sampler.py`:
- Add new `SamplingMode` enum values
- Implement mode-specific frequency generation methods
- Add refinement strategies based on performance feedback

### Custom Reward Functions
Modify `reward_calculator.py`:
- Implement custom EDP alternatives
- Add new constraint handling methods
- Support multi-objective optimization

### Enhanced Feature Engineering
Extend `feature_extractor.py`:
- Add new vLLM metrics extraction
- Implement temporal feature aggregation
- Support external monitoring system integration

### GPU Control Extensions
Enhance `gpu_controller.py`:
- Add support for memory frequency control
- Implement power limit management
- Support multi-GPU coordination

## Best Practices & Production Deployment

### Pre-Deployment Checklist

**System Requirements**:
```bash
# Verify GPU capabilities
nvidia-smi --query-gpu=name,driver_version,power.management --format=csv
nvidia-smi -q -d SUPPORTED_CLOCKS  # Check frequency support

# Test Python environment
python -c "import torch, pynvml, requests, yaml, numpy, scipy; print('✅ All dependencies OK')"

# Verify vLLM endpoint
curl -s http://localhost:8001/metrics | grep -q "vllm" && echo "✅ vLLM accessible" || echo "❌ vLLM not found"
```

**Configuration Validation**:
```bash
# Test configuration file syntax
python -c "import yaml; yaml.safe_load(open('config/config.yaml')); print('✅ Config valid')"

# Validate timing parameters
python -c "
import yaml
config = yaml.safe_load(open('config/config.yaml'))
decision_interval = config.get('control', {}).get('decision_interval', 0.4)
sampling_interval = config.get('metrics', {}).get('sampling_interval', 0.01)
if abs(decision_interval - 0.4) > 0.1:
    print(f'⚠️ Non-standard decision interval: {decision_interval}s')
print('✅ Timing validation complete')
"
```

### Production Monitoring

**Key Metrics to Monitor**:
1. **EDP Trend**: Should decrease over time as system learns
2. **Exploration Balance**: No frequency should have >30% or <2% selection
3. **Convergence Stability**: P-values should stabilize around threshold
4. **Energy Measurement**: Watch for negative deltas (counter resets)
5. **SLO Violations**: Track TTFT/TPOT constraint breaches

**Automated Health Checks**:
```bash
#!/bin/bash
# health_check.sh - Add to cron for periodic monitoring

# Check if system is running
if ! pgrep -f "src.main" > /dev/null; then
    echo "❌ vLLM MAB not running"
    exit 1
fi

# Check recent log activity
if [ $(find logs -name "*.log" -mmin -5 | wc -l) -eq 0 ]; then
    echo "⚠️ No recent log activity"
fi

# Check model convergence status
python -c "
import json
try:
    with open('data/models/model_metadata.json') as f:
        meta = json.load(f)
        phase = meta.get('phase', 'unknown')
        if phase == 'EXPLOITATION':
            print('✅ Model converged and exploiting')
        else:
            print(f'ℹ️ Model in {phase} phase')
except:
    print('⚠️ Model metadata not found')
"

echo "✅ Health check complete"
```

### Security Considerations

**GPU Access Control**:
```bash
# Limit system privileges
sudo usermod -a -G video $USER  # Add user to video group for GPU access
# Avoid running as root

# Monitor GPU frequency changes
sudo nvidia-smi -pm 1  # Enable persistent mode
sudo nvidia-smi -i 0 -pl 300  # Set reasonable power limit
```

**Network Security**:
```bash
# Restrict vLLM metrics endpoint if needed
iptables -A INPUT -p tcp --dport 8001 -s 127.0.0.1 -j ACCEPT
iptables -A INPUT -p tcp --dport 8001 -j DROP
```

### Multi-GPU Configuration

**For Multi-GPU Systems**:
```yaml
# config/multi_gpu_config.yaml
gpu:
  device_id: 0  # Primary GPU for now
  # Future: support multiple device_ids
  multi_gpu_coordination: false  # Not yet implemented

# Run separate instances per GPU
# GPU 0: python -m src.main --config config/gpu0_config.yaml
# GPU 1: python -m src.main --config config/gpu1_config.yaml
```

### Advanced Deployment Patterns

**Containerized Deployment**:
```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3-pip nvidia-utils-525
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python", "-m", "src.main"]
```

**Kubernetes Deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-mab-autoscaler
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-mab
  template:
    metadata:
      labels:
        app: vllm-mab
    spec:
      containers:
      - name: autoscaler
        image: vllm-mab:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: data
          mountPath: /app/data
      volumes:
      - name: config
        configMap:
          name: vllm-mab-config
      - name: data
        persistentVolumeClaim:
          claimName: vllm-mab-data
```

### Backup and Recovery

**Model Backup Strategy**:
```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "backup/models_$DATE.tar.gz" data/models/
find backup/ -name "models_*.tar.gz" -mtime +7 -delete  # Keep 7 days

# Configuration backup
cp config/config.yaml "backup/config_$DATE.yaml"
```

**Disaster Recovery**:
```bash
# Quick recovery procedure
# 1. Restore configuration
cp backup/config_YYYYMMDD_HHMMSS.yaml config/config.yaml

# 2. Restore latest model (optional - system can restart learning)
tar -xzf backup/models_YYYYMMDD_HHMMSS.tar.gz

# 3. Restart system
python -m src.main --reset-model  # Force fresh start if needed
```

## Advanced Development Workflows

### Research & Experimentation

**A/B Testing Framework**:
```bash
# Compare different LinUCB configurations
# Terminal 1: Conservative LinUCB
python -m src.main --config config/conservative_config.yaml --model-suffix "_conservative"

# Terminal 2: Aggressive LinUCB  
python -m src.main --config config/aggressive_config.yaml --model-suffix "_aggressive"

# Analysis comparison
python analysis.py --compare conservative_model.pkl aggressive_model.pkl
```

**Hyperparameter Sweeps**:
```bash
# Automated parameter exploration
for alpha in 1.0 2.0 3.0 5.0 8.0; do
    for forgetting in 0.0 0.1 0.2 0.3; do
        echo "Testing alpha=$alpha, forgetting=$forgetting"
        
        # Update config
        sed -i "s/initial_alpha: .*/initial_alpha: $alpha/" config/sweep_config.yaml
        sed -i "s/forgetting_factor: .*/forgetting_factor: $forgetting/" config/sweep_config.yaml
        
        # Run experiment
        python -m src.main --config config/sweep_config.yaml \
            --model-suffix "_alpha${alpha}_forget${forgetting}" \
            --max-rounds 500
        
        # Quick analysis
        python -c "
import pickle
model = pickle.load(open('data/models/latest_contextual_model_alpha${alpha}_forget${forgetting}.pkl', 'rb'))
print(f'Alpha {alpha}, Forgetting {forgetting}: Avg Reward = {sum(model.reward_history)/len(model.reward_history):.3f}')
        "
    done
done
```

**Custom Metrics Integration**:
```python
# src/custom_metrics.py - Extend feature extraction
def extract_custom_features(vllm_metrics: dict) -> np.ndarray:
    """Add custom workload characterization"""
    
    # GPU utilization from nvidia-ml-py
    gpu_util = get_gpu_utilization()
    
    # Network I/O patterns  
    network_bytes = get_network_activity()
    
    # Custom application metrics
    app_specific = extract_app_metrics()
    
    return np.array([gpu_util, network_bytes, app_specific])

# Integration in feature_extractor.py
def get_features(self) -> np.ndarray:
    base_features = self._extract_base_features()
    custom_features = extract_custom_features(self.latest_metrics)
    return np.concatenate([base_features, custom_features])
```

### Continuous Integration Pipeline

**Automated Testing Pipeline**:
```yaml
# .github/workflows/vllm_mab_ci.yml
name: vLLM MAB CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
        algorithm: [neural, linucb]
    
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Unit tests
      run: |
        pytest tests/ -v --cov=src/
    
    - name: Algorithm-specific tests  
      run: |
        python test_${{ matrix.algorithm }}_model.py
    
    - name: Integration tests
      run: |
        python test_integration.py --algorithm ${{ matrix.algorithm }}
```

**Performance Regression Testing**:
```bash
#!/bin/bash
# scripts/regression_test.sh
set -e

echo "🔄 Running performance regression tests..."

# Baseline model performance
python -m src.main --config config/baseline_config.yaml --max-rounds 100 --model-suffix "_baseline"
BASELINE_REWARD=$(python -c "
import pickle
model = pickle.load(open('data/models/latest_neural_model_baseline.pkl', 'rb'))
print(sum(model.reward_history[-50:])/50)  # Last 50 rounds average
")

echo "📊 Baseline reward: $BASELINE_REWARD"

# Current model performance  
python -m src.main --config config/config.yaml --max-rounds 100 --model-suffix "_current"
CURRENT_REWARD=$(python -c "
import pickle
model = pickle.load(open('data/models/latest_neural_model_current.pkl', 'rb'))
print(sum(model.reward_history[-50:])/50)
")

echo "📊 Current reward: $CURRENT_REWARD"

# Regression check
python -c "
baseline = float('$BASELINE_REWARD')
current = float('$CURRENT_REWARD')
regression = (baseline - current) / abs(baseline) * 100
if regression > 10:  # 10% regression threshold
    print(f'❌ Performance regression detected: {regression:.1f}%')
    exit(1)
else:
    print(f'✅ Performance OK: {regression:.1f}% change')
"
```

### Advanced Analysis Tools

**Real-Time Performance Dashboard**:
```python
# dashboard.py - Live monitoring interface
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

def create_live_dashboard():
    st.title("🎯 vLLM LinUCB Real-Time Dashboard")
    
    # Load latest model
    model_files = list(Path("data/models").glob("*contextual*.pkl"))
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_model, 'rb') as f:
        model = pickle.load(f)
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rounds", len(model.reward_history))
    with col2:
        recent_reward = np.mean(model.reward_history[-50:])
        st.metric("Recent Avg Reward", f"{recent_reward:.3f}")
    with col3:
        exploration_ratio = len([r for r in model.reward_history[-100:] if r > 0]) / 100
        st.metric("Exploration Ratio", f"{exploration_ratio:.1%}")
    with col4:
        if hasattr(model, 'frequency_counts'):
            most_freq = max(model.frequency_counts, key=model.frequency_counts.get)
            st.metric("Preferred Frequency", f"{most_freq}MHz")
    
    # Reward trend plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=model.reward_history, name="Reward"))
    # 50-point moving average
    ma_50 = pd.Series(model.reward_history).rolling(50).mean()
    fig.add_trace(go.Scatter(y=ma_50, name="MA-50"))
    fig.update_layout(title="Reward Evolution", yaxis_title="Reward")
    st.plotly_chart(fig)

if __name__ == "__main__":
    create_live_dashboard()

# Run: streamlit run dashboard.py
```

**Energy Efficiency Analytics**:
```python
# energy_analytics.py - Advanced energy analysis
def analyze_energy_efficiency():
    """Comprehensive energy efficiency analysis"""
    
    # Load performance data
    logs = parse_log_files("logs/")
    
    # Calculate EDP trends
    edp_by_frequency = {}
    for entry in logs:
        freq = entry['frequency']
        energy = entry['energy_delta']
        latency = entry['ttft'] + entry['tpot']
        edp = energy * latency
        
        if freq not in edp_by_frequency:
            edp_by_frequency[freq] = []
        edp_by_frequency[freq].append(edp)
    
    # Find Pareto optimal frequencies
    pareto_frontier = find_pareto_optimal(edp_by_frequency)
    
    # Generate efficiency report
    report = {
        'best_efficiency_freq': min(edp_by_frequency, key=lambda f: np.mean(edp_by_frequency[f])),
        'pareto_frequencies': pareto_frontier,
        'energy_savings': calculate_energy_savings(logs),
        'performance_impact': calculate_performance_impact(logs)
    }
    
    return report
```

## Major Changes and Current Status (v5.0-contextual-bandit)

### Architecture Evolution (v4.0 → v5.0)
**Enhanced Contextual Bandit Features**:
- **Pure Contextual LinUCB**: Maintains frequencies as actions/arms with 7-dimensional workload context
- **Enhanced Action Pruning**: Added extreme frequency instant pruning for early-round optimization (first 20 rounds) and adaptive cascade pruning based on GPU hardware capabilities
- **Optional Memory Frequency Optimization**: Added support for combined core+memory frequency optimization
- **Improved Adaptive Sampling**: Enhanced dual-mode sampling with better maturity-based refinement
- **Advanced Logging System**: GPU-classified logging with detailed round logging and JSON structured data
- **Online Feature Normalization**: Integrated Welford's algorithm for stable feature standardization
- **Optimized Metrics Collection**: Added session reuse and EMA smoothing for improved performance

### Enhanced Smart Action Pruning System (v5.0)
**Intelligent Frequency Management**:
- **Historical performance pruning**: Enhanced with better thresholds and maturity detection (30 rounds)
- **Adaptive cascade pruning**: Dynamic threshold based on GPU hardware capabilities (gpu_max_freq // 2) instead of fixed 800MHz threshold
- **Extreme frequency instant pruning**: NEW - Immediately removes very poor frequencies in early rounds (first 50 rounds, configurable via extreme_pruning_max_rounds)
- **Exploration guarantees**: Enhanced minimum exploration requirements (6 samples)
- **Memory frequency integration**: Supports pruning in combined core+memory frequency optimization
- **Permanent removal**: Enhanced pruning state persistence across model saves/loads

### Mixed Maturity-Based Refinement (v5.0 Enhanced)
**Dual-Strategy Adaptation**:
- **Statistical refinement (immature)**: Enhanced median-based refinement for robust early-stage learning (<100 rounds)
- **Predictive refinement (mature)**: Improved LinUCB model predictions for intelligent refinement (≥100 rounds)
- **Load-aware frequency recommendation**: Percentile-based load classification (p33 percentile for load detection)
- **UCB+EDP hybrid strategy**: Combines UCB potential with historical EDP performance for mature models
- **Emergency SLO refinement**: Immediate frequency space refinement when SLO violations detected (max_freq - 450MHz emergency lower bound)
- **Automatic transition**: Enhanced maturity threshold detection (100 rounds) with better switching logic
- **Best-frequency protection**: Enhanced protection for historical EDP-optimal frequencies
- **Memory frequency support**: Extended refinement strategies to combined frequency optimization

### Top-K Stability Convergence (v5.0 Improved)
**Enhanced Stability Detection**:
- **Action-based stability**: Improved monitoring of top-k most frequently selected actions
- **Configurable parameters**: Enhanced top-k count (default: 3) and stability threshold (default: 60%)
- **Combined frequency support**: Handles both core-only and core+memory frequency actions
- **Faster convergence**: More robust convergence detection with better stability metrics

### Fixed Timing and Enhanced Energy (v5.0)
**Hardcoded Consistency with Enhancements**:
- **0.3-second intervals**: Decision timing remains hardcoded for consistency
- **Automatic energy alignment**: Enhanced energy measurements with improved timing accuracy
- **No manual configuration**: Maintains simplified configuration approach
- **Enhanced idle mode**: Improved idle detection with 210MHz safe frequency reset
- **Memory frequency awareness**: Energy measurement considers combined frequency states

### Optional Memory Frequency Optimization (v5.0 New)
**Combined Frequency Control**:
- **Hardware detection**: Automatic detection of memory frequency control support
- **Combined action space**: Core+memory frequency combinations as single actions
- **Seamless integration**: Works with existing pruning, refinement, and convergence systems
- **Configuration control**: Enable/disable via `enable_memory_frequency_control` setting
- **Fallback support**: Graceful fallback to core-only mode when memory control unavailable