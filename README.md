# AGFT: vLLM Multi-Armed Bandit GPU Autoscaler

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![vLLM](https://img.shields.io/badge/vLLM-Compatible-orange.svg)](https://github.com/vllm-project/vllm)

An intelligent GPU frequency optimization system that uses Contextual LinUCB Multi-Armed Bandit algorithms to automatically adjust NVIDIA GPU clock frequencies in real-time, minimizing Energy-Delay Product (EDP) while maintaining quality of service constraints for vLLM inference servers.

## Overview

AGFT uses a pure contextual bandit approach where:
- **Actions**: GPU frequencies (core and/or memory frequencies)  
- **Context**: 7-dimensional workload feature vectors (queue depth, cache usage, token rates, etc.)
- **Reward**: Normalized Energy-Delay Product with SLO constraints
- **Algorithm**: Contextual LinUCB with smart action pruning and adaptive frequency discovery

**Key capabilities**:
- Real-time energy optimization with 0.3s decision cycles
- Dual-mode operation: SLO-aware vs EDP-optimal
- Smart frequency pruning with cascade and extreme frequency removal
- Combined core and memory frequency optimization
- Automatic model convergence detection and exploitation

## Prerequisites

- NVIDIA GPU with frequency control support (`nvidia-smi -lgc` capability)
- Python 3.8+ with required packages
- vLLM server running with Prometheus metrics endpoint (`/metrics`)
- CUDA drivers and nvidia-smi access
- Sudo privileges for GPU frequency control

## Installation

```bash
# Clone repository
git clone https://github.com/SusCom-Lab/AGFT
cd AGFT

# Create conda environment
conda create -n agft python=3.9 -y
conda activate agft

# Install dependencies
pip install numpy pynvml requests pyyaml
```

## Quick Start

### 1. Configure vLLM Server

Ensure your vLLM server is running with Prometheus metrics enabled:

```bash
# Start vLLM with metrics
vllm serve your-model \
  --port 8000 \
  --disable-log-requests
```

### 2. Configure AGFT

Edit `config/config.yaml`:

```yaml
# vLLM connection
vllm:
  prometheus_url: "http://127.0.0.1:8001/metrics"

# GPU settings
gpu:
  device_id: 0                    # GPU device ID


# Control mode
control:
  ignore_slo: false               # false=SLO-aware mode, true=EDP-optimal mode
  ttft_limit: 2.0                 # Time-to-first-token SLO limit (seconds)
  tpot_limit: 0.25                # Time-per-output-token SLO limit (seconds)

```

### 3. Run AGFT

```bash
# Basic run (requires sudo for GPU frequency control)
sudo -E /path/to/your/conda/envs/agft/bin/python -m src.main --config config/config.yaml

# Example with specific conda path
sudo -E /home/ldaphome/colin/.conda/envs/vllm/bin/python -m src.main --config config/config.yaml

# With debug logging
sudo -E /path/to/your/conda/envs/agft/bin/python -m src.main --log-level DEBUG

# Reset model and start fresh
sudo -E /path/to/your/conda/envs/agft/bin/python -m src.main --reset-model
```

**Note**: Replace `/path/to/your/conda/envs/agft/bin/python` with your actual conda environment Python path. You can find it with `conda info --envs` or `which python` (while in the activated environment).

## Configuration Details

### Core Control Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `control.ignore_slo` | Operating mode | `false` | `false` (SLO-aware), `true` (EDP-optimal) |
| `control.ttft_limit` | Time-to-first-token SLO limit | `2.0` | Seconds |
| `control.tpot_limit` | Time-per-output-token SLO limit | `0.25` | Seconds |
| `control.data_collection_mode` | Data collection only (no frequency changes) | `false` | `true`/`false` |

### GPU Control

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gpu.device_id` | Target GPU device ID | `0` |
| `gpu.frequency_step` | Base frequency step size | `15` MHz |
| `gpu.auto_step` | Auto-detect native GPU frequency steps | `true` |
| `gpu.enable_memory_frequency_control` | Enable memory frequency optimization | `false` |

### LinUCB Algorithm

| Parameter | Description | Default |
|-----------|-------------|---------|
| `linucb.initial_alpha` | Exploration parameter (higher = more exploration) | `10.0` |
| `linucb.enable_action_pruning` | Smart frequency pruning | `true` |
| `linucb.enable_extreme_pruning` | Instant removal of very poor frequencies | `true` |
| `linucb.pruning_threshold` | Pruning threshold (reward gap multiplier) | `4.0` |

## Performance

- **Energy savings**: 15-30% EDP reduction vs fixed frequency
- **Convergence**: Typically 100-200 decision rounds  
- **SLO compliance**: Maintained during optimization (SLO-aware mode)
- **Decision latency**: Fixed 0.3s cycles for real-time operation

## Advanced Features

### Operating Modes

**SLO-Aware Mode** (`ignore_slo: false`):
- Maintains SLO constraints while optimizing energy
- Safe frequency exploration with violation recovery
- Recommended for production environments

**EDP-Optimal Mode** (`ignore_slo: true`):
- Pure energy-delay product optimization
- Full frequency domain exploration
- Maximum efficiency for research environments

### Smart Frequency Management

- **Action Pruning**: Removes consistently poor frequencies based on historical performance
- **Extreme Pruning**: Instantly removes very poor frequencies in early rounds
- **Cascade Pruning**: Hardware-aware frequency removal using adaptive thresholds
- **Memory Optimization**: Combined core and memory frequency tuning (when supported)

## Monitoring

```bash
# Monitor GPU frequency changes
watch -n 1 'nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits'

# View live logs
tail -f logs/*/vllm_gpu_autoscaler_*.log

# Check model convergence status
python -c "
import json
with open('data/models/model_metadata.json') as f:
    meta = json.load(f)
    print(f'Phase: {meta.get(\"phase\", \"unknown\")}')
    print(f'Converged: {meta.get(\"converged\", \"unknown\")}')
"
```

## Testing

```bash
# Test individual components
python -m src.gpu_controller
python -m src.metrics_collector
python -m src.feature_extractor

# Validate configuration
python -c "import yaml; yaml.safe_load(open('config/config.yaml')); print('✅ Config valid')"

# Check GPU capabilities
nvidia-smi --query-gpu=name,driver_version,power.management --format=csv
nvidia-smi -q -d SUPPORTED_CLOCKS
```

## Troubleshooting

**GPU Permission Errors**:
```bash
# Add user to video group and test frequency control
sudo usermod -a -G video $USER
nvidia-smi -i 0 -lgc 1000
```

**vLLM Connection Issues**:
```bash
# Verify vLLM metrics endpoint is accessible
curl http://localhost:8001/metrics | grep "vllm"
```

**Model Loading Issues**:
```bash
# Reset corrupted models
rm data/models/*.pkl
sudo -E $(which python) -m src.main --reset-model
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/name`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature/name`)
5. Open a Pull Request


