# vLLM Multi-Armed Bandit GPU Autoscaler

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![vLLM](https://img.shields.io/badge/vLLM-Compatible-orange.svg)](https://github.com/vllm-project/vllm)

An intelligent GPU frequency optimization system that uses **Contextual LinUCB Multi-Armed Bandit** algorithms to automatically adjust NVIDIA GPU clock frequencies in real-time, minimizing Energy-Delay Product (EDP) while maintaining quality of service constraints for vLLM inference servers.

## 🎯 Key Features

- **🧠 Contextual LinUCB Algorithm**: Pure contextual bandit where frequencies are arms/actions and workload features serve as context
- **⚡ Real-Time Energy Optimization**: Direct GPU power measurement with fixed 0.3s decision cycles
- **🎛️ Adaptive Frequency Discovery**: Intelligent exploration with dual-mode sampling (SLO-aware vs EDP-optimal)
- **✂️ Smart Action Pruning**: Advanced frequency pruning with adaptive cascade and extreme frequency instant removal
- **🔄 Mixed Maturity-Based Refinement**: Statistical refinement for immature models, predictive refinement for mature models
- **💾 Optional Memory Frequency Control**: Combined core and memory frequency optimization when hardware supports it
- **📊 GPU-Classified Logging**: Automatic log organization by GPU model with detailed performance tracking

## 🏗️ Architecture Overview

The system consists of 8 main components working in a control loop:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   vLLM Server   │───▶│ Metrics Collector │───▶│ Feature Extractor│
│ (Prometheus API)│    │  (Session Reuse)  │    │ (7D + Welford)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GPU Controller│◀───│  Main Controller  │◀───│ Contextual LinUCB│
│ (Adaptive Freq) │    │  (Orchestration)  │    │ (Action Pruning) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       ▲
         ▼                       ▼                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│Adaptive Sampler │    │ Reward Calculator│───▶│     Logger      │
│ (Dual-Mode)     │    │   (EDP-Based)    │    │ (GPU-Classified)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **NVIDIA GPU** with frequency control support
- **Python 3.8+** with required packages
- **vLLM server** running with Prometheus metrics endpoint
- **CUDA drivers** and nvidia-smi access


### Installation (优化版)

1. 克隆代码库 (Clone the Repository)

首先，克隆 `AGFT` 的代码库并进入项目目录。
git clone https://github.com/SusCom-Lab/AGFT
cd AGFT
2. 创建并激活 Conda 环境 (Create and Activate Conda Environment)
为了保持项目依赖的整洁，我们强烈建议创建一个新的 Conda 环境。这里我们创建一个名为 agft 的环境，并指定 Python 版本为 3.9 (您可以根据项目需求选择其他版本)。

创建一个名为 agft 的新环境
conda create -n agft python=3.9 -y

激活该环境
conda activate agft
重要提示: 在后续所有操作前，请确保您已经激活了 agft 环境。您会看到命令行提示符前出现 (agft) 字样。

3. 安装依赖项 (Install Dependencies)
在激活的 Conda 环境中，使用 pip 来安装所有必需的 Python 包。

pip install numpy pynvml requests pyyaml matplotlib seaborn scipy
您也可以考虑将这些依赖项整理到一个 requirements.txt 文件中，然后通过 pip install -r requirements.txt 来安装，这样更便于管理。

4. 验证环境与硬件 (Verify Environment and Hardware)
完成安装后，执行以下命令来验证您的 GPU 是否被正确识别以及相关组件是否工作正常。




### Basic Usage

```bash
# Run the autoscaler (recommended method)
sudo -E [your conda path] -m src.main

# Run with debug logging
sudo -E [your conda path] -m src.main --log-level DEBUG

# Run with fresh model (ignore existing models)
sudo -E [your conda path] -m src.main --reset-model

# Run with custom configuration
sudo -E [your conda path] -m src.main --config config/config.yaml

# example
sudo -E /home/ldaphome/colin/.conda/envs/vllm/bin/python -m src.main --config config/config.yaml
```

### Configuration

Edit `config/config.yaml` to customize behavior:

```yaml
# Essential settings
control:
  ignore_slo: false           # SLO-aware mode (false) vs EDP-optimal mode (true)
  ttft_limit: 2.0            # Time-to-first-token limit (seconds)
  tpot_limit: 0.25           # Time-per-output-token limit (seconds)

gpu:
  auto_step: true            # Auto-detect GPU frequency capabilities
  frequency_step: 15         # Base frequency step size (MHz)
  enable_memory_frequency_control: false  # Enable memory freq optimization

linucb:
  initial_alpha: 10.0        # Exploration parameter
  enable_action_pruning: true # Enable smart frequency pruning
  enable_extreme_pruning: true # Enable extreme frequency instant pruning
```

## 📈 Performance & Results

### Energy Efficiency Gains
- **15-30% EDP reduction** compared to fixed frequency operation
- **Automatic adaptation** to varying workload patterns
- **SLO compliance** maintained during optimization

### Adaptive Learning
- **Fast convergence** typically within 100-200 decision rounds
- **Intelligent exploration** with declining alpha decay
- **Robust performance** across different GPU models and workloads

## 🔧 Advanced Features

### Dual-Mode Adaptive Sampling

**SLO-Aware Mode** (`ignore_slo: false`):
- High-to-low frequency search with safety prioritization
- Automatic SLO boundary detection and violation recovery
- Mixed refinement strategies based on model maturity

**EDP-Optimal Mode** (`ignore_slo: true`):
- Pure energy-delay product optimization
- Full frequency domain exploration with reward-driven refinement
- Ideal for research environments and maximum efficiency

### Smart Action Pruning

- **Historical Performance Pruning**: Removes consistently poor frequencies
- **Adaptive Cascade Pruning**: Uses `gpu_max_freq // 2` threshold instead of fixed values
- **Extreme Frequency Instant Pruning**: Immediately removes very poor frequencies in first 50 rounds
- **Exploration Protection**: Ensures minimum exploration before pruning eligibility

### Memory Frequency Optimization

When supported by hardware:
- **Combined optimization** of core and memory frequencies
- **Automatic detection** of memory frequency control capabilities
- **Graceful fallback** to core-only mode when unavailable

## 📊 Monitoring & Analysis

### Real-Time Monitoring

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


## 🛠️ Development & Testing

### Component Testing

```bash
# Test individual components
python -m src.gpu_controller          # GPU control and adaptive sampling
python -m src.metrics_collector       # vLLM metrics collection
python -m src.feature_extractor       # Contextual feature extraction

# Test LinUCB algorithm
python -c "from src.contextual_bandit import ContextualLinUCB; m=ContextualLinUCB(7); print('✅ Contextual LinUCB OK')"

# Test memory frequency support
python -c "from src.gpu_controller import GPUController; g=GPUController(enable_memory_frequency_control=True); print('Memory support:', g.memory_frequency_supported)"
```

### Configuration Validation

```bash
# Validate configuration syntax
python -c "import yaml; yaml.safe_load(open('config/config.yaml')); print('✅ Config valid')"

# Test GPU capabilities
nvidia-smi --query-gpu=name,driver_version,power.management --format=csv
nvidia-smi -q -d SUPPORTED_CLOCKS
```

## 📋 System Requirements

### Hardware
- **NVIDIA GPU** with power management support
- **Frequency control capabilities** via nvidia-smi
- **NVML API access** for energy measurement

### Software
- **Python 3.8+**
- **NVIDIA drivers** (R470+ recommended)
- **vLLM server** with Prometheus metrics enabled
- **CUDA toolkit** (for optimal performance)

### Python Dependencies
```
numpy>=1.21.0
pynvml>=11.0.0
requests>=2.25.0
pyyaml>=5.4.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.7.0
```

## 🔍 Troubleshooting

### Common Issues

**GPU Permission Errors**:
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Test frequency control
nvidia-smi -i 0 -lgc 1000
```

**vLLM Connection Issues**:
```bash
# Verify vLLM is accessible
curl http://localhost:8001/metrics | grep "vllm"
```

**Model Loading Issues**:
```bash
# Check model files
ls -la data/models/contextual_linucb_model_*.pkl

# Verify model integrity
python -c "import pickle; pickle.load(open('data/models/latest_model.pkl', 'rb')); print('✅ Model OK')"
```

For detailed troubleshooting, see [CLAUDE.md](CLAUDE.md#troubleshooting).

## 📖 Documentation

- **[CLAUDE.md](CLAUDE.md)** - Comprehensive technical documentation
- **[Configuration Guide](CLAUDE.md#critical-configuration-parameters)** - Detailed parameter explanations
- **[Development Guide](CLAUDE.md#development-commands)** - Development workflows and testing
- **[Performance Tuning](CLAUDE.md#performance-optimization)** - GPU-specific and workload-specific optimization

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/vllm_mab.git
cd vllm_mab

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/
```


## 🏆 Key Innovations

1. **Pure Contextual Bandit Design**: Frequencies as actions, workload features as context
2. **Adaptive Cascade Pruning**: Hardware-aware frequency pruning using `gpu_max_freq // 2`
3. **Mixed Maturity-Based Refinement**: Different strategies for immature vs mature models
4. **Load-Aware Frequency Recommendation**: Percentile-based load classification
5. **Emergency SLO Recovery**: Immediate frequency adjustment for SLO violations
6. **Online Feature Normalization**: Welford's algorithm for stable feature standardization

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **vLLM Team** for the excellent inference framework
- **NVIDIA** for NVML API and GPU management tools
- **Multi-Armed Bandit Research Community** for theoretical foundations
- **Open Source Contributors** for various dependencies and tools

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/vllm_mab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/vllm_mab/discussions)
- **Documentation**: [CLAUDE.md](CLAUDE.md)

---

**⭐ Star this repo if you find it useful!**

*Built with ❤️ for the energy-efficient AI community*
