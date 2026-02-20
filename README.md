# CARLA Lane Keeping — Multi-Agent PPO

## Overview

This project extends the [single-agent lane-keeping system](https://github.com/YOUR_USERNAME/carla-lane-keeping-ppo-single-agent) to a **multi-agent setting**, where multiple vehicles are trained in parallel within the same CARLA world. Each agent operates independently (no inter-agent coordination), sharing the environment but learning from its own experience stream — effectively providing richer and more diverse training data to the PPO policy.

The work is part of my MSc thesis exploring deep reinforcement learning for autonomous vehicle control.

---

## Key Features

- **Multiple independent agents** running in the same CARLA world simultaneously
- **Custom `VecEnv` wrapper** (`MultiAgentCarlaWrapper`) compatible with Stable-Baselines3
- **Ghost mode**: agents do not collide with each other (physically independent)
- All single-agent features are preserved:
  - Stacked-frame visual observations (4 × 128×128 RGB)
  - Continuous action space: steering, throttle, brake
  - Shaped reward function with collision, lane departure, and smoothness terms
  - PPO with linear LR schedule and entropy annealing
  - TensorBoard logging and model checkpointing

---

## Project Structure

```
carla-lane-keeping-multi-agent-ppo/
├── docs/
│   ├── requirements.txt
│   └── setup_instructions.md
├── main_files/
│   ├── carla_lane_keeping_env.py      # Custom CARLA Gymnasium environment (single agent)
│   ├── multi_agent_wrapper.py         # VecEnv wrapper for multi-agent training
│   ├── training_lane_keeping_ppo.py   # PPO training script (multi-agent enabled)
│   └── lane_keeping_parameters.py     # All hyperparameters including MultiAgentParams
├──LICENSE
└── README.md
```

---

## Multi-Agent Architecture

The `MultiAgentCarlaWrapper` inherits from Stable-Baselines3's `VecEnv`, making it a drop-in replacement for vectorized environments. Each agent is a full `CarlaLaneKeepingEnv` instance with its own vehicle spawned in the CARLA world.

```
PPO Policy
    │
    ▼
MultiAgentCarlaWrapper (VecEnv)
    ├── CarlaLaneKeepingEnv  ──► Vehicle 1  (CARLA World)
    ├── CarlaLaneKeepingEnv  ──► Vehicle 2  (CARLA World)
    └── CarlaLaneKeepingEnv  ──► Vehicle N  (CARLA World)
```

### Multi-Agent Parameters

| Parameter | Value |
|-----------|-------|
| Number of Agents | 2 (configurable) |
| Agent Interaction | Disabled (ghost mode) |
| Spawn Strategy | Random, spaced 50m apart |
| Sync Mode | Synchronous (stable) |

---

## Environment Details

| Parameter | Value |
|-----------|-------|
| Simulator | CARLA 0.9.15 |
| Map | Town04 |
| Target Speed | 8.0 m/s (~30 km/h) |
| Max Episode Steps | 2000 |
| Observation | 4 × 128×128 RGB frames + 6D vehicle state vector |
| Action Space | Continuous: `[steering, throttle, brake]` |
| Framework | Stable-Baselines3 (PPO) |

### Reward Function Summary

| Component | Weight |
|-----------|--------|
| Lane center distance | −1.0 |
| Forward progress | +20.0 |
| Speed maintenance | +5.0 |
| Orientation error | −0.5 |
| Collision | −50.0 |
| Lane departure | −20.0 |
| Action smoothness | −0.1 / −0.05 |
| Episode completion bonus | +60.0 |

---

## PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 5e-4 (linear schedule) |
| Clip Range | 0.2 |
| Entropy Coeff | 0.02 → 0.005 (annealed) |
| GAE Lambda | 0.95 |
| Discount (γ) | 0.99 |
| N Steps | 4096 |
| Batch Size | 512 |
| Epochs per Update | 4 |
| Network | [256, 256] + tanh |
| Total Timesteps | 200,000 |

---

## Requirements

- Windows 10/11 (CARLA runs natively on Windows)
- Python 3.7
- CARLA 0.9.15
- CUDA-capable GPU (recommended — multiple agents are GPU-intensive)
- Sufficient RAM (≥ 16 GB recommended for 2+ agents)

### Python Dependencies

```bash
pip install stable-baselines3[extra]
pip install gymnasium
pip install numpy
pip install tensorboard
```

---

## Getting Started

### 1. Install CARLA

Download [CARLA 0.9.15](https://github.com/carla-simulator/carla/releases/tag/0.9.15) and extract it.

### 2. Configure Paths and Agent Count

Open `lane_keeping_parameters.py` and update:

```python
# Paths
BASE_PATH = "your\\path\\to\\project"
CARLA_EGG_PATH = "your\\path\\to\\carla-0.9.15-py3.7-win-amd64.egg"

# Multi-agent settings
class MultiAgentParams:
    ENABLE_MULTI_AGENT = True
    NUM_AGENTS = 2          # Increase if your hardware supports it
```

### 3. Launch CARLA Server

```bash
cd "C:\path\to\CARLA_0.9.15\WindowsNoEditor"
CarlaUE4.exe -quality-level=Low -benchmark -fps=20
```

> ⚠️ **Important:** Use `-quality-level=Low` when running multiple agents to reduce GPU load.

### 4. Run Training

```bash
python training_lane_keeping_ppo.py
```

The wrapper will automatically spawn all agents with a 2-second delay between each to avoid CARLA spawn conflicts.

### 5. Monitor Training (Optional)

```bash
tensorboard --logdir ./logs
```

---

## Training Notes

- Each agent auto-resets independently when its episode ends
- Failed agent steps are handled gracefully with dummy observations — training continues uninterrupted
- Models are saved every **10,000 timesteps**
- Evaluation runs every **5,000 timesteps** over 5 episodes
- To add more agents, increase `MultiAgentParams.NUM_AGENTS` — but monitor your VRAM and RAM usage

```
For detailed guidance on tuning all parameters, see [instructions.md](docs/setup_instructions.md).
```

## Single-Agent vs Multi-Agent Comparison

| Feature | Single Agent | Multi Agent |
|---------|-------------|-------------|
| Vehicles in world | 1 | 2+ |
| Experience diversity | Lower | Higher |
| Training speed | Baseline | Faster (parallel data) |
| Hardware requirement | Moderate | Higher |
| VecEnv compatible | No | Yes |

---

## Related Project

This is the **multi-agent** extension. The original **single-agent** version is available here:

👉 [carla-lane-keeping-single-agent-ppo](https://github.com/wajdibousnina/carla-lane-keeping-single-agent-ppo/)

---

## Citation / Reference

If you use this code in your research, please cite:

```bibtex
@misc{carla_multiagent_ppo,
  author = {Wajdi Bousnina},
  title = {Carla Lane Keeping Multi-Agent PPO},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/wajdibousnina/carla-lane-keeping-multi-agent-ppo}
}
```

---

## Contact

**Wajdi Bousnina** - wajdibousnina8@gmail.com

Project Link: [https://github.com/wajdibousnina/carla-lane-keeping-multi-agent-ppo](https://github.com/wajdibousnina/carla-lane-keeping-multi-agent-ppo)
