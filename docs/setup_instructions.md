# Installation & Setup Guide

Complete setup instructions for CARLA Lane Keeping Multi-Agent PPO project.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [CARLA Installation](#carla-installation)
3. [Python Environment Setup](#python-environment-setup)
4. [Project Configuration](#project-configuration)
5. [Verification](#verification)
6. [Parameter Tuning Guide](#parameter-tuning-guide)
7. [Common Issues](#common-issues)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 18.04+
- **CPU**: Quad-core 2.5 GHz+
- **RAM**: 8 GB
- **GPU**: NVIDIA GPU with 4GB VRAM (GTX 1060 or better)
- **Storage**: 20 GB free space
- **Python**: 3.7 (exactly - CARLA 0.9.15 requirement)

### Recommended for Multi-Agent
- **CPU**: 6-core 3.0 GHz+ (e.g., i5-12400F)
- **RAM**: 16 GB
- **GPU**: RTX 3060 (12GB) or better
- **Storage**: 50 GB free space (for training logs)

---

## CARLA Installation

### Step 1: Download CARLA

1. Go to [CARLA Releases](https://github.com/carla-simulator/carla/releases)
2. Download **CARLA 0.9.15** for your OS:
   - Windows: `CARLA_0.9.15.zip`
   - Linux: `CARLA_0.9.15.tar.gz`

3. Extract to a permanent location:
   ```
   Windows: C:\Program Files\CARLA_0.9.15\
   Linux: /opt/CARLA_0.9.15/
   ```

### Step 2: Verify CARLA Works

**Windows:**
```bash
cd C:\Program Files\CARLA_0.9.15\WindowsNoEditor
CarlaUE4.exe
```

**Linux:**
```bash
cd /opt/CARLA_0.9.15
./CarlaUE4.sh
```

You should see the CARLA window open. Press `ESC` to quit.

### Step 3: Find Python API Egg File

The egg file is located at:
- **Windows**: `CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg`
- **Linux**: `CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg`

**Save this path** — you'll need it for configuration.

---

## Python Environment Setup

### Step 1: Install Python 3.7

**Critical**: CARLA 0.9.15 requires **exactly Python 3.7**

**Windows:**
1. Download from [python.org](https://www.python.org/downloads/release/python-3710/)
2. Install with "Add to PATH" checked
3. Verify: `python --version` → should show `Python 3.7.x`

**Linux:**
```bash
sudo apt update
sudo apt install python3.7 python3.7-dev python3.7-venv
```

### Step 2: Create Virtual Environment

```bash
cd carla-lane-keeping-multi-agent-ppo

python3.7 -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r docs/requirements.txt
```

**Expected installation time**: 5–10 minutes

### Step 4: Install PyTorch with CUDA (if using GPU)

**For CUDA 11.7** (RTX 30-series):
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

**For CUDA 11.6**:
```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

**CPU only** (not recommended):
```bash
pip install torch==1.13.1 torchvision==0.14.1
```

Verify CUDA:
```python
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

---

## Project Configuration

### Step 1: Clone Repository

```bash
git clone https://github.com/wajdibousnina/carla-lane-keeping-multi-agent-ppo.git
cd carla-lane-keeping-multi-agent-ppo
```

### Step 2: Configure CARLA Path

Open `main_files/lane_keeping_parameters.py` and update:

```python
CARLA_EGG_PATH = "C:\\Program Files\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.15-py3.7-win-amd64.egg"

# Or for Linux:
CARLA_EGG_PATH = "/opt/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
```

> **Important**: Use double backslashes `\\` on Windows!

### Step 3: Configure Training Directory

```python
BASE_PATH = "C:\\your\\preferred\\output\\folder"   # Windows
# Or:
BASE_PATH = "/home/username/carla_training"          # Linux
```

### Step 4: Configure Multi-Agent Settings

```python
class MultiAgentParams:
    ENABLE_MULTI_AGENT = True   # Must be True to activate multi-agent mode
    NUM_AGENTS = 2              # Start with 2; increase only if RAM allows
```

See the [Parameter Tuning Guide](#parameter-tuning-guide) below for full details on every setting.

---

## Verification

### Test 1: CARLA Connection

Start CARLA, then in a Python prompt:

```python
import sys
sys.path.append("YOUR_CARLA_EGG_PATH")
import carla
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
print("✓ Connected to CARLA")
```

### Test 2: Environment Creation

```python
from main_files.carla_lane_keeping_env import CarlaLaneKeepingEnv
env = CarlaLaneKeepingEnv()
# Should print: "✓ CARLA connected successfully"
env.close()
```

### Test 3: Multi-Agent Wrapper

```python
from main_files.multi_agent_wrapper import MultiAgentCarlaWrapper
vec_env = MultiAgentCarlaWrapper(num_agents=2)
# Should print: "✓ Agent 1 ready", "✓ Agent 2 ready", "✓ All 2 agents created!"
vec_env.close()
```

### Test 4: Quick Training Run

```bash
python main_files/training_lane_keeping_ppo.py
```

Reduce `TOTAL_TIMESTEPS` to `5000` for a quick sanity check.

If all tests pass: **✓ Setup complete!**

---

## Parameter Tuning Guide

All training behavior is controlled from a single file: **`main_files/lane_keeping_parameters.py`**. This section explains every configurable section and what to change depending on your goals.

---

### 1. Paths (top of file)

```python
BASE_PATH = "F:\\thesis_carla_lane_keeping"
CARLA_EGG_PATH = "F:\\Program files\\CARLA_0.9.15\\..."
```

These two lines **must** be updated to match your system before anything else will work. `BASE_PATH` is where all output files (models, logs, plots) will be saved.

---

### 2. Environment Parameters

```python
MAX_EPISODE_STEPS = 2000
TARGET_SPEED = 8.0       # m/s ≈ 30 km/h
MIN_SPEED_THRESHOLD = 1.0
```

| Parameter | What it does | When to change it |
|-----------|-------------|-------------------|
| `MAX_EPISODE_STEPS` | Maximum steps before episode is cut off | Increase if agents need more time; decrease to speed up training cycles |
| `TARGET_SPEED` | The speed agents are rewarded for maintaining | Increase for highway-like scenarios; decrease for tight/curved roads |
| `MIN_SPEED_THRESHOLD` | Agents are penalized if speed falls below this | Prevents agents from standing still; raise if agents are lazy |
| `SPAWN_POINT_INDEX` | `None` = random spawn per agent; integer = fixed spawn | Use `None` (default) in multi-agent mode so agents spawn at different locations |
| `WEATHER_RANDOMIZATION` | Randomize weather each episode | Enable later in training to improve generalization |

---

### 3. Multi-Agent Parameters — `MultiAgentParams`

This section is specific to the multi-agent version and controls how the parallel agents are set up.

```python
class MultiAgentParams:
    ENABLE_MULTI_AGENT = True
    NUM_AGENTS = 2
    AGENT_INTERACTION = False
    SPAWN_SPACING = 50.0
    RANDOM_SPAWN = True
    ASYNC_MODE = False
```

| Parameter | What it does | When to change it |
|-----------|-------------|-------------------|
| `ENABLE_MULTI_AGENT` | Activates or deactivates multi-agent mode | Set to `True` to use this wrapper; `False` falls back to single-agent behavior |
| `NUM_AGENTS` | Number of vehicles spawned simultaneously | Start with `2`. Only increase to 3+ if you have ≥16 GB RAM and a high-end GPU |
| `AGENT_INTERACTION` | `False` = ghost mode (agents pass through each other) | Keep `False`; setting `True` enables physical collisions between agents (experimental) |
| `SPAWN_SPACING` | Minimum distance in meters between agent spawn points | Increase if agents are spawning too close and interfering with each other |
| `RANDOM_SPAWN` | Each agent spawns at a random valid location | Keep `True` for training; set `False` only for controlled evaluation |
| `ASYNC_MODE` | Use asynchronous CARLA ticking | Keep `False` — synchronous mode is significantly more stable for multi-agent training |

**RAM requirements by agent count:**

| Agents | Minimum RAM | Recommended RAM |
|--------|------------|----------------|
| 2 | 12 GB | 16 GB |
| 3 | 16 GB | 24 GB |
| 4+ | Not recommended unless on a workstation | — |

> **Tip:** Always start with `NUM_AGENTS = 2` and monitor Task Manager (Windows) or `htop` (Linux) during the first episode. If RAM usage exceeds 85%, do not increase the agent count.

---

### 4. Reward Function — `RewardParams`

This is the most important section for shaping agent behavior. Every value here directly changes what agents "want" to do.

```python
class RewardParams:
    LANE_CENTER_WEIGHT = -1.0
    FORWARD_PROGRESS_WEIGHT = 20.0
    ORIENTATION_WEIGHT = -0.5
    SPEED_REWARD_WEIGHT = 5.0
    MIN_SPEED_PENALTY = -5.0
    COLLISION_PENALTY = -50.0
    LANE_DEPARTURE_PENALTY = -20.0
    REVERSE_PENALTY = -5.0
    STEERING_SMOOTHNESS_WEIGHT = -0.1
    THROTTLE_SMOOTHNESS_WEIGHT = -0.05
    COMPLETION_BONUS = 60.0
```

**How to read these values:** positive = reward (agent wants more of it), negative = penalty (agent tries to avoid it). Larger magnitude = stronger signal.

| Parameter | Effect | Tuning Tip |
|-----------|--------|-----------|
| `LANE_CENTER_WEIGHT` | Penalizes distance from lane center | Make more negative (e.g., `-2.0`) if agents drift; don't make so large agents stop moving |
| `FORWARD_PROGRESS_WEIGHT` | Rewards covering distance forward | The main driving force — if agents are lazy/stationary, increase this |
| `ORIENTATION_WEIGHT` | Penalizes misalignment with road direction | Increase magnitude if agents drive at an angle; too high causes oscillation |
| `SPEED_REWARD_WEIGHT` | Rewards driving close to `TARGET_SPEED` | Reduce if agents overspeed dangerously |
| `MIN_SPEED_PENALTY` | Penalizes being nearly stopped | Increase magnitude if agents stop frequently to "avoid" other penalties |
| `COLLISION_PENALTY` | Large one-time penalty on collision | Keep large relative to other rewards so collision is treated as a hard failure |
| `LANE_DEPARTURE_PENALTY` | Penalty when vehicle fully exits the lane | Should be large but less than `COLLISION_PENALTY` |
| `STEERING_SMOOTHNESS_WEIGHT` | Penalizes abrupt steering changes | Increase magnitude for smoother steering; too large may prevent sharp-turn response |
| `COMPLETION_BONUS` | Bonus for finishing the episode without failure | Helps the agent learn long-horizon behavior |

**Common behavior problems and reward fixes:**

- **Agents stop or slow down too much** → Increase `FORWARD_PROGRESS_WEIGHT` and `MIN_SPEED_PENALTY` magnitude
- **Agents drive in a zigzag** → Increase `ORIENTATION_WEIGHT` and `STEERING_SMOOTHNESS_WEIGHT` magnitude
- **Agents ignore lane boundaries** → Increase `LANE_CENTER_WEIGHT` and `LANE_DEPARTURE_PENALTY` magnitude
- **Agents drive recklessly fast** → Reduce `SPEED_REWARD_WEIGHT` or reduce `TARGET_SPEED`

---

### 5. Observation Space — `ObservationParams`

```python
class ObservationParams:
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    FRAME_STACK = 4
    SPEED_NORMALIZATION = 15.0
    DISTANCE_NORMALIZATION = 4.0
```

| Parameter | What it does | When to change it |
|-----------|-------------|-------------------|
| `IMAGE_HEIGHT` / `IMAGE_WIDTH` | Resolution of camera frames fed to the network | Reduce to `96×96` if running out of VRAM — critical with multiple agents |
| `FRAME_STACK` | How many consecutive frames are stacked as one observation | 4 is standard; reduce to 3 to save memory in multi-agent mode |
| `SPEED_NORMALIZATION` | Divides raw speed to normalize to ~[0,1] range | Set to roughly the max speed you expect the vehicle to reach |
| `DISTANCE_NORMALIZATION` | Divides lane offset distance to normalize | Set to roughly the lane width (4m is typical) |

> **Multi-agent note:** With `NUM_AGENTS = 2`, memory usage doubles. If you encounter VRAM errors, the first thing to reduce is `IMAGE_HEIGHT` and `IMAGE_WIDTH`.

> **Important:** If you change `IMAGE_HEIGHT`, `IMAGE_WIDTH`, or `FRAME_STACK`, the model architecture changes and any saved checkpoint becomes incompatible. Always start fresh training after changing these.

---

### 6. Action Space — `ActionParams`

```python
class ActionParams:
    STEERING_SMOOTHING = 0.3
    THROTTLE_SMOOTHING = 0.3
    BRAKE_SMOOTHING = 0.3
    MAX_STEERING_CHANGE = 0.5
    MAX_THROTTLE_CHANGE = 0.8
    MAX_BRAKE_CHANGE = 0.8
```

Action smoothing blends the new action with the previous one to avoid sudden jerks. A value of `0.3` means: `actual_action = 0.3 × new + 0.7 × previous`.

| Parameter | What it does | When to change it |
|-----------|-------------|-------------------|
| `STEERING_SMOOTHING` | How quickly steering responds to policy output | Increase (towards 1.0) for more responsive steering; decrease for smoother response |
| `MAX_STEERING_CHANGE` | Maximum steering change allowed per step | Reduce to prevent sudden swerves; increase for tight-turn scenarios |
| `MAX_THROTTLE_CHANGE` | Maximum throttle change per step | Reduce for smoother acceleration profile |

---

### 7. PPO Hyperparameters — `PPOParams`

```python
class PPOParams:
    LEARNING_RATE = 5e-4
    LR_SCHEDULE = 'linear'
    CLIP_RANGE = 0.2
    ENTROPY_COEF = 0.02
    ENTROPY_COEF_FINAL = 0.005
    N_STEPS = 4096
    BATCH_SIZE = 512
    N_EPOCHS = 4
    GAE_LAMBDA = 0.95
    GAMMA = 0.99
    NET_ARCH = [256, 256]
```

| Parameter | What it does | When to change it |
|-----------|-------------|-------------------|
| `LEARNING_RATE` | How fast the network weights update | Lower (e.g., `1e-4`) if training is unstable; raise (e.g., `1e-3`) if too slow |
| `LR_SCHEDULE` | `'linear'` decays LR to 0 over training; `'constant'` keeps it fixed | Use `'linear'` for most cases |
| `CLIP_RANGE` | Limits how much the policy changes per update (PPO core) | Keep at `0.2`; lower to `0.1` for more conservative updates |
| `ENTROPY_COEF` | Exploration bonus — encourages trying diverse actions | Increase if agents get stuck in repetitive behavior early on |
| `ENTROPY_COEF_FINAL` | Entropy after annealing — late-stage exploration | Keep low; agents should exploit learned policy by end of training |
| `N_STEPS` | Steps collected per agent before each update | With `N` agents, the policy sees `N × N_STEPS` experiences per update — very efficient |
| `BATCH_SIZE` | Mini-batch size during the update step | Reduce to `256` if getting CUDA out-of-memory errors |
| `N_EPOCHS` | How many times each collected batch is reused per update | Keep at 4; increasing can help with sample efficiency |
| `GAMMA` | Discount factor — how much the agent values future rewards | `0.99` is good for long-horizon lane keeping |
| `NET_ARCH` | Hidden layer sizes of the policy network | Increase to `[512, 512]` for more complex scenarios |

> **Multi-agent training note:** With multiple agents feeding experience simultaneously, the effective experience per update is `NUM_AGENTS × N_STEPS`. This means the policy updates on a much richer and more diverse batch — one of the key advantages of multi-agent training.

---

### 8. Training Parameters — `TrainingParams`

```python
class TrainingParams:
    TOTAL_TIMESTEPS = 200_000
    SAVE_FREQ = 10_000
    EVAL_FREQ = 5_000
    N_EVAL_EPISODES = 5
    PATIENCE = 20_000
```

| Parameter | What it does | When to change it |
|-----------|-------------|-------------------|
| `TOTAL_TIMESTEPS` | Total environment steps before training ends | Increase (500k–1M) for better final performance |
| `SAVE_FREQ` | How often (in steps) a model checkpoint is saved | Decrease for finer recovery options; increase to save disk space |
| `EVAL_FREQ` | How often the model is evaluated on test episodes | Decrease for more frequent feedback |
| `N_EVAL_EPISODES` | Number of episodes used in each evaluation | Increase for a more reliable average |
| `PATIENCE` | Early stopping: stops if no improvement after this many steps | Increase if training needs more time to improve |

---

### 9. Curriculum Learning — `CurriculumParams`

```python
class CurriculumParams:
    ENABLE_CURRICULUM = True
    STAGE_1_EPISODES = 200    # Straight roads, no traffic
    STAGE_2_EPISODES = 300    # Slight curves, some weather
    STAGE_3_EPISODES = inf    # Full complexity
```

Curriculum learning starts agents in easy conditions and progressively increases difficulty. To disable and train on full complexity from the start:
```python
ENABLE_CURRICULUM = False
```

If agents struggle on Stage 1, increase `STAGE_1_EPISODES` to give them more time on simple scenarios before advancing.

---

## Common Issues

### Issue 1: "ImportError: No module named 'carla'"

**Fix**: Update `CARLA_EGG_PATH` in `lane_keeping_parameters.py` with the full absolute path, using double backslashes on Windows.

### Issue 2: "Connection refused" when connecting to CARLA

**Fix**: Start `CarlaUE4.exe`, wait 15–20 seconds, then retry.

### Issue 3: "CUDA out of memory"

With multiple agents this is more likely. Try:
```python
class PPOParams:
    BATCH_SIZE = 256

class ObservationParams:
    IMAGE_HEIGHT = 96
    IMAGE_WIDTH = 96
```

Also reduce `NUM_AGENTS` to 2 if you are on a GPU with less than 8 GB VRAM.

### Issue 4: Python version mismatch

CARLA 0.9.15 requires **exactly Python 3.7**. You must create a venv using Python 3.7.

### Issue 5: Training very slow

1. Verify GPU is in use: `python -c "import torch; print(torch.cuda.is_available())"`
2. Launch CARLA in low resolution: `CarlaUE4.exe -ResX=640 -ResY=480 -quality-level=Low`
3. Always use `-quality-level=Low` when running multiple agents

### Issue 6: Agent spawn failures in multi-agent mode

Each agent spawns with a 2-second delay to avoid CARLA conflicts. If a spawn still fails:
```python
SPAWN_POINT_INDEX = None   # Ensure random spawning is active
SPAWN_SPACING = 80.0       # Increase spacing between agents
```

### Issue 7: Training crashes after many steps

**Fix**: Reduce `FRAME_STACK` from 4 to 3, and consider restarting the CARLA server every 100k steps for long training runs.

---

## Next Steps

After successful setup:

1. **Quick test**: Run with `TOTAL_TIMESTEPS = 5_000` and `NUM_AGENTS = 2` to verify the full pipeline
2. **Monitor training**: Open TensorBoard to watch reward curves across agents
3. **Tune reward weights**: Adjust `RewardParams` based on observed behavior
4. **Scale up**: Increase to full 200k+ timesteps once things look stable

---

## Support

If you encounter issues not covered here, open an issue on the [GitHub repository](https://github.com/wajdibousnina/carla-lane-keeping-multi-agent-ppo/issues) with:
- The full error message
- Your system specs (OS, GPU, RAM)
- Steps to reproduce the problem
