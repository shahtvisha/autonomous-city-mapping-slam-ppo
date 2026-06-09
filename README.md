# Active SLAM + PPO - Autonomous City Mapping

An autonomous agent that maps an unknown urban environment from scratch using only local LiDAR observations (no GPS, no prior map)

**Live demo → [huggingface.co/spaces/TvishaShah/active-slam](https://huggingface.co/spaces/TvishaShah/active-slam)**

![Agent run](https://huggingface.co/spaces/TvishaShah/active-slam/resolve/main/assets/run_centre.gif)
*Left: SLAM belief map built from sensor data alone. Right: ground truth + agent trajectory. Kendall Square, Boston using OpenStreetMap data.*



## Results

| Metric | Value |
|---|---|
| Peak region coverage | **89.7%** (977 steps) |
| Average map accuracy | **91.0%** across 10 episodes |
| Episodes with >70% coverage | 4 / 10 |
| Map size | 60×60 grid · 10 m/cell |
| Trained on | Boston, NYC, Chicago, Paris (real OSM) |



## Architecture

```
LiDAR observation (360°)
        │
        ▼
CityExplorerEnv          — Gymnasium env, 8-directional movement, 60×60 grid
        │
        ▼
OccupancyGrid (SLAM)     — Bayesian log-odds updates, Bresenham ray-casting
        │
        ▼
FastMambaMemorySLAMPolicy
  ├─ MambaSpatialEncoder — local patch + global map → latent vector
  ├─ Mamba SSM           — d=128, 2 layers, O(L) selective state space
  └─ FastNeuralMemoryBank— 500-slot episodic memory, vectorized index_add_
        │
        ▼
FastMambaPPOTrainer      — PPO with 32-step sequence chunks
```

### Key design choices

- **Mamba SSM over attention** — O(L) sequence modeling instead of O(L²). The selective state-space scan lets the policy carry long-horizon context without the memory cost of a transformer.
- **`forward_sequence(B, T, obs)`** — gradients flow through the full 32-step Mamba scan during PPO updates. Earlier implementations fed each observation as a length-1 sequence, eliminating temporal credit assignment.
- **Vectorized memory bank** — `FastNeuralMemoryBank.read()` uses fancy indexing and `index_add_` for O(1) batch reads regardless of batch size.
- **Bayesian SLAM** — log-odds occupancy grid with Bresenham LiDAR ray-casting. Sensor updates are probabilistic, so the map degrades gracefully under noisy returns.
- **Multi-city training** — 50% procedural maps + 50% real OSM city maps (Boston, NYC, Chicago, Paris). Prevents overfitting to a single layout and is what gives the 91% map accuracy on held-out Kendall Square.



## Project Structure

```
envs/city_env.py            — CityExplorerEnv (Gymnasium), procedural + OSM map support
slam/occupancy_grid.py      — Bayesian occupancy grid, Bresenham LiDAR, frontier detection
agent/mamba_memory_fast.py  — FastMambaMemorySLAMPolicy, FastNeuralMemoryBank
agent/mamba_trainer_fast.py — FastMambaPPOTrainer (PPO + Mamba + sequence chunking)
train_mamba_fast.py         — Main training script
real_city.py                — OSM → occupancy grid pipeline
evaluate_model.py           — Batch evaluation across episodes
visualize_agent.py          — Real-time SLAM visualization + GIF export
deploy.py                   — Run agent on a real map with GPS target
hf_space/                   — Gradio demo (deployed to Hugging Face)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch, and `osmnx` for real city maps.

---

## Training

```bash
# Train on procedural maps
python train_mamba_fast.py --policy fast_hybrid

# Train on real OSM city maps
python real_city.py --city "Boston, MA" --size 400  # generates data/boston.json
python train_mamba_fast.py --city-maps data/boston.json data/nyc.json --mix-procedural
```

---

## Evaluation & Visualization

```bash
# Evaluate checkpoint over N episodes
python evaluate_model.py --checkpoint checkpoints/mamba_fast_hybrid_slam.pt --episodes 10

# Watch the agent navigate and save a GIF
python deploy.py --map data/real_grid.json --lat 42.3601 --lon -71.0942

# Interactive visualization
python visualize_agent.py --city-map data/real_grid.json --save-gif
```

---

## Stack

PyTorch · Mamba SSM · PPO · Bayesian Occupancy Grid · Bresenham LiDAR · OpenStreetMap (`osmnx`) · Gymnasium · Gradio
