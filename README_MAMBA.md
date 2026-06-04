# Mamba-based Active SLAM with PPO

## 🚀 Overview

This implementation uses **Mamba** (State Space Models) for long-term spatial memory in Active SLAM, replacing traditional Transformers with a more efficient O(n) complexity architecture.

### Two Architectures Implemented

1. **Pure Mamba**: Sequential spatial reasoning with selective state space models
2. **Hybrid Mamba + Memory Bank**: Combines Mamba with explicit landmark storage for loop closure detection

## 🎯 Key Advantages Over Baseline

| Feature | Baseline (MLP) | Mamba Pure | Mamba + Memory |
|---------|---------------|------------|----------------|
| Complexity | O(1) | O(n) | O(n) |
| Long-term Memory | ❌ | ✅ | ✅✅ |
| Loop Closure | ❌ | ⚠️ | ✅ |
| Sequence Length | N/A | Unlimited | Unlimited |
| Explicit Landmarks | ❌ | ❌ | ✅ |
| Map Accuracy Goal | ~82% | ~88% | **~95%** |

## 📊 Architecture Comparison

### Pure Mamba
```
Observation → Encoder → Mamba Layers (4x) → [Policy | Value]
                         ↓
                    Selective SSM
                    (maintains spatial context)
```

**Pros:**
- Simpler architecture
- Fewer parameters (~2M)
- Faster inference
- Good for medium-sized maps

**Cons:**
- No explicit loop closure
- Implicit memory only

### Hybrid Mamba + Memory Bank
```
Observation → Encoder → Memory Retrieval → Mamba Layers → Fusion → [Policy | Value | Loop Closure]
                         ↓                                    ↑
                    Memory Bank (1000 landmarks)              │
                         └────────────────────────────────────┘
```

**Pros:**
- Explicit landmark storage
- Loop closure detection
- Better long-term consistency
- **Recommended for 90%+ accuracy**

**Cons:**
- More parameters (~2.5M)
- Slightly slower inference
- Requires memory management

## 🔧 Installation

### Additional Dependencies

```bash
pip install torch>=2.0.0
pip install einops  # For efficient tensor operations
```

All other dependencies are in `requirements.txt`.

## 🏃 Quick Start

### 1. Train Pure Mamba

```bash
python train_mamba.py --policy pure --steps 300000
```

### 2. Train Hybrid Mamba + Memory (Recommended)

```bash
python train_mamba.py --policy hybrid --steps 300000
```

### 3. Evaluate Trained Model

```bash
python train_mamba.py --eval --policy hybrid --render
```

### 4. Compare Both Architectures

```bash
python train_mamba.py --compare
```

## 📈 Training Configuration

### Recommended Settings for City Mapping

```bash
python train_mamba.py \
    --policy hybrid \
    --steps 500000 \
    --d-model 256 \
    --n-layers 4 \
    --memory-size 1000 \
    --target-coverage 0.95 \
    --target-score 0.90 \
    --lr 3e-4 \
    --batch 64
```

### For Large Maps (60×60+)

```bash
python train_mamba.py \
    --policy hybrid \
    --width 60 \
    --height 60 \
    --d-model 512 \
    --n-layers 6 \
    --memory-size 2000 \
    --max-steps 1000
```

## 🧠 How Mamba Works

### Selective State Space Model (SSM)

Mamba uses a **selective mechanism** that allows it to:

1. **Remember important landmarks** (high Δ values)
2. **Forget revisited areas** (low Δ values)
3. **Focus on frontiers** (unexplored regions)

The core SSM equation:
```
h_t = A * h_{t-1} + B * x_t    (state update)
y_t = C * h_t + D * x_t        (output)
```

Where Δ (delta) controls the selection:
- High Δ → Remember this observation
- Low Δ → Forget/ignore this observation

### Memory Bank (Hybrid Only)

The memory bank stores:
- **Keys**: Spatial descriptors (128-dim)
- **Values**: Full landmark information (256-dim)
- **Metadata**: Age, usage count

**Retrieval**: Cosine similarity-based, top-k most similar landmarks

**Loop Closure**: Detects when similarity > 0.85 with old landmarks

## 📊 Expected Performance

### Pure Mamba

| Metric | 40×40 Map | 60×60 Map |
|--------|-----------|-----------|
| Coverage | 85-88% | 80-85% |
| Accuracy | 88-90% | 85-88% |
| Training Time | 200k steps | 300k steps |

### Hybrid Mamba + Memory

| Metric | 40×40 Map | 60×60 Map |
|--------|-----------|-----------|
| Coverage | 92-95% | 88-92% |
| Accuracy | **93-96%** | **90-94%** |
| Training Time | 250k steps | 400k steps |
| Loop Closures | 5-10 per episode | 10-20 per episode |

## 🔍 Monitoring Training

### Key Metrics to Watch

1. **Region Coverage**: Should reach >90% by 200k steps
2. **Region Score**: Should reach >85% by 250k steps
3. **Map Accuracy**: Should reach >90% by 300k steps
4. **Loop Closure Prob** (hybrid): Should increase over time
5. **Memory Utilization** (hybrid): Should reach 80-100%

### Training Logs

Logs are saved to:
- `logs/training_mamba_pure.csv`
- `logs/training_mamba_hybrid.csv`

Visualize with:
```bash
python analysis/analyze.py --checkpoint checkpoints/mamba_hybrid_slam.pt
```

## 🎯 Achieving 90%+ Accuracy

### Strategy

1. **Use Hybrid Architecture**: Essential for loop closure
2. **Increase Memory Size**: 1000-2000 landmarks
3. **Train Longer**: 400-500k steps
4. **Larger Model**: d_model=512, n_layers=6
5. **Curriculum Learning**: Start with small maps, gradually increase

### Example Training Sequence

```bash
# Stage 1: Small map (20×20)
python train_mamba.py --policy hybrid --width 20 --height 20 --steps 100000

# Stage 2: Medium map (40×40)
python train_mamba.py --policy hybrid --width 40 --height 40 --steps 200000 \
    --checkpoint checkpoints/mamba_hybrid_slam.pt

# Stage 3: Large map (60×60)
python train_mamba.py --policy hybrid --width 60 --height 60 --steps 300000 \
    --checkpoint checkpoints/mamba_hybrid_slam.pt
```

## 🔬 Technical Details

### Mamba Block Structure

```python
class MambaBlock:
    - Input projection (expand by 2x)
    - Depthwise convolution (local context)
    - SSM parameters (Δ, B, C)
    - Selective scan (core computation)
    - Output projection
```

### Memory Bank Operations

**Write** (every 5 steps):
```python
key = obs_encoded[:, :128]  # Spatial descriptor
memory_bank.write(key, obs_encoded)
```

**Read** (every step):
```python
retrieved, similarities = memory_bank.read(obs_encoded, top_k=10)
memory_context = weighted_sum(retrieved, similarities)
```

**Loop Closure Detection**:
```python
candidates = memory_bank.get_loop_closure_candidates(obs, threshold=0.85)
if candidates:
    # High similarity with old landmark → loop closure!
```

## 📁 File Structure

```
agent/
├── mamba_memory.py       # Mamba architectures
├── mamba_trainer.py      # PPO trainer for Mamba
└── ppo_agent.py          # Baseline (for comparison)

train_mamba.py            # Main training script
checkpoints/
├── mamba_pure_slam.pt    # Pure Mamba checkpoint
└── mamba_hybrid_slam.pt  # Hybrid checkpoint
```

## 🐛 Troubleshooting

### Out of Memory

**Solution**: Reduce batch size or model dimension
```bash
python train_mamba.py --batch 32 --d-model 128
```

### Slow Training

**Solution**: Use pure Mamba or reduce layers
```bash
python train_mamba.py --policy pure --n-layers 2
```

### Low Accuracy

**Solution**: Use hybrid + increase memory size
```bash
python train_mamba.py --policy hybrid --memory-size 2000 --steps 500000
```

### Memory Bank Not Filling

**Solution**: Reduce write frequency
```python
# In mamba_memory.py, line 580
self.write_frequency = 3  # Write more often
```

## 📚 References

1. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**  
   Gu & Dao, 2023  
   https://arxiv.org/abs/2312.00752

2. **Structured State Space Models (S4)**  
   Gu et al., 2021  
   https://arxiv.org/abs/2111.00396

3. **Active Neural SLAM**  
   Chaplot et al., 2020  
   https://arxiv.org/abs/2004.05155

## 🤝 Contributing

To add new features:

1. Modify `agent/mamba_memory.py` for architecture changes
2. Update `agent/mamba_trainer.py` for training logic
3. Test with `python train_mamba.py --compare`

## 📝 License

Same as parent project.

## 🎉 Acknowledgments

- Mamba implementation inspired by the original paper
- Memory bank design based on Neural Turing Machines
- Active SLAM framework from the baseline implementation

---

**For questions or issues, please open a GitHub issue.**

**Goal: 90%+ map accuracy for comprehensive city road mapping** ✅
