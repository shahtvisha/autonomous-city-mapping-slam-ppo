#!/bin/bash
# Round 3: navigation reward shaping
# Key changes from round 2 (env, not hyperparams):
#   - Curriculum start: 50% near-target, 50% random (teaches navigation TO region)
#   - In-region +1/step bonus (dense reward once inside, no local minima vs obstacles)
#   - Target-biased frontier following (0.6 weight toward target, 0.4 agent dist)
# Hyperparams unchanged from round 2 (still focused small region):
nohup caffeinate -dims python -u train_mamba_fast.py \
  --policy fast_hybrid \
  --steps 4000000 \
  --city-maps data/boston.json data/nyc.json data/chicago.json \
  --mix-procedural \
  --seq-len 32 \
  --target-radius 8 \
  --target-coverage 0.85 \
  --target-score 0.80 \
  --max-steps 1200 \
  --ent-coef 0.001 \
  --vf-coef 0.1 \
  --checkpoint checkpoints/mamba_fast_hybrid_slam.pt \
  > logs/overnight.log 2>&1 &
echo "Training PID: $!"
