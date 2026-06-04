"""
export_weights.py — Export trained model to JSON for browser demo
-----------------------------------------------------------------
Run this after training:
    python export_weights.py --checkpoint checkpoints/slam_ppo_v2.pt

Outputs:
    weights.json  — all network weights, loadable in browser
    model_info.json — architecture metadata
"""

import torch
import json
import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.city_env import CityExplorerEnv
from agent.ppo_agent import PPOTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/slam_ppo_v2.pt')
    p.add_argument('--width',  type=int, default=30)
    p.add_argument('--height', type=int, default=30)
    p.add_argument('--towers', type=int, default=5)
    p.add_argument('--out',    default='weights.json')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"No checkpoint found at {args.checkpoint}")
        print("Run training first: python train.py")
        return

    # build env just to get dims
    env = CityExplorerEnv(
        width=args.width, height=args.height, n_towers=args.towers)
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n
    patch_r   = env.patch_r
    G         = env.G
    env.close()

    # load trainer
    trainer = PPOTrainer(
        obs_dim=obs_dim, n_actions=n_actions,
        patch_r=patch_r, global_size=G)
    trainer.load(args.checkpoint)
    trainer.net.eval()

    # extract every layer
    weights = {}
    for name, param in trainer.net.state_dict().items():
        arr = param.cpu().numpy()
        weights[name] = {
            'data':  arr.flatten().tolist(),
            'shape': list(arr.shape),
        }
        print(f"  {name:40s} {list(arr.shape)}")

    # metadata the browser needs to reconstruct the net
    info = {
        'obs_dim':     obs_dim,
        'n_actions':   n_actions,
        'patch_r':     patch_r,
        'global_size': G,
        'hidden_size': 256,
        'local_out':   64,
        'global_out':  64,
        'meta_out':    32,
        'train_steps': trainer.train_steps,
        'layers':      list(weights.keys()),
        'env': {
            'width':  args.width,
            'height': args.height,
            'towers': args.towers,
        }
    }

    # save
    output = {'info': info, 'weights': weights}
    with open(args.out, 'w') as f:
        json.dump(output, f, default=lambda x: int(x) if hasattr(x, 'item') else x)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nExported {len(weights)} layers → {args.out} ({size_mb:.1f} MB)")
    print(f"Train steps: {trainer.train_steps:,}")
    print(f"Obs dim:     {obs_dim}")
    print(f"Actions:     {n_actions}")
    print(f"\nNow open demo_trained.html in your browser.")


if __name__ == '__main__':
    main()