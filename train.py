"""
train.py — Active SLAM + PPO Training
--------------------------------------
Run:
    python train.py                    # train headless
    python train.py --render           # train with live pygame viz
    python train.py --eval             # eval a saved checkpoint
    python train.py --steps 500000     # set total timesteps

Real-world deployment notes:
  - Replace CityExplorerEnv with a ROS2 wrapper that reads
    /velodyne_points → slam.observe_region() and publishes
    /cmd_vel from the agent's action.
  - The trained policy (SLAMPolicyNet) runs on-device (Jetson/RPi).
  - Map is streamed via ros2 topic /map_fused.
"""

import argparse
import time
import os
import numpy as np
from collections import deque

from envs.city_env import CityExplorerEnv
from agent.ppo_agent import PPOTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--steps',      type=int,   default=300_000)
    p.add_argument('--rollout',    type=int,   default=2048)
    p.add_argument('--n-envs',     type=int,   default=1)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--gamma',      type=float, default=0.99)
    p.add_argument('--gae',        type=float, default=0.95)
    p.add_argument('--clip',       type=float, default=0.2)
    p.add_argument('--ent-coef',   type=float, default=0.01)
    p.add_argument('--epochs',     type=int,   default=10)
    p.add_argument('--batch',      type=int,   default=64)
    p.add_argument('--width',      type=int,   default=40)
    p.add_argument('--height',     type=int,   default=40)
    p.add_argument('--towers',     type=int,   default=0)
    p.add_argument('--fov',        type=float, default=7.0)
    p.add_argument('--max-steps',  type=int,   default=600)
    p.add_argument('--task',       default='region',
                   choices=['region', 'legacy_towers'])
    p.add_argument('--target-x',   type=int,   default=None)
    p.add_argument('--target-y',   type=int,   default=None)
    p.add_argument('--target-radius', type=int, default=None)
    p.add_argument('--target-coverage', type=float, default=0.95)
    p.add_argument('--target-score', type=float, default=0.90)
    p.add_argument('--render',     action='store_true')
    p.add_argument('--eval',       action='store_true')
    p.add_argument('--checkpoint', type=str,   default='checkpoints/slam_region.pt')
    p.add_argument('--save-freq',  type=int,   default=25_000)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--city-map', type=str, default=None, help='path to real city JSON e.g. data/boston.json')
    p.add_argument('--load-from', type=str, default=None,
               help='load weights from this checkpoint before training')
    return p.parse_args()


def make_env(args, render=False):
    import json, numpy as np

    city_map = None
    if hasattr(args, 'city_map') and args.city_map:
        with open(args.city_map) as f:
            city_map = json.load(f)

    target_center = None
    if args.target_x is not None and args.target_y is not None:
        target_center = (args.target_x, args.target_y)

    env = CityExplorerEnv(
        width     = city_map['W']  if city_map else args.width,
        height    = city_map['H'] if city_map else args.height,
        n_towers  = args.towers,
        max_steps = args.max_steps,
        fov       = args.fov,
        render_mode = 'human' if render else None,
        seed      = args.seed,
        city_map  = city_map,   # pass real map in
        task      = args.task,
        target_center = target_center,
        target_radius = args.target_radius,
        target_coverage = args.target_coverage,
        target_score = args.target_score,
    )
    return env


def train(args):
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    env = make_env(args, render=args.render)
    obs, _ = env.reset()

    trainer = PPOTrainer(
        obs_dim     = env.observation_space.shape[0],
        n_actions   = env.action_space.n,
        patch_r     = env.patch_r,
        global_size = env.G,
        lr          = args.lr,
        gamma       = args.gamma,
        gae_lambda  = args.gae,
        clip_eps    = args.clip,
        ent_coef    = args.ent_coef,
        n_epochs    = args.epochs,
        batch_size  = args.batch,
    )
    
    # Load checkpoint if exists
    if os.path.exists(args.checkpoint):
        trainer.load(args.checkpoint)
    
    if hasattr(args, 'load_from') and args.load_from and os.path.exists(args.load_from):
        trainer.load(args.load_from)
        print(f"Transfer learning from: {args.load_from}")

    # Metrics
    ep_rewards     = deque(maxlen=20)
    ep_coverages   = deque(maxlen=20)
    ep_scores      = deque(maxlen=20)
    ep_map_accs    = deque(maxlen=20)
    ep_lengths     = deque(maxlen=20)
    ep_reward      = 0.0
    ep_length      = 0
    episode        = 0
    total_steps    = 0
    last_save      = 0
    t0             = time.time()

    log_file = open('logs/training.csv', 'w')
    log_file.write(
        'step,episode,mean_reward,mean_region_coverage,'
        'mean_region_score,mean_map_accuracy,entropy,'
        'policy_loss,value_loss,lr\n')

    print(f"\n{'='*60}")
    print(f"  Active SLAM + PPO Training")
    print(f"  Device: {trainer.device}")
    print(f"  Obs dim: {env.observation_space.shape[0]}")
    print(f"  City: {args.width}×{args.height}")
    print(f"  Task: {args.task} region mapping")
    print(f"  Total steps: {args.steps:,}")
    print(f"{'='*60}\n")

    while total_steps < args.steps:
        action, log_prob, value = trainer.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        trainer.collect(obs, action, log_prob, value, reward, done)

        obs = next_obs if not done else env.reset()[0]
        ep_reward += reward
        ep_length += 1
        total_steps += 1

        if done:
            ep_rewards.append(ep_reward)
            ep_coverages.append(info['region_coverage'])
            ep_scores.append(info['region_score'])
            ep_map_accs.append(info['map_accuracy'])
            ep_lengths.append(ep_length)
            ep_reward = 0.0
            ep_length = 0
            episode += 1

        # PPO update
        if len(trainer._obs) >= args.rollout:
            losses = trainer.train(obs, done)

            if losses and episode > 0:
                mr  = np.mean(ep_rewards)
                mc  = np.mean(ep_coverages) * 100
                ms  = np.mean(ep_scores) * 100
                ma  = np.mean(ep_map_accs) * 100
                fps = total_steps / (time.time() - t0)

                print(
                    f"Step {total_steps:>7,} | Ep {episode:>4} | "
                    f"Reward {mr:>7.1f} | "
                    f"Region {mc:>5.1f}% | "
                    f"Score {ms:>5.1f}% | "
                    f"MapAcc {ma:>5.1f}% | "
                    f"Ent {losses.get('entropy',0):.3f} | "
                    f"FPS {fps:.0f}"
                )

                log_file.write(
                    f"{total_steps},{episode},"
                    f"{mr:.2f},{mc:.2f},{ms:.2f},{ma:.2f},"
                    f"{losses.get('entropy',0):.4f},"
                    f"{losses.get('policy_loss',0):.4f},"
                    f"{losses.get('value_loss',0):.4f},"
                    f"{losses.get('lr',0):.6f}\n"
                )
                log_file.flush()

        # Checkpoint
        if total_steps - last_save >= args.save_freq:
            trainer.save(args.checkpoint)
            last_save = total_steps

    trainer.save(args.checkpoint)
    log_file.close()
    env.close()
    print(f"\nTraining complete. Model saved to {args.checkpoint}")


def evaluate(args):
    """Run the trained policy with rendering."""
    env = make_env(args, render=True)
    obs, _ = env.reset()

    trainer = PPOTrainer(
        obs_dim     = env.observation_space.shape[0],
        n_actions   = env.action_space.n,
        patch_r     = env.patch_r,
        global_size = env.G,
    )
    trainer.load(args.checkpoint)
    trainer.net.eval()

    print(f"\nEvaluating {args.checkpoint} ...")
    ep = 0
    while True:
        action, _, _ = trainer.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            ep += 1
            print(f"Ep {ep} | Region {info['region_coverage']*100:.1f}% | "
                  f"Score {info['region_score']*100:.1f}% | "
                  f"MapAcc {info['map_accuracy']*100:.1f}%")
            obs, _ = env.reset()


if __name__ == '__main__':
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
