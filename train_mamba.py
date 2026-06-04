"""
train_mamba.py — Mamba-based Active SLAM Training
--------------------------------------------------
Train and compare two architectures:
1. Pure Mamba: Sequential spatial reasoning
2. Mamba + Memory Bank: Hybrid with explicit landmark storage

Run:
    python train_mamba.py --policy pure       # Pure Mamba
    python train_mamba.py --policy hybrid     # Mamba + Memory (recommended)
    python train_mamba.py --eval              # Evaluate trained model
    python train_mamba.py --compare           # Compare both architectures

Key improvements over baseline:
- O(n) complexity vs O(n²) for Transformers
- Explicit loop closure detection (hybrid)
- Better long-term spatial memory
- Designed for 90%+ accuracy goal
"""

import argparse
import time
import os
import numpy as np
from collections import deque

from envs.city_env import CityExplorerEnv
from agent.mamba_trainer import MambaPPOTrainer


def parse_args():
    p = argparse.ArgumentParser()
    
    # Model selection
    p.add_argument('--policy', default='hybrid', choices=['pure', 'hybrid'],
                   help='Policy type: pure Mamba or hybrid Mamba+Memory')
    
    # Training
    p.add_argument('--steps', type=int, default=300_000)
    p.add_argument('--rollout', type=int, default=2048)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gae', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--ent-coef', type=float, default=0.01)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=64)
    
    # Mamba architecture
    p.add_argument('--d-model', type=int, default=256,
                   help='Mamba model dimension')
    p.add_argument('--n-layers', type=int, default=4,
                   help='Number of Mamba layers')
    p.add_argument('--memory-size', type=int, default=1000,
                   help='Memory bank size (hybrid only)')
    
    # Environment
    p.add_argument('--width', type=int, default=40)
    p.add_argument('--height', type=int, default=40)
    p.add_argument('--towers', type=int, default=0)
    p.add_argument('--fov', type=float, default=7.0)
    p.add_argument('--max-steps', type=int, default=600)
    p.add_argument('--task', default='region', choices=['region', 'legacy_towers'])
    
    # Region task parameters
    p.add_argument('--target-x', type=int, default=None)
    p.add_argument('--target-y', type=int, default=None)
    p.add_argument('--target-radius', type=int, default=None)
    p.add_argument('--target-coverage', type=float, default=0.95)
    p.add_argument('--target-score', type=float, default=0.90)
    
    # Misc
    p.add_argument('--render', action='store_true')
    p.add_argument('--eval', action='store_true')
    p.add_argument('--compare', action='store_true',
                   help='Train both policies and compare')
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--save-freq', type=int, default=25_000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--city-map', type=str, default=None)
    
    return p.parse_args()


def make_env(args, render=False):
    """Create environment"""
    import json
    
    city_map = None
    if hasattr(args, 'city_map') and args.city_map:
        with open(args.city_map) as f:
            city_map = json.load(f)
    
    target_center = None
    if args.target_x is not None and args.target_y is not None:
        target_center = (args.target_x, args.target_y)
    
    env = CityExplorerEnv(
        width=city_map['W'] if city_map else args.width,
        height=city_map['H'] if city_map else args.height,
        n_towers=args.towers,
        max_steps=args.max_steps,
        fov=args.fov,
        render_mode='human' if render else None,
        seed=args.seed,
        city_map=city_map,
        task=args.task,
        target_center=target_center,
        target_radius=args.target_radius,
        target_coverage=args.target_coverage,
        target_score=args.target_score,
    )
    return env


def train(args):
    """Train Mamba-based policy"""
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Create environment
    env = make_env(args, render=args.render)
    obs, _ = env.reset()
    
    # Create trainer
    trainer = MambaPPOTrainer(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        policy_type=args.policy,
        d_model=args.d_model,
        n_layers=args.n_layers,
        memory_size=args.memory_size,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae,
        clip_eps=args.clip,
        ent_coef=args.ent_coef,
        n_epochs=args.epochs,
        batch_size=args.batch,
    )
    
    # Checkpoint path
    if args.checkpoint is None:
        args.checkpoint = f'checkpoints/mamba_{args.policy}_slam.pt'
    
    # Load if exists
    if os.path.exists(args.checkpoint):
        trainer.load(args.checkpoint)
    
    # Metrics
    ep_rewards = deque(maxlen=20)
    ep_coverages = deque(maxlen=20)
    ep_scores = deque(maxlen=20)
    ep_map_accs = deque(maxlen=20)
    ep_lengths = deque(maxlen=20)
    ep_loop_closures = deque(maxlen=20) if args.policy == 'hybrid' else None
    
    ep_reward = 0.0
    ep_length = 0
    episode = 0
    total_steps = 0
    last_save = 0
    t0 = time.time()
    
    # Logging
    log_file = open(f'logs/training_mamba_{args.policy}.csv', 'w')
    header = ('step,episode,mean_reward,mean_region_coverage,'
              'mean_region_score,mean_map_accuracy,entropy,'
              'policy_loss,value_loss,lr')
    if args.policy == 'hybrid':
        header += ',mean_loop_closure,memory_utilization'
    log_file.write(header + '\n')
    
    print(f"\n{'='*70}")
    print(f"  Mamba-based Active SLAM Training")
    print(f"  Policy: {args.policy.upper()}")
    print(f"  Device: {trainer.device}")
    print(f"  Obs dim: {env.observation_space.shape[0]}")
    print(f"  Model dim: {args.d_model}, Layers: {args.n_layers}")
    if args.policy == 'hybrid':
        print(f"  Memory Bank: {args.memory_size} landmarks")
    print(f"  City: {args.width}×{args.height}")
    print(f"  Task: {args.task} region mapping")
    print(f"  Target: {args.target_coverage*100:.0f}% coverage, "
          f"{args.target_score*100:.0f}% score")
    print(f"  Total steps: {args.steps:,}")
    print(f"{'='*70}\n")
    
    while total_steps < args.steps:
        # Reset episode state
        if ep_length == 0:
            trainer.reset_episode()
        
        # Act
        if args.policy == 'pure':
            action, log_prob, value = trainer.act(obs)
            loop_closure = None
        else:
            action, log_prob, value, loop_closure = trainer.act(obs)
        
        # Step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Collect
        trainer.collect(obs, action, log_prob, value, reward, done, loop_closure)
        
        # Update state
        obs = next_obs if not done else env.reset()[0]
        ep_reward += reward
        ep_length += 1
        total_steps += 1
        
        # Episode end
        if done:
            ep_rewards.append(ep_reward)
            ep_coverages.append(info['region_coverage'])
            ep_scores.append(info['region_score'])
            ep_map_accs.append(info['map_accuracy'])
            ep_lengths.append(ep_length)
            
            if args.policy == 'hybrid' and loop_closure is not None:
                ep_loop_closures.append(loop_closure)
            
            ep_reward = 0.0
            ep_length = 0
            episode += 1
        
        # PPO update
        if len(trainer._obs) >= args.rollout:
            losses = trainer.train(obs, done)
            
            if losses and episode > 0:
                mr = np.mean(ep_rewards)
                mc = np.mean(ep_coverages) * 100
                ms = np.mean(ep_scores) * 100
                ma = np.mean(ep_map_accs) * 100
                fps = total_steps / (time.time() - t0)
                
                # Base metrics
                log_line = (
                    f"Step {total_steps:>7,} | Ep {episode:>4} | "
                    f"Reward {mr:>7.1f} | "
                    f"Region {mc:>5.1f}% | "
                    f"Score {ms:>5.1f}% | "
                    f"MapAcc {ma:>5.1f}% | "
                    f"Ent {losses.get('entropy',0):.3f}"
                )
                
                # Add hybrid-specific metrics
                if args.policy == 'hybrid':
                    mem_stats = trainer.get_memory_stats()
                    mlc = np.mean(ep_loop_closures) if ep_loop_closures else 0
                    log_line += (f" | Loop {mlc:.3f} | "
                               f"Mem {mem_stats['memory_utilization']*100:.0f}%")
                
                log_line += f" | FPS {fps:.0f}"
                print(log_line)
                
                # Write to CSV
                csv_line = (
                    f"{total_steps},{episode},"
                    f"{mr:.2f},{mc:.2f},{ms:.2f},{ma:.2f},"
                    f"{losses.get('entropy',0):.4f},"
                    f"{losses.get('policy_loss',0):.4f},"
                    f"{losses.get('value_loss',0):.4f},"
                    f"{losses.get('lr',0):.6f}"
                )
                
                if args.policy == 'hybrid':
                    mlc = np.mean(ep_loop_closures) if ep_loop_closures else 0
                    mem_util = mem_stats['memory_utilization']
                    csv_line += f",{mlc:.4f},{mem_util:.4f}"
                
                log_file.write(csv_line + '\n')
                log_file.flush()
        
        # Checkpoint
        if total_steps - last_save >= args.save_freq:
            trainer.save(args.checkpoint)
            last_save = total_steps
    
    # Final save
    trainer.save(args.checkpoint)
    log_file.close()
    env.close()
    
    print(f"\n✓ Training complete!")
    print(f"  Model saved to: {args.checkpoint}")
    print(f"  Log saved to: logs/training_mamba_{args.policy}.csv")
    
    # Final statistics
    if episode > 0:
        print(f"\n  Final Performance (last 20 episodes):")
        print(f"    Mean Reward: {np.mean(ep_rewards):.1f}")
        print(f"    Region Coverage: {np.mean(ep_coverages)*100:.1f}%")
        print(f"    Region Score: {np.mean(ep_scores)*100:.1f}%")
        print(f"    Map Accuracy: {np.mean(ep_map_accs)*100:.1f}%")
        
        if args.policy == 'hybrid' and ep_loop_closures:
            print(f"    Loop Closures: {np.mean(ep_loop_closures):.3f}")
            mem_stats = trainer.get_memory_stats()
            print(f"    Memory Utilization: {mem_stats['memory_utilization']*100:.0f}%")


def evaluate(args):
    """Evaluate trained policy"""
    env = make_env(args, render=True)
    obs, _ = env.reset()
    
    trainer = MambaPPOTrainer(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        policy_type=args.policy,
        d_model=args.d_model,
        n_layers=args.n_layers,
        memory_size=args.memory_size,
    )
    
    if args.checkpoint is None:
        args.checkpoint = f'checkpoints/mamba_{args.policy}_slam.pt'
    
    trainer.load(args.checkpoint)
    trainer.net.eval()
    
    print(f"\n{'='*70}")
    print(f"  Evaluating: {args.checkpoint}")
    print(f"  Policy: {args.policy.upper()}")
    print(f"{'='*70}\n")
    
    ep = 0
    while True:
        trainer.reset_episode()
        obs, _ = env.reset()
        done = False
        
        while not done:
            if args.policy == 'pure':
                action, _, _ = trainer.act(obs, deterministic=True)
            else:
                action, _, _, loop_closure = trainer.act(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        ep += 1
        print(f"Ep {ep} | "
              f"Region {info['region_coverage']*100:.1f}% | "
              f"Score {info['region_score']*100:.1f}% | "
              f"MapAcc {info['map_accuracy']*100:.1f}%")
        
        if args.policy == 'hybrid':
            mem_stats = trainer.get_memory_stats()
            print(f"      Memory: {mem_stats['memory_utilization']*100:.0f}% utilized, "
                  f"age {mem_stats['mean_memory_age']:.0f}")


def compare_policies(args):
    """Train both policies and compare performance"""
    print(f"\n{'='*70}")
    print(f"  Comparing Pure Mamba vs Hybrid Mamba+Memory")
    print(f"{'='*70}\n")
    
    results = {}
    
    for policy_type in ['pure', 'hybrid']:
        print(f"\n{'─'*70}")
        print(f"  Training {policy_type.upper()} policy...")
        print(f"{'─'*70}\n")
        
        args.policy = policy_type
        args.checkpoint = f'checkpoints/mamba_{policy_type}_slam_compare.pt'
        args.steps = 100_000  # Shorter for comparison
        
        train(args)
        
        # Evaluate
        env = make_env(args, render=False)
        trainer = MambaPPOTrainer(
            obs_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            policy_type=policy_type,
            d_model=args.d_model,
            n_layers=args.n_layers,
            memory_size=args.memory_size,
        )
        trainer.load(args.checkpoint)
        trainer.net.eval()
        
        # Run evaluation episodes
        coverages, scores, accuracies = [], [], []
        for seed in range(10):
            env = make_env(args, render=False)
            env.reset(seed=seed)
            trainer.reset_episode()
            obs, _ = env.reset()
            done = False
            
            while not done:
                if policy_type == 'pure':
                    action, _, _ = trainer.act(obs, deterministic=True)
                else:
                    action, _, _, _ = trainer.act(obs, deterministic=True)
                
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            coverages.append(info['region_coverage'])
            scores.append(info['region_score'])
            accuracies.append(info['map_accuracy'])
            env.close()
        
        results[policy_type] = {
            'coverage': np.mean(coverages),
            'score': np.mean(scores),
            'accuracy': np.mean(accuracies),
            'coverage_std': np.std(coverages),
            'score_std': np.std(scores),
            'accuracy_std': np.std(accuracies),
        }
    
    # Print comparison
    print(f"\n{'='*70}")
    print(f"  COMPARISON RESULTS (10 episodes each)")
    print(f"{'='*70}\n")
    
    print(f"{'Metric':<20} {'Pure Mamba':<25} {'Hybrid Mamba+Memory':<25}")
    print(f"{'-'*70}")
    
    for metric in ['coverage', 'score', 'accuracy']:
        pure_val = results['pure'][metric] * 100
        pure_std = results['pure'][f'{metric}_std'] * 100
        hybrid_val = results['hybrid'][metric] * 100
        hybrid_std = results['hybrid'][f'{metric}_std'] * 100
        
        print(f"{metric.capitalize():<20} "
              f"{pure_val:>6.1f}% ± {pure_std:>4.1f}%        "
              f"{hybrid_val:>6.1f}% ± {hybrid_std:>4.1f}%")
    
    print(f"\n{'='*70}")
    print(f"  RECOMMENDATION")
    print(f"{'='*70}")
    
    if results['hybrid']['accuracy'] > results['pure']['accuracy']:
        print(f"\n  ✓ Hybrid Mamba+Memory performs better!")
        print(f"    - {(results['hybrid']['accuracy'] - results['pure']['accuracy'])*100:.1f}% "
              f"higher accuracy")
        print(f"    - Explicit memory enables loop closure")
        print(f"    - Recommended for 90%+ accuracy goal")
    else:
        print(f"\n  ✓ Pure Mamba performs better!")
        print(f"    - Simpler architecture")
        print(f"    - Faster inference")
    
    print()


if __name__ == '__main__':
    args = parse_args()
    
    if args.compare:
        compare_policies(args)
    elif args.eval:
        evaluate(args)
    else:
        train(args)
