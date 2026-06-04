"""
train_mamba_fast.py — Fast Mamba Training (5-10x Speedup)
-----------------------------------------------------------
Optimized training with reduced model sizes and efficient operations.

Run:
    python train_mamba_fast.py --policy fast_hybrid    # 10-30 FPS
    python train_mamba_fast.py --policy ultra_fast     # 30-100 FPS
    
Expected speedup: 5-10x faster than original
Expected accuracy: 90-94% (fast_hybrid), 85-90% (ultra_fast)
"""

import argparse
import time
import os
import numpy as np
from collections import deque

from envs.city_env import CityExplorerEnv
from agent.mamba_trainer_fast import FastMambaPPOTrainer


def parse_args():
    p = argparse.ArgumentParser()
    
    # Model selection
    p.add_argument('--policy', default='fast_hybrid', 
                   choices=['fast_hybrid', 'ultra_fast'],
                   help='fast_hybrid: 10-30 FPS, ultra_fast: 30-100 FPS')
    
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
    
    # Model architecture
    p.add_argument('--d-model', type=int, default=None,
                   help='Model dimension (default: 128 for fast_hybrid, 64 for ultra_fast)')
    p.add_argument('--n-layers', type=int, default=2,
                   help='Number of Mamba layers')
    p.add_argument('--memory-size', type=int, default=500,
                   help='Memory bank size (fast_hybrid only)')
    
    # Environment
    p.add_argument('--width', type=int, default=40)
    p.add_argument('--height', type=int, default=40)
    p.add_argument('--towers', type=int, default=0)
    p.add_argument('--fov', type=float, default=7.0)
    p.add_argument('--max-steps', type=int, default=600)
    p.add_argument('--task', default='region')
    
    # Region task
    p.add_argument('--target-x', type=int, default=None)
    p.add_argument('--target-y', type=int, default=None)
    p.add_argument('--target-radius', type=int, default=None)
    p.add_argument('--target-coverage', type=float, default=0.95)
    p.add_argument('--target-score', type=float, default=0.90)
    
    # Misc
    p.add_argument('--render', action='store_true')
    p.add_argument('--eval', action='store_true')
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
    """Train fast Mamba policy"""
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Set default d_model based on policy type
    if args.d_model is None:
        args.d_model = 128 if args.policy == 'fast_hybrid' else 64
    
    # Create environment
    env = make_env(args, render=args.render)
    obs, _ = env.reset()
    
    # Create trainer
    trainer = FastMambaPPOTrainer(
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
    ep_loop_closures = deque(maxlen=20) if args.policy == 'fast_hybrid' else None
    
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
    if args.policy == 'fast_hybrid':
        header += ',mean_loop_closure,memory_utilization'
    log_file.write(header + '\n')
    
    print(f"\n{'='*70}")
    print(f"  Fast Mamba Training (Optimized)")
    print(f"  Policy: {args.policy.upper()}")
    print(f"  Device: {trainer.device}")
    print(f"  Model: d_model={args.d_model}, layers={args.n_layers}")
    if args.policy == 'fast_hybrid':
        print(f"  Memory: {args.memory_size} landmarks")
    print(f"  Expected FPS: {'10-30' if args.policy == 'fast_hybrid' else '30-100'}")
    print(f"  Target: {args.target_coverage*100:.0f}% coverage, "
          f"{args.target_score*100:.0f}% score")
    print(f"{'='*70}\n")
    
    while total_steps < args.steps:
        # Reset episode
        if ep_length == 0:
            trainer.reset_episode()
        
        # Act
        if args.policy == 'ultra_fast':
            action, log_prob, value = trainer.act(obs)
            loop_closure = None
        else:
            action, log_prob, value, loop_closure = trainer.act(obs)
        
        # Step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Collect
        trainer.collect(obs, action, log_prob, value, reward, done, loop_closure)
        
        # Update
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
            
            if args.policy == 'fast_hybrid' and loop_closure is not None:
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
                
                log_line = (
                    f"Step {total_steps:>7,} | Ep {episode:>4} | "
                    f"Reward {mr:>7.1f} | "
                    f"Region {mc:>5.1f}% | "
                    f"Score {ms:>5.1f}% | "
                    f"MapAcc {ma:>5.1f}% | "
                    f"Ent {losses.get('entropy',0):.3f}"
                )
                
                if args.policy == 'fast_hybrid':
                    mem_stats = trainer.get_memory_stats()
                    mlc = np.mean(ep_loop_closures) if ep_loop_closures else 0
                    log_line += (f" | Loop {mlc:.3f} | "
                               f"Mem {mem_stats['memory_utilization']*100:.0f}%")
                
                log_line += f" | FPS {fps:.0f}"
                print(log_line)
                
                # CSV
                csv_line = (
                    f"{total_steps},{episode},"
                    f"{mr:.2f},{mc:.2f},{ms:.2f},{ma:.2f},"
                    f"{losses.get('entropy',0):.4f},"
                    f"{losses.get('policy_loss',0):.4f},"
                    f"{losses.get('value_loss',0):.4f},"
                    f"{losses.get('lr',0):.6f}"
                )
                
                if args.policy == 'fast_hybrid':
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
    print(f"  Model: {args.checkpoint}")
    print(f"  Log: logs/training_mamba_{args.policy}.csv")
    
    if episode > 0:
        print(f"\n  Final Performance:")
        print(f"    Reward: {np.mean(ep_rewards):.1f}")
        print(f"    Coverage: {np.mean(ep_coverages)*100:.1f}%")
        print(f"    Score: {np.mean(ep_scores)*100:.1f}%")
        print(f"    Accuracy: {np.mean(ep_map_accs)*100:.1f}%")
        print(f"    FPS: {total_steps / (time.time() - t0):.0f}")


def evaluate(args):
    """Evaluate trained policy"""
    if args.d_model is None:
        args.d_model = 128 if args.policy == 'fast_hybrid' else 64
    
    env = make_env(args, render=True)
    obs, _ = env.reset()
    
    trainer = FastMambaPPOTrainer(
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
    
    print(f"\nEvaluating: {args.checkpoint}")
    print(f"Policy: {args.policy.upper()}\n")
    
    ep = 0
    while True:
        trainer.reset_episode()
        obs, _ = env.reset()
        done = False
        
        while not done:
            if args.policy == 'ultra_fast':
                action, _, _ = trainer.act(obs, deterministic=True)
            else:
                action, _, _, _ = trainer.act(obs, deterministic=True)
            
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        ep += 1
        print(f"Ep {ep} | "
              f"Region {info['region_coverage']*100:.1f}% | "
              f"Score {info['region_score']*100:.1f}% | "
              f"MapAcc {info['map_accuracy']*100:.1f}%")


if __name__ == '__main__':
    args = parse_args()
    
    if args.eval:
        evaluate(args)
    else:
        train(args)
