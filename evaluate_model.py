"""
evaluate_model.py — Test Trained Models on New Maps
----------------------------------------------------
Evaluate your trained Mamba agent on different city configurations.

Usage:
    # Test on random map
    python evaluate_model.py --checkpoint checkpoints/mamba_fast_hybrid_slam.pt
    
    # Test on specific city
    python evaluate_model.py --checkpoint checkpoints/mamba_fast_hybrid_slam.pt \
        --city-map data/boston_grid.json
    
    # Run multiple episodes
    python evaluate_model.py --checkpoint checkpoints/mamba_fast_hybrid_slam.pt \
        --episodes 10 --save-results
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

from envs.city_env import CityExplorerEnv
from agent.mamba_trainer_fast import FastMambaPPOTrainer


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate trained Mamba SLAM agent')
    
    # Model
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to trained model checkpoint')
    p.add_argument('--policy', type=str, default='fast_hybrid',
                   choices=['fast_hybrid', 'ultra_fast'],
                   help='Policy type (must match checkpoint)')
    
    # Environment
    p.add_argument('--city-map', type=str, default=None,
                   help='Path to city map JSON (if None, generates random)')
    p.add_argument('--width', type=int, default=40,
                   help='Map width (if no city-map provided)')
    p.add_argument('--height', type=int, default=40,
                   help='Map height (if no city-map provided)')
    p.add_argument('--towers', type=int, default=0,
                   help='Number of towers (if no city-map provided)')
    p.add_argument('--fov', type=float, default=7.0,
                   help='Field of view range')
    p.add_argument('--max-steps', type=int, default=600,
                   help='Maximum steps per episode')
    
    # Task
    p.add_argument('--task', type=str, default='region',
                   choices=['region', 'full'],
                   help='Task type')
    p.add_argument('--target-x', type=int, default=None)
    p.add_argument('--target-y', type=int, default=None)
    p.add_argument('--target-radius', type=int, default=None)
    p.add_argument('--target-coverage', type=float, default=0.95)
    p.add_argument('--target-score', type=float, default=0.90)
    
    # Evaluation
    p.add_argument('--episodes', type=int, default=5,
                   help='Number of evaluation episodes')
    p.add_argument('--render', action='store_true',
                   help='Render episodes (slower)')
    p.add_argument('--save-results', action='store_true',
                   help='Save results to JSON')
    p.add_argument('--save-trajectory', action='store_true',
                   help='Save agent trajectory')
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed (None for random)')
    
    # Model architecture (must match training)
    p.add_argument('--d-model', type=int, default=None)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--memory-size', type=int, default=500)
    
    return p.parse_args()


def create_env(args, seed=None):
    """Create evaluation environment"""
    city_map = None
    if args.city_map:
        with open(args.city_map) as f:
            city_map = json.load(f)
        print(f"Loaded city map: {args.city_map}")
        print(f"  Size: {city_map['W']}×{city_map['H']}")
        print(f"  Obstacles: {len(city_map.get('obstacles', []))}")
    
    target_center = None
    if args.target_x is not None and args.target_y is not None:
        target_center = (args.target_x, args.target_y)
    
    env = CityExplorerEnv(
        width=city_map['W'] if city_map else args.width,
        height=city_map['H'] if city_map else args.height,
        n_towers=args.towers,
        max_steps=args.max_steps,
        fov=args.fov,
        render_mode='human' if args.render else None,
        seed=seed if seed is not None else np.random.randint(0, 1000000),
        city_map=city_map,
        task=args.task,
        target_center=target_center,
        target_radius=args.target_radius,
        target_coverage=args.target_coverage,
        target_score=args.target_score,
    )
    
    return env


def load_model(args, obs_dim, n_actions):
    """Load trained model"""
    if args.d_model is None:
        args.d_model = 128 if args.policy == 'fast_hybrid' else 64
    
    trainer = FastMambaPPOTrainer(
        obs_dim=obs_dim,
        n_actions=n_actions,
        policy_type=args.policy,
        d_model=args.d_model,
        n_layers=args.n_layers,
        memory_size=args.memory_size,
    )
    
    trainer.load(args.checkpoint)
    trainer.net.eval()
    
    return trainer


def run_episode(env, trainer, policy_type, save_trajectory=False):
    """Run single evaluation episode"""
    trainer.reset_episode()
    obs, _ = env.reset()
    
    trajectory = []
    episode_data = {
        'steps': 0,
        'reward': 0.0,
        'loop_closures': [] if policy_type == 'fast_hybrid' else None,
    }
    
    done = False
    while not done:
        # Act
        if policy_type == 'ultra_fast':
            action, _, _ = trainer.act(obs, deterministic=True)
            loop_closure = None
        else:
            action, _, _, loop_closure = trainer.act(obs, deterministic=True)
            if loop_closure > 0.85:  # Loop closure threshold
                episode_data['loop_closures'].append(episode_data['steps'])
        
        # Save trajectory
        if save_trajectory:
            trajectory.append({
                'step': episode_data['steps'],
                'position': env.pos.tolist(),
                'action': int(action),
            })
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_data['reward'] += reward
        episode_data['steps'] += 1
    
    # Final metrics
    episode_data.update({
        'region_coverage': info['region_coverage'],
        'region_score': info['region_score'],
        'map_accuracy': info['map_accuracy'],
        'success': info.get('success', False),
        'trajectory': trajectory if save_trajectory else None,
    })
    
    return episode_data


def print_episode_summary(ep_num, data):
    """Print episode summary"""
    print(f"\nEpisode {ep_num}:")
    print(f"  Steps:           {data['steps']}")
    print(f"  Reward:          {data['reward']:.1f}")
    print(f"  Region Coverage: {data['region_coverage']*100:.1f}%")
    print(f"  Region Score:    {data['region_score']*100:.1f}%")
    print(f"  Map Accuracy:    {data['map_accuracy']*100:.1f}%")
    print(f"  Success:         {'✓' if data['success'] else '✗'}")
    
    if data['loop_closures'] is not None:
        print(f"  Loop Closures:   {len(data['loop_closures'])} detected")


def print_final_statistics(results):
    """Print aggregated statistics"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    metrics = {
        'steps': [],
        'reward': [],
        'region_coverage': [],
        'region_score': [],
        'map_accuracy': [],
        'success': [],
    }
    
    for ep_data in results:
        for key in metrics:
            if key == 'success':
                metrics[key].append(1 if ep_data[key] else 0)
            else:
                metrics[key].append(ep_data[key])
    
    print(f"\nAggregated Statistics ({len(results)} episodes):")
    print("-" * 70)
    
    for key, values in metrics.items():
        if key == 'success':
            success_rate = np.mean(values) * 100
            print(f"  Success Rate:        {success_rate:.1f}%")
        elif key in ['region_coverage', 'region_score', 'map_accuracy']:
            mean_val = np.mean(values) * 100
            std_val = np.std(values) * 100
            print(f"  {key.replace('_', ' ').title():20s} {mean_val:.1f}% ± {std_val:.1f}%")
        else:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {key.replace('_', ' ').title():20s} {mean_val:.1f} ± {std_val:.1f}")
    
    # Loop closures
    if results[0]['loop_closures'] is not None:
        total_closures = sum(len(ep['loop_closures']) for ep in results)
        avg_closures = total_closures / len(results)
        print(f"  Loop Closures:       {avg_closures:.1f} per episode")
    
    print("="*70)
    
    # Goal achievement
    avg_coverage = np.mean([ep['region_coverage'] for ep in results]) * 100
    avg_accuracy = np.mean([ep['map_accuracy'] for ep in results]) * 100
    
    print("\n🎯 Goal Achievement:")
    if avg_accuracy >= 90:
        print(f"  ✅ Map Accuracy: {avg_accuracy:.1f}% (Goal: 90%+)")
    else:
        print(f"  ⚠️  Map Accuracy: {avg_accuracy:.1f}% (Goal: 90%+, need more training)")
    
    if avg_coverage >= 85:
        print(f"  ✅ Coverage: {avg_coverage:.1f}% (Good exploration)")
    else:
        print(f"  ⚠️  Coverage: {avg_coverage:.1f}% (Could improve)")


def save_results_to_file(args, results):
    """Save results to JSON"""
    output_dir = Path('evaluation_results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"eval_{args.policy}_{timestamp}.json"
    filepath = output_dir / filename
    
    output = {
        'checkpoint': args.checkpoint,
        'policy': args.policy,
        'city_map': args.city_map,
        'episodes': len(results),
        'timestamp': timestamp,
        'results': results,
        'statistics': {
            'mean_accuracy': float(np.mean([ep['map_accuracy'] for ep in results])),
            'mean_coverage': float(np.mean([ep['region_coverage'] for ep in results])),
            'mean_score': float(np.mean([ep['region_score'] for ep in results])),
            'success_rate': float(np.mean([ep['success'] for ep in results])),
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {filepath}")


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("MAMBA SLAM AGENT EVALUATION")
    print("="*70)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Policy:     {args.policy.upper()}")
    print(f"Episodes:   {args.episodes}")
    
    # Create environment
    env = create_env(args, seed=args.seed)
    print(f"\nEnvironment: {env.W}×{env.H}")
    print(f"Task:        {args.task}")
    print(f"Max Steps:   {args.max_steps}")
    
    # Load model
    print("\nLoading model...")
    trainer = load_model(args, env.observation_space.shape[0], env.action_space.n)
    print("✓ Model loaded")
    
    # Run evaluation
    print(f"\nRunning {args.episodes} episodes...")
    print("-" * 70)
    
    results = []
    for ep in range(1, args.episodes + 1):
        ep_data = run_episode(env, trainer, args.policy, args.save_trajectory)
        results.append(ep_data)
        print_episode_summary(ep, ep_data)
    
    # Print statistics
    print_final_statistics(results)
    
    # Save results
    if args.save_results:
        save_results_to_file(args, results)
    
    env.close()
    print("\n✓ Evaluation complete!\n")


if __name__ == '__main__':
    main()
