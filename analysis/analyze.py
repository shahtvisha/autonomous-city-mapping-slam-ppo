"""
analyze.py — Policy Analysis + Visualization
---------------------------------------------
Run after training to generate:
  1. Training curves (reward, coverage, towers)
  2. Action distribution heatmap
  3. Value function visualization
  4. Policy rollout comparison (random vs trained)
  5. Coverage efficiency over episodes

Usage:
    python analyze.py --checkpoint checkpoints/slam_ppo.pt
    python analyze.py --checkpoint checkpoints/slam_ppo.pt --episodes 20
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import torch

from envs.city_env import CityExplorerEnv
from agent.ppo_agent import PPOTrainer


# ── custom colormap ────────────────────────────────────────────────
SLAM_CMAP = LinearSegmentedColormap.from_list(
    'slam', ['#0d1117', '#16213e', '#1a2e4a', '#4ecca3'])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/slam_ppo.pt')
    p.add_argument('--log',        default='logs/training.csv')
    p.add_argument('--episodes',   type=int, default=10)
    p.add_argument('--width',      type=int, default=40)
    p.add_argument('--height',     type=int, default=40)
    p.add_argument('--towers',     type=int, default=5)
    p.add_argument('--out',        default='analysis/')
    return p.parse_args()


# ── 1. Training curves ─────────────────────────────────────────────

def plot_training_curves(log_path: str, out: str):
    if not os.path.exists(log_path):
        print(f"No log found at {log_path} — skipping curves")
        return

    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('PPO Training Analysis — Active SLAM',
                 color='white', fontsize=14, fontweight='bold', y=0.98)

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor('#131926')
        ax.set_title(title, color='#cccccc', fontsize=11, pad=8)
        ax.set_xlabel(xlabel, color='#888888', fontsize=9)
        ax.set_ylabel(ylabel, color='#888888', fontsize=9)
        ax.tick_params(colors='#666666', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2a3a')
        ax.grid(True, color='#1e2030', linewidth=0.5, alpha=0.7)

    # smooth helper
    def smooth(y, w=15):
        if len(y) < w:
            return y
        return pd.Series(y).rolling(w, min_periods=1).mean().values

    steps = df['step'].values

    # 1a. Episode reward
    ax = axes[0, 0]
    raw = df['mean_reward'].values
    ax.plot(steps, raw, color='#2a3a4a', linewidth=0.5, alpha=0.6)
    ax.plot(steps, smooth(raw), color='#f5a623', linewidth=2, label='smoothed')
    style_ax(ax, 'Mean episode reward', 'steps', 'reward')
    ax.legend(fontsize=8, labelcolor='white', facecolor='#131926')

    # 1b. Coverage
    ax = axes[0, 1]
    raw = df['mean_coverage'].values
    ax.plot(steps, raw, color='#2a3a4a', linewidth=0.5, alpha=0.6)
    ax.plot(steps, smooth(raw), color='#4ecca3', linewidth=2)
    ax.axhline(y=80, color='#4ecca3', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.text(steps[-1], 81, '80% target', color='#4ecca3', fontsize=8, ha='right')
    style_ax(ax, 'Map coverage (%)', 'steps', 'coverage %')

    # 1c. Towers found
    ax = axes[0, 2]
    raw = df['mean_towers'].values
    ax.plot(steps, raw, color='#2a3a4a', linewidth=0.5, alpha=0.6)
    ax.plot(steps, smooth(raw), color='#e94560', linewidth=2)
    style_ax(ax, 'Mean towers found / episode', 'steps', 'towers')

    # 1d. Policy entropy
    ax = axes[1, 0]
    ax.plot(steps, smooth(df['entropy'].values), color='#a78bfa', linewidth=2)
    style_ax(ax, 'Policy entropy (exploration)', 'steps', 'entropy')
    ax.fill_between(steps, smooth(df['entropy'].values), alpha=0.15, color='#a78bfa')

    # 1e. Losses
    ax = axes[1, 1]
    ax.plot(steps, smooth(df['policy_loss'].values), color='#f5a623',
            linewidth=1.5, label='policy')
    ax.plot(steps, smooth(df['value_loss'].values),  color='#4ecca3',
            linewidth=1.5, label='value')
    style_ax(ax, 'PPO losses', 'steps', 'loss')
    ax.legend(fontsize=8, labelcolor='white', facecolor='#131926')

    # 1f. Sample efficiency — coverage per 1k steps
    ax = axes[1, 2]
    eff = df['mean_coverage'].values / (df['step'].values / 1000 + 1e-8)
    ax.plot(steps, smooth(eff, 20), color='#fb923c', linewidth=2)
    style_ax(ax, 'Coverage efficiency (% per 1k steps)', 'steps', 'efficiency')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  Saved → {path}")


# ── 2. Policy rollout analysis ─────────────────────────────────────

def run_rollout(env, trainer, deterministic=True):
    """Run one episode, collect trajectory data."""
    obs, _ = env.reset()
    trajectory = []
    actions = []
    values  = []
    done = False

    while not done:
        action, log_prob, value = trainer.act(obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(env.pos.copy())
        actions.append(action)
        values.append(value)
        done = terminated or truncated
        obs = next_obs

    return trajectory, actions, values, info, env.slam.prob_map.copy()


def plot_rollout_comparison(trainer, args, out):
    """Compare random policy vs trained policy on same map."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('Policy Rollout Analysis',
                 color='white', fontsize=13, fontweight='bold')

    env = CityExplorerEnv(
        width=args.width, height=args.height,
        n_towers=args.towers, seed=42)

    # ── trained policy ──
    traj, acts, vals, info_t, pmap_t = run_rollout(env, trainer, deterministic=True)

    ax = axes[0]
    ax.set_facecolor('#0d1117')
    ax.imshow(pmap_t, cmap=SLAM_CMAP, origin='upper',
              extent=[0, args.width, args.height, 0])

    # obstacle overlay
    obs_arr = env.city.obstacles.astype(float)
    ax.imshow(np.where(obs_arr, 0.9, np.nan),
              cmap='gray', origin='upper', alpha=0.4,
              extent=[0, args.width, args.height, 0])

    # trajectory
    if traj:
        tx = [p[0]+0.5 for p in traj]
        ty = [p[1]+0.5 for p in traj]
        ax.plot(tx, ty, color='#f5a623', linewidth=0.8, alpha=0.7)
        ax.scatter(tx[0], ty[0], c='#4ecca3', s=60, zorder=5, label='start')
        ax.scatter(tx[-1], ty[-1], c='#f5a623', s=60, zorder=5, label='end')

    # towers
    for i, (tx2, ty2) in enumerate(env.city.towers):
        col = '#4ecca3' if i in env._found else '#e94560'
        ax.scatter(tx2+0.5, ty2+0.5, c=col, s=80, marker='^', zorder=6)

    ax.set_title(f'Trained policy\nCoverage {info_t["coverage"]*100:.1f}% · '
                 f'Towers {info_t["towers_found"]}/{args.towers}',
                 color='white', fontsize=10)
    ax.set_xlim(0, args.width); ax.set_ylim(args.height, 0)
    ax.tick_params(colors='#555'); ax.legend(fontsize=8, labelcolor='white',
                                              facecolor='#131926')

    # ── action distribution ──
    ax = axes[1]
    ax.set_facecolor('#131926')
    action_labels = ['→','←','↓','↑','↘','↙','↗','↖']
    counts = np.bincount(acts, minlength=8)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, 8))
    bars = ax.bar(action_labels, counts, color=colors)
    ax.set_title('Action distribution (trained)', color='white', fontsize=10)
    ax.tick_params(colors='#888', labelsize=9)
    for spine in ax.spines.values(): spine.set_edgecolor('#2a2a3a')
    ax.set_facecolor('#131926')
    ax.yaxis.label.set_color('#888')

    # ── value function over trajectory ──
    ax = axes[2]
    ax.set_facecolor('#131926')
    ax.plot(vals, color='#a78bfa', linewidth=1.5)
    ax.fill_between(range(len(vals)), vals, alpha=0.2, color='#a78bfa')
    # mark tower discoveries
    step = 0
    env2 = CityExplorerEnv(width=args.width, height=args.height,
                            n_towers=args.towers, seed=42)
    ax.set_title('Value function V(s) over episode', color='white', fontsize=10)
    ax.set_xlabel('step', color='#888', fontsize=9)
    ax.set_ylabel('V(s)', color='#888', fontsize=9)
    ax.tick_params(colors='#666', labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor('#2a2a3a')
    ax.grid(True, color='#1e2030', linewidth=0.5, alpha=0.7)
    ax.axhline(y=0, color='#444', linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(out, 'rollout_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  Saved → {path}")


# ── 3. Multi-episode stats ─────────────────────────────────────────

def plot_multi_episode(trainer, args, out):
    """Run N episodes, plot distribution of outcomes."""
    print(f"  Running {args.episodes} evaluation episodes...")
    coverages, towers_found, ep_lengths = [], [], []

    for seed in range(args.episodes):
        env = CityExplorerEnv(
            width=args.width, height=args.height,
            n_towers=args.towers, seed=seed)
        _, _, _, info, _ = run_rollout(env, trainer, deterministic=True)
        coverages.append(info['coverage'] * 100)
        towers_found.append(info['towers_found'])
        ep_lengths.append(info['step'])
        env.close()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(f'Policy evaluation — {args.episodes} unseen cities',
                 color='white', fontsize=12, fontweight='bold')

    def style(ax, title, xlabel):
        ax.set_facecolor('#131926')
        ax.set_title(title, color='#cccccc', fontsize=10)
        ax.set_xlabel(xlabel, color='#888', fontsize=9)
        ax.tick_params(colors='#666', labelsize=8)
        for s in ax.spines.values(): s.set_edgecolor('#2a2a3a')

    axes[0].hist(coverages, bins=10, color='#4ecca3', edgecolor='#0d1117', linewidth=0.5)
    axes[0].axvline(np.mean(coverages), color='#f5a623', linewidth=2,
                    label=f'mean={np.mean(coverages):.1f}%')
    axes[0].legend(fontsize=8, labelcolor='white', facecolor='#131926')
    style(axes[0], 'Coverage distribution', 'coverage %')

    tw = np.bincount(towers_found, minlength=args.towers+1)
    axes[1].bar(range(len(tw)), tw, color='#e94560', edgecolor='#0d1117')
    style(axes[1], 'Towers found distribution', '# towers')

    axes[2].hist(ep_lengths, bins=10, color='#a78bfa', edgecolor='#0d1117')
    axes[2].axvline(np.mean(ep_lengths), color='#f5a623', linewidth=2,
                    label=f'mean={np.mean(ep_lengths):.0f}')
    axes[2].legend(fontsize=8, labelcolor='white', facecolor='#131926')
    style(axes[2], 'Episode length distribution', 'steps')

    plt.tight_layout()
    path = os.path.join(out, 'evaluation_stats.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()

    print(f"\n  Evaluation summary ({args.episodes} episodes):")
    print(f"    Coverage:      {np.mean(coverages):.1f}% ± {np.std(coverages):.1f}%")
    print(f"    Towers found:  {np.mean(towers_found):.2f} / {args.towers}")
    print(f"    Episode length:{np.mean(ep_lengths):.0f} steps")
    print(f"  Saved → {path}")


# ── main ───────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("\n=== Active SLAM Policy Analysis ===\n")

    # load trainer
    dummy_env = CityExplorerEnv(
        width=args.width, height=args.height, n_towers=args.towers)
    trainer = PPOTrainer(
        obs_dim   = dummy_env.observation_space.shape[0],
        n_actions = dummy_env.action_space.n,
        patch_r   = dummy_env.patch_r,
        global_size = dummy_env.G,
    )
    dummy_env.close()

    if os.path.exists(args.checkpoint):
        trainer.load(args.checkpoint)
        trainer.net.eval()
    else:
        print(f"  No checkpoint at {args.checkpoint} — analysing untrained policy")

    print("1. Plotting training curves...")
    plot_training_curves(args.log, args.out)

    print("2. Plotting rollout analysis...")
    plot_rollout_comparison(trainer, args, args.out)

    print("3. Running multi-episode evaluation...")
    plot_multi_episode(trainer, args, args.out)

    print(f"\nAll outputs saved to {args.out}/")


if __name__ == '__main__':
    main()