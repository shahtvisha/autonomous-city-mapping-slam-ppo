"""
visualize_agent.py — Watch the trained agent navigate a real city map
----------------------------------------------------------------------
Renders the SLAM belief map being built in real-time alongside ground
truth, with the agent's trajectory and target region overlaid.

Usage:
    python visualize_agent.py
    python visualize_agent.py --city-map data/real_grid.json --target-x 20 --target-y 20
    python visualize_agent.py --speed 0.05   # slow down (seconds per step)
    python visualize_agent.py --save-gif     # save trajectory as GIF
    python visualize_agent.py --save-gif --search 40   # find best run out of 40, then record
"""

import argparse
import json
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

from envs.city_env import CityExplorerEnv
from agent.mamba_trainer_fast import FastMambaPPOTrainer


# ── Stuck detector + BFS recovery ────────────────────────────────────────────

# Action index → (dx, dy), must match CityExplorerEnv.ACTIONS
_DIRS = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
_DIR_TO_ACTION = {d: i for i, d in enumerate(_DIRS)}


class StuckDetector:
    """
    Watches the last `window` positions. If the agent stays within `radius`
    cells of its centroid for that whole window it's considered stuck.

    When stuck, compute a BFS path from current position to the nearest
    frontier cell (free cell adjacent to unknown) and return forced actions
    along that path until the agent is exploring again.
    """

    def __init__(self, window: int = 25, radius: float = 2.5):
        self.window = window
        self.radius = radius
        self._positions: deque = deque(maxlen=window)
        self._recovery_path: list = []   # list of (dx, dy) to follow

    # ── public interface ──────────────────────────────────────────

    def update(self, pos):
        self._positions.append((int(pos[0]), int(pos[1])))

    def is_stuck(self) -> bool:
        if len(self._positions) < self.window:
            return False
        xs = [p[0] for p in self._positions]
        ys = [p[1] for p in self._positions]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        return all(abs(p[0] - cx) + abs(p[1] - cy) <= self.radius
                   for p in self._positions)

    def in_recovery(self) -> bool:
        return len(self._recovery_path) > 0

    def get_action(self, pos, binary_map, obstacles, W, H) -> int:
        """
        Return the next forced action toward the nearest frontier.
        Recomputes path if the current one is exhausted.
        """
        if not self._recovery_path:
            self._recovery_path = self._bfs_to_frontier(pos, binary_map, obstacles, W, H)
            if not self._recovery_path:
                # No reachable frontier — just nudge in a random free direction
                import random
                free_dirs = [(dx, dy) for dx, dy in _DIRS
                             if _free(pos[0]+dx, pos[1]+dy, obstacles, W, H)]
                if free_dirs:
                    dx, dy = random.choice(free_dirs)
                    return _DIR_TO_ACTION[(dx, dy)]
                return 0  # fallback

        dx, dy = self._recovery_path.pop(0)
        return _DIR_TO_ACTION.get((dx, dy), 0)

    def reset(self):
        self._positions.clear()
        self._recovery_path.clear()

    # ── internals ────────────────────────────────────────────────

    def _bfs_to_frontier(self, pos, binary_map, obstacles, W, H) -> list:
        """BFS from pos → nearest frontier. Returns list of (dx,dy) steps."""
        start = (int(pos[0]), int(pos[1]))
        visited = {start}
        # queue entries: (cell, path_of_directions)
        queue = deque([(start, [])])

        while queue:
            (cx, cy), path = queue.popleft()
            # Is this cell a frontier?
            if binary_map[cy, cx] == 0:          # free
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < W and 0 <= ny < H and binary_map[ny, nx] == -1:
                        return path              # reached a frontier — return path

            for dx, dy in _DIRS:
                nx, ny = cx+dx, cy+dy
                if (_free(nx, ny, obstacles, W, H) and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(dx, dy)]))

        return []   # no reachable frontier


def _free(x, y, obstacles, W, H) -> bool:
    return 0 <= x < W and 0 <= y < H and not obstacles[y, x]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/mamba_fast_hybrid_slam.pt')
    p.add_argument('--city-map', default='data/real_grid.json')
    p.add_argument('--target-x', type=int, default=None)
    p.add_argument('--target-y', type=int, default=None)
    p.add_argument('--target-radius', type=int, default=8)
    p.add_argument('--max-steps', type=int, default=1200)
    p.add_argument('--speed', type=float, default=0.08,
                   help='Seconds to pause between steps (0=max speed)')
    p.add_argument('--save-gif', action='store_true')
    p.add_argument('--out', type=str, default='evaluation_results/agent_run.gif',
                   help='Output path for GIF')
    p.add_argument('--near-start', action='store_true',
                   help='Force agent to start near target (demo mode)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--search', type=int, default=0, metavar='N',
                   help='Run N headless episodes first, animate the best one (coverage×0.6 + accuracy×0.4)')
    return p.parse_args()


# ── Image helpers ─────────────────────────────────────────────────────────────

def slam_to_image(slam_map, W, H):
    """Convert SLAM binary_map to RGB: unknown=grey, free=white, occupied=dark."""
    img = np.ones((H, W, 3), dtype=np.float32) * 0.5
    img[slam_map == 0] = [0.95, 0.95, 0.95]
    img[slam_map == 1] = [0.25, 0.25, 0.25]
    return img


def truth_to_image(obstacles, W, H):
    img = np.ones((H, W, 3), dtype=np.float32)
    img[obstacles] = [0.25, 0.25, 0.25]
    return img


def overlay_region(img, region_mask, alpha=0.25):
    out = img.copy()
    out[region_mask] = out[region_mask] * (1 - alpha) + np.array([0.2, 0.5, 1.0]) * alpha
    return out


def overlay_trajectory(img, trajectory, W, H, color=(1.0, 0.3, 0.0)):
    out = img.copy()
    n = len(trajectory)
    for i, (tx, ty) in enumerate(trajectory):
        if 0 <= tx < W and 0 <= ty < H:
            fade = 0.3 + 0.7 * (i / max(n - 1, 1))
            out[ty, tx] = np.array(color) * fade + out[ty, tx] * (1 - fade)
    return out


# ── Headless episode runner (for --search) ────────────────────────────────────

def run_headless(env, trainer, city_map, target, max_steps, seed):
    """Run one episode without rendering. Returns (coverage, map_accuracy)."""
    obs, _ = env.reset(seed=seed, options={'city_map': city_map, 'target': target})
    trainer.reset_episode()
    stuck = StuckDetector(window=25, radius=2.5)
    W, H = city_map['W'], city_map['H']
    info = {}
    for _ in range(max_steps):
        stuck.update(env.pos)
        if stuck.is_stuck() or stuck.in_recovery():
            action = stuck.get_action(
                env.pos, env.slam.binary_map, env.city.obstacles, W, H)
            if not stuck.in_recovery():
                stuck.reset()
        else:
            action, _, _, _ = trainer.act(obs, deterministic=False)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return info.get('region_coverage', 0.0), info.get('map_accuracy', 0.0)


def find_best_seed(env, trainer, city_map, target, max_steps, n_search, base_seed=0):
    """Search n_search seeds, return the seed with best weighted score."""
    print(f"\nSearching {n_search} episodes for best run...")
    best_seed, best_score = base_seed, -1.0
    for i in range(n_search):
        seed = base_seed + i
        cov, acc = run_headless(env, trainer, city_map, target, max_steps, seed)
        score = cov * 0.6 + acc * 0.4
        status = ""
        if score > best_score:
            best_score = score
            best_seed = seed
            status = f"  ← best so far"
        print(f"  Ep {i+1:3d}  seed={seed}  cov={cov*100:.1f}%  acc={acc*100:.1f}%  score={score:.3f}{status}")
    print(f"\nBest seed: {best_seed}  (score={best_score:.3f})\n")
    return best_seed


# ── Main visualization ────────────────────────────────────────────────────────

def run_visualization(args):
    with open(args.city_map) as f:
        city_map = json.load(f)

    W, H = city_map['W'], city_map['H']
    city_name = city_map.get('name', Path(args.city_map).stem)

    # Resolve target
    if args.target_x is not None and args.target_y is not None:
        target = [args.target_x, args.target_y]
    elif 'target' in city_map:
        target = city_map['target']
    else:
        obstacles = np.array(
            [item for row in city_map['obstacles'] for item in row], dtype=bool
        ).reshape(H, W)
        cy, cx = H // 2, W // 2
        target = [cx, cy]
        for r in range(1, max(W, H)):
            found = False
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < W and 0 <= ny < H and not obstacles[ny, nx]:
                        target = [nx, ny]
                        found = True
                        break
                if found:
                    break
            if found:
                break

    print(f"Map:    {city_name}  ({W}×{H}, {city_map.get('cell_size', 10)}m/cell)")
    print(f"Target: {target}  radius={args.target_radius}")

    env = CityExplorerEnv(
        width=W, height=H,
        max_steps=args.max_steps,
        target_radius=args.target_radius,
        target_coverage=0.85,
        target_score=0.80,
        seed=args.seed,
    )

    trainer = FastMambaPPOTrainer(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        policy_type='fast_hybrid',
        d_model=128,
        n_layers=2,
        memory_size=500,
    )
    trainer.load(args.checkpoint)
    trainer.net.eval()

    # Search for best seed if requested
    seed = args.seed
    if args.search > 0:
        seed = find_best_seed(env, trainer, city_map, target,
                              args.max_steps, args.search, base_seed=args.seed)

    # Reset to the chosen seed for the actual visualization run
    trainer.reset_episode()

    if args.near_start:
        import types
        def _near_start(self):
            tx, ty = self._target
            r = self._target_radius()
            for _ in range(200):
                angle = env.np_random.uniform(0, 2 * 3.14159)
                dist  = env.np_random.uniform(0, r * 1.2)
                import numpy as _np
                sx = int(_np.clip(tx + dist * _np.cos(angle), 0, self.W - 1))
                sy = int(_np.clip(ty + dist * _np.sin(angle), 0, self.H - 1))
                if not self.city.obstacles[sy, sx]:
                    return _np.array([sx, sy], dtype=_np.int32)
            return self._random_free_pos()
        env._choose_start = types.MethodType(_near_start, env)

    obs, _ = env.reset(seed=seed, options={'city_map': city_map, 'target': target})
    region_mask = env._region_mask

    # ── Figure layout ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    # Top: main title + subtitle
    fig.suptitle(
        f"Active SLAM  ·  Mamba SSM + PPO",
        color='#e8e8e8', fontsize=16, fontweight='bold', y=0.97,
    )
    fig.text(
        0.5, 0.935,
        f"{city_name}  ·  real OpenStreetMap data  ·  no GPS, no prior map",
        ha='center', color='#888899', fontsize=11,
    )

    axes[0].set_title('SLAM Belief Map', color='#ccddff', fontsize=14, pad=6)
    axes[1].set_title('Ground Truth + Trajectory', color='#ccddff', fontsize=14, pad=6)

    # Bottom: stats bar
    stats_text = fig.text(
        0.5, 0.035,
        'Step    0  |  Coverage  0.0%  |  Score  0.0%  |  Map Accuracy  0.0%',
        ha='center', color='#dddddd', fontsize=13, fontweight='bold',
        fontfamily='monospace',
    )

    # Reserve space: top 14%, bottom 10%
    plt.tight_layout(rect=[0, 0.10, 1, 0.91])

    # Initial renders
    slam_img  = slam_to_image(env.slam.binary_map, W, H)
    truth_img = truth_to_image(env.city.obstacles, W, H)

    slam_disp  = axes[0].imshow(overlay_region(slam_img, region_mask),
                                origin='upper', vmin=0, vmax=1, interpolation='nearest')
    truth_disp = axes[1].imshow(overlay_region(truth_img, region_mask),
                                origin='upper', vmin=0, vmax=1, interpolation='nearest')

    agent_dot_slam,  = axes[0].plot([], [], 'o', color='#ff4444', ms=7, zorder=5)
    agent_dot_truth, = axes[1].plot([], [], 'o', color='#ff4444', ms=7, zorder=5)

    axes[0].plot(target[0], target[1], '*', color='#ffdd00', ms=12, zorder=6)
    axes[1].plot(target[0], target[1], '*', color='#ffdd00', ms=12, zorder=6)

    legend = [
        mpatches.Patch(color=[0.95, 0.95, 0.95], label='Free'),
        mpatches.Patch(color=[0.25, 0.25, 0.25], label='Obstacle'),
        mpatches.Patch(color=[0.5,  0.5,  0.5 ], label='Unknown'),
        mpatches.Patch(color=[0.2,  0.5,  1.0 ], label='Target region'),
        mpatches.Patch(color='#ff4444',           label='Agent'),
    ]
    axes[0].legend(handles=legend, loc='upper right', fontsize=8,
                   facecolor='#1a1a2e', labelcolor='white', framealpha=0.85,
                   edgecolor='#333355')

    trajectory = [tuple(env.pos)]
    step_data = {'step': 0, 'reward': 0.0, 'coverage': 0.0,
                 'score': 0.0, 'accuracy': 0.0, 'done': False,
                 'recovering': False}
    stuck = StuckDetector(window=25, radius=2.5)

    def update(_frame):
        if step_data['done']:
            return

        stuck.update(env.pos)

        if stuck.is_stuck() or stuck.in_recovery():
            # Override policy: BFS toward nearest unexplored frontier
            action = stuck.get_action(
                env.pos, env.slam.binary_map, env.city.obstacles, W, H)
            step_data['recovering'] = True
        else:
            action, _, _, _ = trainer.act(obs, deterministic=False)
            step_data['recovering'] = False

        new_obs, reward, terminated, truncated, info = env.step(action)
        obs[:] = new_obs
        trajectory.append(tuple(env.pos))

        step_data['step']     += 1
        step_data['reward']   += reward
        step_data['coverage']  = info['region_coverage']
        step_data['score']     = info['region_score']
        step_data['accuracy']  = info.get('map_accuracy', 0.0)
        step_data['done']      = terminated or truncated

        # If moving again after recovery, clear the stuck window
        if step_data['recovering'] and not stuck.in_recovery():
            stuck.reset()
            step_data['recovering'] = False

        si = overlay_region(slam_to_image(env.slam.binary_map, W, H), region_mask)
        si = overlay_trajectory(si, trajectory[-80:], W, H)
        slam_disp.set_data(si)
        agent_dot_slam.set_data([env.pos[0]], [env.pos[1]])

        ti = overlay_region(truth_img.copy(), region_mask)
        ti = overlay_trajectory(ti, trajectory[-80:], W, H)
        truth_disp.set_data(ti)
        agent_dot_truth.set_data([env.pos[0]], [env.pos[1]])

        mode = "  [RECOVERY]" if step_data['recovering'] else ""
        stats_text.set_text(
            f"Step {step_data['step']:4d} / {args.max_steps}"
            f"   |   Coverage {step_data['coverage']*100:5.1f}%"
            f"   |   Score {step_data['score']*100:5.1f}%"
            f"   |   Map Accuracy {step_data['accuracy']*100:5.1f}%"
            + mode
        )
        stats_text.set_color('#ffcc44' if step_data['recovering'] else '#dddddd')

        if step_data['done']:
            result = 'SUCCESS' if terminated else 'TIMEOUT'
            color  = '#44ff88' if terminated else '#ffaa44'
            fig.suptitle(
                f"Active SLAM  ·  Mamba SSM + PPO  —  {result}",
                color=color, fontsize=16, fontweight='bold', y=0.97,
            )

        if args.speed > 0 and not args.save_gif:
            plt.pause(args.speed)

        return slam_disp, truth_disp, agent_dot_slam, agent_dot_truth, stats_text

    if args.save_gif:
        from PIL import Image
        import io

        print("Recording GIF...")
        pil_frames = []
        FREEZE_FRAMES = 48

        for frame_i in range(args.max_steps):
            update(frame_i)
            fig.canvas.draw()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=90, facecolor=fig.get_facecolor())
            buf.seek(0)
            pil_frames.append(Image.open(buf).copy())
            buf.close()

            if (frame_i + 1) % 100 == 0:
                print(f"  Frame {frame_i+1}  |  coverage={step_data['coverage']*100:.1f}%")

            if step_data['done']:
                for _ in range(FREEZE_FRAMES):
                    pil_frames.append(pil_frames[-1].copy())
                break

        out_path = Path(args.out)
        out_path.parent.mkdir(exist_ok=True)
        pil_frames[0].save(
            str(out_path), save_all=True, append_images=pil_frames[1:],
            loop=0, duration=83,
        )
        print(f"Saved: {out_path}  ({len(pil_frames)} frames)")
    else:
        ani = FuncAnimation(fig, update, frames=args.max_steps,
                            interval=max(1, int(args.speed * 1000)),
                            blit=False, repeat=False)
        plt.show()

    print(f"\nFinal — Steps: {step_data['step']} | "
          f"Coverage: {step_data['coverage']*100:.1f}% | "
          f"Score: {step_data['score']*100:.1f}% | "
          f"Map Accuracy: {step_data['accuracy']*100:.1f}%")


if __name__ == '__main__':
    args = parse_args()
    run_visualization(args)
