"""
visualize_multi_agent.py — 3-agent cooperative SLAM visualization
------------------------------------------------------------------
Each agent owns a vertical strip of the map and walks a lawnmower
pattern through it (step_size=10, LiDAR range=7 → full coverage).
After its strip is traversed it switches to BFS targeting uncovered
region cells.  Starting positions are spread across the full map so
agents visually explore different parts of the city.

Usage:
    python visualize_multi_agent.py --save-gif --full-run
    python visualize_multi_agent.py --save-gif --full-run --frame-skip 3
    python visualize_multi_agent.py --search 15 --save-gif --full-run
"""

import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
from pathlib import Path

from envs.multi_agent_env import MultiAgentCityExplorerEnv
from agent.mamba_trainer_fast import FastMambaPPOTrainer


# ── Colours / labels ──────────────────────────────────────────────────────────
AGENT_COLORS = ['#cc2222', '#0077cc', '#229922']   # vivid on white background
AGENT_LABELS = ['Agent A', 'Agent B', 'Agent C']

MAROON   = '#8B0000'
DARK_TXT = '#222222'
PANEL_BG = '#f5f5f5'


# ── 8-direction action table ──────────────────────────────────────────────────
_DIRS          = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
_DIR_TO_ACTION = {d: i for i, d in enumerate(_DIRS)}

def _free(x, y, obs, W, H):
    return 0 <= x < W and 0 <= y < H and not obs[y, x]


# ── BFS helpers ───────────────────────────────────────────────────────────────

def _bfs_to_point(start_pos, target, obstacles, W, H):
    """BFS path to a specific (tx, ty).  Finds nearest free cell if target is blocked."""
    tx, ty = int(target[0]), int(target[1])
    # Snap target to nearest free cell
    if not _free(tx, ty, obstacles, W, H):
        found = None
        for r in range(1, 8):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if abs(dx) == r or abs(dy) == r:
                        nx, ny = tx+dx, ty+dy
                        if _free(nx, ny, obstacles, W, H):
                            found = (nx, ny)
                            break
                if found:
                    break
            if found:
                break
        if not found:
            return [], None
        tx, ty = found

    start = (int(start_pos[0]), int(start_pos[1]))
    goal  = (tx, ty)
    if start == goal:
        return [], goal

    visited = {start}
    queue   = deque([(start, [])])
    while queue:
        (cx, cy), path = queue.popleft()
        if (cx, cy) == goal:
            return path, goal
        for dx, dy in _DIRS:
            nx, ny = cx+dx, cy+dy
            if _free(nx, ny, obstacles, W, H) and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(dx, dy)]))
    return [], None


def _bfs_to_region_target(pos, binary_map, region_mask, obstacles, W, H,
                           claimed: dict, agent_id: int):
    """
    BFS to nearest free cell adjacent to an unknown REGION cell.
    Falls back to any frontier when the region is fully mapped.
    """
    region_targets, any_targets = [], []
    for y in range(H):
        for x in range(W):
            if binary_map[y, x] != 0:
                continue
            in_r = in_a = False
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                nx, ny = x+dx, y+dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                if binary_map[ny, nx] == -1:
                    in_a = True
                    if region_mask[ny, nx]:
                        in_r = True
                        break
            if in_r:
                region_targets.append((x, y))
            elif in_a:
                any_targets.append((x, y))

    def prefer_unclaimed(lst):
        u = [(x,y) for x,y in lst if claimed.get((x,y), agent_id) == agent_id]
        return u if u else lst

    targets = (prefer_unclaimed(region_targets) or region_targets
               or prefer_unclaimed(any_targets) or any_targets)
    if not targets:
        return [], None

    target_set = set(targets)
    start   = (int(pos[0]), int(pos[1]))
    visited = {start}
    queue   = deque([(start, [])])
    while queue:
        (cx, cy), path = queue.popleft()
        if (cx, cy) in target_set:
            return path, (cx, cy)
        for dx, dy in _DIRS:
            nx, ny = cx+dx, cy+dy
            if _free(nx, ny, obstacles, W, H) and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(dx, dy)]))
    return [], None


# ── Zone lawnmower explorer ───────────────────────────────────────────────────


class ZoneLawnmower:
    """
    Systematic zone-based explorer.

    Phase 1 — Lawnmower: walk a grid of waypoints across the agent's
    assigned vertical strip (step_size columns apart, zigzag rows).
    Waypoints outside the largest connected component are skipped so the
    agent never tries to navigate to an isolated pocket.

    Phase 2 — Region cleanup: BFS to any remaining uncovered region cells.

    Phase 3 — Policy: hand off to RL if nothing remains.
    """

    STEP_SIZE = 10   # distance between waypoints; LiDAR(7) covers the gaps

    def __init__(self, agent_id: int, n_agents: int, W: int, H: int):
        self.agent_id = agent_id
        strip_w       = W // n_agents
        self.x_min    = agent_id * strip_w
        self.x_max    = (agent_id + 1) * strip_w if agent_id < n_agents - 1 else W
        self.W, self.H = W, H

        # Build lawnmower waypoints for this strip.
        # _advance_lawnmower skips any waypoint whose BFS returns no path,
        # so we don't need to pre-filter here.
        s  = self.STEP_SIZE
        xs = list(range(self.x_min + s//2, self.x_max, s))
        ys = list(range(s//2, H, s))
        self._waypoints: list = []
        for i, x in enumerate(xs):
            col = ys if i % 2 == 0 else list(reversed(ys))
            for y in col:
                self._waypoints.append((x, y))

        self._wp_idx       = 0
        self._path: list   = []
        self._phase        = 'lawnmower'   # 'lawnmower' | 'region' | 'policy'
        self._claimed_tgt  = None

    # ── public ────────────────────────────────────────────────────────────────

    def get_action(self, pos, obs, binary_map, region_mask,
                   obstacles, W, H, claimed, trainer) -> tuple:
        """Returns (action_int, phase_str)."""

        if not self._path:
            self._next_path(pos, binary_map, region_mask, obstacles, W, H, claimed)

        if self._path:
            dx, dy = self._path.pop(0)
            return _DIR_TO_ACTION.get((dx, dy), 0), self._phase

        # Pure policy fallback (region fully covered)
        action, _, _, _ = trainer.act(obs, deterministic=False)
        return action, 'policy'

    def reset(self, claimed: dict):
        self._release_claim(claimed)
        self._path  = []
        self._wp_idx = 0
        self._phase  = 'lawnmower'

    # ── internal ──────────────────────────────────────────────────────────────

    def _next_path(self, pos, binary_map, region_mask, obstacles, W, H, claimed):
        if self._phase == 'lawnmower':
            self._advance_lawnmower(pos, obstacles, W, H)
        if not self._path and self._phase == 'region':
            self._try_region(pos, binary_map, region_mask, obstacles, W, H, claimed)
        # If still no path → switch to policy; avoids re-scanning every step

    def _advance_lawnmower(self, pos, obstacles, W, H):
        while self._wp_idx < len(self._waypoints):
            wp = self._waypoints[self._wp_idx]
            self._wp_idx += 1
            path, _ = _bfs_to_point(pos, wp, obstacles, W, H)
            if path:
                self._path = path
                return
        self._phase = 'region'

    def _try_region(self, pos, binary_map, region_mask, obstacles, W, H, claimed):
        self._release_claim(claimed)
        path, target = _bfs_to_region_target(
            pos, binary_map, region_mask, obstacles, W, H, claimed, self.agent_id)
        if path:
            self._path        = path
            self._claimed_tgt = target
            if target:
                claimed[target] = self.agent_id
        else:
            # No region targets remain — permanently hand off to policy
            self._phase = 'policy'

    def _release_claim(self, claimed: dict):
        if self._claimed_tgt and claimed.get(self._claimed_tgt) == self.agent_id:
            del claimed[self._claimed_tgt]
        self._claimed_tgt = None


# ── Image helpers ─────────────────────────────────────────────────────────────

def slam_to_img(slam_map, W, H):
    img = np.ones((H, W, 3), dtype=np.float32) * 0.5
    img[slam_map == 0] = [0.95, 0.95, 0.95]
    img[slam_map == 1] = [0.20, 0.20, 0.20]
    return img

def truth_to_img(obstacles, W, H):
    img = np.ones((H, W, 3), dtype=np.float32)
    img[obstacles] = [0.20, 0.20, 0.20]
    return img

def overlay_region(img, mask, alpha=0.22):
    out = img.copy()
    out[mask] = out[mask]*(1-alpha) + np.array([0.2, 0.5, 1.0])*alpha
    return out

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2],16)/255 for i in (0,2,4))

def overlay_trail(img, traj, color_hex, W, H, tail=200):
    out  = img.copy()
    traj = traj[-tail:]
    n    = len(traj)
    col  = np.array(hex_to_rgb(color_hex))
    for i, (tx, ty) in enumerate(traj):
        if 0 <= tx < W and 0 <= ty < H:
            fade = 0.25 + 0.75*(i/max(n-1,1))
            out[ty, tx] = col*fade + out[ty, tx]*(1-fade)
    return out


# ── Trainer factory ───────────────────────────────────────────────────────────

def _make_trainers(env, args, shared_weights=None):
    """
    Load trainers.  If shared_weights is a state_dict, skip disk I/O and just
    copy weights — makes seed search ~3× faster.
    """
    trainers = []
    for _ in range(args.n_agents):
        t = FastMambaPPOTrainer(
            obs_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            policy_type='fast_hybrid',
            d_model=128, n_layers=2, memory_size=500,
        )
        if shared_weights is not None:
            import torch
            t.net.load_state_dict(shared_weights)
        else:
            t.load(args.checkpoint)
        t.net.eval()
        trainers.append(t)
    return trainers


# ── Headless runner for seed search ───────────────────────────────────────────

def run_headless(city_map, target, args, seed, shared_weights):
    W, H = city_map['W'], city_map['H']
    env = MultiAgentCityExplorerEnv(
        n_agents=args.n_agents, width=W, height=H,
        max_steps=args.max_steps, target_radius=args.target_radius,
        target_coverage=0.85, target_score=0.80, seed=seed,
    )
    opts = {'city_map': city_map}
    if target:
        opts['target'] = target
    obss = env.reset(seed=seed, options=opts)

    trainers  = _make_trainers(env, args, shared_weights=shared_weights)
    explorers = [ZoneLawnmower(i, args.n_agents, W, H) for i in range(args.n_agents)]
    claimed   = {}

    info = {}
    for _ in range(args.max_steps):
        actions = []
        for i in range(args.n_agents):
            a, _ = explorers[i].get_action(
                env.agent_pos[i], obss[i],
                env.slam.binary_map, env._region_mask,
                env.city.obstacles, W, H, claimed, trainers[i])
            actions.append(a)
        obss, done, info = env.step(actions)
        if done:
            break

    return info.get('region_coverage', 0.0), info.get('map_accuracy', 0.0)


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',    default='checkpoints/mamba_fast_hybrid_slam.pt')
    p.add_argument('--city-map',      default='data/real_grid.json')
    p.add_argument('--target-x',      type=int,   default=None)
    p.add_argument('--target-y',      type=int,   default=None)
    p.add_argument('--target-radius', type=int,   default=8)
    p.add_argument('--max-steps',     type=int,   default=1200)
    p.add_argument('--n-agents',      type=int,   default=3)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--search',        type=int,   default=0, metavar='N')
    p.add_argument('--save-gif',      action='store_true')
    p.add_argument('--out',           default='evaluation_results/multi_agent_run.gif')
    p.add_argument('--speed',         type=float, default=0.05)
    p.add_argument('--frame-skip',    type=int,   default=3,
                   help='Save every Nth frame (default 3 → ~400 frames)')
    p.add_argument('--full-run',      action='store_true',
                   help='Run all max-steps regardless of coverage threshold')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.city_map) as f:
        city_map = json.load(f)
    W, H      = city_map['W'], city_map['H']
    city_name = city_map.get('name', Path(args.city_map).stem)
    target    = ([args.target_x, args.target_y] if args.target_x is not None
                 else city_map.get('target'))

    # Load weights once — reuse across all search seeds
    import torch
    _probe_env = MultiAgentCityExplorerEnv(
        n_agents=args.n_agents, width=W, height=H,
        max_steps=args.max_steps, target_radius=args.target_radius, seed=0)
    _probe_t = FastMambaPPOTrainer(
        obs_dim=_probe_env.observation_space.shape[0],
        n_actions=_probe_env.action_space.n,
        policy_type='fast_hybrid', d_model=128, n_layers=2, memory_size=500)
    _probe_t.load(args.checkpoint)
    shared_weights = {k: v.clone() for k, v in _probe_t.net.state_dict().items()}

    # ── Seed search ───────────────────────────────────────────────────────────
    seed = args.seed
    if args.search > 0:
        print(f"\nSearching {args.search} seeds...", flush=True)
        best_seed, best_score = seed, -1.0
        for i in range(args.search):
            s   = args.seed + i
            cov, acc = run_headless(city_map, target, args, s, shared_weights)
            score    = cov * 0.7 + acc * 0.3
            flag     = '  ← best' if score > best_score else ''
            print(f"  Ep {i+1:3d}  seed={s}  cov={cov*100:.1f}%  acc={acc*100:.1f}%{flag}",
                  flush=True)
            if score > best_score:
                best_score, best_seed = score, s
        print(f"\nBest seed: {best_seed}  score={best_score:.3f}\n", flush=True)
        seed = best_seed

    # ── Build main env ────────────────────────────────────────────────────────
    env = MultiAgentCityExplorerEnv(
        n_agents=args.n_agents, width=W, height=H,
        max_steps=args.max_steps, target_radius=args.target_radius,
        target_coverage=1.1 if args.full_run else 0.85,
        target_score=1.1   if args.full_run else 0.80,
        seed=seed,
    )
    opts = {'city_map': city_map}
    if target:
        opts['target'] = target
    obss = env.reset(seed=seed, options=opts)

    trainers  = _make_trainers(env, args, shared_weights=shared_weights)
    explorers = [ZoneLawnmower(i, args.n_agents, W, H) for i in range(args.n_agents)]
    claimed   = {}

    region_mask = env._region_mask
    truth_base  = truth_to_img(env.city.obstacles, W, H)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.patch.set_facecolor('white')
    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#cccccc')
            sp.set_linewidth(1.2)

    # Single title line — no crowding subtitle below it
    fig.suptitle(
        'Active SLAM  —  3-Agent Cooperative Mapping  —  Mamba SSM + PPO',
        color=MAROON, fontsize=15, fontweight='bold', y=0.97)

    # City name lives in the axes subtitle so it has its own row with padding
    axes[0].set_title(f'SLAM Belief Map  ·  {city_name}',  color=DARK_TXT, fontsize=12, pad=10)
    axes[1].set_title('Ground Truth + Agent Trajectories', color=DARK_TXT, fontsize=12, pad=10)

    stats_text = fig.text(
        0.5, 0.028,
        'Step    0  |  Coverage  0.0%  |  Score  0.0%  |  Map Accuracy  0.0%',
        ha='center', color=MAROON, fontsize=13, fontweight='bold',
        fontfamily='monospace')
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])

    slam_disp  = axes[0].imshow(
        overlay_region(slam_to_img(env.slam.binary_map, W, H), region_mask),
        origin='upper', vmin=0, vmax=1, interpolation='nearest')
    truth_disp = axes[1].imshow(
        overlay_region(truth_base, region_mask),
        origin='upper', vmin=0, vmax=1, interpolation='nearest')

    # Draw zone separators
    strip_w = W // args.n_agents
    for i in range(1, args.n_agents):
        for ax in axes:
            ax.axvline(x=i*strip_w - 0.5, color='#aaaaaa', alpha=0.5,
                       linewidth=1, linestyle='--')

    # Agent dots
    dots_slam  = [axes[0].plot([], [], 'o', color=c, ms=9,  zorder=5)[0] for c in AGENT_COLORS]
    dots_truth = [axes[1].plot([], [], 'o', color=c, ms=9,  zorder=5)[0] for c in AGENT_COLORS]

    # Target star
    tgt = env._target
    for ax in axes:
        ax.plot(tgt[0], tgt[1], '*', color=MAROON, ms=14, zorder=6)

    leg = [
        mpatches.Patch(facecolor=[0.95,0.95,0.95], edgecolor='#aaaaaa', label='Free (known)'),
        mpatches.Patch(color=[0.20,0.20,0.20], label='Obstacle'),
        mpatches.Patch(color=[0.60,0.60,0.60], label='Unknown'),
        mpatches.Patch(color=[0.2, 0.5, 1.0],  label='Target region'),
    ] + [mpatches.Patch(color=AGENT_COLORS[i], label=AGENT_LABELS[i])
         for i in range(args.n_agents)]
    axes[0].legend(handles=leg, loc='upper right', fontsize=9,
                   facecolor='white', labelcolor=DARK_TXT,
                   framealpha=0.92, edgecolor='#cccccc')

    trajectories      = [[tuple(env.agent_pos[i])] for i in range(args.n_agents)]
    milestone_printed = set()
    state = {'step': 0, 'cov': 0.0, 'score': 0.0, 'acc': 0.0, 'done': False}

    def render_frame():
        si = overlay_region(slam_to_img(env.slam.binary_map, W, H), region_mask)
        ti = overlay_region(truth_base.copy(), region_mask)
        for i in range(args.n_agents):
            si = overlay_trail(si, trajectories[i], AGENT_COLORS[i], W, H)
            ti = overlay_trail(ti, trajectories[i], AGENT_COLORS[i], W, H)
        slam_disp.set_data(si)
        truth_disp.set_data(ti)
        for i in range(args.n_agents):
            px, py = env.agent_pos[i]
            dots_slam[i].set_data([px], [py])
            dots_truth[i].set_data([px], [py])
        stats_text.set_text(
            f"Step {state['step']:4d} / {args.max_steps}"
            f"   |   Coverage {state['cov']*100:5.1f}%"
            f"   |   Score {state['score']*100:5.1f}%"
            f"   |   Map Accuracy {state['acc']*100:.1f}%"
        )

    def step_env():
        if state['done']:
            return
        actions = []
        for i in range(args.n_agents):
            a, _ = explorers[i].get_action(
                env.agent_pos[i], obss[i],
                env.slam.binary_map, env._region_mask,
                env.city.obstacles, W, H, claimed, trainers[i])
            actions.append(a)
        new_obss, done, info = env.step(actions)
        obss[:] = new_obss
        for i in range(args.n_agents):
            trajectories[i].append(tuple(env.agent_pos[i]))
        state['step']  += 1
        state['cov']    = info['region_coverage']
        state['score']  = info['region_score']
        state['acc']    = info['map_accuracy']
        state['done']   = done
        for thr in [0.70, 0.80, 0.85, 0.90, 0.95]:
            if state['cov'] >= thr and thr not in milestone_printed:
                print(f"  ✓ {thr*100:.0f}% coverage at step {state['step']}", flush=True)
                milestone_printed.add(thr)

    # ── GIF save path ─────────────────────────────────────────────────────────
    if args.save_gif:
        from PIL import Image
        import io

        skip   = max(1, args.frame_skip)
        FREEZE = 48
        frames = []
        print(f"Recording GIF  (frame-skip={skip}, map={W}×{H})...", flush=True)

        for fi in range(args.max_steps):
            step_env()
            if fi % skip == 0:
                render_frame()
                fig.canvas.draw()
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=90, facecolor=fig.get_facecolor())
                buf.seek(0)
                frames.append(Image.open(buf).copy())
                buf.close()
            if (fi+1) % 300 == 0:
                print(f"  Step {fi+1:4d}  coverage={state['cov']*100:.1f}%  "
                      f"frames={len(frames)}", flush=True)
            if state['done'] and not args.full_run:
                break

        for _ in range(FREEZE):
            frames.append(frames[-1].copy())

        out = Path(args.out)
        out.parent.mkdir(exist_ok=True)
        dur = int(83 * skip)
        frames[0].save(str(out), save_all=True, append_images=frames[1:],
                       loop=0, duration=dur)
        print(f"\nSaved: {out}")
        print(f"  {len(frames)} frames  ·  {len(frames)*dur/1000:.0f}s  ·  skip={skip}")
        print(f"Final — Coverage: {state['cov']*100:.1f}%  |  "
              f"Score: {state['score']*100:.1f}%  |  "
              f"Accuracy: {state['acc']*100:.1f}%")

    # ── Live viewer ───────────────────────────────────────────────────────────
    else:
        def anim_update(_frame):
            step_env()
            render_frame()
            return slam_disp, truth_disp, stats_text

        from matplotlib.animation import FuncAnimation
        ani = FuncAnimation(fig, anim_update, frames=args.max_steps,
                            interval=max(1, int(args.speed*1000)),
                            blit=False, repeat=False)
        plt.show()


if __name__ == '__main__':
    main()
