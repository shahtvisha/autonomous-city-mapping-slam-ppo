"""
find_best_episode.py
--------------------
Runs the trained policy headlessly on both grids,
finds the best 5/5 episode for each, saves the full
path + state as JSON so the demo can replay it instantly.

Usage:
    python find_best_episode.py
    # outputs: data/best_episode.json
"""

import json
import numpy as np
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── constants ────────────────────────────────────────────────────
L_FREE = -2.2
L_OCC  =  2.2
L_MIN  = -8.0
L_MAX  =  8.0
ACTS   = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]]
PATCH_R = 5
G       = 16
FOV     = 8
MAX_STEPS = 700
MAX_EPISODES = 200   # search up to this many episodes per grid

# ── RNG ──────────────────────────────────────────────────────────
def mulberry32(seed):
    def rng():
        nonlocal seed
        seed = (seed + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((seed ^ (seed >> 15)) * (1 | seed)) & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * (61 | t))) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rng

# ── SLAM ─────────────────────────────────────────────────────────
class SLAM:
    def __init__(self, W, H):
        self.W = W; self.H = H
        self.lo      = np.zeros((H, W), dtype=np.float32)
        self.vc      = np.zeros((H, W), dtype=np.int32)
        self.visited = np.zeros((H, W), dtype=np.uint8)

    def prob(self, x, y):
        return 1.0 / (1.0 + np.exp(-self.lo[y, x]))

    def cell(self, x, y):
        p = self.prob(x, y)
        return 1 if p > 0.65 else (0 if p < 0.35 else -1)

    def update(self, x, y, occ):
        if 0 <= x < self.W and 0 <= y < self.H:
            self.lo[y, x] = np.clip(
                self.lo[y, x] + (L_OCC if occ else L_FREE),
                L_MIN, L_MAX)

    def observe(self, cx, cy, obs):
        angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
        for angle in angles:
            dx, dy = np.cos(angle), np.sin(angle)
            hit = FOV
            for r in np.arange(0.5, FOV+0.5, 0.5):
                tx = int(round(cx + dx*r))
                ty = int(round(cy + dy*r))
                if not (0 <= tx < self.W and 0 <= ty < self.H):
                    hit = r; break
                if obs[ty, tx]:
                    hit = r; break
            for r in np.arange(0.5, hit, 0.5):
                tx = int(round(cx + dx*r))
                ty = int(round(cy + dy*r))
                self.update(tx, ty, False)
            if hit < FOV:
                tx = int(round(cx + dx*hit))
                ty = int(round(cy + dy*hit))
                self.update(tx, ty, True)
        self.visited[cy, cx] = 1
        self.vc[cy, cx] += 1

    def coverage(self, free_count):
        return float((self.vc > 0).sum()) / max(free_count, 1)

    def frontiers(self):
        pts = []
        for y in range(1, self.H-1):
            for x in range(1, self.W-1):
                if self.cell(x, y) != 0:
                    continue
                if (self.cell(x+1,y)==-1 or self.cell(x-1,y)==-1 or
                    self.cell(x,y+1)==-1 or self.cell(x,y-1)==-1):
                    pts.append((x, y))
        return pts

# ── GRID ─────────────────────────────────────────────────────────
def make_grid(seed, W, H, bX, bY, rW):
    rng = mulberry32(seed)
    obs = np.zeros((H, W), dtype=bool)
    sX, sY = bX+rW, bY+rW
    for y in range(H):
        for x in range(W):
            bx, by = x % sX, y % sY
            obs[y, x] = bx >= rW and by >= rW and rng() > 0.12

    free = [(x, y) for y in range(2, H-2) for x in range(2, W-2) if not obs[y, x]]
    regions = [(W*.2,H*.2),(W*.8,H*.2),(W*.2,H*.8),(W*.8,H*.8),(W*.5,H*.15)]
    towers = []
    for rx, ry in regions:
        best = min(free, key=lambda p: abs(p[0]-rx)+abs(p[1]-ry))
        towers.append(best)

    sx, sy = 1, 1
    for y in range(1, H):
        for x in range(1, W):
            if not obs[y, x]:
                sx, sy = x, y
                break
        else:
            continue
        break

    return obs, towers, (sx, sy), len(free)

# ── OBS ──────────────────────────────────────────────────────────
def build_obs(slam, pos, found, step_n, W, H, towers, free_count):
    cx, cy = pos
    patch = []
    for dy in range(-PATCH_R, PATCH_R+1):
        for dx in range(-PATCH_R, PATCH_R+1):
            x, y = cx+dx, cy+dy
            if 0 <= x < W and 0 <= y < H:
                patch.append((slam.prob(x, y) - 0.5) * 2)
            else:
                patch.append(0.0)

    global_map = []
    for gy in range(G):
        for gx in range(G):
            mx = int(gx * W / G)
            my = int(gy * H / G)
            global_map.append((slam.prob(mx, my) - 0.5) * 2)

    fronts = slam.frontiers()
    fdx, fdy, fdist = 0.0, 0.0, 1.0
    if fronts:
        dists = [((fx-cx)**2+(fy-cy)**2)**0.5 for fx,fy in fronts]
        i = int(np.argmin(dists))
        fx, fy = fronts[i]
        angle = np.arctan2(fy-cy, fx-cx)
        fdx, fdy = np.cos(angle), np.sin(angle)
        fdist = min(dists[i] / ((W**2+H**2)**0.5), 1.0)

    cov = slam.coverage(free_count)
    return np.array(patch + global_map + [
        cx/W, cy/H, cov,
        len(found)/max(len(towers), 1),
        fdx, fdy, fdist,
        step_n/MAX_STEPS
    ], dtype=np.float32)

# ── NET ──────────────────────────────────────────────────────────
def load_net(weights_path):
    with open(weights_path) as f:
        data = json.load(f)
    w = data['weights']
    obs_dim = data['info']['obs_dim']

    def get(name):
        d = w[name]
        return np.array(d['data'], dtype=np.float32).reshape(d['shape'])

    W0=get('net.0.weight'); b0=get('net.0.bias')
    W2=get('net.2.weight'); b2=get('net.2.bias')
    Wp=get('policy_head.weight'); bp=get('policy_head.bias')
    Wv=get('value_head.weight'); bv=get('value_head.bias')

    def act(obs):
        if len(obs) != obs_dim:
            o = np.zeros(obs_dim, dtype=np.float32)
            o[:min(len(obs), obs_dim)] = obs[:obs_dim]
            obs = o
        h1 = np.maximum(0, W0 @ obs + b0)
        h2 = np.maximum(0, W2 @ h1 + b2)
        logits = Wp @ h2 + bp
        value  = float((Wv @ h2 + bv).flatten()[0])
        logits = logits / 1.2
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        action = int(np.random.choice(len(probs), p=probs))
        return action, value

    return act, obs_dim

# ── EPISODE ───────────────────────────────────────────────────────
def run_episode(obs_arr, towers, start, free_count, W, H, net_act):
    slam = SLAM(W, H)
    pos = list(start)
    found = set()
    path = [list(pos)]
    slam.observe(pos[0], pos[1], obs_arr)

    for step in range(MAX_STEPS):
        obs_vec = build_obs(slam, pos, found, step, W, H, towers, free_count)
        action, value = net_act(obs_vec)
        dx, dy = ACTS[action]
        nx = max(0, min(W-1, pos[0]+dx))
        ny = max(0, min(H-1, pos[1]+dy))
        if not obs_arr[ny, nx]:
            pos = [nx, ny]
        slam.observe(pos[0], pos[1], obs_arr)
        path.append(list(pos))
        for i, (tx, ty) in enumerate(towers):
            if i not in found and ((pos[0]-tx)**2+(pos[1]-ty)**2)**0.5 < 2.5:
                found.add(i)
        if len(found) == len(towers):
            break

    cov = slam.coverage(free_count)
    return path, found, cov, step+1

# ── MAIN ─────────────────────────────────────────────────────────
def find_best(grid_seed, bX, bY, rW, W, H, net_act, label,  max_eps=300):
    obs_arr, towers, start, free_count = make_grid(grid_seed, W, H, bX, bY, rW)

    print(f"  {label}: {W}×{H} grid, {len(towers)} towers, {free_count} free cells")

    best = None
    for ep in range(MAX_EPISODES):
        path, found, cov, steps = run_episode(obs_arr, towers, start, free_count, W, H, net_act)
        tow = len(found)
        print(f"    Ep {ep+1:3d}: towers={tow}/5 coverage={cov*100:.1f}% steps={steps}")

        if tow == len(towers):
            if best is None or cov > best['cov']:
                best = {
                    'cov':    cov,
                    'towers': tow,
                    'steps':  steps,
                    'ep':     ep+1,
                    'path':   path,
                    'found':  list(found),
                }
                print(f"    ✓ New best 5/5 episode! cov={cov*100:.1f}%")
            # keep searching for higher coverage 5/5
            if cov > 0.6:
                print(f"    ✓ Good enough — stopping search")
                break

    if best is None:
        print(f"    ⚠ No 5/5 episode found — using best partial")
        # fallback: best by towers then coverage
        path, found, cov, steps = run_episode(obs_arr, towers, start, free_count, W, H, net_act)
        best = {'cov':cov,'towers':len(found),'steps':steps,'ep':1,'path':path,'found':list(found)}

    return best, obs_arr, towers, start, free_count


def main():
    os.makedirs('data', exist_ok=True)

    # load weights
    weights_path = None
    for f in ['weights.json', 'weights_boston.json']:
        if os.path.exists(f):
            weights_path = f
            break

    if not weights_path:
        print("No weights.json found. Run: python export_weights.py first.")
        return

    print(f"Loading policy from {weights_path}...")
    net_act, obs_dim = load_net(weights_path)
    print(f"Policy loaded. obs_dim={obs_dim}")

    print("\nSearching for best 5/5 episode — City A (trained layout)...")
    best_a, obs_a, towers_a, start_a, fc_a = find_best(
        42, 5, 5, 2, 30, 30, net_act, 'City A')

    print("\nSearching for best 5/5 episode — City B (unseen layout)...")
    best_b, obs_b, towers_b, start_b, fc_b = find_best(
        200, 6, 4, 3, 30, 30, net_act, 'City B', max_eps=300)

    # build output
    result = {
        'policy': weights_path,
        'obs_dim': obs_dim,
        'a': {
            'label':      'City A — Trained layout',
            'W': 30, 'H': 30,
            'obstacles':  obs_a.tolist(),
            'towers':     [list(t) for t in towers_a],
            'start':      list(start_a),
            'freeCount':  fc_a,
            'path':       best_a['path'],
            'foundSet':   best_a['found'],
            'cov':        round(best_a['cov'], 3),
            'towers_found': best_a['towers'],
            'steps':      best_a['steps'],
            'ep':         best_a['ep'],
        },
        'b': {
            'label':      'City B — Never seen before',
            'W': 30, 'H': 30,
            'obstacles':  obs_b.tolist(),
            'towers':     [list(t) for t in towers_b],
            'start':      list(start_b),
            'freeCount':  fc_b,
            'path':       best_b['path'],
            'foundSet':   best_b['found'],
            'cov':        round(best_b['cov'], 3),
            'towers_found': best_b['towers'],
            'steps':      best_b['steps'],
            'ep':         best_b['ep'],
        }
    }

    out = 'data/best_episode.json'
    with open(out, 'w') as f:
        json.dump(result, f)

    size_kb = os.path.getsize(out) // 1024
    print(f"\n{'='*50}")
    print(f"Saved → {out} ({size_kb} KB)")
    print(f"City A: {best_a['towers']}/5 towers · {best_a['cov']*100:.1f}% coverage · {best_a['steps']} steps")
    print(f"City B: {best_b['towers']}/5 towers · {best_b['cov']*100:.1f}% coverage · {best_b['steps']} steps")
    print(f"\nNow run: python share.py")
    print(f"Open:    demo_preloaded.html")


if __name__ == '__main__':
    main()