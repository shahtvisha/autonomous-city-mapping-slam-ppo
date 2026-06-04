"""
CityExplorerEnv - SLAM-guided PPO
---------------------------------
Default task: map a target region. The simulator keeps the ground-truth city
map hidden from the agent, while rewards/evaluation compare the agent's SLAM
belief against truth inside a requested radius.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, List
import pygame
from slam.occupancy_grid import OccupancyGrid, SLAMConfig


class CityMap:
    def __init__(self, width, height, n_towers=5, seed=None):
        self.W = width
        self.H = height
        self.n_towers = n_towers
        self.rng = np.random.default_rng(seed)
        self.obstacles = np.zeros((height, width), dtype=bool)
        self.towers = []
        self._generate()

    def _generate(self):
        W, H = self.W, self.H
        obs = np.zeros((H, W), dtype=bool)
        block_size, road_w = 6, 3
        step = block_size + road_w
        for gy in range(road_w, H-block_size, step):
            for gx in range(road_w, W-block_size, step):
                bw = self.rng.integers(3, block_size+1)
                bh = self.rng.integers(3, block_size+1)
                ox = gx + self.rng.integers(0, block_size-bw+1)
                oy = gy + self.rng.integers(0, block_size-bh+1)
                obs[oy:oy+bh, ox:ox+bw] = True
        self.obstacles = obs
        free_ys, free_xs = np.where(~obs)
        margin = 3
        valid = [(x,y) for x,y in zip(free_xs, free_ys)
                 if margin < x < W-margin and margin < y < H-margin]
        if len(valid) < self.n_towers:
            valid = list(zip(free_xs.tolist(), free_ys.tolist()))
        idxs = self.rng.choice(len(valid), size=min(self.n_towers, len(valid)), replace=False)
        self.towers = [valid[i] for i in idxs]


class CityExplorerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    ACTIONS = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]])

    def __init__(self, width=40, height=40, n_towers=5,
                 max_steps=600, patch_r=5, global_size=16,
                 fov=7.0, render_mode=None, seed=None, city_map=None,
                 task="region", target_center=None, target_radius=None,
                 target_coverage=0.95, target_score=0.90):
        super().__init__()
        self.W = width
        self.H = height
        self.n_towers = n_towers
        self.max_steps = max_steps
        self.patch_r = patch_r
        self.G = global_size
        self.fov = fov
        self.render_mode = render_mode
        self._seed = seed
        self._city_map = city_map
        self.task = task
        self.target_center = target_center
        self.target_radius = target_radius
        self.target_coverage = target_coverage
        self.target_score = target_score

        slam_cfg = SLAMConfig(width=width, height=height, fov_range=fov)
        self.slam = OccupancyGrid(slam_cfg)

        P = 2*patch_r+1
        self.meta_dim = 12
        obs_dim = P*P + global_size*global_size + self.meta_dim
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)

        self.city = None
        self.pos = np.zeros(2, dtype=np.int32)
        self._step = 0
        self._found = set()
        self._prev_coverage = 0.0
        self._prev_frontier_dist = np.inf
        self._cov_milestones = set()
        self._cells_visited = set()
        self._free_cells = 0
        self._target = np.zeros(2, dtype=np.int32)
        self._region_mask = np.zeros((height, width), dtype=bool)
        self._prev_region_coverage = 0.0
        self._prev_region_score = 0.0
        self._prev_entropy = 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._city_map:
            W = self._city_map['W']
            H = self._city_map['H']
            obs_flat = [item for row in self._city_map['obstacles'] for item in row]
            obstacles = np.array(obs_flat, dtype=bool).reshape(H, W)
            self.W = W
            self.H = H
            self.slam = OccupancyGrid(SLAMConfig(width=W, height=H, fov_range=self.fov))
            self.city = type('City', (), {
                'obstacles': obstacles,
                'towers': [tuple(t) for t in self._city_map.get('towers', [])]
            })()
        else:
            s = seed if seed is not None else self._seed
            self.city = CityMap(self.W, self.H, self.n_towers, seed=s)
            self.slam.reset()

        self.n_towers = len(self.city.towers)
        self._free_cells = int((~self.city.obstacles).sum())
        self._target = self._choose_target(options)
        self._region_mask = self._make_region_mask(self._target, self._target_radius())
        self.pos = self._choose_start()
        self._step = 0
        self._found = set()
        self._prev_coverage = 0.0
        self._prev_frontier_dist = np.inf
        self._cov_milestones = set()
        self._cells_visited = {tuple(self.pos)}
        self._update_slam()
        metrics = self._region_metrics()
        self._prev_region_coverage = metrics['region_coverage']
        self._prev_region_score = metrics['region_score']
        self._prev_entropy = self.slam.entropy
        return self._get_obs(), {}

    def step(self, action):
        dx, dy = self.ACTIONS[action]
        nx = int(np.clip(self.pos[0]+dx, 0, self.W-1))
        ny = int(np.clip(self.pos[1]+dy, 0, self.H-1))
        collision = bool(self.city.obstacles[ny, nx])
        if not collision:
            self.pos = np.array([nx, ny])
        self._update_slam()
        self._step += 1
        self._cells_visited.add(tuple(self.pos))
        reward, info = self._compute_reward(collision)
        terminated = self._is_success(info)
        truncated = self._step >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        cx, cy = self.pos
        local = self.slam.local_patch(cx, cy, self.patch_r).flatten()
        try:
            glb = self.slam.global_downsampled(self.G).flatten()
        except Exception:
            glb = np.zeros(self.G*self.G, dtype=np.float32)

        frontiers = self.slam.get_frontiers()
        if frontiers:
            dists = [np.hypot(fx-cx, fy-cy) for fx,fy in frontiers]
            i = int(np.argmin(dists))
            fx, fy = frontiers[i]
            angle = np.arctan2(fy-cy, fx-cx)
            fdir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            fdist = np.array([dists[i]/np.hypot(self.W, self.H)], dtype=np.float32)
        else:
            fdir = np.zeros(2, dtype=np.float32)
            fdist = np.ones(1, dtype=np.float32)

        metrics = self._region_metrics()
        region_cov = metrics['region_coverage']
        region_score = metrics['region_score']
        map_cov = self.slam.coverage
        tx, ty = self._target
        rel = np.array([(tx-cx)/max(self.W, 1), (ty-cy)/max(self.H, 1)], dtype=np.float32)
        target_dist = np.linalg.norm([tx-cx, ty-cy]) / max(np.hypot(self.W, self.H), 1.0)
        radius_norm = self._target_radius() / max(np.hypot(self.W, self.H), 1.0)

        meta = np.array([
            cx/self.W, cy/self.H,
            region_cov,
            region_score,
            map_cov,
            *fdir, *fdist,
            *rel,
            target_dist,
            radius_norm,
            self._step/self.max_steps
        ], dtype=np.float32)

        return np.concatenate([local, glb, meta])

    def _compute_reward(self, collision):
        reward = 0.0
        cx, cy = self.pos
        metrics = self._region_metrics()
        region_cov = metrics['region_coverage']
        region_score = metrics['region_score']

        coverage_gain = region_cov - self._prev_region_coverage
        score_gain = region_score - self._prev_region_score
        entropy_gain = self._prev_entropy - self.slam.entropy

        # Primary signal: improve the reconstruction of the requested region.
        reward += 80.0 * coverage_gain
        reward += 120.0 * score_gain
        reward += 5.0 * max(entropy_gain, 0.0)

        if self.slam.visit_count[cy, cx] == 1:
            reward += 0.2

        frontiers = self.slam.get_frontiers()
        if frontiers:
            nearest = min(np.hypot(fx-cx, fy-cy) for fx,fy in frontiers)
            if self._prev_frontier_dist < np.inf:
                progress = self._prev_frontier_dist - nearest
                if progress > 0:
                    reward += progress * 0.2
            self._prev_frontier_dist = nearest
        else:
            reward += 10.0

        for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]:
            key = ('region_cov', t)
            if region_cov >= t and key not in self._cov_milestones:
                self._cov_milestones.add(key)
                reward += 5.0
            key = ('region_score', t)
            if region_score >= t and key not in self._cov_milestones:
                self._cov_milestones.add(t)
                reward += 8.0

        if region_cov >= self.target_coverage and region_score >= self.target_score:
            reward += 100.0

        if collision:
            reward -= 2.0
        reward -= 0.01

        self._prev_coverage = self.slam.coverage
        self._prev_region_coverage = region_cov
        self._prev_region_score = region_score
        self._prev_entropy = self.slam.entropy

        info = {
            'coverage':        round(region_cov, 3),
            'region_coverage': round(region_cov, 3),
            'region_score':    round(region_score, 3),
            'region_accuracy': round(metrics['region_accuracy'], 3),
            'map_coverage':    round(self.slam.coverage, 3),
            'map_accuracy':    round(metrics['map_accuracy'], 3),
            'target':          self._target.tolist(),
            'target_radius':   self._target_radius(),
            'towers_found':    0,
            'new_towers':      0,
            'step':            self._step,
            'unique_cells':    len(self._cells_visited),
            'free_cells':      self._free_cells,
        }
        return float(reward), info

    def _update_slam(self):
        cx, cy = self.pos
        self.slam.observe_region(cx, cy, self.city.obstacles, int(self.fov))

    def _random_free_pos(self):
        free_ys, free_xs = np.where(~self.city.obstacles)
        i = self.np_random.integers(0, len(free_xs))
        return np.array([free_xs[i], free_ys[i]], dtype=np.int32)

    def _choose_start(self):
        if self._city_map and 'start' in self._city_map:
            sx, sy = self._city_map['start']
            if 0 <= sx < self.W and 0 <= sy < self.H and not self.city.obstacles[sy, sx]:
                return np.array([sx, sy], dtype=np.int32)
        return self._random_free_pos()

    def _target_radius(self):
        if self.target_radius is not None:
            return int(self.target_radius)
        return max(4, min(self.W, self.H) // 5)

    def _choose_target(self, options=None):
        if options and 'target' in options:
            target = options['target']
        elif self.target_center is not None:
            target = self.target_center
        elif self._city_map and 'target' in self._city_map:
            target = self._city_map['target']
        else:
            return self._random_free_pos()

        tx = int(np.clip(target[0], 0, self.W - 1))
        ty = int(np.clip(target[1], 0, self.H - 1))
        return np.array([tx, ty], dtype=np.int32)

    def _make_region_mask(self, center, radius):
        yy, xx = np.ogrid[:self.H, :self.W]
        dist = np.hypot(xx - int(center[0]), yy - int(center[1]))
        return dist <= radius

    def _region_metrics(self):
        bm = self.slam.binary_map
        known = bm >= 0
        pred_occ = bm == 1
        truth_occ = self.city.obstacles

        region = self._region_mask
        region_total = max(int(region.sum()), 1)
        region_known = known & region
        region_correct = region_known & (pred_occ == truth_occ)

        map_known_total = max(int(known.sum()), 1)
        map_correct = known & (pred_occ == truth_occ)

        return {
            'region_coverage': float(region_known.sum()) / region_total,
            'region_score': float(region_correct.sum()) / region_total,
            'region_accuracy': float(region_correct.sum()) / max(int(region_known.sum()), 1),
            'map_accuracy': float(map_correct.sum()) / map_known_total,
        }

    def _is_success(self, info):
        if self.task == "legacy_towers":
            return len(self._found) == self.n_towers
        return (
            info['region_coverage'] >= self.target_coverage and
            info['region_score'] >= self.target_score
        )

    def render(self):
        return None

    def close(self):
        if hasattr(self, '_screen') and self._screen:
            pygame.quit()
