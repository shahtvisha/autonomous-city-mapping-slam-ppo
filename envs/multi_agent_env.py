"""
MultiAgentCityExplorerEnv
-------------------------
N agents sharing a single Bayesian occupancy grid (SLAM).
Each agent runs an independent policy (same checkpoint, separate Mamba state).
The shared SLAM means every agent's LiDAR updates immediately benefit all agents.

Observations are identical in shape to CityExplorerEnv so the existing
checkpoint loads without any changes to the policy.
"""

import numpy as np
from collections import deque
from slam.occupancy_grid import OccupancyGrid, SLAMConfig
from envs.city_env import CityExplorerEnv


_DIRS8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]


def _largest_connected_component(obstacles, W, H):
    """
    BFS over the 8-connected free-cell graph.
    Returns the set of (x, y) cells in the largest connected component.
    Agents that start here are guaranteed to be able to reach each other
    and every lawnmower waypoint that falls in the same component.
    """
    visited = np.zeros((H, W), dtype=bool)
    best = set()
    for sy in range(H):
        for sx in range(W):
            if obstacles[sy, sx] or visited[sy, sx]:
                continue
            component = set()
            q = deque([(sx, sy)])
            visited[sy, sx] = True
            while q:
                x, y = q.popleft()
                component.add((x, y))
                for dx, dy in _DIRS8:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < W and 0 <= ny < H and not obstacles[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((nx, ny))
            if len(component) > len(best):
                best = component
    return best


class MultiAgentCityExplorerEnv:
    """
    Wraps CityExplorerEnv to support N agents on a single shared SLAM grid.

    step(actions)  -> obs_list, done, info
    reset(...)     -> obs_list
    """

    ACTIONS = CityExplorerEnv.ACTIONS      # shared action table

    def __init__(self,
                 n_agents:        int   = 3,
                 width:           int   = 40,
                 height:          int   = 40,
                 max_steps:       int   = 1200,
                 fov:             float = 7.0,
                 target_radius:   int   = 8,
                 target_coverage: float = 0.85,
                 target_score:    float = 0.80,
                 seed:            int   = 42):

        self.n_agents        = n_agents
        self.max_steps       = max_steps
        self.target_coverage = target_coverage
        self.target_score    = target_score

        # Base env handles map generation, SLAM, target selection
        self.base = CityExplorerEnv(
            width=width, height=height,
            n_towers=0,
            max_steps=max_steps,
            fov=fov,
            target_radius=target_radius,
            target_coverage=target_coverage,
            target_score=target_score,
            seed=seed,
        )

        # Mirror observation / action spaces for external callers
        self.observation_space = self.base.observation_space
        self.action_space      = self.base.action_space

        # Per-agent state
        self.agent_pos         = [np.zeros(2, dtype=np.int32)] * n_agents
        self.agent_prev_action = [-1] * n_agents
        self.agent_cells_visited: list[set] = [set() for _ in range(n_agents)]

        self._step           = 0
        self._cov_milestones: set = set()

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self.base.reset(seed=seed, options=options)
        self._step           = 0
        self._cov_milestones = set()
        self.agent_prev_action = [-1] * self.n_agents

        # Spread agents around the target region
        self.agent_pos = self._spread_agents(seed=seed)
        self.agent_cells_visited = [{tuple(p)} for p in self.agent_pos]

        # Initialise shared SLAM from every agent's starting position
        for pos in self.agent_pos:
            self.base.slam.observe_region(
                int(pos[0]), int(pos[1]),
                self.base.city.obstacles,
                int(self.base.fov),
            )

        return [self._obs(i) for i in range(self.n_agents)]

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, actions):
        self._step += 1

        for i, action in enumerate(actions):
            dx, dy = self.ACTIONS[action]
            nx = int(np.clip(self.agent_pos[i][0] + dx, 0, self.W - 1))
            ny = int(np.clip(self.agent_pos[i][1] + dy, 0, self.H - 1))
            if not self.base.city.obstacles[ny, nx]:
                self.agent_pos[i] = np.array([nx, ny])
            self.agent_cells_visited[i].add(tuple(self.agent_pos[i]))

            # Every agent's move updates the shared SLAM
            self.base.slam.observe_region(
                int(self.agent_pos[i][0]), int(self.agent_pos[i][1]),
                self.base.city.obstacles,
                int(self.base.fov),
            )
            self.agent_prev_action[i] = action

        metrics = self.base._region_metrics()
        cov     = metrics['region_coverage']
        score   = metrics['region_score']

        done = (
            (cov >= self.target_coverage and score >= self.target_score)
            or self._step >= self.max_steps
        )

        info = {
            'region_coverage': round(cov,   3),
            'region_score':    round(score, 3),
            'map_accuracy':    round(metrics['map_accuracy'], 3),
            'step':            self._step,
        }
        return [self._obs(i) for i in range(self.n_agents)], done, info

    # ── Observation builder ───────────────────────────────────────────────────

    def _obs(self, i: int) -> np.ndarray:
        """
        Build a 390-dim observation for agent i.
        Local patch comes from agent i's position; global map is shared.
        Identical structure to CityExplorerEnv._get_obs().
        """
        pos = self.agent_pos[i]
        cx, cy = int(pos[0]), int(pos[1])
        slam = self.base.slam

        local = slam.local_patch(cx, cy, self.base.patch_r).flatten()
        try:
            glb = slam.global_downsampled(self.base.G).flatten()
        except Exception:
            glb = np.zeros(self.base.G * self.base.G, dtype=np.float32)

        frontiers = slam.get_frontiers()
        if frontiers:
            dists = [np.hypot(fx - cx, fy - cy) for fx, fy in frontiers]
            j     = int(np.argmin(dists))
            fx, fy = frontiers[j]
            fdir  = np.array([np.cos(np.arctan2(fy-cy, fx-cx)),
                               np.sin(np.arctan2(fy-cy, fx-cx))], dtype=np.float32)
            fdist = np.array([dists[j] / np.hypot(self.W, self.H)], dtype=np.float32)
        else:
            fdir  = np.zeros(2, dtype=np.float32)
            fdist = np.ones(1, dtype=np.float32)

        m       = self.base._region_metrics()
        tx, ty  = self.base._target
        rel     = np.array([(tx-cx)/max(self.W,1), (ty-cy)/max(self.H,1)], dtype=np.float32)
        tdist   = np.linalg.norm([tx-cx, ty-cy]) / max(np.hypot(self.W, self.H), 1.0)
        rnorm   = self.base._target_radius() / max(np.hypot(self.W, self.H), 1.0)

        meta = np.array([
            cx/self.W, cy/self.H,
            m['region_coverage'], m['region_score'], slam.coverage,
            *fdir, *fdist, *rel,
            tdist, rnorm,
            self._step / self.max_steps,
        ], dtype=np.float32)

        return np.concatenate([local, glb, meta])

    # ── Agent placement ───────────────────────────────────────────────────────

    def _spread_agents(self, seed=None) -> list:
        """
        Place agents spread across the full map in distinct zones, restricted
        to the largest connected component of free cells.

        Agents in disconnected pockets (isolated courtyards, dead-end streets)
        can never reach waypoints in other parts of the map, so we pre-filter
        candidates to the LCC before sampling.  If a zone has no LCC cells,
        the agent falls back to any LCC cell.
        """
        rng = np.random.default_rng(seed if seed is not None else 0)
        obs = self.base.city.obstacles

        # Cache LCC so it's only computed once per reset
        if not hasattr(self, '_lcc') or self._lcc is None:
            self._lcc = _largest_connected_component(obs, self.W, self.H)
        lcc = self._lcc

        strip_w = self.W // self.n_agents
        lcc_list = list(lcc)

        positions = []
        for i in range(self.n_agents):
            x_lo = i * strip_w
            x_hi = (i + 1) * strip_w if i < self.n_agents - 1 else self.W

            # Prefer LCC cells in this zone's strip
            strip_lcc = [(x, y) for x, y in lcc_list if x_lo <= x < x_hi]
            pool = strip_lcc if strip_lcc else lcc_list

            idx = int(rng.integers(0, len(pool)))
            x, y = pool[idx]
            positions.append(np.array([x, y], dtype=np.int32))

        return positions

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def W(self):            return self.base.W
    @property
    def H(self):            return self.base.H
    @property
    def slam(self):         return self.base.slam
    @property
    def city(self):         return self.base.city
    @property
    def _region_mask(self): return self.base._region_mask
    @property
    def _target(self):      return self.base._target
