"""
SLAM Occupancy Grid
-------------------
Implements a probabilistic occupancy grid with:
  - Bayesian log-odds updates from sensor readings
  - Frontier detection (boundary between known/unknown)
  - Bresenham raycast for LiDAR simulation
  - Coverage and entropy metrics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# Log-odds constants
L_FREE    = np.log(0.1 / 0.9)   # strong free update
L_OCC     = np.log(0.9 / 0.1)   # strong occupied update
L_MIN     = -10.0
L_MAX     =  10.0
L_PRIOR   =  0.0                 # log-odds of 0.5 (unknown)


@dataclass
class SLAMConfig:
    width:      int   = 50
    height:     int   = 50
    resolution: float = 1.0    # metres per cell
    fov_range:  float = 8.0    # sensor range in cells
    fov_angle:  float = 360.0  # degrees


class OccupancyGrid:
    """
    Probabilistic occupancy grid using log-odds representation.
    Suitable for both simulated and real sensor fusion.
    """

    def __init__(self, cfg: SLAMConfig):
        self.cfg = cfg
        self.W = cfg.width
        self.H = cfg.height
        # log-odds grid: 0 = unknown (p=0.5)
        self.log_odds = np.zeros((self.H, self.W), dtype=np.float32)
        # track visited cells for path history
        self.visited = np.zeros((self.H, self.W), dtype=np.uint8)
        # step counter per cell for recency weighting
        self.visit_count = np.zeros((self.H, self.W), dtype=np.int32)
        self.total_updates = 0

    # ── Core update ────────────────────────────────────────────────

    def update_cell(self, x: int, y: int, occupied: bool):
        """Bayesian log-odds update for a single cell."""
        if not self._in_bounds(x, y):
            return
        delta = L_OCC if occupied else L_FREE
        self.log_odds[y, x] = np.clip(
            self.log_odds[y, x] + delta - L_PRIOR,
            L_MIN, L_MAX
        )

    def raycast_update(self, origin: Tuple[float, float],
                       angle_deg: float, hit_range: float,
                       max_range: float):
        """
        Update cells along a single ray using Bresenham line algorithm.
        Cells up to hit_range are free; cell at hit_range is occupied.
        """
        ox, oy = origin
        angle = np.radians(angle_deg)
        dx, dy = np.cos(angle), np.sin(angle)

        # trace free cells
        end_range = min(hit_range, max_range)
        cells = self._bresenham(ox, oy, ox + dx * end_range, oy + dy * end_range)
        for (cx, cy) in cells[:-1]:
            self.update_cell(cx, cy, occupied=False)

        # mark hit cell as occupied
        if hit_range < max_range:
            cx = int(ox + dx * hit_range)
            cy = int(oy + dy * hit_range)
            self.update_cell(cx, cy, occupied=True)

        self.total_updates += 1

    def observe_region(self, cx: int, cy: int,
                       obstacles: np.ndarray, fov: int):
        """
        Simulate a 360° LiDAR sweep from (cx, cy) with given FOV radius.
        obstacles: boolean array (H, W) — True where obstacles exist.
        """
        angles = np.linspace(0, 360, 72, endpoint=False)
        for angle in angles:
            rad = np.radians(angle)
            ddx, ddy = np.cos(rad), np.sin(rad)
            hit = fov  # default: no hit
            for r in np.arange(0.5, fov + 0.5, 0.5):
                tx = int(round(cx + ddx * r))
                ty = int(round(cy + ddy * r))
                if not self._in_bounds(tx, ty):
                    hit = r
                    break
                if obstacles[ty, tx]:
                    hit = r
                    break
            self.raycast_update((cx, cy), angle, hit, fov)

        # mark current cell visited
        if self._in_bounds(cx, cy):
            self.visited[cy, cx] = 1
            self.visit_count[cy, cx] += 1

    # ── Derived maps ───────────────────────────────────────────────

    @property
    def prob_map(self) -> np.ndarray:
        """Convert log-odds to probability [0,1]."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    @property
    def binary_map(self) -> np.ndarray:
        """
        -1 = unknown, 0 = free, 1 = occupied
        Only cells with enough updates are classified.
        """
        p = self.prob_map
        out = np.full((self.H, self.W), -1, dtype=np.int8)
        out[p > 0.65] = 1   # occupied
        out[p < 0.35] = 0   # free
        return out

    @property
    def coverage(self) -> float:
        """Fraction of cells classified (not unknown)."""
        bm = self.binary_map
        return float(np.sum(bm >= 0)) / (self.W * self.H)

    @property
    def entropy(self) -> float:
        """Mean map entropy — lower = more certain."""
        p = np.clip(self.prob_map, 1e-6, 1 - 1e-6)
        h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        return float(np.mean(h))

    # ── Frontier detection ─────────────────────────────────────────

    def get_frontiers(self) -> List[Tuple[int, int]]:
        """
        Frontiers = free cells adjacent to unknown cells.
        These are the most informative cells for the agent to visit.
        """
        bm = self.binary_map
        frontiers = []
        free_mask = bm == 0
        unknown_mask = bm == -1

        # dilate unknown by 1 cell
        from scipy.ndimage import binary_dilation
        dilated_unknown = binary_dilation(unknown_mask, iterations=1)

        # frontier = free AND adjacent to unknown
        frontier_mask = free_mask & dilated_unknown
        ys, xs = np.where(frontier_mask)
        frontiers = list(zip(xs.tolist(), ys.tolist()))
        return frontiers

    def get_frontier_clusters(self, min_size: int = 3) -> List[Tuple[int, int]]:
        """
        Cluster frontiers and return centroids.
        Useful as high-level waypoint targets for the RL agent.
        """
        frontiers = self.get_frontiers()
        if not frontiers:
            return []
        try:
            from scipy.ndimage import label
            bm = self.binary_map
            frontier_img = np.zeros((self.H, self.W), dtype=np.uint8)
            for fx, fy in frontiers:
                frontier_img[fy, fx] = 1
            labeled, n = label(frontier_img)
            clusters = []
            for i in range(1, n + 1):
                ys, xs = np.where(labeled == i)
                if len(xs) >= min_size:
                    clusters.append((int(np.mean(xs)), int(np.mean(ys))))
            return clusters
        except ImportError:
            return frontiers[:10]

    # ── Observation for RL ─────────────────────────────────────────

    def local_patch(self, cx: int, cy: int, radius: int) -> np.ndarray:
        """
        Extract a (2r+1)×(2r+1) patch of the probability map centred
        at (cx, cy). Pads with 0.5 (unknown) at boundaries.
        Normalised to [-1, 1] for neural network input.
        """
        size = 2 * radius + 1
        patch = np.full((size, size), 0.5, dtype=np.float32)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy
                if self._in_bounds(x, y):
                    patch[dy + radius, dx + radius] = self.prob_map[y, x]
        return (patch - 0.5) * 2.0  # [-1, 1]

    def global_downsampled(self, target_size: int = 16) -> np.ndarray:
        """
        Downsample the full probability map to target_size × target_size.
        Gives the agent a bird's-eye view of the whole map.
        """
        from PIL import Image
        img = (self.prob_map * 255).astype(np.uint8)
        pil = Image.fromarray(img).resize(
            (target_size, target_size), Image.BILINEAR)
        arr = np.array(pil, dtype=np.float32) / 255.0
        return (arr - 0.5) * 2.0

    # ── Utilities ──────────────────────────────────────────────────

    def reset(self):
        self.log_odds[:] = 0.0
        self.visited[:] = 0
        self.visit_count[:] = 0
        self.total_updates = 0

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.W and 0 <= y < self.H

    @staticmethod
    def _bresenham(x0: float, y0: float,
                   x1: float, y1: float) -> List[Tuple[int, int]]:
        """Bresenham line — returns integer cell coords along the ray."""
        cells = []
        x, y = int(round(x0)), int(round(y0))
        xe, ye = int(round(x1)), int(round(y1))
        dx, dy = abs(xe - x), abs(ye - y)
        sx = 1 if x < xe else -1
        sy = 1 if y < ye else -1
        err = dx - dy
        while True:
            cells.append((x, y))
            if x == xe and y == ye:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return cells