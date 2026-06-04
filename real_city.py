"""
real_city.py — Real City Map from OpenStreetMap
------------------------------------------------
Downloads a real city area, converts buildings + roads to an
occupancy grid, and places electricity infrastructure at real
locations (or estimated positions if unavailable).

Usage:
    python real_city.py --city "Manhattan, New York" --size 500
    python real_city.py --city "Boston, MA" --size 300
    python real_city.py --lat 51.5074 --lon -0.1278 --size 400  # London
"""

import numpy as np
import argparse
import json
import os
import pickle
import requests

try:
    import osmnx as ox
    import shapely
    HAS_OSM = True
except ImportError:
    HAS_OSM = False


CELL_SIZE = 5  # metres per grid cell

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--city',   default='Kendall Square, Cambridge, MA')
    p.add_argument('--lat',    type=float, default=None)
    p.add_argument('--lon',    type=float, default=None)
    p.add_argument('--size',   type=int,   default=400,  help='area in metres')
    p.add_argument('--out',    default='data/city_map.pkl')
    p.add_argument('--cell',   type=int,   default=5,    help='metres per cell')
    return p.parse_args()


class RealCityMap:
    """
    Downloads and rasterizes a real city area into an occupancy grid.
    Buildings = occupied. Roads = free. Unknown areas = -1.
    """

    def __init__(self, grid_w, grid_h, cell_size,
                 obstacles, towers, bounds, name="city"):
        self.W = grid_w
        self.H = grid_h
        self.cell_size = cell_size
        self.obstacles = obstacles      # (H, W) bool array
        self.towers = towers            # list of (x, y) grid coords
        self.bounds = bounds            # (min_lat, min_lon, max_lat, max_lon)
        self.name = name

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"City map saved → {path}")

    @staticmethod
    def load(path: str) -> 'RealCityMap':
        with open(path, 'rb') as f:
            return pickle.load(f)

    def summary(self):
        obs_pct = self.obstacles.mean() * 100
        free_pct = 100 - obs_pct
        print(f"\nCity map: {self.name}")
        print(f"  Grid:      {self.W} × {self.H} cells")
        print(f"  Cell size: {self.cell_size}m")
        print(f"  Area:      {self.W*self.cell_size}m × {self.H*self.cell_size}m")
        print(f"  Free:      {free_pct:.1f}%")
        print(f"  Occupied:  {obs_pct:.1f}%")
        print(f"  Towers:    {len(self.towers)}")


def build_from_osmnx(args) -> RealCityMap:
    """Download real OSM data and rasterize."""
    if not HAS_OSM:
        print("osmnx not installed. Run: pip install osmnx shapely")
        return build_synthetic(args)

    print(f"Downloading OSM data for: {args.city or f'({args.lat},{args.lon})'}")
    try:
        ox.settings.log_console = False
        ox.settings.use_cache = True

        dist = args.size // 2
        if args.lat and args.lon:
            point = (args.lat, args.lon)
        else:
            point = ox.geocode(args.city)

        # download buildings and roads
        print("  Fetching buildings...")
        try:
            buildings = ox.features_from_point(
                point, tags={'building': True}, dist=dist)
        except Exception:
            buildings = None

        print("  Fetching roads...")
        G = ox.graph_from_point(point, dist=dist, network_type='drive')
        nodes, edges = ox.graph_to_gdfs(G)

        # get bounds
        bounds = (
            nodes.geometry.y.min(),
            nodes.geometry.x.min(),
            nodes.geometry.y.max(),
            nodes.geometry.x.max(),
        )

        cell = args.cell
        lat_range = bounds[2] - bounds[0]
        lon_range = bounds[3] - bounds[1]
        # degrees to metres approx
        metres_per_lat = 111320
        metres_per_lon = 111320 * np.cos(np.radians((bounds[0]+bounds[2])/2))
        H = max(20, int(lat_range * metres_per_lat / cell))
        W = max(20, int(lon_range * metres_per_lon / cell))
        H = min(H, 80); W = min(W, 80)  # cap for performance

        print(f"  Grid: {W}×{H} cells at {cell}m resolution")
        obstacles = np.zeros((H, W), dtype=bool)

        def geo_to_grid(lat, lon):
            x = int((lon - bounds[1]) / lon_range * (W - 1))
            y = int((bounds[2] - lat) / lat_range * (H - 1))
            return np.clip(x, 0, W-1), np.clip(y, 0, H-1)

        # rasterize buildings
        if buildings is not None:
            print(f"  Rasterizing {len(buildings)} buildings...")
            for _, row in buildings.iterrows():
                try:
                    geom = row.geometry
                    if geom.geom_type in ('Polygon', 'MultiPolygon'):
                        polys = [geom] if geom.geom_type=='Polygon' else geom.geoms
                        for poly in polys:
                            coords = list(poly.exterior.coords)
                            xs = [geo_to_grid(lat,lon)[0] for lon,lat in coords]
                            ys = [geo_to_grid(lat,lon)[1] for lon,lat in coords]
                            # fill polygon
                            from PIL import Image, ImageDraw
                            img = Image.new('L', (W, H), 0)
                            draw = ImageDraw.Draw(img)
                            pts = list(zip(xs, ys))
                            if len(pts) >= 3:
                                draw.polygon(pts, fill=1)
                            obstacles |= np.array(img, dtype=bool)
                except Exception:
                    continue

        # find electricity infrastructure from OSM
        print("  Searching for power infrastructure...")
        towers = []
        try:
            power = ox.features_from_point(
                point,
                tags={'power': ['tower', 'pole', 'substation']},
                dist=dist
            )
            for _, row in power.iterrows():
                try:
                    lat2 = row.geometry.centroid.y
                    lon2 = row.geometry.centroid.x
                    gx, gy = geo_to_grid(lat2, lon2)
                    if not obstacles[gy, gx]:
                        towers.append((gx, gy))
                except Exception:
                    continue
            print(f"  Found {len(towers)} real power infrastructure points")
        except Exception:
            print("  No power data found — placing estimated towers")

        # if no real towers found, place on major road intersections
        if len(towers) < 3:
            print("  Placing towers at estimated positions...")
            towers = _estimate_tower_positions(obstacles, W, H, n=6)

        towers = towers[:8]  # cap at 8

        name = args.city or f"({args.lat:.3f},{args.lon:.3f})"
        city = RealCityMap(W, H, cell, obstacles, towers, bounds, name)
        city.summary()
        return city

    except Exception as e:
        print(f"OSM download failed: {e}")
        print("Falling back to synthetic city map")
        return build_synthetic(args)


def build_synthetic(args) -> RealCityMap:
    """
    Fallback: generate a realistic city-like grid without OSM.
    Uses real urban planning ratios (building coverage ~40%, road ~30%).
    """
    print("Building synthetic urban grid...")
    cell = args.cell
    W = args.size // cell
    H = args.size // cell
    W = min(W, 60); H = min(H, 60)

    obstacles = np.zeros((H, W), dtype=bool)
    rng = np.random.default_rng(42)

    # city block grid — realistic urban density
    block = 8; road = 2; step = block + road
    for gy in range(road, H - block, step):
        for gx in range(road, W - block, step):
            # main building block
            bw = rng.integers(4, block)
            bh = rng.integers(4, block)
            ox2 = gx + rng.integers(0, max(1, block-bw))
            oy2 = gy + rng.integers(0, max(1, block-bh))
            obstacles[oy2:min(oy2+bh,H), ox2:min(ox2+bw,W)] = True
            # internal courtyards (hollow buildings)
            if bw > 5 and bh > 5:
                obstacles[oy2+1:oy2+bh-1, ox2+1:ox2+bw-1] = False

    towers = _estimate_tower_positions(obstacles, W, H, n=6)
    bounds = (42.36, -71.06, 42.37, -71.05)  # Boston-like
    city = RealCityMap(W, H, cell, obstacles, towers, bounds,
                       name=args.city or "Synthetic Urban Grid")
    city.summary()
    return city


def _estimate_tower_positions(obstacles, W, H, n=6):
    """Place towers at road intersections — realistic power line routing."""
    rng = np.random.default_rng(99)
    free_ys, free_xs = np.where(~obstacles)
    valid = [(x,y) for x,y in zip(free_xs,free_ys)
             if 3<x<W-3 and 3<y<H-3]
    if len(valid) < n:
        return valid[:n]
    # spread towers across the map
    towers = []
    regions = [(W//3, H//3), (2*W//3, H//3), (W//3, 2*H//3),
               (2*W//3, 2*H//3), (W//2, H//6), (W//2, 5*H//6)]
    for rx, ry in regions[:n]:
        best = min(valid, key=lambda p: abs(p[0]-rx)+abs(p[1]-ry))
        towers.append(best)
    return towers


def main():
    args = parse_args()
    os.makedirs('data', exist_ok=True)

    city = build_from_osmnx(args)
    city.save(args.out)

    # save as JSON for the web demo too
    json_path = args.out.replace('.pkl', '.json')
    with open(json_path, 'w') as f:
        json.dump({
            'W': int(city.W), 'H': int(city.H),
            'cell_size': int(city.cell_size),
            'name': city.name,
            'obstacles': city.obstacles.tolist(),
            'towers': [[int(x), int(y)] for x,y in city.towers],
        }, f, cls=NumpyEncoder)
    print(f"JSON map → {json_path}")


if __name__ == '__main__':
    main()