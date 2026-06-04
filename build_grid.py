"""
build_grid.py — Build accurate occupancy grid from OpenStreetMap
"""
import argparse, json, os, numpy as np
from PIL import Image, ImageDraw

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lat',    type=float, default=42.3601)
    p.add_argument('--lon',    type=float, default=-71.0942)
    p.add_argument('--radius', type=int,   default=300)
    p.add_argument('--cell',   type=int,   default=10)
    p.add_argument('--out',    type=str,   default='data/real_grid.json')
    p.add_argument('--name',   type=str,   default='Kendall Square, Boston')
    return p.parse_args()

def main():
    args = parse_args()
    lat, lon, radius, cell = args.lat, args.lon, args.radius, args.cell

    try:
        import osmnx as ox
        ox.settings.log_console = False
        ox.settings.use_cache = True
    except ImportError:
        print("pip install osmnx shapely Pillow"); return

    # grid dims
    m_per_lat = 111320
    m_per_lon = 111320 * np.cos(np.radians(lat))
    dlat = radius / m_per_lat
    dlon = radius / m_per_lon
    north, south = lat+dlat, lat-dlat
    east,  west  = lon+dlon, lon-dlon
    W = (2*radius)//cell
    H = (2*radius)//cell
    print(f"Grid: {W}x{H}, area: {W*cell}m x {H*cell}m")

    def to_px(glat, glon):
        x = int((glon-west)/(east-west)*W)
        y = int((north-glat)/(north-south)*H)
        return max(0,min(W-1,x)), max(0,min(H-1,y))

    # canvas: white=free, black=obstacle
    img = Image.new('L', (W, H), 0)   # start all obstacle
    draw = ImageDraw.Draw(img)

    # 1. draw roads as free
    print("Fetching roads...")
    G = ox.graph_from_point((lat,lon), dist=radius, network_type='all')
    _, edges = ox.graph_to_gdfs(G)
    for _, row in edges.iterrows():
        try:
            pts = [to_px(c[1],c[0]) for c in row.geometry.coords]
            hw = row.get('highway','residential')
            if isinstance(hw,list): hw=hw[0]
            w = {'motorway':6,'trunk':5,'primary':5,'secondary':4,
                 'tertiary':3,'residential':3,'service':2,
                 'footway':2,'path':1,'cycleway':1}.get(str(hw),2)
            if len(pts)>=2: draw.line(pts, fill=255, width=w)
        except: pass
    print(f"  {len(edges)} road segments drawn")

    # 2. overdraw buildings as obstacles
    print("Fetching buildings...")
    try:
        bldgs = ox.features_from_point((lat,lon),
                    tags={'building':True}, dist=radius)
        count = 0
        for _, row in bldgs.iterrows():
            try:
                geom = row.geometry
                polys = [geom] if geom.geom_type=='Polygon' else \
                        list(geom.geoms) if geom.geom_type=='MultiPolygon' else []
                for poly in polys:
                    pts = [to_px(c[1],c[0]) for c in poly.exterior.coords]
                    if len(pts)>=3:
                        draw.polygon(pts, fill=0)
                        count+=1
            except: pass
        print(f"  {count} building polygons drawn")
    except Exception as e:
        print(f"  Buildings failed: {e}")

    # 3. convert to obstacle boolean array
    arr = np.array(img)          # 0=obstacle, 255=free
    obs_bool = (arr == 0)        # True where blocked
    free = int((arr>0).sum())
    total = W*H
    print(f"Free: {free}/{total} = {free/total*100:.1f}%")
    print(f"Obstacles: {total-free}/{total}")

    # 4. tower positions — on free cells, spread across map
    free_ys, free_xs = np.where(arr>0)
    free_pts = list(zip(free_xs.tolist(), free_ys.tolist()))
    towers = []
    if free_pts:
        for rx,ry in [(W*.2,H*.2),(W*.8,H*.2),(W*.2,H*.8),
                      (W*.8,H*.8),(W*.5,H*.15),(W*.5,H*.85)]:
            best = min(free_pts, key=lambda p: abs(p[0]-rx)+abs(p[1]-ry))
            if best not in towers: towers.append(list(best))
    else:
        towers = [[5,5],[W-5,5],[5,H-5],[W-5,H-5],[W//2,5],[W//2,H-5]]

    # 5. start position near centre on free cell
    start = [W//2, H//2]
    for r in range(max(W,H)):
        found = False
        for dy in range(-r,r+1):
            for dx in range(-r,r+1):
                x,y = W//2+dx, H//2+dy
                if 0<=x<W and 0<=y<H and arr[y,x]>0:
                    start = [x,y]; found=True; break
            if found: break
        if found: break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result = {
        'name': args.name, 'lat': lat, 'lon': lon,
        'radius': radius, 'cell_size': cell, 'W': W, 'H': H,
        'north': north, 'south': south, 'east': east, 'west': west,
        'obstacles': obs_bool.tolist(),
        'towers': towers, 'start': start,
        'stats': {
            'free_pct':  round(free/total*100,1),
            'obs_pct':   round((total-free)/total*100,1),
            'free_cells': free,
        }
    }

    with open(args.out,'w') as f:
        json.dump(result, f)
    print(f"Saved -> {args.out}")

if __name__ == '__main__':
    main()