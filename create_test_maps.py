"""
create_test_maps.py — Generate Test Maps for Evaluation
--------------------------------------------------------
Create various city configurations to test your trained agent.

Usage:
    python create_test_maps.py --all
    python create_test_maps.py --type grid --size 60
"""

import argparse
import json
import numpy as np
from pathlib import Path


def create_grid_city(width, height, block_size=8, street_width=2):
    """Create a grid-pattern city (like Manhattan)"""
    obstacles = []
    
    for y in range(0, height, block_size + street_width):
        for x in range(0, width, block_size + street_width):
            # Create building block
            for by in range(block_size):
                for bx in range(block_size):
                    ox = x + bx
                    oy = y + by
                    if ox < width and oy < height:
                        obstacles.append([int(ox), int(oy)])
    
    return {
        'W': width,
        'H': height,
        'obstacles': obstacles,
        'name': f'grid_{width}x{height}',
        'description': f'Grid city {width}×{height} with {len(obstacles)} obstacles'
    }


def create_radial_city(width, height, n_rings=3, n_sectors=8):
    """Create a radial city (like Paris)"""
    obstacles = []
    cx, cy = width // 2, height // 2
    
    for ring in range(1, n_rings + 1):
        radius = ring * min(width, height) // (2 * (n_rings + 1))
        
        for sector in range(n_sectors):
            angle_start = sector * 2 * np.pi / n_sectors
            angle_end = (sector + 1) * 2 * np.pi / n_sectors
            
            # Create building in sector
            for angle in np.linspace(angle_start, angle_end, 20):
                for r in range(radius - 3, radius + 3):
                    x = int(cx + r * np.cos(angle))
                    y = int(cy + r * np.sin(angle))
                    if 0 <= x < width and 0 <= y < height:
                        obstacles.append([x, y])
    
    return {
        'W': width,
        'H': height,
        'obstacles': obstacles,
        'name': f'radial_{width}x{height}',
        'description': f'Radial city {width}×{height} with {n_rings} rings'
    }


def create_random_city(width, height, density=0.3, min_building_size=3, max_building_size=8):
    """Create a random city layout"""
    obstacles = []
    occupied = np.zeros((height, width), dtype=bool)
    
    n_buildings = int((width * height * density) / (min_building_size ** 2))
    
    for _ in range(n_buildings):
        # Random building size
        bw = np.random.randint(min_building_size, max_building_size + 1)
        bh = np.random.randint(min_building_size, max_building_size + 1)
        
        # Random position
        x = np.random.randint(0, width - bw)
        y = np.random.randint(0, height - bh)
        
        # Check if space is free
        if not occupied[y:y+bh, x:x+bw].any():
            for by in range(bh):
                for bx in range(bw):
                    obstacles.append([int(x + bx), int(y + by)])
                    occupied[y + by, x + bx] = True
    
    return {
        'W': width,
        'H': height,
        'obstacles': obstacles,
        'name': f'random_{width}x{height}',
        'description': f'Random city {width}×{height} with {len(obstacles)} obstacles'
    }


def create_maze_city(width, height, wall_thickness=1):
    """Create a maze-like city"""
    obstacles = []
    
    # Create maze using recursive division
    def divide(x, y, w, h, horizontal):
        if w < 4 or h < 4:
            return
        
        if horizontal:
            # Horizontal wall
            wall_y = y + np.random.randint(1, h - 1)
            gap_x = x + np.random.randint(0, w)
            
            for wx in range(x, x + w):
                if wx != gap_x:
                    for t in range(wall_thickness):
                        if wall_y + t < y + h:
                            obstacles.append([int(wx), int(wall_y + t)])
            
            divide(x, y, w, wall_y - y, not horizontal)
            divide(x, wall_y + wall_thickness, w, h - (wall_y - y) - wall_thickness, not horizontal)
        else:
            # Vertical wall
            wall_x = x + np.random.randint(1, w - 1)
            gap_y = y + np.random.randint(0, h)
            
            for wy in range(y, y + h):
                if wy != gap_y:
                    for t in range(wall_thickness):
                        if wall_x + t < x + w:
                            obstacles.append([int(wall_x + t), int(wy)])
            
            divide(x, y, wall_x - x, h, not horizontal)
            divide(wall_x + wall_thickness, y, w - (wall_x - x) - wall_thickness, h, not horizontal)
    
    # Start division
    divide(0, 0, width, height, np.random.choice([True, False]))
    
    return {
        'W': width,
        'H': height,
        'obstacles': obstacles,
        'name': f'maze_{width}x{height}',
        'description': f'Maze city {width}×{height} with {len(obstacles)} obstacles'
    }


def create_sparse_city(width, height, n_buildings=10):
    """Create a sparse city with few large buildings"""
    obstacles = []
    
    for _ in range(n_buildings):
        bw = np.random.randint(8, 15)
        bh = np.random.randint(8, 15)
        x = np.random.randint(0, width - bw)
        y = np.random.randint(0, height - bh)
        
        for by in range(bh):
            for bx in range(bw):
                obstacles.append([int(x + bx), int(y + by)])
    
    return {
        'W': width,
        'H': height,
        'obstacles': obstacles,
        'name': f'sparse_{width}x{height}',
        'description': f'Sparse city {width}×{height} with {n_buildings} buildings'
    }


def save_map(city_data, output_dir='data/test_maps'):
    """Save city map to JSON"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"{city_data['name']}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(city_data, f, indent=2)
    
    print(f"✓ Created: {filepath}")
    print(f"  Size: {city_data['W']}×{city_data['H']}")
    print(f"  Obstacles: {len(city_data['obstacles'])}")
    print(f"  Description: {city_data['description']}\n")
    
    return filepath


def parse_args():
    p = argparse.ArgumentParser(description='Generate test maps for evaluation')
    
    p.add_argument('--all', action='store_true',
                   help='Generate all map types')
    p.add_argument('--type', type=str, 
                   choices=['grid', 'radial', 'random', 'maze', 'sparse'],
                   help='Map type to generate')
    p.add_argument('--size', type=int, default=40,
                   help='Map size (width=height)')
    p.add_argument('--width', type=int, default=None,
                   help='Map width (overrides --size)')
    p.add_argument('--height', type=int, default=None,
                   help='Map height (overrides --size)')
    p.add_argument('--output', type=str, default='data/test_maps',
                   help='Output directory')
    
    return p.parse_args()


def main():
    args = parse_args()
    
    width = args.width if args.width else args.size
    height = args.height if args.height else args.size
    
    print("\n" + "="*70)
    print("TEST MAP GENERATOR")
    print("="*70 + "\n")
    
    if args.all:
        print("Generating all map types...\n")
        
        # Small maps (20×20)
        save_map(create_grid_city(20, 20), args.output)
        save_map(create_random_city(20, 20), args.output)
        
        # Medium maps (40×40)
        save_map(create_grid_city(40, 40), args.output)
        save_map(create_radial_city(40, 40), args.output)
        save_map(create_random_city(40, 40), args.output)
        save_map(create_maze_city(40, 40), args.output)
        save_map(create_sparse_city(40, 40, n_buildings=15), args.output)
        
        # Large maps (60×60)
        save_map(create_grid_city(60, 60), args.output)
        save_map(create_radial_city(60, 60, n_rings=4), args.output)
        save_map(create_random_city(60, 60), args.output)
        
        print("="*70)
        print("✓ All maps generated!")
        print(f"  Location: {args.output}/")
        print("\nTest with:")
        print(f"  python evaluate_model.py --checkpoint <model.pt> \\")
        print(f"      --city-map {args.output}/grid_40x40.json")
        
    elif args.type:
        print(f"Generating {args.type} map ({width}×{height})...\n")
        
        if args.type == 'grid':
            city_data = create_grid_city(width, height)
        elif args.type == 'radial':
            city_data = create_radial_city(width, height)
        elif args.type == 'random':
            city_data = create_random_city(width, height)
        elif args.type == 'maze':
            city_data = create_maze_city(width, height)
        elif args.type == 'sparse':
            city_data = create_sparse_city(width, height)
        
        filepath = save_map(city_data, args.output)
        
        print("="*70)
        print("✓ Map generated!")
        print(f"\nTest with:")
        print(f"  python evaluate_model.py --checkpoint <model.pt> \\")
        print(f"      --city-map {filepath}")
    
    else:
        print("Please specify --all or --type <map_type>")
        print("Run with --help for more options")
    
    print()


if __name__ == '__main__':
    main()
