#!/usr/bin/env python3
"""
share.py — One command to share your demo
------------------------------------------
Serves the demo locally and optionally exposes it publicly via ngrok.

Usage:
    python share.py                    # local only: http://localhost:8080
    python share.py --public           # share via ngrok (needs ngrok installed)
    python share.py --city "Boston"    # download real city map first
    python share.py --port 9000        # custom port
"""

import argparse
import os
import sys
import json
import threading
import http.server
import webbrowser
import time
import subprocess


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--port',   type=int,  default=8080)
    p.add_argument('--public', action='store_true', help='expose via ngrok')
    p.add_argument('--city',   type=str,  default=None,
                   help='download real city map e.g. "Kendall Square, Cambridge MA"')
    p.add_argument('--no-browser', action='store_true')
    return p.parse_args()


def download_city(city_name: str):
    """Download a real city map from OSM."""
    print(f"\nDownloading real city map: {city_name}")
    try:
        result = subprocess.run(
            [sys.executable, 'real_city.py',
             '--city', city_name,
             '--size', '400',
             '--out', 'data/city_map.pkl'],
            capture_output=False
        )
        if result.returncode == 0:
            print("City map ready.")
        else:
            print("City download failed — demo will use synthetic map")
    except FileNotFoundError:
        print("real_city.py not found")


def get_share_url(port: int) -> str:
    """Try to get ngrok public URL."""
    try:
        import urllib.request
        with urllib.request.urlopen(f'http://localhost:4040/api/tunnels', timeout=2) as r:
            data = json.loads(r.read())
            tunnels = data.get('tunnels', [])
            for t in tunnels:
                if str(port) in t.get('config', {}).get('addr', ''):
                    return t['public_url']
    except Exception:
        pass
    return None


def start_ngrok(port: int):
    """Start ngrok tunnel."""
    try:
        proc = subprocess.Popen(
            ['ngrok', 'http', str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(2)  # wait for tunnel
        url = get_share_url(port)
        return proc, url
    except FileNotFoundError:
        print("ngrok not found. Install from https://ngrok.com/download")
        return None, None


class SilentHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()


def serve(port: int):
    handler = SilentHandler
    with http.server.HTTPServer(('0.0.0.0', port), handler) as httpd:
        httpd.serve_forever()


def main():
    args = parse_args()

    # ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if not os.path.exists('demo_final.html'):
        print("demo_final.html not found. Make sure you're in the active_slam_ppo directory.")
        sys.exit(1)

    # download real city if requested
    if args.city:
        download_city(args.city)

    # start HTTP server in background
    server_thread = threading.Thread(
        target=serve, args=(args.port,), daemon=True)
    server_thread.start()

    local_url = f'http://localhost:{args.port}/demo_final.html'

    print(f"\n{'='*52}")
    print(f"  Active SLAM + PPO — Live Demo")
    print(f"{'='*52}")
    print(f"  Local:  {local_url}")

    ngrok_proc = None
    if args.public:
        print("  Starting ngrok tunnel...")
        ngrok_proc, public_url = start_ngrok(args.port)
        if public_url:
            share_url = public_url + '/demo_final.html'
            print(f"\n  ✓ Public URL (share this):")
            print(f"    {share_url}")
        else:
            print("  Could not get ngrok URL. Check ngrok is running.")
    else:
        print(f"\n  To share publicly, run:")
        print(f"    python share.py --public")
        print(f"\n  Or install ngrok and run manually:")
        print(f"    ngrok http {args.port}")

    print(f"\n  Dashboard: http://localhost:{args.port}/dashboard.html")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*52}\n")

    if not args.no_browser:
        time.sleep(0.5)
        webbrowser.open(local_url)

    try:
        while True:
            time.sleep(1)
            # refresh ngrok URL if public
            if args.public and ngrok_proc:
                url = get_share_url(args.port)
                if url:
                    pass  # still running
    except KeyboardInterrupt:
        print("\nStopped.")
        if ngrok_proc:
            ngrok_proc.terminate()


if __name__ == '__main__':
    main()