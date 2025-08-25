#!/usr/bin/env python3
import http.server
import socketserver
import webbrowser
import threading
import time
import os

PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.getcwd(), **kwargs)

def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Server running at http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    # Start server in background
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    time.sleep(1)  # Wait for server to start
    
    print("\nAvailable visualizations:")
    print(f"1. Animated Flow Particles: http://localhost:{PORT}/animated_flow_visualization.html")
    print(f"2. OD Matrix Visualization: http://localhost:{PORT}/od_matrix_visualization.html")
    print(f"3. Original Visualization: http://localhost:{PORT}/visualize_flows.html")
    
    # Open the animated visualization
    webbrowser.open(f"http://localhost:{PORT}/animated_flow_visualization.html")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")