#!/usr/bin/env python3
import http.server
import socketserver
import webbrowser
import threading
import time

PORT = 8001  # Different port to avoid conflicts

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def start_server(port=PORT):
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"🚀 Enhanced Flow Visualization Server running at http://localhost:{port}")
            print(f"📊 Enhanced Dashboard: http://localhost:{port}/enhanced_flow_viz.html")
            print(f"📈 Basic Dashboard: http://localhost:{port}/improved_flow_visualization.html")
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {port} is already in use. Trying port {port+1}...")
            start_server(port + 1)
        else:
            raise

if __name__ == "__main__":
    # Start server in background
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait a moment then open browser
    time.sleep(1)
    webbrowser.open(f'http://localhost:8001/enhanced_flow_viz.html')
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")