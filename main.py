"""
Realm of Ash - static file server for Replit.

Serves index.html (and any sibling assets) over HTTP so the browser can
load the ES-module Three.js import. Bind to 0.0.0.0 so Replit's webview
proxy can reach it. Defaults to port 3000; honours $PORT if Replit sets one.

Run locally with:
    python3 main.py
"""
import http.server
import os
import socketserver
import sys

PORT = int(os.environ.get("PORT", 3000))
HOST = "0.0.0.0"


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # No-store so edits show up immediately on refresh during development.
        self.send_header("Cache-Control", "no-store")
        # Allow the page to use Pointer Lock / fullscreen inside Replit's iframe.
        self.send_header("Permissions-Policy", "pointer-lock=(self), fullscreen=(self)")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stdout.write(f"  {self.address_string()}  {fmt % args}\n")
        sys.stdout.flush()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
    # Allow quick restarts without 60s TIME_WAIT.
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((HOST, PORT), Handler) as httpd:
        print(f"Realm of Ash serving at http://{HOST}:{PORT}")
        print("On Replit: open the Webview tab to play.")
        print("Pointer Lock may need 'Open in new tab' if it is blocked inside the iframe.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nshutting down.")


if __name__ == "__main__":
    main()
