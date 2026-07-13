"""
AcousticML Studio - Startup Script
This script starts the FastAPI backend server using uvicorn and opens the web application
automatically in your default browser.
"""

import os
import sys
import subprocess
import time
import webbrowser
import threading

def open_browser():
    # Wait for the server to spin up
    time.sleep(1.5)
    url = "http://127.0.0.1:8000"
    print(f"\n[AcousticML Studio] Launching workspace in browser: {url}")
    webbrowser.open(url)

def main():
    print("[AcousticML Studio] Starting backend server...")
    
    # Run browser launch in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Launch uvicorn server in the main thread
    # Use python -m uvicorn to ensure it runs within the active environment
    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "info"],
            check=True
        )
    except KeyboardInterrupt:
        print("\n[AcousticML Studio] Server stopped by user.")
    except Exception as e:
        print(f"\n[AcousticML Studio] Error starting server: {e}")
        print("Please check if uvicorn is installed in your python environment (pip install uvicorn).")

if __name__ == "__main__":
    main()
