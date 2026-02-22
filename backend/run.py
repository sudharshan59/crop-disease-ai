"""
Entry point for the Crop Disease AI backend server.

Usage:
    cd backend
    python run.py

Loads CNN model, loads quantized LLM, and starts the FastAPI server.
"""

import os
import signal
import socket
import subprocess
import sys

import uvicorn
from app.config import settings


def _kill_port(port: int) -> None:
    """Kill any process currently listening on *port* (Windows & Linux)."""
    try:
        if sys.platform == "win32":
            out = subprocess.check_output(
                f"netstat -ano | findstr :{port} | findstr LISTENING",
                shell=True, text=True, stderr=subprocess.DEVNULL,
            )
            pids = {line.strip().split()[-1] for line in out.splitlines() if line.strip()}
            for pid in pids:
                if pid.isdigit() and int(pid) != os.getpid():
                    print(f"  [auto-fix] Killing PID {pid} on port {port}")
                    subprocess.call(f"taskkill /F /PID {pid}", shell=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            out = subprocess.check_output(
                f"lsof -ti :{port}", shell=True, text=True, stderr=subprocess.DEVNULL,
            )
            for pid in out.split():
                if pid.isdigit() and int(pid) != os.getpid():
                    print(f"  [auto-fix] Killing PID {pid} on port {port}")
                    os.kill(int(pid), signal.SIGKILL)
    except (subprocess.CalledProcessError, Exception):
        pass  # nothing listening — good


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    """Start the FastAPI application with uvicorn."""
    # ── Auto-free port if already in use ──────────────────────────────
    port = settings.API_PORT
    if _port_in_use(port):
        print(f"\n  ⚠  Port {port} is in use — auto-freeing...")
        _kill_port(port)
        import time; time.sleep(1)
        if _port_in_use(port):
            print(f"  ✗  Could not free port {port}. Stop the other process manually.")
            sys.exit(1)
        print(f"  ✓  Port {port} is now free.\n")

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Crop Disease Detection & Treatment Recommendation     ║")
    print("║   AI-Powered Agricultural Health System                  ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║   Host:   {settings.API_HOST:<45} ║")
    print(f"║   Port:   {settings.API_PORT:<45} ║")
    print(f"║   Device: {settings.get_device():<45} ║")
    print(f"║   LLM:    {settings.LLM_BACKEND:<45} ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
