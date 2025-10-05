#!/usr/bin/env python3
"""
Exo-Operator Web Interface Startup Script
=========================================

This script starts the Exo-Operator web interface with proper configuration.
"""

import os
import sys
import subprocess
import signal
import atexit
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import flask
        import pandas
        import numpy
        print("✓ Core dependencies found")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

def check_exso_sdk():
    """Check if Exo-SDK is available."""
    try:
        # Add parent directory to path (relative to web directory)
        web_dir = Path.cwd()
        if web_dir.name == "web":
            parent_dir = web_dir.parent
        else:
            # If not in web directory, try to find it
            parent_dir = Path(__file__).parent.parent
        
        sys.path.insert(0, str(parent_dir))
        
        import exso_sdk
        print("✓ Exo-SDK found")
        return True
    except ImportError:
        print("✗ Exo-SDK not found. Please install it first.")
        print("  Run: uv add -e ..")
        print("  Or: pip install -e ..")
        return False

def cleanup_port(port=2429, force=False, exclude_pids=None):
    """Clean up any processes using the specified port."""
    try:
        import subprocess
        import socket
        
        # Try to bind to the port to check if it's available
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('localhost', port))
            sock.close()
            print(f"✓ Port {port} is available")
            return
        except OSError:
            # Port is in use, try to find and kill the process
            pass
        finally:
            sock.close()
        
        # Cross-platform approach to find processes using the port
        current_pid = os.getpid()
        exclude_pids = exclude_pids or []
        
        # Get parent process ID to avoid killing parent processes
        try:
            parent_pid = os.getppid()
        except:
            parent_pid = None
        
        # Try different methods based on platform
        pids_to_kill = []
        
        # Method 1: Try lsof if available (Unix-like systems)
        try:
            result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                pids_to_kill = result.stdout.strip().split('\n')
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            # lsof not available or timed out, try alternative methods
            pass
        
        # Method 2: Try netstat (if lsof failed)
        if not pids_to_kill:
            try:
                if os.name == 'nt':  # Windows
                    result = subprocess.run(['netstat', '-ano'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if f':{port}' in line and 'LISTENING' in line:
                                parts = line.split()
                                if len(parts) > 4:
                                    pids_to_kill.append(parts[-1])
                else:  # Unix-like systems
                    result = subprocess.run(['netstat', '-tulpn'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if f':{port}' in line and 'LISTEN' in line:
                                parts = line.split()
                                if len(parts) > 6:
                                    pid_part = parts[-1].split('/')[0]
                                    if pid_part.isdigit():
                                        pids_to_kill.append(pid_part)
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass
        
        # Method 3: Try ss (if netstat failed)
        if not pids_to_kill:
            try:
                result = subprocess.run(['ss', '-tulpn'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if f':{port}' in line and 'LISTEN' in line:
                            # Extract PID from the output
                            import re
                            pid_match = re.search(r'pid=(\d+)', line)
                            if pid_match:
                                pids_to_kill.append(pid_match.group(1))
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass
        
        # Kill the processes if found
        if pids_to_kill:
            for pid in pids_to_kill:
                if pid and pid.strip():
                    try:
                        pid_int = int(pid.strip())
                        # Don't kill our own process, parent process, or excluded PIDs
                        if (pid_int != current_pid and 
                            pid_int != parent_pid and 
                            pid_int not in exclude_pids):
                            if force:
                                print(f"Cleaning up process {pid} on port {port}")
                                subprocess.run(['kill', '-9', str(pid_int)], check=False)
                            else:
                                # For non-force cleanup, try graceful termination first
                                print(f"Attempting graceful termination of process {pid} on port {port}")
                                subprocess.run(['kill', '-TERM', str(pid_int)], check=False)
                                time.sleep(0.5)
                                # Check if process is still running
                                try:
                                    subprocess.run(['kill', '-0', str(pid_int)], check=True)
                                    # Process still running, force kill
                                    print(f"Force killing process {pid}")
                                    subprocess.run(['kill', '-9', str(pid_int)], check=False)
                                except subprocess.CalledProcessError:
                                    # Process already terminated
                                    pass
                    except (ValueError, OSError):
                        # Invalid PID or process already gone
                        continue
            print(f"✓ Port {port} cleaned up")
        else:
            print(f"✓ Port {port} appears to be available (no processes found)")
            
    except Exception as e:
        print(f"Warning: Could not clean up port {port}: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print(f"\n\n🛑 Received signal {signum}, shutting down gracefully...")
    cleanup_port(2429, force=True)
    print("👋 Exo-Operator Web Interface stopped")
    sys.exit(0)

def install_requirements():
    """Install web interface requirements."""
    # Use current working directory (should be web directory)
    requirements_file = Path.cwd() / "requirements.txt"
    if requirements_file.exists():
        print("Installing web interface requirements...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            print("✓ Requirements installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install requirements")
            return False
    return True

def main():
    """Main startup function."""
    print("Exo-Operator Web Interface")
    print("=" * 30)
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_port, 2429)
    
    # Ensure we're in the web directory
    script_dir = Path(__file__).parent
    if script_dir.name == "web":
        # We're already in the web directory
        os.chdir(script_dir)
        print("✓ Running from web directory")
    else:
        # Try to find the web directory
        web_dir = script_dir / "web"
        if web_dir.exists() and web_dir.is_dir():
            os.chdir(web_dir)
            print(f"✓ Changed to web directory: {web_dir}")
        else:
            print("✗ Could not find web directory")
            print(f"  Script location: {script_dir}")
            print(f"  Expected web directory: {web_dir}")
            sys.exit(1)
    
    # Only clean up port if we're not in a restart situation
    # Check if this is a fresh start by looking for existing processes
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('localhost', 2429))
            sock.close()
            print("✓ Port 2429 is available")
        except OSError:
            print("🧹 Found existing processes on port 2429, cleaning up...")
            cleanup_port(2429, force=False)
            time.sleep(1)  # Give the port time to be released
        finally:
            sock.close()
    except Exception as e:
        print(f"Warning: Could not check port status: {e}")
    
    # Install requirements if needed
    if not check_dependencies():
        print("\nInstalling missing dependencies...")
        if not install_requirements():
            print("✗ Failed to install dependencies")
            sys.exit(1)
    
    # Check Exo-SDK
    if not check_exso_sdk():
        print("\nPlease install the Exo-SDK first:")
        print("  cd ..")
        print("  pip install -e .")
        sys.exit(1)
    
    # Start the application
    print("\n🚀 Starting Exo-Operator Web Interface...")
    print("   Open your browser and go to: http://localhost:2429")
    print("   Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Ensure we're in the web directory for Flask to work properly
        web_dir = Path.cwd()
        if web_dir.name != "web":
            web_dir = Path(__file__).parent
            os.chdir(web_dir)
            print(f"✓ Changed to web directory: {web_dir}")
        
        # Add the web directory to Python path
        sys.path.insert(0, str(web_dir))
        
        from app import app
        
        # Determine if we should use debug mode
        # Only enable debug mode when running directly from web directory
        # and not when imported by another script
        use_debug = (Path(__file__).parent.name == "web" and 
                    os.path.basename(sys.argv[0]) == "run.py")
        
        if use_debug:
            print("✓ Debug mode enabled")
        else:
            print("✓ Debug mode disabled (imported or running from outside web directory)")
        
        app.run(debug=use_debug, host='0.0.0.0', port=2429)
    except KeyboardInterrupt:
        print("\n\n👋 Exo-Operator Web Interface stopped")
        cleanup_port(2429, force=True)
    except Exception as e:
        print(f"\n✗ Error starting application: {e}")
        cleanup_port(2429, force=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
