#!/usr/bin/env python3
"""
Exo-Operator Web Interface Launcher
===================================

This script launches the Exo-Operator web interface from any directory.
"""

import os
import sys
from pathlib import Path

def main():
    """Launch the web interface."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    web_dir = script_dir / "web"
    
    if not web_dir.exists():
        print("✗ Web directory not found")
        print(f"  Expected: {web_dir}")
        sys.exit(1)
    
    # Change to web directory
    os.chdir(web_dir)
    print(f"✓ Changed to web directory: {web_dir}")
    
    # Add web directory to Python path
    sys.path.insert(0, str(web_dir))
    
    # Import and run the web application
    try:
        from run import main as run_main
        run_main()
    except ImportError as e:
        print(f"✗ Failed to import web application: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error starting web application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
