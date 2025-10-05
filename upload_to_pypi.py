#!/usr/bin/env python3
"""
Script to upload exso-sdk to PyPI
"""
#pypi-AgENdGVzdC5weXBpLm9yZwIkNjc5YzkxOWUtNWMxNy00MjFlLTg2MDAtYjcwNTJiOTBhZDAxAAIqWzMsIjNmOWRhZjQ1LTE2MzQtNDI3OC1iNTJjLTc1Zjg1NDNjYmY1YSJdAAAGID4rGqvjPdyUx50y9xOMZjt0DQ9nKco9GVp2ljSpiuYT
import subprocess
import sys
import os

def check_requirements():
    """Check if required tools are available."""
    print("🔍 Checking requirements...")
    
    # Check if twine is installed
    try:
        subprocess.run([sys.executable, "-c", "import twine"], check=True, capture_output=True)
        print("✓ Twine is available")
    except subprocess.CalledProcessError:
        print("❌ Twine not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "twine"], check=True)
    
    # Check if build is available
    try:
        subprocess.run([sys.executable, "-c", "import build"], check=True, capture_output=True)
        print("✓ Build is available")
    except subprocess.CalledProcessError:
        print("❌ Build not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "build"], check=True)

def build_package():
    """Build the package."""
    print("\n🔨 Building package...")
    
    # Clean previous builds
    if os.path.exists("dist"):
        import shutil
        shutil.rmtree("dist")
    
    # Build the package
    subprocess.run([sys.executable, "-m", "build"], check=True)
    print("✓ Package built successfully")

def check_package():
    """Check the built package."""
    print("\n🧪 Checking package...")
    
    # Check the wheel
    subprocess.run([sys.executable, "-m", "twine", "check", "dist/*"], check=True)
    print("✓ Package check passed")

def upload_to_testpypi():
    """Upload to TestPyPI first."""
    print("\n🚀 Uploading to TestPyPI...")
    
    # Upload to TestPyPI
    subprocess.run([
        sys.executable, "-m", "twine", "upload", 
        "--repository", "testpypi",
        "dist/*"
    ], check=True)
    
    print("✓ Uploaded to TestPyPI successfully!")
    print("\n📝 TestPyPI installation command:")
    print("pip install --index-url https://test.pypi.org/simple/ exso-sdk")

def upload_to_pypi():
    """Upload to PyPI."""
    print("\n🚀 Uploading to PyPI...")
    
    # Upload to PyPI
    subprocess.run([
        sys.executable, "-m", "twine", "upload",
        "dist/*"
    ], check=True)
    
    print("✓ Uploaded to PyPI successfully!")
    print("\n📝 PyPI installation command:")
    print("pip install exso-sdk")

def main():
    """Main function."""
    print("🚀 Exo-SDK PyPI Upload Script")
    print("=" * 50)
    
    try:
        # Check requirements
        check_requirements()
        
        # Build package
        build_package()
        
        # Check package
        check_package()
        
        # Ask user for upload destination
        print("\n📤 Upload Options:")
        print("1. TestPyPI (recommended for testing)")
        print("2. PyPI (production)")
        print("3. Both")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            upload_to_testpypi()
        elif choice == "2":
            upload_to_pypi()
        elif choice == "3":
            upload_to_testpypi()
            print("\n" + "="*50)
            input("Press Enter to continue to PyPI upload...")
            upload_to_pypi()
        else:
            print("❌ Invalid choice. Exiting.")
            return
        
        print("\n🎉 Upload completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during upload: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Upload cancelled by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()
