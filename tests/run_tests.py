#!/usr/bin/env python3
"""
Test Runner for Exo-SDK Tests
=============================

This script provides an easy way to run all SDK tests.
"""

import sys
import os
import subprocess
import time

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} failed!")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Exo-SDK Test Runner")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("test_basic.py"):
        print("❌ Please run this script from the tests/ directory")
        print("💡 cd tests/ && python run_tests.py")
        sys.exit(1)
    
    # Check if exso-sdk is installed
    try:
        import exso_sdk
        print(f"✅ Found exso-sdk package version: {getattr(exso_sdk, '__version__', 'unknown')}")
    except ImportError:
        print("❌ exso-sdk package not found!")
        print("💡 Install it with:")
        print("   pip install -i https://test.pypi.org/simple/ exso-sdk")
        print("   or")
        print("   pip install exso-sdk")
        sys.exit(1)
    
    # Run tests
    tests = [
        ("python3 test_basic.py", "Basic SDK Functionality Test"),
        ("python3 model_import.py", "Model Import Test"),
        ("python3 demo_pypi_package.py", "Demo Package Test"),
        ("python3 test_complete_sdk.py", "Complete SDK Test Suite"),
        ("python3 test_accuracy_verification.py", "Accuracy Verification Test")
    ]
    
    results = []
    for cmd, description in tests:
        success = run_command(cmd, description)
        results.append((description, success))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {description}")
    
    print(f"\n🎯 Overall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All tests passed! Your SDK is working perfectly!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
