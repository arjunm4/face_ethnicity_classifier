import os
import sys
import platform
import subprocess

def system_check():
    print("SYSTEM INFORMATION")
    print("=" * 50)
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Python Executable: {sys.executable}")
    
    # Check if we're in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Running in virtual environment")
        print(f"Virtual Environment Path: {sys.prefix}")
    else:
        print("⚠ Not running in virtual environment")
    
    print(f"\nCurrent Working Directory: {os.getcwd()}")
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage('.')
    print(f"Available Disk Space: {free // (1024**3)} GB")
    
    # Check memory (if psutil is available)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Available RAM: {memory.available // (1024**3)} GB / {memory.total // (1024**3)} GB")
    except ImportError:
        print("Memory info not available (psutil not installed)")
    
    print("\nPROJECT STRUCTURE CHECK")
    print("=" * 50)
    required_dirs = [
        'data', 'models', 'utils', 'notebooks', 'scripts', 'results',
        'data/train', 'data/val', 'data/test'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (missing)")
    
    print(f"\nTotal files in project: {sum(len(files) for _, _, files in os.walk('.'))}")
    print(f"Total directories in project: {sum(len(dirs) for _, dirs, _ in os.walk('.'))}")

if __name__ == "__main__":
    system_check()
