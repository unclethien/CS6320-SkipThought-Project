#!/usr/bin/env python
# Setup script to install transformers and verify OS module

import subprocess
import sys

def check_install_transformers():
    """Check if transformers is installed, install if not"""
    try:
        import transformers
        print(f"✓ Transformers library is already installed (version {transformers.__version__})")
    except ImportError:
        print("Installing transformers library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
        try:
            import transformers
            print(f"✓ Successfully installed transformers library (version {transformers.__version__})")
        except ImportError:
            print("❌ Failed to install transformers library")
            return False
    return True

def verify_os_module():
    """Verify that the OS module is available"""
    try:
        import os
        print(f"✓ OS module is available (part of Python standard library)")
        print(f"  Current working directory: {os.getcwd()}")
        print(f"  Python executable: {sys.executable}")
    except ImportError:
        print("❌ OS module not available (this should never happen with standard Python installations)")
        return False
    return True

def verify_environment():
    """Verify that transformers library and other requirements are available"""
    try:
        import torch
        print(f"✓ PyTorch is installed (version {torch.__version__})")
        print(f"  CUDA availability: {'Enabled' if torch.cuda.is_available() else 'Not available'}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch is not installed. This is required for the Skip-Thought-GAN project.")
        print("  Run 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'")
        return False
    return True

if __name__ == "__main__":
    print("Setting up environment for Skip-Thought-GAN project...")
    transformers_ok = check_install_transformers()
    os_ok = verify_os_module()
    env_ok = verify_environment()
    
    if transformers_ok and os_ok and env_ok:
        print("\n✅ Environment setup complete. You're ready to run Skip-Thought-GAN.py!")
    else:
        print("\n❌ Environment setup incomplete. Please fix the issues mentioned above.")