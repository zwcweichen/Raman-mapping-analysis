import subprocess
import sys

def check_installation(package_list):
    missing_packages = []
    for package in package_list:
        try:
            __import__(package)
            print(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("未安裝模組正在進行安裝...")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])