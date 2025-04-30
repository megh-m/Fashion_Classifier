# data_setup.py
import os
import subprocess

def setup_dvc():
    # Initialize DVC
    subprocess.run(["dvc", "init"])
    
    # Configure DVC remote storage (e.g., local)
    subprocess.run(["dvc", "remote", "add", "-d", "myremote", "/path/to/remote"])
    
    # Add initial dataset to DVC
    subprocess.run(["dvc", "add", "data/fashion-mnist"])
    
    # Commit the DVC files to git
    subprocess.run(["git", "add", ".dvc", ".dvcignore", "data/fashion-mnist.dvc"])
    subprocess.run(["git", "commit", "-m", "Initialize DVC with Fashion-MNIST dataset"])

if __name__ == "__main__":
    # Create directories
    os.makedirs("data/fashion-mnist", exist_ok=True)
    os.makedirs("data/user-images", exist_ok=True)
    
    setup_dvc()
