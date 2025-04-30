# dvc_helper.py
import subprocess
import os
import json

def init_dvc():
    """Initialize DVC if not already initialized"""
    if not os.path.exists('.dvc'):
        subprocess.run(["dvc", "init"])
        subprocess.run(["dvc", "remote", "add", "-d", "myremote", "/home/megh-m/app-remo"])
        print("DVC initialized successfully")
    else:
        print("DVC already initialized")

def track_data(data_path):
    """Add data to DVC tracking"""
    try:
        subprocess.run(["dvc", "add", data_path], check=True)
        print(f"Added {data_path} to DVC tracking")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error adding {data_path} to DVC: {str(e)}")
        return False

def push_data():
    """Push tracked data to remote storage"""
    try:
        subprocess.run(["dvc", "push"], check=True)
        print("Successfully pushed data to remote storage")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error pushing data: {str(e)}")
        return False

def pull_data():
    """Pull data from remote storage"""
    try:
        subprocess.run(["dvc", "pull"], check=True)
        print("Successfully pulled data from remote storage")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error pulling data: {str(e)}")
        return False
