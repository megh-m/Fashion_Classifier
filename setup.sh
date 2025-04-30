#!/bin/bash
# setup.sh

# Create necessary directories
mkdir -p data/fashion-mnist
mkdir -p data/user-images/correct
mkdir -p data/user-images/incorrect
mkdir -p models

# Initialize git repository if not exists
if [ ! -d .git ]; then
    git init
    echo "models/" > .gitignore
    echo "data/fashion-mnist/" >> .gitignore
    git add .gitignore
    git commit -m "Initial commit"
fi

# Initialize DVC
dvc init
dvc remote add -d myremote /home/megh_m/app-remo

# Download the Fashion-MNIST dataset if not present
pip install --no-cache-dir -r requirements.txt
python3 -c "from torchvision.datasets import FashionMNIST; FashionMNIST('./data', download=True)"

# Train initial model - THIS IS THE REQUIRED ADDITION
echo "Training initial model..."
python3 train.py

# Add model to DVC tracking
dvc add models/fashion_model.pth
dvc add models/model_metadata.json

# Make retraining script executable
chmod +x periodic_retrain.sh

echo "Setup completed successfully. Model trained and ready for containerization."
