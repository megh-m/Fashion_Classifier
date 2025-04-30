# retrain.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import glob
import datetime
import shutil
import subprocess
import json
import requests
from model import FashionCNN
from prometheus_client import Gauge, Summary
import time
from filelock import FileLock
import logging

# Define metrics for monitoring
retraining_metrics = {
    "accuracy": Gauge('fashion_model_retraining_accuracy', 'Model accuracy during retraining'),
    "loss": Gauge('fashion_model_retraining_loss', 'Retraining loss value'),
    "user_images_count": Gauge('fashion_model_user_images_total', 'Total user images used')
}

class UserImageDataset(Dataset):
    """Custom dataset for user-submitted images with proper filename handling"""
    def __init__(self, image_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        
        # Use sanitized class names matching filename patterns
        self.class_map = {
            "T-shirt-top": 0, "Trouser": 1, "Pullover": 2, 
            "Dress": 3, "Coat": 4, "Sandal": 5, 
            "Shirt": 6, "Sneaker": 7, "Bag": 8, "Ankle boot": 9
        }

        self._load_images(image_dir, "correct")
        self._load_images(image_dir, "incorrect")
        
        print(f"Dataset initialized with {len(self.images)} valid images")

    def _load_images(self, base_dir, category):
        dir_path = os.path.join(base_dir, category)
        if not os.path.exists(dir_path):
            return
            
        for img_path in glob.glob(os.path.join(dir_path, "*")):
            try:
                filename = os.path.basename(img_path)
                class_part = filename.split("_")[0]
                
                if class_part in self.class_map:
                    self.images.append(img_path)
                    self.labels.append(self.class_map[class_part])
                else:
                    print(f"Ignoring invalid class: {class_part} in {filename}")
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img_path = self.images[idx]
            image = Image.open(img_path).convert('L').resize((28, 28))
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return torch.zeros((1, 28, 28)), 0

# Add metrics
RETRAIN_TIME = Summary('retraining_duration_seconds', 'Time spent retraining')
# Modify function signature
@RETRAIN_TIME.time()
def retrain_model(min_images=10, epochs=5, batch_size=32, learning_rate=0.0005):
    """Retrain model with layer freezing and user images"""
    
    print("\n=== Starting Retraining Process ===")
    
    # Check user images
    user_images = glob.glob("data/user-images/**/*.*", recursive=True)
    retraining_metrics["user_images_count"].set(len(user_images))
    
    if len(user_images) < min_images:
        print(f"Aborting: Found {len(user_images)} images (needs {min_images})")
        return False

    # Setup device and transformations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load data
    try:
        test_loader = load_fashion_mnist(batch_size)
        user_dataset = UserImageDataset("data/user-images", transform)
        user_loader = DataLoader(user_dataset, batch_size=batch_size, shuffle=True)
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        return False

    # Model setup with layer freezing
    model = FashionCNN().to(device)
    try:
        model.load_state_dict(torch.load("models/fashion_model.pth", map_location=device))
        print("Loaded existing model for fine-tuning")
        
        # Freeze all layers except last two FC layers
        for name, param in model.named_parameters():
            if 'fc' not in name:  # Freeze conv and pooling layers
                param.requires_grad = False
                print(f"Frozen layer: {name}")
            else:
                param.requires_grad = True
                print(f"Trainable layer: {name}")
                
    except Exception as e:
        print(f"Model loading error: {str(e)}")
        return False

    # Optimizer with filtered parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_accuracy = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, labels in user_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(user_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        retraining_metrics["loss"].set(avg_loss)
        retraining_metrics["accuracy"].set(accuracy)

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "models/fashion_model.pth")
            print(f"New best model saved with accuracy {accuracy:.2f}%")

    # Final save and DVC tracking
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/fashion_model_retrained_{timestamp}.pth"
    #torch.save(model.state_dict(), model_path)
    model_lock = FileLock("models/fashion_model.pth.lock")
    with model_lock:
        # Save to temporary file first
        temp_path = "models/fashion_model.temp.pth"
        torch.save(model.state_dict(), temp_path)
        
        # Atomic replacement
        os.replace(temp_path, "models/fashion_model.pth")
        # Update metadata
        metadata = {
            "accuracy": best_accuracy,
            "parameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "trainable_layers": [name for name, _ in model.named_parameters() if _.requires_grad]
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with open("models/model_metadata.json", "w") as f:
            json.dump(metadata, f)
        print("Model updated atomically")

    # Signal app.py to reload (requires adding endpoint)
    try:
        response = requests.post("http://host.docker.internal:5001/reload-model", data = {"api_key":"your-secret-key"})
        response.raise_for_status()
    except Exception as e:
        print(f"Model reload signal failed: {str(e)}")
    
    
    # DVC tracking
    subprocess.run(["dvc", "add", "models/fashion_model.pth", "models/model_metadata.json", model_path])
    print("=== Retraining Complete ===")
    return True

def load_fashion_mnist(batch_size=64):
    """Load Fashion-MNIST test set for validation"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_set = datasets.FashionMNIST(
        root='./data', 
        train=False,
        download=True,
        transform=transform
    )
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retrain fashion classifier")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--min-images", type=int, default=10)
    
    args = parser.parse_args()
    
    if retrain_model(
        min_images=args.min_images,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    ):
        print("Retraining successful")
    else:
        print("Retraining failed")
