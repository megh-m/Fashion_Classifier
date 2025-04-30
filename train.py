# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import datetime
import json
import subprocess
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import FashionCNN
from prometheus_client import Counter, Gauge

# Define metrics for monitoring
training_metrics = {
    "accuracy": Gauge('fashion_model_training_accuracy', 'Model accuracy during training'),
    "loss": Gauge('fashion_model_training_loss', 'Training loss value'),
    "epoch_counter": Counter('fashion_model_training_epochs_total', 'Total number of training epochs completed')
}

def load_fashion_mnist(batch_size=64):
    """
    Load the Fashion-MNIST dataset for training and testing
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load training data
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Load test data
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(epochs=10, batch_size=64, learning_rate=0.001):
    """
    Train the FashionCNN model on the Fashion-MNIST dataset
    """
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Load data
    train_loader, test_loader = load_fashion_mnist(batch_size)
    
    # Initialize model
    model = FashionCNN().to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            
            if i % 100 == 99:
                avg_loss = running_loss / 100
                print(f'Epoch {epoch+1}, Batch {i+1}: loss {avg_loss:.4f}')
                training_metrics["loss"].set(avg_loss)
                running_loss = 0.0
        
        # Update epoch counter
        training_metrics["epoch_counter"].inc()
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        accuracy = 100 * correct / total
        avg_test_loss = test_loss / len(test_loader)
        
        # Update metrics
        training_metrics["accuracy"].set(accuracy)
        
        print(f'Epoch {epoch+1} - Accuracy: {accuracy:.2f}%, Test Loss: {avg_test_loss:.4f}')
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/fashion_model.pth")
    
    # Save model metadata
    model_metadata = {
        "accuracy": accuracy,
        "parameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        },
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    with open("models/model_metadata.json", "w") as f:
        json.dump(model_metadata, f)
    
    # Add model to DVC
    subprocess.run(["dvc", "add", "models/fashion_model.pth"])
    subprocess.run(["dvc", "add", "models/model_metadata.json"])
    
    print(f"Training completed with final accuracy: {accuracy:.2f}%")
    print("Model saved to models/fashion_model.pth")
    
    return model, accuracy

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a CNN model on Fashion-MNIST")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    model, accuracy = train_model(
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        learning_rate=args.learning_rate
    )
