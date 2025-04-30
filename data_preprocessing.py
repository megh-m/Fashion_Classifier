# data_preprocessing.py
from PIL import Image
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_fashion_mnist(batch_size=64):
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

def preprocess_user_image(image_path):
    """Preprocess user-uploaded images for model prediction"""
    # Open image and convert to grayscale
    image = Image.open(image_path).convert('L')
    
    # Resize to 28x28 (same as Fashion-MNIST)
    image = image.resize((28, 28))
    
    # Apply same transforms as training data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Transform and add batch dimension
    tensor = transform(image).unsqueeze(0)
    
    return tensor

def save_user_image(image_path, destination_dir, class_name, image_id=None):
    """Save user images to appropriate directory for future retraining"""
    os.makedirs(destination_dir, exist_ok=True)
    
    # Generate unique ID if not provided
    if image_id is None:
        import uuid
        image_id = str(uuid.uuid4())
    
    # Extract file extension
    _, ext = os.path.splitext(image_path)
    
    # Create destination path
    destination = os.path.join(destination_dir, f"{class_name}_{image_id}{ext}")
    
    # Copy the image
    import shutil
    shutil.copy2(image_path, destination)
    
    return destination
