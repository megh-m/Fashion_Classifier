# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionCNN(nn.Module):
    def __init__(self):
        """
        Initialize the CNN architecture for Fashion-MNIST classification.
        
        Architecture:
        - 2 Convolutional layers with max pooling
        - Dropout for regularization
        - 2 Fully connected layers
        """
        super(FashionCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # Input size: 64 feature maps * 7 * 7 (image size after conv and pooling)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes for Fashion-MNIST
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, 1, 28, 28]
            
        Returns:
            x: Output tensor with logits for 10 classes
        """
        # First block: Conv -> BatchNorm -> ReLU -> MaxPool
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block: Conv -> BatchNorm -> ReLU -> MaxPool
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # First FC layer with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Output layer
        x = self.fc2(x)
        
        # Apply log softmax for numerical stability
        return F.log_softmax(x, dim=1)

# For testing the model architecture
if __name__ == "__main__":
    # Create a random tensor with the shape of a Fashion-MNIST image
    x = torch.randn(1, 1, 28, 28)
    
    # Initialize the model
    model = FashionCNN()
    
    # Print model summary
    print(model)
    
    # Forward pass
    output = model(x)
    
    # Print output shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected number of classes: 10")
