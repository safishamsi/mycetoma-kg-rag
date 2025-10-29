"""
InceptionV3-based classifier for mycetoma histopathology images.

This module implements the vision component of the KG-RAG system,
providing both classification and feature extraction capabilities.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple


class InceptionV3Classifier(nn.Module):
    """
    InceptionV3 model for binary mycetoma classification.
    
    Architecture:
        - Base: Pre-trained InceptionV3 on ImageNet
        - Modified: Final FC layer for binary classification
        - Feature extraction: 2048-dimensional features before classification
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        pretrained: Whether to use ImageNet pre-trained weights
        dropout: Dropout rate before final classification layer
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        super(InceptionV3Classifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pre-trained InceptionV3
        self.inception = models.inception_v3(pretrained=pretrained)
        
        # Get input features for FC layer
        in_features = self.inception.fc.in_features  # 2048
        
        # Replace final fully connected layer
        self.inception.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
        # For auxiliary classifier (used during training)
        if hasattr(self.inception, 'AuxLogits'):
            aux_in_features = self.inception.AuxLogits.fc.in_features
            self.inception.AuxLogits.fc = nn.Linear(aux_in_features, num_classes)
        
        # Feature extractor (everything except final FC)
        # We'll use this for Knowledge Graph similarity search
        self.feature_extractor = nn.Sequential(
            *list(self.inception.children())[:-1]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 299, 299)
        
        Returns:
            Logits of shape (batch_size, num_classes)
            
        Note:
            During training with InceptionV3, auxiliary outputs may be returned.
            Use `model.training` to check mode.
        """
        return self.inception(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 2048-dimensional feature vectors for Knowledge Graph retrieval.
        
        This is a key method for the KG-RAG system - these features are stored
        in Neo4j and used for visual similarity search.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 299, 299)
        
        Returns:
            Feature tensor of shape (batch_size, 2048)
        """
        self.eval()  # Always use eval mode for feature extraction
        with torch.no_grad():
            # Pass through all layers except final FC
            features = self.feature_extractor(x)
            
            # Global average pooling (if not already done)
            if features.dim() == 4:
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            
            # Flatten
            features = features.view(features.size(0), -1)
            
        return features
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 299, 299)
        
        Returns:
            Tuple of (predicted_classes, probabilities)
            - predicted_classes: Tensor of shape (batch_size,)
            - probabilities: Tensor of shape (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            
            # Handle InceptionOutputs during training
            if isinstance(logits, tuple):
                logits = logits[0]
            
            # Softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Get predicted class
            preds = torch.argmax(probs, dim=1)
            
        return preds, probs
    
    def freeze_base(self):
        """
        Freeze all layers except the final classification layer.
        Useful for fine-tuning.
        """
        # Freeze all parameters
        for param in self.inception.parameters():
            param.requires_grad = False
        
        # Unfreeze final FC layer
        for param in self.inception.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_num_params(self) -> dict:
        """
        Get number of parameters in the model.
        
        Returns:
            Dictionary with total, trainable, and frozen parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen
        }


def load_checkpoint(
    checkpoint_path: str,
    num_classes: int = 2,
    device: Optional[torch.device] = None
) -> InceptionV3Classifier:
    """
    Load a trained InceptionV3 checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        num_classes: Number of classes
        device: Device to load model on
    
    Returns:
        Loaded InceptionV3Classifier model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = InceptionV3Classifier(num_classes=num_classes, pretrained=False)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = InceptionV3Classifier(num_classes=2, pretrained=True)
    
    print("Model initialized successfully!")
    print(f"Parameters: {model.get_num_params()}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 299, 299)
    output = model(dummy_input)
    print(f"Output shape: {output.shape if not isinstance(output, tuple) else output[0].shape}")
    
    # Test feature extraction
    features = model.extract_features(dummy_input)
    print(f"Feature shape: {features.shape}")
    
    # Test prediction
    preds, probs = model.predict(dummy_input)
    print(f"Predictions shape: {preds.shape}")
    print(f"Probabilities shape: {probs.shape}")
