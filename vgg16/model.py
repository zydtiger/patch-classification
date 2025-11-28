import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights


class VGG16Classifier(nn.Module):
    """
    VGG16 classifier with custom head for histopathology patch classification.

    Features:
    - Pretrained VGG16 backbone
    - Custom classifier head: 25088 -> 4096 -> 1024 -> num_classes
    - Feature extraction mode support
    - Higher dropout (0.5) to combat overfitting
    - Differential learning rate support
    """

    def __init__(self, num_classes=9, feature_extract=True):
        """
        Initialize VGG16 classifier

        Args:
            num_classes (int): Number of output classes (default: 9 for NCT-CRC-HE-100K)
            feature_extract (bool): If True, freeze backbone features and only train classifier
        """
        super(VGG16Classifier, self).__init__()

        # Store feature extraction mode
        self.feature_extract = feature_extract

        # Load pretrained VGG16
        self.backbone = models.vgg16(weights=VGG16_Weights.DEFAULT)

        # Freeze features if using as feature extractor
        if self.feature_extract:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Get input size for classifier (VGG16: 512 * 7 * 7 = 25088)
        num_features = 25088

        # Replace classifier head with custom architecture
        # Higher dropout rates to combat VGG16's overfitting tendencies
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Higher dropout for VGG16
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Consistent dropout rate
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        """Forward pass through VGG16 backbone and custom classifier"""
        return self.backbone(x)

    def get_layer_params(self):
        """
        Get parameters grouped by layer for differential learning rates.

        Returns:
            dict: Dictionary containing parameter groups for features and classifier
        """
        if self.feature_extract:
            # Only classifier parameters are trainable in feature extraction mode
            return {"classifier": list(self.backbone.classifier.parameters())}
        else:
            # Both features and classifier are trainable in fine-tuning mode
            features_params = list(self.backbone.features.parameters())
            classifier_params = list(self.backbone.classifier.parameters())
            return {"features": features_params, "classifier": classifier_params}
