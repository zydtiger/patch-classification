import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNet50Classifier(nn.Module):
    """
    ResNet50-based classifier for histopathology image classification.

    Supports both feature extraction (freeze backbone) and fine-tuning modes.
    """

    def __init__(self, num_classes=9, feature_extract=True):
        """
        Initialize ResNet50 classifier.

        Args:
            num_classes (int): Number of output classes (default: 9 for tissue types)
            feature_extract (bool): If True, freeze backbone layers for feature extraction
        """
        super(ResNet50Classifier, self).__init__()

        self.num_classes = num_classes
        self.feature_extract = feature_extract

        # Load pretrained ResNet50 with best available ImageNet weights
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze layers if using as feature extractor
        if self.feature_extract:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get number of features from the original final layer
        num_features = self.backbone.fc.in_features

        # Replace final classification layer with custom classifier
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: Class logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)

    def get_layer_params(self):
        """
        Get parameters grouped by layer for differential learning rates.

        Returns:
            dict: Dictionary containing parameter groups
        """
        if self.feature_extract:
            # Only classifier parameters are trainable
            return {"classifier": list(self.backbone.fc.parameters())}
        else:
            # Separate backbone and classifier parameters
            backbone_params = []
            classifier_params = list(self.backbone.fc.parameters())

            # Exclude classifier from backbone parameters
            for name, param in self.backbone.named_parameters():
                if not name.startswith("fc"):
                    backbone_params.append(param)

            return {"backbone": backbone_params, "classifier": classifier_params}
