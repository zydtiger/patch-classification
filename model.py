import torch
import torch.nn as nn


class PatchMIL(nn.Module):
    def __init__(self, num_patches, input_dim, num_classes):
        super(PatchMIL, self).__init__()

        self.num_patches = num_patches

        self.encoder = nn.Sequential(
            # First block
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            # Second block
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
        )

        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.classifier = nn.Sequential(
            # First block
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            # Second block
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            # Output layer
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # Encode patches
        # [B, num_patches, input_dim] -> [B*num_patches, input_dim]
        x = x.view(-1, x.size(-1))
        # [B*num_patches, input_dim] -> [B*num_patches, 256]
        x = self.encoder(x)
        # [B*num_patches, 256] -> [B, num_patches, 256]
        x = x.view(-1, self.num_patches, 256)

        # Calculate attention weights
        # [B, num_patches, 256] -> [B, num_patches, 1]
        attn_weights = self.attention(x)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Apply attention weights and sum across patches
        # [B, num_patches, 256] * [B, num_patches, 1] -> [B, num_patches, 256] -> [B, 256]
        weighted_encoding = torch.sum(x * attn_weights, dim=1)

        # Classify the weighted sum
        # [B, 256] -> [B, num_classes]
        output = self.classifier(weighted_encoding)

        return output


class MulticlassMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MulticlassMLP, self).__init__()

        self.network = nn.Sequential(
            # First block - reduce dimensionality significantly
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Second block - further feature extraction
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            # Third block - final feature refinement
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Output layer
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)
