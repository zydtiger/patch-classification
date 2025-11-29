# ResNet50 Histopathology Classification

This directory implements a specialized ResNet50 training pipeline for histopathology patch classification on the NCT-CRC-HE-100K dataset. The pipeline leverages a pretrained ResNet50 backbone with a custom classifier head optimized for 9-class tissue classification (ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM). The training process supports both feature extraction mode (freeze backbone) and fine-tuning mode (update backbone and classifier) with differential learning rates for backbone and classifier layers. The data loading pipeline processes raw TIF images directly, applying ImageNet standard normalization and histopathology-specific augmentations including random rotation (15°), horizontal/vertical flipping, color jittering, and random erasing for improved robustness. Training is managed with an early stopping mechanism with patience-based validation loss monitoring, automatic checkpoint management, and support for class imbalance handling through balanced sampling and class weighting.

## Architecture

### Model Structure

Sample torchinfo summary input size is (B, 3, 224, 224)

===============================================================================================
Layer (type:depth-idx) Output Shape Param #
===============================================================================================
ResNet50Classifier [1, 9] --
├─ResNet: 1-1 [1, 9] --
│ └─Conv2d: 2-1 [1, 64, 112, 112] 9,408
│ └─BatchNorm2d: 2-2 [1, 64, 112, 112] 128
│ └─ReLU: 2-3 [1, 64, 112, 112] --
│ └─MaxPool2d: 2-4 [1, 64, 56, 56] --
│ └─Sequential: 2-5 [1, 256, 56, 56] --
│ │ └─Bottleneck: 3-1 [1, 256, 56, 56] 75,008
│ │ └─Bottleneck: 3-2 [1, 256, 56, 56] 70,400
│ │ └─Bottleneck: 3-3 [1, 256, 56, 56] 70,400
│ └─Sequential: 2-6 [1, 512, 28, 28] --
│ │ └─Bottleneck: 3-4 [1, 512, 28, 28] 379,392
│ │ └─Bottleneck: 3-5 [1, 512, 28, 28] 280,064
│ │ └─Bottleneck: 3-6 [1, 512, 28, 28] 280,064
│ │ └─Bottleneck: 3-7 [1, 512, 28, 28] 280,064
│ └─Sequential: 2-7 [1, 1024, 14, 14] --
│ │ └─Bottleneck: 3-8 [1, 1024, 14, 14] 1,512,448
│ │ └─Bottleneck: 3-9 [1, 1024, 14, 14] 1,117,184
│ │ └─Bottleneck: 3-10 [1, 1024, 14, 14] 1,117,184
│ │ └─Bottleneck: 3-11 [1, 1024, 14, 14] 1,117,184
│ │ └─Bottleneck: 3-12 [1, 1024, 14, 14] 1,117,184
│ │ └─Bottleneck: 3-13 [1, 1024, 14, 14] 1,117,184
│ └─Sequential: 2-8 [1, 2048, 7, 7] --
│ │ └─Bottleneck: 3-14 [1, 2048, 7, 7] 6,039,552
│ │ └─Bottleneck: 3-15 [1, 2048, 7, 7] 4,462,592
│ │ └─Bottleneck: 3-16 [1, 2048, 7, 7] 4,462,592
│ └─AdaptiveAvgPool2d: 2-9 [1, 2048, 1, 1] --
│ └─Sequential: 2-10 [1, 9] --
│ │ └─Linear: 3-17 [1, 512] 1,049,088
│ │ └─BatchNorm1d: 3-18 [1, 512] 1,024
│ │ └─ReLU: 3-19 [1, 512] --
│ │ └─Dropout: 3-20 [1, 512] --
│ │ └─Linear: 3-21 [1, 256] 131,328
│ │ └─BatchNorm1d: 3-22 [1, 256] 512
│ │ └─ReLU: 3-23 [1, 256] --
│ │ └─Dropout: 3-24 [1, 256] --
│ │ └─Linear: 3-25 [1, 9] 2,313
===============================================================================================
Total params: 24,692,297
Trainable params: 24,692,297
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 4.09
===============================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 177.84
Params size (MB): 98.77
Estimated Total Size (MB): 277.21
===============================================================================================

### Architecture Breakdown

**Backbone Structure:**
- **Initial Convolution**: 7×7 conv with 64 filters, stride=2, padding=3 → output [1, 64, 112, 112]
- **Max Pooling**: 3×3 max pool with stride=2 → output [1, 64, 56, 56]
- **Residual Blocks**: Four sequential stages with bottleneck blocks:
  - **Stage 1**: 3 bottleneck blocks, output [1, 256, 56, 56]
  - **Stage 2**: 4 bottleneck blocks, output [1, 512, 28, 28]
  - **Stage 3**: 6 bottleneck blocks, output [1, 1024, 14, 14]
  - **Stage 4**: 3 bottleneck blocks, output [1, 2048, 7, 7]
- **Global Pooling**: Adaptive average pooling → [1, 2048, 1, 1]

**Custom Classifier Head:**
- **First Layer**: Linear(2048 → 512) + BatchNorm1d + ReLU + Dropout(0.5) - 1,049,088 parameters
- **Second Layer**: Linear(512 → 256) + BatchNorm1d + ReLU + Dropout(0.4) - 131,328 parameters
- **Output Layer**: Linear(256 → 9) - 2,313 parameters for 9-class classification

**Key Architecture Features:**
- **Total Parameters**: 24.7M with all layers trainable in fine-tuning mode
- **Memory Footprint**: ~277MB total model size with ~178MB activation memory
- **Computational Complexity**: 4.09 GMACs (8.18 GFLOPs) for forward pass
- **Residual Connections**: Skip connections enabling training of very deep networks
- **Bottleneck Design**: 1×1, 3×3, 1×1 convolutions reducing computational cost while maintaining representational power
