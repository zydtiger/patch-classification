# VGG16 Histopathology Classification

This directory implements a specialized VGG16 training pipeline for histopathology patch classification on the NCT-CRC-HE-100K dataset. The pipeline leverages a pretrained VGG16 backbone with a custom classifier head optimized for 9-class tissue classification (ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM). The training process incorporates VGG16-specific architectural optimizations including enhanced data augmentation strategies and different learning rates for backbone and classifier to compensate for VGG16's tendency to overfit on smaller datasets. The model trains in fine-tuning mode, updating the feature extracting backbone and classifier simultaneously, with a focus on training the classifier. The data loading pipeline applys ImageNet standard normalization and histopathology-specific augmentations such as random rotation (20°), color jittering, and random erasing to improve model robustness. Training is managed with an early stopping mechanism with patience-based validation loss monitoring, ensuring optimal model selection while preventing overfitting.

## Architecture

### Model Structure

Sample torchinfo summary input size is (B, 3, 224, 224)

==========================================================================================
Layer (type:depth-idx) Output Shape Param #
==========================================================================================
VGG16Classifier [1, 9] --
├─VGG: 1-1 [1, 9] --
│ └─Sequential: 2-1 [1, 512, 7, 7] --
│ │ └─Conv2d: 3-1 [1, 64, 224, 224] 1,792
│ │ └─ReLU: 3-2 [1, 64, 224, 224] --
│ │ └─Conv2d: 3-3 [1, 64, 224, 224] 36,928
│ │ └─ReLU: 3-4 [1, 64, 224, 224] --
│ │ └─MaxPool2d: 3-5 [1, 64, 112, 112] --
│ │ └─Conv2d: 3-6 [1, 128, 112, 112] 73,856
│ │ └─ReLU: 3-7 [1, 128, 112, 112] --
│ │ └─Conv2d: 3-8 [1, 128, 112, 112] 147,584
│ │ └─ReLU: 3-9 [1, 128, 112, 112] --
│ │ └─MaxPool2d: 3-10 [1, 128, 56, 56] --
│ │ └─Conv2d: 3-11 [1, 256, 56, 56] 295,168
│ │ └─ReLU: 3-12 [1, 256, 56, 56] --
│ │ └─Conv2d: 3-13 [1, 256, 56, 56] 590,080
│ │ └─ReLU: 3-14 [1, 256, 56, 56] --
│ │ └─Conv2d: 3-15 [1, 256, 56, 56] 590,080
│ │ └─ReLU: 3-16 [1, 256, 56, 56] --
│ │ └─MaxPool2d: 3-17 [1, 256, 28, 28] --
│ │ └─Conv2d: 3-18 [1, 512, 28, 28] 1,180,160
│ │ └─ReLU: 3-19 [1, 512, 28, 28] --
│ │ └─Conv2d: 3-20 [1, 512, 28, 28] 2,359,808
│ │ └─ReLU: 3-21 [1, 512, 28, 28] --
│ │ └─Conv2d: 3-22 [1, 512, 28, 28] 2,359,808
│ │ └─ReLU: 3-23 [1, 512, 28, 28] --
│ │ └─MaxPool2d: 3-24 [1, 512, 14, 14] --
│ │ └─Conv2d: 3-25 [1, 512, 14, 14] 2,359,808
│ │ └─ReLU: 3-26 [1, 512, 14, 14] --
│ │ └─Conv2d: 3-27 [1, 512, 14, 14] 2,359,808
│ │ └─ReLU: 3-28 [1, 512, 14, 14] --
│ │ └─Conv2d: 3-29 [1, 512, 14, 14] 2,359,808
│ │ └─ReLU: 3-30 [1, 512, 14, 14] --
│ │ └─MaxPool2d: 3-31 [1, 512, 7, 7] --
│ └─AdaptiveAvgPool2d: 2-2 [1, 512, 7, 7] --
│ └─Sequential: 2-3 [1, 9] --
│ │ └─Linear: 3-32 [1, 4096] 102,764,544
│ │ └─ReLU: 3-33 [1, 4096] --
│ │ └─Dropout: 3-34 [1, 4096] --
│ │ └─Linear: 3-35 [1, 1024] 4,195,328
│ │ └─ReLU: 3-36 [1, 1024] --
│ │ └─Dropout: 3-37 [1, 1024] --
│ │ └─Linear: 3-38 [1, 9] 9,225
==========================================================================================
Total params: 121,683,785
Trainable params: 121,683,785
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 15.47
==========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 108.42
Params size (MB): 486.74
Estimated Total Size (MB): 595.76
==========================================================================================

### Architecture Breakdown

**Backbone Structure:**

- **Input Size**: [B, 3, 224, 224] (batch_size, channels, height, width)
- **Initial Convolution Block**: Two 3×3 conv layers with 64 filters → output [B, 64, 224, 224]
- **Pooling Layer 1**: 2×2 max pool with stride=2 → output [B, 64, 112, 112]
- **Convolution Block 2**: Two 3×3 conv layers with 128 filters → output [B, 128, 112, 112]
- **Pooling Layer 2**: 2×2 max pool with stride=2 → output [B, 128, 56, 56]
- **Convolution Block 3**: Three 3×3 conv layers with 256 filters → output [B, 256, 56, 56]
- **Pooling Layer 3**: 2×2 max pool with stride=2 → output [B, 256, 28, 28]
- **Convolution Block 4**: Three 3×3 conv layers with 512 filters → output [B, 512, 28, 28]
- **Pooling Layer 4**: 2×2 max pool with stride=2 → output [B, 512, 14, 14]
- **Convolution Block 5**: Three 3×3 conv layers with 512 filters → output [B, 512, 14, 14]
- **Pooling Layer 5**: 2×2 max pool with stride=2 → output [B, 512, 7, 7]
- **Global Pooling**: Adaptive average pooling → [B, 512, 7, 7]

**Custom Classifier Head:**

- **First Layer**: Linear(512×7×7 → 4096) + ReLU + Dropout(0.5) - 102,764,544 parameters
- **Second Layer**: Linear(4096 → 1024) + ReLU + Dropout(0.5) - 4,195,328 parameters
- **Output Layer**: Linear(1024 → 9) - 9,225 parameters for 9-class classification

**Key Architecture Features:**

- **Total Parameters**: 121.7M with all layers trainable in fine-tuning mode (significantly larger than ResNet50)
- **Memory Footprint**: ~596MB total model size with ~108MB activation memory
- **Computational Complexity**: 15.47 mult-adds (30.94 GFLOPs, since 1 mult-add = 2 FLOPs) for forward pass (higher computational cost than ResNet50)
- **Sequential Convolution Pattern**: Consistent 3×3 kernel size throughout the network with increasing depth
- **Progressive Feature Learning**: Increasing filter counts (64→128→256→512) enable hierarchical feature extraction
- **Large Fully Connected Layers**: Traditional VGG architecture with dense classifier layers requiring careful regularization
- **Uniform Convolution Design**: All convolutional layers use 3×3 kernels with stride=1 and padding=1, maintaining spatial dimensions until pooling
- **Max Pooling Strategy**: Five 2×2 max pooling layers progressively reduce spatial dimensions by factor of 32 overall
