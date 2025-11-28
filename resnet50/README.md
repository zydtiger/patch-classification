# ResNet50 Histopathology Classification

This module implements a ResNet50-based classifier for histopathology image classification, providing a direct comparison to the existing embedding-based approach in the parent directory.

## Overview

The ResNet50 implementation works directly with raw TIF images from the NCT-CRC-HE-100K dataset, using transfer learning from ImageNet-pretrained weights. It maintains architectural consistency with the existing codebase while providing an alternative approach based on convolutional neural networks rather than transformer embeddings.

## Key Features

### Model Architecture
- **Pretrained ResNet50**: Uses ImageNet-pretrained weights as backbone
- **Custom Classifier Head**: Multi-layer classifier for 9 tissue types
- **Transfer Learning Modes**:
  - Feature extraction (freeze backbone)
  - Fine-tuning (unfreeze layers with differential learning rates)

### Training Pipeline
- **Consistent CLI**: Same typer-based interface as existing code
- **Flexible Augmentation**: Multiple transform configurations
- **Class Imbalance Handling**: Optional class weights and balanced sampling
- **Checkpoint Management**: Resume training and best model saving

### Evaluation & Visualization
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score
- **Visualization Tools**: Confusion matrices, training curves, Grad-CAM
- **Comparison Tools**: Side-by-side comparison with embedding-based models
- **Whole Slide Classification**: Sliding window inference for large images

## Directory Structure

```
resnet50/
├── model.py                    # ResNet50 classifier architecture
├── train.py                    # Training script with CLI interface
├── utils.py                    # Data loading and utility functions
├── transforms.py               # Image augmentation pipeline
├── eval/
│   ├── test.py                 # Model evaluation and comparison
│   ├── classify.py             # Whole slide image classification
│   └── vis_utils.py            # Visualization utilities
└── README.md                   # This file
```

## Quick Start

### 1. Training

#### Basic Training (Feature Extraction)
```bash
python resnet50/train.py \
    --name resnet50_experiment \
    --data_root data/NCT-CRC-HE-100K \
    --feature-extract \
    --batch-size 32 \
    --num-epochs 50
```

#### Fine-tuning with Differential Learning Rates
```bash
python resnet50/train.py \
    --name resnet50_finetune \
    --data_root data/NCT-CRC-HE-100K \
    --feature-extract False \
    --lr 1e-3 \
    --backbone-lr 1e-4 \
    --batch-size 16 \
    --num-epochs 100
```

#### Progressive Fine-tuning (Freeze → Unfreeze)
```bash
python resnet50/train.py \
    --name resnet50_progressive \
    --data_root data/NCT-CRC-HE-100K \
    --feature-extract \
    --unfreeze-epochs 20 \
    --lr 1e-3 \
    --backbone-lr 1e-4 \
    --batch-size 32 \
    --num-epochs 60
```

### 2. Evaluation

#### Single Model Evaluation
```bash
python resnet50/eval/test.py \
    --checkpoint-path runs/resnet50_experiment/checkpoints/checkpoint_epoch_25.pth \
    --data_root data/NCT-CRC-HE-100K \
    --output-dir eval_results
```

#### Model Comparison
```bash
python resnet50/eval/test.py compare-checkpoints \
    runs/exp1/checkpoints/checkpoint_epoch_25.pth \
    runs/exp2/checkpoints/checkpoint_epoch_30.pth \
    --data_root data/NCT-CRC-HE-100K \
    --output-dir comparison_results
```

### 3. Whole Slide Classification

#### Single Image Classification
```bash
python resnet50/eval/classify.py classify-single \
    --model-path runs/resnet50_experiment/checkpoints/latest.pth \
    --image-path path/to/test_image.tif \
    --output-dir classification_results \
    --patch-size 224 \
    --stride 112
```

#### Batch Classification
```bash
python resnet50/eval/classify.py batch-classify \
    --model-path runs/resnet50_experiment/checkpoints/latest.pth \
    --input-dir path/to/test_images/ \
    --output-dir batch_results \
    --file-pattern "*.tif"
```

### 4. Dataset Testing
```bash
python resnet50/train.py test-dataset \
    --data_root data/NCT-CRC-HE-100K \
    --transform-config standard \
    --batch-size 4
```

## Configuration Options

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--name` | Required | Experiment name for logging |
| `--data_root` | `data/NCT-CRC-HE-100K` | Root directory of images |
| `--num_classes` | 9 | Number of output classes |
| `--img_size` | 224 | Input image size |
| `--batch_size` | 32 | Training batch size |
| `--learning_rate` | 1e-3 | Learning rate for classifier |
| `--backbone_lr` | 1e-4 | Learning rate for backbone (fine-tuning) |
| `--weight_decay` | 1e-4 | L2 regularization |
| `--num_epochs` | 50 | Maximum training epochs |
| `--test_size` | 0.2 | Validation set fraction |
| `--device` | auto | Device (auto/cuda/mps/cpu) |

### Advanced Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--feature_extract` | True | Freeze backbone for feature extraction |
| `--unfreeze_epochs` | 0 | Epoch to unfreeze for fine-tuning |
| `--balanced_sampling` | False | Use balanced sampling for class imbalance |
| `--use_class_weights` | False | Use class weights in loss function |
| `--transform_config` | standard | Augmentation config (standard/strong/minimal) |
| `--random_seed` | 42 | Random seed for reproducibility |

### Transform Configurations

#### Standard (Default)
- Basic medical image augmentations
- Horizontal/vertical flip (50%)
- Random rotation (±15°)
- Conservative color jitter
- ImageNet normalization

#### Strong
- Additional aggressive augmentations
- Random resized crop
- Affine transformations
- Random erasing
- Higher augmentation intensity

#### Minimal
- Only essential transforms
- Resize and normalization
- No data augmentation

## Model Architecture Details

### ResNet50 Backbone
- Pretrained on ImageNet
- 50-layer residual network
- ~23 million parameters
- Feature extraction: 2048-dim embeddings

### Custom Classifier Head
```
Linear(2048 → 512)
BatchNorm1d(512)
ReLU
Dropout(0.5)

Linear(512 → 256)
BatchNorm1d(256)
ReLU
Dropout(0.4)

Linear(256 → 9)  # 9 tissue classes
```

### Transfer Learning Strategies

#### 1. Feature Extraction
- Freeze all ResNet50 layers
- Train only classifier head
- Faster training, less prone to overfitting
- Good baseline approach

#### 2. Fine-tuning
- Unfreeze some/all ResNet50 layers
- Use differential learning rates
- Lower LR for backbone, higher LR for classifier
- Better performance with sufficient data

#### 3. Progressive Fine-tuning
- Start with feature extraction
- Unfreeze after N epochs
- Gradual adaptation to medical domain
- Balances stability and performance

## Class Information

The model classifies 9 tissue types from NCT-CRC-HE-100K:

| Class | Description | Typical Appearance |
|-------|-------------|-------------------|
| ADI | Adipose tissue | Fat cells, white space |
| BACK | Background | Empty areas, artifacts |
| DEB | Debris | Dead cells, waste |
| LYM | Lymphocytes | Small round cells |
| MUC | Mucus | Gel-like substance |
| MUS | Muscle | Fibrous tissue |
| NORM | Normal mucosa | Healthy epithelium |
| STR | Stroma | Support tissue |
| TUM | Tumor | Cancerous cells |

## Performance Benchmarks

### Expected Performance
- **Feature Extraction**: ~85-90% accuracy
- **Fine-tuning**: ~90-95% accuracy
- **Training Time**: 1-4 hours (depending on GPU)
- **Memory Usage**: 8-16GB GPU memory

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended
- **CPU**: Multi-core CPU for data loading
- **RAM**: 16GB+ system memory
- **Storage**: 2GB+ for models and checkpoints

## Integration with Existing Codebase

### Reused Components
- `TrainingManager`: Training orchestration
- `EarlyStopping`: Overfitting prevention
- `TIFDataset`: Image loading (from embed.py)
- Device management utilities
- Checkpoint management system

### Compatible Features
- TensorBoard logging
- CLI parameter naming
- Experiment organization
- Visualization utilities

### Model Comparison
The implementation includes tools for direct comparison:
- Side-by-side metrics
- Confusion matrix comparison
- Class-wise performance analysis
- Comprehensive evaluation reports

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
- Reduce batch size (16 → 8 → 4)
- Use gradient accumulation
- Enable mixed precision training

#### Slow Training
- Increase number of workers
- Use SSD for data storage
- Enable pin_memory
- Check data loading bottlenecks

#### Poor Performance
- Try different learning rates
- Adjust augmentation strength
- Use class weights for imbalance
- Consider progressive fine-tuning

#### Data Loading Errors
- Verify dataset structure
- Check TIF file integrity
- Ensure correct data paths
- Test with smaller dataset

### Debug Mode
Enable verbose logging:
```bash
python resnet50/train.py \
    --name debug_experiment \
    --data_root data/NCT-CRC-HE-100K \
    --batch-size 4 \
    --num-epochs 2 \
    --num-workers 1
```

## Contributing

### Adding New Features
1. Follow existing code patterns
2. Update documentation
3. Add comprehensive tests
4. Maintain CLI consistency

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add comprehensive docstrings
- Include error handling

### Testing
```bash
# Test dataset loading
python resnet50/train.py test-dataset

# Test model creation
python -c "from resnet50.model import create_model; print(create_model())"

# Test transforms
python -c "from resnet50.transforms import get_train_transforms; print(get_train_transforms())"
```

## Citation

If you use this ResNet50 implementation in your research, please cite:

```bibtex
@misc{resnet50_histopathology,
  title={ResNet50-based Classification for Histopathology Images},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/patch-classification}
}
```

Also cite the original dataset:
```bibtex
@article{kather2016multi,
  title={Multi-class texture analysis in colorectal cancer histology},
  author={Kather, Jakob Nikolas and Weis, Cleo-Aron and Bianconi, Francesco and Melchers, Susanne M and Schad, Lothar R and Gaiser, Tim and Marx, Alexander and Z{\"o}llner, Frank G},
  journal={Scientific reports},
  volume={6},
  pages={27988},
  year={2016},
  publisher={Nature Publishing Group}
}
```