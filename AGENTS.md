# Patch Classification Project

This project implements deep learning models for histopathology patch classification, supporting both CLS token-based classification and patch-level Multiple Instance Learning (MIL).

## Data Structure

```
data/
└── NCT-CRC-HE-100K/
    ├── ADI/     # Adipose tissue
    ├── BACK/    # Background
    ├── DEB/     # Debris
    ├── LYM/     # Lymphocytes
    ├── MUC/     # Mucus
    ├── MUS/     # Muscle
    ├── NORM/    # Normal mucosa
    ├── STR/     # Strroma
    └── TUM/     # Tumor

# Generated embeddings
data/NCT-CRC-HE-100K-embeddings/
    ├── {class_name}-{filename}.npy  # Embedding files (261 x 1280)
    └── ...

# Model checkpoints
checkpoints/
    └── checkpoint_epoch_{N}.pth

# Training logs
runs/
    └── {experiment_name}/
```

**Embedding File Format**: Each `.npy` file contains a 261×1280 array where:
- Row 0: CLS token embedding (for classification mode)
- Rows 1-4: Skip tokens
- Rows 5-256: Patch embeddings (251 patches, for MIL mode)

## Core Files

### Model Architecture
- **[`model.py`](model.py)** - Neural network models
  - `PatchMIL`: Attention-based Multiple Instance Learning for patch-level classification
  - `MulticlassMLP`: Simple feedforward network for CLS token classification

### Training Pipeline
- **[`train.py`](train.py)** - Main training script with CLI interface
  - Supports both `cls` and `patch` modes via `--dataset-type` argument
  - Conditional model selection and dataloader configuration
  - Checkpoint management and resumption capabilities

### Data Handling
- **[`dataloader.py`](dataloader.py)** - Dataset classes and data loading
  - `EmbeddingDataset`: Loads and processes `.npy` embedding files
  - Supports both CLS token extraction and patch-level embedding modes
  - Automatic class mapping and one-hot encoding

- **[`embed.py`](embed.py)** - Embedding generation script
  - Processes TIF images using pretrained vision transformers
  - Generates 261×1280 embeddings for each histopathology image
  - Saves embeddings in structured format for downstream training

### Training Infrastructure
- **[`training_manager.py`](training_manager.py)** - Training orchestration
  - `TrainingManager`: Handles training loops, validation, checkpointing
  - TensorBoard logging, metrics tracking, early stopping integration

- **[`early_stopping.py`](early_stopping.py)** - Early stopping implementation
  - Monitors validation metrics with patience and min_delta thresholds
  - Prevents overfitting and saves best model checkpoints

### Utilities
- **[`utils.py`](utils.py)** - Helper functions
  - Device selection (CUDA/MPS/CPU)
  - Data loader creation with optimal configurations
  - Checkpoint management utilities

### Evaluation & Visualization
- **[`eval/classify.py`](eval/classify.py)** - Inference and classification
  - Model loading and batch inference on whole slide images
  - Patch-wise classification with visualization overlays

- **[`eval/test.py`](eval/test.py)** - Model evaluation script
  - Performance metrics calculation on test datasets
  - Confusion matrix and classification reports

- **[`eval/vis_utils.py`](eval/vis_utils.py)** - Visualization utilities
  - Custom colormaps and overlay generation
  - Heatmap creation for classification results

## Key Features

### Dual Training Modes
- **CLS Mode**: Uses CLS token embeddings with `MulticlassMLP` for image-level classification
- **Patch Mode**: Uses patch embeddings with `PatchMIL` for attention-based multiple instance learning

### CLI Interface
```bash
# Patch-level MIL training
python train.py --name experiment --dataset-type patch --embed-dir ./data/embeddings --embed-dim 1280 --num-classes 9

# CLS token classification training
python train.py --name experiment --dataset-type cls --embed-dir ./data/embeddings --embed-dim 1280 --num-classes 9
```

### Model Architectures
- **PatchMIL**: Attention mechanism over 251 patches per image
- **MulticlassMLP**: Simple feedforward network for direct classification

### Hardware Optimization
- Automatic device detection (CUDA > MPS > CPU)
- Optimized DataLoader settings with multiprocessing
- Memory-efficient batch processing

## Dependencies

Dependencies are managed with `uv` via [`pyproject.toml`](pyproject.toml). Run `uv sync` to create or update the local environment with PyTorch, timm, typer, and the rest of the project dependencies.
