from typing import List, Tuple, Optional

import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, device: str
) -> Tuple[List[int], np.ndarray, List[int]]:
    """
    Evaluate the model and return true labels, predicted probabilities and predicted labels
    """
    model.eval()
    true_labels = []
    pred_probs = []
    pred_labels = []

    with torch.inference_mode():
        pbar = tqdm(data_loader, desc="Running model")
        for data, _, targets in pbar:
            data = data.to(device)
            outputs = model(data)  # shape: [batch, num_classes]
            probs = (
                torch.softmax(outputs, dim=1).cpu().numpy()
            )  # shape: [batch, num_classes]
            pred_probs.append(probs)
            pred = np.argmax(probs, axis=1)
            pred_labels.extend(pred)

            true_labels.extend(targets)

    pred_probs = np.concatenate(
        pred_probs, axis=0
    )  # shape: [n_samples, num_classes]

    return true_labels, pred_probs, pred_labels


def plot_confusion_matrix(
    true_labels: List[int],
    pred_labels: List[int],
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> matplotlib.figure.Figure:
    """
    Plot confusion matrix with metrics for multiclass

    Args:
        true_labels: List of true class labels
        pred_labels: List of predicted class labels
        num_classes: Number of classes
        class_names: Optional list of class names for labels. If None, uses numeric labels.
    """
    labels = np.arange(0, num_classes)

    # Set tick labels based on class_names if provided, otherwise use numeric labels
    if class_names is not None:
        if len(class_names) != num_classes:
            raise ValueError(f"Length of class_names ({len(class_names)}) must match num_classes ({num_classes})")
        tick_labels = class_names
    else:
        tick_labels = [str(i + 1) for i in labels]  # Default: numeric labels starting from 1

    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    recall = recall_score(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        annot_kws={"size": 10},
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Add metrics text box
    plt.figtext(
        0.5,
        0.03,
        f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}",
        ha="center",
        fontsize=10,
        bbox=dict(
            facecolor="white",
            boxstyle="round,pad=0.3",
            edgecolor="black",
            linewidth=1,
        ),
    )

    plt.subplots_adjust(bottom=0.13)
    plt.tight_layout(rect=(0, 0.05, 1, 0.97))

    return plt.gcf()


def plot_multiclass_roc_curve(
    true_labels: List[int], pred_probs: np.ndarray, num_classes: int, class_names: Optional[List[str]] = None
) -> matplotlib.figure.Figure:
    """
    Plot ROC curve for multiclass (one-vs-rest) and macro-average ROC

    Args:
        true_labels: List of true class labels
        pred_probs: Predicted probabilities for each class
        num_classes: Number of classes
        class_names: Optional list of class names for labels. If None, uses numeric labels.
    """
    # Set tick labels based on class_names if provided, otherwise use numeric labels
    if class_names is not None:
        if len(class_names) != num_classes:
            raise ValueError(f"Length of class_names ({len(class_names)}) must match num_classes ({num_classes})")
        tick_labels = class_names
    else:
        tick_labels = [f"Class {i+1}" for i in range(num_classes)]  # Default: numeric labels starting from 1

    # Binarize the labels for one-vs-rest ROC, same as OHE in this case
    y_true_bin = np.array(
        label_binarize(true_labels, classes=np.arange(0, num_classes))
    )
    # y_true_bin: shape [n_samples, num_classes]
    # pred_probs: shape [n_samples, num_classes]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure(figsize=(8, 6))

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], pred_probs[:, i])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], pred_probs[:, i])
        plt.plot(
            fpr[i],
            tpr[i],
            linestyle="--",
            label=f"{tick_labels[i]} (AUC = {roc_auc[i]:.2f})",
        )

    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        # Use scipy.interpolate.interp1d for modern interpolation
        interp_func = interp1d(fpr[i], tpr[i], kind='linear', bounds_error=False, fill_value=0.0)
        mean_tpr += interp_func(all_fpr)
    mean_tpr /= num_classes

    macro_auc = np.mean(list(roc_auc.values()))
    plt.plot(
        all_fpr,
        mean_tpr,
        color="navy",
        linestyle="-",
        linewidth=4,
        label=f"Macro-average ROC (AUC = {macro_auc:.2f})",
    )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve")
    plt.legend()
    plt.tight_layout()

    return plt.gcf()


def visualize_model_performance(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]:
    """
    Comprehensive model evaluation and visualization for multiclass

    Args:
        model: Trained neural network model
        data_loader: DataLoader for test/validation data
        device: Device to run inference on
        num_classes: Number of classes
        class_names: Optional list of class names for confusion matrix labels
    """
    true_labels, pred_probs, pred_labels = evaluate_model(
        model, data_loader, device
    )
    cm_fig = plot_confusion_matrix(true_labels, pred_labels, num_classes, class_names)
    roc_fig = plot_multiclass_roc_curve(true_labels, pred_probs, num_classes, class_names)

    return cm_fig, roc_fig
