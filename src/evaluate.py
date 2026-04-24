import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc)
from dataset import get_dataloaders
from model import AIImageDetector
DATASET_DIR = "dataset"
OUTPUT_DIR  = "outputs"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Real", "Fake"]
@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval() 
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)[:, 1]  
        preds   = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)
def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual",    fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Confusion matrix saved → {save_path}")
def plot_roc_curve(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="royalblue", lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate",  fontsize=13)
    ax.set_title("ROC Curve", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] ROC curve saved → {save_path}")
def main():
    _, test_loader = get_dataloaders(DATASET_DIR)
    model = AIImageDetector(num_classes=2, pretrained=False).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"),
                   map_location=DEVICE))
    labels, preds, probs = get_predictions(model, test_loader, DEVICE)
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec  = recall_score(labels, preds)
    f1   = f1_score(labels, preds)
    print("\n" + "="*45)
    print("  EVALUATION RESULTS")
    print("="*45)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print("="*45)
    print("\nDetailed Classification Report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))
    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"Accuracy  : {acc*100:.2f}%\n")
        f.write(f"Precision : {prec*100:.2f}%\n")
        f.write(f"Recall    : {rec*100:.2f}%\n")
        f.write(f"F1-Score  : {f1*100:.2f}%\n\n")
        f.write(classification_report(labels, preds, target_names=CLASS_NAMES))
    plot_confusion_matrix(labels, preds,
                          os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plot_roc_curve(labels, probs,
                   os.path.join(OUTPUT_DIR, "roc_curve.png"))
if __name__ == "__main__":
    main()
