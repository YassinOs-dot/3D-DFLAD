import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
import seaborn as sns


def compute_thresholds(preds, labels, name="Image"):
    """
    Compute thresholds for anomaly detection.

    - thr_good : quantile-based threshold from good samples
    - thr_anom : quantile-based threshold from anomalous samples
    - final_thr: balanced threshold (EER, where FPR ≈ FNR)

    preds  : array-like anomaly scores (N,)
    labels : array-like true labels (0=good, 1=anomaly)
    name   : str, for reporting
    """
    preds = np.array(preds).reshape(-1)
    labels = np.array(labels).reshape(-1)

    # Separate good vs anomaly scores
    good_scores = preds[labels == 0]
    anom_scores = preds[labels == 1]

    n_good = len(good_scores)
    n_anom = len(anom_scores)

    # --- Quantile thresholds ---
    thr_good = np.quantile(good_scores, 0.95) if n_good > 0 else None
    thr_anom = np.quantile(anom_scores, 0.05) if n_anom > 0 else None

    # --- Balanced threshold (EER = Equal Error Rate) ---
    thresholds = np.unique(preds)
    fpr_list, fnr_list, diff_list = [], [], []

    for t in thresholds:
        pred_labels = (preds >= t).astype(int)
        fp = np.sum((pred_labels == 1) & (labels == 0))
        fn = np.sum((pred_labels == 0) & (labels == 1))

        fpr = fp / n_good if n_good > 0 else 0
        fnr = fn / n_anom if n_anom > 0 else 0

        fpr_list.append(fpr)
        fnr_list.append(fnr)
        diff_list.append(abs(fpr - fnr))

    best_idx = np.argmin(diff_list)
    final_thr = thresholds[best_idx]
    eer_fpr, eer_fnr = fpr_list[best_idx], fnr_list[best_idx]

    # --- Predictions using final threshold ---
    final_preds = (preds >= final_thr).astype(int)

    # --- Evaluation ---
    cm = confusion_matrix(labels, final_preds)
    report = classification_report(labels, final_preds, target_names=["Good", "Anomalous"])

    print(f"\n===== {name} Thresholds & Evaluation =====")
    print("Quantile threshold (good):", thr_good)
    print("Quantile threshold (anom):", thr_anom)
    print("Final balanced threshold:", final_thr)
    print("At final threshold -> FPR:", eer_fpr, "FNR:", eer_fnr)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    return {
        "thr_good": thr_good,
        "thr_anom": thr_anom,
        "final_thr": final_thr,
        "eer_fpr": eer_fpr,
        "eer_fnr": eer_fnr,
        "confusion_matrix": cm,
        "classification_report": report
    }



def evaluate_and_plot_metrics(
    image_preds, image_labels,
    pixel_preds, pixel_labels,
    det_th=10, seg_th=34.46,
    output_dir="results"
):
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # IMAGE-LEVEL (Detection)
    # -------------------------
    image_preds_bin = (image_preds > det_th).astype(int)
    cm_det = confusion_matrix(image_labels, image_preds_bin)

    det_report = classification_report(image_labels, image_preds_bin, output_dict=True)
    det_auc = roc_auc_score(image_labels, image_preds)
    det_ap = average_precision_score(image_labels, image_preds)

    # Save metrics
    with open(os.path.join(output_dir, "detection_metrics.txt"), "w") as f:
        f.write("=== Detection (Image-level) ===\n")
        f.write(f"AUC: {det_auc:.4f}\n")
        f.write(f"AP: {det_ap:.4f}\n")
        f.write(classification_report(image_labels, image_preds_bin))

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_det, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Good", "Anomaly"],
                yticklabels=["Good", "Anomaly"])
    plt.title("Detection Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(output_dir, "detection_cm.png"))
    plt.close()

    # -------------------------
    # PIXEL-LEVEL (Segmentation)
    # -------------------------
    pixel_preds_bin = (pixel_preds > seg_th).astype(int)
    cm_seg = confusion_matrix(pixel_labels.flatten(), pixel_preds_bin.flatten())

    seg_report = classification_report(pixel_labels.flatten(), pixel_preds_bin.flatten(), output_dict=True)
    seg_auc = roc_auc_score(pixel_labels.flatten(), pixel_preds.flatten())
    seg_ap = average_precision_score(pixel_labels.flatten(), pixel_preds.flatten())

    # Save metrics
    with open(os.path.join(output_dir, "segmentation_metrics.txt"), "w") as f:
        f.write("=== Segmentation (Pixel-level) ===\n")
        f.write(f"AUC: {seg_auc:.4f}\n")
        f.write(f"AP: {seg_ap:.4f}\n")
        f.write(classification_report(pixel_labels.flatten(), pixel_preds_bin.flatten()))

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_seg, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Good", "Anomaly"],
                yticklabels=["Good", "Anomaly"])
    plt.title("Segmentation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(output_dir, "segmentation_cm.png"))
    plt.close()

    print(f"✅ Results saved in {output_dir}")
  
from sklearn.metrics import roc_curve, auc

def plot_roc_curves(image_preds, image_labels, pixel_preds, pixel_labels):
    fig, axs = plt.subplots(1,2, figsize=(12,5))

    # Image-level ROC
    fpr, tpr, _ = roc_curve(image_labels, image_preds)
    axs[0].plot(fpr, tpr, label=f"AUC = {auc(fpr,tpr):.3f}")
    axs[0].plot([0,1],[0,1], 'k--')
    axs[0].set_title("Image-level ROC")
    axs[0].set_xlabel("FPR")
    axs[0].set_ylabel("TPR")
    axs[0].legend()

    # Pixel-level ROC
    fpr, tpr, _ = roc_curve(pixel_labels, pixel_preds)
    axs[1].plot(fpr, tpr, label=f"AUC = {auc(fpr,tpr):.3f}")
    axs[1].plot([0,1],[0,1], 'k--')
    axs[1].set_title("Pixel-level ROC")
    axs[1].set_xlabel("FPR")
    axs[1].set_ylabel("TPR")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
#-----------------------------
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_pr_curves(image_preds, image_labels, pixel_preds, pixel_labels):
    fig, axs = plt.subplots(1,2, figsize=(12,5))

    # Image-level PR
    prec, rec, _ = precision_recall_curve(image_labels, image_preds)
    ap = average_precision_score(image_labels, image_preds)
    axs[0].plot(rec, prec, label=f"AP = {ap:.3f}")
    axs[0].set_title("Image-level PR curve")
    axs[0].set_xlabel("Recall")
    axs[0].set_ylabel("Precision")
    axs[0].legend()

    # Pixel-level PR
    prec, rec, _ = precision_recall_curve(pixel_labels, pixel_preds)
    ap = average_precision_score(pixel_labels, pixel_preds)
    axs[1].plot(rec, prec, label=f"AP = {ap:.3f}")
    axs[1].set_title("Pixel-level PR curve")
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
#----------------------------------
