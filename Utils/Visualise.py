import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tifffile as tiff
from scipy.ndimage import label
import matplotlib.patches as patches
import natsort  # for natural sorting
from skimage import measure


def visualize_image_scores(images, image_preds, image_labels, output_dir="results/image_scores"):
    os.makedirs(output_dir, exist_ok=True)
    colors = ["blue", "red"]  # blue=good, red=anomaly

    x = range(len(image_preds))
    plt.figure(figsize=(12,6))
    # Convert image_labels to integers before using as indices
    plt.scatter(x, image_preds, c=[colors[int(l)] for l in image_labels], s=40)
    plt.plot(x, image_preds, alpha=0.6)
    plt.xlabel("Image Index")
    plt.ylabel("Anomaly Score")
    plt.title("Image-level Anomaly Scores")
    plt.savefig(os.path.join(output_dir, "image_scores.png"))
    plt.close()

#-----------------


def visualize_heatmaps(test_loader, pixel_preds, pixel_labels, output_dir="results/heatmaps"):
    """
    Visualize heatmaps (pixel_preds) alongside GT masks (pixel_labels).
    test_loader   : DataLoader for the test set
    pixel_preds   : array of predicted anomaly maps (N,H,W)
    pixel_labels  : array of GT masks (N,H,W)
    """
    os.makedirs(output_dir, exist_ok=True)

    img_idx = 0
    for sample, _, _ in test_loader:
        # sample is a tuple: (img, resized_organized_pc, resized_depth_map_3channel)
        img_tensor = sample[0] # This is the batched image tensor (1, C, H, W)

        # Iterate through the batch (batch size is 1 in this case)
        for i in range(img_tensor.shape[0]):
            img = img_tensor[i].permute(1, 2, 0).numpy() # (H, W, C) numpy array
            pred_map = pixel_preds[img_idx]
            gt_mask = pixel_labels[img_idx]
            pred_norm = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-8)

            s_map_blur = cv2.GaussianBlur(pred_norm, (5,5), 1)
            s_map_thresh = (s_map_blur > 0.7).astype(np.float32) * s_map_blur
            # --- ensure shapes ---
            if s_map_thresh.ndim == 0:
                print(f"[WARNING] pred_map {img_idx} is scalar, skipping")
                img_idx += 1
                continue
            if img.ndim == 2:  # grayscale -> expand to RGB
                img = np.stack([img]*3, axis=-1)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax in axes:
                ax.axis('off') # Turn off axes for cleaner visualization

            axes[0].imshow(img.astype(np.uint8))
            axes[0].set_title("Input Image")

            axes[1].imshow(s_map_thresh, cmap="jet")
            axes[1].set_title("Predicted Heatmap")

            axes[2].imshow(gt_mask, cmap="gray")
            axes[2].set_title("Ground Truth Mask")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sample_{img_idx:04d}.png"))
            plt.close()

            img_idx += 1

#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------


# ------- loaders -------
def preprocess_image(path, normalize=True):
    img = np.array(Image.open(path).convert("RGB"))
    if normalize:
        img = img / 255.0
    return img

def preprocess_xyz(path, normalize=True):
    xyz = tiff.imread(path).astype(np.float32)
    if normalize:
        xyz = (xyz - xyz.min()) / (xyz.max() - xyz.min() + 1e-8)
    return xyz

# ------- utils -------
def resize_mask_nn(mask, target_hw):
    H, W = target_hw
    mask_u8 = mask.astype(np.uint8)
    return np.array(Image.fromarray(mask_u8).resize((W, H), resample=Image.NEAREST))

def compute_bboxes_from_gt(gt_mask, min_area=10):
    gt_bin = gt_mask > 0
    labeled, n_components = label(gt_bin)
    bboxes = []
    for comp_idx in range(1, n_components + 1):
        ys, xs = np.where(labeled == comp_idx)
        if ys.size == 0 or ys.size < min_area:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes

def visualize_with_precise_heatmap_bboxes(
    rgb_paths, xyz_paths, gt_paths, pred_maps,
    output_dir="results/fault", min_area=10, top_quantile=0.95
):
    import cv2
    """
    Draw bounding boxes from GT (red) and tightly fit only the most anomalous heatmap regions (blue).

    top_quantile : float [0-1], only pixels with anomaly score above this quantile are considered
    """
    os.makedirs(output_dir, exist_ok=True)

    rgb_paths = natsort.natsorted(rgb_paths)
    xyz_paths = natsort.natsorted(xyz_paths)
    gt_paths  = natsort.natsorted(gt_paths)
    assert len(pred_maps) == len(rgb_paths), "pred_maps must match number of images"

    n = len(rgb_paths)

    for idx in range(n):
        img_rgb = preprocess_image(rgb_paths[idx])
        img_xyz = preprocess_xyz(xyz_paths[idx])
        gt_raw  = np.array(Image.open(gt_paths[idx]))

        H, W = img_rgb.shape[:2]
        if gt_raw.shape[:2] != (H, W):
            gt_mask = resize_mask_nn(gt_raw, (H, W))
        else:
            gt_mask = gt_raw

        # GT boxes
        gt_bboxes = compute_bboxes_from_gt(gt_mask, min_area=min_area)

        # Prediction map processing
        pred_map = pred_maps[idx]
        if pred_map.shape != (H, W):
            pred_map = np.array(Image.fromarray(pred_map.astype(np.float32)).resize((W, H), resample=Image.BILINEAR))
        pred_norm = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-8)
        pred_norm = cv2.GaussianBlur(pred_norm, (5,5), 1)
        pred_norm = (pred_norm > 0.8).astype(np.float32) * pred_norm
        # High-anomaly heatmap: only top quantile
        threshold = np.quantile(pred_norm, top_quantile)
        anomaly_mask = pred_norm >= threshold
        pred_bboxes = compute_bboxes_from_gt(anomaly_mask.astype(np.uint8), min_area=min_area)

        # Plotting
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax in axes:
            ax.axis("off")

        # 1) RGB with GT + precise heatmap anomaly boxes
        axes[0].imshow(img_rgb)
        axes[0].set_title("RGB + GT & Top Anomaly BBoxes")
        for (x0, y0, x1, y1) in gt_bboxes:
            rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
        for (x0, y0, x1, y1) in pred_bboxes:
            rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor='blue', facecolor='none')
            axes[0].add_patch(rect)

        # 2) XYZ
        axes[1].imshow(img_xyz, cmap="gray")
        axes[1].set_title("XYZ (tif)")

        # 3) GT mask
        axes[2].imshow(gt_mask, cmap="gray")
        axes[2].set_title("GT Mask")

        # 4) Prediction map
        axes[3].imshow(pred_norm, cmap="jet")
        axes[3].set_title("Prediction Map")

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"sample_{idx:03d}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")


#---------------------------
# pls be sure that the first defected folder in your test set in drive/mydrive/rope/ is /combined else you should adjust the folder name to the first one
#to assert that the predictions is fitted to the gts

#--------------------------------------------------------------------
#--------------------------------------------------------------------

# pred_maps: list of HxW numpy arrays from your patchcore or anomaly scores
# Make sure the length matches rgb_paths

def visualize_hmaps(test_loader, pixel_preds, pixel_labels, output_dir="results/hmaps"):
    """
    Visualize anomaly heatmaps overlayed on input images, with outlines.
    """
    os.makedirs(output_dir, exist_ok=True)

    img_idx = 0
    for sample, _, _ in test_loader:
        img_tensor = sample[0]  # (B,C,H,W)

        for i in range(img_tensor.shape[0]):
            img = img_tensor[i].permute(1, 2, 0).cpu().numpy()
            pred_map = pixel_preds[img_idx]
            gt_mask = pixel_labels[img_idx]

            # --- safety checks ---
            if pred_map.ndim == 0:
                print(f"[WARNING] pred_map {img_idx} is scalar, skipping")
                img_idx += 1
                continue
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)

            # --- normalize prediction map to [0,1] ---
            pred_norm = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-8)
            s_map_blur = cv2.GaussianBlur(pred_norm, (5,5), 1)
            s_map_thresh = (s_map_blur > 0.8).astype(np.float32) * s_map_blur
            # --- overlay heatmap (transparent colormap) ---
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax in axes: ax.axis("off")

            axes[0].imshow(img.astype(np.uint8))
            axes[0].set_title("Input Image")

            axes[1].imshow(img.astype(np.uint8))
            axes[1].imshow(s_map_thresh, cmap="jet", alpha=0.5)  # 🔥 change colormap here
            axes[1].set_title("Overlay Heatmap")

            # --- add outlines from GT mask (white) and anomaly map (red) ---
            axes[2].imshow(img.astype(np.uint8))
            axes[2].imshow(gt_mask, cmap="gray", alpha=0.3)

            # draw GT contours in green
            contours_gt = measure.find_contours(gt_mask, 0.5)
            for contour in contours_gt:
                axes[2].plot(contour[:, 1], contour[:, 0], linewidth=2, color="lime")

            # draw predicted anomaly contours in red
            contours_pred = measure.find_contours(s_map_thresh, 0.7)  # threshold = 0.7
            for contour in contours_pred:
                axes[2].plot(contour[:, 1], contour[:, 0], linewidth=2, color="red")

            axes[2].set_title("Overlay + Contours")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sample_{img_idx:04d}.png"))
            plt.close()

            img_idx += 1
