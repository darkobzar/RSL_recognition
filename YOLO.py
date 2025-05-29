import os
import random
import yaml
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from time import time
import numpy as np
from PIL import Image

def get_iou(box1, box2):

    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_sq = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_sq = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = box1_sq + box2_sq - inter
    return inter / union if union != 0 else 0

def get_bbox(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            lbl, cx, cy, w, h = map(float, parts)
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes

def test_iou(model, test_txt_path):

    st = time()
   
    with open(test_txt_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]

    iou_scores = []

    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue

        lbl_path = img_path.replace('images', 'labels')
        lbl_path = os.path.splitext(lbl_path)[0] + '.txt'

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        result = model(img)[0]

        pred_bbox = result.boxes.xyxy.cpu().numpy()
        gt_bbox = get_bbox(lbl_path, w, h)

        for gt in gt_bbox:
            best_iou = 0
            for pred_box in pred_bbox:
                iou = get_iou(gt, pred_bbox)
                best_iou = max(best_iou, iou)
            iou_scores.append(best_iou)

    mean_iou = np.mean(iou_scores) if iou_scores else 0.0

    return mean_iou, time() - st


def classify_metrics(model, test_txt_path, window_size=8):
    
    with open(test_txt_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]

    y_true_all = []
    y_pred_all = []

    for i in range(0, len(image_paths) - window_size + 1):

        window_paths = image_paths[i:i + window_size]

        true_classes_window = []
        pred_classes_window = []

        for img_path in window_paths:
            if not os.path.exists(img_path):
                break

            results = model(img_path)[0]
            preds = results.boxes.cls.cpu().numpy().astype(int) if results.boxes.cls is not None else []
            pred_classes_window.extend(preds)

            label_path = img_path.replace("images", "labels")
            label_path = os.path.splitext(label_path)[0] + ".txt"
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            true_classes_window.append(int(parts[0]))

        if len(true_classes_window) < window_size or len(pred_classes_window) < window_size:
            continue

        gt_mode = Counter(true_classes_window).most_common(1)[0][0]
        pred_mode = Counter(pred_classes_window).most_common(1)[0][0]

        if gt_mode is not None and pred_mode is not None:
            y_true_all.append(gt_mode)
            y_pred_all.append(pred_mode)

    acc = accuracy_score(y_true_all, y_pred_all)
    prec = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    rec = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)

    print(f"Accuracy:  {acc}")
    print(f"Precision: {prec}")
    print(f"Recall:    {rec}")
    print(f"F1 Score:  {f1}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }