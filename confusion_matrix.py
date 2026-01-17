import csv
import os
import sys

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix


def load_mask(path):
    img = Image.open(path).convert('L')
    return np.array(img) > 0

def load_all_masks(folder):
    files = sorted(os.listdir(folder))
    return [load_mask(os.path.join(folder, f)) for f in files if f.endswith(('.png', '.jpg', '.tif'))]

def compute_confusion(y_true_masks, y_pred_masks):
    y_true_flat = np.concatenate([m.flatten() for m in y_true_masks])
    y_pred_flat = np.concatenate([m.flatten() for m in y_pred_masks])
    return confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])

def save_confusion_matrix(cm, path):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['', 'pred_background', 'pred_landslide'])
        writer.writerow(['real_background', cm[0][0], cm[0][1]])
        writer.writerow(['real_landslide', cm[1][0], cm[1][1]])


def compute_confusion_matrix(zone, model_name):
    for confidence in np.arange(0.05, 0.45, 0.05):
        path_out = f"runs/{model_name}/{zone}/confidence_{confidence:.2f}/confusion_matrix.csv"
        path_true_masks = f"H:/Landslides/data/{zone}/segmentation_512_512/mask/"
        path_pred_masks = f"runs/{model_name}/{zone}/confidence_{confidence:.2f}/mask/"

        true_masks = load_all_masks(path_true_masks)
        pred_masks = load_all_masks(path_pred_masks)

        cm = compute_confusion(true_masks, pred_masks)
        save_confusion_matrix(cm, path_out)
        print(f"confusion matrix: {path_out}")