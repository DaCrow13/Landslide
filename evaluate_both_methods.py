import os
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
import traceback
import pandas as pd

# --- Patching ---
def patched_model_forward(self, x, *args, **kwargs):
    if hasattr(self, '_orig_forward'):
        return self._orig_forward(x, *args, **kwargs)
    return super(SegmentationModel, self).forward(x, *args, **kwargs)

def global_feature_hook(*args, **kwargs): pass

SegmentationModel.patched_model_forward = patched_model_forward
SegmentationModel.global_feature_hook = global_feature_hook
# ----------------

def evaluate_project():
    """
    Scans the validation dataset and computes BOTH:
    1. The Professor's Patch-Wise Metrics (Does the image contain a landslide?)
    2. Our Pixel-Wise Metrics (Does this specific pixel contain a landslide?)
    """
    try:
        model_path = "final_model.pt"
        source_dir = "A19_5cm/val/images"
        label_dir = "A19_5cm/val/labels"
        conf_threshold = 0.10
        
        # Post-processing filters (from our robust model)
        MIN_AREA = 300
        MAX_LINEARITY = 4.0
        
        print(f"Loading model: {model_path}...")
        model = YOLO(model_path)
        
        images = sorted([f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.tif', '.png'))])
        print(f"Evaluating {len(images)} images in {source_dir}...")
        
        # --- Metrics Accumulators ---
        # 1. Patch-Wise (Image Level)
        pw_tp, pw_fp, pw_tn, pw_fn = 0, 0, 0, 0
        
        # 2. Pixel-Wise (Pixel Level)
        px_tp, px_fp, px_tn, px_fn = 0, 0, 0, 0
        
        for idx, filename in enumerate(images):
            if idx % 50 == 0 and idx > 0:
                print(f"Processed {idx}/{len(images)}...")
                
            img_path = os.path.join(source_dir, filename)
            base_name = os.path.splitext(filename)[0]
            label_path = os.path.join(label_dir, base_name + ".txt")
            
            # --- 1. Get Ground Truth (Original Mask) ---
            # We need to build the GT mask from the YOLO txt format
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            has_gt_landslide = False
            
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = list(map(float, line.strip().split()))
                        if len(parts) > 1:
                            coords = parts[1:]
                            points = []
                            for i in range(0, len(coords), 2):
                                x_pt = int(coords[i] * w)
                                y_pt = int(coords[i+1] * h)
                                points.append([x_pt, y_pt])
                            pts = np.array([points], dtype=np.int32)
                            cv2.fillPoly(gt_mask, pts, 1)
                            has_gt_landslide = True
                            
            # --- 2. Get Prediction (Our Mask) ---
            results = model.predict(source=img_path, conf=conf_threshold, verbose=False)
            r = results[0]
            
            raw_mask = np.zeros((h, w), dtype=np.uint8)
            if r.masks is not None:
                for mask in r.masks.xy:
                    pts = np.array(mask, dtype=np.int32)
                    cv2.fillPoly(raw_mask, [pts], 1)
                    
            # Apply our filters
            pred_mask = np.zeros((h, w), dtype=np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
            
            has_pred_landslide = False
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < MIN_AREA: continue 
                
                component_mask = (labels == i).astype(np.uint8)
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                is_linear = False
                if len(contours) > 0:
                    cnt = contours[0]
                    if len(cnt) >= 5:
                        (x,y), (MA, ma), angle = cv2.fitEllipse(cnt)
                        major = max(MA, ma)
                        minor = min(MA, ma)
                        if minor > 0 and (major/minor) > MAX_LINEARITY:
                            is_linear = True
                
                if not is_linear:
                    pred_mask[labels == i] = 1
                    has_pred_landslide = True
            
            # --- 3. Compute Patch-Wise (Professor's logic) ---
            # Rule: If GT has > 0 pixels -> Real = 1, Else Real = 0
            # Rule: If Pred has > 0 pixels -> Pred = 1, Else Pred = 0
            if has_gt_landslide and has_pred_landslide:
                pw_tp += 1
            elif not has_gt_landslide and has_pred_landslide:
                pw_fp += 1
            elif has_gt_landslide and not has_pred_landslide:
                pw_fn += 1
            elif not has_gt_landslide and not has_pred_landslide:
                pw_tn += 1
                
            # --- 4. Compute Pixel-Wise (Our original logic) ---
            # Compare gt_mask (0/1) with pred_mask (0/1) element-wise
            px_tp += np.sum((gt_mask == 1) & (pred_mask == 1))
            px_fp += np.sum((gt_mask == 0) & (pred_mask == 1))
            px_fn += np.sum((gt_mask == 1) & (pred_mask == 0))
            px_tn += np.sum((gt_mask == 0) & (pred_mask == 0))
            
        # --- Final Metrics ---
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        
        def safe_div(n, d): return n / d if d > 0 else 0
        
        # Print Patch-Wise
        pw_precision = safe_div(pw_tp, pw_tp + pw_fp)
        pw_recall = safe_div(pw_tp, pw_tp + pw_fn)
        pw_f1 = safe_div(2 * pw_precision * pw_recall, pw_precision + pw_recall)
        pw_accuracy = safe_div(pw_tp + pw_tn, pw_tp + pw_tn + pw_fp + pw_fn)
        
        print("\n1. PATCH-WISE EVALUATION (Metodo del Professore)")
        print(f"   [Valuta l'immagine intera: Contiene frane? Sì/No]")
        print(f"   TP (Frane trovate correttamente): {pw_tp}")
        print(f"   FP (Falsi allarmi su sfondo):     {pw_fp}")
        print(f"   FN (Frane non viste):             {pw_fn}")
        print(f"   TN (Sfondi ignorati corrette.):   {pw_tn}")
        print(f"   --------------------")
        print(f"   Accuracy:  {pw_accuracy:.2%}")
        print(f"   Precision: {pw_precision:.2%}")
        print(f"   Recall:    {pw_recall:.2%}")
        print(f"   F1-Score:  {pw_f1:.2%}")
        
        # Print Pixel-Wise
        px_precision = safe_div(px_tp, px_tp + px_fp)
        px_recall = safe_div(px_tp, px_tp + px_fn)
        px_f1 = safe_div(2 * px_precision * px_recall, px_precision + px_recall)
        px_accuracy = safe_div(px_tp + px_tn, px_tp + px_tn + px_fp + px_fn)
        
        print("\n2. PIXEL-WISE EVALUATION (Nostro Metodo)")
        print(f"   [Valuta la segmentazione esatta di ogni singolo pixel]")
        print(f"   TP Pixel: {px_tp:,}")
        print(f"   FP Pixel: {px_fp:,}")
        print(f"   FN Pixel: {px_fn:,}")
        print(f"   TN Pixel: {px_tn:,}")
        print(f"   --------------------")
        print(f"   Accuracy:  {px_accuracy:.2%}")
        print(f"   Precision: {px_precision:.2%}")
        print(f"   Recall:    {px_recall:.2%}")
        print(f"   F1-Score:  {px_f1:.2%}")
        print("="*50)

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    evaluate_project()
