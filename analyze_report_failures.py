import os
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
import traceback

# --- Patching ---
def patched_model_forward(self, x, *args, **kwargs):
    if hasattr(self, '_orig_forward'):
        return self._orig_forward(x, *args, **kwargs)
    return super(SegmentationModel, self).forward(x, *args, **kwargs)

def global_feature_hook(*args, **kwargs): pass

SegmentationModel.patched_model_forward = patched_model_forward
SegmentationModel.global_feature_hook = global_feature_hook
# ----------------

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def analyze_failures():
    try:
        model_path = "final_model.pt"
        source_dir = "A19_5cm/val/images"
        label_dir = "A19_5cm/val/labels"
        conf_threshold = 0.25
        
        print(f"Analyzing failures using model: {model_path}")
        model = YOLO(model_path)
        
        images = sorted([f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.tif'))])
        
        hallucinations = []
        misses = []
        misalignments = []
        
        print(f"Scanning {len(images)} images...")
        
        for filename in images:
            try:
                img_path = os.path.join(source_dir, filename)
                base_name = os.path.splitext(filename)[0]
                
                # Predict
                results = model.predict(source=img_path, conf=conf_threshold, verbose=False)
                r = results[0]
                
                if r.masks is None and not os.path.exists(os.path.join(label_dir, base_name + ".txt")):
                    continue # Both empty, skip
                    
                h, w = r.orig_shape
                
                # GT Mask
                gt_mask = np.zeros((h, w), dtype=np.uint8)
                label_path = os.path.join(label_dir, base_name + ".txt")
                has_gt = False
                if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                    has_gt = True
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
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

                # Pred Mask
                pred_mask = np.zeros((h, w), dtype=np.uint8)
                has_pred = False
                if r.masks is not None:
                    has_pred = True
                    for mask in r.masks.xy:
                        if len(mask) > 0: # Check empty segment
                            pts = np.array(mask, dtype=np.int32)
                            cv2.fillPoly(pred_mask, [pts], 1)
                
                # Analyze
                if has_pred and not has_gt:
                    hallucinations.append(filename)
                elif has_gt and not has_pred:
                    misses.append(filename)
                elif has_gt and has_pred:
                    iou = compute_iou(gt_mask, pred_mask)
                    if iou < 0.1:
                        misalignments.append((filename, iou))
                        
            except Exception as inner_e:
                print(f"Error processing {filename}: {inner_e}")
                continue

        print("-" * 30)
        print("ANALYSIS RESULTS (Worst Offenders):")
        print(f"Total Images: {len(images)}")
        print(f"PURE HALLUCINATIONS (Pred but no GT): {len(hallucinations)}")
        if len(hallucinations) > 0:
            print(f"  Examples: {hallucinations[:5]}")
            
        print(f"COMPLETE MISSES (GT but no Pred): {len(misses)}")
        if len(misses) > 0:
            print(f"  Examples: {misses[:5]}")
            
        print(f"MAJOR MISALIGNMENTS (IoU < 0.1): {len(misalignments)}")
        if len(misalignments) > 0:
            print(f"  Examples: {misalignments[:5]}")
            
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    analyze_failures()
