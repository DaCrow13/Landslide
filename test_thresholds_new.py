import torch
import os
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel

# --- Patching indispensabile ---
def patched_model_forward(self, x, *args, **kwargs):
    if hasattr(self, '_orig_forward'):
        return self._orig_forward(x, *args, **kwargs)
    return super(SegmentationModel, self).forward(x, *args, **kwargs)

def global_feature_hook(*args, **kwargs): pass

SegmentationModel.patched_model_forward = patched_model_forward
SegmentationModel.global_feature_hook = global_feature_hook
# -------------------------------

def test_thresholds_new_model():
    # Use the NEW fine-tuned model
    model_path = "runs/fine_tune_hnm/train/weights/best.pt"
    source_dir = "A19_5cm/val/images"
    thresholds = [0.05, 0.15, 0.25]
    
    print(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Pre-scan to identify Files with Ground Truth
    label_dir = "A19_5cm/val/labels"
    gt_files = set()
    if os.path.exists(label_dir):
        for f in os.listdir(label_dir):
            if f.endswith('.txt') and os.path.getsize(os.path.join(label_dir, f)) > 0:
                gt_files.add(f.replace('.txt', ''))

    print(f"Found {len(gt_files)} ground truth files for reference.")
    print("-" * 50)
    
    print(f"{'Conf':<10} | {'Total':<10} | {'TP':<10} | {'FP':<10} | {'Precision':<10}")

    for conf in thresholds:
        results = model.predict(source=source_dir, conf=conf, stream=True, verbose=False)
        
        detections = []
        tp_count = 0
        fp_count = 0
        
        for r in results:
            if r.masks is not None and len(r.masks) > 0:
                filename = os.path.basename(r.path)
                detections.append(filename)
                
                # Check rough TP/FP status
                # If file in GT set -> It's a TP (contains at least one landslide) 
                # (Simplification: assumes checking "presence" not IoU)
                base_name = os.path.splitext(filename)[0]
                if base_name in gt_files:
                    tp_count += 1
                else:
                    fp_count += 1
        
        total = len(detections)
        prec = (tp_count / total * 100) if total > 0 else 0
        print(f"{conf:<10} | {total:<10} | {tp_count:<10} | {fp_count:<10} | {prec:.1f}%")

if __name__ == "__main__":
    test_thresholds_new_model()
