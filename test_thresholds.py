import torch
import os
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel

# --- Patching (same as scan_results) ---
def patched_model_forward(self, x, *args, **kwargs):
    if hasattr(self, '_orig_forward'):
        return self._orig_forward(x, *args, **kwargs)
    return super(SegmentationModel, self).forward(x, *args, **kwargs)

def global_feature_hook(*args, **kwargs): pass

SegmentationModel.patched_model_forward = patched_model_forward
SegmentationModel.global_feature_hook = global_feature_hook
# ---------------------------------------

def test_thresholds():
    model_path = "best_clean.pt"
    source_dir = "A19_5cm/val/images"
    thresholds = [0.05, 0.15, 0.25, 0.40, 0.60]
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Pre-scan to identify Files with Ground Truth
    label_dir = "A19_5cm/val/labels"
    gt_files = set()
    if os.path.exists(label_dir):
        for f in os.listdir(label_dir):
            if f.endswith('.txt') and os.path.getsize(os.path.join(label_dir, f)) > 0:
                gt_files.add(f.replace('.txt', '')) # Base name only (no extension)

    print(f"Found {len(gt_files)} ground truth files for reference.")
    print("-" * 50)
    
    results_summary = {}

    for conf in thresholds:
        print(f"Testing confidence: {conf}")
        results = model.predict(source=source_dir, conf=conf, stream=True, verbose=False)
        
        detections = []
        tp_count = 0
        fp_count = 0
        
        for r in results:
            if r.masks is not None and len(r.masks) > 0:
                filename = os.path.basename(r.path)
                detections.append(filename)
                
                # Check rough TP/FP status
                base_name = os.path.splitext(filename)[0]
                if base_name in gt_files:
                    tp_count += 1
                else:
                    fp_count += 1
        
        results_summary[conf] = {
            'total': len(detections),
            'tp': tp_count,
            'fp': fp_count
        }
        print(f"  -> Total: {len(detections)} | Possible TPs: {tp_count} | Likely FPs: {fp_count}")

    print("-" * 50)
    print("Summary Table:")
    print(f"{'Conf':<10} | {'Total':<10} | {'Est. TPs':<10} | {'Est. FPs':<10} | {'Precision':<10}")
    for conf in thresholds:
        d = results_summary[conf]
        prec = (d['tp'] / d['total']) * 100 if d['total'] > 0 else 0
        print(f"{conf:<10} | {d['total']:<10} | {d['tp']:<10} | {d['fp']:<10} | {prec:.1f}%")

if __name__ == "__main__":
    test_thresholds()
