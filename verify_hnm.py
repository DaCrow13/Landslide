import os
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel

# --- Patching ---
def patched_model_forward(self, x, *args, **kwargs):
    if hasattr(self, '_orig_forward'):
        return self._orig_forward(x, *args, **kwargs)
    return super(SegmentationModel, self).forward(x, *args, **kwargs)

def global_feature_hook(*args, **kwargs): pass

SegmentationModel.patched_model_forward = patched_model_forward
SegmentationModel.global_feature_hook = global_feature_hook
# ----------------

def verify_hard_negatives():
    model_path = "runs/fine_tune_hnm/train/weights/best.pt"
    train_img_dir = "A19_5cm/train/images"
    train_lbl_dir = "A19_5cm/train/labels"
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # 1. Identify likely Hard Negatives
    # They are files in train folder that map to empty label files
    # AND likely match the naming convention we saw earlier (Stigliano_...)
    
    candidates = []
    
    for f in os.listdir(train_img_dir):
        if not f.endswith(('.jpg', '.tif')):
            continue
            
        base_name = os.path.splitext(f)[0]
        label_path = os.path.join(train_lbl_dir, base_name + ".txt")
        
        is_empty = False
        if not os.path.exists(label_path):
            is_empty = True
        elif os.path.getsize(label_path) == 0:
            is_empty = True
            
        if is_empty:
            candidates.append(os.path.join(train_img_dir, f))
            
    print(f"Found {len(candidates)} empty samples in training set (Potential Hard Negatives).")
    print("Testing a subset (checking if they trigger detections)...")
    
    # Predict
    results = model.predict(source=candidates, conf=0.05, stream=True, verbose=False)
    
    failures = 0
    passed = 0
    
    for r in results:
        fname = os.path.basename(r.path)
        if r.masks is not None and len(r.masks) > 0:
            print(f"[FAIL] Detection on empty file: {fname} ({len(r.masks)} masks)")
            failures += 1
        else:
            # print(f"[PASS] No detection on: {fname}")
            passed += 1
            
    total = failures + passed
    print("-" * 30)
    print(f"Total Tested: {total}")
    print(f"Correctly Ignored (PASS): {passed}")
    print(f"Incorrectly Detected (FAIL): {failures}")
    
    if total > 0:
        success_rate = passed / total
        print(f"Suppression Success Rate: {success_rate:.1%}")

if __name__ == "__main__":
    verify_hard_negatives()
