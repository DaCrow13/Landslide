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

def verify_robust():
    model_path = "runs/robust_training/train/weights/best.pt"
    train_img_dir = "A19_5cm/train/images"
    train_lbl_dir = "A19_5cm/train/labels"
    
    print(f"Loading model: {model_path}")
    if not os.path.exists(model_path):
        print("Model file not found!")
        return

    model = YOLO(model_path)
    
    # Identify Hard Negatives (Empty/Missing label files in Train)
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
            
    print(f"Found {len(candidates)} empty samples in training set.")
    print("Testing specific known FP: Stigliano_5cm_patch_2_1024_0.tif (if present)...")
    
    # Specific check
    target_fp = os.path.join(train_img_dir, "Stigliano_5cm_patch_2_1024_0.tif")
    if target_fp in candidates:
        res = model.predict(target_fp, conf=0.15, verbose=False) # Check at reasonable conf
        for r in res:
            if r.masks:
                print(f"[FAIL] Specific Target {os.path.basename(target_fp)} STILL DETECTED!")
            else:
                print(f"[PASS] Specific Target {os.path.basename(target_fp)} CORRECTLY IGNORED.")
    else:
        print("Specific target not found in candidates list.")

    print("-" * 30)
    print("Testing random sample of 100 empty files...")
    import random
    if len(candidates) > 100:
        subset = random.sample(candidates, 100)
    else:
        subset = candidates
        
    failures = 0
    
    results = model.predict(source=subset, conf=0.15, stream=True, verbose=False)
    for r in results:
        if r.masks:
            failures += 1
            
    total = len(subset)
    passed = total - failures
    
    print(f"Total Tested: {total}")
    print(f"Correctly Ignored (PASS): {passed}")
    print(f"Incorrectly Detected (FAIL): {failures}")
    print(f"Suppression Rate: {passed/total:.1%}")

if __name__ == "__main__":
    verify_robust()
