from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
import os

# --- Patching ---
def patched_model_forward(self, x, *args, **kwargs):
    if hasattr(self, '_orig_forward'):
        return self._orig_forward(x, *args, **kwargs)
    return super(SegmentationModel, self).forward(x, *args, **kwargs)

def global_feature_hook(*args, **kwargs): pass

SegmentationModel.patched_model_forward = patched_model_forward
SegmentationModel.global_feature_hook = global_feature_hook
# ----------------

def test_single():
    model = YOLO("runs/fine_tune_hnm/train/weights/best.pt")
    # This file was identified as a FP and moved to train
    test_file = "A19_5cm/train/images/Stigliano_5cm_patch_2_1024_0.tif" 
    
    if not os.path.exists(test_file):
        print(f"File not found: {test_file}")
        return
        
    results = model.predict(test_file, conf=0.05)
    for r in results:
        if r.masks:
            print(f"STILL DETECTED: {len(r.masks)} masks found.")
        else:
            print("CLEAN: No detections (Success!).")

if __name__ == "__main__":
    test_single()
