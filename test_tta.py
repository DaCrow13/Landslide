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

def test_tta_recovery():
    model_path = "final_model.pt"
    source_dir = "A19_5cm/val/images"
    label_dir = "A19_5cm/val/labels"
    # We focus on the files identified as misses previously
    # For simplicity, we scan all and check recall improvement
    
    print(f"Testing TTA Recovery with model: {model_path}")
    model = YOLO(model_path)
    
    # Identify misses first (Standard inference)
    print("Step 1: Identifying Baseline Misses...")
    all_files = sorted([f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.tif'))])
    
    missed_files = []
    
    # Fast check of what is currently missed
    results_std = model.predict(source=source_dir, conf=0.25, verbose=False, stream=True)
    
    # We need to match result to filename carefully since stream might not preserve order if we didn't sort input?
    # YOLO stream is iterator. Usually order is preserved if source is list.
    # Let's be safe and iterate manually or trust list source.
    
    # Actually, let's just loop over the known likely misses (Bella files) to save time?
    # No, let's do a proper check on a subset or all.
    # Let's check ALL "Bella" files since that's where the problem is.
    
    bella_files = [f for f in all_files if "Bella" in f]
    print(f"Focusing on {len(bella_files)} Bella files...")
    
    baseline_recovered = 0
    tta_recovered = 0
    total_gt = 0
    
    for f in bella_files:
        img_path = os.path.join(source_dir, f)
        lbl_path = os.path.join(label_dir, os.path.splitext(f)[0] + ".txt")
        
        # Check if GT exists
        if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
            continue
            
        total_gt += 1
        
        # Standard Prediction
        r_std = model.predict(img_path, conf=0.25, verbose=False)[0]
        if r_std.masks is not None:
            baseline_recovered += 1
        
        # TTA Prediction
        # augment=True enables TTA (scale, flip)
        r_tta = model.predict(img_path, conf=0.25, augment=True, verbose=False)[0]
        if r_tta.masks is not None:
            tta_recovered += 1
            
    print("-" * 30)
    print(f"Total Bella Files with GT: {total_gt}")
    print(f"Detected WITHOUT TTA: {baseline_recovered}")
    print(f"Detected WITH TTA: {tta_recovered}")
    
    improvement = tta_recovered - baseline_recovered
    print(f"Recovered by TTA: {improvement}")

if __name__ == "__main__":
    test_tta_recovery()
