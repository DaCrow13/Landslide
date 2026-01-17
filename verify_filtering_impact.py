import os
import cv2
import numpy as np
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

def verify_impact():
    model_path = "final_model.pt"
    source_dir = "A19_5cm/val/images"
    conf_threshold = 0.10
    
    # Filter Params
    MIN_AREA = 300
    MAX_LINEARITY = 4.0
    
    model = YOLO(model_path)
    images = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.tif'))]
    
    print(f"Verifying Filtering Impact on {len(images)} images...")
    
    total_detections_raw = 0
    total_detections_kept = 0
    removed_small = 0
    removed_linear = 0
    
    for filename in images:
        img_path = os.path.join(source_dir, filename)
        results = model.predict(source=img_path, conf=conf_threshold, verbose=False)
        r = results[0]
        
        if r.masks is None:
            continue
            
        # Get raw polygons to count individual detections
        # r.masks.xy is list of arrays
        
        for mask_poly in r.masks.xy:
            total_detections_raw += 1
            
            # Reconstruct mask for this poly to measure props
            # (Approximation: fitEllipse needs points, area needs contour)
            pts = np.array(mask_poly, dtype=np.int32)
            
            # Area
            area = cv2.contourArea(pts)
            if area < MIN_AREA:
                removed_small += 1
                continue
                
            # Linearity
            is_linear = False
            if len(pts) >= 5:
                (x,y), (MA, ma), angle = cv2.fitEllipse(pts)
                major = max(MA, ma)
                minor = min(MA, ma)
                if minor > 0 and (major/minor) > MAX_LINEARITY:
                    removed_linear += 1
                    is_linear = True
            
            if not is_linear:
                total_detections_kept += 1

    print("-" * 30)
    print("FILTERING IMPACT REPORT")
    print(f"Total Raw Detections (@ 0.10): {total_detections_raw}")
    print(f"Removed (Small Noise < {MIN_AREA}px): {removed_small}")
    print(f"Removed (Roads/Linear > {MAX_LINEARITY}): {removed_linear}")
    print(f"Final Valid Detections: {total_detections_kept}")
    print(f"Noise Reduction Rate: {(total_detections_raw - total_detections_kept) / total_detections_raw * 100:.1f}%")

if __name__ == "__main__":
    verify_impact()
