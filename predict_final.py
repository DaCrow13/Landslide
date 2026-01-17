import os
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
import argparse

# --- Patching for Custom Model Loading ---
def patched_model_forward(self, x, *args, **kwargs):
    if hasattr(self, '_orig_forward'):
        return self._orig_forward(x, *args, **kwargs)
    return super(SegmentationModel, self).forward(x, *args, **kwargs)

def global_feature_hook(*args, **kwargs): pass

SegmentationModel.patched_model_forward = patched_model_forward
SegmentationModel.global_feature_hook = global_feature_hook
# -----------------------------------------

def run_prediction(source, output_dir="predictions_output", conf=0.10):
    """
    Runs the Robust Landslide Detection Model with Post-Processing Filters.
    
    Args:
        source (str): Path to image or directory of images.
        output_dir (str): Directory to save results.
        conf (float): Confidence threshold (default 0.10 for high sensitivity).
    """
    
    # 1. Configuration
    MODEL_PATH = "final_model.pt"
    MIN_AREA = 300        # Minimum pixel area to keep (removes small noise)
    MAX_LINEARITY = 4.0   # Maximum aspect ratio (removes roads)
    
    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model '{MODEL_PATH}' not found!")
        return
        
    print(f"Loading Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # 3. Prepare Output
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Predict
    print(f"Running inference on '{source}' with Conf={conf}...")
    results = model.predict(source=source, conf=conf, verbose=False, stream=True)
    
    processed_count = 0
    saved_count = 0
    
    for r in results:
        processed_count += 1
        filename = os.path.basename(r.path)
        base_name = os.path.splitext(filename)[0]
        
        img = r.orig_img
        if img is None: continue
        
        h, w = img.shape[:2]
        
        # Build Raw Mask
        raw_mask = np.zeros((h, w), dtype=np.uint8)
        if r.masks is not None:
             for mask in r.masks.xy:
                pts = np.array(mask, dtype=np.int32)
                cv2.fillPoly(raw_mask, [pts], 1)
                
        # --- Apply Filters ---
        filtered_mask = np.zeros((h, w), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
        
        detections_kept = 0
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Area Filter
            if area < MIN_AREA:
                continue 
                
            # Linearity Filter
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
                filtered_mask[labels == i] = 1
                detections_kept += 1
                
        # Save Result if detections exist
        if detections_kept > 0:
            # Draw Red Contours on Image
            viz_img = img.copy()
            contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(viz_img, contours, -1, (0, 0, 255), 3) # Red BGR
            
            save_path = os.path.join(output_dir, f"pred_{filename}")
            cv2.imwrite(save_path, viz_img)
            saved_count += 1
            print(f"Saved: {save_path} ({detections_kept} landslides)")
            
    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Total Images Scanned: {processed_count}")
    print(f"Images with Landslides Found: {saved_count}")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    # Default behavior: run on validation set validation
    # User can edit this call to point to new data
    target_source = "A19_5cm/val/images"
    run_prediction(target_source)
