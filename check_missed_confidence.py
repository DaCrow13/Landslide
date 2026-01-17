import os
from ultralytics import YOLO
import numpy as np

def check_confidence():
    model_path = "final_model.pt"
    source_dir = "A19_5cm/val/images"
    label_dir = "A19_5cm/val/labels"
    
    print(f"Checking latent confidence with model: {model_path}")
    model = YOLO(model_path)
    
    files = sorted([f for f in os.listdir(source_dir) if "Bella" in f])
    
    print(f"Scanning {len(files)} Bella files for hidden detections...")
    
    recoverable = 0
    total_gt = 0
    
    for f in files:
        img_path = os.path.join(source_dir, f)
        lbl_path = os.path.join(label_dir, os.path.splitext(f)[0] + ".txt")
        
        if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
            continue
            
        total_gt += 1
        
        # Predict with very low conf to capture everything
        results = model.predict(img_path, conf=0.01, verbose=False)[0]
        
        max_conf = 0.0
        if results.boxes:
            # check box confidences
            max_conf = results.boxes.conf.max().item()
            
        print(f"File: {f} | GT: Yes | Max Conf: {max_conf:.4f}")
        
        if max_conf >= 0.10: # If we lowered threshold to 0.10, would we find it?
            recoverable += 1
            
    print("-" * 30)
    print(f"Total Bella GT Files: {total_gt}")
    print(f"Recoverable with Conf >= 0.10: {recoverable}")

if __name__ == "__main__":
    check_confidence()
