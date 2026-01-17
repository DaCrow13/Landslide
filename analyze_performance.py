import os

def analyze():
    det_path = "visualizations/detected_list.txt"
    label_dir = "A19_5cm/val/labels"
    
    if not os.path.exists(det_path):
        print(f"File not found: {det_path}")
        return

    with open(det_path, 'r') as f:
        detections = [line.strip() for line in f.readlines() if line.strip()]

    tps = 0
    fps = 0
    
    print(f"Total detections: {len(detections)}")
    
    for det in detections:
        # detection: Name.jpg
        base_name = os.path.splitext(det)[0]
        label_name = base_name + ".txt"
        label_path = os.path.join(label_dir, label_name)
        
        is_tp = False
        status = "FP"
        reason = "No label file"
        
        if os.path.exists(label_path):
            size = os.path.getsize(label_path)
            if size > 0:
                is_tp = True
                status = "TP"
                reason = f"Label size: {size}"
            else:
                reason = "Label file empty"
        
        print(f"{det} -> {status} ({reason})")
        
        if is_tp:
            tps += 1
        else:
            fps += 1
            
    precision = tps / len(detections) if len(detections) > 0 else 0
    print("-" * 30)
    print(f"True Positives: {tps}")
    print(f"False Positives: {fps}")
    print(f"Precision: {precision:.2%}")

if __name__ == "__main__":
    analyze()
