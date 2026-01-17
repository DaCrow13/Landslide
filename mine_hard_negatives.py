import os
import shutil

def mine_hard_negatives():
    det_list_path = "visualizations/detected_list.txt"
    val_img_dir = "A19_5cm/val/images"
    val_lbl_dir = "A19_5cm/val/labels"
    
    train_img_dir = "A19_5cm/train/images"
    train_lbl_dir = "A19_5cm/train/labels"
    
    if not os.path.exists(det_list_path):
        print("Detected list not found.")
        return

    with open(det_list_path, 'r') as f:
        detections = [line.strip() for line in f.readlines() if line.strip()]
        
    print(f"Analisi {len(detections)} rilevamenti per mining...")
    
    moved_count = 0
    
    for filename in detections:
        # filename es: Stigliano...jpg
        base_name = os.path.splitext(filename)[0]
        label_name = base_name + ".txt"
        
        # Try finding the source image with .tif
        tif_name = base_name + ".tif"
        val_img_path = os.path.join(val_img_dir, tif_name)
        
        # If not found, try original name just in case
        if not os.path.exists(val_img_path):
            val_img_path = os.path.join(val_img_dir, filename)
        
        val_lbl_path = os.path.join(val_lbl_dir, label_name)
        
        # Check se è un FP (cioè label size == 0 o not exists)
        is_fp = False
        if not os.path.exists(val_lbl_path):
            is_fp = True
        elif os.path.getsize(val_lbl_path) == 0:
            is_fp = True
            
        if is_fp:
            # È un Hard Negative! Spostiamolo in Training.
            if os.path.exists(val_img_path):
                # Target name (keep extension)
                target_name = os.path.basename(val_img_path)
                
                # Move image
                shutil.move(val_img_path, os.path.join(train_img_dir, target_name))
                
                # Ensure empty label exists in train
                train_lbl_path = os.path.join(train_lbl_dir, label_name)
                with open(train_lbl_path, 'w') as f:
                    pass # Create empty file
                
                # Remove from val label if it existed (was empty anyway)
                if os.path.exists(val_lbl_path):
                    os.remove(val_lbl_path)
                    
                print(f"MOVED (Hard Negative): {target_name}")
                moved_count += 1
            else:
                # Only warn if checking extension failed
                pass
                
    print(f"-" * 30)
    print(f"Totale Hard Negatives spostati in Training: {moved_count}")

if __name__ == "__main__":
    mine_hard_negatives()
