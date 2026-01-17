import cv2
import numpy as np
import os
import sys

def draw_ground_truth(image_path, label_path, output_path):
    # 1. Carica Immagine
    if not os.path.exists(image_path):
        print(f"Immagine non trovata: {image_path}")
        return

    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    # 2. Carica Etichette (YOLO format: class x1 y1 x2 y2 ... normalized)
    if not os.path.exists(label_path):
        print(f"Label non trovata: {label_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    mask_overlay = img.copy()
    
    found_labels = False
    for line in lines:
        parts = list(map(float, line.strip().split()))
        cls = int(parts[0])
        coords = parts[1:]
        
        # Converte coordinate normalizzate in pixel
        pts = []
        for i in range(0, len(coords), 2):
            x_pt = int(coords[i] * w)
            y_pt = int(coords[i+1] * h)
            pts.append([x_pt, y_pt])
            
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Disegna poligono (Verde per Ground Truth)
        cv2.fillPoly(mask_overlay, [pts], (0, 255, 0)) # Verde
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        found_labels = True

    # 3. Applica trasparenza
    alpha = 0.4
    cv2.addWeighted(mask_overlay, alpha, img, 1 - alpha, 0, img)
    
    # Scrivi testo
    cv2.putText(img, "GROUND TRUTH (VERDE)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 4. Salva
    cv2.imwrite(output_path, img)
    print(f"Salvato Ground Truth visuale in: {output_path}")

if __name__ == "__main__":
    # File specifici richiesti
    files_to_viz = [
        ("A19_5cm/val/images/Stigliano_5cm_patch_136_4096_4096.tif", "A19_5cm/val/labels/Stigliano_5cm_patch_136_4096_4096.txt", "visualizations/GT_Stigliano_136.jpg"),
        ("A19_5cm/val/images/Bella_5cm_patch_9_512_1024.tif", "A19_5cm/val/labels/Bella_5cm_patch_9_512_1024.txt", "visualizations/GT_Bella_9.jpg"),
        ("A19_5cm/val/images/Stigliano_5cm_patch_48_0_1536.tif", "A19_5cm/val/labels/Stigliano_5cm_patch_48_0_1536.txt", "visualizations/GT_Stigliano_48.jpg")
    ]
    
    if not os.path.exists("visualizations"):
        os.makedirs("visualizations")

    for img, lbl, out in files_to_viz:
        draw_ground_truth(img, lbl, out)
