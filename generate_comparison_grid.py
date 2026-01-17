import cv2
import numpy as np
import os
import glob

# Costanti per i colori
COLOR_GT = (0, 255, 0)      # Verde
COLOR_PRED = (255, 0, 0)    # Blu (BGR)

def load_yolo_polygon(label_path, w, h):
    """Legge un file label YOLO e restituisce una lista di poligoni pixel."""
    if not os.path.exists(label_path):
        return []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    polygons = []
    for line in lines:
        parts = list(map(float, line.strip().split()))
        coords = parts[1:] # Ignora class ID
        pts = []
        for i in range(0, len(coords), 2):
            pts.append([int(coords[i]*w), int(coords[i+1]*h)])
        polygons.append(np.array(pts, np.int32).reshape((-1, 1, 2)))
    return polygons

def create_comparison_grid(image_name, base_dir, output_dir):
    """Crea una griglia 1x3: Originale | GT | Predizione"""
    
    # Percorsi
    img_path = os.path.join(base_dir, "images", image_name)
    label_path = os.path.join(base_dir, "labels", image_name.replace('.tif', '.txt').replace('.jpg', '.txt'))
    
    # Questo è un po' tricky: dobbiamo caricare la predizione salvata da YOLO
    # o ridisegnarla noi se abbiamo i dati. 
    # Per semplicità, carichiamo la predizione già generata da `visual_results` se esiste,
    # altrimenti usiamo l'originale per la colonna "Predizione" (che sarà vuota se non rilevata)
    pred_viz_path = os.path.join("visualizations", "visual_results", image_name.replace('.tif', '.jpg'))
    
    if not os.path.exists(img_path):
        print(f"Skip: {img_path} not found")
        return

    # 1. Immagine Originale
    img_orig = cv2.imread(img_path)
    if img_orig is None: return
    h, w, _ = img_orig.shape
    
    # Resize per leggibilità se troppo grande
    TARGET_H = 512
    scale = TARGET_H / h
    TARGET_W = int(w * scale)
    
    img_orig = cv2.resize(img_orig, (TARGET_W, TARGET_H))
    
    # 2. Ground Truth Image
    img_gt = img_orig.copy()
    polys_gt = load_yolo_polygon(label_path, TARGET_W, TARGET_H)
    
    if polys_gt:
        overlay = img_gt.copy()
        for poly in polys_gt:
            cv2.fillPoly(overlay, [poly], COLOR_GT)
            cv2.polylines(img_gt, [poly], True, COLOR_GT, 2)
        cv2.addWeighted(overlay, 0.3, img_gt, 0.7, 0, img_gt)
    
    # 3. Prediction Image
    # Se abbiamo già l'immagine generata da YOLO predict, usiamola
    if os.path.exists(pred_viz_path):
        img_pred = cv2.imread(pred_viz_path)
        img_pred = cv2.resize(img_pred, (TARGET_W, TARGET_H))
    else:
        # Se non c'è, magari è perché non è stata rilevata (o file name mismatch)
        img_pred = img_orig.copy()
        cv2.putText(img_pred, "No Detection", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Aggiungi titoli
    def add_title(img, text, color=(255, 255, 255)):
        # Striscia nera in alto
        bar = np.zeros((40, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return np.vstack([bar, img])

    img_orig_viz = add_title(img_orig, "1. Immagine Satellitare")
    img_gt_viz = add_title(img_gt, "2. Realtà (Ground Truth)", COLOR_GT)
    img_pred_viz = add_title(img_pred, "3. Modello (Predizione)", (255, 100, 0)) # Bluastro

    # Concatena orizzontalmente
    grid = np.hstack([img_orig_viz, img_gt_viz, img_pred_viz])
    
    out_name = os.path.join(output_dir, f"COMPARE_{image_name.replace('.tif', '')}.jpg")
    cv2.imwrite(out_name, grid)
    print(f"Generato confronto: {out_name}")

if __name__ == "__main__":
    BASE_VAL = "A19_5cm/val"
    OUT_DIR = "visualizations/comparisons"
    
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        
    # Lista dei True Positives identificati
    files_of_interest = [
        "Bella_5cm_patch_9_512_1024.tif",
        "Stigliano_5cm_patch_116_2048_3584.tif",
        "Stigliano_5cm_patch_119_3584_3584.tif",
        "Stigliano_5cm_patch_133_2560_4096.tif",
        "Stigliano_5cm_patch_134_3072_4096.tif",
        "Stigliano_5cm_patch_135_3584_4096.tif",
        "Stigliano_5cm_patch_136_4096_4096.tif",
        "Stigliano_5cm_patch_137_4608_4096.tif",
        "Stigliano_5cm_patch_149_2560_4608.tif",
        "Stigliano_5cm_patch_150_3072_4608.tif",
        "Stigliano_5cm_patch_151_3584_4608.tif",
        "Stigliano_5cm_patch_152_4096_4608.tif",
        "Stigliano_5cm_patch_153_4608_4608.tif",
        "Stigliano_5cm_patch_168_4096_5120.tif",
        "Stigliano_5cm_patch_183_3584_5632.tif",
        "Stigliano_5cm_patch_88_4096_2560.tif"
    ]
    
    print("Generazione griglie di confronto...")
    for f in files_of_interest:
        create_comparison_grid(f, BASE_VAL, OUT_DIR)
