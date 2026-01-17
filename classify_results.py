import os
import shutil

def classify_detections(detection_list_path, label_dir, output_dir):
    with open(detection_list_path, 'r') as f:
        files = [line.strip() for line in f.readlines() if line.strip()]
        
    tp_dir = os.path.join(output_dir, "True_Positives")
    fp_dir = os.path.join(output_dir, "False_Positives")
    
    if not os.path.exists(tp_dir): os.makedirs(tp_dir)
    if not os.path.exists(fp_dir): os.makedirs(fp_dir)
        
    print(f"Classificazione di {len(files)} rilevazioni...")
    
    tp_count = 0
    fp_count = 0
    
    for filename in files:
        # Converti nome immagine in nome label
        label_name = filename.replace('.jpg', '.txt').replace('.tif', '')
        # A volte l'estensione originale è nel nome se lo script di visualizzazione l'ha lasciata
        # I file in detected_list.txt sono tipo "Nome_patch.jpg"
        # Le label sono in "Nome_patch.txt"
        
        # Facciamo un match più robusto
        base_name = os.path.splitext(filename)[0] # Via .jpg
        label_path = os.path.join(label_dir, base_name + ".txt")
        
        # Se il file non esiste, proviamo ad aggiungere .tif prima di togliere .jpg?
        # No, le label sono Nome.txt.
        
        is_true_positive = False
        if os.path.exists(label_path):
            if os.path.getsize(label_path) > 0:
                is_true_positive = True
        
        # Copia l'immagine di confronto se esiste, altrimenti quella di visualizzazione
        # Cerchiamo prima in comparisons perché è più chiara
        comp_src = os.path.join("visualizations", "comparisons", f"COMPARE_{base_name}.jpg")
        viz_src = os.path.join("visualizations", "visual_results", filename)
        
        target_src = comp_src if os.path.exists(comp_src) else viz_src
        
        if is_true_positive:
            shutil.copy(target_src, os.path.join(tp_dir, filename))
            tp_count += 1
        else:
            shutil.copy(target_src, os.path.join(fp_dir, filename))
            fp_count += 1
            
    print(f"Classificazione Completata.")
    print(f"Veri Positivi (Successi): {tp_count}")
    print(f"Falsi Positivi (Allucinazioni): {fp_count}")
    print(f"Cartella TP: {tp_dir}")
    print(f"Cartella FP: {fp_dir}")

if __name__ == "__main__":
    DET_LIST = "visualizations/detected_list.txt"
    LBL_DIR = "A19_5cm/val/labels"
    OUT_DIR = "visualizations/classified_results"
    
    classify_detections(DET_LIST, LBL_DIR, OUT_DIR)
