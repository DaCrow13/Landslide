import torch
import os
import sys
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel

# --- Patching per compatibilità (come in predict_contrastive) ---
def patched_model_forward(self, x, *args, **kwargs):
    if hasattr(self, '_orig_forward'):
        return self._orig_forward(x, *args, **kwargs)
    return super(SegmentationModel, self).forward(x, *args, **kwargs)

def global_feature_hook(*args, **kwargs): pass

SegmentationModel.patched_model_forward = patched_model_forward
SegmentationModel.global_feature_hook = global_feature_hook
# -------------------------------------------------------------

def scan_for_detections(model_path, source_dir):
    print(f"Scansione rilevamenti con modello: {model_path}")
    model = YOLO(model_path)
    
    # Esegue predizione stream (senza salvare file, solo in memoria)
    results = model.predict(source=source_dir, conf=0.05, stream=True, verbose=False)
    
    detected_files = []
    
    print("\nFile con frane rilevate:")
    for r in results:
        # Se ci sono box o maschere
        if r.masks is not None and len(r.masks) > 0:
            filename = os.path.basename(r.path)
            # Salviamo il nome file e il numero di oggetti trovati
            count = len(r.masks)
            print(f"- {filename} ({count} frane)")
            detected_files.append(filename)
            
    if not detected_files:
        print("\nNessuna frana rilevata in nessun file.")
    else:
        print(f"\nTotale immagini con rilevamenti: {len(detected_files)}")

if __name__ == "__main__":
    MODEL = "best_clean.pt"
    SOURCE = "A19_5cm/val/images"
    scan_for_detections(MODEL, SOURCE)
