import torch
import os
import sys

# 1. Iniezione dinamica per compatibilità unpickling
from ultralytics.nn.tasks import SegmentationModel

def patched_model_forward(self, x, *args, **kwargs):
    if hasattr(self, '_orig_forward'):
        return self._orig_forward(x, *args, **kwargs)
    return super(SegmentationModel, self).forward(x, *args, **kwargs)

def global_feature_hook(*args, **kwargs): pass

# Attacchiamo i metodi alla classe per farli trovare all'unpickler
SegmentationModel.patched_model_forward = patched_model_forward
SegmentationModel.global_feature_hook = global_feature_hook

from ultralytics import YOLO
import contrastive_utils # Per sicurezza importiamo anche il modulo utility

def predict_landslides(model_path, source_dir, output_dir):
    """
    Esegue l'inferenza utilizzando il modello YOLOv8-seg e salva i risultati.
    """
    if not os.path.exists(model_path):
        print(f"Errore: Modello non trovato in {model_path}")
        return

    # Crea la directory di output se non esiste
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Caricamento modello da: {model_path}")
    
    try:
        # Tentativo di caricamento standard
        model = YOLO(model_path)
    except Exception as e:
        print(f"Errore durante il caricamento: {e}")
        print("Il modello salvato contiene riferimenti a funzioni non più presenti.")
        print("Consiglio: Avvia un nuovo addestramento con il codice aggiornato.")
        return

    # Esegue la predizione
    print(f"Inizio predizione su: {source_dir}")
    results = model.predict(
        source=source_dir,
        conf=0.05,
        save=True,
        project=output_dir,
        name='visual_results',
        exist_ok=True
    )
    print(f"Predizione completata in: {output_dir}/visual_results")

if __name__ == "__main__":
    DEFAULT_MODEL = "best_clean.pt"
    DEFAULT_SOURCE = "A19_5cm/val/images"
    DEFAULT_OUTPUT = "visualizations"

    model_p = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    source_p = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SOURCE
    output_p = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUTPUT

    predict_landslides(model_p, source_p, output_p)
