from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import SegmentationModel

# --- Patching indispensabile per caricare il modello custom ---
def patched_model_forward(self, x, *args, **kwargs):
    if hasattr(self, '_orig_forward'):
        return self._orig_forward(x, *args, **kwargs)
    return super(SegmentationModel, self).forward(x, *args, **kwargs)

def global_feature_hook(*args, **kwargs): pass

SegmentationModel.patched_model_forward = patched_model_forward
SegmentationModel.global_feature_hook = global_feature_hook
# -----------------------------------------------------------

def fine_tune():
    # Load previous best model
    # Il patching sopra deve avvenire PRIMA di caricare il modello
    model = YOLO("best_clean.pt")
    
    print("Avvio Fine-Tuning con Hard Negative Mining (Ritentativo)...")
    
    model.train(
        data="yaml/A19_5cm.yaml", 
        project="runs/fine_tune_hnm",
        epochs=15, 
        imgsz=512, 
        batch=16, 
        lr0=0.001,
        lrf=0.01,
        device=0 if torch.cuda.is_available() else 'cpu',
        exist_ok=True,
        plots=True
    )
    
    print("Fine-Tuning completato.")

if __name__ == "__main__":
    fine_tune()
