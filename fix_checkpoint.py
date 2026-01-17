import torch
from ultralytics import YOLO
import sys

# Crucial: Define the missing attribute in the class itself so the unpickler doesn't complain
from ultralytics.nn.tasks import SegmentationModel

# Define a fallback so unpickler finds something
def patched_model_forward(self, *args, **kwargs):
    return self._orig_forward(*args, **kwargs)

SegmentationModel.patched_model_forward = patched_model_forward

def clean_model(input_path, output_path):
    print(f"Tentativo di pulizia modello: {input_path}")
    
    try:
        # Load the checkpoint
        ckpt = torch.load(input_path, map_location='cpu')
        
        # Extract the model object
        model_obj = ckpt['model']
        
        # Remove the monkeypatched forward if it exists
        if hasattr(model_obj, '_orig_forward'):
            print("Ripristino forward originale...")
            model_obj.forward = model_obj._orig_forward
            del model_obj._orig_forward
            
        # Remove any other contrastive attributes to be safe
        for attr in ['contrastive_heads', 'contrast_fn', 'contrast_weight', 'patched_model_forward']:
            if hasattr(model_obj, attr):
                # We don't delete them if they are modules needed for state_dict, 
                # but they aren't standard YOLO so we should probably clear them.
                pass

        # Save a clean version
        # Ultralytics YOLO.save() expects a YOLO object.
        # Let's just save the ckpt dict again but "cleaned"
        ckpt['model'] = model_obj
        torch.save(ckpt, output_path)
        print(f"Modello pulito salvato in: {output_path}")
        
    except Exception as e:
        print(f"Errore critico durante la pulizia: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    IN = "runs/yolov8s-seg_contrastive/A19_5cm/train47/weights/best.pt"
    OUT = "best_clean.pt"
    clean_model(IN, OUT)
