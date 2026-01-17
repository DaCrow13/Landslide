from ultralytics import YOLO
import torch

def train_robust():
    # Start from base COCO model to learn the new balance cleanly
    # (Avoids inheriting the 'confusion' of previous models)
    model = YOLO("yolov8s-seg.pt")
    
    print("Avvio Training Robusto (da zero)...")
    
    model.train(
        data="yaml/A19_5cm.yaml", 
        project="runs/robust_training",
        epochs=50,          # More epochs for proper convergence
        patience=15,        # Early stopping if no improvement
        imgsz=640,          # Higher resolution (helps with small landslides)
        batch=16, 
        device=0 if torch.cuda.is_available() else 'cpu',
        exist_ok=True,
        plots=True,
        
        # Hyperparameters for robustness
        lr0=0.01,           # Standard initial learning rate
        lrf=0.01,           # Final learning rate
        dropout=0.1,        # Add dropout to prevent overfitting
        weight_decay=0.0005,
        
        # Augmentations (Standard YOLOv8 is already good, but let's ensure they are active)
        mosaic=1.0,
        mixup=0.1,          # Slight mixup to generalize better
    )
    
    print("Training Robusto completato.")

if __name__ == "__main__":
    train_robust()
