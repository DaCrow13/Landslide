import sys
from contrastive_utils import ContrastiveTrainer, log_debug

def train_contrastive(zone, model_name, epochs=100):
    log_debug(f"Starting Optimized Contrastive Training (Weight 0.05) for {zone}")
    args = dict(
        model=f"{model_name}.pt",
        data=f"yaml/{zone}.yaml",
        project=f"runs/{model_name}_contrastive/{zone}",
        epochs=int(epochs),
        imgsz=1024,
        device=0,
        batch=8,
        patience=25, 
        save=True,
        lr0=0.0005,  
        cos_lr=True,
        warmup_epochs=10, 
        amp=False, # Stabilized with AMP disabled
        workers=0, 
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        close_mosaic=20 
    )
    trainer = ContrastiveTrainer(overrides=args)
    trainer.train()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train_contrastive.py <zone> <model_name> [epochs]")
    else:
        train_contrastive(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 100)
