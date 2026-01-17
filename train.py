import yaml
from ultralytics import YOLO
import sys
import torch
from ultralytics.utils import loss
from ultralytics.data import build



def train(zone, model_name):
    model = YOLO(f"{model_name}.pt")
    model.train(data=f"yaml/{zone}.yaml",
                project=f"runs/{model_name}/{zone}",
                epochs=1,
                imgsz=512,
                verbose=True,
                exist_ok=True,
                device=0 if torch.cuda.is_available() else 'cpu',
                batch=32,
                single_cls=True,
                box=7.5,
                cls=0.5,
                dfl=1.5)
