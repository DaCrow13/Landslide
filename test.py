import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO


def save_masks(results, save_mask_dir):
    for result in results:
        image_path = result.path
        image_name = os.path.basename(image_path)
        name_no_ext = os.path.splitext(image_name)[0]

        original_img = cv2.imread(image_path)
        height, width = original_img.shape[:2]

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
        else:
            combined_mask = np.zeros((height, width), dtype=np.uint8)

        save_path = os.path.join(save_mask_dir, f"{name_no_ext}_mask.png")
        cv2.imwrite(save_path, combined_mask)


def test(zone, model_name):
    for confidence in np.arange(0.05, 0.45, 0.05):
        model_path = f"runs/{model_name}/{zone}/train/weights/last.pt"
        model = YOLO(model_path)

        PATH_OUT = f"runs/{model_name}/{zone}/confidence_{confidence:.2f}"
        save_mask_dir = f"{PATH_OUT}/mask"
        os.makedirs(save_mask_dir, exist_ok=True)

        data_path = os.path.join(os.getcwd(), zone, "test", "images")
        results = model.predict(source=data_path, retina_masks=True,
                                save=True, imgsz=512, conf=confidence, show_boxes=True, exist_ok=True,
                                project=PATH_OUT, stream=True, name=f'predict_{confidence:.2f}')

        save_masks(results, save_mask_dir)