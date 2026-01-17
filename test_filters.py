import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def apply_filters():
    model_path = "final_model.pt"
    source_dir = "A19_5cm/val/images"
    output_pdf = "Filtered_Report_Test.pdf"
    conf_threshold = 0.10
    
    # Filter Parameters
    MIN_AREA = 300       # Remove small noise (trees/bushes)
    MAX_LINEARITY = 4.0  # Remove long thin lines (roads). (MajorAxis / MinorAxis)
    
    print(f"Testing filters: Area < {MIN_AREA}, Linearity > {MAX_LINEARITY}")
    model = YOLO(model_path)
    
    images = sorted([f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.tif'))])
    
    # We only process images where filters MIGHT apply (i.e., there are detections)
    
    with PdfPages(output_pdf) as pdf:
        processed_count = 0
        
        for filename in images:
            img_path = os.path.join(source_dir, filename)
            
            # Predict
            results = model.predict(source=img_path, conf=conf_threshold, verbose=False)
            r = results[0]
            
            if r.masks is None:
                continue
                
            img_rgb = cv2.cvtColor(r.orig_img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            
            # Raw Mask
            raw_mask = np.zeros((h, w), dtype=np.uint8)
            current_masks_data = [] # Store individual masks for analysis
            if r.masks:
                current_masks_data = r.masks.data.cpu().numpy() # Warning: this might be resized.
                # Safer to use polygons for full res
                for mask in r.masks.xy:
                    pts = np.array(mask, dtype=np.int32)
                    cv2.fillPoly(raw_mask, [pts], 1)
            
            # Apply Filtering Logic on Connected Components of the Raw Mask
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
            
            filtered_mask = np.zeros((h, w), dtype=np.uint8)
            
            params_text = []
            
            has_change = False
            
            for i in range(1, num_labels): # Skip background 0
                area = stats[i, cv2.CC_STAT_AREA]
                
                # Aspect Ratio / Linearity check
                # Note: stats doesn't give orientation. We need moments or minAreaRect for that.
                # Let's extract the component contour.
                component_mask = (labels == i).astype(np.uint8)
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                keep = True
                status = "Keep"
                
                if len(contours) > 0:
                    cnt = contours[0]
                    
                    # 1. Area Filter
                    if area < MIN_AREA:
                        keep = False
                        status = "Removed (Small)"
                    else:
                        # 2. Linearity Filter (Roads)
                        if len(cnt) >= 5: # Need points for ellipse
                            (x,y), (MA, ma), angle = cv2.fitEllipse(cnt)
                            # ma is major axis, MA is minor (or vice versa depending on opencv version typically (MA, ma))
                            # Actually fitEllipse returns (center, axes, angle). axes=(minor, major) usually.
                            major = max(MA, ma)
                            minor = min(MA, ma)
                            
                            if minor > 0:
                                ratio = major / minor
                                if ratio > MAX_LINEARITY:
                                    keep = False
                                    status = f"Removed (Road, Ratio {ratio:.1f})"
                
                if keep:
                    filtered_mask[labels == i] = 1
                else:
                    has_change = True
                    params_text.append(status)
            
            if not has_change and processed_count >= 5: 
                # Skip if no interesting filtering happening (limit output size), 
                # but show at least 5 unchanging ones to confirm safety.
                continue
                
            processed_count += 1
            if processed_count > 20: # Limit to 20 examples
                break
                
            # Plot
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f"{filename}", fontsize=14)
            
            # Original
            ax[0].imshow(img_rgb)
            ax[0].set_title("Original")
            
            # Raw Pred
            ax[1].imshow(img_rgb)
            ax[1].imshow(raw_mask, alpha=0.4, cmap='Reds', vmin=0, vmax=1)
            ax[1].set_title(f"Raw Prediction (Conf {conf_threshold})")
            
            # Filtered Pred
            ax[2].imshow(img_rgb)
            ax[2].imshow(filtered_mask, alpha=0.4, cmap='Greens', vmin=0, vmax=1)
            ax[2].set_title("Filtered Prediction")
            
            # Text info
            if params_text:
                fig.text(0.5, 0.05, "Actions: " + ", ".join(params_text[:3]) + ("..." if len(params_text)>3 else ""), ha='center')
            
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Added filter comparison for {filename}")

    print(f"Filter test report generated: {output_pdf}")

if __name__ == "__main__":
    apply_filters()
