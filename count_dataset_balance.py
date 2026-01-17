import os

def count_labels():
    train_label_dir = "A19_5cm/train/labels"
    train_img_dir = "A19_5cm/train/images"
    
    if not os.path.exists(train_label_dir):
        print(f"Directory non trovata: {train_label_dir}")
        return

    total_images = len([f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.tif'))])
    label_files = [f for f in os.listdir(train_label_dir) if f.endswith('.txt')]
    
    empty_labels = 0
    non_empty_labels = 0
    
    for lf in label_files:
        path = os.path.join(train_label_dir, lf)
        if os.path.getsize(path) == 0:
            empty_labels += 1
        else:
            non_empty_labels += 1
            
    # Images without any label file are also background in YOLOv8 usually, 
    # but strictly speaking we should have empty txt files.
    # Let's assume implied background if no txt exists, but let's count explicitly.
    
    orphaned_images = total_images - len(label_files)
    
    # Total background = empty .txt + no .txt
    total_background = empty_labels + orphaned_images
    
    print(f"--- Statistiche Training Set ---")
    print(f"Totale Immagini: {total_images}")
    print(f"Immagini con Frane (Non-empty labels): {non_empty_labels}")
    print(f"Immagini di Sfondo (Empty/No labels): {total_background}")
    
    if non_empty_labels > 0:
        ratio = total_background / non_empty_labels
        print(f"Rapporto Sfondo/Frana: {ratio:.2f}")
    else:
        print("Rapporto: N/A (0 frane)")

if __name__ == "__main__":
    count_labels()
