import os

def count_positives(label_dir):
    if not os.path.exists(label_dir):
        print(f"Directory non trovata: {label_dir}")
        return

    total_files = 0
    positive_files = 0
    
    files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    total_files = len(files)
    
    print(f"Analisi di {total_files} file di etichetta in: {label_dir}\n")
    
    for f in files:
        path = os.path.join(label_dir, f)
        # Un file è positivo se ha una dimensione > 0 (cioè contiene annotazioni)
        if os.path.getsize(path) > 0:
            positive_files += 1
            print(f"Positivo: {f} (Size: {os.path.getsize(path)} bytes)")
            
    print(f"\nRisultato:")
    print(f"Totale Immagini: {total_files}")
    print(f"Immagini CON Frane (Ground Truth): {positive_files}")
    print(f"Immagini SENZA Frane (Background): {total_files - positive_files}")

if __name__ == "__main__":
    count_positives("A19_5cm/val/labels")
