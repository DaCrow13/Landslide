# Documentazione Progetto: Landslide Detection (Esame Big Data)

Questo documento illustra l'architettura, le funzionalità e il workflow tecnico del sistema di rilevamento frane sviluppato per l'esame di Big Data. Il progetto si focalizza sull'integrazione di tecniche di **Contrastive Learning** e **Hard Negative Mining** per migliorare la robustezza di un modello YOLOv8 in scenari di forte sbilanciamento delle classi.

---

## 1. Analisi del Dataset e Contesto Big Data
Il progetto affronta il problema della rilevazione di frane in immagini aeree ad alta risoluzione (5cm/pixel). Nel contesto "Big Data", la sfida non è data solo dal volume, ma dalla **complessità data‑centrica**: le frane sono eventi rari e visivamente simili ad altri elementi geologici.

### Statistiche del Dataset (Training Set)
*   **Totale Immagini**: 2411
*   **Immagini con Frane (Positivi)**: 1124
*   **Immagini di Sfondo (Negativi)**: 1287
*   **Rapporto Sfondo/Frana**: ~1.15 (bilanciato artificialmente tramite Hard Negative Mining)
*   **Risoluzione Nativa**: Patch estratte da ortofoto di grandi dimensioni, processate a 640x640 pixel.

### Esempi di Immagini e Rilevazioni
Di seguito sono riportati due esempi di rilevazioni corrette (True Positives) dove il modello identifica la morfologia della frana separandola dal contesto boschivo o agricolo.

![Esempio TP 1 - Area Bella](C:\Users\Studente1\.gemini\antigravity\brain\c6ec4625-c4d4-476d-97b4-4c487a0a6183\example_tp_1.jpg)
*Figura 1: A sinistra l'originale, a destra la maschera predetta (verde). Notare la precisione sui bordi della colata.*

![Esempio TP 2 - Area Stigliano](C:\Users\Studente1\.gemini\antigravity\brain\c6ec4625-c4d4-476d-97b4-4c487a0a6183\example_tp_2.jpg)
*Figura 2: Rilevazione di una frana in un contesto di terreno nudo, distinta con successo dalle aree circostanti.*

### Esempi di Ambiguità
Le immagini di sfondo "difficile" includono:
*   Letti di fiumi in secca (texture granulare simile allo smottamento).
*   Strade sterrate o cantieri.
*   Zone di erosione naturale non classificate come frane.

---

## 2. Introduzione al Modello: YOLOv8-seg
Abbiamo scelto **YOLOv8-segmentation** (You Only Look Once v8) come baseline per le sue caratteristiche allo stato dell'arte:
*   **Anchor-Free Detection**: Riduce la complessità dei parametri e migliora la velocità.
*   **Decoupled Head**: Separa i rami di classificazione, regressione (box) e segmentazione (mask).
*   **Spatial Pyramid Pooling (SPPF)**: Permette di catturare feature a diverse scale, fondamentale per frane di varie dimensioni.

---

## 3. Integrazione Dettagliata del Contrastive Learning
Per superare i limiti della Cross-Entropy standard, abbiamo integrato un modulo di **Supervised Contrastive Learning (SupCon)** direttamente nel backbone di YOLO.

### Architettura dell'Integrazione
1.  **Feature Hooks**: Abbiamo inserito dei "ganci" (hooks) nei layer intermedi del backbone (Layer 4, 6 e 9) per estrarre le mappe di feature prima della classificazione finale.
2.  **Projection Heads**: Le feature estratte passano attraverso testine di proiezione composte da:
    *   `Conv2d (1x1)` -> `ReLU` -> `Conv2d (1x1, 128 dim)`.
    *   L'obiettivo è mappare le feature in uno spazio latente a 128 dimensioni dove il contrasto è più efficace.
3.  **Boundary-Aware Sampling**:
    *   Invece di campionare pixel a caso, il sistema identifica i **confini** delle frane (tramite erosione/dilatazione delle maschere).
    *   I "Positivi" sono campionati vicino al bordo della frana.
    *   I "Hard Negativi" sono campionati nelle aree immediatamente adiacenti (background immediato), dove la confusione è massima.

### La Loss Contrastiva
Utilizziamo la `SupContrastLoss`, che minimizza la distanza tra le rappresentazioni di pixel appartenenti alla classe "Frana" e massimizza quella rispetto ai pixel del "Background Difficile". La loss totale è definita come:
$$L_{total} = L_{YOLO} + \lambda \cdot L_{Contrastive}$$
con $\lambda = 0.05$ per non destabilizzare il gradiente principale.

---

## 4. Analisi Comparativa: Baseline vs Contrastive

| Caratteristica | YOLOv8 Baseline | YOLOv8 + Contrastive (Progetto) |
| :--- | :--- | :--- |
| **Allucinazioni** | Elevate su fiumi e strade | Significativamente ridotte (HNM) |
| **Decision Boundary** | Molto netto, sensibile al rumore | Più robusto, basato su feature latenti |
| **Recall** | Alto (ma sporco) | Inizialmente basso (modello "conservativo") |
| **Precision** | Bassissima (<20% su fiumi) | Migliorata (~43% con soglie filtrate) |
| **Metodologia** | Solo Supervisione Standard | Supervisione + Contrastive Injection |

---

## 5. Dinamiche di Training e Convergenza
Un'osservazione tecnica fondamentale emersa durante lo sviluppo riguarda il comportamento della funzione di perdita durante l'integrazione degli Hard Negatives.

*   **Initial Loss Shock**: Nelle prime fasi del retraining robusto, si osserva un picco della `box_loss`. Questo è dovuto allo "shock" che il modello subisce quando gli vengono presentati i letti di fiume (Hard Negatives) con maschere vuote: il modello deve "disimparare" bruscamente le associazioni errate apprese in precedenza.
*   **Stabilità Post-20 Epoche**: Dopo la fase di shock, la loss contrastiva agisce come un regolarizzatore, portando a una convergenza più stabile e a uno spazio latente meglio strutturato.

---

## 6. Elaborazioni Passo-Passo
1.  **Mining degli Hard Negatives**: Analisi automatica degli errori del modello baseline per isolare i 21 "fiumi critici".
2.  **Injection Contrastiva**: Reinserimento di questi negativi nel training con maschere vuote e abilitazione della SupCon Loss.
3.  **Calibration**: Abbassamento della soglia di confidenza a 0.10 e applicazione di filtri morfologici (Area > 300px) per bilanciare la nuova natura conservativa del modello.

---

## 6. Mappa dei File Tecnici

| File | Ruolo Tecnico |
| :--- | :--- |
| `contrastive_loss.py` | Implementazione della perdita contrastiva (PyTorch). |
| `contrastive_utils.py` | Logica di hooks, sampling e proiezione delle feature. |
| `train_robust.py` | Pipeline di training che integra i moduli sopra citati. |
| `predict_final.py` | Sistema di inferenza post-addestramento con filtri geometrici. |
| `analyze_performance.py` | Script per il calcolo delle performance su dataset di test. |
| `best_clean.pt` | Modello finale ottimizzato. |

---
*Documentazione redatta per il corso di Ingegneria dei Big Data.*
