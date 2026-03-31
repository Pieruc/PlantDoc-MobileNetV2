# Classificazione Automatica di Fitopatologie "In the Wild" con MobileNetV2

Questo repository contiene il codice sorgente e i risultati sperimentali sviluppati per la tesi di laurea: **"Architetture Basate su Convoluzione per Riconoscere le Malattie delle Piante"**. 

L'obiettivo del progetto è valutare l'efficacia dell'architettura **MobileNetV2** nell'estrazione di *feature* patologiche sul dataset **PlantDoc**, caratterizzato da immagini acquisite in condizioni non controllate (*in the wild*).

## Struttura del Progetto

- `dataset/`: Cartella destinata a ospitare i dati grezzi (`train` e `test`) scaricati dalla sorgente originale.
- `dataset/processed/`: Sottocartella generata automaticamente contenente le immagini estratte, organizzate in `train/` e `val/`.
- `src/`: Directory contenente tutti gli script Python (`.py`) del progetto.
- `models/`: Directory in cui vengono salvati i pesi dei modelli addestrati (`.h5`).
- `results/`: Directory dedicata al salvataggio dei grafici (curve di apprendimento, matrici di confusione e F1-Score).

## Requisiti e Installazione

Il progetto è stato sviluppato in **Python 3.9** e ottimizzato per sfruttare l'accelerazione hardware **NVIDIA CUDA**. 

1. Clonare la repository:
   ```bash
   git clone [https://github.com/tuo-username/nome-repo.git](https://github.com/tuo-username/nome-repo.git)
   cd nome-repo
   ```

2. Installare le dipendenze richieste:
   ```bash
   pip install -r requirements.txt
   ```

## Pipeline Sperimentale

Tutti i comandi vanno lanciati dalla cartella principale del progetto.

### 0. Download e Preparazione del Dataset
Prima di avviare la pipeline, è necessario scaricare il dataset originale:
1. Andare sul repository ufficiale di [PlantDoc su GitHub](https://github.com/pratikkayal/PlantDoc-Dataset).
2. Scaricare le cartelle `train` e `test` (che contengono sia le immagini che i file `.xml`).
3. Inserire le cartelle scaricate all'interno della directory `dataset/` di questo progetto.

Una volta posizionati i dati, lanciare lo script di pre-elaborazione per generare la cartella `dataset/processed/`:
```bash
python src/dataset_prep.py
```

### 1. Addestramento Multiclasse (28 Classi)
Esegue la pipeline di Transfer Learning (Warm-up + Fine-Tuning) con Data Augmentation:
```bash
python src/train_multiclass.py
```

### 2. Valutazione e Analisi
Genera la matrice di confusione e il grafico dell'F1-Score disaggregato per le 28 classi partendo dal modello salvato:
```bash
python src/evaluate.py
```

### 3. Validazione Binaria (Top-2)
Esegue il test di controllo sulle due classi maggioritarie per verificare la solidità dell'architettura:
```bash
python src/train_binary.py
```

## Risultati Principali

- **Accuratezza Multiclasse:** ~49% (Limite imposto dalla rumorosità del dataset PlantDoc e dal forte *class imbalance*).
- **Accuratezza Binaria (Top-2):** ~98-100% (Conferma della capacità di generalizzazione di MobileNetV2 su dati bilanciati).
- **Efficienza:** Modello ottimizzato per il deployment in ambito Edge Computing (impronta di memoria ~14 MB).

## Autore

**Leonardo Pierucci** Tesi di Laurea – Università degli Studi [Inserisci Nome]