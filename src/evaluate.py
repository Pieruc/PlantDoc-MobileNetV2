"""
Script di valutazione del modello predittivo (Inference).
Genera il Classification Report (Precision, Recall, F1-Score in configurazione
Macro e Weighted) e salva la Matrice di Confusione.

Autore: Leonardo Pierucci
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configurazione
VAL_DIR = os.path.join('dataset', 'processed', 'val')
MODEL_PATH = os.path.join('models', 'mobilenet_multiclass_final.h5')
RESULTS_DIR = 'results'

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Errore: Modello {MODEL_PATH} non trovato. Eseguire prima il training.")

    print("\n--- Caricamento Modello e Dati ---")
    model = load_model(MODEL_PATH)
    
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR, target_size=(224, 224), batch_size=32, 
        class_mode='categorical', shuffle=False
    )

    class_names = list(val_generator.class_indices.keys())
    tutti_gli_indici = np.arange(len(class_names))

    print("\n--- Calcolo Predizioni ---")
    predictions = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes

    print("\n================ REPORT DI CLASSIFICAZIONE ================")
    # Stampa i valori Macro e Weighted F1-Score
    print(classification_report(y_true, y_pred, labels=tutti_gli_indici, target_names=class_names))
    print("===========================================================")

    print("\n--- Generazione Matrice di Confusione ---")
    cm = confusion_matrix(y_true, y_pred, labels=tutti_gli_indici)

    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                square=True, cbar_kws={"shrink": .75})

    plt.title('Matrice di Confusione - PlantDoc', fontsize=20, pad=20)
    plt.ylabel('Ground Truth', fontsize=16)
    plt.xlabel('Prediction', fontsize=16)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=300)
    print("Matrice salvata in 'results/confusion_matrix.png'")

    # Genera il report come dizionario
    report_dict = classification_report(
        y_true, 
        y_pred, 
        labels=np.arange(len(class_names)), 
        target_names=class_names, 
        output_dict=True, 
        zero_division=0
    )

    # Rimuovi le medie globali per isolare solo le classi
    report_dict.pop('accuracy', None)
    report_dict.pop('macro avg', None)
    report_dict.pop('weighted avg', None)

    # Crea un DataFrame e ordinalo per F1-Score decrescente
    df_metrics = pd.DataFrame(report_dict).transpose()
    df_metrics = df_metrics.sort_values(by='f1-score', ascending=False)

    # Disegna il grafico a barre orizzontali
    plt.figure(figsize=(12, 10))
    sns.barplot(x=df_metrics['f1-score'], y=df_metrics.index, palette='viridis')
    plt.title('F1-Score per singola classe (Ordinato dal migliore al peggiore)')
    plt.xlabel('F1-Score')
    plt.ylabel('Classe Patologica')
    plt.xlim(0, 1.05) # L'F1-Score va da 0 a 1
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('results/f1_score_per_class.png', dpi=300)
    print("Grafico F1-Score per classe salvato in results/f1_score_per_class.png")

if __name__ == "__main__":
    main()
