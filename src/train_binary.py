"""
Script per il task di validazione binaria (Top-2 Classi).
Estrae le due classi maggioritarie dal dataset e addestra un classificatore binario 
per verificare la convergenza del modello al netto dello sbilanciamento del dataset completo.

Autore: Leonardo Pierucci
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

TRAIN_DIR = os.path.join('dataset', 'processed', 'train')
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

def main():
    # Trova le 2 classi più numerose
    class_counts = {name: len(os.listdir(os.path.join(TRAIN_DIR, name))) 
                    for name in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, name))}
    top_2 = sorted(class_counts, key=class_counts.get, reverse=True)[:2]
    print(f"Task Binario sulle classi: {top_2}")

    # Generatori con Validation Split 80/20
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)
    
    train_gen = datagen.flow_from_directory(
        TRAIN_DIR, target_size=(224, 224), batch_size=16, 
        class_mode='categorical', classes=top_2, subset='training', shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        TRAIN_DIR, target_size=(224, 224), batch_size=16, 
        class_mode='categorical', classes=top_2, subset='validation', shuffle=False
    )

    # Modello Binario Rapido
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_gen, validation_data=val_gen, epochs=15)

    # Salvataggio del modello
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(os.path.join(MODELS_DIR, 'mobilenet_binary_top2.h5'))
    
    # Generazione dei Grafici (Accuracy e Loss)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Grafico Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    axes[0].set_title('Accuracy - Task Binario Top 2')
    axes[0].set_xlabel('Epoche')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Grafico Loss
    axes[1].plot(history.history['loss'], label='Train Loss', color='blue')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', color='orange')
    axes[1].set_title('Loss - Task Binario Top 2')
    axes[1].set_xlabel('Epoche')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'binary_top2_accuracy.png'))
    print("Task Binario e salvataggio grafici completati.")

if __name__ == "__main__":
    main()
