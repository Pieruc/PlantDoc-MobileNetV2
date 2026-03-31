"""
Script per l'addestramento del modello MobileNetV2 sul dataset PlantDoc (Multiclasse).
Implementa una pipeline di Transfer Learning in due fasi (Warm-up e Fine-Tuning) 
con Data Augmentation stocastica per mitigare l'overfitting.

Autore: Leonardo Pierucci
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import os

# Configurazione Percorsi Relativi
TRAIN_DIR = os.path.join('dataset', 'processed', 'train')
VAL_DIR = os.path.join('dataset', 'processed', 'val')
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_PHASE_1 = 20
EPOCHS_PHASE_2 = 25

def main():
    # --- 1. DATA AUGMENTATION ---
    print("\n--- Preparazione Dati e Augmentation ---")
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=90,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=[0.6, 1.4],
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.6, 1.4],
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical',
        classes=list(train_generator.class_indices.keys()), shuffle=False
    )

    # --- 2. FASE 1: WARM-UP ---
    print("\n--- FASE 1: Warm-up Classificatore ---")
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False 

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    history_1 = model.fit(
        train_generator, epochs=EPOCHS_PHASE_1, validation_data=val_generator,
        callbacks=[EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)]
    )

    # --- 3. FASE 2: FINE-TUNING ---
    print("\n--- FASE 2: Fine-Tuning ---")
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks_2 = [
        EarlyStopping(patience=8, monitor='val_loss', restore_best_weights=True),
        ModelCheckpoint(os.path.join(MODELS_DIR, 'mobilenet_multiclass_final.h5'), save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
    ]

    history_2 = model.fit(
        train_generator, epochs=EPOCHS_PHASE_1 + EPOCHS_PHASE_2,
        initial_epoch=history_1.epoch[-1], validation_data=val_generator, callbacks=callbacks_2
    )

    # --- 4. SALVATAGGIO GRAFICI ---
    print("\n--- Salvataggio Log e Grafici ---")
    acc = history_1.history['accuracy'] + history_2.history['accuracy']
    val_acc = history_1.history['val_accuracy'] + history_2.history['val_accuracy']
    loss = history_1.history['loss'] + history_2.history['loss']
    val_loss = history_1.history['val_loss'] + history_2.history['val_loss']

    pd.DataFrame({'accuracy': acc, 'val_accuracy': val_acc, 'loss': loss, 'val_loss': val_loss}).to_csv(
        os.path.join(RESULTS_DIR, 'training_history_multiclass.csv'), index=False)

    plt.figure(figsize=(14, 6))
    
    # Grafico Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy', color='blue')
    plt.plot(val_acc, label='Validation Accuracy', color='orange')
    plt.axvline(x=len(history_1.history['accuracy']), color='g', linestyle='--', label='Inizio Fine-Tuning')
    plt.title('Accuracy - Training e Fine-Tuning')
    plt.xlabel('Epoche')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Grafico Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.axvline(x=len(history_1.history['accuracy']), color='g', linestyle='--', label='Inizio Fine-Tuning')
    plt.title('Loss - Training e Fine-Tuning')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'multiclass_learning_curves.png'))
    print("Addestramento e salvataggio grafici completati.")

if __name__ == "__main__":
    main()
