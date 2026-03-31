"""
Script per il parsing dei file XML e l'organizzazione del dataset PlantDoc.
Estrae le etichette delle patologie fogliari e suddivide le immagini in sottocartelle
per renderle compatibili con Keras ImageDataGenerator.

Autore: Leonardo Pierucci
"""

import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Percorsi relativi (assumendo l'esecuzione dalla root del progetto)
SOURCE_TRAIN_DIR = os.path.join('dataset', 'train')
SOURCE_TEST_DIR = os.path.join('dataset', 'test')
DEST_ROOT_DIR = os.path.join('dataset', 'processed')

def organizza_immagini(source_dir, dest_split_name):
    print(f"\n--- Elaborazione Split: {dest_split_name} ---")
    dest_path = os.path.join(DEST_ROOT_DIR, dest_split_name)
    os.makedirs(dest_path, exist_ok=True)

    if not os.path.exists(source_dir):
        print(f"Errore: Cartella sorgente {source_dir} non trovata.")
        return

    xml_files = [f for f in os.listdir(source_dir) if f.endswith('.xml')]
    count = 0
    
    for xml_file in tqdm(xml_files, desc=f"Processando {dest_split_name}"):
        try:
            tree = ET.parse(os.path.join(source_dir, xml_file))
            root = tree.getroot()
            
            object_tag = root.find('object')
            if object_tag is None:
                continue 
                
            class_name = object_tag.find('name').text.strip().replace(" ", "_")
            base_name = os.path.splitext(xml_file)[0]
            
            image_name = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                if os.path.exists(os.path.join(source_dir, base_name + ext)):
                    image_name = base_name + ext
                    break
            
            if image_name:
                class_dir = os.path.join(dest_path, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                src_img_path = os.path.join(source_dir, image_name)
                dst_img_path = os.path.join(class_dir, image_name)
                shutil.copy(src_img_path, dst_img_path)
                count += 1
                
        except Exception as e:
            print(f"Errore parsing {xml_file}: {e}")

    print(f"✅ Completato! Spostate {count} immagini in {dest_path}")

if __name__ == "__main__":
    organizza_immagini(SOURCE_TRAIN_DIR, 'train')
    organizza_immagini(SOURCE_TEST_DIR, 'val')