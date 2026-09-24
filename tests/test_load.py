"""Script de prueba: carga una imagen y la muestra."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from pathlib import Path
import cv2
from config import IMAGES_DIR, PLOTS_DIR

def main():
    # Buscar imagenes recursivamente en todas las subcarpetas
    image_files = list(Path(IMAGES_DIR).rglob("*.jpg")) + \
                  list(Path(IMAGES_DIR).rglob("*.jpeg")) + \
                  list(Path(IMAGES_DIR).rglob("*.png"))
    
    if not image_files:
        print("No images found in:", IMAGES_DIR)
        print("Make sure you downloaded and moved the dataset.")
        return
    
    first_image = str(image_files[0])
    print(f"Loading: {first_image}")
    
    img = cv2.imread(first_image)
    if img is None:
        print("Could not read the image.")
        return
    
    print(f"Size: {img.shape} (height, width, channels)")
    print(f"Total images found: {len(image_files)}")
    
    # Guardar una miniatura para verificar visualmente
    thumb = cv2.resize(img, (300, 300))
    out_path = os.path.join(PLOTS_DIR, "test_thumbnail.jpg")
    cv2.imwrite(out_path, thumb)
    print(f"Thumbnail saved to: {out_path}")
    print("Open it to confirm everything works.")

if __name__ == "__main__":
    main()