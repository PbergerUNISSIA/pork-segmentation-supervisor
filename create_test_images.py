"""
====================================================================
PORK SEGMENTATION SUPERVISOR - TEST IMAGE GENERATOR
====================================================================
Génère des images de test synthétiques pour tester le système.

USAGE:
    python create_test_images.py --count 5
====================================================================
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config.settings import settings


def create_synthetic_carcass_image(
    width: int = 512,
    height: int = 512,
    seed: int = 42
) -> np.ndarray:
    """
    Crée une image synthétique de carcasse.

    Args:
        width: Largeur de l'image
        height: Hauteur de l'image
        seed: Seed pour reproductibilité

    Returns:
        np.ndarray: Image RGB (H, W, 3)
    """
    np.random.seed(seed)

    # Fond gris
    image = np.ones((height, width, 3), dtype=np.uint8) * 200

    # Ajouter du bruit
    noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    # Dessiner une forme elliptique (carcasse)
    center_x = width // 2 + np.random.randint(-50, 50)
    center_y = height // 2 + np.random.randint(-50, 50)
    axes_major = width // 3 + np.random.randint(-30, 30)
    axes_minor = height // 4 + np.random.randint(-20, 20)
    angle = np.random.randint(0, 180)

    # Couleur chair
    color = (
        180 + np.random.randint(-30, 30),  # R
        120 + np.random.randint(-30, 30),  # G
        130 + np.random.randint(-30, 30),  # B
    )

    cv2.ellipse(
        image,
        (center_x, center_y),
        (axes_major, axes_minor),
        angle,
        0,
        360,
        color,
        -1
    )

    # Ajouter des détails (graisse, marbrure)
    num_blobs = np.random.randint(3, 8)
    for _ in range(num_blobs):
        blob_x = np.random.randint(center_x - axes_major, center_x + axes_major)
        blob_y = np.random.randint(center_y - axes_minor, center_y + axes_minor)
        blob_size = np.random.randint(10, 40)

        blob_color = (
            220 + np.random.randint(-20, 20),
            220 + np.random.randint(-20, 20),
            200 + np.random.randint(-20, 20),
        )

        cv2.circle(image, (blob_x, blob_y), blob_size, blob_color, -1)

    # Flouter légèrement pour plus de réalisme
    image = cv2.GaussianBlur(image, (5, 5), 0)

    return image


def create_synthetic_mask(
    width: int = 512,
    height: int = 512,
    seed: int = 42
) -> np.ndarray:
    """
    Crée un masque de segmentation synthétique.

    Args:
        width: Largeur du masque
        height: Hauteur du masque
        seed: Seed pour reproductibilité

    Returns:
        np.ndarray: Masque binaire (H, W)
    """
    np.random.seed(seed)

    # Masque noir
    mask = np.zeros((height, width), dtype=np.uint8)

    # Dessiner une forme elliptique (zone de carcasse)
    center_x = width // 2 + np.random.randint(-50, 50)
    center_y = height // 2 + np.random.randint(-50, 50)
    axes_major = width // 3 + np.random.randint(-30, 30)
    axes_minor = height // 4 + np.random.randint(-20, 20)
    angle = np.random.randint(0, 180)

    cv2.ellipse(
        mask,
        (center_x, center_y),
        (axes_major, axes_minor),
        angle,
        0,
        360,
        255,
        -1
    )

    # Ajouter du bruit au masque
    noise_mask = np.random.rand(height, width) > 0.98
    mask[noise_mask] = 255 - mask[noise_mask]

    return mask


def main():
    """Génère des images de test."""
    parser = argparse.ArgumentParser(description="Generate test images for Pork Segmentation Supervisor")
    parser.add_argument("--count", type=int, default=5, help="Number of images to generate")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    args = parser.parse_args()

    print("=" * 70)
    print("🖼️  GENERATING TEST IMAGES")
    print("=" * 70)

    # Créer les dossiers si nécessaires
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Output directories:")
    print(f"   Images: {settings.UPLOAD_DIR}")
    print(f"   Masks:  {settings.ANNOTATIONS_DIR}")

    print(f"\n🎨 Generating {args.count} synthetic images...")

    for i in range(args.count):
        seed = 42 + i

        # Générer image
        image = create_synthetic_carcass_image(
            width=args.width,
            height=args.height,
            seed=seed
        )

        # Générer masque
        mask = create_synthetic_mask(
            width=args.width,
            height=args.height,
            seed=seed
        )

        # Sauvegarder l'image
        image_filename = f"test_carcass_{i+1:03d}.jpg"
        image_path = settings.UPLOAD_DIR / image_filename
        image_pil = Image.fromarray(image)
        image_pil.save(image_path, quality=95)

        # Sauvegarder le masque
        mask_filename = f"test_mask_{i+1:03d}.png"
        mask_path = settings.ANNOTATIONS_DIR / mask_filename
        mask_pil = Image.fromarray(mask)
        mask_pil.save(mask_path)

        print(f"   ✅ Generated: {image_filename} + {mask_filename}")

    print("\n" + "=" * 70)
    print(f"✅ GENERATED {args.count} TEST IMAGES SUCCESSFULLY")
    print("=" * 70)

    print("\n💡 Next steps:")
    print("   1. Run: streamlit run app.py")
    print("   2. Upload the generated images from:")
    print(f"      {settings.UPLOAD_DIR.absolute()}")
    print("   3. Test the prediction workflow")


if __name__ == "__main__":
    main()
