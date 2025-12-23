"""
====================================================================
PORK SEGMENTATION SUPERVISOR - IMAGE PROCESSOR
====================================================================
Preprocessing et post-processing d'images pour segmentation.
====================================================================
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
from config.settings import settings
from utils.logger import logger, log_function_call


class ImageProcessor:
    """Processeur d'images pour pipeline de segmentation."""

    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        if target_size is None:
            size = settings.MODEL_INPUT_SIZE
            self.target_size = (size, size)
        else:
            self.target_size = target_size
        self.supported_formats = settings.get_supported_formats_list()
        logger.info(f"ImageProcessor initialized with target_size={self.target_size}")

    @log_function_call
    def load_image(self, image_path: Union[str, Path], color_mode: str = "RGB") -> np.ndarray:
        """Charge une image depuis un fichier."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        extension = image_path.suffix.lower().lstrip('.')
        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported format: {extension}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        if color_mode == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_mode == "GRAY":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        logger.debug(f"Loaded image: {image_path.name}, shape={image.shape}")
        return image

    @log_function_call
    def resize(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """Redimensionne une image à la taille cible."""
        if target_size is None:
            target_size = self.target_size
        target_size_cv = (target_size[1], target_size[0])
        resized = cv2.resize(image, target_size_cv, interpolation=interpolation)
        logger.debug(f"Resized image from {image.shape} to {resized.shape}")
        return resized

    def normalize(self, image: np.ndarray, mode: str = "float") -> np.ndarray:
        """Normalise une image."""
        if mode == "float":
            if image.dtype == np.uint8:
                normalized = image.astype(np.float32) / 255.0
            else:
                normalized = image.astype(np.float32)
        elif mode == "standard":
            mean = np.mean(image)
            std = np.std(image)
            normalized = (image.astype(np.float32) - mean) / (std + 1e-7)
        elif mode == "minmax":
            min_val = np.min(image)
            max_val = np.max(image)
            normalized = (image.astype(np.float32) - min_val) / (max_val - min_val + 1e-7)
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
        return normalized

    def denormalize(self, image: np.ndarray, to_uint8: bool = True) -> np.ndarray:
        """Dénormalise une image de [0, 1] vers [0, 255]."""
        denormalized = image * 255.0
        if to_uint8:
            denormalized = denormalized.astype(np.uint8)
        return denormalized

    def create_overlay(self, image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.5) -> np.ndarray:
        """Crée une visualisation overlay."""
        if image.dtype != np.uint8:
            image = self.denormalize(image, to_uint8=True)
        overlay = image.copy()
        mask_binary = (mask > 0).astype(np.uint8)
        colored_mask = np.zeros_like(image)
        colored_mask[mask_binary == 1] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)
        return overlay

    def create_contours(self, image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """Dessine les contours du masque sur l'image."""
        if image.dtype != np.uint8:
            image = self.denormalize(image, to_uint8=True)
        result = image.copy()
        mask_binary = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_bgr = (color[2], color[1], color[0])
        cv2.drawContours(result, contours, -1, color_bgr, thickness)
        return result

    @log_function_call
    def save_image(self, image: np.ndarray, save_path: Union[str, Path], color_mode: str = "RGB") -> None:
        """Sauvegarde une image dans un fichier."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if color_mode == "RGB" and image.ndim == 3:
            image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_to_save = image
        success = cv2.imwrite(str(save_path), image_to_save)
        if not success:
            raise RuntimeError(f"Failed to save image to {save_path}")
        logger.info(f"Image saved: {save_path}")

    def save_mask(self, mask: np.ndarray, save_path: Union[str, Path]) -> None:
        """Sauvegarde un masque binaire."""
        save_path = Path(save_path)
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        self.save_image(mask_uint8, save_path, color_mode="GRAY")

    def __repr__(self) -> str:
        return f"ImageProcessor(target_size={self.target_size})"
