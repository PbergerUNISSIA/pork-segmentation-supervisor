"""
====================================================================
PORK SEGMENTATION SUPERVISOR - MODEL MANAGER
====================================================================
Gestionnaire de modèle TensorFlow Lite pour inférence optimisée.
====================================================================
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from config.settings import settings
from utils.logger import logger, log_function_call


class ModelManager:
    """Gestionnaire de modèle TensorFlow Lite pour segmentation."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.interpreter: Optional[tf.lite.Interpreter] = None
        self.input_details: Optional[dict] = None
        self.output_details: Optional[dict] = None
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
        self._load_model()
        logger.success(f"ModelManager initialized with model: {self.model_path}")

    @log_function_call
    def _load_model(self) -> None:
        """Charge le modèle TFLite et alloue les tensors."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        logger.info(f"Loading TFLite model from {self.model_path}")

        try:
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]
            self.input_shape = tuple(self.input_details['shape'])
            self.output_shape = tuple(self.output_details['shape'])

            logger.info(f"Model loaded successfully")
            logger.debug(f"Input shape: {self.input_shape}")
            logger.debug(f"Output shape: {self.output_shape}")
            logger.debug(f"Input dtype: {self.input_details['dtype']}")
            logger.debug(f"Output dtype: {self.output_details['dtype']}")

            if self.input_details['dtype'] == np.uint8:
                logger.info("Model is INT8 quantized (optimized)")
                input_scale, input_zero = self._get_quantization_params("input")
                logger.debug(f"Input quantization: scale={input_scale}, zero_point={input_zero}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def _get_quantization_params(self, tensor_type: str = "input") -> Tuple[float, int]:
        """Récupère les paramètres de quantization INT8."""
        details = self.input_details if tensor_type == "input" else self.output_details
        quantization = details['quantization_parameters']
        scale = quantization['scales'][0] if len(quantization['scales']) > 0 else 1.0
        zero_point = quantization['zero_points'][0] if len(quantization['zero_points']) > 0 else 0
        return scale, zero_point

    @log_function_call
    def predict(self, image: np.ndarray, return_probabilities: bool = False) -> np.ndarray:
        """Effectue une prédiction de segmentation sur une image."""
        start_time = time.time()
        input_tensor = self._prepare_input(image)
        self.interpreter.set_tensor(self.input_details['index'], input_tensor)
        self.interpreter.invoke()
        output_tensor = self.interpreter.get_tensor(self.output_details['index'])
        result = self._process_output(output_tensor, return_probabilities)
        elapsed = time.time() - start_time
        logger.debug(f"Inference completed in {elapsed*1000:.1f}ms")
        return result

    def _prepare_input(self, image: np.ndarray) -> np.ndarray:
        """Prépare l'image pour l'inférence."""
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        expected_shape = self.input_shape
        if image.shape != expected_shape:
            raise ValueError(
                f"Invalid input shape. Expected {expected_shape}, got {image.shape}. "
                f"Please resize image to {expected_shape[1]}x{expected_shape[2]}"
            )

        if self.input_details['dtype'] == np.uint8:
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        else:
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
        return image

    def _process_output(self, output_tensor: np.ndarray, return_probabilities: bool) -> np.ndarray:
        """Post-traite la sortie du modèle."""
        if self.output_details['dtype'] == np.uint8:
            scale, zero_point = self._get_quantization_params("output")
            output_tensor = (output_tensor.astype(np.float32) - zero_point) * scale

        if return_probabilities:
            return np.squeeze(output_tensor, axis=0)
        else:
            if output_tensor.shape[-1] > 1:
                mask = np.argmax(output_tensor, axis=-1)
            else:
                mask = (output_tensor > 0.5).astype(np.uint8)
            return np.squeeze(mask, axis=0)

    def get_model_info(self) -> dict:
        """Retourne les informations du modèle."""
        return {
            "model_path": str(self.model_path),
            "model_version": settings.MODEL_VERSION,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "input_dtype": str(self.input_details['dtype']),
            "output_dtype": str(self.output_details['dtype']),
            "is_quantized": self.input_details['dtype'] == np.uint8,
            "model_size_mb": self.model_path.stat().st_size / (1024 * 1024)
        }

    def __repr__(self) -> str:
        return f"ModelManager(model={self.model_path.name}, version={settings.MODEL_VERSION}, input_shape={self.input_shape})"
