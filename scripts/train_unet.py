"""
train_unet.py
Script d'entraînement U-Net adapté pour TrainingPipeline
Basé sur votre architecture avec Boundary Loss
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import albumentations as A
from pathlib import Path
from datetime import datetime

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Conv2DTranspose,
                                      concatenate, BatchNormalization, Activation, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Import boundary loss et métriques
try:
    from boundary_loss import (
        combined_loss_v2,
        dice_coefficient,
        dice_muscle,
        dice_fat,
    )
except ImportError:
    print("⚠️  boundary_loss.py non trouvé, utilisation de loss standard")
    combined_loss_v2 = 'categorical_crossentropy'
    dice_coefficient = None
    dice_muscle = None
    dice_fat = None

# ---------------------------
# Paramètres
# ---------------------------
IMG_HEIGHT, IMG_WIDTH = 512, 512
NUM_CLASSES = 3

# ---------------------------
# Data Augmentation
# ---------------------------
def get_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.RandomBrightnessContrast(p=0.2)
    ])

# ---------------------------
# DataGenerator
# ---------------------------
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, augmentations=None):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.mask_paths = [
            os.path.join(mask_dir, os.path.basename(f))
            for f in self.image_paths
        ]
        self.batch_size = batch_size
        self.augment = augmentations
        self.indexes = np.arange(len(self.image_paths))

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        images = []
        masks = []

        for i in batch_indexes:
            img = load_img(self.image_paths[i], target_size=(IMG_HEIGHT, IMG_WIDTH))
            img = img_to_array(img) / 255.0

            mask = load_img(self.mask_paths[i], color_mode='grayscale',
                          target_size=(IMG_HEIGHT, IMG_WIDTH))
            mask = img_to_array(mask)[..., 0].astype(np.int32)

            if self.augment:
                augmented = self.augment(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']

            mask_cat = tf.keras.utils.to_categorical(mask, num_classes=NUM_CLASSES)

            images.append(img)
            masks.append(mask_cat)

        return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

# ---------------------------
# Bloc de convolution
# ---------------------------
def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# ---------------------------
# U-Net STANDARD à 4 niveaux
# ---------------------------
def unet_multiclass(input_size=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    """
    Architecture U-Net STANDARD à 4 niveaux

    Encoder : 512 → 256 → 128 → 64 → 32
    Decoder : 32 → 64 → 128 → 256 → 512
    """
    inputs = Input(input_size)

    # ENCODER (Contracting Path) - 4 niveaux
    c1 = conv_block(inputs, 32)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 256)
    p4 = MaxPooling2D((2, 2))(c4)

    # BOTTLENECK : 32×32
    c5 = conv_block(p4, 512)
    c5 = Dropout(0.5)(c5)

    # DECODER (Expansive Path) - 4 niveaux
    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, 256)

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, 128)

    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, 64)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, 32)

    # SORTIE : 512×512
    outputs = Conv2D(num_classes, (1, 1), activation='softmax', name='output')(c9)

    model = Model(inputs, outputs, name='UNet_Multiclass_BoundaryLoss')

    return model

# ---------------------------
# Conversion en TFLite INT8
# ---------------------------
def convert_to_tflite(model, output_path, representative_dataset_gen=None):
    """Convertit le modèle Keras en TFLite INT8."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Quantization INT8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if representative_dataset_gen:
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    # Sauvegarder
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"✅ Modèle TFLite sauvegardé: {output_path}")
    print(f"   Taille: {len(tflite_model) / 1024 / 1024:.2f} MB")

# ---------------------------
# MAIN
# ---------------------------
def main(args):
    print("=" * 70)
    print("  ENTRAÎNEMENT U-NET POUR ACTIVE LEARNING PIPELINE")
    print("=" * 70)
    print()

    # Chemins
    train_images = Path(args.train_dir) / "images"
    train_masks = Path(args.train_dir) / "masks"
    val_images = Path(args.val_dir) / "images"
    val_masks = Path(args.val_dir) / "masks"

    print(f"📂 Configuration:")
    print(f"   Train images: {train_images}")
    print(f"   Train masks: {train_masks}")
    print(f"   Val images: {val_images}")
    print(f"   Val masks: {val_masks}")
    print(f"   Output: {args.output}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print()

    # Vérifier que les dossiers existent
    for path in [train_images, train_masks, val_images, val_masks]:
        if not path.exists():
            print(f"❌ Dossier introuvable: {path}")
            sys.exit(1)

    # Créer les générateurs
    print("📂 Chargement des données...")
    train_gen = DataGenerator(
        str(train_images),
        str(train_masks),
        batch_size=args.batch_size,
        augmentations=get_augmentations()
    )
    val_gen = DataGenerator(
        str(val_images),
        str(val_masks),
        batch_size=args.batch_size,
        augmentations=None
    )

    print(f"   ✓ Batches train: {len(train_gen)}")
    print(f"   ✓ Batches val: {len(val_gen)}")
    print()

    # Construire le modèle
    print("🏗️  Construction du modèle U-Net...")
    model = unet_multiclass()
    print(f"   ✓ Paramètres: {model.count_params():,}")
    print()

    # Compiler
    print("⚙️  Compilation du modèle...")

    metrics = ['accuracy']
    if dice_coefficient:
        metrics.extend([dice_coefficient, dice_muscle, dice_fat])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=combined_loss_v2 if combined_loss_v2 != 'categorical_crossentropy' else 'categorical_crossentropy',
        metrics=metrics
    )
    print("   ✓ Compilation réussie")
    print()

    # Callbacks
    print("📋 Configuration des callbacks...")

    checkpoint_dir = Path(args.output).parent
    checkpoint_path = checkpoint_dir / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"

    callbacks = [
        ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            mode='min',
            min_lr=1e-7
        )
    ]

    print("   ✓ Callbacks configurés")
    print()

    # Training
    print("🚀 LANCEMENT TRAINING")
    print("=" * 70)
    print()

    try:
        history = model.fit(
            train_gen,
            epochs=args.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )

        print()
        print("=" * 70)
        print("✅ TRAINING TERMINÉ")
        print("=" * 70)

    except Exception as e:
        print(f"❌ ERREUR durant training: {e}")
        raise

    # Sauvegarder modèle Keras
    print()
    print("💾 Sauvegarde du modèle...")

    keras_output = str(args.output).replace('.tflite', '.keras')
    model.save(keras_output)
    print(f"   ✓ Modèle Keras sauvegardé: {keras_output}")

    # Conversion TFLite INT8
    print()
    print("🔄 Conversion en TFLite INT8...")

    def representative_dataset():
        for i in range(min(100, len(val_gen))):
            batch = val_gen[i]
            yield [batch[0][:1].astype(np.float32)]

    convert_to_tflite(model, args.output, representative_dataset)

    # Sauvegarder métriques
    print()
    print("📊 Sauvegarde des métriques...")

    metrics_output = str(args.output).replace('.tflite', '_metrics.json')

    final_metrics = {
        'train_loss': float(history.history['loss'][-1]),
        'val_loss': float(history.history['val_loss'][-1]),
        'train_accuracy': float(history.history['accuracy'][-1]),
        'val_accuracy': float(history.history['val_accuracy'][-1]),
    }

    if 'dice_coefficient' in history.history:
        final_metrics['train_dice'] = float(history.history['dice_coefficient'][-1])
        final_metrics['val_dice'] = float(history.history['val_dice_coefficient'][-1])
        final_metrics['val_dice_muscle'] = float(history.history['val_dice_muscle'][-1])
        final_metrics['val_dice_fat'] = float(history.history['val_dice_fat'][-1])

    with open(metrics_output, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print(f"   ✓ Métriques sauvegardées: {metrics_output}")

    # Résumé final
    print()
    print("=" * 70)
    print("📊 RÉSUMÉ FINAL")
    print("=" * 70)
    print(f"   • Training Loss     : {final_metrics['train_loss']:.4f}")
    print(f"   • Validation Loss   : {final_metrics['val_loss']:.4f}")
    print(f"   • Training Accuracy : {final_metrics['train_accuracy']:.4f}")
    print(f"   • Val Accuracy      : {final_metrics['val_accuracy']:.4f}")

    if 'val_dice' in final_metrics:
        print(f"   • Validation Dice   : {final_metrics['val_dice']:.4f}")
        print(f"   • Val Dice Muscle   : {final_metrics['val_dice_muscle']:.4f}")
        print(f"   • Val Dice Fat      : {final_metrics['val_dice_fat']:.4f}")

    print()
    print("=" * 70)
    print("✨ TERMINÉ !")
    print("=" * 70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entraînement U-Net pour Active Learning")
    parser.add_argument("--train-dir", required=True, help="Dossier train (contient images/ et masks/)")
    parser.add_argument("--val-dir", required=True, help="Dossier validation (contient images/ et masks/)")
    parser.add_argument("--output", required=True, help="Chemin de sortie du modèle TFLite")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre d'epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Taille du batch")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    main(args)
