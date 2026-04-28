"""
Model Training Pipeline — generates vision CNN and XGBoost risk models.

Improvement #1: Synthetic training images use spectrally realistic color
distributions per vegetation class instead of pure random noise. Each
species has a characteristic RGB mean + variance that mimics its real
spectral signature (green canopy, brown bark, yellow cured grass, etc.).

Improvement #2: XGBoost risk model is self-trained from CA simulation
outputs rather than random data, ensuring internally consistent feature
distributions.

Improvement #6: 4-class risk output (Low/Moderate/High/Extreme).
"""

import os
import logging

import numpy as np
import tensorflow as tf
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# 13 fuel classes — order must match vision_node.CLASS_NAMES
CLASSES = [
    "Pinus halepensis", "Quercus ilex", "Quercus suber",
    "Stipa tenacissima", "Pistacia lentiscus", "Arbutus unedo",
    "Tetraclinis articulata", "Cistus monspeliensis",
    "Calicotome villosa", "Juniperus phoenicea",
    "Erica arborea", "Olea europaea", "Barren",
]

# Approximate RGB spectral signatures per class (mean R, G, B, std)
# These create distinguishable distributions for the CNN to learn from,
# unlike pure random noise which teaches nothing.
SPECTRAL_PROFILES = {
    "Pinus halepensis":       (60, 95, 50, 20),    # Dark green conifer
    "Quercus ilex":           (55, 85, 45, 18),    # Deep green broadleaf
    "Quercus suber":          (90, 80, 50, 22),    # Grey-brown bark + green
    "Stipa tenacissima":      (170, 160, 90, 30),  # Yellow-brown cured grass
    "Pistacia lentiscus":     (70, 110, 55, 15),   # Bright green shrub
    "Arbutus unedo":          (80, 100, 60, 20),   # Medium green + red berries
    "Tetraclinis articulata": (65, 90, 55, 18),    # Blue-green conifer
    "Cistus monspeliensis":   (100, 120, 80, 25),  # Light green + white flowers
    "Calicotome villosa":     (120, 130, 50, 25),  # Yellowish-green thorny
    "Juniperus phoenicea":    (50, 75, 45, 15),    # Very dark green
    "Erica arborea":          (85, 95, 75, 20),    # Grey-green heather
    "Olea europaea":          (75, 100, 65, 18),   # Silver-green olive
    "Barren":                 (160, 140, 110, 35),  # Sandy/rocky soil
}


def generate_spectral_images(n_per_class=50, img_size=64):
    """
    Generate synthetic training images with class-specific spectral
    distributions instead of pure random noise.

    Each image is a 64x64 patch with RGB values drawn from the species'
    characteristic color distribution + Gaussian texture noise.
    """
    images = []
    labels = []

    for idx, cls_name in enumerate(CLASSES):
        r_mean, g_mean, b_mean, std = SPECTRAL_PROFILES[cls_name]

        for _ in range(n_per_class):
            # Base color from spectral profile
            img = np.zeros((img_size, img_size, 3), dtype=np.float32)
            img[:, :, 0] = np.random.normal(r_mean, std, (img_size, img_size))
            img[:, :, 1] = np.random.normal(g_mean, std, (img_size, img_size))
            img[:, :, 2] = np.random.normal(b_mean, std, (img_size, img_size))

            # Add texture variation (patches of lighter/darker areas)
            for _ in range(3):
                px = np.random.randint(0, img_size - 16)
                py = np.random.randint(0, img_size - 16)
                patch_size = np.random.randint(8, 16)
                brightness = np.random.uniform(0.7, 1.3)
                img[px:px+patch_size, py:py+patch_size] *= brightness

            img = np.clip(img, 0, 255).astype(np.float32) / 255.0
            images.append(img)
            labels.append(idx)

    return np.array(images), np.array(labels)


def train_vision_model():
    """Train 13-class CNN with spectrally realistic synthetic data."""
    logging.info("--- Training Exhaustive Vision Model (13 Classes) ---")
    logging.info("Generating spectrally realistic synthetic data...")

    num_classes = len(CLASSES)
    x_train, y_train = generate_spectral_images(n_per_class=50)

    # Shuffle
    perm = np.random.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]

    logging.info(f"Training set: {x_train.shape[0]} images, "
                 f"{num_classes} classes")

    # Build CNN
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    logging.info("Fitting model (3 epochs on spectral data)...")
    model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=1)

    # Save baseline
    model_path = "models/vision_baseline.keras"
    model.save(model_path)
    logging.info(f"Saved baseline model to {model_path}")

    # Quantize to TFLite INT8
    logging.info("Reducing model size via TFLite INT8 Quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        for i in range(50):
            yield [x_train[i:i + 1]]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    tflite_path = "models/vision_quantized.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    logging.info(f"Saved quantized model to {tflite_path}")


def train_risk_model():
    """
    Train 4-class XGBoost from CA simulation outputs (#2 + #6).
    Uses the self_train() method in AIRiskAnalyzer.
    """
    logging.info("--- Self-Training Risk Analyzer from CA Simulations ---")

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from simulation.ca_engine import CellularAutomataEngine
    from database.botanical_db import BotanicalDatabase
    from analyzer.risk_analyzer import AIRiskAnalyzer

    ca = CellularAutomataEngine(grid_size=(30, 30))  # Smaller for speed
    db = BotanicalDatabase()
    analyzer = AIRiskAnalyzer(model_path="models/regional_risk.json")
    analyzer.self_train(ca, db, n_samples=300)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_vision_model()
    train_risk_model()
    logging.info("=== Full model pipeline complete ===")
