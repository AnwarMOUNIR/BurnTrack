import os
import numpy as np
import tensorflow as tf
import xgboost as xgb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def train_vision_model():
    logging.info("--- Training Exhaustive Vision Model (13 Classes) ---")
    
    # 1. Exhaustive Taxonomy
    classes = [
        "Pinus halepensis", "Quercus ilex", "Quercus suber", "Stipa tenacissima",
        "Pistacia lentiscus", "Arbutus unedo", "Tetraclinis articulata", 
        "Cistus monspeliensis", "Calicotome villosa", "Juniperus phoenicea",
        "Erica arborea", "Olea europaea", "Barren"
    ]
    num_classes = len(classes)
    
    x_train = np.random.random((500, 64, 64, 3)).astype(np.float32)
    y_train = np.random.randint(0, num_classes, size=(500,))
    
    # 2. Build more robust CNN
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # 3. Compile Model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # 4. Train
    logging.info("Fitting model...")
    model.fit(x_train, y_train, epochs=1, verbose=0)
    
    # 5. Save baseline
    model_path = "models/vision_baseline.keras"
    model.save(model_path)
    logging.info(f"Saved baseline model to {model_path}")
    
    # 6. "Reduce" (Quantize to TFLite INT8)
    logging.info("Reducing model size via TFLite INT8 Quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset for quantization (using part of training data)
    def representative_data_gen():
        for i in range(20):
            yield [x_train[i:i+1]]
    
    converter.representative_dataset = representative_data_gen
    # Ensure fully quantized
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    tflite_path = "models/vision_quantized.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    logging.info(f"Saved quantized model to {tflite_path}")

def train_risk_model():
    logging.info("--- Training Risk Analyzer (XGBoost) ---")
    
    # 1. Generate Synthetic Data
    # Features: burned_pct, spread_rate, intensity, gas_ppm
    x_train = np.random.rand(200, 4)
    y_train = np.random.randint(0, 2, size=(200,)) # Binary risk: 0 (Moderate), 1 (Extreme)
    
    # 2. Train XGBoost
    clf = xgb.XGBClassifier(n_estimators=10, max_depth=3, learning_rate=0.1)
    logging.info("Fitting XGBoost classifier...")
    clf.fit(x_train, y_train)
    
    # 3. Save Model
    model_path = "models/regional_risk.json"
    clf.save_model(model_path)
    logging.info(f"Saved risk model to {model_path}")

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    train_vision_model()
    train_risk_model()
    logging.info("Model pipeline complete.")
