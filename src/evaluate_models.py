import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import os

from train_models import generate_spectral_images

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def evaluate_vision():
    logging.info("--- Evaluating Exhaustive Vision Model (13 Classes) ---")
    interpreter = tf.lite.Interpreter(model_path="models/vision_quantized.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Generate spectrally realistic test data
    x_test, y_true = generate_spectral_images(n_per_class=10) # 130 images
    y_pred = []

    for i in range(len(x_test)):
        input_dtype = input_details[0]["dtype"]
        if input_dtype == np.uint8:
            # For UINT8, tflite expects 0-255
            input_data = (x_test[i:i+1] * 255).astype(np.uint8)
        elif input_dtype == np.int8:
            scale = input_details[0]["quantization_parameters"]["scales"][0]
            zp = input_details[0]["quantization_parameters"]["zero_points"][0]
            input_data = (x_test[i:i+1] / scale + zp).astype(np.int8)
        else:
            input_data = x_test[i:i+1].astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        y_pred.append(np.argmax(output))

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return metrics

def evaluate_risk():
    logging.info("--- Evaluating Risk Model (XGBoost) ---")
    model = xgb.XGBClassifier()
    model.load_model("models/regional_risk.json")

    # Generate synthetic features
    # features: burned_pct, spread_rate, intensity, gas_ppm
    x_test = np.random.rand(100, 4)
    x_test[:, 0] *= 100 # burned pct
    x_test[:, 1] *= 50  # spread
    x_test[:, 2] *= 3000 # intensity
    x_test[:, 3] = x_test[:, 3] * 900 + 300 # gas
    
    # Random 4-class true labels
    y_true = np.random.randint(0, 4, size=(100,))
    y_pred = model.predict(x_test)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return metrics

def save_explanation():
    content = """Model Evaluation Metrics Explanation
======================================

1. Accuracy:
   - What it means: The percentage of total predictions that were correct.
   - Use case: Best when classes are balanced (e.g., equal amounts of Aleppo Pine and Alfa Grass).

2. Precision:
   - What it means: Of all the times the model predicted a certain class (e.g., 'Extreme Risk'), how often was it actually correct?
   - Use case: High precision means fewer "False Alarms".

3. Recall (Sensitivity):
   - What it means: Of all the real instances of a class (e.g., real 'Extreme Risk'), how many did the model successfully catch?
   - Use case: In fire safety, Recall is CRITICAL. You don't want to miss a real fire risk.

4. F1-Score:
   - What it means: The harmonic mean of Precision and Recall.
   - Use case: Provides a single balance between not having false alarms (Precision) and not missing real events (Recall).

Note on North African Fuel Models:
The vision model evaluates 13 classes covering North African fire-carrying biomass.
The Risk model evaluates 4-class severity: Low, Moderate, High, Extreme.

Note on Synthetic Data:
Current metrics are computed against synthetic (spectral/physics-based) test data to validate the
pipeline architecture. Metrics will become fully representative once field-collected imagery
is used for training (Phase 2).
"""
    with open("models/metrics_explanation.txt", "w") as f:
        f.write(content)
    logging.info("Saved metrics explanation to models/metrics_explanation.txt")

if __name__ == "__main__":
    if not os.path.exists("models/vision_quantized.tflite") or not os.path.exists("models/regional_risk.json"):
        logging.error("Models not found. Run training first.")
    else:
        v_metrics = evaluate_vision()
        r_metrics = evaluate_risk()
        save_explanation()

        print("\n--- VISION METRICS ---")
        for k, v in v_metrics.items(): print(f"{k}: {v:.4f}")
        
        print("\n--- RISK METRICS ---")
        for k, v in r_metrics.items(): print(f"{k}: {v:.4f}")
