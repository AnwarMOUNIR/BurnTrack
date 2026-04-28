"""
Edge Vision Node — TFLite CNN inference with NGRDI aridity detection.

Improvement #1: Better synthetic training data with realistic spectral
signatures per vegetation type (not pure random noise).

Improvement #7: NGRDI (Normalized Green-Red Difference Index) computed
directly from RGB frames for aridity assessment, as described in the
research paper §4.2:

    NGRDI = (Green - Red) / (Green + Red)

    - Positive NGRDI → healthy, photosynthetic vegetation ("Green")
    - Near-zero NGRDI → transitional, senescent vegetation ("cured")
    - Negative NGRDI → dead, highly combustible vegetation ("Dead")

This replaces the hardcoded aridity placeholder with actual computed
values from camera frames.
"""

import logging
import os

import numpy as np

# Mapping from model output index to species name
# Must match the training label order in train_models.py
CLASS_NAMES = [
    "Pinus halepensis", "Quercus ilex", "Quercus suber",
    "Stipa tenacissima", "Pistacia lentiscus", "Arbutus unedo",
    "Tetraclinis articulata", "Cistus monspeliensis",
    "Calicotome villosa", "Juniperus phoenicea",
    "Erica arborea", "Olea europaea", "Barren",
]


class VisionNode:
    def __init__(self, model_path="models/vision_quantized.tflite"):
        self.model_path = model_path
        logging.info(f"Initializing Edge Vision Node with {model_path}...")

        if os.path.exists(model_path):
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(
                model_path=self.model_path
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self._model_loaded = True
        else:
            logging.warning(
                f"Vision model not found at {model_path}. "
                f"Run train_models.py first."
            )
            self._model_loaded = False

    @staticmethod
    def compute_ngrdi(frame):
        """
        Compute the Normalized Green-Red Difference Index from an RGB frame.

        NGRDI = (G - R) / (G + R)

        Returns the mean NGRDI value across the frame and the inferred
        aridity state.
        """
        frame_f = frame.astype(np.float32)
        red = frame_f[:, :, 0]
        green = frame_f[:, :, 1]

        # Avoid division by zero
        denominator = green + red
        denominator = np.where(denominator == 0, 1, denominator)

        ngrdi = (green - red) / denominator
        mean_ngrdi = float(np.mean(ngrdi))

        # Classify aridity from NGRDI (thresholds from §4.2)
        if mean_ngrdi > 0.05:
            aridity = "Green"
        elif mean_ngrdi > -0.05:
            aridity = "cured"
        else:
            aridity = "Dead"

        return mean_ngrdi, aridity

    def analyze_frame(self, frame):
        """
        Run TFLite inference on an RGB frame for fuel type classification,
        then compute NGRDI for aridity assessment.

        Args:
            frame: numpy array of shape (H, W, 3) in RGB uint8,
                   or string "dummy_image_data" for demo mode.

        Returns:
            Dict with fuel_type, aridity_state, confidence, ngrdi_value.
        """
        # Generate a demo frame if running in simulation mode
        if isinstance(frame, str):
            frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        # --- NGRDI Aridity Detection (Improvement #7) ---
        ngrdi_value, aridity_state = self.compute_ngrdi(frame)

        # --- Fuel Type Classification ---
        if not self._model_loaded:
            return {
                "fuel_type": "Stipa tenacissima",
                "aridity_state": aridity_state,
                "confidence": 0.0,
                "ngrdi_value": round(ngrdi_value, 4),
            }

        # Preprocess: resize to model input shape and normalize
        import cv2
        input_shape = self.input_details[0]["shape"]
        h, w = input_shape[1], input_shape[2]
        resized = cv2.resize(frame, (w, h))

        input_dtype = self.input_details[0]["dtype"]
        if input_dtype == np.float32:
            input_data = resized.astype(np.float32) / 255.0
        elif input_dtype == np.int8:
            # INT8 quantized model
            scale = self.input_details[0]["quantization_parameters"]["scales"]
            zp = self.input_details[0]["quantization_parameters"]["zero_points"]
            input_data = (resized.astype(np.float32) / 255.0 / scale[0] + zp[0])
            input_data = input_data.astype(np.int8)
        else:
            input_data = resized.astype(np.uint8)

        input_data = np.expand_dims(input_data, axis=0)

        self.interpreter.set_tensor(
            self.input_details[0]["index"], input_data
        )
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )
        output_data = output_data.flatten().astype(np.float32)

        # Dequantize if needed
        if self.output_details[0].get("quantization_parameters"):
            params = self.output_details[0]["quantization_parameters"]
            if len(params.get("scales", [])) > 0:
                output_data = (
                    (output_data - params["zero_points"][0]) *
                    params["scales"][0]
                )

        pred_idx = int(np.argmax(output_data))
        confidence = float(np.max(output_data))

        if 0 <= pred_idx < len(CLASS_NAMES):
            fuel_type = CLASS_NAMES[pred_idx]
        else:
            fuel_type = "Barren"

        return {
            "fuel_type": fuel_type,
            "aridity_state": aridity_state,
            "confidence": round(confidence, 4),
            "ngrdi_value": round(ngrdi_value, 4),
        }
