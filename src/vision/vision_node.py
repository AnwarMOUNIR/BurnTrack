import logging
import tensorflow as tf
import numpy as np

class VisionNode:
    def __init__(self, model_path="models/vision_quantized.tflite"):
        self.model_path = model_path
        logging.info(f"Initializing Edge Vision Node with {model_path}...")
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def analyze_frame(self, frame):
        """
        Takes an RGB frame, runs inference.
        """
        # Prepare dummy input if frame is a string
        if isinstance(frame, str):
            # Scale to 0-255 and cast to uint8 for INT8 model
            input_data = (np.random.random((1, 64, 64, 3)) * 255).astype(np.uint8)
        else:
            # Ensure input is uint8
            input_data = frame.astype(np.uint8)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Mapping predicted class back to labels
        classes = [
            "Pinus halepensis", "Quercus ilex", "Quercus suber", "Stipa tenacissima",
            "Pistacia lentiscus", "Arbutus unedo", "Tetraclinis articulata", 
            "Cistus monspeliensis", "Calicotome villosa", "Juniperus phoenicea",
            "Erica arborea", "Olea europaea", "Barren"
        ]
        fuel_type = classes[np.argmax(output_data)]
        
        return {"fuel_type": fuel_type, "aridity_state": "cured"}
