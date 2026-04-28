import logging
import xgboost as xgb
import numpy as np

class AIRiskAnalyzer:
    def __init__(self, model_path="models/regional_risk.json"):
        self.model_path = model_path
        logging.info(f"Loading XGBoost Risk Classification Bridge from {model_path}...")
        self.model = xgb.XGBClassifier()
        self.model.load_model(self.model_path)

    def evaluate_risk(self, ca_damage_metrics, current_gas_ppm):
        """
        Takes the physics output and micro-sensors, evaluates final risk.
        """
        # Prepare input features: burned_pct, spread_rate, intensity, gas_ppm
        features = np.array([[
            ca_damage_metrics["total_area_burned_pct"],
            ca_damage_metrics["rate_of_spread_m_min"],
            ca_damage_metrics["peak_intensity"],
            current_gas_ppm
        ]])
        
        prediction = self.model.predict(features)[0]
        return "Extreme" if prediction == 1 else "Moderate"
