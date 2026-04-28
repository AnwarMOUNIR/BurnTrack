"""
Risk Analyzer — XGBoost bridge between CA simulation output and risk labels.

Improvement #6: 4-class risk output (Low, Moderate, High, Extreme)
instead of binary, matching the architecture described in TXT.txt research.

The model is trained on actual CA simulation outputs (Improvement #2)
via the self_train() method, ensuring feature distributions match
what the model sees during live inference.
"""

import logging
import os

import numpy as np
import xgboost as xgb


RISK_LABELS = {0: "Low", 1: "Moderate", 2: "High", 3: "Extreme"}


class AIRiskAnalyzer:
    def __init__(self, model_path="models/regional_risk.json"):
        self.model_path = model_path
        self.model = xgb.XGBClassifier()

        if os.path.exists(self.model_path):
            logging.info(
                f"Loading XGBoost Risk Classification Bridge "
                f"from {model_path}..."
            )
            self.model.load_model(self.model_path)
            self._trained = True
        else:
            logging.warning(
                f"Risk model not found at {model_path}. "
                f"Run self_train() or train_models.py first."
            )
            self._trained = False

    def evaluate_risk(self, ca_damage_metrics, current_gas_ppm):
        """
        Evaluate final risk from CA simulation output + MQ135 gas reading.

        Returns one of: 'Low', 'Moderate', 'High', 'Extreme'.
        """
        features = np.array([[
            ca_damage_metrics["total_area_burned_pct"],
            ca_damage_metrics["rate_of_spread_m_min"],
            ca_damage_metrics["peak_intensity"],
            current_gas_ppm,
        ]])

        if not self._trained:
            # Deterministic fallback if model isn't trained yet
            return self._rule_based_fallback(ca_damage_metrics, current_gas_ppm)

        prediction = int(self.model.predict(features)[0])
        return RISK_LABELS.get(prediction, "Unknown")

    @staticmethod
    def _rule_based_fallback(metrics, gas_ppm):
        """Deterministic fallback when no trained model is available."""
        score = (
            metrics["total_area_burned_pct"] * 0.3 +
            metrics["rate_of_spread_m_min"] * 0.3 +
            min(metrics["peak_intensity"] / 100.0, 30) +
            min(gas_ppm / 100.0, 10)
        )
        if score > 50:
            return "Extreme"
        elif score > 35:
            return "High"
        elif score > 20:
            return "Moderate"
        return "Low"

    def self_train(self, ca_engine, botanical_db, n_samples=500):
        """
        Improvement #2: Generate training data from the CA simulation
        itself, ensuring the XGBoost model sees realistic feature
        distributions that match live inference inputs.

        Runs n_samples simulations with randomized wind, slope, fuel,
        and gas conditions. Labels are computed from a physics-based
        risk heuristic.
        """
        logging.info(
            f"Self-training risk model from {n_samples} CA simulations..."
        )

        species_list = [
            k for k in botanical_db.flora_matrix.keys() if k != "Barren"
        ]
        aridity_states = ["Dead", "cured", "Green"]

        X = []
        y = []

        for i in range(n_samples):
            # Randomize conditions
            wind = np.random.uniform(0, 20)
            slope = np.random.uniform(0, 35)
            species = np.random.choice(species_list)
            aridity = np.random.choice(aridity_states)
            gas_ppm = np.random.uniform(300, 1200)

            flora = botanical_db.query_flammability(species, aridity)

            # Run a fast single simulation (fewer steps for speed)
            result = ca_engine.run_simulation(
                wind, slope, flora, steps=60
            )

            features = [
                result["total_area_burned_pct"],
                result["rate_of_spread_m_min"],
                result["peak_intensity"],
                gas_ppm,
            ]
            X.append(features)

            # Physics-based labeling heuristic
            risk_score = (
                result["total_area_burned_pct"] * 0.3 +
                result["rate_of_spread_m_min"] * 0.3 +
                min(result["peak_intensity"] / 100.0, 30) +
                min(gas_ppm / 100.0, 10)
            )

            if risk_score > 50:
                label = 3  # Extreme
            elif risk_score > 35:
                label = 2  # High
            elif risk_score > 20:
                label = 1  # Moderate
            else:
                label = 0  # Low

            y.append(label)

        X = np.array(X)
        y = np.array(y)

        # Ensure all 4 classes are represented (add minimal samples if needed)
        for label in range(4):
            if label not in y:
                # Add a single sample for the missing class
                X = np.vstack([X, X[0:1]])
                y = np.append(y, label)

        # Create fresh 4-class XGBoost (don't reuse old 2-class model)
        self.model = xgb.XGBClassifier(
            n_estimators=50, max_depth=4,
            learning_rate=0.1, objective="multi:softmax",
            num_class=4, verbosity=0,
        )
        self.model.fit(X, y)

        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        self.model.save_model(self.model_path)
        self._trained = True

        # Log class distribution
        unique, counts = np.unique(y, return_counts=True)
        dist = {RISK_LABELS[u]: c for u, c in zip(unique, counts)}
        logging.info(f"Training complete. Class distribution: {dist}")
        logging.info(f"Saved self-trained model to {self.model_path}")
