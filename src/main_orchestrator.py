"""
BurnTrack Main Orchestrator — ties together all subsystems.

Pipeline flow:
    1. Vision Node → fuel type + NGRDI aridity (#7)
    2. Satellite Processor → biome-aware NDVI/NBR (#3)
    3. Botanical Database → Rothermel fuel metrics
    4. Hexagonal CA Engine → ensemble fire simulation (#4, #5)
    5. Risk Analyzer → 4-class XGBoost risk (#2, #6)
    6. LoRa Serializer → compact telemetry payload (#9)
"""

import time
import logging
from vision.vision_node import VisionNode
from vision.satellite_processor import SatelliteProcessor
from database.botanical_db import BotanicalDatabase
from simulation.ca_engine import CellularAutomataEngine
from analyzer.risk_analyzer import AIRiskAnalyzer
from telemetry.lora_serializer import LoRaSerializer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_rover_pipeline():
    logging.info("--- Starting Autonomous Forecasting Loop ---")

    # Initialize all subsystems
    vision = VisionNode()
    satellite = SatelliteProcessor()
    botanical_db = BotanicalDatabase()
    physics_engine = CellularAutomataEngine()
    risk_analyzer = AIRiskAnalyzer()
    lora = LoRaSerializer()

    # --- Rover's current position (simulated GPS) ---
    current_lat = 36.70
    current_lon = 3.00

    # 1. Satellite macro-data (#3: biome-aware NDVI)
    sat_data = satellite.get_vegetation_indices(current_lat, current_lon)

    # 2. Vision micro-perception (#7: NGRDI aridity)
    logging.info("Taking camera snapshot...")
    vision_results = vision.analyze_frame(frame="dummy_image_data")
    logging.info(
        f"Identified {vision_results['aridity_state']} "
        f"{vision_results['fuel_type']} "
        f"(NGRDI={vision_results['ngrdi_value']:.3f}). Querying DB..."
    )

    # 3. Botanical knowledge retrieval
    flora_metrics = botanical_db.query_flammability(
        vision_results["fuel_type"],
        vision_results["aridity_state"],
    )

    # 4. Sensor readings (simulated hardware)
    current_wind = 5.0   # m/s from anemometer
    current_slope = 10.0  # degrees from IMU
    current_gas_ppm = 850  # ppm from MQ135

    # 5. Ensemble CA simulation (#4: hex grid, #5: ensemble)
    damage_metrics = physics_engine.run_ensemble(
        current_wind, current_slope, flora_metrics,
        steps=80, n_runs=5,
    )

    # 6. AI risk assessment (#2: self-trained, #6: 4-class)
    final_risk = risk_analyzer.evaluate_risk(damage_metrics, current_gas_ppm)
    logging.info("=" * 40)
    logging.info(f"FINAL PREDICTED RISK: {final_risk}")
    logging.info("=" * 40)

    # 7. LoRa telemetry (#9: semantic payload)
    payload = lora.build_payload(
        lat=current_lat, lon=current_lon,
        vision_result=vision_results,
        ca_metrics=damage_metrics,
        risk_label=final_risk,
        gas_ppm=current_gas_ppm,
    )
    logging.info(
        f"LoRa Payload ({payload['size_bytes']}B): {payload['json']}"
    )

    # --- Navigation suggestions ---
    logging.info("Scanning 5km radius for high-risk data collection targets...")
    targets = [
        {"lat": 36.71, "lon": 3.01,
         "priority": "High", "note": "Dense Pinus halepensis Cluster"},
        {"lat": 36.68, "lon": 3.015,
         "priority": "Medium", "note": "Dry Stipa Steppe"},
    ]
    for t in targets:
        logging.info(
            f"SUGGESTED NEXT POINT: {t['lat']}, {t['lon']} | "
            f"Priority: {t['priority']} ({t['note']})"
        )


if __name__ == "__main__":
    run_rover_pipeline()
