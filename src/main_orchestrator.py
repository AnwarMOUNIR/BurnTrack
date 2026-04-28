import time
import logging
from vision.vision_node import VisionNode
from vision.satellite_processor import SatelliteDataProcessor
from database.botanical_db import BotanicalDatabase
from simulation.ca_engine import CellularAutomataEngine
from analyzer.risk_analyzer import AIRiskAnalyzer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_rover_pipeline():
    logging.info("--- Starting Autonomous Forecasting Loop ---")
    
    # Initialize all modules
    vision = VisionNode()
    satellite = SatelliteDataProcessor()
    botanical_db = BotanicalDatabase()
    physics_engine = CellularAutomataEngine()
    risk_analyzer = AIRiskAnalyzer()

    # 0. Macro-Planning (Satellite)
    logging.info("Querying Sentinel-2 Macro-Data for current coordinates...")
    macro_data = satellite.get_regional_vegetation_index(lat=36.7, lon=3.0)
    logging.info(f"Satellite Data - NDVI: {macro_data['ndvi']:.2f}, NBR: {macro_data['nbr']:.2f}")

    # 1. Vision Perception
    logging.info("Taking camera snapshot...")
    vision_results = vision.analyze_frame(frame="dummy_image_data")
    
    # 2. Database Knowledge Retrieval
    logging.info(f"Identified {vision_results['aridity_state']} {vision_results['fuel_type']}. Querying DB...")
    flora_metrics = botanical_db.query_flammability(
        vision_results['fuel_type'], 
        vision_results['aridity_state']
    )
    
    # 3. Physics Simulation
    # Assuming we pulled wind (5m/s) and slope (10 deg) from our hardware sensors
    current_wind = 5.0
    current_slope = 10.0
    damage_metrics = physics_engine.run_simulation(current_wind, current_slope, flora_metrics)
    
    # 4. AI Risk Assessment
    # Assuming we pulled 850 PPM from the MQ135
    current_gas_ppm = 850
    final_risk = risk_analyzer.evaluate_risk(damage_metrics, current_gas_ppm)
    
    logging.info("========================================")
    logging.info(f"FINAL PREDICTED RISK: {final_risk}")
    logging.info("========================================")

    # 5. Suggest Next Path
    targets = satellite.suggest_data_collection_points(center_lat=36.7, center_lon=3.0)
    for t in targets:
        logging.info(f"SUGGESTED NEXT POINT: {t['lat']}, {t['lon']} | Priority: {t['priority']} ({t['reason']})")

if __name__ == "__main__":
    run_rover_pipeline()
