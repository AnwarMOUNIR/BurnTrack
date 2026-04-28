"""
Offline Botanical Knowledge Base for North African fire-carrying flora.

Contains physics-based fuel metrics used by the Cellular Automata engine
and the XGBoost risk analyzer. Parameters align with the Rothermel surface
fire spread model inputs as documented in the project research paper
(Soutenance/Rover Control, Sensing, and Simulation.pdf, §7.2).

Metrics per species:
    burn_rate       : Relative flammability probability [0, 1].
    ignition_delay  : Approximate time-to-ignition under radiant heat (s).
    heat_release    : Heat of combustion (kJ/kg dry mass).
                      Mediterranean vegetation: ~16,000–22,000 kJ/kg.
    sav_ratio       : Surface-area-to-volume ratio (cm^-1).
                      Drives heat transfer rate. Grass ~60–90, trees ~30–55.
    fuel_load       : Dry biomass density (kg/m^2).
                      Rothermel parameter ω₀ from old soutenance.
    fuel_depth      : Fuel bed depth (m).
                      Rothermel parameter δ from old soutenance.

Sources:
    - SAV for Pinus halepensis: 52–59 cm^-1 (Mediterranean pine studies, MDPI)
    - Quercus suber ignition: 61–118s TTI (ResearchGate cork bark thermal studies)
    - Heat of combustion range: 18–20 MJ/kg standard (USDA wildfire fuel data)
    - 3D Fuels IFPL database for bulk density and fuel depth baselines
    - Stipa tenacissima fire behavior from North African steppe literature
"""

import logging


class BotanicalDatabase:
    def __init__(self, db_path="data/botanical/local_flora.duckdb"):
        self.db_path = db_path

        # Physics-based fuel metrics for North African / Mediterranean species
        # Ordered roughly by SAV (fire-spread dominance) from high to low
        self.flora_matrix = {
            # --- Grasses & fine fuels (highest SAV, fastest spread) ---
            "Stipa tenacissima": {
                "burn_rate": 0.98, "ignition_delay": 1.5,
                "heat_release": 16000, "sav_ratio": 90,
                "fuel_load": 0.4, "fuel_depth": 0.6,
            },

            # --- Shrubs & maquis (high SAV, volatile oils) ---
            "Cistus monspeliensis": {
                "burn_rate": 0.96, "ignition_delay": 2.1,
                "heat_release": 21000, "sav_ratio": 85,
                "fuel_load": 1.2, "fuel_depth": 1.0,
            },
            "Erica arborea": {
                "burn_rate": 0.87, "ignition_delay": 3.5,
                "heat_release": 19000, "sav_ratio": 80,
                "fuel_load": 1.0, "fuel_depth": 1.5,
            },
            "Calicotome villosa": {
                "burn_rate": 0.88, "ignition_delay": 3.2,
                "heat_release": 17000, "sav_ratio": 75,
                "fuel_load": 0.9, "fuel_depth": 1.2,
            },
            "Pistacia lentiscus": {
                "burn_rate": 0.92, "ignition_delay": 4.5,
                "heat_release": 20000, "sav_ratio": 65,
                "fuel_load": 1.4, "fuel_depth": 1.8,
            },
            "Juniperus phoenicea": {
                "burn_rate": 0.94, "ignition_delay": 4.8,
                "heat_release": 20500, "sav_ratio": 60,
                "fuel_load": 1.3, "fuel_depth": 2.5,
            },

            # --- Conifers (moderate SAV, high heat, crown fire risk) ---
            "Pinus halepensis": {
                "burn_rate": 0.95, "ignition_delay": 6.8,
                "heat_release": 19000, "sav_ratio": 55,
                "fuel_load": 1.8, "fuel_depth": 0.3,
            },
            "Tetraclinis articulata": {
                "burn_rate": 0.90, "ignition_delay": 5.0,
                "heat_release": 19500, "sav_ratio": 50,
                "fuel_load": 1.5, "fuel_depth": 0.4,
            },

            # --- Broadleaf trees (low SAV, fire-resistant bark/leaves) ---
            "Arbutus unedo": {
                "burn_rate": 0.85, "ignition_delay": 12.0,
                "heat_release": 18500, "sav_ratio": 45,
                "fuel_load": 1.1, "fuel_depth": 0.5,
            },
            "Quercus suber": {
                # Cork oak: thick bark provides exceptional fire insulation.
                # TTI = 61–118s in radiant heat tests; bark belly reaches
                # lethal temp at ~230s. Using 90s as mid-range TTI.
                "burn_rate": 0.65, "ignition_delay": 90.0,
                "heat_release": 18000, "sav_ratio": 38,
                "fuel_load": 1.6, "fuel_depth": 0.3,
            },
            "Quercus ilex": {
                "burn_rate": 0.70, "ignition_delay": 20.0,
                "heat_release": 17500, "sav_ratio": 35,
                "fuel_load": 1.5, "fuel_depth": 0.4,
            },
            "Olea europaea": {
                # Olive: dense wood, high oil content → high heat of combustion
                # but low flammability due to thick leaves and low SAV.
                "burn_rate": 0.60, "ignition_delay": 15.0,
                "heat_release": 22000, "sav_ratio": 30,
                "fuel_load": 1.2, "fuel_depth": 0.3,
            },

            # --- Sentinel class ---
            "Barren": {
                "burn_rate": 0.0, "ignition_delay": 999,
                "heat_release": 0, "sav_ratio": 0,
                "fuel_load": 0.0, "fuel_depth": 0.0,
            },
        }

    def query_flammability(self, fuel_type, aridity_state):
        """
        Retrieve fuel metrics and apply aridity/moisture modifier.

        Aridity states (from vision model):
            'Dead'  : DFMC < 10%. Fully desiccated, maximum combustibility.
            'cured' : Transitional. Senescent vegetation, moderate moisture.
            'Green' : Live fuel, high moisture content, fire-resistant.

        The aridity multiplier adjusts burn_rate, which feeds directly
        into the CA engine's p_veg vegetation factor.
        """
        metrics = self.flora_matrix.get(
            fuel_type, self.flora_matrix["Barren"]
        ).copy()

        # Apply aridity multiplier (affects p_veg in CA propagation)
        if aridity_state == "Dead":
            metrics["burn_rate"] *= 1.2
        elif aridity_state == "cured":
            metrics["burn_rate"] *= 1.0  # Transitional — no modifier
        elif aridity_state == "Green":
            metrics["burn_rate"] *= 0.6

        # Cap burn_rate at 1.0 (it is a probability)
        metrics["burn_rate"] = min(metrics["burn_rate"], 1.0)

        return metrics
