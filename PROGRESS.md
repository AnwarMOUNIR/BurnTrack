# Project Rover-AI: Progress & Technical Milestone Report
*Last Updated: April 28, 2026 | Prepared for: S6 Soutenance (Presentation Tomorrow)*

## 1. Accomplishments & System State

### A. AI & Computer Vision (Edge-Optimized)
- **Exhaustive Taxonomy:** Implemented a **13-class** botanical vision model specifically for North African ecosystems. This covers >90% of the dominant fire-carrying biomass in the Maghreb.
  - *Key Species:* Pinus halepensis, Stipa tenacissima (Alfa), Quercus suber (Cork Oak), Tetraclinis articulata, Argania spinosa, etc.
- **TFLite INT8 Quantization:** Successfully reduced the model from 32-bit floats to **8-bit integers**.
  - *Impact:* ~4x reduction in size and significantly lower CPU usage on Raspberry Pi 4.
- **Inference Strategy:** Configured for **0.5 FPS** (one frame per 2 seconds) to preserve CPU headroom for motor kinematics and GPS fusion.
- **Verification:** Automated evaluation script (`src/evaluate_models.py`) added to track Precision, Recall, and F1-Score.

### B. Regional Risk Analysis
- **XGBoost Integration:** Deployed an XGBoost classifier for regional risk assessment. It is extremely lightweight and handles multi-factor inference (Vision Output + Sensor Data + Simulation Results).
- **Tabular Data Bridge:** The model evaluates 4 critical parameters: Burn Area %, Spread Rate, Intensity, and MQ135 Gas PPM.

### C. Botanical Knowledge Base
- **Physics-Based Matrix:** Created `src/database/botanical_db.py` containing real-world Mediterranean fuel metrics:
  - **SAV Ratio:** Surface-area-to-volume ratio for spread speed.
  - **Heat Release:** Caloric content for fire intensity.
  - **Ignition Delay:** Time-to-ignition based on plant chemistry (terpenes/oils).

### D. Simulation Engine (CA)
- **Cellular Automata:** A 50x50 physics grid that simulates fire propagation in forward-time, taking wind speed, slope, and botanical metrics as inputs.

### E. Macro-Perception (Satellite Integration)
- **Sentinel-2 Pipeline:** Added `src/vision/satellite_processor.py` to incorporate macro-level vegetation indices (NDVI/NBR).
- **Multi-Scale AI:** The system now compares **Micro-Vision** (Rover Camera) with **Macro-Vision** (Satellite Maps) to validate fuel classification and plan data collection routes.
- **Mission Planning:** Implemented logic to suggest high-priority data collection points based on regional fuel density and historical aridity.

---

## 2. Infrastructure & Environment
- **Venv Setup:** Resolved all `scikit-learn` and `tensorflow` dependency issues.
- **Workflow Tools:** 
  - `src/train_models.py`: One-command training, compilation, and reduction.
  - `src/evaluate_models.py`: Generates performance metrics and saves an explanation file (`models/metrics_explanation.txt`).
  - `src/main_orchestrator.py`: The live pipeline executing the full perception-to-inference loop.

---

## 3. Next Phase: The "Moving Model" & Field Deployment

### Phase 1: High-Precision Navigation
- **The Swerve Kinematics:** Implementation of the **12-parameter motor control model** (4 wheels x [Angle, Velocity, Position]).
- **GPS Path Parcour:** Integrating micro-coordinate path following where the rover chooses motor parameters dynamically to stay on path despite terrain friction.

### Phase 2: Offline Knowledge Retrieval (RAG)
- **Local DuckDB Deployment:** Replacing the dictionary-based flora matrix with a local geospatial DuckDB instance to handle mission-specific topographic maps without internet.

### Phase 3: Hardware Sensor Fusion
- **MQ135 & Wind Anemometer:** Calibrating real-time analog inputs into the XGBoost feature vector for "Live Risk" updates during the parcour.

### Phase 4: LoRa Semantic Telemetry
- **Payload Optimization:** Designing the condensed hex payload for long-range transmission back to the base station in dead zones.

---

## 4. Quick-Run Commands (For Demo)
- **Train & Optimize:** `python3 src/train_models.py`
- **Evaluate & Explain:** `python3 src/evaluate_models.py`
- **Run Full Pipeline:** `python3 src/main_orchestrator.py`
