"""
CA Grid Exporter — dumps simulation state to JSON for the 3D visualizer.
"""

import json
import os
import numpy as np
import logging

from simulation.ca_engine import CellularAutomataEngine
from database.botanical_db import BotanicalDatabase


def export_simulation(output_path="viz/simulation_data.json",
                      grid_size=(30, 30), steps=60):
    """
    Run a CA simulation and export each step's grid state as JSON
    for the Three.js 3D visualizer.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    db = BotanicalDatabase()
    flora = db.query_flammability("Stipa tenacissima", "Dead")

    engine = CellularAutomataEngine(grid_size=grid_size)
    rows, cols = grid_size

    # We need to capture per-step state, so we run manually
    grid = np.zeros((rows, cols), dtype=np.int8)
    cx, cy = rows // 2, cols // 2
    grid[cx, cy] = 1

    p_veg = engine._vegetation_factor(flora)
    frames = [grid.copy().tolist()]

    wind_speed = 8.0
    slope = 15.0

    for step in range(steps):
        new_grid = grid.copy()
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 1:
                    continue
                new_grid[r, c] = 2
                for nr, nc, angle in engine._get_neighbors(r, c):
                    if grid[nr, nc] != 0:
                        continue
                    pw = engine._wind_factor(wind_speed, 0.0, angle)
                    ps = engine._slope_factor(slope, angle)
                    p_burn = min(0.58 * (1.0 + p_veg) * pw * ps, 0.99)
                    if np.random.random() < p_burn:
                        new_grid[nr, nc] = 1

        grid = new_grid
        frames.append(grid.copy().tolist())

        if np.sum(grid == 1) == 0:
            break

    data = {
        "rows": rows,
        "cols": cols,
        "frames": frames,
        "params": {
            "wind_speed": wind_speed,
            "slope": slope,
            "fuel": "Stipa tenacissima",
            "aridity": "Dead",
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)

    logging.info(
        f"Exported {len(frames)} frames to {output_path} "
        f"({os.path.getsize(output_path) / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    export_simulation()
