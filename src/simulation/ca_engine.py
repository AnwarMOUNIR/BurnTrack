"""
Stochastic Cellular Automata Engine for Wildfire Simulation.

Implements a hexagonal-grid CA model with the propagation probability
from the project's research paper (§7.2):

    p_burn = p0 * (1 + p_veg) * p_w * p_s

Hexagonal grids provide equidistant neighbors, eliminating the anisotropic
artifacts of square grids where diagonal distance = sqrt(2) (§6.1).

Uses cube coordinates (q, r, s) where q + r + s = 0 for efficient
hex grid indexing (Red Blob Games implementation guide, ref [71]).

Supports ensemble simulation: running N stochastic trials and averaging
results for stable, defensible risk metrics (ref: old soutenance
"simulations d'ensemble").

References:
    [35] NHESS cellular automata wildfire models
    [37] Universidade de Lisboa CA propagation model
    [64] HexFire: A Flexible and Accessible Wildfire Simulator (PMC)
    [66] NHESS fire prevention using CA
    [71] Red Blob Games hex grid implementation
    [76] AIMS Press stochastic CA with heterogeneous vegetation
"""

import numpy as np
import logging

# Empirical tuning coefficients from CA wildfire literature
_C1 = 0.045       # Base wind acceleration coefficient
_C2 = 0.131       # Directional wind sensitivity coefficient
_A_SLOPE = 0.078  # Slope acceleration coefficient
_P0 = 0.58        # Baseline ignition probability


class CellularAutomataEngine:
    """
    Stochastic Cellular Automata engine on a hexagonal grid for
    forward-time fire propagation simulation.

    Cell states:
        0 = S0 (Unburned, combustible)
        1 = S1 (Actively Burning)
        2 = S2 (Burned out / Ashes)

    The hex grid is stored as a 2D array using offset coordinates
    (odd-r layout) for storage, with neighbor lookups adjusted for
    the hex topology. Each hex cell has exactly 6 equidistant neighbors.
    """

    # Odd-r hex neighbor offsets: 6 directions
    # Even rows and odd rows have different column offsets
    _EVEN_ROW_NEIGHBORS = [
        (-1, 0), (-1, -1),  # NE, NW
        (0, -1), (0, 1),    # W, E
        (1, 0),  (1, -1),   # SE, SW
    ]
    _ODD_ROW_NEIGHBORS = [
        (-1, 1), (-1, 0),   # NE, NW
        (0, -1), (0, 1),    # W, E
        (1, 1),  (1, 0),    # SE, SW
    ]

    # Directional angles for each neighbor (radians from East)
    # Approximated for flat-top hex orientation
    _NEIGHBOR_ANGLES = [
        np.pi / 6,      # NE  (30°)
        5 * np.pi / 6,  # NW  (150°)
        np.pi,          # W   (180°)
        0.0,            # E   (0°)
        -np.pi / 6,     # SE  (-30° / 330°)
        -5 * np.pi / 6, # SW  (-150° / 210°)
    ]

    def __init__(self, grid_size=(50, 50)):
        self.rows, self.cols = grid_size
        logging.info(
            f"Initializing Hexagonal CA Physics Grid "
            f"({self.rows}x{self.cols}, {self.rows * self.cols} cells)..."
        )

    def _get_neighbors(self, r, c):
        """Return list of (row, col, angle) for valid hex neighbors."""
        offsets = (
            self._ODD_ROW_NEIGHBORS if r % 2 == 1
            else self._EVEN_ROW_NEIGHBORS
        )
        neighbors = []
        for i, (dr, dc) in enumerate(offsets):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc, self._NEIGHBOR_ANGLES[i]))
        return neighbors

    @staticmethod
    def _vegetation_factor(flora_metrics):
        """
        Compute p_veg from the botanical database metrics.
        Combines burn_rate with normalized SAV ratio.
        """
        burn_rate = flora_metrics.get("burn_rate", 0.5)
        sav_ratio = flora_metrics.get("sav_ratio", 50)
        sav_norm = min(sav_ratio / 90.0, 1.0)
        return burn_rate * (0.4 + 0.6 * sav_norm)

    @staticmethod
    def _wind_factor(wind_speed, wind_angle_rad, spread_angle):
        """
        Compute p_w using the exponential wind model:
            p_w = exp(V * (c1 + c2 * (cos(theta) - 1)))

        theta = angle between wind direction and spread direction.
        """
        if wind_speed < 0.01:
            return 1.0
        theta = spread_angle - wind_angle_rad
        return np.exp(wind_speed * (_C1 + _C2 * (np.cos(theta) - 1)))

    @staticmethod
    def _slope_factor(slope_deg, spread_angle):
        """
        Compute p_s: uphill spread is exponentially accelerated.
        Uphill direction is assumed to be North (pi/2).
        """
        if abs(slope_deg) < 0.5:
            return 1.0
        slope_rad = np.radians(abs(slope_deg))
        tan_slope = np.tan(slope_rad)
        # Component of spread in the uphill direction
        uphill_component = np.sin(spread_angle)
        return np.exp(_A_SLOPE * tan_slope * uphill_component)

    def run_simulation(self, wind_speed, slope, flora_metrics,
                       steps=100, wind_direction=0.0):
        """
        Execute a single stochastic CA fire propagation simulation.

        Args:
            wind_speed: Wind speed in m/s.
            slope: Terrain slope in degrees.
            flora_metrics: Dict from BotanicalDatabase.
            steps: Number of discrete time steps.
            wind_direction: Wind direction in radians (0 = East).

        Returns:
            Dict with total_area_burned_pct, rate_of_spread_m_min,
            peak_intensity.
        """
        grid = np.zeros((self.rows, self.cols), dtype=np.int8)

        # Ignite center cell
        cx, cy = self.rows // 2, self.cols // 2
        grid[cx, cy] = 1

        p_veg = self._vegetation_factor(flora_metrics)
        heat_release = flora_metrics.get("heat_release", 15000)
        peak_burning = 0
        active_steps = 0

        for step in range(steps):
            new_grid = grid.copy()
            burning = 0

            for r in range(self.rows):
                for c in range(self.cols):
                    if grid[r, c] != 1:
                        continue

                    burning += 1
                    new_grid[r, c] = 2  # Burn out

                    for nr, nc, angle in self._get_neighbors(r, c):
                        if grid[nr, nc] != 0:
                            continue

                        pw = self._wind_factor(
                            wind_speed, wind_direction, angle
                        )
                        ps = self._slope_factor(slope, angle)
                        p_burn = min(_P0 * (1.0 + p_veg) * pw * ps, 0.99)

                        if np.random.random() < p_burn:
                            new_grid[nr, nc] = 1

            peak_burning = max(peak_burning, burning)
            if burning > 0:
                active_steps += 1

            grid = new_grid

            # Early termination
            if burning == 0 and np.sum(grid == 1) == 0:
                break

        # --- Metrics ---
        total_cells = self.rows * self.cols
        burned = np.sum(grid >= 1)
        burned_pct = (burned / total_cells) * 100.0

        if active_steps > 0:
            positions = np.argwhere(grid >= 1)
            if len(positions) > 0:
                dists = np.sqrt(
                    (positions[:, 0] - cx) ** 2 +
                    (positions[:, 1] - cy) ** 2
                )
                spread_rate = (np.max(dists) / active_steps) * 60.0
            else:
                spread_rate = 0.0
        else:
            spread_rate = 0.0

        peak_intensity = peak_burning * (heat_release / 1000.0)

        return {
            "total_area_burned_pct": round(burned_pct, 1),
            "rate_of_spread_m_min": round(spread_rate, 1),
            "peak_intensity": round(peak_intensity, 1),
        }

    def run_ensemble(self, wind_speed, slope, flora_metrics,
                     steps=100, wind_direction=0.0, n_runs=10):
        """
        Run N stochastic simulations and return averaged metrics.

        Ensemble averaging smooths stochastic variance and produces
        stable, defensible risk scores (ref: old soutenance
        "simulations d'ensemble").

        Args:
            n_runs: Number of independent simulation runs.
            (other args: same as run_simulation)

        Returns:
            Dict with averaged metrics + std deviations.
        """
        logging.info(
            f"Running {n_runs}-trial ensemble "
            f"(wind={wind_speed}m/s, slope={slope}°, "
            f"fuel_burn_rate={flora_metrics.get('burn_rate', '?')})..."
        )

        results = []
        for _ in range(n_runs):
            r = self.run_simulation(
                wind_speed, slope, flora_metrics,
                steps=steps, wind_direction=wind_direction,
            )
            results.append(r)

        # Average and std
        keys = ["total_area_burned_pct", "rate_of_spread_m_min",
                "peak_intensity"]
        avg = {}
        for k in keys:
            vals = [r[k] for r in results]
            avg[k] = round(np.mean(vals), 1)
            avg[f"{k}_std"] = round(np.std(vals), 1)

        logging.info(
            f"Ensemble complete: {avg['total_area_burned_pct']}% burned "
            f"(±{avg['total_area_burned_pct_std']}%), "
            f"spread={avg['rate_of_spread_m_min']} m/min, "
            f"intensity={avg['peak_intensity']} kW/m"
        )

        return avg
