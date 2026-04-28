"""
Stochastic Cellular Automata Engine for Wildfire Simulation.

Implements the propagation probability model documented in the project's
research paper (Soutenance/Rover Control, Sensing, and Simulation.pdf, §7):

    p_burn = p0 * (1 + p_veg) * p_w * p_s

Where:
    p0   = baseline ignition probability
    p_veg = vegetation/fuel factor (from botanical DB burn_rate & SAV)
    p_w  = wind factor: exp(V * (c1 + c2*(cos(theta) - 1)))
    p_s  = slope factor: exp(a_s * tan(alpha))

References:
    [35] NHESS cellular automata wildfire models
    [37] Universidade de Lisboa CA propagation model
    [66] NHESS fire prevention using CA
    [76] AIMS Press stochastic CA with heterogeneous vegetation
"""

import numpy as np
import logging

# Empirical tuning coefficients from CA wildfire literature
_C1 = 0.045   # Base wind acceleration coefficient
_C2 = 0.131   # Directional wind sensitivity coefficient
_A_SLOPE = 0.078  # Slope acceleration coefficient
_P0 = 0.58    # Baseline ignition probability (flat, no wind, standard fuel)


class CellularAutomataEngine:
    """
    A stochastic Cellular Automata engine for forward-time fire
    propagation simulation on a 2D grid.

    Cell states:
        0 = S0 (Unburned, combustible)
        1 = S1 (Actively Burning)
        2 = S2 (Burned out / Ashes)

    The propagation probability from a burning cell to an unburned
    neighbor is computed per the multiplicative weighting model from
    the project's documented Rothermel-inspired CA formulation.
    """

    def __init__(self, grid_size=(50, 50)):
        self.rows, self.cols = grid_size
        logging.info(f"Initializing Cellular Automata Physics Grid ({self.rows}x{self.cols})...")

    @staticmethod
    def _vegetation_factor(flora_metrics):
        """
        Compute p_veg from the botanical database metrics.

        High burn_rate and high SAV ratio => more combustible.
        The factor scales [0, ~1.5] to multiplicatively boost p0.
        """
        burn_rate = flora_metrics.get("burn_rate", 0.5)
        sav_ratio = flora_metrics.get("sav_ratio", 50)

        # Normalize SAV: typical Mediterranean range is 30–90 cm^-1
        sav_norm = min(sav_ratio / 90.0, 1.0)

        # p_veg combines the fuel's inherent flammability with its geometry
        return burn_rate * (0.4 + 0.6 * sav_norm)

    @staticmethod
    def _wind_factor(wind_speed, wind_angle_rad, dx, dy):
        """
        Compute p_w using the exponential wind model from the research:
            p_w = exp(V * (c1 + c2 * (cos(theta) - 1)))

        Where theta is the angle between the wind direction vector and
        the vector from the burning cell to the target neighbor.

        Args:
            wind_speed: Wind speed in m/s.
            wind_angle_rad: Wind direction in radians (0 = East, pi/2 = North).
            dx, dy: Offset to the neighbor cell (col, row delta).
        """
        if wind_speed < 0.01:
            return 1.0

        # Direction from burning cell to neighbor
        spread_angle = np.arctan2(dy, dx)

        # Angle between wind direction and spread direction
        theta = spread_angle - wind_angle_rad

        pw = np.exp(wind_speed * (_C1 + _C2 * (np.cos(theta) - 1)))
        return pw

    @staticmethod
    def _slope_factor(slope_deg, dy):
        """
        Compute p_s: uphill spread is exponentially accelerated,
        downhill spread is dampened.

        Uses: p_s = exp(a_s * tan(alpha)) for uphill,
              p_s = exp(-a_s * |tan(alpha)|) for downhill.

        Only the row-axis component (dy) determines uphill/downhill.
        """
        if abs(slope_deg) < 0.5:
            return 1.0

        slope_rad = np.radians(abs(slope_deg))
        tan_slope = np.tan(slope_rad)

        if dy > 0:  # Spreading uphill
            return np.exp(_A_SLOPE * tan_slope)
        elif dy < 0:  # Spreading downhill
            return np.exp(-_A_SLOPE * tan_slope)
        else:  # Lateral spread — slope has minimal effect
            return 1.0

    def run_simulation(self, wind_speed, slope, flora_metrics, steps=100):
        """
        Execute a forward-time stochastic CA fire propagation simulation.

        The simulation implements the documented propagation probability:
            p_burn = p0 * (1 + p_veg) * p_w * p_s

        Args:
            wind_speed: Wind speed in m/s.
            slope: Terrain slope in degrees.
            flora_metrics: Dict with burn_rate, sav_ratio, heat_release,
                           ignition_delay from BotanicalDatabase.
            steps: Number of discrete time steps.

        Returns:
            Dict with total_area_burned_pct, rate_of_spread_m_min,
            peak_intensity.
        """
        logging.info(
            f"Running {steps}-step CA propagation "
            f"(wind={wind_speed}m/s, slope={slope}°, "
            f"fuel={flora_metrics.get('burn_rate', '?')})..."
        )

        # --- Grid initialization ---
        grid = np.zeros((self.rows, self.cols), dtype=np.int8)

        # Ignite center cell
        cx, cy = self.rows // 2, self.cols // 2
        grid[cx, cy] = 1

        # Compute static factors once (they don't change per step)
        p_veg = self._vegetation_factor(flora_metrics)

        # Wind direction: assume wind blows East (+x direction) = 0 rad
        wind_angle_rad = 0.0

        # 8-connected neighbor offsets: (row_delta, col_delta)
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        # Diagonal distance penalty (sqrt(2) further)
        def distance_weight(dr, dc):
            return 1.0 / np.sqrt(dr * dr + dc * dc)

        # --- Track statistics ---
        heat_release = flora_metrics.get("heat_release", 15000)
        peak_burning_cells = 0
        burn_counts = []

        # --- Main simulation loop ---
        for step in range(steps):
            new_grid = grid.copy()
            burning_cells = 0

            for r in range(self.rows):
                for c in range(self.cols):
                    if grid[r, c] != 1:
                        continue

                    burning_cells += 1
                    # Burning cell transitions to burned-out
                    new_grid[r, c] = 2

                    # Attempt ignition of each unburned neighbor
                    for dr, dc in neighbors:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                            continue
                        if grid[nr, nc] != 0:
                            continue

                        # Compute propagation probability
                        pw = self._wind_factor(wind_speed, wind_angle_rad, dc, dr)
                        ps = self._slope_factor(slope, dr)
                        dw = distance_weight(dr, dc)

                        p_burn = _P0 * (1.0 + p_veg) * pw * ps * dw
                        p_burn = min(p_burn, 0.99)

                        if np.random.random() < p_burn:
                            new_grid[nr, nc] = 1

            peak_burning_cells = max(peak_burning_cells, burning_cells)
            burn_counts.append(burning_cells)
            grid = new_grid

            # Early termination if fire is fully extinguished
            if burning_cells == 0 and np.sum(grid == 1) == 0:
                logging.info(f"Fire extinguished at step {step + 1}")
                break

        # --- Compute output metrics ---
        total_cells = self.rows * self.cols
        burned_cells = np.sum(grid >= 1)  # burning + burned-out
        burned_pct = (burned_cells / total_cells) * 100.0

        # Rate of spread: max distance from ignition / active steps
        active_steps = len([c for c in burn_counts if c > 0])
        if active_steps > 0:
            burned_positions = np.argwhere(grid >= 1)
            if len(burned_positions) > 0:
                distances = np.sqrt(
                    (burned_positions[:, 0] - cx) ** 2 +
                    (burned_positions[:, 1] - cy) ** 2
                )
                max_distance = np.max(distances)
                # Each cell ≈ 1m, each step ≈ 1s → convert to m/min
                spread_rate = (max_distance / active_steps) * 60.0
            else:
                spread_rate = 0.0
        else:
            spread_rate = 0.0

        # Peak intensity: peak simultaneous burning cells * heat release
        # Approximation of Byram's fire line intensity (kW/m)
        peak_intensity = peak_burning_cells * (heat_release / 1000.0)

        damage_metrics = {
            "total_area_burned_pct": round(burned_pct, 1),
            "rate_of_spread_m_min": round(spread_rate, 1),
            "peak_intensity": round(peak_intensity, 1),
        }

        logging.info(
            f"Simulation complete: {burned_pct:.1f}% burned, "
            f"spread={spread_rate:.1f} m/min, "
            f"intensity={peak_intensity:.1f} kW/m"
        )

        return damage_metrics
