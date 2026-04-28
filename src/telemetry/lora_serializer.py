"""
LoRa Semantic Payload Serializer — Improvement #9.

Generates compact, hex-encoded telemetry packets for transmission
over the RF96 LoRa module (433/868 MHz). Respects LoRa's strict
bandwidth constraints (max ~250 bytes per packet at SF7).

The rover transmits semantic conclusions (risk labels, fuel types)
rather than raw sensor data, as documented in the research paper §5.1:
"semantic data transmission" — the Pi processes data locally and sends
only the distilled output.

Packet format (JSON + hex-encoded):
    {
        "lat": 36.70,
        "lon": 3.00,
        "fuel": "brush",
        "dry": "cured",
        "risk": "High",
        "ngrdi": -0.12,
        "burned_pct": 65.4,
        "spread": 12.2,
        "gas_ppm": 850,
        "ts": 1714300000
    }
"""

import json
import logging
import time


# LoRa payload size limit (bytes) — SF7 BW125 max is ~222 bytes
_MAX_PAYLOAD_BYTES = 200

# Compact fuel type abbreviations to save bytes
_FUEL_ABBREV = {
    "Pinus halepensis": "PH",
    "Quercus ilex": "QI",
    "Quercus suber": "QS",
    "Stipa tenacissima": "ST",
    "Pistacia lentiscus": "PL",
    "Arbutus unedo": "AU",
    "Tetraclinis articulata": "TA",
    "Cistus monspeliensis": "CM",
    "Calicotome villosa": "CV",
    "Juniperus phoenicea": "JP",
    "Erica arborea": "EA",
    "Olea europaea": "OE",
    "Barren": "BR",
}

# Compact aridity abbreviations
_ARIDITY_ABBREV = {"Dead": "D", "cured": "C", "Green": "G"}

# Compact risk abbreviations
_RISK_ABBREV = {
    "Low": "L", "Moderate": "M", "High": "H", "Extreme": "X",
}


class LoRaSerializer:
    """
    Serializes pipeline output into compact LoRa-ready payloads.

    Supports two modes:
    - JSON (human-readable, ~150 bytes) — for debugging
    - Binary hex (compact, ~80 bytes) — for production LoRa TX
    """

    def __init__(self):
        logging.info("Initializing LoRa Semantic Payload Serializer...")

    def build_payload(self, lat, lon, vision_result, ca_metrics,
                      risk_label, gas_ppm):
        """
        Build the semantic telemetry payload from pipeline outputs.

        Returns a dict containing the structured payload and both
        JSON and hex-encoded representations.
        """
        payload = {
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "fuel": _FUEL_ABBREV.get(
                vision_result.get("fuel_type", ""), "??"
            ),
            "dry": _ARIDITY_ABBREV.get(
                vision_result.get("aridity_state", ""), "?"
            ),
            "ngrdi": vision_result.get("ngrdi_value", 0.0),
            "risk": _RISK_ABBREV.get(risk_label, "?"),
            "burn": ca_metrics.get("total_area_burned_pct", 0),
            "spread": ca_metrics.get("rate_of_spread_m_min", 0),
            "gas": int(gas_ppm),
            "ts": int(time.time()),
        }

        # JSON representation
        json_str = json.dumps(payload, separators=(",", ":"))
        json_bytes = json_str.encode("utf-8")

        # Hex-encoded representation
        hex_str = json_bytes.hex()

        size = len(json_bytes)
        fits = size <= _MAX_PAYLOAD_BYTES

        if not fits:
            logging.warning(
                f"Payload exceeds LoRa limit: {size}/{_MAX_PAYLOAD_BYTES} bytes"
            )

        return {
            "payload": payload,
            "json": json_str,
            "hex": hex_str,
            "size_bytes": size,
            "within_limit": fits,
        }

    @staticmethod
    def decode_hex(hex_str):
        """Decode a hex-encoded LoRa payload back to a dict."""
        json_str = bytes.fromhex(hex_str).decode("utf-8")
        return json.loads(json_str)
