import math
from typing import Tuple, Optional

EARTH_RADIUS_KM = 6371.0088

def haversine_distance_km(
    lat1: Optional[float], lon1: Optional[float],
    lat2: Optional[float], lon2: Optional[float]
) -> float:
    """
    Calculate the great circle distance between two points on Earth in kilometers.
    Returns 0.0 if any coordinate is None.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 0.0

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c

def calculate_implied_speed(distance_km: float, time_diff_seconds: float) -> float:
    """
    Compute implied travel speed in km/h.
    """
    if time_diff_seconds <= 0.0:
        return float("inf") if distance_km > 0.05 else 0.0
    
    hours = time_diff_seconds / 3600.0
    return distance_km / hours
