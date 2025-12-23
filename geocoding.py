import requests
from typing import Optional, Tuple

def geocode_arcgis(address: str, timeout: int = 10) -> Optional[Tuple[float, float, str]]:
    url = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
    params = {
        "f": "json",
        "singleLine": address,
        "maxLocations": 1,
        "outFields": "Match_addr"
    }
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return None
    top = candidates[0]
    lat = float(top["location"]["y"])
    lon = float(top["location"]["x"])
    formatted = top.get("address") or address
    return lat, lon, formatted

def geocode_with_fallback(address: str, timeout: int = 10) -> Tuple[float, float, str]:
    address = (address or "").strip()
    if not address:
        raise ValueError("Geocoding failed: empty address")

    last_err: Optional[Exception] = None

    # 1) Nominatim (may fail on Streamlit Cloud due to IP blocking)
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(
            user_agent="allsat-ai-streamlit/1.0 (email=your-email@example.com)",
            timeout=timeout
        )
        loc = geolocator.geocode(address)
        if loc:
            return float(loc.latitude), float(loc.longitude), getattr(loc, "address", address)
    except Exception as e:
        last_err = e

    # 2) ArcGIS (no key) - retry once
    for _ in range(2):
        try:
            arc = geocode_arcgis(address, timeout=timeout)
            if arc:
                return arc
        except Exception as e:
            last_err = e

    raise ValueError(f"Geocoding failed (all providers): {address}. Last error: {last_err}")
