import requests
from typing import Optional, Tuple

def geocode_locationiq(address: str, api_key: str, timeout: int = 10) -> Optional[Tuple[float, float, str]]:
    """
    LocationIQ geocoding - Free tier: 5,000 requests/day
    Get API key at: https://locationiq.com/ (free signup, no credit card)
    """
    url = "https://us1.locationiq.com/v1/search.php"
    params = {
        "key": api_key,
        "q": address,
        "format": "json",
        "limit": 1
    }
    
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        
        if not data or len(data) == 0:
            return None
            
        result = data[0]
        lat = float(result["lat"])
        lon = float(result["lon"])
        formatted = result.get("display_name", address)
        
        return lat, lon, formatted
    except Exception as e:
        print(f"LocationIQ geocoding error: {e}")
        return None


def geocode_positionstack(address: str, api_key: str, timeout: int = 10) -> Optional[Tuple[float, float, str]]:
    """
    PositionStack geocoding - Free tier: 25,000 requests/month
    Get API key at: https://positionstack.com/ (free signup)
    """
    url = "http://api.positionstack.com/v1/forward"  # Note: Free tier uses HTTP only
    params = {
        "access_key": api_key,
        "query": address,
        "limit": 1
    }
    
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        
        if not data.get("data") or len(data["data"]) == 0:
            return None
            
        result = data["data"][0]
        lat = float(result["latitude"])
        lon = float(result["longitude"])
        formatted = result.get("label", address)
        
        return lat, lon, formatted
    except Exception as e:
        print(f"PositionStack geocoding error: {e}")
        return None


def geocode_arcgis(address: str, api_key: Optional[str] = None, timeout: int = 10) -> Optional[Tuple[float, float, str]]:
    """
    ArcGIS geocoding with optional authentication.
    Free tier: 20,000 geocodes/month (not stored)
    Get API key at: https://developers.arcgis.com/
    """
    url = "https://geocode-api.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
    
    params = {
        "f": "json",
        "singleLine": address,
        "maxLocations": 1,
        "outFields": "Match_addr",
        "forStorage": "false"
    }
    
    if api_key:
        params["token"] = api_key
    
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        
        if "error" in data:
            error_msg = data["error"].get("message", "Unknown error")
            print(f"ArcGIS API Error: {error_msg}")
            return None
        
        candidates = data.get("candidates") or []
        if not candidates:
            return None
            
        top = candidates[0]
        lat = float(top["location"]["y"])
        lon = float(top["location"]["x"])
        formatted = (
            top.get("attributes", {}).get("Match_addr") or 
            top.get("address") or 
            address
        )
        
        return lat, lon, formatted
    except Exception as e:
        print(f"ArcGIS geocoding error: {e}")
        return None


def geocode_nominatim_fallback(address: str, timeout: int = 10) -> Optional[Tuple[float, float, str]]:
    """
    Nominatim - Last resort, often blocked on shared hosting.
    Rate limit: 1 request/second
    """
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(
            user_agent="ALLSAT_AI_EWIS/1.0",
            timeout=timeout
        )
        loc = geolocator.geocode(address)
        if loc:
            return float(loc.latitude), float(loc.longitude), getattr(loc, "address", address)
        return None
    except Exception as e:
        print(f"Nominatim geocoding error: {e}")
        return None


def geocode_with_fallback(
    address: str, 
    timeout: int = 10,
    locationiq_api_key: Optional[str] = None,
    positionstack_api_key: Optional[str] = None,
    arcgis_api_key: Optional[str] = None
) -> Tuple[float, float, str]:
    """
    Multi-provider geocoding with smart fallback strategy.
    
    Priority order (based on reliability on Streamlit Cloud):
    1. LocationIQ (if API key provided) - FREE tier: 5,000/day
    2. PositionStack (if API key provided) - FREE tier: 25,000/month  
    3. ArcGIS (if API key provided) - FREE tier: 20,000/month
    4. Nominatim - FREE but often blocked on Streamlit Cloud
    
    All providers except Nominatim require free API keys.
    
    Recommended: Get a free LocationIQ key at https://locationiq.com/
    - No credit card required
    - 5,000 requests/day free
    - Works perfectly on Streamlit Cloud
    
    Args:
        address: Address string to geocode
        timeout: Request timeout in seconds
        locationiq_api_key: LocationIQ API key (recommended!)
        positionstack_api_key: PositionStack API key
        arcgis_api_key: ArcGIS API key
        
    Returns:
        Tuple of (latitude, longitude, formatted_address)
        
    Raises:
        ValueError: If all providers fail
    """
    address = (address or "").strip()
    if not address:
        raise ValueError("Geocoding failed: empty address")

    last_err: Optional[Exception] = None
    providers_tried = []

    # Provider 1: LocationIQ (BEST for Streamlit Cloud - reliable and fast)
    if locationiq_api_key:
        try:
            result = geocode_locationiq(address, api_key=locationiq_api_key, timeout=timeout)
            if result:
                print(f"✅ Geocoded via LocationIQ")
                return result
            providers_tried.append("LocationIQ (no results)")
        except Exception as e:
            last_err = e
            providers_tried.append(f"LocationIQ (error: {str(e)[:50]})")
    else:
        providers_tried.append("LocationIQ (skipped - no API key)")

    # Provider 2: PositionStack (Good alternative - 25k/month free)
    if positionstack_api_key:
        try:
            result = geocode_positionstack(address, api_key=positionstack_api_key, timeout=timeout)
            if result:
                print(f"✅ Geocoded via PositionStack")
                return result
            providers_tried.append("PositionStack (no results)")
        except Exception as e:
            last_err = e
            providers_tried.append(f"PositionStack (error: {str(e)[:50]})")
    else:
        providers_tried.append("PositionStack (skipped - no API key)")

    # Provider 3: ArcGIS (Solid option if you have a key)
    if arcgis_api_key:
        try:
            result = geocode_arcgis(address, api_key=arcgis_api_key, timeout=timeout)
            if result:
                print(f"✅ Geocoded via ArcGIS")
                return result
            providers_tried.append("ArcGIS (no results)")
        except Exception as e:
            last_err = e
            providers_tried.append(f"ArcGIS (error: {str(e)[:50]})")
    else:
        providers_tried.append("ArcGIS (skipped - no API key)")

    # Provider 4: Nominatim (LAST RESORT - often blocked on Streamlit Cloud)
    try:
        result = geocode_nominatim_fallback(address, timeout=timeout)
        if result:
            print(f"✅ Geocoded via Nominatim")
            return result
        providers_tried.append("Nominatim (no results)")
    except Exception as e:
        last_err = e
        providers_tried.append(f"Nominatim (error: {str(e)[:50]})")

    # All providers failed
    providers_summary = "\n  - ".join(providers_tried)
    error_msg = (
        f"Geocoding failed for '{address}' - all providers failed:\n"
        f"  - {providers_summary}\n"
        f"Last error: {last_err}\n\n"
    )
    raise ValueError(error_msg)