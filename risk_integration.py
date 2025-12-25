"""
Integration helpers for ALLSAT AI - EWIS
Functions to integrate risk map generation into the Streamlit app
"""

from risk_map_generator import RiskMapGenerator, create_risk_visualization
from pathlib import Path
from typing import Dict, Optional
import tempfile


def generate_risk_maps_from_data(
    ndvi_path: str,
    ndmi_path: str,
    era5_nc_path: str,
    output_dir: Optional[str] = None,
    thresholds: Optional[dict] = None
) -> Dict:
    """
       
    This is the main integration point for the Streamlit app.
    Call this after you've downloaded both Sentinel-2 and ERA5-Land data.
    
    Args:
        ndvi_path: Path to NDVI GeoTIFF from Sentinel-2
        ndmi_path: Path to NDMI GeoTIFF from Sentinel-2  
        era5_nc_path: Path to ERA5-Land NetCDF file
        output_dir: Where to save outputs (default: temp directory)
        
    Returns:
        Dict containing:
            - 'drought_risk_path': Path to drought risk GeoTIFF
            - 'wildfire_risk_path': Path to wildfire risk GeoTIFF
            - 'drought_risk_data': Numpy array (0-1 scale)
            - 'wildfire_risk_data': Numpy array (0-1 scale)
            - 'statistics': Risk statistics and class distributions
            - 'coverage_percent': % of valid pixels
            - 'visualization': BytesIO buffer with risk maps figure
    
    
    """
    
    # Use temp directory if none provided
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='ewis_risk_')
    
    # Create risk map generator
    generator = RiskMapGenerator()
    
    # Generate risk maps
    results = generator.generate_combined_risk_maps(
        ndvi_path=ndvi_path,
        ndmi_path=ndmi_path,
        era5_nc_path=era5_nc_path,
        output_dir=output_dir,
        save_intermediate=False,  # Set to True if you want component layers
        thresholds=thresholds
    )
    
    # Print summary
    generator.print_risk_summary(results['statistics'])
    
    # Create visualization
    viz_buffer = create_risk_visualization(
        drought_risk_data=results['drought_risk_data'],
        wildfire_risk_data=results['wildfire_risk_data']
    )
    
    results['visualization'] = viz_buffer
    
    return results

def get_risk_score_with_label(risk_value: float) -> dict:
    """
    Get risk score with all associated metadata
    
    Args:
        risk_value: Risk score (0-1)
        
    Returns:
        Dict with label, color, recommendation, score
    """
    label, color, recommendation = interpret_risk_level(risk_value)
    
    return {
        'score': risk_value,
        'label': label,
        'color': color,
        'recommendation': recommendation,
        'score_formatted': f"{risk_value:.2f}"
    }

def interpret_risk_level(risk_value: float) -> tuple:
    """
    Convert 0-1 risk value to human-readable category and color
    
    Args:
        risk_value: Risk score between 0 and 1
        
    Returns:
        Tuple of (category_name, color_code, recommendation)
    """
    if risk_value < 0.33:
        return (
            "Low", 
            "#2ecc71",  # Green
            "Conditions are favorable. Continue standard monitoring."
        )
    elif risk_value < 0.66:
        return (
            "Moderate",
            "#f1c40f",  # Yellow
            "Increased monitoring recommended. Prepare mitigation strategies."
        )
    else:
        return (
            "High",
            "#e74c3c",  # Red
            "Critical conditions detected. Immediate action may be required."
        )


def format_risk_report(results: Dict, location_info: Dict) -> str:
    """
    Create a formatted text report of risk assessment results
    
    Args:
        results: Output from generate_risk_maps_from_data()
        location_info: Dict with 'address', 'latitude', 'longitude', etc.
        
    Returns:
        Formatted string report
    """
    stats = results['statistics']
    
    drought_category, drought_color, drought_rec = interpret_risk_level(
        stats['drought']['mean']
    )
    wildfire_category, wildfire_color, wildfire_rec = interpret_risk_level(
        stats['wildfire']['mean']
    )
    
    report = f"""
{'='*80}
ALLSAT AI - ENVIRONMENTAL RISK ASSESSMENT
{'='*80}

LOCATION INFORMATION:
  Address: {location_info.get('address', 'N/A')}
  Coordinates: {location_info.get('latitude', 'N/A'):.4f}°N, {location_info.get('longitude', 'N/A'):.4f}°W
  Coverage: {results.get('coverage_percent', 0):.1f}% valid data

{'='*80}
DROUGHT RISK ASSESSMENT
{'='*80}

Overall Risk Level: {drought_category.upper()} ({stats['drought']['mean']:.3f})
Risk Range: {stats['drought']['min']:.3f} - {stats['drought']['max']:.3f}

Pixel Classification:
"""
    
    for class_name, class_stats in stats['drought']['classes'].items():
        report += f"  • {class_name}: {class_stats['count']:,} pixels ({class_stats['percentage']:.1f}%)\n"
    
    report += f"\n💡 Recommendation: {drought_rec}\n"
    
    report += f"""
{'='*80}
WILDFIRE RISK ASSESSMENT
{'='*80}

Overall Risk Level: {wildfire_category.upper()} ({stats['wildfire']['mean']:.3f})
Risk Range: {stats['wildfire']['min']:.3f} - {stats['wildfire']['max']:.3f}

Pixel Classification:
"""
    
    for class_name, class_stats in stats['wildfire']['classes'].items():
        report += f"  • {class_name}: {class_stats['count']:,} pixels ({class_stats['percentage']:.1f}%)\n"
    
    report += f"\n💡 Recommendation: {wildfire_rec}\n"
    
    report += f"""
{'='*80}
METHODOLOGY
{'='*80}

Drought Risk Formula:
  35% Moisture Stress (NDMI) + 25% Vegetation Stress (NDVI) 
  + 20% Precipitation Deficit + 20% Heat Stress

Wildfire Risk Formula:
  40% Moisture Stress (NDMI) + 20% Heat Stress 
  + 20% Precipitation Deficit + 20% Critical Fuel Condition (NDMI < 0.10)

Risk Scale:
  0.00-0.33: Low Risk (favorable conditions)
  0.33-0.66: Moderate Risk (monitor closely)
  0.66-1.00: High Risk (action required)

Data Sources:
  • Sentinel-2 L2A: Vegetation indices (NDVI, NDMI)
  • ERA5-Land: Temperature, precipitation, soil moisture

{'='*80}
"""
    
    return report


def get_risk_alert_level(stats: Dict) -> Dict:
    """
    Determine overall alert status based on risk statistics
    
    Args:
        stats: Statistics dictionary from risk map generation
        
    Returns:
        Dict with alert info for UI display
    """
    drought_mean = stats['drought']['mean']
    wildfire_mean = stats['wildfire']['mean']
    
    # Determine highest priority alert
    max_risk = max(drought_mean, wildfire_mean)
    
    if max_risk >= 0.66:
        severity = "CRITICAL"
        icon = "🔴"
        message = "High risk conditions detected. Immediate attention recommended."
    elif max_risk >= 0.33:
        severity = "WARNING"
        icon = "🟡"
        message = "Moderate risk conditions detected. Increased monitoring advised."
    else:
        severity = "NORMAL"
        icon = "🟢"
        message = "Conditions are within normal ranges. Continue standard monitoring."
    
    return {
        'severity': severity,
        'icon': icon,
        'message': message,
        'drought_risk': drought_mean,
        'wildfire_risk': wildfire_mean,
        'max_risk': max_risk
    }