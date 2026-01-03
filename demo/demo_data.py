"""
ALLSAT AI - Demo Data Module
Hardcoded data for 5 demo locations × 3 analysis types
"""

from typing import Dict, Any, Optional

# ============================================================================
# LOCATION DATA STRUCTURE
# ============================================================================

DEMO_LOCATIONS = {
    'prineville': {
        'name': 'Prineville',
        'display_name': 'Prineville, Oregon',
        'coordinates': {
            'lat': 44.30,
            'lon': -120.83
        }
    },
    'bend': {
        'name': 'Bend',
        'display_name': 'Bend, Oregon',
        'coordinates': {
            'lat': 44.06,
            'lon': -121.31
        }
    },
    'portland': {
        'name': 'Portland',
        'display_name': 'Portland, Oregon',
        'coordinates': {
            'lat': 45.52,
            'lon': -122.68
        }
    },
    'dallas': {
        'name': 'Dallas',
        'display_name': 'Dallas, Texas',
        'coordinates': {
            'lat': 32.77,
            'lon': -96.79
        }
    },
    'nashville': {
        'name': 'Nashville',
        'display_name': 'Nashville, Tennessee',
        'coordinates': {
            'lat': 36.13,
            'lon': -86.67
        }
    }
}

# ============================================================================
# VEGETATION DATA (Sentinel-2)
# ============================================================================

VEGETATION_DATA = {
    'prineville': {
        # Scene metadata
        'scene_date': '2024-12-31',  # From Sentinel-2
        'cloud_coverage': 31.5,
        
        # NDVI/NDMI metrics
        'metrics': {
            'ndvi': {
                'mean': 0.369,
                'median': 0.3316,
                'std': 0.204,
                'min': -1.000,
                'max': 1.000,
                'coverage_percent': 68.5
            },
            'ndmi': {
                'mean': -0.015,
                'median': -0.0405,
                'std': 0.177,
                'min': -1.000,
                'max': 1.000,
                'coverage_percent': 68.5
            },
            'correlation': 0.559,  # NDVI-NDMI correlation
            'valid_pixels': 1548792,
            'total_pixels': 2260647
        },
        
        # Vegetation health classification (derived from NDVI ranges)
        'health_classification': {
            'bare_soil': 15.0,  # < 0.2
            'sparse': 25.0,  # 0.2-0.3
            'moderate': 35.0,  # 0.3-0.6
            'dense': 25.0  # > 0.6
        },
        
        # Alerts
        'alerts': [
            {
                'severity': 'CRITICAL',
                'title': 'MOISTURE STRESS - (NDMI SCENARIO)',
                'message': 'This demonstration scenario displays a pre-generated NDMI value of -0.015 compared against an example threshold (0.100) to illustrate how moisture stress may be presented in EWIS outputs.',
                'recommendation': 'Pre-generated demonstration scenario. No live analysis occurs.'
            },
            {
                'severity': 'MEDIUM',
                'title': 'DATA QUALITY - (CLOUD SCENARIO)',
                'message': 'This demonstration scenario displays a pre-generated cloud coverage of 31.5% compared against a threshold (20%) to illustrate how data quality issues may be presented in EWIS outputs.',
                'recommendation': 'Pre-generated demonstration scenario. No live analysis occurs.'
            }
        ],
        
      
        'images': {
            'sentinel': 'prineville/sentinel.png',
        }
    },
    
    'bend': {
        'scene_date': '2024-12-31',
        'cloud_coverage': 13.3,
        
        # NDVI/NDMI metrics
        'metrics': {
            'ndvi': {
                'mean': 0.464,
                'median': 0.4222,
                'std': 0.224,
                'min': -1.000,
                'max': 1.000,
                'coverage_percent': 86.7
            },
            'ndmi': {
                'mean': 0.158,
                'median': 0.0667,
                'std': 0.304,
                'min': -1.000,
                'max': 1.000,
                'coverage_percent': 86.7
            },
            'correlation': 0.557,
            'valid_pixels': 1959242,
            'total_pixels': 2259700
        },
        
        # Vegetation health classification
        'health_classification': {
            'bare_soil': 10.0,
            'sparse': 20.0,
            'moderate': 40.0,
            'dense': 30.0
        },
        
        # Alerts
        'alerts': [],
        
        'images': {
            'sentinel': 'bend/sentinel.png',
        }
    },
    
    'portland': {
        'scene_date': '2024-12-31',
        'cloud_coverage': 24.7,
        
        # NDVI/NDMI metrics
        'metrics': {
            'ndvi': {
                'mean': 0.595,
                'median': 0.6222,
                'std': 0.350,
                'min': -1.000,
                'max': 1.000,
                'coverage_percent': 75.3
            },
            'ndmi': {
                'mean': 0.165,
                'median': 0.0667,
                'std': 0.245,
                'min': -1.000,
                'max': 1.000,
                'coverage_percent': 75.3
            },
            'correlation': 0.557,
            'valid_pixels': 1702160,
            'total_pixels': 2260500
        },
        
        # Vegetation health classification
        'health_classification': {
            'bare_soil': 8.0,
            'sparse': 15.0,
            'moderate': 32.0,
            'dense': 45.0
        },
        
        # Alerts
        'alerts': [],
        
        'images': {
            'sentinel': 'portland/sentinel.png',
        }
    },

    'dallas': {
        'scene_date': '2024-12-31',
        'cloud_coverage': 0.0,
        
        # NDVI/NDMI metrics
        'metrics': {
            'ndvi': {
                'mean': 0.261,
                'median': 0.2944,
                'std': 0.255,
                'min': -1.000,
                'max': 1.000,
                'coverage_percent': 100.0
            },
            'ndmi': {
                'mean': -0.098,
                'median': -0.1033,
                'std': 0.156,
                'min': -1.000,
                'max': 1.000,
                'coverage_percent': 100.0
            },
            'correlation': 0.141,
            'valid_pixels': 2254623,
            'total_pixels': 2255220
        },
        
        # Vegetation health classification
        'health_classification': {
            'bare_soil': 40.0,
            'sparse': 35.0,
            'moderate': 20.0,
            'dense': 5.0
        },
        
        # Alerts
        'alerts': [
            {
                'severity': 'HIGH',
                'title': 'VEGETATION HEALTH - (NDVI SCENARIO)',
                'message': 'This demonstration scenario displays a pre-generated NDVI value of 0.261 compared against an example threshold (0.300) to illustrate how vegetation health may be presented in EWIS outputs.',
                'recommendation': 'Pre-generated demonstration scenario. No live analysis occurs.'
            },
            {
                'severity': 'CRITICAL',
                'title': 'MOISTURE STRESS - (NDMI SCENARIO)',
                'message': 'This demonstration scenario displays a pre-generated NDMI value of -0.098 compared against an example threshold (0.100) to illustrate how moisture stress may be presented in EWIS outputs.',
                'recommendation': 'Pre-generated demonstration scenario. No live analysis occurs.'
            }
        ],
        
        'images': {
            'sentinel': 'dallas/sentinel.png',
        }
    },
    
    'nashville': {
        'scene_date': '2024-12-31',
        'cloud_coverage': 0.9,
        
        # NDVI/NDMI metrics
        'metrics': {
            'ndvi': {
                'mean': 0.415,
                'median': 0.2944,
                'std': 0.225,
                'min': -1.000,
                'max': 1.000,
                'coverage_percent': 99.1
            },
            'ndmi': {
                'mean': -0.055,
                'median': -0.1033,
                'std': 0.161,
                'min': -1.000,
                'max': 1.000,
                'coverage_percent': 99.1
            },
            'correlation': 0.141,
            'valid_pixels': 2236019,
            'total_pixels': 2256000
        },
        
        # Vegetation health classification
        'health_classification': {
            'bare_soil': 20.0,
            'sparse': 25.0,
            'moderate': 35.0,
            'dense': 20.0
        },
        
        # Alerts
        'alerts': [
            {
                'severity': 'CRITICAL',
                'title': 'MOISTURE STRESS - (NDMI SCENARIO)',
                'message': 'This demonstration scenario displays a pre-generated NDMI value of -0.055 compared against an example threshold (0.100) to illustrate how moisture stress may be presented in EWIS outputs.',
                'recommendation': 'Pre-generated demonstration scenario. No live analysis occurs.'
            }
        ],
        
        'images': {
            'sentinel': 'nashville/sentinel.png',
        }
    }
}

# ============================================================================
# CLIMATE DATA (ERA5-Land)
# ============================================================================

CLIMATE_DATA = {
    'prineville': {
        # Date range
        'date_range': {
            'start': '2025-10-01',
            'end': '2025-11-01'
        },
        
        # Climate metrics
        'metrics': {
            'temperature': {
                'mean_c': 3.8,
                'max_c': 4.6,
                'min_c': 3.1,
                'std_c': 0.5  # Estimated from graph
            },
            'precipitation': {
                'total_mm': 1.2,
                'mean_daily_mm': 0.6,
                'max_daily_mm': 0.6,
                'days_with_precip': 2  # Estimated from monthly chart
            },
            'soil_moisture': {
                'mean': 0.224,
                'min': 0.194,
                'max': 0.254
            }
        },
        
        # Climate alerts
        'alerts': [
            {
                'severity': 'HIGH',
                'title': 'PRECIPITATION - (DROUGHT SCENARIO)',
                'message': 'This demonstration scenario displays a pre-generated precipitation value of 1.2mm compared against an example threshold (100mm) to illustrate how drought risk may be presented in EWIS outputs.',
                'recommendation': 'Pre-generated demonstration scenario. No live analysis occurs.'
            },
            {
                'severity': 'MEDIUM',
                'title': 'SOIL MOISTURE - (DRY SCENARIO)',
                'message': 'This demonstration scenario displays a pre-generated soil moisture value of 0.194 compared against an example threshold (0.200) to illustrate how dry soil conditions may be presented in EWIS outputs.',
                'recommendation': 'Pre-generated demonstration scenario. No live analysis occurs.'
            }
        ],
        
        # Trends
        'trends': {
            'temperature_trend': 'decreasing',  # From graph: 4.6°C to 3.1°C
            'precipitation_trend': 'stable',
            'soil_moisture_trend': 'increasing',  # From graph: 0.194 to 0.254
            'drought_indicators': ['Low precipitation', 'Below-average soil moisture']
        },
        
        # Image paths
        'images': {
            'climate': 'prineville/climate.png',
        }
    },
    
    'bend': {
        'date_range': {
            'start': '2025-10-01',
            'end': '2025-11-01'
        },
        
        # Climate metrics
        'metrics': {
            'temperature': {
                'mean_c': 3.2,
                'max_c': 3.8,
                'min_c': 2.7,
                'std_c': 0.4
            },
            'precipitation': {
                'total_mm': 1.2,
                'mean_daily_mm': 0.6,
                'max_daily_mm': 0.7,
                'days_with_precip': 2
            },
            'soil_moisture': {
                'mean': 0.242,
                'min': 0.219,
                'max': 0.264
            }
        },
        
        # Climate alerts
        'alerts': [
            {
                'severity': 'HIGH',
                'title': 'PRECIPITATION - (DROUGHT SCENARIO)',
                'message': 'This demonstration scenario displays a pre-generated precipitation value of 1.2mm compared against an example threshold (100mm) to illustrate how drought risk may be presented in EWIS outputs.',
                'recommendation': 'Pre-generated demonstration scenario. No live analysis occurs.'
            }
        ],
        
        # Trends
        'trends': {
            'temperature_trend': 'decreasing',
            'precipitation_trend': 'stable',
            'soil_moisture_trend': 'increasing',
            'drought_indicators': ['Low precipitation']
        },
        
        'images': {
            'climate': 'bend/climate.png',
        }
    },
    
    'portland': {
        'date_range': {
            'start': '2025-10-01',
            'end': '2025-11-01'
        },
        
        # Climate metrics
        'metrics': {
            'temperature': {
                'mean_c': 7.7,
                'max_c': 8.2,
                'min_c': 7.3,
                'std_c': 0.3
            },
            'precipitation': {
                'total_mm': 4.3,
                'mean_daily_mm': 2.2,
                'max_daily_mm': 3.1,
                'days_with_precip': 2
            },
            'soil_moisture': {
                'mean': 0.355,
                'min': 0.311,
                'max': 0.399
            }
        },
        
        # Climate alerts
        'alerts': [
            {
                'severity': 'HIGH',
                'title': 'PRECIPITATION - (DROUGHT SCENARIO)',
                'message': 'This demonstration scenario displays a pre-generated precipitation value of 4.3mm compared against an example threshold (100mm) to illustrate how drought risk may be presented in EWIS outputs.',
                'recommendation': 'Pre-generated demonstration scenario. No live analysis occurs.'
            }
        ],
        
        # Trends
        'trends': {
            'temperature_trend': 'stable',
            'precipitation_trend': 'stable',
            'soil_moisture_trend': 'increasing',
            'drought_indicators': ['Low precipitation']
        },
        
        'images': {
            'climate': 'portland/climate.png',
        }
    },
    
    'dallas': {
        'date_range': {
            'start': '2025-10-01',
            'end': '2025-11-01'
        },
        
        # Climate metrics
        'metrics': {
            'temperature': {
                'mean_c': 16.2,
                'max_c': 18.8,
                'min_c': 13.7,
                'std_c': 1.8
            },
            'precipitation': {
                'total_mm': 3.1,
                'mean_daily_mm': 1.6,
                'max_daily_mm': 1.6,
                'days_with_precip': 2
            },
            'soil_moisture': {
                'mean': 0.321,
                'min': 0.288,
                'max': 0.355
            }
        },
        
        # Climate alerts
        'alerts': [
            {
                'severity': 'HIGH',
                'title': 'PRECIPITATION - (DROUGHT SCENARIO)',
                'message': 'This demonstration scenario displays a pre-generated precipitation value of 3.1mm compared against an example threshold (100mm) to illustrate how drought risk may be presented in EWIS outputs.',
                'recommendation': 'Pre-generated demonstration scenario. No live analysis occurs.'
            }
        ],
        
        # Trends
        'trends': {
            'temperature_trend': 'decreasing',
            'precipitation_trend': 'stable',
            'soil_moisture_trend': 'increasing',
            'drought_indicators': ['Low precipitation', 'High temperatures']
        },
        
        'images': {
            'climate': 'dallas/climate.png',
        }
    },
    
    'nashville': {
        'date_range': {
            'start': '2025-10-01',
            'end': '2025-11-01'
        },
        
        # Climate metrics
        'metrics': {
            'temperature': {
                'mean_c': 10.9,
                'max_c': 13.4,
                'min_c': 8.3,
                'std_c': 1.7
            },
            'precipitation': {
                'total_mm': 3.1,
                'mean_daily_mm': 1.6,
                'max_daily_mm': 2.0,
                'days_with_precip': 2
            },
            'soil_moisture': {
                'mean': 0.337,
                'min': 0.312,
                'max': 0.362
            }
        },
        
        # Climate alerts
        'alerts': [
            {
                'severity': 'HIGH',
                'title': 'PRECIPITATION - (DROUGHT SCENARIO)',
                'message': 'This demonstration scenario displays a pre-generated precipitation value of 3.1mm compared against an example threshold (100mm) to illustrate how drought risk may be presented in EWIS outputs.',
                'recommendation': 'Pre-generated demonstration scenario. No live analysis occurs.'
            }
        ],
        
        # Trends
        'trends': {
            'temperature_trend': 'decreasing',
            'precipitation_trend': 'stable',
            'soil_moisture_trend': 'increasing',
            'drought_indicators': ['Low precipitation']
        },
        
        'images': {
            'climate': 'nashville/climate.png',
        }
    }
}

# ============================================================================
# COMBINED RISK DATA (Both)
# ============================================================================

RISK_DATA = {
    'prineville': {
        # Drought risk
        'drought_risk': {
            'mean_score': 0.51,
            'min_score': 0.25,
            'max_score': 0.86,
            'risk_level': 'Moderate',
            'risk_color': "#f1c40f",  # Yellow
            'class_distribution': {
                'low': 22.9,  # % of pixels
                'moderate': 63.0,
                'high': 14.1
            },
            'class_pixels': {
                'low': 206107,
                'moderate': 566667,
                'high': 127336
            },
            'recommendation': 'This demonstration scenario displays a pre-generated drought risk level of Moderate Risk value 0.51 compared against threshold range of 0.25 - 0.86 to illustrate how drought risk may be presented in EWIS outputs.'
        },
        
        # Wildfire risk
        'wildfire_risk': {
            'mean_score': 0.43,
            'min_score': 0.25,
            'max_score': 0.46,
            'risk_level': 'Moderate',
            'risk_color': "#f1c40f",  # Yellow
            'class_distribution': {
                'low': 14.4,
                'moderate': 85.6,
                'high': 0.0
            },
            'class_pixels': {
                'low': 129293,
                'moderate': 770817,
                'high': 0
            },
            'recommendation': 'This demonstration scenario displays a pre-generated wildfire risk level of Moderate Risk value 0.43 compared against threshold range of 0.25 - 0.46 to illustrate how wildfire risk may be presented in EWIS outputs.'
        },
        
        # Overall alert
        'overall_alert': {
            'severity': 'WARNING',
            'icon': '🟡',
            'message': 'Moderate risk conditions displayed.'
        },
        
        # Coverage
        'coverage_percent': 68.5,
        
        # Image paths
        'images': {
            'risk': 'prineville/combined.png',
        }
    },
    
    'bend': {
        # Drought risk
        'drought_risk': {
            'mean_score': 0.39,
            'min_score': 0.23,
            'max_score': 0.86,
            'risk_level': 'Moderate',
            'risk_color': '#f1c40f',
            'class_distribution': {
                'low': 56.6,
                'moderate': 36.8,
                'high': 6.7
            },
            'class_pixels': {
                'low': 691899,
                'moderate': 449828,
                'high': 81477
            },
            'recommendation': 'This demonstration scenario displays a pre-generated drought risk level of Moderate Risk value 0.39 compared against threshold range of 0.23 - 0.86 to illustrate how drought risk may be presented in EWIS outputs.'
        },
        
        # Wildfire risk
        'wildfire_risk': {
            'mean_score': 0.35,
            'min_score': 0.23,
            'max_score': 0.46,
            'risk_level': 'Moderate',
            'risk_color': '#f1c40f',
            'class_distribution': {
                'low': 49.1,
                'moderate': 50.9,
                'high': 0.0
            },
            'class_pixels': {
                'low': 600804,
                'moderate': 622400,
                'high': 0
            },
            'recommendation': 'This demonstration scenario displays a pre-generated wildfire risk level of Moderate Risk value 0.35 compared against threshold range of 0.23 - 0.46 to illustrate how wildfire risk may be presented in EWIS outputs.'
        },
        
        # Overall alert
        'overall_alert': {
            'severity': 'WARNING',
            'icon': '🟡',
            'message': 'Moderate risk conditions displayed.'
        },
        
        # Coverage
        'coverage_percent': 86.7,
        
        'images': {
            'risk': 'bend/combined.png',
        }
    },
    
    'portland': {
        # Drought risk
        'drought_risk': {
            'mean_score': 0.36,
            'min_score': 0.23,
            'max_score': 0.86,
            'risk_level': 'Moderate',
            'risk_color': '#f1c40f',
            'class_distribution': {
                'low': 67.3,
                'moderate': 26.0,
                'high': 6.7
            },
            'class_pixels': {
                'low': 916956,
                'moderate': 354437,
                'high': 91430
            },
            'recommendation': 'This demonstration scenario displays a pre-generated drought risk level of Moderate Risk value 0.36 compared against threshold range of 0.23 - 0.86 to illustrate how drought risk may be presented in EWIS outputs.'
        },
        
        # Wildfire risk
        'wildfire_risk': {
            'mean_score': 0.33,
            'min_score': 0.23,
            'max_score': 0.46,
            'risk_level': 'Low',
            'risk_color': '#2ecc71',
            'class_distribution': {
                'low': 60.3,
                'moderate': 39.7,
                'high': 0.0
            },
            'class_pixels': {
                'low': 821426,
                'moderate': 541397,
                'high': 0
            },
            'recommendation': 'This demonstration scenario displays a pre-generated wildfire risk level of Low Risk value 0.33 compared against threshold range of 0.23 - 0.46 to illustrate how wildfire risk may be presented in EWIS outputs.'
        },
        
        # Overall alert
        'overall_alert': {
            'severity': 'WARNING',
            'icon': '🟡',
            'message': 'Moderate risk conditions displayed.'
        },
        
        # Coverage
        'coverage_percent': 75.3,
        
        'images': {
            'risk': 'portland/combined.png',
        }
    },
    
    'dallas': {
        'drought_risk': {
            'mean_score': 0.62,
            'min_score': 0.28,
            'max_score': 0.89,
            'risk_level': 'Moderate',
            'risk_color': '#f1c40f',
            'class_distribution': {
                'low': 9.9,
                'moderate': 59.7,
                'high': 30.4
            },
            'class_pixels': {
                'low': 154630,
                'moderate': 929568,
                'high': 473845
            },
            'recommendation': 'This demonstration scenario displays a pre-generated drought risk level of Moderate Risk value 0.62 compared against threshold range of 0.28 - 0.89 to illustrate how drought risk may be presented in EWIS outputs.'
        },
        
        # Wildfire risk
        'wildfire_risk': {
            'mean_score': 0.47,
            'min_score': 0.28,
            'max_score': 0.49,
            'risk_level': 'Moderate',
            'risk_color': '#f1c40f',
            'class_distribution': {
                'low': 9.1,
                'moderate': 90.9,
                'high': 0.0
            },
            'class_pixels': {
                'low': 141810,
                'moderate': 1416233,
                'high': 0
            },
            'recommendation': 'This demonstration scenario displays a pre-generated wildfire risk level of Moderate Risk value 0.47 compared against threshold range of 0.28 - 0.49 to illustrate how wildfire risk may be presented in EWIS outputs.'
        },
        
        # Overall alert
        'overall_alert': {
            'severity': 'WARNING',
            'icon': '🟡',
            'message': 'Moderate risk conditions displayed.'
        },
        
        # Coverage
        'coverage_percent': 100.0,
        
        'images': {
            'risk': 'dallas/combined.png',
        }
    },
    
    'nashville': {
        'drought_risk': {
            'mean_score': 0.52,
            'min_score': 0.26,
            'max_score': 0.87,
            'risk_level': 'Moderate',
            'risk_color': '#f1c40f',
            'class_distribution': {
                'low': 22.0,
                'moderate': 69.0,
                'high': 9.0
            },
            'class_pixels': {
                'low': 325841,
                'moderate': 1022925,
                'high': 133488
            },
            'recommendation': 'This demonstration scenario displays a pre-generated drought risk level of Moderate Risk value 0.52 compared against threshold range of 0.26 - 0.87 to illustrate how drought risk may be presented in EWIS outputs.'
        },
        
        # Wildfire risk
        'wildfire_risk': {
            'mean_score': 0.43,
            'min_score': 0.26,
            'max_score': 0.47,
            'risk_level': 'Moderate',
            'risk_color': '#f1c40f',
            'class_distribution': {
                'low': 15.6,
                'moderate': 84.4,
                'high': 0.0
            },
            'class_pixels': {
                'low': 231488,
                'moderate': 1250766,
                'high': 0
            },
            'recommendation': 'This demonstration scenario displays a pre-generated wildfire risk level of Low Risk value 0.33 compared against threshold range of 0.23 - 0.46 to illustrate how wildfire risk may be presented in EWIS outputs.'
        },
        
        # Overall alert
        'overall_alert': {
            'severity': 'WARNING',
            'icon': '🟡',
            'message': 'Moderate risk conditions displayed.'
        },
        
        # Coverage
        'coverage_percent': 99.1,
        
        'images': {
            'risk': 'nashville/combined.png',
        }
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_location_info(location_key: str) -> Optional[Dict[str, Any]]:
    """Get location metadata"""
    return DEMO_LOCATIONS.get(location_key.lower())

def get_vegetation_data(location_key: str) -> Optional[Dict[str, Any]]:
    """Get vegetation analysis data for a location"""
    return VEGETATION_DATA.get(location_key.lower())

def get_climate_data(location_key: str) -> Optional[Dict[str, Any]]:
    """Get climate analysis data for a location"""
    return CLIMATE_DATA.get(location_key.lower())

def get_risk_data(location_key: str) -> Optional[Dict[str, Any]]:
    """Get combined risk data for a location"""
    return RISK_DATA.get(location_key.lower())

def get_all_location_keys() -> list:
    """Get list of all available location keys"""
    return list(DEMO_LOCATIONS.keys())

def get_location_display_name(location_key: str) -> str:
    """Get formatted display name for a location"""
    loc = DEMO_LOCATIONS.get(location_key.lower())
    return loc['display_name'] if loc else location_key.title()