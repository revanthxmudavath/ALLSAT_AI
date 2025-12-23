"""
Sentinel-2 Data Processing Module for ALLSAT AI
Contains SentinelDataManager and VegetationAlertSystem classes
"""

import os
import math
import numpy as np
import rasterio
from datetime import datetime
from rasterio.transform import from_bounds
from sentinelhub import (
    SHConfig, 
    DataCollection, 
    SentinelHubCatalog,
    SentinelHubRequest,
    BBox, 
    bbox_to_dimensions, 
    CRS, 
    MimeType,
)
from geocoding import geocode_with_fallback


def setup_cdse_config(client_id, client_secret, profile_name="cdse"):
    """Setup and save configuration for Copernicus Data Space Ecosystem"""
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.save(profile_name)
    print(f"✅ Configuration saved as profile: '{profile_name}'")
    return config


def load_cdse_config(profile_name="cdse"):
    """Load saved Copernicus Data Space Ecosystem configuration"""
    config = SHConfig(profile_name)
    print(f"✅ Loaded config profile: '{profile_name}'")
    print(f"   Base URL: {config.sh_base_url}")
    return config


class SentinelDataManager:
    """Manage Sentinel-2 data search, download, and processing"""
    
    def __init__(self, config, arcgis_api_key=None, locationiq_api_key=None):
        self.config = config
        self.arcgis_api_key = arcgis_api_key
        self.locationiq_api_key = locationiq_api_key
        if "dataspace.copernicus.eu" not in config.sh_base_url:
            raise ValueError(f"Wrong base URL: {config.sh_base_url}")
        print(f"🔧 SentinelDataManager initialized")
        print(f"   Using: {config.sh_base_url}")
        self.catalog = SentinelHubCatalog(config=self.config)
        
    def geocode_address(self, address):
        """Convert address to coordinates"""
        print(f"\n{'='*80}\nGEOCODING ADDRESS\n{'='*80}")
        print(f"Input: {address}")
        lat, lon, formatted = geocode_with_fallback(address, timeout=10, arcgis_api_key=self.arcgis_api_key, locationiq_api_key=self.locationiq_api_key)
        print(f"✅ Found location:\n   Coordinates: {lat:.4f}°N, {lon:.4f}°W\n   Address: {formatted}\n{'='*80}\n")
        return lat, lon, formatted

    def create_bbox_from_point(self, lat, lon, buffer_km=20):
        """Create bounding box around a point"""
        lat_buffer = (buffer_km / 111.0)
        lon_buffer = (buffer_km / (111.0 * math.cos(math.radians(lat))))
        bbox_coords = [lon - lon_buffer, lat - lat_buffer, lon + lon_buffer, lat + lat_buffer]
        bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
        print(f"{'='*80}\nBOUNDING BOX CREATED\n{'='*80}")
        print(f"Center: {lat:.4f}°N, {lon:.4f}°W\nBuffer: {buffer_km} km radius\n{'='*80}\n")
        return bbox, bbox_coords

    def search_sentinel2_data(self, bbox, time_interval, max_cloud_coverage=20):
        """Search for Sentinel-2 data in the catalog"""
        print(f"{'='*80}\nSEARCHING SENTINEL-2 CATALOG\n{'='*80}")
        print(f"Time range: {time_interval[0]} to {time_interval[1]}")
        search_iterator = self.catalog.search(
            DataCollection.SENTINEL2_L2A,
            bbox=bbox, time=time_interval,
            filter=f"eo:cloud_cover < {max_cloud_coverage}",
            fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"], "exclude": []}
        )
        results = list(search_iterator)
        print(f"✅ Found {len(results)} Sentinel-2 images")
        if results:
            print(f"\nAvailable images:")
            for i, result in enumerate(results[:5], 1):
                date = result['properties']['datetime'][:10]
                cloud = result['properties'].get('eo:cloud_cover', 'N/A')
                print(f"   [{i}] Date: {date}, Cloud: {cloud:.1f}%")
        print(f"{'='*80}\n")
        return results

    def download_ndvi_ndmi(
        self,
        bbox,
        time_interval,
        resolution=20,
        output_dir='sentinel_data',
        max_cloud_coverage=30,
        mosaicking_order=None
    ):
        """Download Sentinel-2 data and calculate NDVI and NDMI"""
        print(f"{'='*80}\nDOWNLOADING & PROCESSING SENTINEL-2 DATA\n{'='*80}")
        os.makedirs(output_dir, exist_ok=True)
        size = bbox_to_dimensions(bbox, resolution=resolution)
        print(f"Output size: {size[0]} x {size[1]} pixels")
        
        cdse_collection = DataCollection.SENTINEL2_L2A.define_from(
            "sentinel-2-l2a-cdse", service_url=self.config.sh_base_url
        )

        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: [{ bands: ["B04", "B08", "B11", "SCL"] }],
                output: [
                    {id: "ndvi", bands: 1, sampleType: "FLOAT32"},
                    {id: "ndmi", bands: 1, sampleType: "FLOAT32"}
                ]
            };
        }
        function evaluatePixel(sample) {
          let ndvi = -9999;
          let ndmi = -9999;

          // Mask only real clouds + snow
          if ([8, 9, 10, 11].includes(sample.SCL)) {
            return { ndvi: [ndvi], ndmi: [ndmi] };
          }

          let ndvi_denom = sample.B08 + sample.B04;
          let ndmi_denom = sample.B08 + sample.B11;

          if (ndvi_denom !== 0) {
            ndvi = (sample.B08 - sample.B04) / ndvi_denom;
          }

          if (ndmi_denom !== 0) {
            ndmi = (sample.B08 - sample.B11) / ndmi_denom;
          }

          return { ndvi: [ndvi], ndmi: [ndmi] };
        }
        """
        
        other_args = {"dataFilter": {"maxCloudCoverage": max_cloud_coverage}}
        if mosaicking_order:
            other_args["processing"] = {"mosaickingOrder": mosaicking_order}

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[SentinelHubRequest.input_data(
                data_collection=cdse_collection,
                time_interval=time_interval,
                other_args=other_args
            )],
            responses=[
                SentinelHubRequest.output_response('ndvi', MimeType.TIFF),
                SentinelHubRequest.output_response('ndmi', MimeType.TIFF)
            ],
            bbox=bbox, size=size, config=self.config
        )
        
        print(f"Downloading data...")
        data = request.get_data()
        if not data: 
            raise ValueError("No data returned.")
        
        ndvi_array = data[0]['ndvi.tif']
        ndmi_array = data[0]['ndmi.tif']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ndvi_path = os.path.join(output_dir, f'ndvi_{timestamp}.tif')
        ndmi_path = os.path.join(output_dir, f'ndmi_{timestamp}.tif')
        
        transform = from_bounds(bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y, size[0], size[1])
        
        for path, array in [(ndvi_path, ndvi_array), (ndmi_path, ndmi_array)]:
            with rasterio.open(path, 'w', driver='GTiff', height=size[1], width=size[0], count=1, 
                               dtype=array.dtype, crs='EPSG:4326', transform=transform, nodata=-9999) as dst:
                dst.write(array, 1)
        
        print(f"\n✅ Data downloaded successfully:\n   NDVI: {ndvi_path}\n   NDMI: {ndmi_path}\n{'='*80}\n")
        return {'ndvi': ndvi_path, 'ndmi': ndmi_path, 'ndvi_data': ndvi_array, 'ndmi_data': ndmi_array}

    def analyze_indices(self, ndvi_data, ndmi_data):
        """Analyze NDVI and NDMI data"""
        def get_clean_data(data):
            return data[(data != -9999) & (~np.isnan(data)) & (~np.isinf(data))]
        
        ndvi_valid = get_clean_data(ndvi_data)
        ndmi_valid = get_clean_data(ndmi_data)
        
        total_pixels = ndvi_data.size
        valid_pixels = ndvi_valid.size
        invalid_pixels = total_pixels - valid_pixels
        invalid_percentage = (invalid_pixels / total_pixels) * 100
        
        stats = {
            'ndvi_mean': float(np.mean(ndvi_valid)) if ndvi_valid.size > 0 else 0.0,
            'ndvi_min': float(np.min(ndvi_valid)) if ndvi_valid.size > 0 else 0.0,
            'ndvi_max': float(np.max(ndvi_valid)) if ndvi_valid.size > 0 else 0.0,
            'ndvi_std': float(np.std(ndvi_valid)) if ndvi_valid.size > 0 else 0.0,
            'ndmi_mean': float(np.mean(ndmi_valid)) if ndmi_valid.size > 0 else 0.0,
            'ndmi_min': float(np.min(ndmi_valid)) if ndmi_valid.size > 0 else 0.0,
            'ndmi_max': float(np.max(ndmi_valid)) if ndmi_valid.size > 0 else 0.0,
            'ndmi_std': float(np.std(ndmi_valid)) if ndmi_valid.size > 0 else 0.0,
            'total_pixels': total_pixels,
            'valid_pixels': valid_pixels,
            'invalid_pixels': invalid_pixels,
            'cloud_percentage': invalid_percentage
        }
        
        print(f"{'='*80}")
        print(f"RESULTS")
        print(f"{'='*80}")
        print(f"Valid Data Coverage: {100 - invalid_percentage:.1f}%")
        print(f"🌱 NDVI Mean: {stats['ndvi_mean']:.3f}")
        print(f"💧 NDMI Mean: {stats['ndmi_mean']:.3f}")
        print(f"{'='*80}\n")
        
        return stats


class VegetationAlertSystem:
    """Alert system for vegetation indices from Sentinel-2"""
    
    def __init__(self, thresholds=None):
        """
        Initialize alert system with thresholds
        
        Args:
            thresholds: Dict with threshold values, e.g.:
                {
                    'ndvi_min': 0.3,     # Alert if NDVI < 0.3 (poor vegetation)
                    'ndvi_max': 0.9,     # Alert if NDVI > 0.9 (unusually high)
                    'ndmi_min': 0.1,     # Alert if NDMI < 0.1 (drought stress)
                    'cloud_max': 30      # Alert if cloud cover > 30%
                }
        """
        self.thresholds = thresholds or {
            'ndvi_min': 0.3,
            'ndvi_max': 0.9,
            'ndmi_min': 0.1,
            'cloud_max': 30
        }
    
    def check_thresholds(self, stats):
        """
        Check if any indices exceed thresholds
        
        Args:
            stats: Statistics dict from Sentinel analysis
            
        Returns:
            list: Alert dictionaries
        """
        alerts = []
        
        # NDVI alerts
        if stats['ndvi_mean'] is not None:
            # Low NDVI - Poor vegetation health
            if 'ndvi_min' in self.thresholds and stats['ndvi_mean'] < self.thresholds['ndvi_min']:
                severity = 'CRITICAL' if stats['ndvi_mean'] < self.thresholds['ndvi_min'] * 0.7 else 'HIGH'
                alerts.append({
                    'type': 'VEGETATION HEALTH - LOW NDVI',
                    'parameter': 'ndvi_mean',
                    'value': stats['ndvi_mean'],
                    'threshold': self.thresholds['ndvi_min'],
                    'deviation': self.thresholds['ndvi_min'] - stats['ndvi_mean'],
                    'severity': severity,
                    'message': f"Average NDVI of {stats['ndvi_mean']:.3f} indicates poor vegetation health (threshold: {self.thresholds['ndvi_min']:.3f})",
                    'recommendation': "Vegetation density is below normal. Check for crop stress, disease, or soil issues."
                })
            
            # High NDVI - Unusually dense vegetation
            if 'ndvi_max' in self.thresholds and stats['ndvi_mean'] > self.thresholds['ndvi_max']:
                alerts.append({
                    'type': 'VEGETATION DENSITY - HIGH NDVI',
                    'parameter': 'ndvi_mean',
                    'value': stats['ndvi_mean'],
                    'threshold': self.thresholds['ndvi_max'],
                    'deviation': stats['ndvi_mean'] - self.thresholds['ndvi_max'],
                    'severity': 'MEDIUM',
                    'message': f"Average NDVI of {stats['ndvi_mean']:.3f} indicates unusually dense vegetation (threshold: {self.thresholds['ndvi_max']:.3f})",
                    'recommendation': "Vegetation is very dense. This may be normal or could indicate overgrowth."
                })
        
        # NDMI alerts
        if stats['ndmi_mean'] is not None:
            # Low NDMI - Drought stress
            if 'ndmi_min' in self.thresholds and stats['ndmi_mean'] < self.thresholds['ndmi_min']:
                severity = 'CRITICAL' if stats['ndmi_mean'] < self.thresholds['ndmi_min'] * 0.5 else 'HIGH'
                alerts.append({
                    'type': 'MOISTURE STRESS - LOW NDMI',
                    'parameter': 'ndmi_mean',
                    'value': stats['ndmi_mean'],
                    'threshold': self.thresholds['ndmi_min'],
                    'deviation': self.thresholds['ndmi_min'] - stats['ndmi_mean'],
                    'severity': severity,
                    'message': f"Average NDMI of {stats['ndmi_mean']:.3f} indicates moisture stress (threshold: {self.thresholds['ndmi_min']:.3f})",
                    'recommendation': "Plants are experiencing water stress. Consider irrigation or check soil moisture."
                })
        
        # Cloud coverage alert
        if 'cloud_max' in self.thresholds and stats['cloud_percentage'] > self.thresholds['cloud_max']:
            alerts.append({
                'type': 'DATA QUALITY - HIGH CLOUD COVERAGE',
                'parameter': 'cloud_percentage',
                'value': stats['cloud_percentage'],
                'threshold': self.thresholds['cloud_max'],
                'deviation': stats['cloud_percentage'] - self.thresholds['cloud_max'],
                'severity': 'MEDIUM',
                'message': f"Cloud coverage of {stats['cloud_percentage']:.1f}% exceeds threshold of {self.thresholds['cloud_max']}%",
                'recommendation': "Data quality may be affected. Consider using data from a different date."
            })
        
        return alerts
    
    def format_alert_report(self, location_info, stats, alerts):
        """
        Format alerts into a readable report
        
        Args:
            location_info: Location dictionary
            stats: Statistics dictionary
            alerts: List of alerts
            
        Returns:
            str: Formatted report
        """
        
        report = f"""
{'='*80}
ALLSAT AI - VEGETATION MONITORING ALERT
{'='*80}

CLIENT LOCATION:
  Address: {location_info['address']}
  Coordinates: {location_info['latitude']:.4f}°N, {location_info['longitude']:.4f}°W
  Coverage Area: {location_info['buffer_km']} km radius

DATA QUALITY:
  Cloud Coverage: {stats['cloud_percentage']:.1f}%
  Valid Pixels: {stats['valid_pixels']:,} / {stats['total_pixels']:,}

{'='*80}
VEGETATION INDICES
{'='*80}

🌱 NDVI (Normalized Difference Vegetation Index)
   Purpose: Measures vegetation health and density
   Range: -1.0 to +1.0 (higher = healthier vegetation)
   
   Your Values:
   • Average: {stats['ndvi_mean']:.3f}
   • Minimum: {stats['ndvi_min']:.3f}
   • Maximum: {stats['ndvi_max']:.3f}
   • Std Dev: {stats['ndvi_std']:.3f}
   
   Interpretation:
   • < 0.2  = Bare soil, rocks, water
   • 0.2-0.3 = Sparse vegetation
   • 0.3-0.6 = Moderate vegetation ✓
   • 0.6-0.8 = Dense vegetation ✓✓
   • > 0.8   = Very dense vegetation

💧 NDMI (Normalized Difference Moisture Index)
   Purpose: Measures plant water content and stress
   Range: -1.0 to +1.0 (higher = more moisture)
   
   Your Values:
   • Average: {stats['ndmi_mean']:.3f}
   • Minimum: {stats['ndmi_min']:.3f}
   • Maximum: {stats['ndmi_max']:.3f}
   • Std Dev: {stats['ndmi_std']:.3f}
   
   Interpretation:
   • < 0.0  = Severe drought stress ⚠️
   • 0.0-0.2 = Moderate stress
   • 0.2-0.4 = Adequate moisture ✓
   • > 0.4   = High moisture content

{'='*80}
ALERT STATUS
{'='*80}
"""
        
        if alerts:
            report += f"\n⚠️  {len(alerts)} ALERT(S) DETECTED:\n\n"
            for i, alert in enumerate(alerts, 1):
                report += f"[{i}] [{alert['severity']}] {alert['type']}\n"
                report += f"    {alert['message']}\n"
                report += f"    Value: {alert['value']:.3f} | Threshold: {alert['threshold']:.3f}\n"
                report += f"    💡 {alert.get('recommendation', 'Monitor the situation.')}\n\n"
        else:
            report += f"\n✅ NO ALERTS - All parameters within normal range\n\n"
        
        report += f"{'='*80}\n"
        report += f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"{'='*80}\n"
        
        return report


