"""
ERA5-Land Data Processing Module for ALLSAT AI
Contains ERA5DataManager and ClimateAlertSystem classes
"""

import cdsapi
import xarray as xr
import zipfile
import os
import numpy as np
import pandas as pd
import math
from datetime import datetime


class CoordinateError(Exception):
    """Raised when coordinates are outside the data range"""
    pass


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers"""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


class ERA5DataManager:
    """Manage ERA5-Land data download and processing"""
    
    def __init__(self, cds_key):
        """
        Initialize ERA5 Data Manager
        
        Args:
            cds_key: CDS API key (UID:API-key format)
        """
        self.cds_key = cds_key
        self.client = cdsapi.Client(
            url="https://cds.climate.copernicus.eu/api",
            key=cds_key
        )
        print("🔧 ERA5DataManager initialized")
    
    def fetch_era5_for_location(self, lat, lon, start_date, end_date, variables, output_dir='era5_data'):
        """
        Fetch ERA5-Land data for a specific location
        
        Args:
            lat: Latitude (decimal degrees)
            lon: Longitude (decimal degrees)
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
            variables: List of variable names
            output_dir: Directory to save data
            
        Returns:
            xarray.Dataset: ERA5 data
        """
        print(f"\n{'='*80}\nFETCHING ERA5-LAND DATA\n{'='*80}")
        print(f"Location: {lat:.4f}°N, {lon:.4f}°W")
        print(f"Period: {start_date} to {end_date}")
        print(f"Variables: {', '.join(variables)}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Define bounding box (0.5 degree buffer around point)
        buffer = 0.5
        area = [lat + buffer, lon - buffer, lat - buffer, lon + buffer]
        
        # Parse years and months from date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        years = [str(y) for y in range(start_dt.year, end_dt.year + 1)]
        months = [f"{m:02d}" for m in range(1, 13)]
        
        request = {
            'variable': variables,
            'year': years,
            'month': months,
            'area': area,
            'format': 'netcdf',
        }
        
        download_file = os.path.join(output_dir, f'era5_data_{lat}_{lon}.zip')
        
        print(f"\n📡 Downloading data from CDS API...")
        
        
        self.client.retrieve(
            'reanalysis-era5-land-monthly-means',
            request,
            download_file
        )
        
        # Check if the file is a ZIP archive and extract it if so
        if zipfile.is_zipfile(download_file):
            print(f"📦 Downloaded file is a ZIP archive. Extracting...")
            extract_dir = os.path.join(output_dir, f'era5_data_{lat}_{lon}_extracted')
            with zipfile.ZipFile(download_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find the .nc file inside the extracted directory
            found_files = [f for f in os.listdir(extract_dir) if f.endswith('.nc')]
            if not found_files:
                raise ValueError("No .nc file found in the extracted archive.")
            
            nc_path = os.path.join(extract_dir, found_files[0])
            ds = xr.open_dataset(nc_path, engine='netcdf4')
            final_path = nc_path
        else:
            # It's likely a direct NetCDF file
            ds = xr.open_dataset(download_file, engine='netcdf4')
            final_path = download_file

        print(f"✅ Data downloaded successfully")
        print(f"{'='*80}\n")
        
        return ds, final_path
    
    def validate_coordinates(self, ds, client_lat, client_lon, max_distance_km=50):
        """
        Validate that client coordinates are within the dataset bounds
        
        Args:
            ds: xarray Dataset
            client_lat: Client latitude
            client_lon: Client longitude
            max_distance_km: Maximum acceptable distance from data boundary
            
        Returns:
            bool: True if coordinates are valid
            
        Raises:
            CoordinateError: If coordinates are outside bounds or too far away
        """
        # Get data bounds
        lat_min = float(ds.latitude.min())
        lat_max = float(ds.latitude.max())
        lon_min = float(ds.longitude.min())
        lon_max = float(ds.longitude.max())
        
        # Check if coordinates are within bounds (with small tolerance)
        tolerance = 0.01  # ~1km
        lat_in_bounds = (lat_min - tolerance) <= client_lat <= (lat_max + tolerance)
        lon_in_bounds = (lon_min - tolerance) <= client_lon <= (lon_max + tolerance)
        
        if not lat_in_bounds or not lon_in_bounds:
            # Calculate distance to nearest data point
            if not lat_in_bounds:
                nearest_lat = lat_min if client_lat < lat_min else lat_max
            else:
                nearest_lat = client_lat
                
            if not lon_in_bounds:
                nearest_lon = lon_min if client_lon < lon_min else lon_max
            else:
                nearest_lon = client_lon
            
            distance = haversine_distance(client_lat, client_lon, nearest_lat, nearest_lon)
            
            raise CoordinateError(
                f"\n{'='*80}\n"
                f"❌ CLIENT COORDINATES OUT OF RANGE\n"
                f"{'='*80}\n"
                f"Client location: {client_lat:.4f}°N, {client_lon:.4f}°W\n"
                f"Data coverage:   {lat_min:.4f}° to {lat_max:.4f}°N\n"
                f"                 {lon_min:.4f}° to {lon_max:.4f}°W\n"
                f"Distance to nearest data: {distance:.0f} km\n\n"
                f"{'='*80}\n"
            )
        
        # Find actual nearest point and verify distance
        ds_nearest = ds.sel(latitude=client_lat, longitude=client_lon, method='nearest')
        actual_lat = float(ds_nearest.latitude)
        actual_lon = float(ds_nearest.longitude)
        
        distance = haversine_distance(client_lat, client_lon, actual_lat, actual_lon)
        
        if distance > max_distance_km:
            raise CoordinateError(
                f"\n{'='*80}\n"
                f"⚠️  NEAREST DATA POINT TOO FAR\n"
                f"{'='*80}\n"
                f"Client location: {client_lat:.4f}°N, {client_lon:.4f}°W\n"
                f"Nearest data at: {actual_lat:.4f}°N, {actual_lon:.4f}°W\n"
                f"Distance: {distance:.1f} km (max allowed: {max_distance_km} km)\n"
                f"{'='*80}\n"
            )
        
        return True, actual_lat, actual_lon, distance
    
    def process_era5_for_alerts(self, netcdf_path, client_lat, client_lon, thresholds, max_distance_km=50, verbose=True, start_date=None, end_date=None):
        """
        Process ERA5 data and check for threshold violations
        
        Args:
            netcdf_path: Path to NetCDF file
            client_lat: Client latitude
            client_lon: Client longitude
            thresholds: Dict of thresholds
            max_distance_km: Max distance for nearest point
            verbose: Print detailed output
            
        Returns:
            tuple: (data dict, alerts list)
        """
        if zipfile.is_zipfile(netcdf_path):
            raise ValueError(f"Expected .nc, got ZIP: {netcdf_path}")
        
        with open(netcdf_path, "rb") as f:
            magic = f.read(4)
        
        if not (magic.startswith(b"CDF") or magic == b"\x89HDF"): 
            raise ValueError(f"File is not NetCDF/HDF5. Magic={magic!r}. Path={netcdf_path}")

        # Open dataset
        ds = xr.open_dataset(netcdf_path, engine='netcdf4')
        if start_date and end_date: 
            ds = ds.sel(valid_time=slice(pd.to_datetime(start_date), pd.to_datetime(end_date)))
        
        # Validate coordinates
        if verbose:
            print(f"{'='*80}")
            print(f"VALIDATING COORDINATES")
            print(f"{'='*80}")
        
        is_valid, actual_lat, actual_lon, distance = self.validate_coordinates(
            ds, client_lat, client_lon, max_distance_km
        )
        
        if verbose:
            print(f"✅ Coordinates valid")
            print(f"   Requested: {client_lat:.4f}°N, {client_lon:.4f}°W")
            print(f"   Actual:    {actual_lat:.4f}°N, {actual_lon:.4f}°W")
            print(f"   Distance:  {distance:.1f} km")
            print(f"{'='*80}\n")
        
        # Extract data for the nearest point
        data_point = ds.sel(latitude=actual_lat, longitude=actual_lon, method='nearest')
        
        # Extract variables
        temp_k = data_point['t2m'].values  # Temperature in Kelvin
        temp_c = temp_k - 273.15  # Convert to Celsius
        temp_f = (temp_c * 9/5) + 32  # Convert to Fahrenheit
        
        precip_m = data_point['tp'].values  # Precipitation in meters
        precip_mm = precip_m * 1000  # Convert to millimeters
        
        soil_moisture = data_point['swvl1'].values  # Soil moisture (m³/m³)
        
        # Get time information
        times = pd.to_datetime(data_point['valid_time'].values)
        
        # Compile data
        data = {
            'period_start': times[0],
            'period_end': times[-1],
            'n_months': len(times),
            'actual_lat': actual_lat,
            'actual_lon': actual_lon,
            'distance_km': distance,
            
            # Temperature metrics (Celsius)
            'avg_temp_c': float(np.mean(temp_c)),
            'max_temp_c': float(np.max(temp_c)),
            'min_temp_c': float(np.min(temp_c)),
            
            # Temperature in Fahrenheit
            'avg_temp_f': float(np.mean(temp_f)),
            'max_temp_f': float(np.max(temp_f)),
            'min_temp_f': float(np.min(temp_f)),
            
            # Precipitation metrics (millimeters)
            'total_precip_mm': float(np.sum(precip_mm)),
            'avg_precip_mm': float(np.mean(precip_mm)),
            'max_precip_mm': float(np.max(precip_mm)),
            
            # Soil moisture metrics
            'avg_soil_moisture': float(np.mean(soil_moisture)),
            'min_soil_moisture': float(np.min(soil_moisture)),
            'max_soil_moisture': float(np.max(soil_moisture)),
            
            # Time series data for plotting
            'times': times,
            'temp_c': temp_c,
            'temp_f': temp_f,
            'precip_mm': precip_mm,
            'soil_moisture': soil_moisture,
        }
        
        # Print summary if verbose
        if verbose:
            print(f"{'='*80}")
            print(f"ENVIRONMENTAL DATA SUMMARY")
            print(f"{'='*80}")
            print(f"Period: {data['period_start'].strftime('%Y-%m-%d')} to {data['period_end'].strftime('%Y-%m-%d')} ({data['n_months']} months)\n")
            
            print(f"🌡️  TEMPERATURE:")
            print(f"   Average: {data['avg_temp_c']:6.1f}°C ({data['avg_temp_f']:6.1f}°F)")
            print(f"   Maximum: {data['max_temp_c']:6.1f}°C ({data['max_temp_f']:6.1f}°F)")
            print(f"   Minimum: {data['min_temp_c']:6.1f}°C ({data['min_temp_f']:6.1f}°F)")
            
            print(f"\n💧 PRECIPITATION:")
            print(f"   Total:        {data['total_precip_mm']:6.1f} mm")
            print(f"   Avg/month:    {data['avg_precip_mm']:6.1f} mm")
            print(f"   Max in month: {data['max_precip_mm']:6.1f} mm")
            
            print(f"\n🌱 SOIL MOISTURE:")
            print(f"   Average: {data['avg_soil_moisture']:.3f} m³/m³")
            print(f"   Minimum: {data['min_soil_moisture']:.3f} m³/m³")
            print(f"   Maximum: {data['max_soil_moisture']:.3f} m³/m³")
            print(f"{'='*80}\n")
        
        # Check thresholds and generate alerts
        alerts = []
        
        # Temperature alerts
        if 'max_temp_c' in thresholds and data['max_temp_c'] > thresholds['max_temp_c']:
            alerts.append({
                'type': 'TEMPERATURE - EXCESSIVE HEAT',
                'parameter': 'max_temp_c',
                'value': data['max_temp_c'],
                'threshold': thresholds['max_temp_c'],
                'deviation': data['max_temp_c'] - thresholds['max_temp_c'],
                'severity': 'CRITICAL' if data['max_temp_c'] > thresholds['max_temp_c'] * 1.15 else 'HIGH',
                'message': f"Maximum temperature of {data['max_temp_c']:.1f}°C exceeds threshold of {thresholds['max_temp_c']}°C",
                'recommendation': "Extreme heat detected. Monitor crops for heat stress and consider irrigation."
            })
        
        if 'min_temp_c' in thresholds and data['min_temp_c'] < thresholds['min_temp_c']:
            alerts.append({
                'type': 'TEMPERATURE - FREEZE WARNING',
                'parameter': 'min_temp_c',
                'value': data['min_temp_c'],
                'threshold': thresholds['min_temp_c'],
                'deviation': thresholds['min_temp_c'] - data['min_temp_c'],
                'severity': 'CRITICAL' if data['min_temp_c'] < thresholds['min_temp_c'] * 1.15 else 'HIGH',
                'message': f"Minimum temperature of {data['min_temp_c']:.1f}°C below threshold of {thresholds['min_temp_c']}°C",
                'recommendation': "Freezing temperatures detected. Risk of frost damage to crops."
            })
        
        # Precipitation alerts
        if 'total_precip_mm' in thresholds and data['total_precip_mm'] > thresholds['total_precip_mm']:
            alerts.append({
                'type': 'PRECIPITATION - EXCESSIVE RAINFALL',
                'parameter': 'total_precip_mm',
                'value': data['total_precip_mm'],
                'threshold': thresholds['total_precip_mm'],
                'deviation': data['total_precip_mm'] - thresholds['total_precip_mm'],
                'severity': 'CRITICAL' if data['total_precip_mm'] > thresholds['total_precip_mm'] * 1.3 else 'HIGH',
                'message': f"Total precipitation of {data['total_precip_mm']:.1f}mm exceeds threshold of {thresholds['total_precip_mm']}mm",
                'recommendation': "Excessive rainfall detected. Monitor for flooding and waterlogging."
            })
        
        if 'min_precip_mm' in thresholds and data['total_precip_mm'] < thresholds['min_precip_mm']:
            alerts.append({
                'type': 'PRECIPITATION - DROUGHT RISK',
                'parameter': 'total_precip_mm',
                'value': data['total_precip_mm'],
                'threshold': thresholds['min_precip_mm'],
                'deviation': thresholds['min_precip_mm'] - data['total_precip_mm'],
                'severity': 'HIGH',
                'message': f"Total precipitation of {data['total_precip_mm']:.1f}mm below threshold of {thresholds['min_precip_mm']}mm",
                'recommendation': "Low precipitation detected. Drought risk - consider irrigation."
            })
        
        # Soil moisture alerts
        if 'min_soil_moisture' in thresholds and data['min_soil_moisture'] < thresholds['min_soil_moisture']:
            alerts.append({
                'type': 'SOIL MOISTURE - DRY CONDITIONS',
                'parameter': 'min_soil_moisture',
                'value': data['min_soil_moisture'],
                'threshold': thresholds['min_soil_moisture'],
                'deviation': thresholds['min_soil_moisture'] - data['min_soil_moisture'],
                'severity': 'CRITICAL' if data['min_soil_moisture'] < thresholds['min_soil_moisture'] * 0.7 else 'MEDIUM',
                'message': f"Soil moisture of {data['min_soil_moisture']:.3f} below threshold of {thresholds['min_soil_moisture']:.3f}",
                'recommendation': "Dry soil conditions detected. Plants may be experiencing water stress."
            })
        
        # Print alerts summary
        if verbose:
            print(f"{'='*80}")
            print(f"ALERT STATUS")
            print(f"{'='*80}")
            
            if alerts:
                print(f"⚠️  {len(alerts)} ALERT(S) DETECTED:\n")
                for i, alert in enumerate(alerts, 1):
                    print(f"  [{i}] [{alert['severity']}] {alert['type']}")
                    print(f"      {alert['message']}")
                    print(f"      Deviation: {alert['deviation']:.2f}\n")
            else:
                print(f"✅ NO ALERTS - All parameters within normal range\n")
            
            print(f"{'='*80}\n")
        
        # Close dataset
        ds.close()
        
        return data, alerts


class ClimateAlertSystem:
    """Alert system for ERA5-Land climate data"""
    
    def __init__(self, thresholds=None):
        """
        Initialize climate alert system with thresholds
        
        Args:
            thresholds: Dict with threshold values
        """
        self.thresholds = thresholds or {
            'max_temp_c': 35,
            'min_temp_c': -10,
            'total_precip_mm': 500,
            'min_precip_mm': 100,
            'min_soil_moisture': 0.20,
        }
    
    def format_climate_report(self, location_info, data, alerts):
        """
        Format climate data into a readable report
        
        Args:
            location_info: Location dictionary
            data: Climate data dictionary
            alerts: List of alerts
            
        Returns:
            str: Formatted report
        """
        report = f"""
{'='*80}
ALLSAT AI - CLIMATE MONITORING ALERT
{'='*80}

CLIENT LOCATION:
  Address: {location_info['address']}
  Coordinates: {location_info['latitude']:.4f}°N, {location_info['longitude']:.4f}°W
  Data Point: {data['actual_lat']:.4f}°N, {data['actual_lon']:.4f}°W
  Distance: {data['distance_km']:.1f} km

TIME PERIOD:
  Start: {data['period_start'].strftime('%Y-%m-%d')}
  End: {data['period_end'].strftime('%Y-%m-%d')}
  Duration: {data['n_months']} months

{'='*80}
CLIMATE METRICS
{'='*80}

🌡️  TEMPERATURE (2m above ground)
   Average: {data['avg_temp_c']:.1f}°C ({data['avg_temp_f']:.1f}°F)
   Maximum: {data['max_temp_c']:.1f}°C ({data['max_temp_f']:.1f}°F)
   Minimum: {data['min_temp_c']:.1f}°C ({data['min_temp_f']:.1f}°F)

💧 PRECIPITATION
   Total:        {data['total_precip_mm']:.1f} mm
   Average/month: {data['avg_precip_mm']:.1f} mm
   Max in month:  {data['max_precip_mm']:.1f} mm

🌱 SOIL MOISTURE (Top Layer)
   Average: {data['avg_soil_moisture']:.3f} m³/m³
   Minimum: {data['min_soil_moisture']:.3f} m³/m³
   Maximum: {data['max_soil_moisture']:.3f} m³/m³

{'='*80}
ALERT STATUS
{'='*80}
"""
        
        if alerts:
            report += f"\n⚠️  {len(alerts)} ALERT(S) DETECTED:\n\n"
            for i, alert in enumerate(alerts, 1):
                report += f"[{i}] [{alert['severity']}] {alert['type']}\n"
                report += f"    {alert['message']}\n"
                report += f"    Value: {alert['value']:.2f} | Threshold: {alert['threshold']:.2f}\n"
                report += f"    💡 {alert.get('recommendation', 'Monitor conditions.')}\n\n"
        else:
            report += f"\n✅ NO ALERTS - All parameters within normal range\n\n"
        
        report += f"{'='*80}\n"
        report += f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"{'='*80}\n"
        
        return report