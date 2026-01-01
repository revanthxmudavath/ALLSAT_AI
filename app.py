# """
# ALLSAT AI - Environmental Warning & Intelligence System (EWIS)
# Streamlit Prototype for Sentinel-2 Vegetation Monitoring & ERA5-Land Climate Analysis
# """

# import streamlit as st
# import numpy as np
# import rasterio
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from datetime import datetime, timedelta
# from io import BytesIO
# import tempfile
# from geocoding import geocode_with_fallback

# # Import custom modules
# from sentinel_processor import (
#     SentinelDataManager,
#     VegetationAlertSystem,
#     setup_cdse_config,
#     load_cdse_config
# )
# from era5_processor import (
#     ERA5DataManager,
#     ClimateAlertSystem,
#     CoordinateError
# )

# from risk_integration import (
#     generate_risk_maps_from_data,
#     format_risk_report,
#     get_risk_alert_level,
#     interpret_risk_level
# )

# from pathlib import Path 

# # Page config
# st.set_page_config(
#     page_title="ALLSAT AI - EWIS",
#     page_icon="🛰️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# try:
#     arcgis_api_key = st.secrets["ARC_GIS_API_KEY"]
    
# except KeyError:
#     arcgis_api_key = None
#     print("⚠️ ArcGIS key not found in secrets")

# try:
#     locationiq_api_key = st.secrets["LOCATIONIQ_API_KEY"]
    
# except KeyError:
#     locationiq_api_key = None
#     print("⚠️ LocationIQ key not found in secrets")

# # Custom CSS for better UI
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 1rem;
#     }
#     .sub-header {
#         font-size: 1.2rem;
#         text-align: center;
#         color: #666;
#         margin-bottom: 2rem;
#     }
#     .metric-card {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin: 0.5rem 0;
#     }
#     .alert-critical {
#         background-color: #ffebee;
#         padding: 1rem;
#         border-left: 4px solid #f44336;
#         margin: 0.5rem 0;
#     }
#     .alert-high {
#         background-color: #fff3e0;
#         padding: 1rem;
#         border-left: 4px solid #ff9800;
#         margin: 0.5rem 0;
#     }
#     .alert-medium {
#         background-color: #e3f2fd;
#         padding: 1rem;
#         border-left: 4px solid #2196f3;
#         margin: 0.5rem 0;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# def visualize_tiff_files(ndvi_path, ndmi_path, nodata=-9999):
#     """
#     Comprehensive NDVI/NDMI visualization with 7 analysis panels
    
#     Args:
#         ndvi_path: Path to NDVI GeoTIFF
#         ndmi_path: Path to NDMI GeoTIFF
#         nodata: No-data value to mask
        
#     Returns:
#         BytesIO: Buffer containing the complete analysis figure
#     """
    
#     # Load data
#     with rasterio.open(ndvi_path) as src:
#         ndvi = src.read(1).astype(np.float32)
#         ndvi_masked = np.ma.masked_where(ndvi == nodata, ndvi)
    
#     with rasterio.open(ndmi_path) as src:
#         ndmi = src.read(1).astype(np.float32)
#         ndmi_masked = np.ma.masked_where(ndmi == nodata, ndmi)
    
#     # Get valid data only (for statistics)
#     ndvi_valid = ndvi_masked.compressed()
#     ndmi_valid = ndmi_masked.compressed()
    
#     # Calculate statistics
#     ndvi_stats = {
#         'mean': np.mean(ndvi_valid),
#         'median': np.median(ndvi_valid),
#         'std': np.std(ndvi_valid),
#         'min': np.min(ndvi_valid),
#         'max': np.max(ndvi_valid)
#     }
    
#     ndmi_stats = {
#         'mean': np.mean(ndmi_valid),
#         'median': np.median(ndmi_valid),
#         'std': np.std(ndmi_valid),
#         'min': np.min(ndmi_valid),
#         'max': np.max(ndmi_valid)
#     }
    
#     # Calculate correlation (only where both have valid data)
#     valid_mask = ~ndvi_masked.mask & ~ndmi_masked.mask
#     ndvi_for_corr = ndvi[valid_mask]
#     ndmi_for_corr = ndmi[valid_mask]
#     correlation = np.corrcoef(ndvi_for_corr, ndmi_for_corr)[0, 1]
    
#     # Calculate coverage
#     total_pixels = ndvi.size
#     valid_pixels = len(ndvi_valid)
#     coverage_pct = (valid_pixels / total_pixels) * 100
    
#     # =========================================================================
#     # CREATE FIGURE WITH 7 PANELS
#     # =========================================================================
    
#     fig = plt.figure(figsize=(20, 12))
#     gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
#     # Title
#     fig.suptitle(
#         'NDVI vs NDMI Analysis',
#         fontsize=18,
#         fontweight='bold',
#         y=0.98
#     )
    
#     # -------------------------------------------------------------------------
#     # PANEL 1: NDVI Map (Top Left)
#     # -------------------------------------------------------------------------
#     ax1 = fig.add_subplot(gs[0, 0])
    
#     cmap_ndvi = plt.cm.RdYlGn.copy()
#     cmap_ndvi.set_bad(color='white', alpha=0.3)
    
#     im1 = ax1.imshow(ndvi_masked, cmap=cmap_ndvi, vmin=-0.2, vmax=0.8)
#     ax1.set_title('NDVI (Vegetation Health)', fontsize=12, fontweight='bold')
#     ax1.axis('off')
    
#     cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
#     cbar1.set_label('NDVI', fontsize=10)
    
#     # -------------------------------------------------------------------------
#     # PANEL 2: NDMI Map (Top Center-Left)
#     # -------------------------------------------------------------------------
#     ax2 = fig.add_subplot(gs[0, 1])
    
#     cmap_ndmi = plt.cm.BrBG.copy()
#     cmap_ndmi.set_bad(color='white', alpha=0.3)
    
#     im2 = ax2.imshow(ndmi_masked, cmap=cmap_ndmi, vmin=-0.4, vmax=0.6)
#     ax2.set_title('NDMI (Moisture Content)', fontsize=12, fontweight='bold')
#     ax2.axis('off')
    
#     cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
#     cbar2.set_label('NDMI', fontsize=10)
    
#     # -------------------------------------------------------------------------
#     # PANEL 3: Difference Map (NDVI - NDMI) (Top Center-Right)
#     # -------------------------------------------------------------------------
#     ax3 = fig.add_subplot(gs[0, 2])
    
#     difference = ndvi_masked - ndmi_masked
    
#     im3 = ax3.imshow(difference, cmap='RdBu_r', vmin=-0.2, vmax=0.8)
#     ax3.set_title('Difference (NDVI - NDMI)', fontsize=12, fontweight='bold')
#     ax3.axis('off')
    
#     cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
#     cbar3.set_label('NDVI - NDMI', fontsize=10)
    
#     # -------------------------------------------------------------------------
#     # PANEL 4: Distribution Comparison (Top Right)
#     # -------------------------------------------------------------------------
#     ax4 = fig.add_subplot(gs[1, 0])
    
#     ax4.hist(
#         ndvi_valid,
#         bins=50,
#         alpha=0.6,
#         color='green',
#         label='NDVI',
#         density=True,
#         edgecolor='darkgreen'
#     )
#     ax4.hist(
#         ndmi_valid,
#         bins=50,
#         alpha=0.6,
#         color='blue',
#         label='NDMI',
#         density=True,
#         edgecolor='darkblue'
#     )
    
#     ax4.set_xlabel('Index Value', fontsize=10)
#     ax4.set_ylabel('Density', fontsize=10)
#     ax4.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
#     ax4.legend(loc='upper left')
#     ax4.grid(True, alpha=0.3)
#     ax4.set_xlim(-1, 1)
    
#     # -------------------------------------------------------------------------
#     # PANEL 5: NDVI vs NDMI Scatter Plot 
#     # -------------------------------------------------------------------------
#     ax5 = fig.add_subplot(gs[1, 1:3])
    
#     # Downsample if too many points (for performance)
#     if len(ndvi_for_corr) > 50000:
#         sample_idx = np.random.choice(len(ndvi_for_corr), 50000, replace=False)
#         ndvi_sample = ndvi_for_corr[sample_idx]
#         ndmi_sample = ndmi_for_corr[sample_idx]
#     else:
#         ndvi_sample = ndvi_for_corr
#         ndmi_sample = ndmi_for_corr
    
#     scatter = ax5.scatter(
#         ndmi_sample,
#         ndvi_sample,
#         c=ndvi_sample,
#         cmap='RdYlGn',
#         s=1,
#         alpha=0.5,
#         vmin=-0.2,
#         vmax=0.8
#     )
    
#     ax5.plot([-1, 1], [-1, 1], 'r--', linewidth=1.5, alpha=0.5, label='1:1 line')
    
#     ax5.text(
#         0.05,
#         0.95,
#         f'r = {correlation:.3f}',
#         transform=ax5.transAxes,
#         fontsize=12,
#         fontweight='bold',
#         verticalalignment='top',
#         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#     )
    
#     ax5.set_xlabel('NDMI', fontsize=11, fontweight='bold')
#     ax5.set_ylabel('NDVI', fontsize=11, fontweight='bold')
#     ax5.set_title('NDVI vs NDMI Scatter', fontsize=12, fontweight='bold')
#     ax5.grid(True, alpha=0.3)
#     ax5.set_xlim(-1, 1)
#     ax5.set_ylim(-1, 1)
#     ax5.legend(loc='lower right')
    
#     plt.colorbar(scatter, ax=ax5, label='NDVI Value')
    
#     # -------------------------------------------------------------------------
#     # PANEL 6: Statistical Comparison (Middle Center-Right)
#     # -------------------------------------------------------------------------
#     ax6 = fig.add_subplot(gs[2, 0])
    
#     bp = ax6.boxplot(
#         [ndvi_valid, ndmi_valid],
#         tick_labels=['NDVI', 'NDMI'],
#         patch_artist=True,
#         showmeans=True,
#         meanline=True
#     )
    
#     bp['boxes'][0].set_facecolor('lightgreen')
#     bp['boxes'][1].set_facecolor('lightblue')
    
#     ax6.set_ylabel('Index Value', fontsize=10, fontweight='bold')
#     ax6.set_title('Statistical Comparison', fontsize=12, fontweight='bold')
#     ax6.grid(True, alpha=0.3, axis='y')
#     ax6.set_ylim(-1, 1)
    
#     # -------------------------------------------------------------------------
#     # PANEL 7: Valid Data Coverage Map (Middle Right)
#     # -------------------------------------------------------------------------
#     ax7 = fig.add_subplot(gs[2, 1])
    
#     coverage_mask = (~ndvi_masked.mask).astype(int)
    
#     im7 = ax7.imshow(coverage_mask, cmap='RdYlGn', vmin=0, vmax=1)
#     ax7.set_title('Valid Data Coverage', fontsize=12, fontweight='bold')
#     ax7.axis('off')
    
#     ax7.text(
#         0.5,
#         -0.1,
#         f'{coverage_pct:.1f}% valid',
#         transform=ax7.transAxes,
#         fontsize=11,
#         fontweight='bold',
#         ha='center',
#         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7)
#     )
    
#     # -------------------------------------------------------------------------
#     # PANEL 8: NDVI Statistics Table (Bottom Left)
#     # -------------------------------------------------------------------------
#     ax8 = fig.add_subplot(gs[2, 2])
#     ax8.axis('off')
    
#     ndvi_table_data = [
#         ['Metric', 'Value'],
#         ['Mean', f'{ndvi_stats["mean"]:.4f}'],
#         ['Median', f'{ndvi_stats["median"]:.4f}'],
#         ['Std Dev', f'{ndvi_stats["std"]:.4f}'],
#         ['Min', f'{ndvi_stats["min"]:.4f}'],
#         ['Max', f'{ndvi_stats["max"]:.4f}'],
#         ['Range', f'[{ndvi_stats["min"]:.1f}, {ndvi_stats["max"]:.1f}]']
#     ]
    
#     table1 = ax8.table(
#         cellText=ndvi_table_data,
#         loc='center',
#         cellLoc='left',
#         colWidths=[0.3, 0.3],
#         bbox=[0.0, 0.0, 0.45, 1.0]
#     )
    
#     table1.auto_set_font_size(False)
#     table1.set_fontsize(10)
#     table1.scale(1, 2)
    
#     for i in range(2):
#         table1[(0, i)].set_facecolor('#4CAF50')
#         table1[(0, i)].set_text_props(weight='bold', color='white')
    
#     for i in range(1, len(ndvi_table_data)):
#         for j in range(2):
#             if i % 2 == 0:
#                 table1[(i, j)].set_facecolor('#f0f0f0')
    
#     ax8.text(
#         0.225,
#         1.05,
#         'NDVI STATISTICS',
#         transform=ax8.transAxes,
#         fontsize=12,
#         fontweight='bold',
#         ha='center',
#         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7)
#     )
    
#     ndmi_table_data = [
#         ['Metric', 'Value'],
#         ['Mean', f'{ndmi_stats["mean"]:.4f}'],
#         ['Median', f'{ndmi_stats["median"]:.4f}'],
#         ['Std Dev', f'{ndmi_stats["std"]:.4f}'],
#         ['Min', f'{ndmi_stats["min"]:.4f}'],
#         ['Max', f'{ndmi_stats["max"]:.4f}'],
#         ['Range', f'[{ndmi_stats["min"]:.1f}, {ndmi_stats["max"]:.1f}]']
#     ]
    
#     table2 = ax8.table(
#         cellText=ndmi_table_data,
#         loc='center',
#         cellLoc='left',
#         colWidths=[0.3, 0.3],
#         bbox=[0.55, 0.0, 0.45, 1.0]
#     )
    
#     table2.auto_set_font_size(False)
#     table2.set_fontsize(10)
#     table2.scale(1, 2)
    
#     for i in range(2):
#         table2[(0, i)].set_facecolor('#2196F3')
#         table2[(0, i)].set_text_props(weight='bold', color='white')
    
#     for i in range(1, len(ndmi_table_data)):
#         for j in range(2):
#             if i % 2 == 0:
#                 table2[(i, j)].set_facecolor('#f0f0f0')
    
#     ax8.text(
#         0.775,
#         1.05,
#         'NDMI STATISTICS',
#         transform=ax8.transAxes,
#         fontsize=12,
#         fontweight='bold',
#         ha='center',
#         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
#     )
    
       
#     buf = BytesIO()
#     plt.savefig(
#         buf,
#         format='png',
#         dpi=150,
#         bbox_inches='tight',
#         facecolor='white',
#         edgecolor='none'
#     )
#     buf.seek(0)
#     plt.close()
    
#     return buf


# def visualize_climate_data(data):
#     """
#     Visualize ERA5 climate data time series
    
#     Args:
#         data: Climate data dictionary with time series
        
#     Returns:
#         BytesIO: Buffer containing the matplotlib figure
#     """
#     fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
#     times = data['times']
    
#     # Temperature plot
#     axes[0].plot(times, data['temp_c'], 'r-', linewidth=2, label='Temperature')
#     axes[0].set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
#     axes[0].set_title('Temperature Over Time', fontsize=14, fontweight='bold')
#     axes[0].grid(True, alpha=0.3)
#     axes[0].axhline(y=0, color='blue', linestyle='--', alpha=0.5, label='Freezing Point')
#     axes[0].legend()
    
#     # Precipitation plot
#     axes[1].bar(times, data['precip_mm'], color='steelblue', alpha=0.7, width=20)
#     axes[1].set_ylabel('Precipitation (mm)', fontsize=12, fontweight='bold')
#     axes[1].set_title('Monthly Precipitation', fontsize=14, fontweight='bold')
#     axes[1].grid(True, alpha=0.3, axis='y')
    
#     # Soil moisture plot
#     axes[2].plot(times, data['soil_moisture'], 'g-', linewidth=2, marker='o', label='Soil Moisture')
#     axes[2].set_ylabel('Soil Moisture (m³/m³)', fontsize=12, fontweight='bold')
#     axes[2].set_xlabel('Date', fontsize=12, fontweight='bold')
#     axes[2].set_title('Soil Moisture (Top Layer)', fontsize=14, fontweight='bold')
#     axes[2].grid(True, alpha=0.3)
#     axes[2].legend()
    
#     # Format x-axis
#     for ax in axes:
#         ax.tick_params(axis='x', rotation=45)
    
#     plt.tight_layout()
    
#     # Save to buffer
#     buf = BytesIO()
#     plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
#     buf.seek(0)
#     plt.close()
    
#     return buf

# def process_sentinel_data(address, client_id, client_secret, buffer_km, days_back, max_cloud, resolution, thresholds):
#     """
#     Main processing function for Sentinel-2 data
    
#     Returns:
#         dict: Results including stats, alerts, and file paths
#     """
#     # Create output directory
#     output_dir = tempfile.mkdtemp(prefix='sentinel_')
    
#     # Setup config
#     try:
#         config = load_cdse_config("cdse")
#     except:
#         config = setup_cdse_config(client_id, client_secret, "cdse")
    
#     # Initialize manager
#     manager = SentinelDataManager(config, arcgis_api_key=arcgis_api_key, locationiq_api_key=locationiq_api_key)
    
#     # Geocode address
#     st.info(f"🗺️ Geocoding address: {address}")
#     lat, lon, formatted_address = manager.geocode_address(address)
    
#     # Create bounding box
#     bbox, bbox_coords = manager.create_bbox_from_point(lat, lon, buffer_km)
    
#     # Define time interval
#     end_date = datetime.now()
#     start_date = end_date - timedelta(days=days_back)
#     time_interval = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
#     # Search catalog
#     st.info(f"🔍 Searching Sentinel-2 catalog from {time_interval[0]} to {time_interval[1]}")
#     results = manager.search_sentinel2_data(bbox, time_interval, max_cloud)
    
#     if len(results) == 0:
#         st.error("❌ No suitable images found for this location and time period")
#         return None
    
#     st.success(f"✅ Found {len(results)} images with cloud coverage < {max_cloud}%")
    
#     # Download and process
#     st.info("📡 Downloading and processing satellite imagery...")
#     output = manager.download_ndvi_ndmi(
#         bbox,
#         time_interval,
#         resolution,
#         output_dir,
#         max_cloud_coverage=max_cloud,
#         mosaicking_order="leastCC"
#     )
    
#     # Analyze indices
#     st.info("📊 Analyzing vegetation indices...")
#     stats = manager.analyze_indices(output['ndvi_data'], output['ndmi_data'])
    
#     # Check thresholds and generate alerts
#     alert_system = VegetationAlertSystem(thresholds)
#     alerts = alert_system.check_thresholds(stats)
    
#     # Generate report
#     location_info = {
#         'address': formatted_address,
#         'latitude': lat,
#         'longitude': lon,
#         'buffer_km': buffer_km
#     }
    
#     report = alert_system.format_alert_report(location_info, stats, alerts)
    
#     return {
#         'location': location_info,
#         'statistics': stats,
#         'alerts': alerts,
#         'report': report,
#         'files': {
#             'ndvi': output['ndvi'],
#             'ndmi': output['ndmi']
#         }
#     }


# def process_era5_data(address, lat, lon, cds_key, start_date, end_date, thresholds):
#     """
#     Main processing function for ERA5-Land climate data
    
#     Returns:
#         dict: Results including climate data, alerts, and file paths
#     """
#     # Create output directory
#     output_dir = tempfile.mkdtemp(prefix='era5_')
    
#     # Initialize manager
#     manager = ERA5DataManager(cds_key)
    
#     # Fetch data
    
#     variables = [
#         '2m_temperature',
#         'total_precipitation',
#         'volumetric_soil_water_layer_1'
#     ]
    
#     try:
#         ds, netcdf_path = manager.fetch_era5_for_location(
#             lat, lon, start_date, end_date, variables, output_dir
#         )
        
#         st.success("✅ Data downloaded successfully")
        
#         # Process data and check alerts
#         st.info("📊 Analyzing climate data...")
#         data, alerts = manager.process_era5_for_alerts(
#             netcdf_path, lat, lon, thresholds, max_distance_km=50, verbose=True, start_date=start_date, end_date=end_date
#         )
        
#         # Generate report
#         location_info = {
#             'address': address,
#             'latitude': lat,
#             'longitude': lon
#         }
        
#         alert_system = ClimateAlertSystem(thresholds)
#         report = alert_system.format_climate_report(location_info, data, alerts)
        
#         return {
#             'location': location_info,
#             'data': data,
#             'alerts': alerts,
#             'report': report,
#             'netcdf_path': netcdf_path
#         }
        
#     except CoordinateError as e:
#         st.error(f"❌ Coordinate Error: {str(e)}")
#         return None
#     except Exception as e:
#         st.error(f"❌ Error processing ERA5 data: {str(e)}")
#         st.exception(e)
#         return None

# def display_results(results):
#     """Display analysis results in Streamlit UI"""
    
#     st.markdown("---")
#     st.markdown("## 📍 Location Information")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Address", results['location']['address'])
#         st.metric("Latitude", f"{results['location']['latitude']:.4f}°N")
#     with col2:
#         st.metric("Longitude", f"{results['location']['longitude']:.4f}°W")
#         st.metric("Coverage Area", f"{results['location']['buffer_km']} km radius")
    
#     st.markdown("---")
#     st.markdown("## 🌱 Vegetation Indices")
    
#     stats = results['statistics']
    
#     # NDVI metrics
#     st.markdown("### NDVI (Normalized Difference Vegetation Index)")
#     st.caption("Measures vegetation health and density (-1.0 to +1.0)")
    
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Average", f"{stats['ndvi_mean']:.3f}")
#     with col2:
#         st.metric("Minimum", f"{stats['ndvi_min']:.3f}")
#     with col3:
#         st.metric("Maximum", f"{stats['ndvi_max']:.3f}")
#     with col4:
#         st.metric("Std Dev", f"{stats['ndvi_std']:.3f}")
    
#     # NDVI interpretation
#     with st.expander("📖 NDVI Interpretation Guide"):
#         st.markdown("""
#         - **< 0.2**: Bare soil, rocks, water
#         - **0.2 - 0.3**: Sparse vegetation
#         - **0.3 - 0.6**: Moderate vegetation ✓
#         - **0.6 - 0.8**: Dense vegetation ✓✓
#         - **> 0.8**: Very dense vegetation
#         """)
    
#     st.markdown("---")
    
#     # NDMI metrics
#     st.markdown("### NDMI (Normalized Difference Moisture Index)")
#     st.caption("Measures plant water content and stress (-1.0 to +1.0)")
    
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Average", f"{stats['ndmi_mean']:.3f}")
#     with col2:
#         st.metric("Minimum", f"{stats['ndmi_min']:.3f}")
#     with col3:
#         st.metric("Maximum", f"{stats['ndmi_max']:.3f}")
#     with col4:
#         st.metric("Std Dev", f"{stats['ndmi_std']:.3f}")
    
#     # NDMI interpretation
#     with st.expander("📖 NDMI Interpretation Guide"):
#         st.markdown("""
#         - **< 0.0**: Severe drought stress ⚠️
#         - **0.0 - 0.2**: Moderate stress
#         - **0.2 - 0.4**: Adequate moisture ✓
#         - **> 0.4**: High moisture content
#         """)
    
#     st.markdown("---")
    
#     # Data Quality
#     st.markdown("## 📊 Data Quality")
#     valid_pixels = stats['valid_pixels']
#     total_pixels = stats['total_pixels']
#     valid_pct = (valid_pixels / total_pixels * 100) if total_pixels else 0.0
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Cloud Coverage", f"{stats['cloud_percentage']:.1f}%")
#     with col2:
#         st.metric("Valid Pixels", f"{valid_pixels:,} / {total_pixels:,} ({valid_pct:.1f}%)")
    
#     st.markdown("---")
    
#     # Alerts
#     st.markdown("## ⚠️ Alert Status")
    
#     if results['alerts']:
#         st.warning(f"**{len(results['alerts'])} Alert(s) Detected**")
        
#         for i, alert in enumerate(results['alerts'], 1):
#             severity_class = f"alert-{alert['severity'].lower()}"
            
#             st.markdown(f"""
#             <div class="{severity_class}" style="color: #333;">
#                 <h4 style="color: #000; margin: 0 0 0.5rem 0;">[{i}] [{alert['severity']}] {alert['type']}</h4>
#                 <p style="color: #333; margin: 0.25rem 0;"><strong>Message:</strong> {alert['message']}</p>
#                 <p style="color: #333; margin: 0.25rem 0;"><strong>Value:</strong> {alert['value']:.3f} | <strong>Threshold:</strong> {alert['threshold']:.3f}</p>
#                 <p style="color: #555; margin: 0.25rem 0;">💡 <em>{alert.get('recommendation', 'Monitor the situation closely.')}</em></p>
#             </div>
#             """, unsafe_allow_html=True)
#     else:
#         st.success("✅ **No Alerts** - All parameters within normal range")
    
#     st.markdown("---")
    
#     # Visualizations
#     st.markdown("## 🗺️ Satellite Imagery Analysis")
    
#     st.info("Generating detailed NDVI/NDMI analysis panels...")
    
#     try:
#         img_buffer = visualize_tiff_files(
#             results['files']['ndvi'],
#             results['files']['ndmi']
#         )
#         st.image(img_buffer)
#     except Exception as e:
#         st.error(f"Error generating visualizations: {str(e)}")
    
#     st.markdown("---")
    
#     # Download report
#     st.markdown("## 📄 Download Report")
    
#     st.download_button(
#         label="📥 Download Full Report",
#         data=results['report'],
#         file_name=f"allsat_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
#         mime="text/plain"
#     )


# def display_era5_results(results):
#     """Display ERA5 climate analysis results in Streamlit UI"""
    
#     st.markdown("---")
#     st.markdown("## 📍 Location Information")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Address", results['location']['address'])
#         st.metric("Latitude", f"{results['location']['latitude']:.4f}°N")
#     with col2:
#         st.metric("Longitude", f"{results['location']['longitude']:.4f}°W")
#         st.metric("Data Distance (Grid Mismatch)", f"{results['data']['distance_km']:.1f} km")
    
#     st.markdown("---")
#     st.markdown("## 🌡️ Climate Metrics")
    
#     data = results['data']
    
#     # Temperature metrics
#     st.markdown("### Temperature (2m above ground)")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Average", f"{data['avg_temp_c']:.1f}°C ({data['avg_temp_f']:.1f}°F)")
#     with col2:
#         st.metric("Maximum", f"{data['max_temp_c']:.1f}°C ({data['max_temp_f']:.1f}°F)")
#     with col3:
#         st.metric("Minimum", f"{data['min_temp_c']:.1f}°C ({data['min_temp_f']:.1f}°F)")
    
#     st.markdown("---")
    
#     # Precipitation metrics
#     st.markdown("### Precipitation")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Total", f"{data['total_precip_mm']:.1f} mm")
#     with col2:
#         st.metric("Average/Month", f"{data['avg_precip_mm']:.1f} mm")
#     with col3:
#         st.metric("Max in Month", f"{data['max_precip_mm']:.1f} mm")
    
#     st.markdown("---")
    
#     # Soil moisture metrics
#     st.markdown("### Soil Moisture (Top Layer)")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Average", f"{data['avg_soil_moisture']:.3f} m³/m³")
#     with col2:
#         st.metric("Minimum", f"{data['min_soil_moisture']:.3f} m³/m³")
#     with col3:
#         st.metric("Maximum", f"{data['max_soil_moisture']:.3f} m³/m³")
    
#     st.markdown("---")
    
#     # Time period
#     st.markdown("## 📅 Time Period")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Start Date", data['period_start'].strftime('%Y-%m-%d'))
#     with col2:
#         st.metric("End Date", data['period_end'].strftime('%Y-%m-%d'))
#     with col3:
#         st.metric("Duration", f"{data['n_months']} months")
    
#     st.markdown("---")
    
#     # Alerts
#     st.markdown("## ⚠️ Alert Status")
    
#     if results['alerts']:
#         st.warning(f"**{len(results['alerts'])} Alert(s) Detected**")
        
#         for i, alert in enumerate(results['alerts'], 1):
#             severity_class = f"alert-{alert['severity'].lower()}"
            
#             st.markdown(f"""
#             <div class="{severity_class}">
#                 <h4 style="color: #000; margin: 0 0 0.5rem 0;">[{i}] [{alert['severity']}] {alert['type']}</h4>
#                 <p style="color: #333; margin: 0.25rem 0;"><strong>Message:</strong> {alert['message']}</p>
#                 <p style="color: #333; margin: 0.25rem 0;"><strong>Value:</strong> {alert['value']:.3f} | <strong>Threshold:</strong> {alert['threshold']:.3f}</p>
#                 <p style="color: #555; margin: 0.25rem 0;">💡 <em>{alert.get('recommendation', 'Monitor the situation closely.')}</em></p>
#             </div>
#             """, unsafe_allow_html=True)
#     else:
#         st.success("✅ **No Alerts** - All parameters within normal range")
    
#     st.markdown("---")
    
#     # Visualizations
#     st.markdown("## 📈 Climate Data Trends")
    
#     st.info("Generating time series visualizations...")
    
#     try:
#         img_buffer = visualize_climate_data(data)
#         st.image(img_buffer)
#     except Exception as e:
#         st.error(f"Error generating visualizations: {str(e)}")
    
#     st.markdown("---")
    
#     # Download report
#     st.markdown("## 📄 Download Report")
    
#     st.download_button(
#         label="📥 Download Climate Report",
#         data=results['report'],
#         file_name=f"allsat_climate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
#         mime="text/plain"
#     )

# def display_risk_assessment(risk_results: dict, address: str, lat: float, lon: float):
#     """
#     Display combined risk assessment with scores and visualization
#     """
#     from risk_integration import interpret_risk_level, get_risk_alert_level, format_risk_report
    
#     stats = risk_results['statistics']
    
#     # Get risk levels
#     drought_mean = stats['drought']['mean']
#     wildfire_mean = stats['wildfire']['mean']
    
#     drought_label, drought_color, drought_rec = interpret_risk_level(drought_mean)
#     wildfire_label, wildfire_color, wildfire_rec = interpret_risk_level(wildfire_mean)
    
#     alert_info = get_risk_alert_level(stats)
    
#     # Display Alert Banner
#     if alert_info['severity'] == "CRITICAL":
#         st.error(f"{alert_info['icon']} **{alert_info['severity']}**: {alert_info['message']}")
#     elif alert_info['severity'] == "WARNING":
#         st.warning(f"{alert_info['icon']} **{alert_info['severity']}**: {alert_info['message']}")
#     else:
#         st.success(f"{alert_info['icon']} **{alert_info['severity']}**: {alert_info['message']}")
    
#     # Risk Scores Side by Side
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("### 🌾 Drought Risk")
#         st.markdown(f"""
#         <div style="
#             background-color: {drought_color}22; 
#             border-left: 5px solid {drought_color}; 
#             padding: 20px; 
#             border-radius: 5px;
#             margin: 10px 0;
#         ">
#             <h1 style="color: {drought_color}; margin: 0;">{drought_mean:.2f}</h1>
#             <h3 style="margin: 5px 0;">{drought_label} Risk</h3>
#             <p style="margin: 5px 0; font-size: 0.9em;">{drought_rec}</p>
#             <hr style="border-color: {drought_color};">
#             <p style="margin: 5px 0;"><strong>Range:</strong> {stats['drought']['min']:.2f} - {stats['drought']['max']:.2f}</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Distribution
#         with st.expander("📊 Risk Distribution"):
#             for class_name, class_stats in stats['drought']['classes'].items():
#                 st.metric(
#                     label=class_name,
#                     value=f"{class_stats['percentage']:.1f}%",
#                     delta=f"{class_stats['count']:,} pixels"
#                 )
    
#     with col2:
#         st.markdown("### 🔥 Wildfire Risk")
#         st.markdown(f"""
#         <div style="
#             background-color: {wildfire_color}22; 
#             border-left: 5px solid {wildfire_color}; 
#             padding: 20px; 
#             border-radius: 5px;
#             margin: 10px 0;
#         ">
#             <h1 style="color: {wildfire_color}; margin: 0;">{wildfire_mean:.2f}</h1>
#             <h3 style="margin: 5px 0;">{wildfire_label} Risk</h3>
#             <p style="margin: 5px 0; font-size: 0.9em;">{wildfire_rec}</p>
#             <hr style="border-color: {wildfire_color};">
#             <p style="margin: 5px 0;"><strong>Range:</strong> {stats['wildfire']['min']:.2f} - {stats['wildfire']['max']:.2f}</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Distribution
#         with st.expander("📊 Risk Distribution"):
#             for class_name, class_stats in stats['wildfire']['classes'].items():
#                 st.metric(
#                     label=class_name,
#                     value=f"{class_stats['percentage']:.1f}%",
#                     delta=f"{class_stats['count']:,} pixels"
#                 )
    
#     # Visualization
#     st.markdown("### 🗺️ Risk Maps")
#     st.image(
#         risk_results['visualization'],
#         caption=f"Combined Risk Assessment - {address}",
#         width='stretch'
#     )
    
#     # Download Report
#     st.markdown("### 📥 Download Full Report")
    
#     location_info = {
#         'address': address,
#         'latitude': lat,
#         'longitude': lon
#     }
    
#     report_text = format_risk_report(risk_results, location_info)
    
#     st.download_button(
#         label="📄 Download Risk Assessment Report",
#         data=report_text,
#         file_name=f"risk_assessment_{address.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
#         mime="text/plain"
#     )

# def main():
#     """Main Streamlit application"""
    
#     # Header
#     st.markdown('<div class="main-header">🛰️ ALLSAT AI - EWIS</div>', unsafe_allow_html=True)
#     st.markdown('<div class="sub-header">Environmental Warning & Intelligence System</div>', unsafe_allow_html=True)
#     st.caption("Sentinel-2 Vegetation Monitoring + ERA5-Land Climate Analysis")
    
#     # Sidebar - Configuration
#     st.sidebar.header("⚙️ Configuration")
    
#     # Analysis type selection
#     st.sidebar.subheader("Analysis Type")
#     analysis_type = st.sidebar.radio(
#         "Select Analysis",
#         ["Vegetation (Sentinel-2)", "Climate (ERA5-Land)", "Both"],
#         help="Choose which analysis to run"
#     )
    
#     st.sidebar.markdown("---")
    
#     # API Credentials
#     st.sidebar.subheader("API Credentials")
    
#     show_sentinel = analysis_type in ["Vegetation (Sentinel-2)", "Both"]
#     show_era5 = analysis_type in ["Climate (ERA5-Land)", "Both"]
    
#     if show_sentinel:
#         client_id = st.sidebar.text_input(
#             "Copernicus Client ID",
#             value=st.secrets.get("COPERNICUS_CLIENT_ID", ""),
#             type="password",
#             help="Sentinel Hub OAuth Client ID"
#         )
#         client_secret = st.sidebar.text_input(
#             "Copernicus Client Secret",
#             value=st.secrets.get("COPERNICUS_CLIENT_SECRET", ""),
#             type="password",
#             help="Sentinel Hub OAuth Client Secret"
#         )
#     else:
#         client_id, client_secret = None, None
    
#     if show_era5:
#         cds_key = st.sidebar.text_input(
#             "CDS API Key",
#             value=st.secrets.get("CDS_API_KEY", ""),
#             type="password",
#             help="Format: UID:API-key from https://cds.climate.copernicus.eu/"
#         )
#     else:
#         cds_key = None
    
#     st.sidebar.markdown("---")
    
#     # Sentinel-2 Parameters
#     if show_sentinel:
#         st.sidebar.subheader("Sentinel-2 Parameters")
        
#         buffer_km = st.sidebar.slider(
#             "Coverage Radius (km)",
#             min_value=10,
#             max_value=50,
#             value=30,
#             step=5,
#             help="Radius around the location"
#         )
        
#         days_back = st.sidebar.slider(
#             "Days to Look Back",
#             min_value=7,
#             max_value=30,
#             value=25,
#             step=1,
#             help="Search window for imagery"
#         )
        
#         max_cloud = st.sidebar.slider(
#             "Max Cloud Coverage (%)",
#             min_value=10,
#             max_value=50,
#             value=30,
#             step=5
#         )
        
#         resolution = st.sidebar.select_slider(
#             "Resolution (m/pixel)",
#             options=[10, 20, 40, 60],
#             value=40,
#             help="Lower = more detail, slower"
#         )
        
#         st.sidebar.markdown("**Vegetation Alert Thresholds**")
#         ndvi_min = st.sidebar.number_input("NDVI Min", 0.0, 1.0, 0.3, 0.05)
#         ndmi_min = st.sidebar.number_input("NDMI Min", -1.0, 1.0, 0.1, 0.05)
#         cloud_max_threshold = st.sidebar.number_input("Cloud Alert (%)", 10, 100, 30, 5)
        
#         sentinel_thresholds = {
#             'ndvi_min': ndvi_min,
#             'ndmi_min': ndmi_min,
#             'cloud_max': cloud_max_threshold
#         }
#         st.sidebar.markdown("---")
#     else:
#         buffer_km, days_back, max_cloud, resolution, sentinel_thresholds = None, None, None, None, None
    
#     # ERA5 Parameters
#     if show_era5:
#         st.sidebar.subheader("ERA5-Land Parameters")
        
#         start_date = st.sidebar.date_input(
#             "Start Date",
#             value=datetime(2025, 10, 1),
#             help="Beginning of climate data period"
#         )
#         end_date = st.sidebar.date_input(
#             "End Date",
#             value=datetime.now(),
#             help="End of climate data period"
#         )
        
#         st.sidebar.markdown("**Climate Alert Thresholds**")
#         max_temp_c = st.sidebar.number_input("Max Temp (°C)", -50, 60, 35, 1)
#         min_temp_c = st.sidebar.number_input("Min Temp (°C)", -50, 60, -10, 1)
#         total_precip_mm = st.sidebar.number_input("Max Precip (mm)", 0, 2000, 500, 50)
#         min_precip_mm = st.sidebar.number_input("Min Precip (mm)", 0, 1000, 100, 10)
#         min_soil_moisture = st.sidebar.number_input("Min Soil Moisture", 0.0, 1.0, 0.20, 0.05)
        
#         era5_thresholds = {
#             'max_temp_c': max_temp_c,
#             'min_temp_c': min_temp_c,
#             'total_precip_mm': total_precip_mm,
#             'min_precip_mm': min_precip_mm,
#             'min_soil_moisture': min_soil_moisture
#         }
#     else:
#         start_date, end_date, era5_thresholds = None, None, None
    
#     # Main interface
#     st.markdown("### 📍 Enter Location")
    
#     col1, col2 = st.columns([3, 1])
    
#     with col1:
#         address = st.text_input(
#             "Location Address",
#             placeholder="e.g., Prineville, Oregon or 123 Farm Road, Iowa City, IA",
#             help="Enter any address or location name"
#         )
    
#     with col2:
#         st.write("")
#         st.write("")
#         analyze_button = st.button("🚀 Analyze", type="primary", width='stretch')
    
#     # Process analysis
#     if analyze_button:
#         if not address:
#             st.error("❌ Please enter a location address")
#             return
        
#         # Validate credentials
#         if show_sentinel and (not client_id or not client_secret):
#             st.error("❌ Please provide Copernicus OAuth credentials in the sidebar")
#             return
        
#         if show_era5 and not cds_key:
#             st.error("❌ Please provide CDS API Key in the sidebar")
#             return
        
#         # Store results
#         sentinel_results = None
#         era5_results = None
#         geocoded_lat, geocoded_lon = None, None
        
#         # Process Sentinel-2
#         if show_sentinel:
#             with st.spinner("🛰️ Processing Sentinel-2 data... (2-5 minutes)"):
#                 try:
#                     sentinel_results = process_sentinel_data(
#                         address=address,
#                         client_id=client_id,
#                         client_secret=client_secret,
#                         buffer_km=buffer_km,
#                         days_back=days_back,
#                         max_cloud=max_cloud,
#                         resolution=resolution,
#                         thresholds=sentinel_thresholds
#                     )
                    
#                     if sentinel_results:
#                         st.success("✅ Sentinel-2 analysis complete!")
#                         geocoded_lat = sentinel_results['location']['latitude']
#                         geocoded_lon = sentinel_results['location']['longitude']
                        
#                 except Exception as e:
#                     st.error(f"❌ Sentinel-2 Error: {str(e)}")
#                     st.exception(e)
        
#         # Process ERA5
#         if show_era5:
#             # Get coordinates (from Sentinel if available, otherwise geocode)
#             if geocoded_lat is None:
#                 st.info("🗺️ Geocoding address.")
#             try:
#                 geocoded_lat, geocoded_lon, _ = geocode_with_fallback(address, timeout=10, arcgis_api_key=arcgis_api_key, locationiq_api_key=locationiq_api_key)
#             except Exception as e:
#                 st.error(f"❌ Geocoding error: {str(e)}")
#                 return
            
#             st.info(f"📡 Fetching ERA5-Land climate data from CDS API...")
#             with st.spinner("🌡️ Processing ERA5-Land data... (2-3 minutes)"):
#                 try:
#                     era5_results = process_era5_data(
#                         address=address,
#                         lat=geocoded_lat,
#                         lon=geocoded_lon,
#                         cds_key=cds_key,
#                         start_date=start_date.strftime('%Y-%m-%d'),
#                         end_date=end_date.strftime('%Y-%m-%d'),
#                         thresholds=era5_thresholds
#                     )
                    
#                     if era5_results:
#                         st.success("✅ ERA5-Land analysis complete!")
                        
#                 except Exception as e:
#                     st.error(f"❌ ERA5 Error: {str(e)}")
#                     st.exception(e)
        
#         # Display results
#         if sentinel_results or era5_results:
#             st.markdown("---")
#             st.markdown("## 📊 Analysis Results")
            
#             if sentinel_results and era5_results:
#                 # Show both in tabs
#                 tab1, tab2, tab3 = st.tabs(["Vegetation (Sentinel-2)", "Climate (ERA5-Land)", "Combined Drought and Wildfire Risk Scores"])
                
#                 with tab1:
#                     display_results(sentinel_results)
                
#                 with tab2:
#                     display_era5_results(era5_results)
                
#                 with tab3:
#                     st.markdown(" Combined Risk Assessment")
    
#                     with st.spinner(" Generating drought and wildfire risk maps..."):
#                         try:
#                             risk_results = generate_risk_maps_from_data(
#                                 ndvi_path=sentinel_results['files']['ndvi'],
#                                 ndmi_path=sentinel_results['files']['ndmi'],
#                                 era5_nc_path=era5_results['netcdf_path'],
#                                 output_dir=None,  # Uses temp directory
#                                 thresholds=None,  # or pass your thresholds dict if you have one
#                                 lat=geocoded_lat,
#                                 lon=geocoded_lon,
#                             )
                            
#                             # Display Risk Scores
#                             display_risk_assessment(risk_results, address, geocoded_lat, geocoded_lon)
                            
#                         except Exception as e:
#                             st.error(f"❌ Risk Assessment Error: {str(e)}")
#                             st.exception(e)

                
            
#             elif sentinel_results:
#                 display_results(sentinel_results)
            
#             elif era5_results:
#                 display_era5_results(era5_results)
    
#     # Footer
#     st.markdown("---")
#     st.markdown("""
#     <div style="text-align: center; color: #666; padding: 1rem;">
#         <p><strong>ALLSAT AI - Environmental Warning & Intelligence System</strong></p>
#         <p>Sentinel-2 Satellite Imagery + ERA5-Land Climate Data from Copernicus</p>
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()



# -----------------------------------------------------------------------------
# APP STATIC - DEMO
"""
ALLSAT AI - Environmental Warning & Intelligence System (EWIS)
Static Demo Version - Prineville, Oregon
"""

import streamlit as st
from pathlib import Path
import sys

# Add demo folder to Python path so we can import demo_data
sys.path.insert(0, str(Path(__file__).parent / "demo"))

from demo_data import (
    get_location_info,
    get_vegetation_data,
    get_climate_data,
    get_risk_data,
    get_all_location_keys,
    get_location_display_name
)

# Page config
st.set_page_config(
    page_title="ALLSAT AI - EWIS Demo",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (copied from original app.py)
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-critical {
        background-color: #ffebee;
        padding: 1rem;
        border-left: 4px solid #f44336;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #fff3e0;
        padding: 1rem;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #e3f2fd;
        padding: 1rem;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
    }
    .location-button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

def display_vegetation_analysis(location_key):
    """Display vegetation analysis results"""
    veg_data = get_vegetation_data(location_key)
    loc_info = get_location_info(location_key)
    
    if not veg_data or not loc_info:
        st.error(f"❌ No vegetation data available for {location_key}")
        return
    
    # Location Information
    st.markdown("## 📍 Location Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Address", loc_info['display_name'])
        st.metric("Latitude", f"{loc_info['coordinates']['lat']:.4f}°N")
    with col2:
        st.metric("Longitude", f"{loc_info['coordinates']['lon']:.4f}°W")
        st.metric("Coverage Area", "30 km radius")
    
    st.markdown("---")
    
    # Vegetation Indices
    st.markdown("## 🌱 Vegetation Indices")
    
    metrics = veg_data['metrics']
    
    # NDVI metrics
    st.markdown("### NDVI (Normalized Difference Vegetation Index)")
    st.caption("Measures vegetation health and density (-1.0 to +1.0)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average", f"{metrics['ndvi']['mean']:.3f}")
    with col2:
        st.metric("Minimum", f"{metrics['ndvi']['min']:.3f}")
    with col3:
        st.metric("Maximum", f"{metrics['ndvi']['max']:.3f}")
    with col4:
        st.metric("Std Dev", f"{metrics['ndvi']['std']:.3f}")
    
    # NDVI interpretation
    with st.expander("📖 NDVI Interpretation Guide"):
        st.markdown("""
        - **< 0.2**: Bare soil, rocks, water
        - **0.2 - 0.3**: Sparse vegetation
        - **0.3 - 0.6**: Moderate vegetation ✓
        - **0.6 - 0.8**: Dense vegetation ✓✓
        - **> 0.8**: Very dense vegetation
        """)
    
    st.markdown("---")
    
    # NDMI metrics
    st.markdown("### NDMI (Normalized Difference Moisture Index)")
    st.caption("Measures plant water content and stress (-1.0 to +1.0)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average", f"{metrics['ndmi']['mean']:.3f}")
    with col2:
        st.metric("Minimum", f"{metrics['ndmi']['min']:.3f}")
    with col3:
        st.metric("Maximum", f"{metrics['ndmi']['max']:.3f}")
    with col4:
        st.metric("Std Dev", f"{metrics['ndmi']['std']:.3f}")
    
    # NDMI interpretation
    with st.expander("📖 NDMI Interpretation Guide"):
        st.markdown("""
        - **< 0.0**: Severe drought stress ⚠️
        - **0.0 - 0.2**: Moderate stress
        - **0.2 - 0.4**: Adequate moisture ✓
        - **> 0.4**: High moisture content
        """)
    
    st.markdown("---")
    
    # Data Quality
    st.markdown("## 📊 Data Quality")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cloud Coverage", f"{veg_data['cloud_coverage']:.1f}%")
    with col2:
        valid_pixels = metrics['valid_pixels']
        total_pixels = metrics['total_pixels']
        coverage_pct = metrics['ndvi']['coverage_percent']
        st.metric("Valid Pixels", f"{valid_pixels:,} / {total_pixels:,} ({coverage_pct:.1f}%)")
    
    st.markdown("---")
    
    # Alerts
    st.markdown("## ⚠️ Alert Status")
    
    if veg_data['alerts']:
        st.warning(f"**{len(veg_data['alerts'])} Alert(s) Detected**")
        
        for i, alert in enumerate(veg_data['alerts'], 1):
            severity_class = f"alert-{alert['severity'].lower()}"
            
            st.markdown(f"""
            <div class="{severity_class}" style="color: #333;">
                <h4 style="color: #000; margin: 0 0 0.5rem 0;">[{i}] [{alert['severity']}] {alert['title']}</h4>
                <p style="color: #333; margin: 0.25rem 0;"><strong>Message:</strong> {alert['message']}</p>
                <p style="color: #555; margin: 0.25rem 0;">💡 <em>{alert['recommendation']}</em></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ **No Alerts** - All parameters within normal range")
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("## 🗺️ Satellite Imagery Analysis")
    
    # Display the sentinel image
    if 'sentinel' in veg_data['images']:
        # Use Path to handle both Windows and Unix separators
        image_path = Path("demo") / veg_data['images']['sentinel']
        
        if image_path.exists():
            st.image(str(image_path), width='stretch')
        else:
            # Try alternate location at root level
            alt_path = Path(veg_data['images']['sentinel'])
            if alt_path.exists():
                st.image(str(alt_path), width='stretch')
            else:
                st.warning(f"⚠️ Visualization not available. Looking for: {image_path} or {alt_path}")
    else:
        st.warning("⚠️ No vegetation imagery available for this location")

def display_climate_analysis(location_key):
    """Display climate analysis results"""
    climate_data = get_climate_data(location_key)
    loc_info = get_location_info(location_key)
    
    if not climate_data or not loc_info:
        st.error(f"❌ No climate data available for {location_key}")
        return
    
    # Location Information
    st.markdown("## 📍 Location Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Address", loc_info['display_name'])
        st.metric("Latitude", f"{loc_info['coordinates']['lat']:.4f}°N")
    with col2:
        st.metric("Longitude", f"{loc_info['coordinates']['lon']:.4f}°W")
        st.metric("Data Distance (Grid Mismatch)", "0.1 km")
    
    st.markdown("---")
    
    # Climate Metrics
    st.markdown("## 🌡️ Climate Metrics")
    
    metrics = climate_data['metrics']
    
    # Temperature
    st.markdown("### Temperature (2m above ground)")
    col1, col2, col3 = st.columns(3)
    with col1:
        temp_c = metrics['temperature']['mean_c']
        temp_f = temp_c * 9/5 + 32
        st.metric("Average", f"{temp_c:.1f}°C ({temp_f:.1f}°F)")
    with col2:
        max_c = metrics['temperature']['max_c']
        max_f = max_c * 9/5 + 32
        st.metric("Maximum", f"{max_c:.1f}°C ({max_f:.1f}°F)")
    with col3:
        min_c = metrics['temperature']['min_c']
        min_f = min_c * 9/5 + 32
        st.metric("Minimum", f"{min_c:.1f}°C ({min_f:.1f}°F)")
    
    st.markdown("---")
    
    # Precipitation
    st.markdown("### Precipitation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total", f"{metrics['precipitation']['total_mm']:.1f} mm")
    with col2:
        st.metric("Average/Month", f"{metrics['precipitation']['mean_daily_mm']:.1f} mm")
    with col3:
        st.metric("Max in Month", f"{metrics['precipitation']['max_daily_mm']:.1f} mm")
    
    st.markdown("---")
    
    # Soil Moisture
    st.markdown("### Soil Moisture (Top Layer)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average", f"{metrics['soil_moisture']['mean']:.3f} m³/m³")
    with col2:
        st.metric("Minimum", f"{metrics['soil_moisture']['min']:.3f} m³/m³")
    with col3:
        st.metric("Maximum", f"{metrics['soil_moisture']['max']:.3f} m³/m³")
    
    st.markdown("---")
    
    # Time Period
    st.markdown("## 📅 Time Period")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Start Date", climate_data['date_range']['start'])
    with col2:
        st.metric("End Date", climate_data['date_range']['end'])
    with col3:
        st.metric("Duration", "1 month")
    
    st.markdown("---")
    
    # Alerts
    st.markdown("## ⚠️ Alert Status")
    
    if climate_data['alerts']:
        st.warning(f"**{len(climate_data['alerts'])} Alert(s) Detected**")
        
        for i, alert in enumerate(climate_data['alerts'], 1):
            severity_class = f"alert-{alert['severity'].lower()}"
            
            st.markdown(f"""
            <div class="{severity_class}" style="color: #333;">
                <h4 style="color: #000; margin: 0 0 0.5rem 0;">[{i}] [{alert['severity']}] {alert['title']}</h4>
                <p style="color: #333; margin: 0.25rem 0;"><strong>Message:</strong> {alert['message']}</p>
                <p style="color: #555; margin: 0.25rem 0;">💡 <em>{alert['recommendation']}</em></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ **No Alerts** - All parameters within normal range")
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("## 📈 Climate Analysis Charts")
    
    # Display the combined analysis image
    if 'combined_analysis' in climate_data['images']:
        image_path = Path("demo") / climate_data['images']['combined_analysis']
        if image_path.exists():
            st.image(str(image_path), width='stretch')
        else:
            alt_path = Path(climate_data['images']['combined_analysis'])
            if alt_path.exists():
                st.image(str(alt_path), width='stretch')
            else:
                st.warning(f"⚠️ Visualization not available. Looking for: {image_path} or {alt_path}")
    elif 'climate' in climate_data['images']:
        image_path = Path("demo") / climate_data['images']['climate']
        if image_path.exists():
            st.image(str(image_path), width='stretch')
        else:
            alt_path = Path(climate_data['images']['climate'])
            if alt_path.exists():
                st.image(str(alt_path), width='stretch')
            else:
                st.warning(f"⚠️ Visualization not available. Looking for: {image_path} or {alt_path}")
    else:
        st.warning("⚠️ No climate visualizations available for this location")

def display_risk_assessment(location_key):
    """Display combined risk assessment"""
    risk_data = get_risk_data(location_key)
    loc_info = get_location_info(location_key)
    
    if not risk_data or not loc_info:
        st.error(f"❌ No risk data available for {location_key}")
        return
    
    # Overall Alert Banner
    alert = risk_data['overall_alert']
    if alert['severity'] == 'CRITICAL':
        st.error(f"{alert['icon']} **{alert['severity']}**: {alert['message']}")
    elif alert['severity'] == 'WARNING':
        st.warning(f"{alert['icon']} **{alert['severity']}**: {alert['message']}")
    else:
        st.success(f"{alert['icon']} **{alert['severity']}**: {alert['message']}")
    
    st.markdown("---")
    
    # Risk Scores Side by Side
    col1, col2 = st.columns(2)
    
    # Drought Risk
    with col1:
        drought = risk_data['drought_risk']
        st.markdown(f"## 🌾 Drought Risk")
        
        st.markdown(f"""
        <div style="background-color: {drought['risk_color']}22; padding: 2rem; border-radius: 0.5rem; border-left: 4px solid {drought['risk_color']};">
            <h1 style="margin: 0; color: #000;">{drought['mean_score']:.2f}</h1>
            <h3 style="margin: 0.5rem 0; color: #000;">{drought['risk_level']} Risk</h3>
            <p style="color: #FFF; margin: 1rem 0;">{drought['recommendation']}</p>
            <hr style="border-color: {drought['risk_color']};">
            <p style="color: #FFF; margin: 0.5rem 0;"><strong>Range:</strong> {drought['min_score']:.2f} - {drought['max_score']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk Distribution
        with st.expander("📊 Risk Distribution"):
            dist = drought['class_distribution']
            pixels = drought['class_pixels']
            st.markdown(f"""
            **Low (0.00-0.33)**  
            {dist['low']:.1f}%  
            ↑ {pixels['low']:,} pixels
            
            **Moderate (0.33-0.66)**  
            {dist['moderate']:.1f}%  
            ↑ {pixels['moderate']:,} pixels
            
            **High (0.66-1.00)**  
            {dist['high']:.1f}%  
            ↑ {pixels['high']:,} pixels
            """)
    
    # Wildfire Risk
    with col2:
        wildfire = risk_data['wildfire_risk']
        st.markdown(f"## 🔥 Wildfire Risk")
        
        st.markdown(f"""
        <div style="background-color: {wildfire['risk_color']}22; padding: 2rem; border-radius: 0.5rem; border-left: 4px solid {wildfire['risk_color']};">
            <h1 style="margin: 0; color: #000;">{wildfire['mean_score']:.2f}</h1>
            <h3 style="margin: 0.5rem 0; color: #000;">{wildfire['risk_level']} Risk</h3>
            <p style="color: #FFF; margin: 1rem 0;">{wildfire['recommendation']}</p>
            <hr style="border-color: {wildfire['risk_color']};">
            <p style="color: #FFF; margin: 0.5rem 0;"><strong>Range:</strong> {wildfire['min_score']:.2f} - {wildfire['max_score']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk Distribution
        with st.expander("📊 Risk Distribution"):
            dist = wildfire['class_distribution']
            pixels = wildfire['class_pixels']
            st.markdown(f"""
            **Low (0.00-0.33)**  
            {dist['low']:.1f}%  
            ↑ {pixels['low']:,} pixels
            
            **Moderate (0.33-0.66)**  
            {dist['moderate']:.1f}%  
            ↑ {pixels['moderate']:,} pixels
            
            **High (0.66-1.00)**  
            {dist['high']:.1f}%  
            ↑ {pixels['high']:,} pixels
            """)
    
    st.markdown("---")
    
    # Risk Maps Visualization
    st.markdown("## 🗺️ Risk Maps")
    
    if 'combined_visualization' in risk_data['images']:
        image_path = Path("demo") / risk_data['images']['combined_visualization']
        if image_path.exists():
            st.image(str(image_path), width='stretch')
        else:
            alt_path = Path(risk_data['images']['combined_visualization'])
            if alt_path.exists():
                st.image(str(alt_path), width='stretch')
            else:
                st.warning(f"⚠️ Visualization not available. Looking for: {image_path} or {alt_path}")
    elif 'risk' in risk_data['images']:
        image_path = Path("demo") / risk_data['images']['risk']
        if image_path.exists():
            st.image(str(image_path), width='stretch')
        else:
            alt_path = Path(risk_data['images']['risk'])
            if alt_path.exists():
                st.image(str(alt_path), width='stretch')
            else:
                st.warning(f"⚠️ Visualization not available. Looking for: {image_path} or {alt_path}")
    else:
        st.warning("⚠️ No risk visualizations available for this location")
    
    st.markdown("---")
    
    # Methodology
    with st.expander("📖 Risk Assessment Methodology"):
        st.markdown("""
        ### Drought Risk Formula
        35% Moisture Stress (NDMI) + 25% Vegetation Stress (NDVI) + 20% Precipitation Deficit + 20% Heat Stress
        
        ### Wildfire Risk Formula
        40% Moisture Stress (NDMI) + 20% Heat Stress + 20% Precipitation Deficit + 20% Critical Fuel Condition (NDMI < 0.10)
        
        ### Risk Scale
        - **0.00-0.33**: Low Risk (favorable conditions)
        - **0.33-0.66**: Moderate Risk (monitor closely)
        - **0.66-1.00**: High Risk (action required)
        
        ### Data Sources
        - Sentinel-2 L2A: Vegetation indices (NDVI, NDMI)
        - ERA5-Land: Temperature, precipitation, soil moisture
        """)

def main():
    # Header
    st.markdown('<h1 class="main-header">🛰️ ALLSAT AI - EWIS</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Environmental Warning & Intelligence System - Demo</p>', unsafe_allow_html=True)
    
    # Sidebar - Analysis Type Selection
    st.sidebar.markdown("## 📊 Analysis Settings")
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["vegetation", "climate", "both"],
        format_func=lambda x: {
            "vegetation": "Vegetation (Sentinel-2)",
            "climate": "Climate (ERA5-Land)",
            "both": "Combined Risk Assessment"
        }[x]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About This Demo")
    st.sidebar.info("""
    This demo showcases ALLSAT AI's environmental monitoring capabilities using pre-computed data for 5 locations.
    
    **Data Sources:**
    - Sentinel-2 L2A (Last 30 days)
    - ERA5-Land (Oct-Nov 2025)
    """)
    
    # Main content
    st.markdown("## 🗺️ Select Location")
    
    # Location buttons (horizontal)
    locations = ['prineville', 'bend', 'portland', 'Dallas', 'nashville']
    location_names = ['Prineville', 'Bend', 'Portland', 'Dallas', 'Nashville']
    
    cols = st.columns(5)
    selected_location = None
    
    for i, (loc_key, loc_name) in enumerate(zip(locations, location_names)):
        with cols[i]:
            if st.button(loc_name, key=f"btn_{loc_key}", width='stretch'):
                selected_location = loc_key
    
    # Display results based on selection
    if selected_location:
        st.markdown("---")
        st.markdown(f"## 📊 Analysis Results - {get_location_display_name(selected_location)}")
        
        if analysis_type == "vegetation":
            display_vegetation_analysis(selected_location)
        elif analysis_type == "climate":
            display_climate_analysis(selected_location)
        else:  # both
            display_risk_assessment(selected_location)
    else:
        # Empty state
        st.info("👆 **Please select a location above to view analysis results**")
        
        st.markdown("---")
        st.markdown("### Available Locations")
        
        for loc_key, loc_name in zip(locations, location_names):
            loc_info = get_location_info(loc_key)
            if loc_info:
                st.markdown(f"**{loc_name}**: {loc_info['coordinates']['lat']:.2f}°N, {loc_info['coordinates']['lon']:.2f}°W")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>ALLSAT AI - Environmental Warning & Intelligence System</strong></p>
        <p>Sentinel-2 Satellite Imagery + ERA5-Land Climate Data from Copernicus</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()