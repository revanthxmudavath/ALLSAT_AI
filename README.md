# ALLSAT AI - Environmental Warning & Intelligence System (EWIS)

A Streamlit prototype for comprehensive environmental monitoring using Sentinel-2 satellite imagery and ERA5-Land climate data.

## Features

🛰️ **Sentinel-2 Vegetation Analysis**
- Real-time satellite imagery from Copernicus Data Space Ecosystem
- Automated NDVI (vegetation health) and NDMI (moisture content) calculation
- Cloud masking and data quality analysis
- Visual .tif file rendering

🌡️ **ERA5-Land Climate Analysis**
- Historical climate data from Copernicus Climate Data Store
- Temperature, precipitation, and soil moisture tracking
- Time series visualization
- Multi-month trend analysis

📊 **Integrated Monitoring**
- Run analyses independently or together
- Automated alert system with configurable thresholds
- Downloadable analysis reports for both datasets
- Single address input for all analyses

## Quick Start

### Prerequisites

1. **Copernicus Data Space Account** (for Sentinel-2)
   - Register at: https://dataspace.copernicus.eu/
   - Create OAuth credentials in your account settings
   - Save your Client ID and Client Secret

2. **Climate Data Store Account** (for ERA5-Land)
   - Register at: https://cds.climate.copernicus.eu/
   - Get your API key from User Profile
   - Format: `UID:API-key`

### Local Development

1. **Clone and Install**
```bash
pip install -r requirements.txt
```

2. **Set Up Secrets**

Create `.streamlit/secrets.toml`:
```toml
COPERNICUS_CLIENT_ID = "your_client_id_here"
COPERNICUS_CLIENT_SECRET = "your_client_secret_here"
CDS_API_KEY = "UID:API-key"
```

3. **Run the App**
```bash
streamlit run app.py
```

## Deployment to Streamlit Cloud

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Select your repository and branch
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Configure Secrets in Streamlit Cloud**
   - In your app settings, go to "Secrets"
   - Add:
   ```toml
   COPERNICUS_CLIENT_ID = "your_client_id_here"
   COPERNICUS_CLIENT_SECRET = "your_client_secret_here"
   CDS_API_KEY = "UID:API-key"
   ```

## Usage

1. **Select Analysis Type**
   - Sentinel-2 only (vegetation)
   - ERA5-Land only (climate)
   - Both analyses

2. **Enter Location**
   - Address, city, or coordinates

3. **Configure Parameters** (sidebar):
   - **Sentinel-2**: Coverage radius, days back, cloud coverage, resolution
   - **ERA5-Land**: Date range (start/end dates)
   - **Alert Thresholds**: Set limits for both analyses

4. **Click "Analyze"**
   - Sentinel-2: 2-5 minutes
   - ERA5-Land: 5-15 minutes (CDS queue dependent)
   - Both: Run sequentially

5. **Review Results**
   - Statistics and metrics
   - Alerts and recommendations
   - Visualizations (satellite .tifs or climate time series)
   - Download reports

## Understanding the Metrics

### Sentinel-2 Vegetation Indices

#### NDVI (Normalized Difference Vegetation Index)
- **Range**: -1.0 to +1.0
- **< 0.2**: Bare soil, rocks, water
- **0.2-0.3**: Sparse vegetation
- **0.3-0.6**: Moderate vegetation ✓
- **0.6-0.8**: Dense vegetation ✓✓
- **> 0.8**: Very dense vegetation

#### NDMI (Normalized Difference Moisture Index)
- **Range**: -1.0 to +1.0
- **< 0.0**: Severe drought stress ⚠️
- **0.0-0.2**: Moderate stress
- **0.2-0.4**: Adequate moisture ✓
- **> 0.4**: High moisture content

### ERA5-Land Climate Metrics

#### Temperature (2m above ground)
- **Average**: Mean temperature over period
- **Maximum**: Highest monthly temperature
- **Minimum**: Lowest monthly temperature
- Values in both Celsius and Fahrenheit

#### Precipitation
- **Total**: Cumulative rainfall over period (mm)
- **Average**: Mean monthly precipitation
- **Maximum**: Highest single month rainfall

#### Soil Moisture (Top Layer)
- **Range**: 0.0 to 1.0 m³/m³
- **< 0.15**: Very dry soil
- **0.15-0.25**: Dry soil
- **0.25-0.35**: Adequate moisture ✓
- **> 0.35**: High moisture content

## Technical Details

### Architecture
- **Frontend**: Streamlit with tab-based UI
- **Satellite Data**: Sentinel-2 L2A (Copernicus Data Space)
- **Climate Data**: ERA5-Land Reanalysis (Copernicus Climate Data Store)
- **Geocoding**: Nominatim (OpenStreetMap)
- **Image Processing**: Rasterio, NumPy
- **Visualization**: Matplotlib

### Data Processing Pipelines

**Sentinel-2 Pipeline:**
1. Geocode address to lat/lon coordinates
2. Create bounding box with specified radius
3. Search Sentinel-2 catalog for recent imagery
4. Download B04 (Red), B08 (NIR), B11 (SWIR) bands
5. Calculate NDVI and NDMI indices
6. Apply cloud masking using Scene Classification Layer
7. Analyze statistics and check alert thresholds
8. Generate visualizations and report

**ERA5-Land Pipeline:**
1. Geocode address to coordinates
2. Define bounding box (0.5° buffer)
3. Request monthly data from CDS API
4. Download 2m temperature, precipitation, soil moisture
5. Extract time series for nearest grid point
6. Validate coordinates and distance
7. Calculate statistics and check thresholds
8. Generate time series plots and report

### Processing Time
- **Sentinel-2**: 2-5 minutes
- **ERA5-Land**: 5-15 minutes (depends on CDS queue)
- **Both**: 7-20 minutes total
- **Factors**: Resolution, coverage area, network speed, API queues

## File Structure

```
.
├── app.py                   # Main Streamlit application (both analyses)
├── sentinel_processor.py    # Sentinel-2 processing logic
├── era5_processor.py        # ERA5-Land processing logic
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── DEPLOYMENT.md           # Deployment guide
├── PROJECT_STRUCTURE.md    # Architecture documentation
├── .gitignore              # Git ignore rules
├── quickstart.sh           # Linux/Mac setup script
├── quickstart.bat          # Windows setup script
└── .streamlit/
    ├── config.toml         # Streamlit UI configuration
    └── secrets.toml.template  # API credentials template
```

## Current Capabilities

✅ Sentinel-2 vegetation monitoring (NDVI/NDMI)
✅ ERA5-Land climate analysis (temp/precip/soil moisture)
✅ Combined analysis mode
✅ Interactive visualizations (.tif rendering and time series plots)
✅ Downloadable reports for both analyses
✅ Configurable alert thresholds

## Future Enhancements

- [ ] Multi-temporal analysis (trend detection over months/years)
- [ ] Email/SMS alerts for threshold violations
- [ ] Historical data comparison and anomaly detection
- [ ] Batch processing for multiple locations
- [ ] Export to GeoTIFF/KML/Shapefile formats
- [ ] Integration with additional data sources (MODIS, Landsat)
- [ ] Advanced analytics (crop yield prediction, fire risk)
- [ ] Mobile-responsive improvements

## Troubleshooting

### Sentinel-2 Issues

**"No suitable images found"**
- Try increasing the number of days to look back
- Increase max cloud coverage threshold
- Try a different location

**"Geocoding failed"**
- Check internet connection
- Try a more specific address
- Use coordinates directly: "lat, lon" format

**Processing timeout**
- Reduce coverage radius
- Increase resolution (lower detail)
- Check Copernicus API status

### ERA5-Land Issues

**"CDS API queue timeout"**
- CDS can have long queues (5-15 minutes normal)
- Try during off-peak hours
- Check CDS service status: https://cds.climate.copernicus.eu/

**"Coordinate validation error"**
- ERA5 data has 0.1° resolution (~11km grid)
- Location will snap to nearest grid point
- Max 50km distance allowed

**"Invalid CDS API key"**
- Verify format: `UID:API-key` (not just the key)
- Check at: https://cds.climate.copernicus.eu/user
- Ensure account is activated

### General Issues

**"Missing dependencies"**
- Run: `pip install -r requirements.txt`
- For conda users: some packages may need conda-forge channel

**"Streamlit secrets not found"**
- Ensure `.streamlit/secrets.toml` exists
- Copy from `secrets.toml.template`
- For deployment, use Streamlit Cloud secrets

## Credits

- **Satellite Data**: ESA Copernicus Programme (Sentinel-2)
- **Climate Data**: ECMWF ERA5-Land Reanalysis
- **APIs**: 
  - Sentinel Hub / Copernicus Data Space Ecosystem
  - Copernicus Climate Data Store (CDS)
- **Geocoding**: OpenStreetMap Nominatim

## License

This is a prototype for demonstration purposes.

---

**Built by**: Revanth for ALLSAT AI  
**Purpose**: Environmental monitoring and early warning system prototype  
**Dataset Sources**: Copernicus Programme (Sentinel-2 + ERA5-Land)