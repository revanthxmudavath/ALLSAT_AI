from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
import time

import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds, array_bounds
import geopandas as gpd
from rasterio.features import geometry_mask

warnings.filterwarnings("ignore")


class RiskMapGenerator:
    def __init__(self, nodata_value: float = -9999.0):
        self.nodata = float(nodata_value)

    # -----------------------------
    # Public API
    # -----------------------------
    def generate_combined_risk_maps(
        self,
        ndvi_path: str,
        ndmi_path: str,
        era5_nc_path: str,
        output_dir: str = "risk_outputs",
        save_intermediate: bool = False,
        aoi_shapefile: Optional[str] = None,
        thresholds: Optional[dict] = None,
    ) -> Dict:
        thresholds = thresholds or {}

        # Hardcoded defaults (override via thresholds dict)
        ndvi_min = float(thresholds.get("ndvi_min", 0.30))
        ndmi_min = float(thresholds.get("ndmi_min", 0.10))
        min_precip_mm = float(thresholds.get("min_precip_mm", 100.0))
        min_temp_c = float(thresholds.get("min_temp_c", -10.0))
        max_temp_c = float(thresholds.get("max_temp_c", 35.0))
        fuel_ndmi_critical = float(thresholds.get("fuel_ndmi_critical", 0.10))

        # Recipe params
        ndvi_drop_offset = float(thresholds.get("ndvi_drop_offset", 0.05))
        ndvi_drop_scale = float(thresholds.get("ndvi_drop_scale", 0.15))
        ndmi_drop_offset = float(thresholds.get("ndmi_drop_offset", 0.04))
        ndmi_drop_scale = float(thresholds.get("ndmi_drop_scale", 0.12))

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("GENERATING COMBINED RISK MAPS")
        print("=" * 80 + "\n")

        # 1) NDVI master grid
        print("📊 Step 1: Loading NDVI (master grid)...")
        ndvi_data, ndvi_profile, ndvi_nodata = self._read_raster(ndvi_path)
        ndvi_data = ndvi_data.astype("float32")
        if ndvi_nodata is not None:
            ndvi_data[ndvi_data == ndvi_nodata] = np.nan

        inside_mask = np.ones(ndvi_data.shape, dtype=bool)
        if aoi_shapefile:
            print("🗺️  Applying AOI shapefile mask...")
            inside_mask = self._build_aoi_mask(aoi_shapefile, ndvi_profile)
            ndvi_data[~inside_mask] = np.nan

        print(f"   Grid size: {ndvi_profile['width']} x {ndvi_profile['height']} pixels")
        print(f"   CRS: {ndvi_profile['crs']}")

        # 2) NDMI aligned to NDVI
        print("\n📊 Step 2: Loading and aligning NDMI...")
        ndmi_data = self._align_raster_to_master(ndmi_path, ndvi_profile, resampling=Resampling.bilinear)
        ndmi_data[~inside_mask] = np.nan

        # 3) ERA5 aligned to NDVI
        print("\n🌡️ Step 3: Processing ERA5-Land climate data...")
        climate = self._process_era5_to_grid(era5_nc_path, ndvi_profile)

        temperature = climate["temperature"]
        precipitation = climate["precipitation"]
        soil_moisture = climate["soil_moisture"]

        # apply AOI mask
        temperature[~inside_mask] = np.nan
        precipitation[~inside_mask] = np.nan
        soil_moisture[~inside_mask] = np.nan

        # stats
        def _safe_minmax(arr):
            if np.all(np.isnan(arr)):
                return (np.nan, np.nan)
            return (float(np.nanmin(arr)), float(np.nanmax(arr)))

        tmin, tmax = _safe_minmax(temperature)
        pmin, pmax = _safe_minmax(precipitation)
        smin, smax = _safe_minmax(soil_moisture)

        print(f"   Temperature range: {tmin:.2f}°C to {tmax:.2f}°C")
        print(f"   Precipitation range: {pmin:.2f} to {pmax:.2f} mm (sum over selected range)")
        print(f"   Soil moisture range: {smin:.4f} to {smax:.4f} m³/m³")

        # 4) valid mask (only used layers)
        print("\n🔥 Step 4: Calculating risk components...")
        valid_mask = (
            (~np.isnan(ndvi_data))
            & (~np.isnan(ndmi_data))
            & (~np.isnan(temperature))
            & (~np.isnan(precipitation))
        )

        num_valid = int(np.sum(valid_mask))
        total_pixels = int(ndvi_data.size)
        coverage_pct = (num_valid / total_pixels) * 100.0 if total_pixels else 0.0
        print(f"   Valid pixels: {num_valid:,} / {total_pixels:,} ({coverage_pct:.1f}%)")

        # NDVI/NDMI threshold-drop stress
        print("\n   A. Vegetation Stress (NDVI_drop recipe)")
        ndvi_drop = np.maximum(0.0, ndvi_min - ndvi_data)
        veg_stress = self._clamp01((ndvi_drop - ndvi_drop_offset) / (ndvi_drop_scale + 1e-6))
        veg_stress[~valid_mask] = np.nan

        print("   B. Moisture Stress (NDMI_drop recipe)")
        ndmi_drop = np.maximum(0.0, ndmi_min - ndmi_data)
        moisture_stress = self._clamp01((ndmi_drop - ndmi_drop_offset) / (ndmi_drop_scale + 1e-6))
        moisture_stress[~valid_mask] = np.nan

        # Heat threshold
        print("   C. Heat Stress (threshold-based)")
        heat = self._normalize_risk_threshold(
            temperature,
            threshold_low=min_temp_c,
            threshold_high=max_temp_c,
            valid_mask=valid_mask,
        )

        # Dryness threshold deficit
        print("   D. Dryness (threshold-based precipitation deficit)")
        dryness = np.full(precipitation.shape, np.nan, dtype="float32")
        precip_def = (min_precip_mm - precipitation) / (min_precip_mm + 1e-6)
        dryness_vals = self._clamp01(precip_def)
        dryness[valid_mask] = dryness_vals[valid_mask]

        # Fuel condition
        print("   E. Fuel Condition (NDMI critical threshold)")
        fuel_critical = (ndmi_data < fuel_ndmi_critical).astype("float32")
        fuel_critical[~valid_mask] = np.nan

        # Combine
        print("\n🎯 Step 5: Computing combined risk indices...")
        drought_risk = (
            0.35 * moisture_stress
            + 0.25 * veg_stress
            + 0.20 * dryness
            + 0.20 * heat
        )
        drought_risk = np.clip(drought_risk, 0.0, 1.0).astype("float32")
        drought_risk[~valid_mask] = np.nan

        """
        While the base formula uses NDMI only as a critical dryness threshold, 
        we apply fuel availability gating to the moisture stress component 
        to prevent unrealistic wildfire risk in urban areas, water bodies, 
        and bare ground where combustible vegetation is absent. 
        This enhancement improves map interpretability while 
        maintaining the core risk drivers from the methodology
        
        """
        
        fuel_critical_binary = (ndmi_data < 0.10).astype("float32")  # For critical flag
        fuel_availability = np.clip((ndmi_data - 0.05) / 0.25, 0.0, 1.0)  # For gating

        # Apply selectively
        veg_fuel_risk = 0.40 * moisture_stress * fuel_availability  # Gate moisture
        climate_risk = 0.20 * heat + 0.20 * dryness  # Keep full (ungated)
        fuel_risk = 0.20 * fuel_critical_binary  # Binary critical fuel

        wildfire_risk = veg_fuel_risk + climate_risk + fuel_risk
        # Stats
        print("\n📈 Step 6: Computing statistics...")
        stats = self._calculate_risk_statistics(drought_risk, wildfire_risk, valid_mask)

        # Save
        print("\n💾 Step 7: Saving risk maps...")
        drought_path = output_path / "drought_risk_map.tif"
        wildfire_path = output_path / "wildfire_risk_map.tif"
        self._save_risk_geotiff(drought_path, drought_risk, ndvi_profile)
        self._save_risk_geotiff(wildfire_path, wildfire_risk, ndvi_profile)

        if save_intermediate:
            self._save_risk_geotiff(output_path / "component_veg_stress.tif", veg_stress, ndvi_profile)
            self._save_risk_geotiff(output_path / "component_moisture_stress.tif", moisture_stress, ndvi_profile)
            self._save_risk_geotiff(output_path / "component_heat.tif", heat, ndvi_profile)
            self._save_risk_geotiff(output_path / "component_dryness.tif", dryness, ndvi_profile)
            self._save_risk_geotiff(output_path / "component_fuel_critical.tif", fuel_critical, ndvi_profile)

        print("\n✅ Outputs:")
        print(f"   Drought:  {drought_path}")
        print(f"   Wildfire: {wildfire_path}")

        return {
            "drought_risk_path": str(drought_path),
            "wildfire_risk_path": str(wildfire_path),
            "drought_risk_data": drought_risk,
            "wildfire_risk_data": wildfire_risk,
            "statistics": stats,
            "coverage_percent": coverage_pct,
        }

    # -----------------------------
    # Raster utils
    # -----------------------------
    def _read_raster(self, path: str) -> Tuple[np.ndarray, dict, Optional[float]]:
        with rasterio.open(path) as src:
            data = src.read(1)
            profile = src.profile
            nodata = src.nodata
        return data, profile, nodata

    def _align_raster_to_master(
        self,
        src_path: str,
        master_profile: dict,
        resampling: Resampling = Resampling.bilinear,
    ) -> np.ndarray:
        with rasterio.open(src_path) as src:
            if (
                src.width == master_profile["width"]
                and src.height == master_profile["height"]
                and src.transform == master_profile["transform"]
                and src.crs == master_profile["crs"]
            ):
                data = src.read(1).astype("float32")
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                return data

            print(f"   Resampling {Path(src_path).name} to master grid...")
            dst_array = np.full(
                (master_profile["height"], master_profile["width"]),
                np.nan,
                dtype="float32",
            )
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=master_profile["transform"],
                dst_crs=master_profile["crs"],
                dst_nodata=np.nan,
                resampling=resampling,
            )
            return dst_array.astype("float32")

    def _build_aoi_mask(self, shp_path: str, master_profile: dict) -> np.ndarray:
        gdf = gpd.read_file(shp_path)
        if gdf.empty:
            raise ValueError(f"AOI shapefile is empty: {shp_path}")
        geom = gdf.geometry.unary_union
        if gdf.crs is not None and gdf.crs != master_profile["crs"]:
            gdf = gdf.to_crs(master_profile["crs"])
            geom = gdf.geometry.unary_union
        mask = geometry_mask(
            [geom],
            out_shape=(master_profile["height"], master_profile["width"]),
            transform=master_profile["transform"],
            invert=True,
        )
        return mask

    # -----------------------------
    # ERA5 processing
    # -----------------------------
    def _open_era5_dataset(self, nc_path: str) -> xr.Dataset:
        engines = ["netcdf4", "h5netcdf", "scipy", None]
        last_err = None
        for eng in engines:
            try:
                t0 = time.time()
                print(f"   -> Trying xr.open_dataset(engine={eng}, decode_times=False)...")
                ds = xr.open_dataset(
                    nc_path,
                    engine=eng,
                    decode_times=False,
                    cache=False,
                    mask_and_scale=True,
                )
                print(f"   ✅ Opened dataset in {time.time()-t0:.2f}s (engine={eng})")
                return ds
            except Exception as e:
                last_err = e
                print(f"   ⚠️ Failed engine={eng}: {type(e).__name__}: {e}")
        raise RuntimeError(f"Could not open NetCDF {nc_path}. Last error: {last_err}")

    def _find_time_dim(self, da: xr.DataArray) -> Optional[str]:
        for d in da.dims:
            if "time" in d.lower():
                return d
        return None

    def _process_era5_to_grid(self, nc_path: str, master_profile: dict) -> Dict[str, np.ndarray]:
        print("   Opening ERA5-Land NetCDF...")
        ds = self._open_era5_dataset(nc_path)

        try:
            print(f"   Dataset dims: {dict(ds.dims)}")
            print(f"   Dataset vars: {list(ds.data_vars)}")

            lat_dim = "latitude" if "latitude" in ds.dims else "lat"
            lon_dim = "longitude" if "longitude" in ds.dims else "lon"

            # Subset to raster bounds only when master CRS is EPSG:4326
            crs = str(master_profile.get("crs", ""))
            if "4326" in crs:
                h = master_profile["height"]
                w = master_profile["width"]
                west, south, east, north = array_bounds(h, w, master_profile["transform"])

                # Buffer (small)
                buf_lat = max(0.01, (north - south) * 0.02)
                buf_lon = max(0.01, (east - west) * 0.02)
                lat_min, lat_max = south - buf_lat, north + buf_lat
                lon_min, lon_max = west - buf_lon, east + buf_lon

                print("   Subsetting ERA5 to raster bounds:")
                print(f"     lat [{lat_min:.4f}, {lat_max:.4f}] lon [{lon_min:.4f}, {lon_max:.4f}]")

                lat_vals = ds[lat_dim].values
                lon_vals = ds[lon_dim].values
                lat_asc = lat_vals[0] < lat_vals[-1]
                lon_asc = lon_vals[0] < lon_vals[-1]
                lat_slice = slice(lat_min, lat_max) if lat_asc else slice(lat_max, lat_min)
                lon_slice = slice(lon_min, lon_max) if lon_asc else slice(lon_max, lon_min)
                ds = ds.sel({lat_dim: lat_slice, lon_dim: lon_slice})
                print(f"   After subset dims: {dict(ds.dims)}")

            lats = ds[lat_dim].values
            lons = ds[lon_dim].values
            expected_shape = (len(lats), len(lons))

            def _nan_grid() -> np.ndarray:
                return np.full(expected_shape, np.nan, dtype="float32")

            # Temperature
            temp_var = next((n for n in ["t2m", "temperature_2m", "2m_temperature"] if n in ds), None)
            if temp_var:
                da = ds[temp_var]
                tdim = self._find_time_dim(da)
                if tdim:
                    da = da.mean(dim=tdim)
                temperature = da.values.astype("float32") - 273.15
            else:
                print("   ⚠️ Temperature variable not found; using NaNs")
                temperature = _nan_grid()

            # Precipitation (sum over time)
            precip_var = next((n for n in ["tp", "total_precipitation"] if n in ds), None)
            if precip_var:
                da = ds[precip_var]
                tdim = self._find_time_dim(da)
                if tdim:
                    da = da.sum(dim=tdim)
                precipitation = da.values.astype("float32") * 1000.0  # m -> mm
            else:
                print("   ⚠️ Precipitation variable not found; using NaNs")
                precipitation = _nan_grid()

            # Soil moisture (mean over time)
            soil_var = next((n for n in ["swvl1", "volumetric_soil_water_layer_1"] if n in ds), None)
            if soil_var:
                da = ds[soil_var]
                tdim = self._find_time_dim(da)
                if tdim:
                    da = da.mean(dim=tdim)
                soil_moisture = da.values.astype("float32")
            else:
                print("   ⚠️ Soil moisture variable not found; using NaNs")
                soil_moisture = _nan_grid()

            # Ensure 2D
            def _ensure_2d(name: str, arr: np.ndarray) -> np.ndarray:
                if arr.ndim == 2:
                    return arr
                print(f"   ⚠️ {name} not 2D after reduction (ndim={arr.ndim}); using NaNs")
                return _nan_grid()

            temperature = _ensure_2d("temperature", temperature)
            precipitation = _ensure_2d("precipitation", precipitation)
            soil_moisture = _ensure_2d("soil_moisture", soil_moisture)

            # Flip latitude if descending
            if len(lats) >= 2 and lats[0] > lats[-1]:
                print("   Flipping latitude...")
                lats = lats[::-1]
                temperature = np.flipud(temperature)
                precipitation = np.flipud(precipitation)
                soil_moisture = np.flipud(soil_moisture)

            era5_transform = from_bounds(
                float(np.min(lons)),
                float(np.min(lats)),
                float(np.max(lons)),
                float(np.max(lats)),
                len(lons),
                len(lats),
            )

            print("   Resampling ERA5 layers to Sentinel-2 grid...")
            resampled = {}
            for name, src_arr in {
                "temperature": temperature,
                "precipitation": precipitation,
                "soil_moisture": soil_moisture,
            }.items():
                dst = np.full(
                    (master_profile["height"], master_profile["width"]),
                    np.nan,
                    dtype="float32",
                )
                reproject(
                    source=src_arr.astype("float32"),
                    destination=dst,
                    src_transform=era5_transform,
                    src_crs="EPSG:4326",
                    src_nodata=np.nan,
                    dst_transform=master_profile["transform"],
                    dst_crs=master_profile["crs"],
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
                resampled[name] = dst

            return resampled

        finally:
            try:
                ds.close()
            except Exception:
                pass

    # -----------------------------
    # Helpers
    # -----------------------------
    def _clamp01(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, 1.0).astype("float32")

    def _normalize_risk_threshold(
        self,
        data: np.ndarray,
        threshold_low: float,
        threshold_high: float,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        out = np.full(data.shape, np.nan, dtype="float32")
        if threshold_high <= threshold_low:
            return out
        risk = (data - threshold_low) / (threshold_high - threshold_low)
        risk = np.clip(risk, 0.0, 1.0).astype("float32")
        out[valid_mask] = risk[valid_mask]
        return out

    def _save_risk_geotiff(self, output_path: Path, data: np.ndarray, base_profile: dict) -> None:
        profile = base_profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=self.nodata,
            compress="deflate",
            bigtiff="IF_SAFER",
        )
        out_data = np.where(np.isnan(data), self.nodata, data).astype("float32")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(out_data, 1)

    def _calculate_risk_statistics(
        self,
        drought_risk: np.ndarray,
        wildfire_risk: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Dict:
        drought_valid = drought_risk[valid_mask]
        wildfire_valid = wildfire_risk[valid_mask]

        bins = [0.0, 0.33, 0.66, 1.0 + 1e-6]
        labels = ["Low (0.00-0.33)", "Moderate (0.33-0.66)", "High (0.66-1.00)"]

        stats = {
            "drought": {
                "mean": float(np.nanmean(drought_valid)) if drought_valid.size else float("nan"),
                "min": float(np.nanmin(drought_valid)) if drought_valid.size else float("nan"),
                "max": float(np.nanmax(drought_valid)) if drought_valid.size else float("nan"),
            },
            "wildfire": {
                "mean": float(np.nanmean(wildfire_valid)) if wildfire_valid.size else float("nan"),
                "min": float(np.nanmin(wildfire_valid)) if wildfire_valid.size else float("nan"),
                "max": float(np.nanmax(wildfire_valid)) if wildfire_valid.size else float("nan"),
            },
        }

        for risk_name, risk_data in [("drought", drought_valid), ("wildfire", wildfire_valid)]:
            total = int(len(risk_data))
            stats[risk_name]["classes"] = {}
            for i, label in enumerate(labels):
                m = (risk_data >= bins[i]) & (risk_data < bins[i + 1])
                count = int(np.sum(m))
                pct = (100.0 * count / total) if total > 0 else 0.0
                stats[risk_name]["classes"][label] = {"count": count, "percentage": pct}

        return stats
    def print_risk_summary(self, stats: Dict) -> None:
        """
        Print a formatted summary of risk statistics to console
        
        Args:
            stats: Statistics dictionary from _calculate_risk_statistics
        """
        print("\n" + "="*80)
        print("RISK ASSESSMENT SUMMARY")
        print("="*80)
        
        for risk_type in ['drought', 'wildfire']:
            print(f"\n{risk_type.upper()} RISK:")
            print(f"  Mean:  {stats[risk_type]['mean']:.3f}")
            print(f"  Range: {stats[risk_type]['min']:.3f} - {stats[risk_type]['max']:.3f}")
            print(f"  Distribution:")
            
            for class_name, class_stats in stats[risk_type]['classes'].items():
                pct = class_stats['percentage']
                count = class_stats['count']
                print(f"    {class_name:25s}: {pct:5.1f}% ({count:,} pixels)")
        
        print("\n" + "="*80)


def create_risk_visualization(
    drought_risk_data: np.ndarray,
    wildfire_risk_data: np.ndarray,
    nodata: float = -9999.0,
) -> "BytesIO":
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    drought_masked = np.ma.masked_where(
        (drought_risk_data == nodata) | np.isnan(drought_risk_data),
        drought_risk_data,
    )
    wildfire_masked = np.ma.masked_where(
        (wildfire_risk_data == nodata) | np.isnan(wildfire_risk_data),
        wildfire_risk_data,
    )

    colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
    cmap = LinearSegmentedColormap.from_list("risk", colors, N=100)
    cmap.set_bad(color="white", alpha=0.3)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    im1 = axes[0].imshow(drought_masked, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title("Drought Risk Map", fontsize=14, fontweight="bold", pad=15)
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(wildfire_masked, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Wildfire Risk Map", fontsize=14, fontweight="bold", pad=15)
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf