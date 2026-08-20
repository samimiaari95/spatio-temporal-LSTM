"""
Download Vapor Pressure Deficit from HydroData – calculated from NLDAS2 variables
==================================================================================
Variables downloaded:
  - NLDAS2: air_temp (K), specific_humidity (kg/kg), atmospheric_pressure (Pa)
  - conus1_baseline_mod: ground_temp (K), sensible_heat (w/m2), latent_heat (w/m2)

VPD Calculation Method:
  VPD is calculated from air temperature and specific humidity using:
  1. Saturation vapor pressure (Buck equation): e_s = 6.1121 * exp((18.678 - T/234.5) * T / (257.14 + T))
     where T is temperature in Celsius
  2. Actual vapor pressure from specific humidity: e_a = q * P / (0.622 + 0.378 * q)
     where q is specific humidity (kg/kg) and P is atmospheric pressure (Pa)
   3. VPD = e_s - e_a (in kPa)

Strategy:
  - One NetCDF file per month: VPD_conus1_YYYY-MM.nc
  - Each month downloaded in N_TILES_X × N_TILES_Y spatial tiles
  - API date_end exclusion is auto-detected and corrected
  - Each request << 2 GB (API limit)

Notes:
  - Run hf.get_catalog_entries(dataset="NLDAS2") to list available variables
  - Run hf.get_catalog_entries(dataset="conus1_baseline_mod") to list available variables
"""

import numpy as np
import hf_hydrodata as hf
import netCDF4 as nc
from datetime import datetime, date, timedelta, timezone
from pyproj import Proj
import calendar, os, math, sys, argparse
import time

# ─── User Parameters ──────────────────────────────────────────────────────────
LAT_MIN_REQ, LON_MIN_REQ = 32.0, -110.0
LAT_MAX_REQ, LON_MAX_REQ = 45.0,  -90.0
GRID            = "conus1"
DATASET_NLDAS   = "NLDAS2"
DATASET_CONUS1  = "conus1_baseline_mod"
TEMPORAL_RES    = "hourly"
DATE_START      = date(2002, 10, 1)
DATE_END        = date(2003,  9, 30)
OUT_DIR         = "VPD_calculated"
# NOTE: TARGET_GB is reduced to 0.5 because we download 3 variables per tile request
# (air_temp, specific_humidity, atmospheric_pressure), tripling the data volume
TARGET_GB       = 0.5

# Variables to download
VARIABLES_NLDAS = ["air_temp", "specific_humidity", "atmospheric_pressure"]
VARIABLES_CONUS1 = ["ground_temp", "sensible_heat", "latent_heat"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and calculate Vapor Pressure Deficit (VPD) from HydroData on CONUS1 grid."
    )
    parser.add_argument("--date-start", type=str, default=DATE_START.isoformat(),
                        help="Start date YYYY-MM-DD (default: %(default)s)")
    parser.add_argument("--date-end", type=str, default=DATE_END.isoformat(),
                        help="End date YYYY-MM-DD (default: %(default)s)")
    parser.add_argument("--out-dir", type=str, default=OUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--temporal-res", type=str, default=TEMPORAL_RES,
                        help="Temporal resolution: 'daily' or 'hourly' (default: %(default)s)")
    return parser.parse_args()


# Parse CLI arguments if provided
if len(sys.argv) > 1:
    args = parse_args()
    DATE_START   = datetime.strptime(args.date_start, "%Y-%m-%d").date()
    DATE_END     = datetime.strptime(args.date_end, "%Y-%m-%d").date()
    OUT_DIR      = args.out_dir
    TEMPORAL_RES = args.temporal_res

# ─── CONUS1 Projection ────────────────────────────────────────────────────────
PROJ_STR = ("+proj=lcc +lat_1=33 +lat_2=45 +lon_0=-96.0 "
            "+lat_0=39 +a=6378137.0 +b=6356752.31")
ORIGIN_X, ORIGIN_Y = -1885055.4995, -604957.0654
RES = 1000.0
X_MAX, Y_MAX = 3342, 1888

proj = Proj(PROJ_STR)


def latlon_to_colrow(lat, lon):
    mx, my = proj(lon, lat)
    return (mx - ORIGIN_X) / RES, (my - ORIGIN_Y) / RES


def colrow_to_latlon(col, row):
    lon, lat = proj(ORIGIN_X + col * RES, ORIGIN_Y + row * RES, inverse=True)
    return lat, lon


def tile_latlon_extent(tx0, ty0, tx1, ty1):
    """Return (lat_min, lat_max, lon_min, lon_max) for a grid tile."""
    corners = [
        colrow_to_latlon(tx0, ty0),
        colrow_to_latlon(tx1, ty0),
        colrow_to_latlon(tx0, ty1),
        colrow_to_latlon(tx1, ty1),
    ]
    lats = [c[0] for c in corners]
    lons = [c[1] for c in corners]
    return min(lats), max(lats), min(lons), max(lons)


def month_chunks(d_start, d_end):
    """List of (month_start, month_end) pairs within [d_start, d_end]."""
    chunks = []
    cur = d_start
    while cur <= d_end:
        last = date(cur.year, cur.month,
                    calendar.monthrange(cur.year, cur.month)[1])
        chunks.append((cur, min(last, d_end)))
        cur = min(last, d_end) + timedelta(days=1)
    return chunks


def calculate_vpd(air_temp_k, specific_humidity, atmospheric_pressure):
    """
    Calculate Vapor Pressure Deficit (VPD) from temperature and humidity.
    
    Parameters:
    -----------
    air_temp_k : numpy.ndarray
        Air temperature in Kelvin
    specific_humidity : numpy.ndarray
        Specific humidity in kg/kg
    atmospheric_pressure : numpy.ndarray
        Atmospheric pressure in Pa
    
    Returns:
    --------
    vpd_kpa : numpy.ndarray
        Vapor Pressure Deficit in kPa
    """
    # Convert temperature from Kelvin to Celsius
    T_celsius = air_temp_k - 273.15
    
    # Calculate saturation vapor pressure using Buck equation (in hPa)
    # e_s = 6.1121 * exp((18.678 - T/234.5) * T / (257.14 + T))
    e_s_hpa = 6.1121 * np.exp((18.678 - T_celsius / 234.5) * T_celsius / (257.14 + T_celsius))
    
    # Calculate actual vapor pressure from specific humidity and atmospheric pressure
    # e_a = q * P / (0.622 + 0.378 * q)
    # where q is specific humidity (kg/kg) and P is atmospheric pressure (Pa)
    e_a_pa = specific_humidity * atmospheric_pressure / (0.622 + 0.378 * specific_humidity)
    
    # Convert both to kPa (1 hPa = 0.1 kPa, 1 Pa = 0.001 kPa)
    e_s_kpa = e_s_hpa * 0.1
    e_a_kpa = e_a_pa * 0.001
    
    # Calculate VPD in kPa
    vpd_kpa = e_s_kpa - e_a_kpa
    
    # Clip negative values (can occur due to numerical errors)
    vpd_kpa = np.maximum(vpd_kpa, 0.0)
    
    return vpd_kpa


def main():
    try:
        hf.register_api_pin(email='miaarisami@gmail.com', pin='1995')
    except Exception:
        pass

    os.makedirs(OUT_DIR, exist_ok=True)

    # Always download hourly data; aggregate to daily mean before writing
    download_res = "hourly"
    steps_per_day = 24

    print("=" * 70)
    print("  HydroData – Vapor Pressure Deficit CONUS1 (Calculated from NLDAS2)")
    print(f"  Datasets: {DATASET_NLDAS}, {DATASET_CONUS1}")
    print(f"  Variables: {', '.join(VARIABLES_NLDAS)}")
    print(f"  Download res: hourly  →  Output res: daily (mean)")
    print(f"  VPD units: kPa")
    print("=" * 70)

    # ── 1. Grid domain ────────────────────────────────────────────────────────────
    print("\n[1/5] Computing grid domain from lat/lon bounds...")
    corners_latlon = [
        (LAT_MIN_REQ, LON_MIN_REQ),
        (LAT_MIN_REQ, LON_MAX_REQ),
        (LAT_MAX_REQ, LON_MIN_REQ),
        (LAT_MAX_REQ, LON_MAX_REQ),
    ]
    cols_c, rows_c = zip(*[latlon_to_colrow(la, lo) for la, lo in corners_latlon])

    X0 = max(0,     int(np.floor(min(cols_c))))
    Y0 = max(0,     int(np.floor(min(rows_c))))
    X1 = min(X_MAX, int(np.ceil (max(cols_c))))
    Y1 = min(Y_MAX, int(np.ceil (max(rows_c))))
    NX = X1 - X0
    NY = Y1 - Y0

    # Effective bounding box in lat/lon
    lat_sw, lon_sw = colrow_to_latlon(X0, Y0)
    lat_ne, lon_ne = colrow_to_latlon(X1, Y1)
    print(f"  Requested bbox : lat [{LAT_MIN_REQ}, {LAT_MAX_REQ}]  "
          f"lon [{LON_MIN_REQ}, {LON_MAX_REQ}]")
    print(f"  Effective bbox : lat [{lat_sw:.3f}, {lat_ne:.3f}]  "
          f"lon [{lon_sw:.3f}, {lon_ne:.3f}]")
    print(f"  Grid size      : {NX} × {NY} cells  "
          f"(grid indices x[{X0}:{X1}] y[{Y0}:{Y1}] — internal only)")

    # ── 2. Detect API date_end exclusion ─────────────────────────────────────────
    print("\n[2/5] Detecting API date_end behaviour...")
    probe = hf.get_gridded_data({
        "dataset"             : DATASET_NLDAS,
        "variable"            : VARIABLES_NLDAS[0],
        "temporal_resolution" : download_res,
        "date_start"          : DATE_START.isoformat(),
        "date_end"            : (DATE_START + timedelta(days=1)).isoformat(),
        "grid_bounds"         : [X0, Y0, X0 + 2, Y0 + 2],
        "grid"                : GRID,
    })
    expected_steps = 2 * steps_per_day
    api_offset     = expected_steps - probe.shape[0]
    api_day_offset = 1 if api_offset > 0 else 0
    print(f"  API returned {probe.shape[0]} step(s)  "
          f"→ date_end {'EXcluded' if api_day_offset else 'included'}")

    # ── 3. Spatial tiles ──────────────────────────────────────────────────────────
    print("\n[3/5] Computing spatial tiles...")
    max_days   = 31
    max_cells  = int(TARGET_GB * 1e9 / (max_days * steps_per_day * 4))
    n_tiles    = math.ceil(NX * NY / max_cells)
    n_tiles_x  = max(1, math.ceil(math.sqrt(n_tiles * NX / NY)))
    n_tiles_y  = max(1, math.ceil(n_tiles / n_tiles_x))
    tile_nx    = math.ceil(NX / n_tiles_x)
    tile_ny    = math.ceil(NY / n_tiles_y)

    dlon_tile = abs(lon_ne - lon_sw) / n_tiles_x
    dlat_tile = abs(lat_ne - lat_sw) / n_tiles_y
    print(f"  {n_tiles_x} × {n_tiles_y} tiles  "
          f"(~{dlon_tile:.2f}° lon × {dlat_tile:.2f}° lat per tile, "
          f"{tile_nx} × {tile_ny} grid cells)")

    # ── 4. 2D lat/lon grid for the full domain ────────────────────────────────────
    print("\n[4/5] Computing 2D lat/lon arrays...")
    col_v = np.arange(X0, X0 + NX, dtype=np.float64)
    row_v = np.arange(Y0, Y0 + NY, dtype=np.float64)
    col_2d, row_2d = np.meshgrid(col_v, row_v)
    lons_2d, lats_2d = proj(ORIGIN_X + col_2d * RES,
                             ORIGIN_Y + row_2d * RES, inverse=True)
    lats = lats_2d.astype(np.float32)   # (NY, NX)
    lons = lons_2d.astype(np.float32)
    print(f"  lat range: [{lats.min():.3f}, {lats.max():.3f}]  "
          f"lon range: [{lons.min():.3f}, {lons.max():.3f}]")

    # ── 5. Main loop: one NetCDF per month ───────────────────────────────────────
    months     = month_chunks(DATE_START, DATE_END)
    total_req  = n_tiles_x * n_tiles_y * len(months)
    req_num    = 0
    all_errors = []

    print(f"\n[5/5] {len(months)} months  ×  {n_tiles_x}×{n_tiles_y} tiles"
          f"  =  {total_req} requests")
    print(f"  Files → {os.path.abspath(OUT_DIR)}/VPD_conus1_YYYY-MM.nc\n")

    for ms, me in months:
        me_req     = me + timedelta(days=api_day_offset)
        days_real  = (me - ms).days + 1
        steps_real = days_real * steps_per_day
        nc_file    = os.path.join(OUT_DIR,
                                  f"VPD_conus1_{ms.strftime('%Y-%m')}.nc")

        if os.path.exists(nc_file):
            size_mb = os.path.getsize(nc_file) / 1e6
            print(f"  ── {ms} → {me}  SKIP (already exists, {size_mb:.1f} MB)")
            req_num += n_tiles_x * n_tiles_y
            continue

        print(f"  ── {ms} → {me}  ({days_real} days, {steps_real} hourly steps"
              f" → {days_real} daily steps)"
              f"  →  {os.path.basename(nc_file)}")

        # Accumulator for hourly VPD across all tiles in this month
        vpd_hourly = np.zeros((steps_real, NY, NX), dtype=np.float64)

        # ── Tile loop: download hourly data and accumulate VPD ────────────────────
        for ti in range(n_tiles_y):
            ty0 = Y0 + ti * tile_ny
            ty1 = min(Y0 + (ti + 1) * tile_ny, Y1)
            ly0 = ti * tile_ny
            ly1 = ly0 + (ty1 - ty0)

            for tj in range(n_tiles_x):
                time.sleep(1)  # Avoid overwhelming the API
                tx0 = X0 + tj * tile_nx
                tx1 = min(X0 + (tj + 1) * tile_nx, X1)
                lx0 = tj * tile_nx
                lx1 = lx0 + (tx1 - tx0)
                req_num += 1
                est_gb  = (tx1 - tx0) * (ty1 - ty0) * steps_real * 4 / 1e9

                tlat_min, tlat_max, tlon_min, tlon_max = tile_latlon_extent(
                    tx0, ty0, tx1, ty1)

                print(f"    [{req_num:3d}/{total_req}] "
                      f"tile({tj+1},{ti+1})  "
                      f"lat[{tlat_min:.2f},{tlat_max:.2f}]  "
                      f"lon[{tlon_min:.2f},{tlon_max:.2f}]  "
                      f"~{est_gb:.3f} GB ... ",
                      end="", flush=True)
                try:
                    air_temp_data = hf.get_gridded_data({
                        "dataset"             : DATASET_NLDAS,
                        "variable"            : "air_temp",
                        "temporal_resolution" : download_res,
                        "date_start"          : ms.isoformat(),
                        "date_end"            : me_req.isoformat(),
                        "grid_bounds"         : [tx0, ty0, tx1, ty1],
                        "grid"                : GRID,
                    })
                    n_got = air_temp_data.shape[0]

                    specific_humidity_data = hf.get_gridded_data({
                        "dataset"             : DATASET_NLDAS,
                        "variable"            : "specific_humidity",
                        "temporal_resolution" : download_res,
                        "date_start"          : ms.isoformat(),
                        "date_end"            : me_req.isoformat(),
                        "grid_bounds"         : [tx0, ty0, tx1, ty1],
                        "grid"                : GRID,
                    })

                    atmospheric_pressure_data = hf.get_gridded_data({
                        "dataset"             : DATASET_NLDAS,
                        "variable"            : "atmospheric_pressure",
                        "temporal_resolution" : download_res,
                        "date_start"          : ms.isoformat(),
                        "date_end"            : me_req.isoformat(),
                        "grid_bounds"         : [tx0, ty0, tx1, ty1],
                        "grid"                : GRID,
                    })

                    vpd_data = calculate_vpd(
                        air_temp_data,
                        specific_humidity_data,
                        atmospheric_pressure_data
                    )
                    vpd_hourly[0:n_got, ly0:ly1, lx0:lx1] = vpd_data

                    print(f"OK  {air_temp_data.shape}")

                except Exception as e:
                    print(f"ERROR: {e}")
                    all_errors.append(
                        f"{ms} tile({tj+1},{ti+1}) "
                        f"lat[{tlat_min:.2f},{tlat_max:.2f}] "
                        f"lon[{tlon_min:.2f},{tlon_max:.2f}]: {e}")

        # ── Aggregate hourly → daily mean ─────────────────────────────────────────
        vpd_daily = vpd_hourly.reshape(days_real, 24, NY, NX).mean(axis=1).astype(np.float32)

        # ── Write daily NetCDF ────────────────────────────────────────────────────
        time_units = f"days since {ms.isoformat()} 00:00:00"
        with nc.Dataset(nc_file, "w", format="NETCDF4") as ds:
            ds.title          = "Vapor Pressure Deficit (Daily Mean) – Calculated from NLDAS2 on ParFlow CONUS1 grid"
            ds.source         = "HydroData / HydroFrame (hf_hydrodata)"
            ds.dataset        = f"{DATASET_NLDAS} + {DATASET_CONUS1}"
            ds.variables_used = f"{', '.join(VARIABLES_NLDAS)}"
            ds.temporal_res   = "daily"
            ds.date_start     = ms.isoformat()
            ds.date_end       = me.isoformat()
            ds.bbox_requested = (f"lat [{LAT_MIN_REQ}, {LAT_MAX_REQ}]  "
                                 f"lon [{LON_MIN_REQ}, {LON_MAX_REQ}]")
            ds.bbox_effective = (f"lat [{lat_sw:.4f}, {lat_ne:.4f}]  "
                                 f"lon [{lon_sw:.4f}, {lon_ne:.4f}]")
            ds.grid           = GRID
            ds.grid_projection= PROJ_STR
            ds.created        = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            ds.Conventions    = "CF-1.8"
            ds.vpd_formula    = ("VPD calculated using Buck equation for saturation vapor pressure "
                                 "and specific humidity with atmospheric pressure for actual vapor pressure. "
                                 "Hourly VPD aggregated to daily mean.")

            # ── Dimensions ────────────────────────────────────────────────────────
            ds.createDimension("time", days_real)
            ds.createDimension("lat",  NY)
            ds.createDimension("lon",  NX)

            # ── Time variable ─────────────────────────────────────────────────────
            tv           = ds.createVariable("time", "f8", ("time",))
            tv.units     = time_units
            tv.calendar  = "standard"
            tv.long_name = "time"
            tv.axis      = "T"
            tv[:]        = np.arange(days_real, dtype=np.float64)

            # ── Coordinate variables ──────────────────────────────────────────────
            latv               = ds.createVariable("lat", "f4", ("lat", "lon"))
            latv.long_name     = "latitude"
            latv.standard_name = "latitude"
            latv.units         = "degrees_north"
            latv[:]            = lats

            lonv               = ds.createVariable("lon", "f4", ("lat", "lon"))
            lonv.long_name     = "longitude"
            lonv.standard_name = "longitude"
            lonv.units         = "degrees_east"
            lonv[:]            = lons

            # ── Vapor Pressure Deficit variable ───────────────────────────────────
            vpd = ds.createVariable(
                "vapor_pressure_deficit", "f4", ("time", "lat", "lon"),
                zlib=True, complevel=4,
                chunksizes=(1, min(NY, 128), min(NX, 128)),
                fill_value=np.float32(-9999.0)
            )
            vpd.long_name     = "Vapor Pressure Deficit (Daily Mean)"
            vpd.standard_name = "water_vapor_pressure_deficit"
            vpd.units         = "kPa"
            vpd.coordinates   = "lat lon"
            vpd.missing_value = np.float32(-9999.0)
            vpd.comment       = ("VPD calculated from hourly NLDAS2 air temperature and "
                                 "specific humidity, then averaged to daily mean. "
                                 "Source: HydroFrame hf_hydrodata.")
            vpd[:]            = vpd_daily

        size_mb = os.path.getsize(nc_file) / 1e6
        print(f"  ✓ {os.path.basename(nc_file)}  {size_mb:.1f} MB\n")

    # ── Final summary ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  Files generated : {len(months)}  in  {os.path.abspath(OUT_DIR)}/")
    for ms, me in months:
        f = os.path.join(OUT_DIR, f"VPD_conus1_{ms.strftime('%Y-%m')}.nc")
        mb = os.path.getsize(f) / 1e6 if os.path.exists(f) else 0
        print(f"    {os.path.basename(f)}  {mb:.1f} MB")
    if all_errors:
        print(f"\n  ERRORS ({len(all_errors)}):")
        for e in all_errors:
            print(f"    - {e}")
    else:
        print("\n  All requests completed without errors. ✓")


if __name__ == "__main__":
    main()
