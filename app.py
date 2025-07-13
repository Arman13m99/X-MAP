import os
from datetime import datetime
import re # For splitting vendor codes
import threading # For opening browser without blocking
import webbrowser # For opening browser
import time # For delay before opening browser
import random # For population heatmap point generation
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, Polygon
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import hashlib
from functools import lru_cache

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
PUBLIC_DIR = os.path.join(BASE_DIR, 'public')

# --- Initialize Flask App ---
app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path='')
CORS(app)

# Enable response compression
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600  # Cache static files for 1 hour
app.config['JSON_SORT_KEYS'] = False  # Don't sort JSON keys for slightly faster response

# --- Global Data Variables ---
df_orders = None
df_vendors = None
gdf_marketing_areas = {}
gdf_tehran_region = None
gdf_tehran_main_districts = None
city_id_map = {1: "mashhad", 2: "tehran", 5: "shiraz"}
city_name_to_id_map = {v: k for k, v in city_id_map.items()}

# Simple cache for coverage calculations
coverage_cache = {}
CACHE_SIZE = 100  # Limit cache size to prevent memory issues

# City boundaries for grid generation (approximate)
city_boundaries = {
    "tehran": {"min_lat": 35.5, "max_lat": 35.85, "min_lng": 51.1, "max_lng": 51.7},
    "mashhad": {"min_lat": 36.15, "max_lat": 36.45, "min_lng": 59.35, "max_lng": 59.8},
    "shiraz": {"min_lat": 29.5, "max_lat": 29.75, "min_lng": 52.4, "max_lng": 52.7}
}

# --- Helper Functions ---
def safe_tolist(series):
    if series.empty:
        return []
    cleaned = series.dropna().unique()
    if pd.api.types.is_numeric_dtype(cleaned.dtype):
        return [item.item() if hasattr(item, 'item') else item for item in cleaned]
    return cleaned.tolist()

def load_tehran_shapefile(filename):
    shp_path = os.path.join(SRC_DIR, 'polygons', 'tehran_districts', filename)
    tried_encodings = [None, 'cp1256', 'utf-8']
    loaded_gdf = None
    for enc in tried_encodings:
        try:
            gdf_temp = gpd.read_file(shp_path, encoding=enc if enc else None)
            print(f"Loaded {filename} using encoding='{enc or 'default'}'")
            loaded_gdf = gdf_temp
            break
        except Exception as e:
            print(f"  • Trying encoding {enc!r} for {filename}: failed with: {e}")
    if loaded_gdf is None:
        print(f"Could not load {filename}")
        return gpd.GeoDataFrame()
    if loaded_gdf.crs and loaded_gdf.crs.to_string() != "EPSG:4326":
        try:
            loaded_gdf = loaded_gdf.to_crs("EPSG:4326")
            print(f"  • Reprojected {filename} to EPSG:4326")
        except Exception as e:
            print(f"  • Failed to reproject {filename}: {e}")
    print(f"{filename} columns: {list(loaded_gdf.columns)}")
    return loaded_gdf

def get_district_names_from_gdf(gdf, default_prefix="District"):
    if gdf is None or gdf.empty: return []
    name_cols = ['Name', 'name', 'NAME', 'Region', 'REGION_N', 'NAME_MAHAL', 'NAME_1', 'NAME_2', 'district']
    for col in name_cols:
        if col in gdf.columns and gdf[col].dtype == 'object':
            return sorted(safe_tolist(gdf[col].astype(str)))
    for col in gdf.columns: # Fallback
        if col != 'geometry' and gdf[col].dtype == 'object':
            return sorted(safe_tolist(gdf[col].astype(str)))
    return [f"{default_prefix} {i+1}" for i in range(len(gdf))]

def generate_random_points_in_polygon(poly, num_points):
    """Generates a specified number of random points within a given Shapely polygon."""
    points = []
    min_x, min_y, max_x, max_y = poly.bounds
    while len(points) < num_points:
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if random_point.within(poly):
            points.append(random_point)
    return points

def generate_coverage_grid(city_name, grid_size_meters=200):
    """Generate a grid of points for coverage analysis."""
    if city_name not in city_boundaries:
        return []
    
    bounds = city_boundaries[city_name]
    
    # Convert grid size from meters to approximate degrees
    # At these latitudes, 1 degree ≈ 111 km
    grid_size_deg = grid_size_meters / 111000.0
    
    grid_points = []
    lat = bounds["min_lat"]
    while lat <= bounds["max_lat"]:
        lng = bounds["min_lng"]
        while lng <= bounds["max_lng"]:
            grid_points.append({"lat": lat, "lng": lng})
            lng += grid_size_deg
        lat += grid_size_deg
    
    return grid_points

def calculate_coverage_for_grid_vectorized(grid_points, df_vendors_filtered, city_name):
    """
    Calculate vendor coverage for all grid points using vectorized operations.
    Much faster than calculating point by point.
    """
    if df_vendors_filtered.empty or not grid_points:
        return []
    
    # Filter vendors with valid data
    valid_vendors = df_vendors_filtered.dropna(subset=['latitude', 'longitude', 'radius'])
    if valid_vendors.empty:
        return []
    
    # Convert to numpy arrays for faster computation
    grid_lats = np.array([p['lat'] for p in grid_points])
    grid_lngs = np.array([p['lng'] for p in grid_points])
    
    vendor_lats = valid_vendors['latitude'].values
    vendor_lngs = valid_vendors['longitude'].values
    vendor_radii = valid_vendors['radius'].values * 1000  # Convert km to meters
    
    # Pre-extract vendor attributes
# Around line 150
    if 'business_line' in valid_vendors:
        # First, access the categorical methods (.cat), add the new category, then fill NaNs
        vendor_business_lines = valid_vendors['business_line'].cat.add_categories(['Unknown']).fillna('Unknown').values
    else:
        vendor_business_lines = None

    if 'grade' in valid_vendors:
        # Proactively apply the same fix for the 'grade' column, as it will have the same problem
        vendor_grades = valid_vendors['grade'].cat.add_categories(['Unknown']).fillna('Unknown').values
    else:
        vendor_grades = None
    
    coverage_results = []
    
    # Process in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(grid_points), batch_size):
        batch_end = min(i + batch_size, len(grid_points))
        batch_lats = grid_lats[i:batch_end]
        batch_lngs = grid_lngs[i:batch_end]
        
        # Vectorized distance calculation for the batch
        # Using broadcasting to calculate distances from all batch points to all vendors
        lat_diff = batch_lats[:, np.newaxis] - vendor_lats[np.newaxis, :]
        lng_diff = batch_lngs[:, np.newaxis] - vendor_lngs[np.newaxis, :]
        
        # Approximate distance in meters (good enough for our scale)
        distances_meters = np.sqrt((lat_diff * 111000)**2 + (lng_diff * 111000 * np.cos(np.radians(batch_lats[:, np.newaxis])))**2)
        
        # Check which vendors cover each point
        coverage_matrix = distances_meters <= vendor_radii[np.newaxis, :]
        
        # Process results for each point in batch
        for j, point_idx in enumerate(range(i, batch_end)):
            covering_vendors = np.where(coverage_matrix[j])[0]
            
            coverage_data = {
                "lat": grid_points[point_idx]['lat'],
                "lng": grid_points[point_idx]['lng'],
                "total_vendors": len(covering_vendors),
                "by_business_line": {},
                "by_grade": {}
            }
            
            if len(covering_vendors) > 0:
                # Count by business line
                if vendor_business_lines is not None:
                    bl_counts = {}
                    for vendor_idx in covering_vendors:
                        bl = vendor_business_lines[vendor_idx]
                        bl_counts[bl] = bl_counts.get(bl, 0) + 1
                    coverage_data["by_business_line"] = bl_counts
                
                # Count by grade
                if vendor_grades is not None:
                    grade_counts = {}
                    for vendor_idx in covering_vendors:
                        grade = vendor_grades[vendor_idx]
                        grade_counts[grade] = grade_counts.get(grade, 0) + 1
                    coverage_data["by_grade"] = grade_counts
            
            coverage_results.append(coverage_data)
    
    return coverage_results

def find_marketing_area_for_points(points, city_name):
    """Find which marketing area each point belongs to."""
    if city_name not in gdf_marketing_areas or gdf_marketing_areas[city_name].empty:
        return [None] * len(points)
    
    gdf_areas = gdf_marketing_areas[city_name]
    marketing_areas = []
    
    # Create a spatial index for faster lookup if available
    try:
        from shapely.strtree import STRtree
        area_geoms = gdf_areas.geometry.values
        area_names = gdf_areas['name'].values if 'name' in gdf_areas.columns else [f"Area_{i}" for i in range(len(gdf_areas))]
        tree = STRtree(area_geoms)
        
        for point in points:
            point_geom = Point(point['lng'], point['lat'])
            # Find potential matches using spatial index
            result = tree.query(point_geom, predicate='contains')
            if len(result) > 0:
                # Take the first matching area
                marketing_areas.append(area_names[result[0]])
            else:
                marketing_areas.append(None)
    except ImportError:
        # Fallback to non-indexed method
        for point in points:
            point_geom = Point(point['lng'], point['lat'])
            found = False
            for idx, area in gdf_areas.iterrows():
                if area.geometry.contains(point_geom):
                    area_name = area.get('name', f"Area_{idx}")
                    marketing_areas.append(area_name)
                    found = True
                    break
            if not found:
                marketing_areas.append(None)
    
    return marketing_areas

def remove_outliers_and_normalize(df, value_column, lower_percentile=5, upper_percentile=95):
    """
    Remove outliers using percentile method and normalize values to 0-100 range.
    Returns a copy of the dataframe with normalized values.
    """
    if df.empty or value_column not in df.columns:
        return df
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Remove rows where the value is null or zero
    df_copy = df_copy[df_copy[value_column].notna() & (df_copy[value_column] > 0)]
    
    if df_copy.empty:
        print(f"No valid {value_column} data after removing nulls/zeros")
        return df_copy
    
    # Calculate percentiles for outlier removal
    lower_bound = df_copy[value_column].quantile(lower_percentile / 100)
    upper_bound = df_copy[value_column].quantile(upper_percentile / 100)
    
    print(f"{value_column} bounds: {lower_bound:,.0f} to {upper_bound:,.0f}")
    
    # Remove outliers
    df_copy = df_copy[(df_copy[value_column] >= lower_bound) & (df_copy[value_column] <= upper_bound)]
    
    if df_copy.empty:
        print(f"No data left after outlier removal for {value_column}")
        return df_copy
    
    # Normalize to 0-100 range
    min_val = df_copy[value_column].min()
    max_val = df_copy[value_column].max()
    
    if max_val > min_val:
        df_copy[f'{value_column}_normalized'] = ((df_copy[value_column] - min_val) / (max_val - min_val)) * 100
    else:
        df_copy[f'{value_column}_normalized'] = 50  # If all values are the same, set to middle value
    
    # Log transformation for better distribution (optional, helps with skewed data)
    # This helps when you have many small values and few large values
    df_copy[f'{value_column}_log_normalized'] = np.log1p(df_copy[value_column])
    log_min = df_copy[f'{value_column}_log_normalized'].min()
    log_max = df_copy[f'{value_column}_log_normalized'].max()
    
    if log_max > log_min:
        df_copy[f'{value_column}_log_normalized'] = ((df_copy[f'{value_column}_log_normalized'] - log_min) / (log_max - log_min)) * 100
    else:
        df_copy[f'{value_column}_log_normalized'] = 50
    
    print(f"{value_column} normalization complete: {len(df_copy)} points")
    return df_copy

def aggregate_heatmap_points(df, lat_col, lng_col, value_col, precision=4):
    """
    Aggregate heatmap points by rounding coordinates to create heat accumulation.
    This ensures areas with multiple orders show more heat.
    """
    if df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Round coordinates to aggregate nearby points
    df_copy['lat_rounded'] = df_copy[lat_col].round(precision)
    df_copy['lng_rounded'] = df_copy[lng_col].round(precision)
    
    # Aggregate values for the same rounded location
    aggregated = df_copy.groupby(['lat_rounded', 'lng_rounded']).agg({
        value_col: 'sum'
    }).reset_index()
    
    # Rename columns to standard names for output
    aggregated['lat'] = aggregated['lat_rounded']
    aggregated['lng'] = aggregated['lng_rounded']
    aggregated['value'] = aggregated[value_col]
    
    return aggregated[['lat', 'lng', 'value']]

def aggregate_user_heatmap_points(df, lat_col, lng_col, user_col, precision=4):
    """
    Aggregate unique users by location for user heatmap.
    """
    if df.empty:
        return df
    
    # Create a copy
    df_copy = df.copy()
    
    # Round coordinates
    df_copy['lat_rounded'] = df_copy[lat_col].round(precision)
    df_copy['lng_rounded'] = df_copy[lng_col].round(precision)
    
    # Count unique users per location
    aggregated = df_copy.groupby(['lat_rounded', 'lng_rounded'])[user_col].nunique().reset_index()
    
    # Rename columns
    aggregated['lat'] = aggregated['lat_rounded']
    aggregated['lng'] = aggregated['lng_rounded']
    aggregated['value'] = aggregated[user_col]
    
    # Normalize values
    if len(aggregated) > 0:
        max_val = aggregated['value'].max()
        min_val = aggregated['value'].min()
        if max_val > min_val:
            aggregated['value'] = ((aggregated['value'] - min_val) / (max_val - min_val)) * 100
        else:
            aggregated['value'] = 50
    
    return aggregated[['lat', 'lng', 'value']]

def load_data():
    global df_orders, df_vendors, gdf_marketing_areas, gdf_tehran_region, gdf_tehran_main_districts
    print("Loading data...")
    start_time = time.time()
    
    # 1. Order Data - with optimized dtypes
    order_file = os.path.join(SRC_DIR, 'order', 'x_map_order.csv')
    try:
        # Specify dtypes to reduce memory usage and speed up loading
        dtype_dict = {
            'city_id': 'Int64',
            'business_line': 'category',
            'marketing_area': 'category',
            'vendor_code': 'str',
            'organic': 'int8'
        }
        
        df_orders = pd.read_csv(order_file, dtype=dtype_dict, parse_dates=['created_at'])
        df_orders['city_name'] = df_orders['city_id'].map(city_id_map).astype('category')
        
        # Only add missing columns, don't override existing ones
        for col in ['customer_latitude', 'customer_longitude', 'payable_price', 'voucher_value', 'order_id', 'user_id']:
            if col not in df_orders.columns: 
                df_orders[col] = np.nan
        
        if 'organic' not in df_orders.columns:
            # If organic column doesn't exist, create a random one for demo
            df_orders['organic'] = np.random.choice([0, 1], size=len(df_orders), p=[0.7, 0.3]).astype('int8')
                
        print(f"Orders loaded: {len(df_orders)} rows in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Error loading order data: {e}")
        df_orders = pd.DataFrame()
        
    # 2. Vendor Data
    vendor_file = os.path.join(SRC_DIR, 'vendor', 'x_map_vendor.csv')
    graded_file = os.path.join(SRC_DIR, 'vendor', 'graded.csv')
    try:
        vendor_start = time.time()
        # Optimize dtypes for vendors too
        vendor_dtype = {
            'city_id': 'Int64',
            'vendor_code': 'str',
            'status_id': 'float32',
            'visible': 'float32',
            'open': 'float32',
            'radius': 'float32'
        }
        
        df_vendors_raw = pd.read_csv(vendor_file, dtype=vendor_dtype)
        df_vendors_raw['city_name'] = df_vendors_raw['city_id'].map(city_id_map).astype('category')
        
        try:
            df_graded_data = pd.read_csv(graded_file, dtype={'vendor_code': 'str', 'grade': 'category'})
            if 'vendor_code' in df_vendors_raw.columns and 'vendor_code' in df_graded_data.columns:
                df_vendors_raw['vendor_code'] = df_vendors_raw['vendor_code'].astype(str)
                df_graded_data['vendor_code'] = df_graded_data['vendor_code'].astype(str)
                df_vendors = pd.merge(df_vendors_raw, df_graded_data[['vendor_code', 'grade']], on='vendor_code', how='left')
                if 'grade' in df_vendors.columns and pd.api.types.is_categorical_dtype(df_vendors['grade']):
                    df_vendors['grade'] = df_vendors['grade'].cat.add_categories(['Ungraded'])
                df_vendors['grade'] = df_vendors['grade'].fillna('Ungraded').astype('category')
                print(f"Grades loaded and merged. Found {df_vendors['grade'].notna().sum()} graded vendors.")
            else:
                print("Warning: 'vendor_code' column missing in vendors or graded CSV. Grades not merged.")
                df_vendors = df_vendors_raw
                if 'grade' not in df_vendors.columns: 
                    df_vendors['grade'] = pd.Categorical(['Ungraded'] * len(df_vendors))
        except Exception as eg:
            print(f"Error loading or merging graded.csv: {eg}. Proceeding without grades.")
            df_vendors = df_vendors_raw
            if 'grade' not in df_vendors.columns: 
                df_vendors['grade'] = pd.Categorical(['Ungraded'] * len(df_vendors))
        
        # Add business_line to vendors by joining with orders (use mode to get most common)
        if not df_orders.empty and 'vendor_code' in df_vendors.columns and 'vendor_code' in df_orders.columns:
            vendor_bl = df_orders.groupby('vendor_code')['business_line'].agg(lambda x: x.mode()[0] if not x.empty else np.nan)
            df_vendors = df_vendors.merge(vendor_bl.rename('business_line'), left_on='vendor_code', right_index=True, how='left')
            if 'business_line' in df_vendors.columns:
                df_vendors['business_line'] = df_vendors['business_line'].astype('category')
        
        for col in ['latitude', 'longitude', 'vendor_name', 'radius', 'status_id', 'visible', 'open', 'vendor_code']:
            if col not in df_vendors.columns: df_vendors[col] = np.nan
            
        if 'visible' in df_vendors.columns: df_vendors['visible'] = pd.to_numeric(df_vendors['visible'], errors='coerce')
        if 'open' in df_vendors.columns: df_vendors['open'] = pd.to_numeric(df_vendors['open'], errors='coerce')
        if 'status_id' in df_vendors.columns: df_vendors['status_id'] = pd.to_numeric(df_vendors['status_id'], errors='coerce')
        if 'vendor_code' in df_vendors.columns: df_vendors['vendor_code'] = df_vendors['vendor_code'].astype(str)
        
        # Store original radius for reset functionality
        if 'radius' in df_vendors.columns:
            df_vendors['original_radius'] = df_vendors['radius'].copy()
            
        print(f"Vendors loaded: {len(df_vendors)} rows in {time.time() - vendor_start:.2f}s")
    except Exception as e:
        print(f"Error loading vendor data: {e}")
        df_vendors = pd.DataFrame()
        if df_vendors is not None and 'grade' not in df_vendors.columns: 
            df_vendors['grade'] = pd.Categorical(['Ungraded'] * len(df_vendors))
            
    # 3. Polygon Data - Load in parallel if possible
    poly_start = time.time()
    marketing_areas_base = os.path.join(SRC_DIR, 'polygons', 'tapsifood_marketing_areas')
    for city_file_name in ['mashhad_polygons.csv', 'tehran_polygons.csv', 'shiraz_polygons.csv']:
        city_name_key = city_file_name.split('_')[0]
        file_path = os.path.join(marketing_areas_base, city_file_name)
        try:
            df_poly = pd.read_csv(file_path, encoding='utf-8')
            if 'WKT' not in df_poly.columns: df_poly['WKT'] = None
            df_poly['geometry'] = df_poly['WKT'].apply(lambda x: wkt.loads(x) if pd.notna(x) else None)
            if 'name' not in df_poly.columns: 
                df_poly['name'] = [f"{city_name_key}_area_{i+1}" for i in range(len(df_poly))]
            else:
                df_poly['name'] = df_poly['name'].astype('category')
            gdf_marketing_areas[city_name_key] = gpd.GeoDataFrame(df_poly, geometry='geometry', crs="EPSG:4326").dropna(subset=['geometry'])
            print(f"Marketing areas for {city_name_key} loaded: {len(gdf_marketing_areas[city_name_key])} polygons")
        except Exception as e:
            print(f"Error loading marketing areas for {city_name_key} from {file_path}: {e}")
            gdf_marketing_areas[city_name_key] = gpd.GeoDataFrame()
            
    gdf_tehran_region = load_tehran_shapefile('RegionTehran_WGS1984.shp')
    gdf_tehran_main_districts = load_tehran_shapefile('Tehran_WGS1984.shp')
    
    total_time = time.time() - start_time
    print(f"Data loading complete in {total_time:.2f} seconds.")

# --- Serve Static Files (Frontend) ---
@app.route('/')
def serve_index():
    return send_from_directory(PUBLIC_DIR, 'index.html')

@app.route('/api/initial-data', methods=['GET'])
def get_initial_data():
    if df_orders is None or df_vendors is None:
        return jsonify({"error": "Data not loaded properly"}), 500
    
    cities = [{"id": cid, "name": name} for cid, name in city_id_map.items()]
    business_lines = []
    if not df_orders.empty and 'business_line' in df_orders.columns:
        business_lines = safe_tolist(df_orders['business_line'])
    
    marketing_area_names_by_city = {}
    for city_key, gdf in gdf_marketing_areas.items():
        if not gdf.empty and 'name' in gdf.columns:
            marketing_area_names_by_city[city_key] = sorted(safe_tolist(gdf['name'].astype(str)))
        else: 
            marketing_area_names_by_city[city_key] = []
    
    tehran_region_districts = get_district_names_from_gdf(gdf_tehran_region, "Region Tehran")
    tehran_main_districts = get_district_names_from_gdf(gdf_tehran_main_districts, "Main Tehran")
    
    vendor_statuses = []
    if not df_vendors.empty and 'status_id' in df_vendors.columns:
        # Filter out NaN values before converting to int
        status_series = df_vendors['status_id'].dropna()
        if not status_series.empty:
            vendor_statuses = sorted([int(x) for x in status_series.unique()])
    
    vendor_grades = []
    if not df_vendors.empty and 'grade' in df_vendors.columns:
        vendor_grades = sorted(safe_tolist(df_vendors['grade'].astype(str)))
    
    return jsonify({
        "cities": cities,
        "business_lines": business_lines,
        "marketing_areas_by_city": marketing_area_names_by_city,
        "tehran_region_districts": tehran_region_districts,
        "tehran_main_districts": tehran_main_districts,
        "vendor_statuses": vendor_statuses,
        "vendor_grades": vendor_grades
    })

def enrich_polygons_with_stats(gdf_polygons, name_col, df_v_filtered, df_o_filtered, df_o_all_for_city):
    """
    Enriches a polygon GeoDataFrame with vendor and user statistics.
    Args:
        gdf_polygons (gpd.GeoDataFrame): The polygons to enrich.
        name_col (str): The name of the unique identifier column in the polygon GDF.
        df_v_filtered (pd.DataFrame): Pre-filtered vendor data.
        df_o_filtered (pd.DataFrame): Pre-filtered order data (by date, bl, etc.).
        df_o_all_for_city (pd.DataFrame): Order data filtered only by city (for total user count).
    Returns:
        gpd.GeoDataFrame: The enriched GeoDataFrame.
    """
    if gdf_polygons is None or gdf_polygons.empty:
        return gpd.GeoDataFrame()
    enriched_gdf = gdf_polygons.copy()
    # --- 1. Vendor Enrichment ---
    if not df_v_filtered.empty and not df_v_filtered.dropna(subset=['latitude', 'longitude']).empty:
        gdf_v_filtered_for_enrich = gpd.GeoDataFrame(
            df_v_filtered.dropna(subset=['latitude', 'longitude']),
            geometry=gpd.points_from_xy(df_v_filtered.longitude, df_v_filtered.latitude),
            crs="EPSG:4326"
        )
        joined_vendors = gpd.sjoin(gdf_v_filtered_for_enrich, enriched_gdf, how="inner", predicate="within")
        # Total vendor count
        vendor_counts = joined_vendors.groupby(name_col).size().rename('vendor_count')
        enriched_gdf = enriched_gdf.merge(vendor_counts, how='left', left_on=name_col, right_index=True)
        # Vendor count by grade
        if 'grade' in joined_vendors.columns:
            grade_counts_series = joined_vendors.groupby([name_col, 'grade'], observed=True).size().unstack(fill_value=0)
            grade_counts_dict = grade_counts_series.apply(lambda row: {k: v for k, v in row.items() if v > 0}, axis=1).to_dict()
            enriched_gdf['grade_counts'] = enriched_gdf[name_col].astype(str).map(grade_counts_dict)
        else:
             enriched_gdf['grade_counts'] = None
    else:
        enriched_gdf['vendor_count'] = 0
        enriched_gdf['grade_counts'] = None
    
    enriched_gdf['vendor_count'] = enriched_gdf['vendor_count'].fillna(0).astype(int)
    # --- 2. Unique User Enrichment (Date-Ranged & Total) ---
    has_user_id = 'user_id' in df_o_all_for_city.columns
    if has_user_id:
        # A) Date-Ranged Unique Users
        if not df_o_filtered.empty and not df_o_filtered.dropna(subset=['customer_latitude', 'customer_longitude']).empty:
            gdf_orders_filtered = gpd.GeoDataFrame(
                df_o_filtered.dropna(subset=['customer_latitude', 'customer_longitude']),
                geometry=gpd.points_from_xy(df_o_filtered.customer_longitude, df_o_filtered.customer_latitude),
                crs="EPSG:4326"
            )
            joined_orders_filtered = gpd.sjoin(gdf_orders_filtered, enriched_gdf, how="inner", predicate="within")
            user_counts_filtered = joined_orders_filtered.groupby(name_col, observed=True)['user_id'].nunique().rename('unique_user_count')
            enriched_gdf = enriched_gdf.merge(user_counts_filtered, how='left', left_on=name_col, right_index=True)
        
        # B) Total (All-Time) Unique Users for the city
        if not df_o_all_for_city.empty and not df_o_all_for_city.dropna(subset=['customer_latitude', 'customer_longitude']).empty:
            gdf_orders_all = gpd.GeoDataFrame(
                df_o_all_for_city.dropna(subset=['customer_latitude', 'customer_longitude']),
                geometry=gpd.points_from_xy(df_o_all_for_city.customer_longitude, df_o_all_for_city.customer_latitude),
                crs="EPSG:4326"
            )
            joined_orders_all = gpd.sjoin(gdf_orders_all, enriched_gdf, how="inner", predicate="within")
            user_counts_all = joined_orders_all.groupby(name_col, observed=True)['user_id'].nunique().rename('total_unique_user_count')
            enriched_gdf = enriched_gdf.merge(user_counts_all, how='left', left_on=name_col, right_index=True)
    enriched_gdf['unique_user_count'] = enriched_gdf.get('unique_user_count', pd.Series(0, index=enriched_gdf.index)).fillna(0).astype(int)
    enriched_gdf['total_unique_user_count'] = enriched_gdf.get('total_unique_user_count', pd.Series(0, index=enriched_gdf.index)).fillna(0).astype(int)
    
    # --- 3. Population-based Metrics (if Pop data exists) ---
    if 'Pop' in enriched_gdf.columns:
        enriched_gdf['Pop'] = pd.to_numeric(enriched_gdf['Pop'], errors='coerce').fillna(0)
        enriched_gdf['vendor_per_10k_pop'] = enriched_gdf.apply(
            lambda row: (row['vendor_count'] / row['Pop']) * 10000 if row['Pop'] > 0 else 0, axis=1
        )
    if 'PopDensity' in enriched_gdf.columns:
        enriched_gdf['PopDensity'] = pd.to_numeric(enriched_gdf['PopDensity'], errors='coerce').fillna(0)
        
    return enriched_gdf

@app.route('/api/map-data', methods=['GET'])
def get_map_data():
    if df_orders is None or df_vendors is None:
        return jsonify({"error": "Server data not loaded"}), 500
    try:
        # Start timing
        request_start = time.time()
        
        # --- Parsing of filters ---
        city_name = request.args.get('city', default="tehran", type=str)
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        start_date = pd.to_datetime(start_date_str) if start_date_str else None
        end_date = pd.to_datetime(end_date_str).replace(hour=23, minute=59, second=59) if end_date_str else None
        selected_business_lines = [bl.strip() for bl in request.args.getlist('business_lines') if bl.strip()]
        vendor_codes_input_str = request.args.get('vendor_codes_filter', default="", type=str)
        selected_vendor_codes_for_vendors_only = [code.strip() for code in re.split(r'[\s,;\n]+', vendor_codes_input_str) if code.strip()]
        selected_vendor_status_ids = [int(s.strip()) for s in request.args.getlist('vendor_status_ids') if s.strip().isdigit()]
        selected_vendor_grades = [g.strip() for g in request.args.getlist('vendor_grades') if g.strip()]
        vendor_visible_str = request.args.get('vendor_visible', default="any", type=str)
        vendor_is_open_str = request.args.get('vendor_is_open', default="any", type=str)
        vendor_area_main_type = request.args.get('vendor_area_main_type', default="all", type=str)
        selected_vendor_area_sub_types = [s.strip() for s in request.args.getlist('vendor_area_sub_type') if s.strip()]
        heatmap_type_req = request.args.get('heatmap_type_request', default="none", type=str)
        area_type_display = request.args.get('area_type_display', default="tapsifood_marketing_areas", type=str)
        selected_polygon_sub_types = [s.strip() for s in request.args.getlist('area_sub_type_filter') if s.strip()]
        
        # New parameters for radius modifier
        radius_modifier = request.args.get('radius_modifier', default=1.0, type=float)
        radius_mode = request.args.get('radius_mode', default='percentage', type=str)
        radius_fixed = request.args.get('radius_fixed', default=3.0, type=float)
        
        heatmap_data = []
        vendor_markers = []
        polygons_geojson = {"type": "FeatureCollection", "features": []}
        coverage_grid_data = []
        
        # --- Vendor filtering logic ---
        df_v_filtered = df_vendors.copy()
        
        # Check if required columns exist
        required_vendor_columns = ['latitude', 'longitude', 'vendor_code']
        missing_columns = [col for col in required_vendor_columns if col not in df_v_filtered.columns]
        if missing_columns:
            print(f"Warning: Missing vendor columns: {missing_columns}")
            vendor_markers = []
        else:
            # Apply radius modifier based on mode
            if 'radius' in df_v_filtered.columns and 'original_radius' in df_v_filtered.columns:
                if radius_mode == 'fixed':
                    # Set all radii to the fixed value
                    df_v_filtered['radius'] = radius_fixed
                else:
                    # Apply percentage modifier
                    df_v_filtered['radius'] = df_v_filtered['original_radius'] * radius_modifier
            
            # Only filter if columns exist
            df_v_filtered = df_v_filtered.dropna(subset=['latitude', 'longitude', 'vendor_code'])
            
            if city_name != "all" and 'city_name' in df_v_filtered.columns: 
                df_v_filtered = df_v_filtered[df_v_filtered['city_name'] == city_name]
            
            if selected_vendor_codes_for_vendors_only and 'vendor_code' in df_v_filtered.columns: 
                df_v_filtered = df_v_filtered[df_v_filtered['vendor_code'].astype(str).isin(selected_vendor_codes_for_vendors_only)]
            
            if vendor_area_main_type != 'all' and selected_vendor_area_sub_types and not df_v_filtered.empty:
                if vendor_area_main_type == 'tapsifood_marketing_areas' and not df_orders.empty and 'marketing_area' in df_orders.columns:
                    temp_orders_ma = df_orders[df_orders['marketing_area'].isin(selected_vendor_area_sub_types)]
                    relevant_vendor_codes_ma = temp_orders_ma['vendor_code'].astype(str).dropna().unique()
                    df_v_filtered = df_v_filtered[df_v_filtered['vendor_code'].isin(relevant_vendor_codes_ma)]
                elif city_name == 'tehran':
                    target_gdf = None
                    if vendor_area_main_type == 'tehran_region_districts': target_gdf = gdf_tehran_region.copy() if gdf_tehran_region is not None else None
                    elif vendor_area_main_type == 'tehran_main_districts': target_gdf = gdf_tehran_main_districts.copy() if gdf_tehran_main_districts is not None else None
                    if target_gdf is not None and not target_gdf.empty:
                        name_col = next((col for col in ['Name', 'NAME_MAHAL'] if col in target_gdf.columns), None)
                        if name_col: target_gdf = target_gdf[target_gdf[name_col].isin(selected_vendor_area_sub_types)]
                        if not target_gdf.empty:
                            gdf_vendors_to_filter = gpd.GeoDataFrame(df_v_filtered, geometry=gpd.points_from_xy(df_v_filtered.longitude, df_v_filtered.latitude), crs="EPSG:4326")
                            vendors_in_area = gpd.sjoin(gdf_vendors_to_filter, target_gdf, how="inner", predicate="within")
                            codes_in_area = vendors_in_area['vendor_code'].unique() if not vendors_in_area.empty else []
                            df_v_filtered = df_v_filtered[df_v_filtered['vendor_code'].isin(codes_in_area)]
            
            if not df_v_filtered.empty and selected_business_lines and not df_orders.empty and 'business_line' in df_orders.columns:
                temp_orders = df_orders[df_orders['business_line'].isin(selected_business_lines)]
                relevant_vendor_codes = temp_orders['vendor_code'].astype(str).dropna().unique()
                df_v_filtered = df_v_filtered[df_v_filtered['vendor_code'].isin(relevant_vendor_codes)]
            
            if not df_v_filtered.empty:
                if selected_vendor_status_ids and 'status_id' in df_v_filtered.columns: 
                    df_v_filtered = df_v_filtered[df_v_filtered['status_id'].isin(selected_vendor_status_ids)]
                if selected_vendor_grades and 'grade' in df_v_filtered.columns: 
                    df_v_filtered = df_v_filtered[df_v_filtered['grade'].isin(selected_vendor_grades)]
                if vendor_visible_str != "any" and 'visible' in df_v_filtered.columns: 
                    df_v_filtered = df_v_filtered[df_v_filtered['visible'] == int(vendor_visible_str)]
                if vendor_is_open_str != "any" and 'open' in df_v_filtered.columns: 
                    df_v_filtered = df_v_filtered[df_v_filtered['open'] == int(vendor_is_open_str)]
            
            if not df_v_filtered.empty:
                # Convert to regular Python types for JSON serialization
                vendor_markers = df_v_filtered.replace({np.nan: None}).to_dict(orient='records')
            else:
                vendor_markers = []
        
        # --- Prepare filtered and total order dataframes for enrichment ---
        df_orders_filtered = df_orders.copy()
        if city_name != "all": df_orders_filtered = df_orders_filtered[df_orders_filtered['city_name'] == city_name]
        df_orders_all_for_city = df_orders_filtered.copy() # All orders in the city, before date/bl filters
        if start_date: df_orders_filtered = df_orders_filtered[df_orders_filtered['created_at'] >= start_date]
        if end_date: df_orders_filtered = df_orders_filtered[df_orders_filtered['created_at'] <= end_date]
        if selected_business_lines: df_orders_filtered = df_orders_filtered[df_orders_filtered['business_line'].isin(selected_business_lines)]
        
        # --- Heatmap generation with organic/non-organic order density ---
        if heatmap_type_req in ["order_density", "order_density_organic", "order_density_non_organic", "user_density"]:
            df_hm_source = df_orders_filtered.dropna(subset=['customer_latitude', 'customer_longitude'])
            
            if not df_hm_source.empty:
                if heatmap_type_req == "order_density":
                    # Total order density
                    df_hm_source['order_count'] = 1
                    df_aggregated = aggregate_heatmap_points(
                        df_hm_source, 
                        'customer_latitude', 
                        'customer_longitude', 
                        'order_count',
                        precision=4
                    )
                elif heatmap_type_req == "order_density_organic":
                    # Organic orders only
                    if 'organic' in df_hm_source.columns:
                        df_organic = df_hm_source[df_hm_source['organic'] == 1]
                        if not df_organic.empty:
                            df_organic['order_count'] = 1
                            df_aggregated = aggregate_heatmap_points(
                                df_organic,
                                'customer_latitude',
                                'customer_longitude',
                                'order_count',
                                precision=4
                            )
                        else:
                            df_aggregated = pd.DataFrame(columns=['lat', 'lng', 'value'])
                    else:
                        df_aggregated = pd.DataFrame(columns=['lat', 'lng', 'value'])
                elif heatmap_type_req == "order_density_non_organic":
                    # Non-organic orders only
                    if 'organic' in df_hm_source.columns:
                        df_non_organic = df_hm_source[df_hm_source['organic'] == 0]
                        if not df_non_organic.empty:
                            df_non_organic['order_count'] = 1
                            df_aggregated = aggregate_heatmap_points(
                                df_non_organic,
                                'customer_latitude',
                                'customer_longitude',
                                'order_count',
                                precision=4
                            )
                        else:
                            df_aggregated = pd.DataFrame(columns=['lat', 'lng', 'value'])
                    else:
                        df_aggregated = pd.DataFrame(columns=['lat', 'lng', 'value'])
                elif heatmap_type_req == "user_density":
                    # User density heatmap
                    if 'user_id' in df_hm_source.columns:
                        df_aggregated = aggregate_user_heatmap_points(
                            df_hm_source,
                            'customer_latitude',
                            'customer_longitude',
                            'user_id',
                            precision=4
                        )
                    else:
                        df_aggregated = pd.DataFrame(columns=['lat', 'lng', 'value'])
                
                # Normalize the counts to 0-100 range for consistency
                if not df_aggregated.empty and heatmap_type_req != "user_density":
                    max_count = df_aggregated['value'].max()
                    min_count = df_aggregated['value'].min()
                    if max_count > min_count:
                        df_aggregated['value'] = ((df_aggregated['value'] - min_count) / (max_count - min_count)) * 100
                    else:
                        df_aggregated['value'] = 50
                    
                heatmap_data = df_aggregated.to_dict(orient='records')
        
        # Population heatmap remains the same
        elif heatmap_type_req == "population" and city_name == "tehran":
            print("Generating population heatmap...")
            gdf_pop_source = None
            if area_type_display == "tehran_main_districts" and gdf_tehran_main_districts is not None:
                gdf_pop_source = gdf_tehran_main_districts
            elif area_type_display == "tehran_region_districts" and gdf_tehran_region is not None:
                 gdf_pop_source = gdf_tehran_region
            elif area_type_display == "all_tehran_districts":
                 if gdf_tehran_main_districts is not None:
                    gdf_pop_source = gdf_tehran_main_districts
            
            if gdf_pop_source is not None and 'Pop' in gdf_pop_source.columns:
                if selected_polygon_sub_types:
                    name_cols_poly = ['Name', 'NAME_MAHAL']
                    actual_name_col = next((col for col in name_cols_poly if col in gdf_pop_source.columns), None)
                    if actual_name_col:
                        gdf_pop_source = gdf_pop_source[gdf_pop_source[actual_name_col].isin(selected_polygon_sub_types)]
                
                point_density_divisor = 1000 
                temp_points = []
                for _, row in gdf_pop_source.iterrows():
                    population = pd.to_numeric(row['Pop'], errors='coerce')
                    if pd.notna(population) and population > 0:
                        num_points = int(population / point_density_divisor)
                        if num_points > 0:
                            generated_points = generate_random_points_in_polygon(row['geometry'], num_points)
                            for point in generated_points:
                                temp_points.append({'lat': point.y, 'lng': point.x, 'value': 1})
                heatmap_data = temp_points
                print(f"Generated {len(heatmap_data)} points for population heatmap.")
        
        # --- Coverage Grid Generation ---
        if area_type_display == "coverage_grid":
            print(f"Generating coverage grid for {city_name}")
            
            # Create cache key based on filters
            vendor_codes_for_cache = []
            if not df_v_filtered.empty and 'vendor_code' in df_v_filtered.columns:
                vendor_codes_for_cache = sorted(df_v_filtered['vendor_code'].tolist())
            
            cache_key = hashlib.md5(json.dumps({
                'city': city_name,
                'vendor_codes': vendor_codes_for_cache,
                'radius_modifier': radius_modifier,
                'radius_mode': radius_mode,
                'radius_fixed': radius_fixed
            }, sort_keys=True).encode()).hexdigest()
            
            # Check cache first
            if cache_key in coverage_cache:
                print(f"Using cached coverage grid")
                coverage_grid_data = coverage_cache[cache_key]
            else:
                grid_points = generate_coverage_grid(city_name)
                print(f"Generated {len(grid_points)} grid points")
                
                # Use vectorized calculation for better performance
                coverage_results = calculate_coverage_for_grid_vectorized(grid_points, df_v_filtered, city_name)
                
                # Find marketing areas for grid points
                marketing_areas = find_marketing_area_for_points(grid_points, city_name)
                
                # Combine coverage data with marketing area info
                coverage_grid_data = []
                for i, coverage in enumerate(coverage_results):
                    if coverage['total_vendors'] > 0:  # Only include points with coverage
                        coverage_grid_data.append({
                            'lat': coverage['lat'],
                            'lng': coverage['lng'],
                            'coverage': coverage,
                            'marketing_area': marketing_areas[i]
                        })
                
                # Cache the result (limit cache size)
                if len(coverage_cache) > CACHE_SIZE:
                    # Remove oldest entries
                    coverage_cache.clear()
                coverage_cache[cache_key] = coverage_grid_data
                
                print(f"Filtered to {len(coverage_grid_data)} coverage points with vendors")
        
        # --- Centralized Polygon Display & Enrichment Logic ---
        final_polygons_gdf = None
        if area_type_display != "none" and area_type_display != "coverage_grid":
            gdf_to_enrich = None
            name_col_to_use = None
            
            if area_type_display == "tapsifood_marketing_areas" and city_name in gdf_marketing_areas:
                gdf_to_enrich = gdf_marketing_areas[city_name]
                name_col_to_use = 'name'
            elif city_name == "tehran":
                if area_type_display == "tehran_region_districts" and gdf_tehran_region is not None:
                    gdf_to_enrich = gdf_tehran_region
                    name_col_to_use = 'Name'
                elif area_type_display == "tehran_main_districts" and gdf_tehran_main_districts is not None:
                    gdf_to_enrich = gdf_tehran_main_districts
                    name_col_to_use = 'NAME_MAHAL'
            
            if gdf_to_enrich is not None and not gdf_to_enrich.empty and name_col_to_use is not None:
                final_polygons_gdf = enrich_polygons_with_stats(
                    gdf_polygons=gdf_to_enrich,
                    name_col=name_col_to_use,
                    df_v_filtered=df_v_filtered,
                    df_o_filtered=df_orders_filtered,
                    df_o_all_for_city=df_orders_all_for_city
                )
            elif area_type_display == "all_tehran_districts" and city_name == 'tehran':
                enriched_list = []
                if gdf_tehran_region is not None:
                    enriched_list.append(enrich_polygons_with_stats(gdf_tehran_region, 'Name', df_v_filtered, df_orders_filtered, df_orders_all_for_city))
                if gdf_tehran_main_districts is not None:
                    enriched_list.append(enrich_polygons_with_stats(gdf_tehran_main_districts, 'NAME_MAHAL', df_v_filtered, df_orders_filtered, df_orders_all_for_city))
                if enriched_list:
                    final_polygons_gdf = pd.concat(enriched_list, ignore_index=True)
            
            if final_polygons_gdf is not None and not final_polygons_gdf.empty:
                if selected_polygon_sub_types:
                    name_cols_poly = ['name', 'Name', 'NAME_MAHAL']
                    actual_name_col = next((col for col in name_cols_poly if col in final_polygons_gdf.columns), None)
                    if actual_name_col:
                        final_polygons_gdf = final_polygons_gdf[final_polygons_gdf[actual_name_col].astype(str).isin(selected_polygon_sub_types)]
                if not final_polygons_gdf.empty:
                    clean_gdf = final_polygons_gdf.copy()
                    for col in clean_gdf.columns:
                        if col == 'geometry': continue
                        if pd.api.types.is_numeric_dtype(clean_gdf[col].dtype):
                            clean_gdf[col] = clean_gdf[col].astype(object).where(pd.notna(clean_gdf[col]), None)
                        else:
                            clean_gdf[col] = clean_gdf[col].astype(object).where(pd.notna(clean_gdf[col]), None)
                    polygons_geojson = clean_gdf.__geo_interface__
        
        # Log request time
        request_time = time.time() - request_start
        print(f"Request processed in {request_time:.2f}s")
        
        response_data = {
            "vendors": vendor_markers, 
            "heatmap_data": heatmap_data, 
            "polygons": polygons_geojson,
            "coverage_grid": coverage_grid_data,
            "processing_time": request_time
        }
        
        # Use ujson if available for faster JSON serialization
        try:
            import ujson
            return app.response_class(
                ujson.dumps(response_data),
                mimetype='application/json'
            )
        except ImportError:
            return jsonify(response_data)
            
    except Exception as e:
        import traceback
        print(f"Error in /api/map-data: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# --- Function to Open Browser ---
def open_browser():
    """Opens the web browser to the app's URL after a short delay."""
    time.sleep(1)
    webbrowser.open_new("http://127.0.0.1:5001/")

# --- Main Execution ---
if __name__ == '__main__':
    load_data()
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
         threading.Thread(target=open_browser).start()
    app.run(debug=True, port=5001, use_reloader=True)