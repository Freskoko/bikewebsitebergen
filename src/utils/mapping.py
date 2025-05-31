import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
import math


def map_bergen_with_points() -> None:
    # bergen
    bergen_lat_min, bergen_lat_max = 60.325, 60.45
    bergen_lon_min, bergen_lon_max = 5.25, 5.40

    # map
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(bergen_lon_min, bergen_lon_max)
    ax.set_ylim(bergen_lat_min, bergen_lat_max)

    # bergen base
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)

    # Formatting
    plt.title('Bergen, Norway', fontsize=16, fontweight='bold')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

    # Example of how to add your own points:
    print("To add your own data:")
    print("# your_gdf = gpd.GeoDataFrame(your_df, geometry=gpd.points_from_xy(your_df.longitude, your_df.latitude), crs='EPSG:4326')")
    print("# your_gdf.plot(ax=ax, color='red', markersize=50)")

def map_bergen_sentrum_with_points(points : dict = None) -> None:
    # bergen
    # bergen_lat_min, bergen_lat_max = 60.360, 60.420
    # bergen_lon_min, bergen_lon_max = 5.30, 5.36

    bergen_lat_min, bergen_lat_max = 60.375, 60.420
    bergen_lon_min, bergen_lon_max = 5.30, 5.35

    # aspect ratio
    lat_range = bergen_lat_max - bergen_lat_min
    lon_range = bergen_lon_max - bergen_lon_min
    center_lat = (bergen_lat_max + bergen_lat_min) / 2
    lat_correction = math.cos(math.radians(center_lat))
    aspect_ratio = (lon_range * lat_correction) / lat_range

    # create fig
    fig_height = 16
    fig_width = fig_height * aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # bounds
    ax.set_xlim(bergen_lon_min, bergen_lon_max)
    ax.set_ylim(bergen_lat_min, bergen_lat_max)

    # basemap
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron)

    # clean
    plt.title('Bergen, Norway', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    ax.set_aspect('equal')
    plt.tight_layout()

    if points:
        # start stop
        start_long, start_lat, start_name = points["start"]
        ax.scatter(start_long, start_lat, c="red", s=10, zorder=5)
        ax.text(start_long, start_lat, f'start ({start_name})', fontsize=7, ha='left', va='bottom', zorder=6)

        for stop_point in points["end_list"]:
            # end stop
            stop_long, stop_lat, stop_name = stop_point
            ax.scatter(stop_long, stop_lat, c="blue", s=10, zorder=5)
            ax.text(stop_long, stop_lat, f'stop ({stop_name})', fontsize=7, ha='left', va='bottom', zorder=6)

            # lines
            ax.plot([start_long, stop_long], [start_lat, stop_lat], color='black', alpha=0.5, linewidth=0.8, zorder=4)

    plt.show()


if __name__ == "__main__":
    points = {
        "start":(5.33, 60.39, "lake"),
        "end_list":[
            (5.31, 60.40, "edge"),
            (5.31, 60.41, "ocean")
        ]
    }
    map_bergen_sentrum_with_points()