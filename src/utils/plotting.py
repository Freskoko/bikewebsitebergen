import pandas as pd
import plotly.graph_objects as go


def create_map_fig(
    base_station_name="Bergen jernbanestasjon", size_width=1980, size_height=1200
):
    fig = go.Figure()

    all_stations_df = pd.read_csv("data/out/stations_df.csv")
    final_df = pd.read_csv("data/out/stations_trips_df.csv")

    # filter trips to other stations from the base station
    trips_from_base = final_df[final_df["start_station_name"] == base_station_name]

    # loop through all other stations
    for i in range(len(all_stations_df)):
        if (
            all_stations_df.iloc[i]["station_name"] != base_station_name
        ):  # if the station is not the base station
            target_station_name = all_stations_df.iloc[i]["station_name"]
            # num of trips
            num_trips = (
                trips_from_base[target_station_name].values[0]
                if target_station_name in trips_from_base.columns
                else 0
            )

            fig.add_trace(
                go.Scattermapbox(
                    mode="lines",
                    showlegend=False,
                    lon=[
                        all_stations_df[
                            all_stations_df["station_name"] == base_station_name
                        ]["station_longitude"].values[0],
                        all_stations_df.iloc[i]["station_longitude"],
                    ],
                    lat=[
                        all_stations_df[
                            all_stations_df["station_name"] == base_station_name
                        ]["station_latitude"].values[0],
                        all_stations_df.iloc[i]["station_latitude"],
                    ],
                )
            )

            # line text (todo fix)
            fig.add_trace(
                go.Scattermapbox(
                    mode="markers",
                    lon=[
                        (
                            all_stations_df[
                                all_stations_df["station_name"] == base_station_name
                            ]["station_longitude"].values[0]
                            + all_stations_df.iloc[i]["station_longitude"]
                        )
                        / 2
                    ],
                    lat=[
                        (
                            all_stations_df[
                                all_stations_df["station_name"] == base_station_name
                            ]["station_latitude"].values[0]
                            + all_stations_df.iloc[i]["station_latitude"]
                        )
                        / 2
                    ],
                    showlegend=False,
                    marker=go.scattermapbox.Marker(size=0),
                    hovertemplate=f"Trips from {base_station_name} to {target_station_name}: {num_trips}",
                )
            )

    fig.add_trace(
        go.Scattermapbox(
            mode="markers",
            lon=all_stations_df["station_longitude"],
            lat=all_stations_df["station_latitude"],
            marker=go.scattermapbox.Marker(size=8),
            text=all_stations_df["station_name"],
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        autosize=False,  # autosize off
        width=size_width,
        height=size_height,
        margin={"l": 0, "t": 0, "b": 0, "r": 0},
        mapbox={
            "style": "open-street-map",
            "center": {
                "lon": all_stations_df[
                    all_stations_df["station_name"] == base_station_name
                ]["station_longitude"].values[0],
                "lat": all_stations_df[
                    all_stations_df["station_name"] == base_station_name
                ]["station_latitude"].values[0],
            },
            "zoom": 9.8,
        },
    )
    return fig
