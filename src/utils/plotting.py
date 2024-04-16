import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BASE_WIDTH = 1980 / 1.8
BASE_HEIGHT = 1200 / 1.8


def create_map_fig(
    base_station_name="Bergen jernbanestasjon",
    size_width=BASE_WIDTH,
    size_height=BASE_HEIGHT,
):
    fig = go.Figure()

    all_stations_df = pd.read_csv("src/data/out/stations_df.csv")
    final_df = pd.read_csv("src/data/out/stations_trips_df.csv")

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

            # markers at stations
            fig.add_trace(
                go.Scattermapbox(
                    mode="markers",
                    lon=[all_stations_df.iloc[i]["station_longitude"]],
                    lat=[all_stations_df.iloc[i]["station_latitude"]],
                    showlegend=False,
                    marker=go.scattermapbox.Marker(size=8),
                    hovertemplate=f"{target_station_name}  trips here: {num_trips}",
                )
            )

    # base station
    fig.add_trace(
        go.Scattermapbox(
            mode="markers",
            lon=[
                all_stations_df[all_stations_df["station_name"] == base_station_name][
                    "station_longitude"
                ].values[0]
            ],
            lat=[
                all_stations_df[all_stations_df["station_name"] == base_station_name][
                    "station_latitude"
                ].values[0]
            ],
            showlegend=False,
            marker=go.scattermapbox.Marker(size=10, color="red"),
            hovertemplate=f"{base_station_name}",
        )
    )

    fig.update_layout(
        autosize=False,  # autosize off
        width=size_width,
        height=size_height,
        margin={"l": 0, "t": 0, "b": 0, "r": 0},
        mapbox={
            "uirevision": "constant",
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


def create_station_graph(
    base_station_name="Bergen jernbanestasjon",
    size_width=BASE_WIDTH,
    size_height=BASE_HEIGHT,
):

    fig = go.Figure()

    # all_stations_df = pd.read_csv("data/out/stations_df.csv")
    # final_df = pd.read_csv("src/data/out/stations_trips_df.csv")
    final_df = pd.read_csv("src/data/out/stations_trips_df.csv")

    station_df = final_df.loc[final_df["start_station_name"] == base_station_name]
    station_df = station_df.drop(columns="avg_duration")

    df = station_df.transpose()

    # rename the columns to 'visits'
    df.columns = ["visits"]

    print(df)
    df = df.drop(index="start_station_name")
    df["visits"] = df["visits"].astype(int)

    # sort by 'visits' in descending order and take the top 5
    df = df.sort_values("visits", ascending=False).head(5)

    print(df)

    fig = px.bar(df, x=df.index, y="visits", height=400)
    fig.update_layout(
        autosize=False, width=size_width, height=size_height  # autosize off
    )

    return fig


if __name__ == "__main__":
    create_station_graph()
