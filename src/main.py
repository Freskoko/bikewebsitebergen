import marimo

__generated_with = "0.13.14"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    import os
    return mo, os, pd, plt, sns, train_test_split


@app.cell
def _():
    RANDOM_STATE = 42
    return (RANDOM_STATE,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Ideas
    
        variation in trip length per station
    
        predict length?
        """
    )
    return


@app.cell
def _(os, pd):
    # load all csv to one big df

    data_dir = "/home/henrik/project/bikewebsitebergen/src/data/"
    file_list = os.listdir(data_dir)
    all_dfs = []

    for file_name in file_list:
        if file_name.endswith(".csv"):
            file_path = os.path.join(data_dir, file_name)

            total_df = pd.read_csv(file_path)
            all_dfs.append(total_df)

    df = pd.concat(all_dfs)
    print(df.head())
    return (df,)


@app.cell
def _(RANDOM_STATE, df, train_test_split):
    train, val_test = train_test_split(df, train_size = 0.7, random_state = RANDOM_STATE)
    validation, test = train_test_split(val_test, train_size = 0.5, random_state = RANDOM_STATE)
    return (train,)


@app.cell
def _(train):
    train.to_csv("train_data.csv")
    return


@app.cell
def _(mo, train):
    longest = train[train["start_station_id"] == 807]["duration"].max()
    mo.md(f"""longest = {longest}""")
    return


@app.cell
def _():

    #ids = [807,816,640,132,817]
    # ids = train["start_station_id"]
    # sub_df_smaller = train[train["start_station_id"].isin(ids)]
    return


@app.cell
def _(pd, train):
    train_minutes = train.copy()
    train_minutes["duration"] = pd.to_numeric(train_minutes["duration"])/60
    return (train_minutes,)


@app.cell
def _(plt, sns, train_minutes):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=train_minutes, x="start_station_name", y="duration", showfliers=False)

    plt.xlabel("Start Station Name")
    plt.ylabel("Duration (minutes)")
    plt.title("Boxplot of Duration by Start Station Name")
    # plt.tight_layout()
    plt.show()
    return


@app.cell
def _(plt, sns, train_minutes):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=train_minutes, x="end_station_name", y="duration", showfliers=False)

    plt.xlabel("End Station Name")
    plt.ylabel("Duration (minutes)")
    plt.title("Boxplot of Duration by End Station Name")
    # plt.tight_layout()
    plt.show()
    return


@app.cell
def _(train):
    # trips per station start
    trips_per_station_start = train["start_station_name"].value_counts(ascending=False)
    return (trips_per_station_start,)


@app.cell
def _(plt, sns, trips_per_station_start):
    plt.figure(figsize=(12, 6))
    sns.barplot(trips_per_station_start)

    plt.xlabel("Station name")
    plt.ylabel("Trips started here")
    plt.title("Station name vs amount of trips started there")
    plt.show()
    return


@app.cell
def _(plt, sns, trips_per_station_start):
    plt.figure(figsize=(12, 6))
    sns.barplot(trips_per_station_start.head())

    plt.xlabel("Station name")
    plt.ylabel("Trips started here")
    plt.title("Ten most common locations to start a trip")
    plt.show()
    return


@app.cell
def _(plt, sns, trips_per_station_start):
    plt.figure(figsize=(12, 6))
    sns.barplot(trips_per_station_start.tail())

    plt.xlabel("Station name")
    plt.ylabel("Trips started here")
    plt.title("Ten least common locations to start a trip")
    plt.show()
    return


@app.cell
def _(train):
    # trips per station end
    trips_per_station_end = train["end_station_name"].value_counts(ascending=False)
    return (trips_per_station_end,)


@app.cell
def _(plt, sns, trips_per_station_start):
    plt.figure(figsize=(12, 6))
    sns.barplot(trips_per_station_start)

    plt.xlabel("Station name")
    plt.ylabel("Trips ended here")
    plt.title("Station name vs amount of trips ended there")
    plt.show()
    return


@app.cell
def _(plt, sns, trips_per_station_end):
    plt.figure(figsize=(12, 6))
    sns.barplot(trips_per_station_end.head())

    plt.xlabel("Station name")
    plt.ylabel("Trips ended here")
    plt.title("Ten most common locations to end a trip")
    plt.show()
    return


@app.cell
def _(plt, sns, trips_per_station_end):
    plt.figure(figsize=(12, 6))
    sns.barplot(trips_per_station_end.tail())

    plt.xlabel("Station name")
    plt.ylabel("Trips ended here")
    plt.title("Ten least common locations to end a trip")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"# Lets look at a specific station")
    return


@app.cell
def _(train):
    train_specific_station = train[train["start_station_name"] == "Torgallmenningen"]
    return (train_specific_station,)


@app.cell
def _(train_specific_station):
    train_specific_value_count_end = train_specific_station["end_station_name"].value_counts(ascending=False)
    return (train_specific_value_count_end,)


@app.cell
def _(plt, sns, train_specific_value_count_end):
    plt.figure(figsize=(12, 6))
    sns.barplot(train_specific_value_count_end)

    plt.xlabel("Station name")
    plt.ylabel("Trips ended here")
    plt.title("Station name vs amount of trips ended there")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
