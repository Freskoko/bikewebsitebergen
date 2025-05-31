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
    import numpy as np
    from sklearn.impute import SimpleImputer
    return SimpleImputer, mo, np, os, pd, plt, sns, train_test_split


@app.cell
def _():
    RANDOM_STATE = 99
    return (RANDOM_STATE,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Ideas
    
        Use more data in hopes of generalizing
        """
    )
    return


@app.cell
def _(os, pd):
    # load all csv to one big df

    data_dir = "/home/henrik/project/bikewebsitebergen/src/data/cycle"
    file_list = os.listdir(data_dir)
    all_dfs = []

    for file_name in file_list:
        if file_name.endswith(".csv"):
            file_path = os.path.join(data_dir, file_name)

            total_df = pd.read_csv(file_path)
            all_dfs.append(total_df)

    df = pd.concat(all_dfs)
    return (df,)


@app.cell
def _(df, pd):
    # get station info
    start_stations = df[['start_station_id', 'start_station_name', 'start_station_description', 'start_station_latitude', 'start_station_longitude']].copy()
    start_stations.columns = ['station_id', 'station_name', 'station_description', 'station_latitude', 'station_longitude']

    # Combine and get unique stations
    all_stations = pd.concat([start_stations], ignore_index=True)
    station_static_data = all_stations.drop_duplicates(subset=['station_id']).reset_index(drop=True)
    station_static_data_dict = station_static_data.set_index('station_name').to_dict('index')

    station_static_data
    return (station_static_data_dict,)


@app.cell
def _(RANDOM_STATE, df, train_test_split):
    train, temp = train_test_split(
        df,
        train_size=0.8,
        random_state=RANDOM_STATE,
        stratify=df["end_station_name"]
    )

    validation, test = train_test_split(
        temp,
        train_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp["end_station_name"]
    )
    return test, train, validation


@app.cell
def _(mo, train):
    longest = train[train["start_station_id"] == 807]["duration"].max()
    mo.md(f"""longest = {longest}""")
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
    mo.md(r"Lets look at a specific station")
    return


@app.cell
def _(train):
    specific_start_station_name = "Torgallmenningen"
    train_specific_station = train[train["start_station_name"] == specific_start_station_name]
    return specific_start_station_name, train_specific_station


@app.cell
def _(train_specific_station):
    train_specific_value_count_end = train_specific_station["end_station_name"].value_counts(ascending=False)
    train_specific_value_count_end
    return (train_specific_value_count_end,)


@app.cell
def _(plt, sns, train_specific_value_count_end):
    plt.figure(figsize=(12, 6))
    sns.barplot(train_specific_value_count_end)

    plt.xlabel("Station name")
    plt.ylabel("Trips ended at station")
    plt.title("Where trips end after starting at the 'Torgallmenningen' station")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"Seems like this station has a few very popular end points, lets look closer.")
    return


@app.cell
def _(plt, sns, train_specific_value_count_end):
    plt.figure(figsize=(12, 6))
    sns.barplot(train_specific_value_count_end.head(6))

    plt.xlabel("Station name")
    plt.ylabel("Trips ended at station")
    plt.title("Where trips end after starting at the 'Torgallmenningen' station (top 10)")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"Interesting, let's use a map to visualize this better.")
    return


@app.cell
def _(train_specific_station):
    train_specific_station.head(n=1)["end_station_name"]
    return


@app.cell
def _(
    specific_start_station_name,
    station_static_data_dict,
    train_specific_station,
):
    import utils.mapping
    import importlib

    importlib.reload(utils.mapping)

    from utils.mapping import map_bergen_sentrum_with_points

    start = (
        station_static_data_dict[specific_start_station_name]["station_longitude"],
        station_static_data_dict[specific_start_station_name]["station_latitude"],
        specific_start_station_name
    )
    end_list = [
        (
            station_static_data_dict[name]["station_longitude"],
            station_static_data_dict[name]["station_latitude"],
            name
        )
        for name in train_specific_station.head(n=6)["end_station_name"]
    ]

    points = {
        "start": start,
        "end_list": end_list
    }
    map_bergen_sentrum_with_points(points=points)
    return


@app.cell
def _(mo):
    mo.md(r"Interesting! Seems like people often travel to Nordnes, or sandviken from this stop.")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Lets try to predict the end stop of a given trip!
        Lets first try just for the single stop, "Torgallmenningen"
        """
    )
    return


@app.cell
def _(specific_start_station_name, test, train, validation):
    specific_start_station_name

    train_torg = train[train["start_station_name"] == specific_start_station_name]
    val_torg = validation[validation["start_station_name"] == specific_start_station_name]
    test_torg = test[test["start_station_name"] == specific_start_station_name]

    train_torg
    return test_torg, train_torg, val_torg


@app.cell
def _(mo):
    mo.md(r"Certain information cant be used for prediction, since it directly relates to a stop, or is info about the starting stop (which is the same for all trips)")
    return


@app.cell
def _(test_torg, train_torg, val_torg):
    to_drop = ["start_station_id", "start_station_description", "start_station_latitude", "start_station_longitude"] # start station
    to_drop += ["end_station_id", "end_station_description", "end_station_latitude", "end_station_longitude"] # end station
    to_drop += ["start_station_name"] # same for all

    train_torg_c1 = train_torg.drop(to_drop,axis=1)
    val_torg_c1 = val_torg.drop(to_drop,axis=1)
    test_torg_c1 = test_torg.drop(to_drop,axis=1)

    train_torg_c1
    return test_torg_c1, train_torg_c1, val_torg_c1


@app.cell
def _(test_torg_c1, train_torg_c1, val_torg_c1):
    Y_train_torg = train_torg_c1["end_station_name"]
    Y_val_torg = val_torg_c1["end_station_name"]
    Y_test_torg = test_torg_c1["end_station_name"]

    X_train_torg = train_torg_c1.drop("end_station_name", axis = 1)
    X_val_torg = val_torg_c1.drop("end_station_name", axis = 1)
    X_test_torg = test_torg_c1.drop("end_station_name", axis = 1)
    return (
        X_test_torg,
        X_train_torg,
        X_val_torg,
        Y_test_torg,
        Y_train_torg,
        Y_val_torg,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        Lets do some feature engineering
    
        Firstly, lets change the date cols to be usable
        """
    )
    return


@app.cell
def _(X_test_torg, X_train_torg, X_val_torg, pd):
    def transform_date_cols(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        date_cols = [
            "started_at","ended_at"
        ]
        for col in date_cols:
            df_copy[col] = pd.to_datetime(df_copy[col], format='ISO8601')
            df_copy[f'{col}_year'] = df_copy[col].dt.year
            df_copy[f'{col}_month'] = df_copy[col].dt.month
            df_copy[f'{col}_day'] = df_copy[col].dt.day
            df_copy[f'{col}_hour'] = df_copy[col].dt.hour

        df_copy = df_copy.drop(date_cols, axis = 1)

        return df_copy

    X_train_torg_dated = transform_date_cols(X_train_torg)
    X_val_torg_dated = transform_date_cols(X_val_torg)
    X_test_torg_dated = transform_date_cols(X_test_torg)

    X_train_torg_dated
    return X_test_torg_dated, X_train_torg_dated, X_val_torg_dated


@app.cell
def _(np, os, pd):
    # load all csv to one big df

    data_dir_weather = "/home/henrik/project/bikewebsitebergen/src/data/weather"
    all_weather_dfs = []

    for file_name_weather in os.listdir(data_dir_weather):
        if file_name_weather.endswith(".csv"):
            file_path_weather = os.path.join(data_dir_weather, file_name_weather)
            all_weather_dfs.append(pd.read_csv(file_path_weather))

    weather_df = pd.concat(all_weather_dfs)

    weather_df['Lufttemperatur'] = weather_df['Lufttemperatur'].where(weather_df['Lufttemperatur'] <= 50, np.nan)

    weather_df['Relativ luftfuktighet'] = weather_df['Relativ luftfuktighet'].where(weather_df['Relativ luftfuktighet'] <= 1000, np.nan)

    weather_df = weather_df.drop("Solskinstid",axis=1)
    weather_df
    return (weather_df,)


@app.cell
def _(X_test_torg_dated, X_train_torg_dated, X_val_torg_dated, pd, weather_df):
    def add_weather_cols(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        weather_df_copy = weather_df.copy()

        df_copy['merge_hour'] = pd.to_datetime(
            df_copy['started_at_year'].astype(str) + '-' +
            df_copy['started_at_month'].astype(str).str.zfill(2) + '-' +
            df_copy['started_at_day'].astype(str).str.zfill(2) + ' ' +
            df_copy['started_at_hour'].astype(str).str.zfill(2) + ':00:00'
        )

        weather_df_copy['merge_hour'] = pd.to_datetime(
            weather_df_copy['Dato'] + ' ' + weather_df_copy['Tid']
        ).dt.floor('h')  # Floor to the hour

        weather_df_copy = weather_df_copy.drop_duplicates(subset='merge_hour')

        merged = pd.merge(
            df_copy,
            weather_df_copy[['merge_hour', 'Lufttemperatur', 'Relativ luftfuktighet', 'Vindstyrke', 'Vindkast']],
            how='left',
            on='merge_hour'
        )

        merged = merged.rename(columns={
            'Lufttemperatur': 'temperature',
            'Relativ luftfuktighet': 'humidity',
            'Vindstyrke': 'wind_speed',
            'Vindkast': 'wind_gust'
        })

        merged = merged.drop('merge_hour', axis=1)

        return merged

    X_train_torg_weather = add_weather_cols(X_train_torg_dated, weather_df)
    X_val_torg_weather = add_weather_cols(X_val_torg_dated, weather_df)
    X_test_torg_weather = add_weather_cols(X_test_torg_dated, weather_df)

    X_train_torg_weather
    return X_test_torg_weather, X_train_torg_weather, X_val_torg_weather


@app.cell
def _(mo):
    mo.md(r"Lets make sure the datatypes are correct")
    return


@app.cell
def _(X_train_torg_weather):
    X_train_torg_weather.info()
    return


@app.cell
def _(Y_test_torg, Y_train_torg, Y_val_torg, np):
    from sklearn.preprocessing import LabelEncoder

    # maps unseen to custom
    class RobustLabelEncoder(LabelEncoder):
        def transform(self, y):
            try:
                return super().transform(y)
            except ValueError:
                return np.array([self.classes_.tolist().index(x) if x in self.classes_ else -1 for x in y])

    le = RobustLabelEncoder()
    Y_train_torg_encoded = le.fit_transform(Y_train_torg)
    Y_val_torg_encoded = le.transform(Y_val_torg)
    Y_test_torg_encoded = le.transform(Y_test_torg)
    return Y_train_torg_encoded, Y_val_torg_encoded


@app.cell
def _(mo):
    mo.md(r"Lets handle any nan values")
    return


@app.cell
def _(X_train_torg_weather):
    X_train_torg_weather.isna().sum()
    return


@app.cell
def _(mo):
    mo.md(r"Seems like some data is missing, lets inpute this data")
    return


@app.cell
def _(
    SimpleImputer,
    X_test_torg_weather,
    X_train_torg_weather,
    X_val_torg_weather,
    np,
):
    # these have nan
    cols_to_impute = ['temperature', 'humidity', 'wind_speed', 'wind_gust']

    X_train_imputed = X_train_torg_weather.copy()
    X_val_imputed = X_val_torg_weather.copy()
    X_test_imputed = X_test_torg_weather.copy()

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    X_train_imputed[cols_to_impute] = imp.fit_transform(X_train_imputed[cols_to_impute])
    X_val_imputed[cols_to_impute] = imp.transform(X_val_imputed[cols_to_impute])
    X_test_imputed[cols_to_impute] = imp.transform(X_test_imputed[cols_to_impute])

    X_test_imputed.isna().sum().sum()
    return X_train_imputed, X_val_imputed


@app.cell
def _(mo):
    mo.md(
        r"""
        Looks good, all missing data has been imputed.
    
        Now we are ready to predict.
        """
    )
    return


@app.cell
def _(Y_test_torg, Y_train_torg, Y_val_torg):
    # First, let's properly inspect our raw labels before encoding
    print("Unique station names in training:", len(Y_train_torg.unique()))
    print("Unique station names in validation:", len(Y_val_torg.unique()))
    print("Unique station names in test:", len(Y_test_torg.unique()))

    # Find stations that are in validation but not in training
    unseen_in_val = set(Y_val_torg.unique()) - set(Y_train_torg.unique())
    print("\nStations in validation but not training:", unseen_in_val)

    # Find stations that are in test but not in training
    unseen_in_test = set(Y_test_torg.unique()) - set(Y_train_torg.unique())
    print("Stations in test but not training:", unseen_in_test)
    return


@app.cell
def _(
    X_train_imputed,
    X_val_imputed,
    Y_train_torg_encoded,
    Y_val_torg_encoded,
):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.dummy import DummyClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import balanced_accuracy_score,confusion_matrix
    from sklearn.neighbors import KNeighborsClassifier

    # smote?

    models = {
        'Dummy': {
            'model': DummyClassifier(strategy='stratified'),
            'params': {}
        },
        'SVC': {
            'model': SVC(class_weight='balanced'),
            'params': {
                'model__C': [0.1, 1, 10],
                'model__kernel': ['linear', 'rbf', 'poly'],
                'model__gamma': ['scale', 'auto']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(class_weight='balanced'),
            'params': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'model__n_neighbors': [3, 5, 10, 15],
                'model__weights': ['uniform', 'distance']
            }
        }
    }

    results_train = {}
    results_val = {}
    confusion_matrices = {}

    for name, cfg in models.items():
        print(f"Training {name}...")

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', cfg['model'])
        ])

        params = None
        if cfg['params']:
            grid = GridSearchCV(pipe, cfg['params'], cv=3, scoring='balanced_accuracy', n_jobs=-1)
            grid.fit(X_train_imputed, Y_train_torg_encoded)
            best_model = grid.best_estimator_
            params = grid.best_estimator_
        else:
            best_model = pipe.fit(X_train_imputed, Y_train_torg_encoded)

        y_pred_train = best_model.predict(X_train_imputed)
        results_train[name] = (balanced_accuracy_score(Y_train_torg_encoded, y_pred_train), params)

        y_pred_val = best_model.predict(X_val_imputed)
        results_val[name] = (balanced_accuracy_score(Y_val_torg_encoded, y_pred_val), params)

        confusion_matrices[name] = confusion_matrix(Y_val_torg_encoded, y_pred_val)

    return confusion_matrices, results_train, results_val


@app.cell
def _(confusion_matrices, plt, sns):
    for name_2, cm_2 in confusion_matrices.items():
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_2, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name_2}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    return


@app.cell
def _(results_train):
    results_train
    return


@app.cell
def _(results_val):
    results_val
    return


if __name__ == "__main__":
    app.run()
