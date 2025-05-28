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
    return mo, pd, plt, sns, train_test_split


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
def _(pd):
    df = pd.read_csv("/home/henrik/project/bikewebsitebergen/src/data/04.csv")
    print(df)
    return (df,)


@app.cell
def _(RANDOM_STATE, df, train_test_split):
    train, val_test = train_test_split(df, train_size = 70, random_state = RANDOM_STATE)
    validation, test = train_test_split(val_test, train_size = 50, random_state = RANDOM_STATE)
    return (train,)


@app.cell
def _(train):
    sub_df = train[["duration", "start_station_id", "start_station_name"]]
    return (sub_df,)


@app.cell
def _(mo, sub_df):
    mo.md(f"""sub_df = {sub_df}""")
    return


@app.cell
def _(sub_df):
    longest = sub_df[sub_df["start_station_id"] == 807]["duration"].max()
    return (longest,)


@app.cell
def _(longest, mo):
    mo.md(f"""longest = {longest}""")
    return


@app.cell
def _(sub_df):

    #ids = [807,816,640,132,817]
    ids = sub_df["start_station_id"]
    sub_df_smaller = sub_df[sub_df["start_station_id"].isin(ids)]
    return (sub_df_smaller,)


@app.cell
def _(mo, sub_df_smaller):
    mo.md(f"""sub_df_smaller = {sub_df_smaller}""")
    return


@app.cell
def _(pd, sub_df_smaller):
    sub_df_cleaned = sub_df_smaller.copy()
    sub_df_cleaned["duration"] = pd.to_numeric(sub_df_cleaned["duration"])/60
    return (sub_df_cleaned,)


@app.cell
def _(mo, sub_df_cleaned):
    mo.md(rf"""sub_df_cleaned = {sub_df_cleaned}""")
    return


@app.cell
def _(plt, sns, sub_df_cleaned):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=sub_df_cleaned, x="start_station_name", y="duration", showfliers=False)

    plt.xlabel("Start Station ID")
    plt.ylabel("Duration (minutes)")
    plt.title("Boxplot of Duration by Start Station ID")
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
