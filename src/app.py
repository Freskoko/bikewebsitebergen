import json
import os

import plotly
from flask import Flask, render_template, request

from utils.plotting import create_map_fig, create_station_graph

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def biketraffic():
    location = "Gågaten"
    if request.method == "POST":
        location = request.form["location"]

    if location != "":
        fig = create_map_fig(base_station_name=location)
        fig2 = create_station_graph(base_station_name=location)
    else:
        fig = create_map_fig()
        fig2 = create_station_graph()

    graph1JSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template(
        "biketraffic.html",
        graph1JSON=graph1JSON,
        graph2JSON=graph2JSON,
        location=location,
    )


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT"))
