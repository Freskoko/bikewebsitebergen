from flask import Flask, render_template, request
import os
import json
import plotly

from utils.plotting import create_map_fig
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def biketraffic():
    location = ""
    if request.method == "POST":
        location = request.form['location']
        
    if location != "":
        fig = create_map_fig(base_station_name=location)
    else:
        fig = create_map_fig()

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("biketraffic.html", graphJSON=graphJSON)


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT"))
