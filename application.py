import os
import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Base directory of this file (works both locally and on AWS)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Flask app
application = Flask(__name__)
app = application

# Load trained model + scaler from models/ folder
with open(os.path.join(BASE_DIR, "models", "ridge.pkl"), "rb") as f:
    ridge_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "models", "scaler.pkl"), "rb") as f:
    standard_scaler = pickle.load(f)


@app.route("/")
def index():
    """Landing page."""
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        # Get form values
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        # 9 features only – in the same order as used during training
        features = np.array(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        )

        # Scale + predict
        new_data_scaled = standard_scaler.transform(features)
        result = ridge_model.predict(new_data_scaled)

        # Pass the prediction to template
        return render_template("home.html", results=result[0])
    else:
        # GET request – just show empty form
        return render_template("home.html")


if __name__ == "__main__":
    # For local development
    app.run(debug=True)
