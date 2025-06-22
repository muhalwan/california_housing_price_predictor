import pickle
from pathlib import Path
from flask import Flask, render_template, request, flash
import numpy as np

app = Flask(__name__)
app.secret_key = 'supersecretkey'

MODEL_PATH = Path("models") / "xgb_model.pkl"
SCALER_PATH = Path("models") / "scaler.pkl"

FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
    "Population", "AveOccup", "Latitude", "Longitude"
]

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError as e:
    raise RuntimeError(f"Could not load model or scaler. Please run train.py first. Error: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    if request.method == 'POST':
        try:
            input_data = [float(request.form[key]) for key in FEATURE_NAMES]
            
            data_scaled = scaler.transform([input_data])

            raw_prediction = model.predict(data_scaled)[0]
            final_prediction = raw_prediction * 100000

            prediction_text = f"${final_prediction:,.2f}"
            
        except (ValueError, KeyError) as e:
            flash(f"Invalid input. Please ensure all fields are filled with numbers. Error: {e}", "danger")
        except Exception as e:
            flash(f"An unexpected error occurred: {e}", "danger")
            
    return render_template('index.html', feature_names=FEATURE_NAMES, prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
