from flask import Flask, render_template, request, flash
import numpy as np
import joblib
from pathlib import Path

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = 'your_very_secret_key'

# --- Feature Names ---
FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]


# --- Load Model and Scaler Locally ---
def load_model_locally():
    model_path = Path("models/xgb_model.joblib")
    scaler_path = Path("models/scaler.joblib")

    if not model_path.exists() or not scaler_path.exists():
        print("❌ Error: Model or scaler file not found. Please run train.py first.")
        return None, None

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("✅ Model and scaler loaded successfully from local files.")
        return model, scaler
    except Exception as e:
        print(f"❌ Error loading local model: {e}")
        return None, None


model, scaler = load_model_locally()


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    if request.method == 'POST':
        if not model or not scaler:
            flash("Model is not available. Please run the training script and restart the server.", "danger")
            return render_template('index.html', feature_names=FEATURE_NAMES, prediction=None)

        try:
            input_data = [float(request.form[key]) for key in FEATURE_NAMES]

            data_scaled = scaler.transform([input_data])

            raw_prediction = model.predict(data_scaled)[0]

            final_prediction = raw_prediction * 100000
            prediction_text = f"${final_prediction:,.2f}"

        except (ValueError, KeyError):
            flash("Invalid input. Please ensure all fields are filled with valid numbers.", "warning")
        except Exception as e:
            flash(f"An unexpected error occurred: {e}", "danger")

    return render_template('index.html', feature_names=FEATURE_NAMES, prediction=prediction_text)


if __name__ == '__main__':
    app.run(debug=True, port=5001)