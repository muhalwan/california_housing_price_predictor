# California House Price Predictor

This project provides a high-performance machine learning model and a modern web application to predict median house values for districts in California. It leverages an optimized XGBoost model tuned with Optuna and a user-friendly interface built with Flask and Bootstrap.

 <!-- Placeholder: Replace with a real screenshot URL -->

---

## Key Features

- **High-Performance Model**: Utilizes an XGBoost Regressor, a powerful gradient boosting library, for accurate predictions.
- **Hyperparameter Tuning**: Employs Optuna to automate the search for the best model hyperparameters, achieving an **R² score of 0.862**.
- **Overfitting Prevention**: Implements early stopping to prevent the model from overfitting the training data.
- **Modern Web UI**: A clean, responsive, and user-friendly web interface built with Flask and Bootstrap.
- **Robust Backend**: Includes error handling and input validation for a stable user experience.
- **Reproducible Workflow**: The entire training and optimization process is captured in a single, easy-to-run script.

## Directory Structure

```
California_House_Price_Predictor/
│
├── app.py              # Flask application script
├── train.py            # Script to train, tune, and save the model
├── requirements.txt    # Project dependencies
├── .gitignore          # Files to be ignored by Git
│
├── models/             # Contains the saved model and scaler
│   ├── xgb_model.pkl
│   └── scaler.pkl
│
└── templates/
    └── index.html      # HTML template for the Flask app
```

## Getting Started

Follow these steps to get the project running on your local machine.

### Prerequisites

- Python 3.8+

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/muhalwan/california_housing_price_predictor.git
    cd california_housing_price_predictor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv .venv
    .venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Train the Model (Optional):**
    If you want to run the training and hyperparameter tuning process yourself, execute the `train.py` script. This will find the best model and save it to the `models/` directory.
    ```bash
    python train.py
    ```

2.  **Launch the Web Application:**
    Start the Flask server to launch the web interface.
    ```bash
    python app.py
    ```

3.  **Access the App:**
    Open your web browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Usage

- Fill in the property details in the web form.
- Click the "Predict Price" button.
- The estimated median house value for the district will be displayed.