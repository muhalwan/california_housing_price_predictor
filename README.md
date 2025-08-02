# 🏡 California House Price Predictor

This project provides a high-performance machine learning model and a web application to predict median house values for districts in California. It leverages an optimized **XGBoost model** tuned with Optuna and a user-friendly interface built with Flask.

## ✨ Key Features

-   **High-Performance Model**: Utilizes an XGBoost Regressor for accurate predictions.
-   **Automated Hyperparameter Tuning**: Employs **Optuna** to find the best model hyperparameters, achieving an impressive **R² score of 0.8629**.
-   **Local First**: The model is trained and saved locally, ready for manual upload to any model hosting service.
-   **Web UI**: A clean and responsive web interface built with Flask allows for easy interaction with the model.
-   **Reproducible**: The entire training workflow is captured in an easy-to-run script.

## 📈 Model Performance & Results

The final model was trained with optimized hyperparameters and evaluated on a held-out test set, yielding excellent results:

| Metric                          | Score  |
| ------------------------------- | ------ |
| **Test Set R-squared (R²)** | 0.8629 |
| **Test Set MSE** | 0.1880 |
| **Test Set RMSE** | 0.4336 |
| **Test Set MAE** | 0.2794 |

### Optimal Hyperparameters

The following hyperparameters were identified by Optuna as the best combination for this task:

-   `n_estimators`: **2548**
-   `learning_rate`: **0.0294**
-   `max_depth`: **7**
-   `subsample`: **0.9336**
-   `colsample_bytree`: **0.6983**
-   `min_child_weight`: **1**

## 🚀 Getting Started

Follow these instructions to get the project running on your local machine.

### Prerequisites

-   Python 3.10 or 3.11
-   A virtual environment tool (`venv`)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/muhalwan/california_housing_price_predictor.git](https://github.com/muhalwan/california_housing_price_predictor.git)
    cd california_housing_price_predictor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Train the Model:**

    This script trains the model, tunes it with Optuna, and saves the final `xgb_model.joblib` and `scaler.joblib` files into a `models/` directory.
    ```bash
    python train.py
    ```

2.  **Launch the Web Application:**

    This will start a local Flask web server that uses the model files you just created.
    ```bash
    python app.py
    ```

3.  **Access the App:**
    Open your web browser and navigate to **`http://127.0.0.1:5001`**. You can now input new data to get a house price prediction!
