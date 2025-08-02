import numpy as np
import optuna
import joblib
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def train_model():
    """Main function to train the model, tune it, and save it."""

    # Load data
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)

    # Split data
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, test_size=0.4, random_state=42, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.5, random_state=42, shuffle=True
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- 1. Optuna Objective Function ---
    def objective(trial):
        """Objective function for Optuna hyperparameter tuning."""
        params = {
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        }

        model = XGBRegressor(**params, early_stopping_rounds=50)

        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )

        preds = model.predict(X_val_scaled)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse

    # --- 2. Hyperparameter Tuning ---
    print("Starting hyperparameter tuning with Optuna...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, timeout=1200)

    print("\nHyperparameter tuning finished.")
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (RMSE): {best_trial.value:.4f}")
    print("  Best hyperparameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # --- 3. Final Model Training ---
    print("\nTraining the final model with the best hyperparameters...")
    best_params = best_trial.params
    best_params.update({
        'objective': 'reg:squarederror',
        'random_state': 42,
    })

    X_train_full_scaled = np.concatenate((X_train_scaled, X_val_scaled), axis=0)
    y_train_full = np.concatenate((y_train, y_val), axis=0)

    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train_full_scaled, y_train_full, verbose=False)
    print("✅ Final model training finished.")

    # --- 4. Evaluation ---
    y_pred = final_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Test-set metrics with optimized model:")
    print(f"   MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"    R²: {r2:.4f}\n")

    # --- 5. Save Model and Scaler---
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "xgb_model.joblib"
    scaler_path = output_dir / "scaler.joblib"

    joblib.dump(final_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print("\nTraining complete.")


if __name__ == '__main__':
    train_model()