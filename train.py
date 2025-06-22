import pickle
from pathlib import Path
import numpy as np
import optuna
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, DMatrix

# Load data
X, y = fetch_california_housing(return_X_y=True)

# Split data
X_train, X_val_test, y_train, y_val_test = train_test_split(
    X, y, test_size=0.4, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_val_test, y_val_test, test_size=0.5, random_state=42, shuffle=True
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# 1. Define the objective function for Optuna
def objective(trial):
    """Objective function for Optuna hyperparameter tuning."""
    params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'reg:squarederror',
        'random_state': 42,
        'early_stopping_rounds': 50,
        'n_estimators': trial.suggest_int('n_estimators', 500, 2500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }
    
    model = XGBRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

# 2. Create a study object and optimize
print("Starting hyperparameter tuning with Optuna...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, timeout=600)

print("\nHyperparameter tuning finished.")
print("Best trial:")
best_trial = study.best_trial
print(f"  Value (RMSE): {best_trial.value:.4f}")
print("  Best hyperparameters: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# 3. Train the final model with the best hyperparameters
print("\nTraining the final model with the best hyperparameters...")
best_params = best_trial.params
best_params.update({
    'tree_method': 'hist',
    'device': 'cuda',
    'objective': 'reg:squarederror',
    'random_state': 42,
})

# Combine train and validation sets for final training
X_train_full = np.concatenate((X_train, X_val), axis=0)
y_train_full = np.concatenate((y_train, y_val), axis=0)

model = XGBRegressor(**best_params)
model.fit(X_train_full, y_train_full, verbose=False)
print("Final model training finished.")


# Predict on the test set using the Booster object for GPU efficiency
booster = model.get_booster()
dtest = DMatrix(X_test, feature_names=[f'feature_{i}' for i in range(X_test.shape[1])])
y_pred = booster.predict(dtest)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nTest-set metrics with optimized model:")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R2  : {r2:.4f}\n")

# Save model and scaler
output_dir = Path("models")
output_dir.mkdir(exist_ok=True)
with open(output_dir / "xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(output_dir / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Saved optimized model to xgb_model.pkl and scaler to scaler.pkl.")