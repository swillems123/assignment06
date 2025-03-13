import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the diabetes dataset
def load_data():
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    feature_names = diabetes.feature_names
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, feature_names

# Preprocess the data
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Train models
def train_models(X_train, y_train):
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Random Forest Regression
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Support Vector Regression
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train)
    
    return lr_model, rf_model, svr_model

# Evaluate model performance
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return y_pred, {"model": model_name, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

# Compare model performance
def compare_models(results):
    models = [result["model"] for result in results]
    r2_scores = [result["r2"] for result in results]
    rmse_scores = [result["rmse"] for result in results]
    
    # Create comparison bar chart
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(models, r2_scores, color=['blue', 'green', 'red'])
    plt.title('R² Score Comparison (higher is better)')
    plt.ylim(0, max(r2_scores) + 0.1)
    
    plt.subplot(1, 2, 2)
    plt.bar(models, rmse_scores, color=['blue', 'green', 'red'])
    plt.title('RMSE Comparison (lower is better)')
    
    plt.tight_layout()
    plt.show()
    
    # Find the best model
    best_idx = r2_scores.index(max(r2_scores))
    print(f"\nBest performing model: {models[best_idx]}")
    print(f"R² Score: {r2_scores[best_idx]:.4f}")

# Plot feature importance for Random Forest
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

def main():
    # Load data
    X, y, feature_names = load_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Train models
    lr_model, rf_model, svr_model = train_models(X_train, y_train)
    
    # Evaluate models
    _, lr_results = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    _, rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    _, svr_results = evaluate_model(svr_model, X_test, y_test, "SVR")
    
    # Compare models
    results = [lr_results, rf_results, svr_results]
    compare_models(results)
    
    # Plot feature importance for Random Forest
    plot_feature_importance(rf_model, feature_names)
    
    # Plot actual vs predicted for best model (assuming Random Forest)
    y_pred_rf = rf_model.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_rf, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Random Forest: Actual vs Predicted')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()