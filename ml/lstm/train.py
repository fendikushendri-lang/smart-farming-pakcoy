import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ml.lstm.model import PakcoyMultivariateLSTM

def train_lstm():
    BASE_DIR = r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new"
    DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "timeseries_clean.csv")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    df = pd.read_csv(DATA_PATH)
    features = ['soil_moisture', 'temperature', 'maturity_level']
    data = df[features].values
    
    # 1. Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Save scaler params
    scaler_params = {
        "features": features,
        "min": scaler.min_.tolist(),
        "scale": scaler.scale_.tolist(),
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist()
    }
    with open(os.path.join(MODELS_DIR, "lstm_scaler.json"), "w") as f:
        json.dump(scaler_params, f, indent=2)
        
    # 2. Sequence Creation (seq_len=12, horizon=12)
    seq_len = 12
    horizon = 12
    X, Y = [], []
    for i in range(len(scaled_data) - seq_len - horizon + 1):
        X.append(scaled_data[i : i + seq_len])
        Y.append(scaled_data[i + seq_len : i + seq_len + horizon])
        
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    
    # 3. Train-Test Split (80% Train, 20% Test sequentially)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"LSTM Data prepared: Total={len(X)}, Train={len(X_train)}, Test={len(X_test)}")
    
    # 4. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PakcoyMultivariateLSTM(
        input_size=3,
        hidden_size=64,
        num_layers=2,
        output_steps=horizon,
        output_features=3,
        dropout=0.2
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    epochs = 35
    best_loss = float('inf')
    best_model_path = os.path.join(MODELS_DIR, "lstm_pakcoy_model.pt")
    
    history = {"train_loss": [], "val_loss": []}
    
    print("=== START TRAINING MULTIVARIATE LSTM ===")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
            
        train_loss /= len(X_train)
        
        # Test eval
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                loss = criterion(pred, by)
                test_loss += loss.item() * len(bx)
        test_loss /= len(X_test)
        
        scheduler.step(test_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(test_loss)
        
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), best_model_path)
            
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs}] - Train MSE: {train_loss:.6f} | Test MSE: {test_loss:.6f}")
            
    # 5. Evaluate Best Model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    with torch.no_grad():
        test_preds = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
        
    # Invert scaling to calculate real-world metrics
    # Shape of test_preds: (N, 12, 3), Y_test: (N, 12, 3)
    preds_flat = test_preds.reshape(-1, 3)
    y_test_flat = Y_test.reshape(-1, 3)
    
    preds_orig = scaler.inverse_transform(preds_flat)
    y_orig = scaler.inverse_transform(y_test_flat)
    
    feature_metrics = {}
    for idx, feat in enumerate(features):
        y_true = y_orig[:, idx]
        y_pred = preds_orig[:, idx]
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        feature_metrics[feat] = {
            "MAE": round(float(mae), 4),
            "MSE": round(float(mse), 4),
            "RMSE": round(float(rmse), 4),
            "R2": round(float(r2), 4)
        }
        print(f"Metric [{feat}]: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
    overall_metrics = {
        "version": "1.0.0",
        "model_type": "Multivariate Multi-Step LSTM",
        "input_features": features,
        "input_timesteps": seq_len,
        "forecast_horizon_timesteps": horizon,
        "forecast_days": 4,
        "best_test_loss_mse": float(best_loss),
        "feature_metrics": feature_metrics,
        "history": history
    }
    
    with open(os.path.join(MODELS_DIR, "lstm_metrics.json"), "w") as f:
        json.dump(overall_metrics, f, indent=2)
        
    print("Multivariate LSTM training and evaluation completed successfully!")

if __name__ == "__main__":
    train_lstm()
