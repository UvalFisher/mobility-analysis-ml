import itertools
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("trips_with_clusters.csv")
features = ["hour_bin", "origin_cluster", "destination_cluster", "TripDistance_km"]
target = "TripDuration_min"
cat_features = ["hour_bin", "origin_cluster", "destination_cluster"]

encoder = OneHotEncoder(sparse_output=False)
X_cat = encoder.fit_transform(df[cat_features])
X_num = df[["TripDistance_km"]].values
X = np.hstack((X_cat, X_num))

scaler_y = StandardScaler()
y = scaler_y.fit_transform(df[[target]].values).flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

hidden_sizes = [16, 32, 64, 128, 256]
learning_rates = [0.05, 0.01, 0.005, 0.001]
epoch_options = [50, 100, 200, 300]
results = []
best_model_state, best_model_params = None, None
best_test_loss = float("inf")

for lr, hs, epochs in itertools.product(learning_rates, hidden_sizes, epoch_options):
    start = time.time()
    model = Net(X.shape[1], hs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses, test_losses = [], []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test)
            test_losses.append(test_loss.item())

    with torch.no_grad():
        y_pred = scaler_y.inverse_transform(test_pred.numpy()).flatten()
        y_true = scaler_y.inverse_transform(y_test.numpy()).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    results.append({"LR": lr, "HiddenSize": hs, "Epochs": epochs,
                    "TestLoss": test_losses[-1], "MAE": mae, "RMSE": rmse,
                    "Runtime_sec": round(time.time() - start, 2),
                    "TrainLosses": train_losses, "TestLosses": test_losses})

    if test_losses[-1] < best_test_loss:
        best_test_loss = test_losses[-1]
        best_model_state = model.state_dict()
        best_model_params = {"input_size": X.shape[1], "hidden_size": hs}

results_df = pd.DataFrame(results).sort_values("TestLoss")
print(results_df[["LR", "HiddenSize", "Epochs", "TestLoss", "MAE", "RMSE", "Runtime_sec"]].head(10).to_string(index=False))

torch.save({"model_state_dict": best_model_state, "model_params": best_model_params}, "best_model.pth")

best = results_df.iloc[0]
plt.figure(figsize=(8, 4))
plt.plot(best["TrainLosses"], label="Train Loss")
plt.plot(best["TestLosses"], label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Best ANN Training Run")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
