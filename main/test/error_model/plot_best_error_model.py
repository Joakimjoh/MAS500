import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error
from xgboost import XGBRegressor
from scipy.spatial.distance import cdist

# === Load and clean data ===
df = pd.read_csv("/home/student/Documents/MAS500/test/error_model/raw_test_data.csv")
df["x (m)"] = df["x (m)"].apply(lambda s: float(s.strip("[]")))
df["y (m)"] = df["y (m)"].apply(lambda s: float(s.strip("[]")))
df["z (m)"] = df["z (m)"].apply(lambda s: float(s.strip("[]")))

X = df[["x (m)", "y (m)"]].values
y = df["z (m)"].values
results = []

# === Model 3: XGB + Poly2 + RBF ===
scaler3 = StandardScaler().fit(X)
X_scaled3 = scaler3.transform(X)
poly3 = PolynomialFeatures(degree=2, include_bias=False)
X_poly3 = poly3.fit_transform(X_scaled3)

sigma = 0.3

xq = np.quantile(X_scaled3[:, 0], [0.01, 0.99])
yq = np.quantile(X_scaled3[:, 1], [0.01, 0.99])
x_centers = np.linspace(xq[0], xq[1], 7)
y_centers = np.linspace(yq[0], yq[1], 7)
xc, yc = np.meshgrid(x_centers, y_centers)
rbf_centers3 = np.vstack([xc.ravel(), yc.ravel()]).T
dist3 = cdist(X_scaled3, rbf_centers3, "sqeuclidean")
X_rbf3 = np.exp(-dist3 / (2 * sigma ** 2))
X_combined3 = np.hstack([X_poly3, X_rbf3])

scaler_y3 = StandardScaler().fit(y.reshape(-1, 1))
y_norm3 = scaler_y3.transform(y.reshape(-1, 1)).ravel()

start_train = time.time()
m3 = XGBRegressor(n_estimators=30, learning_rate=0.16, max_depth=8,
                  subsample=0.8, colsample_bytree=0.8, random_state=42)
m3.fit(X_combined3, y_norm3)
train_time = time.time() - start_train

start_pred = time.time()
y_pred = scaler_y3.inverse_transform(m3.predict(X_combined3).reshape(-1, 1)).ravel()
pred_time = time.time() - start_pred

results.append({
    "Model": "XGB + Poly2 + RBF",
    "Train Time (s)": train_time,
    "Pred Time (s)": pred_time,
    "MAE": mean_absolute_error(y, y_pred),
    "RMSE": mean_squared_error(y, y_pred, squared=False),
    "R2": r2_score(y, y_pred),
    "Max Error": max_error(y, y_pred)
})

# === Print Results ===
df_results = pd.DataFrame(results)
print("\nModel Comparison Results:")
print(df_results.to_string(index=False))

# === Visualize prediction at py = 120 ===
df_py = df[df["py"] == 120].copy()
X_py = df_py[["x (m)", "y (m)"]].values
y_actual_py = np.abs(df_py["z (m)"].values)
x_vals_py = df_py["x (m)"].values

X_py_scaled = scaler3.transform(X_py)
X_py_poly = poly3.transform(X_py_scaled)
dist_py = cdist(X_py_scaled, rbf_centers3, "sqeuclidean")
X_py_rbf = np.exp(-dist_py / (2 * sigma ** 2))
X_py_combined = np.hstack([X_py_poly, X_py_rbf])
y_pred_py = np.abs(scaler_y3.inverse_transform(m3.predict(X_py_combined).reshape(-1, 1)).ravel())

plt.figure(figsize=(12, 7))
plt.plot(x_vals_py, y_actual_py, label="Actual Z", color="black", linewidth=4)
plt.plot(x_vals_py, y_pred_py, label="XGB + Poly2 + RBF", linestyle='--', linewidth=4)
plt.xlabel("x (m)", fontsize=18)
plt.ylabel("z (m)", fontsize=18)
plt.title("Absolute Z Prediction Comparison at py = 120", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(False)  # Disable grid
plt.tight_layout()
plt.show()

# === Visualize prediction at px = 160 ===
df_px = df[df["px"] == 160].copy()
X_px = df_px[["x (m)", "y (m)"]].values
y_actual_px = np.abs(df_px["z (m)"].values)
y_vals_px = df_px["y (m)"].values

X_px_scaled = scaler3.transform(X_px)
X_px_poly = poly3.transform(X_px_scaled)
dist_px = cdist(X_px_scaled, rbf_centers3, "sqeuclidean")
X_px_rbf = np.exp(-dist_px / (2 * sigma ** 2))
X_px_combined = np.hstack([X_px_poly, X_px_rbf])
y_pred_px = np.abs(scaler_y3.inverse_transform(m3.predict(X_px_combined).reshape(-1, 1)).ravel())

plt.figure(figsize=(12, 7))
plt.plot(y_vals_px, y_actual_px, label="Actual Z", color="black", linewidth=4)
plt.plot(y_vals_px, y_pred_px, label="XGB + Poly2 + RBF", linestyle='--', linewidth=4)
plt.xlabel("y (m)", fontsize=18)
plt.ylabel("z (m)", fontsize=18)
plt.title("Absolute Z Prediction Comparison at px = 160", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(False)  # Disable grid
plt.tight_layout()
plt.show()