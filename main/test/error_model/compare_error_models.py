import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error, max_error
)
from xgboost import XGBRegressor
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata

# === Load and clean data ===
df = pd.read_csv("/home/student/Documents/MAS500/test/error_model/raw_test_data.csv")
df["x (m)"] = df["x (m)"].apply(lambda s: float(s.strip("[]")))
df["y (m)"] = df["y (m)"].apply(lambda s: float(s.strip("[]")))
df["z (m)"] = df["z (m)"].apply(lambda s: float(s.strip("[]")))

X = df[["x (m)", "y (m)"]].values
y = df["z (m)"].values
results = []

# Common parameters
poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly2 = poly2.fit_transform(X)
scaler_std = StandardScaler().fit(X)
X_scaled = scaler_std.transform(X)
sigma = 0.3
xq = np.quantile(X_scaled[:, 0], [0.01, 0.99])
yq = np.quantile(X_scaled[:, 1], [0.01, 0.99])
x_centers = np.linspace(xq[0], xq[1], 7)
y_centers = np.linspace(yq[0], yq[1], 7)
xc, yc = np.meshgrid(x_centers, y_centers)
rbf_centers = np.vstack([xc.ravel(), yc.ravel()]).T
dist = cdist(X_scaled, rbf_centers, "sqeuclidean")
X_rbf = np.exp(-dist / (2 * sigma ** 2))

# === Model 1: Linear ===
start_train = time.time()
m1 = LinearRegression().fit(X, y)
train_time = time.time() - start_train
start_pred = time.time()
y_pred = m1.predict(X)
pred_time = time.time() - start_pred
results.append({
    "Model": "Linear",
    "Train Time (s)": train_time,
    "Pred Time (s)": pred_time,
    "MAE": mean_absolute_error(y, y_pred),
    "RMSE": mean_squared_error(y, y_pred, squared=False),
    "R2": r2_score(y, y_pred),
    "Max Error": max_error(y, y_pred)
})

# === Model 2: Poly2 + Linear ===
start_train = time.time()
m2 = LinearRegression().fit(X_poly2, y)
train_time = time.time() - start_train
start_pred = time.time()
y_pred = m2.predict(X_poly2)
pred_time = time.time() - start_pred
results.append({
    "Model": "Linear + Poly2",
    "Train Time (s)": train_time,
    "Pred Time (s)": pred_time,
    "MAE": mean_absolute_error(y, y_pred),
    "RMSE": mean_squared_error(y, y_pred, squared=False),
    "R2": r2_score(y, y_pred),
    "Max Error": max_error(y, y_pred)
})

# === Model 3: Linear + RBF ===
start_train = time.time()
m3 = LinearRegression().fit(X_rbf, y)
train_time = time.time() - start_train
start_pred = time.time()
y_pred = m3.predict(X_rbf)
pred_time = time.time() - start_pred
results.append({
    "Model": "Linear + RBF",
    "Train Time (s)": train_time,
    "Pred Time (s)": pred_time,
    "MAE": mean_absolute_error(y, y_pred),
    "RMSE": mean_squared_error(y, y_pred, squared=False),
    "R2": r2_score(y, y_pred),
    "Max Error": max_error(y, y_pred)
})

# === Model 7: XGB + Poly2 ===
X_poly2_scaled = poly2.transform(X_scaled)
scaler_y = StandardScaler().fit(y.reshape(-1, 1))
y_norm = scaler_y.transform(y.reshape(-1, 1)).ravel()
start_train = time.time()
m7 = XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8, random_state=42)
m7.fit(X_poly2_scaled, y_norm)
train_time = time.time() - start_train
start_pred = time.time()
y_pred = scaler_y.inverse_transform(m7.predict(X_poly2_scaled).reshape(-1, 1)).ravel()
pred_time = time.time() - start_pred
results.append({
    "Model": "XGB + Poly2",
    "Train Time (s)": train_time,
    "Pred Time (s)": pred_time,
    "MAE": mean_absolute_error(y, y_pred),
    "RMSE": mean_squared_error(y, y_pred, squared=False),
    "R2": r2_score(y, y_pred),
    "Max Error": max_error(y, y_pred)
})

# === Model 8: XGB + RBF ===
scaler_y_rbf = StandardScaler().fit(y.reshape(-1, 1))
y_norm_rbf = scaler_y_rbf.transform(y.reshape(-1, 1)).ravel()
start_train = time.time()
m8 = XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8, random_state=42)
m8.fit(X_rbf, y_norm_rbf)
train_time = time.time() - start_train
start_pred = time.time()
y_pred = scaler_y_rbf.inverse_transform(m8.predict(X_rbf).reshape(-1, 1)).ravel()
pred_time = time.time() - start_pred
results.append({
    "Model": "XGB + RBF",
    "Train Time (s)": train_time,
    "Pred Time (s)": pred_time,
    "MAE": mean_absolute_error(y, y_pred),
    "RMSE": mean_squared_error(y, y_pred, squared=False),
    "R2": r2_score(y, y_pred),
    "Max Error": max_error(y, y_pred)
})

# === Model 9: XGB + Poly2 + RBF ===
X_combined_scaled = np.hstack([poly2.transform(X_scaled), X_rbf])
scaler_y_combo = StandardScaler().fit(y.reshape(-1, 1))
y_norm_combo = scaler_y_combo.transform(y.reshape(-1, 1)).ravel()
start_train = time.time()
m9 = XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8, random_state=42)
m9.fit(X_combined_scaled, y_norm_combo)
train_time = time.time() - start_train
start_pred = time.time()
y_pred = scaler_y_combo.inverse_transform(m9.predict(X_combined_scaled).reshape(-1, 1)).ravel()
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

# === Model 10: Random Forest ===
start_train = time.time()
m10 = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
m10.fit(X, y)
train_time = time.time() - start_train
start_pred = time.time()
y_pred = m10.predict(X)
pred_time = time.time() - start_pred
results.append({
    "Model": "Random Forest",
    "Train Time (s)": train_time,
    "Pred Time (s)": pred_time,
    "MAE": mean_absolute_error(y, y_pred),
    "RMSE": mean_squared_error(y, y_pred, squared=False),
    "R2": r2_score(y, y_pred),
    "Max Error": max_error(y, y_pred)
})

# === Model 11: LightGBM ===
start_train = time.time()
m11 = LGBMRegressor(n_estimators=300, learning_rate=0.08, max_depth=6, random_state=42)
m11.fit(X, y)
train_time = time.time() - start_train
start_pred = time.time()
y_pred = m11.predict(X)
pred_time = time.time() - start_pred
results.append({
    "Model": "LightGBM",
    "Train Time (s)": train_time,
    "Pred Time (s)": pred_time,
    "MAE": mean_absolute_error(y, y_pred),
    "RMSE": mean_squared_error(y, y_pred, squared=False),
    "R2": r2_score(y, y_pred),
    "Max Error": max_error(y, y_pred)
})

# === Model 6: XGBoost (raw) ===
start_train = time.time()
m6 = XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8, random_state=42)
m6.fit(X, y)
train_time = time.time() - start_train
start_pred = time.time()
y_pred = m6.predict(X)
pred_time = time.time() - start_pred
results.append({
    "Model": "XGB",
    "Train Time (s)": train_time,
    "Pred Time (s)": pred_time,
    "MAE": mean_absolute_error(y, y_pred),
    "RMSE": mean_squared_error(y, y_pred, squared=False),
    "R2": r2_score(y, y_pred),
    "Max Error": max_error(y, y_pred)
})

# === Model 4: Linear + Poly2 + RBF ===
X_poly2_scaled = poly2.fit_transform(X_scaled)
X_combined = np.hstack([X_poly2_scaled, X_rbf])
start_train = time.time()
m4 = LinearRegression().fit(X_combined, y)
train_time = time.time() - start_train
start_pred = time.time()
y_pred = m4.predict(X_combined)
pred_time = time.time() - start_pred
results.append({
    "Model": "Linear + Poly2 + RBF",
    "Train Time (s)": train_time,
    "Pred Time (s)": pred_time,
    "MAE": mean_absolute_error(y, y_pred),
    "RMSE": mean_squared_error(y, y_pred, squared=False),
    "R2": r2_score(y, y_pred),
    "Max Error": max_error(y, y_pred)
})

# === Model 5: Ridge + Poly2 + Scaled ===
X_poly2_scaled = poly2.transform(X_scaled)
start_train = time.time()
m5 = Ridge(alpha=1.0).fit(X_poly2_scaled, y)
train_time = time.time() - start_train
start_pred = time.time()
y_pred = m5.predict(X_poly2_scaled)
pred_time = time.time() - start_pred
results.append({
    "Model": "Ridge + Poly2 + Scaled",
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

# === Save to CSV ===
df_results.to_csv("/home/student/Documents/MAS500/test/error_model/model_comparison_results.csv", index=False)

plt.figure(figsize=(10, 6))

# Multiply the MAE values by 1000 for scaling
df_results["MAE"] = df_results["MAE"] * 1000

# Plot the bar chart
plt.bar(df_results["Model"], df_results["MAE"], color='skyblue')

# Set font size for all text elements
plt.ylabel("Mean Absolute Error (mm)", fontsize=24)
plt.title("Model Comparison: MAE", fontsize=24)

# Increase text size for x-ticks, and rotate them to avoid overlap
plt.xticks(rotation=15, fontsize=24)

# Set text size for y-ticks (numbers on y-axis)
plt.yticks(fontsize=24)

# Adjust layout to avoid text overlap
plt.tight_layout()

plt.show()

# === Extract points where py == 120 ===
df_subset = df[df["py"] == 120].copy()
X_subset = df_subset[["x (m)", "y (m)"]].values
y_actual = np.abs(df_subset["z (m)"].values)
x_vals = df_subset["x (m)"].values

# Predict with all models in correct order
y_pred_linear = np.abs(m1.predict(X_subset))

X_poly2_sub = poly2.transform(X_subset)
y_pred_poly2 = np.abs(m2.predict(X_poly2_sub))

X_subset_scaled = scaler_std.transform(X_subset)
dist_subset_rbf = cdist(X_subset_scaled, rbf_centers, "sqeuclidean")
X_rbf_subset = np.exp(-dist_subset_rbf / (2 * sigma ** 2))
y_pred_rbf_linear = np.abs(m3.predict(X_rbf_subset))

X_poly2_sub_scaled = poly2.transform(X_subset_scaled)
X_combined_sub = np.hstack([X_poly2_sub_scaled, X_rbf_subset])
y_pred_poly2_rbf_linear = np.abs(m4.predict(X_combined_sub))

X_poly2_ridge_sub = poly2.transform(X_subset_scaled)
y_pred_ridge = np.abs(m5.predict(X_poly2_ridge_sub))

y_pred_xgb_raw = np.abs(m6.predict(X_subset))

X_poly2_sub_scaled = poly2.transform(X_subset_scaled)
y_pred_xgb_poly = np.abs(scaler_y.inverse_transform(m7.predict(X_poly2_sub_scaled).reshape(-1, 1)).ravel())

y_pred_xgb_rbf = np.abs(scaler_y_rbf.inverse_transform(m8.predict(X_rbf_subset).reshape(-1, 1)).ravel())

y_pred_xgb_poly_rbf = np.abs(scaler_y_combo.inverse_transform(m9.predict(X_combined_sub).reshape(-1, 1)).ravel())

y_pred_rf = np.abs(m10.predict(X_subset))
y_pred_lgb = np.abs(m11.predict(X_subset))

# === Plot all predictions at py = 120 ===
plt.figure(figsize=(12, 7))
plt.plot(x_vals, y_actual, label="Actual Z", color="black", linewidth=4)
plt.plot(x_vals, y_pred_linear, label="Linear", linestyle='--', linewidth=4)
plt.plot(x_vals, y_pred_poly2, label="Linear + Poly2", linestyle='--', linewidth=4)
plt.plot(x_vals, y_pred_rbf_linear, label="Linear + RBF", linestyle='--', linewidth=4)
plt.plot(x_vals, y_pred_poly2_rbf_linear, label="Linear + Poly2 + RBF", linestyle='--', linewidth=4)
plt.plot(x_vals, y_pred_ridge, label="Ridge + Poly2 + Scaled", linestyle='--', linewidth=4)
plt.plot(x_vals, y_pred_xgb_raw, label="XGBoost (raw)", linestyle='--', linewidth=4)
plt.plot(x_vals, y_pred_xgb_poly, label="XGB + Poly2", linestyle='--', linewidth=4)
plt.plot(x_vals, y_pred_xgb_rbf, label="XGB + RBF", linestyle='--', linewidth=4)
plt.plot(x_vals, y_pred_xgb_poly_rbf, label="XGB + Poly2 + RBF", linestyle='--', linewidth=4)
plt.plot(x_vals, y_pred_rf, label="Random Forest", linestyle='--', linewidth=4)
plt.plot(x_vals, y_pred_lgb, label="LightGBM", linestyle='--', linewidth=4)

# Set font size for all text elements
plt.xlabel("x (m)", fontsize=24)
plt.ylabel("|z| (m)", fontsize=24)
plt.title("Absolute Z Prediction Comparison at py = 120", fontsize=24)

# Adjust the font size for the legend (make it smaller)
plt.legend(fontsize=18)

# Remove the grid and set text size for ticks
plt.grid(False)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.tight_layout()
plt.show()