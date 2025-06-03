import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.spatial.distance import cdist

# === Load and clean data ===
df = pd.read_csv("/home/student/Documents/MAS500/test/raw_test_data.csv")
df["x (m)"] = df["x (m)"].apply(lambda s: float(s.strip("[]")))
df["y (m)"] = df["y (m)"].apply(lambda s: float(s.strip("[]")))
df["z (m)"] = df["z (m)"].apply(lambda s: float(s.strip("[]")))

X = df[["x (m)", "y (m)"]].values
y = df["z (m)"].values
results = []

# === Model 1: XGB + Poly2 ===
scaler1 = StandardScaler().fit(X)
X_scaled1 = scaler1.transform(X)
poly1 = PolynomialFeatures(degree=2, include_bias=False)
X_poly1 = poly1.fit_transform(X_scaled1)
scaler_y1 = StandardScaler().fit(y.reshape(-1, 1))
y_norm1 = scaler_y1.transform(y.reshape(-1, 1)).ravel()

start_train = time.time()
m1 = XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8, random_state=42)
m1.fit(X_poly1, y_norm1)
train_time = time.time() - start_train

start_pred = time.time()
y_pred = scaler_y1.inverse_transform(m1.predict(X_poly1).reshape(-1, 1)).ravel()
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

# === Model 2: XGB + RBF ===
scaler2 = StandardScaler().fit(X)
X_scaled2 = scaler2.transform(X)
xq = np.quantile(X_scaled2[:, 0], [0.01, 0.99])
yq = np.quantile(X_scaled2[:, 1], [0.01, 0.99])
x_centers = np.linspace(xq[0], xq[1], 7)
y_centers = np.linspace(yq[0], yq[1], 7)
xc, yc = np.meshgrid(x_centers, y_centers)
rbf_centers2 = np.vstack([xc.ravel(), yc.ravel()]).T
dist2 = cdist(X_scaled2, rbf_centers2, "sqeuclidean")
sigma = 0.3
X_rbf2 = np.exp(-dist2 / (2 * sigma ** 2))

scaler_y2 = StandardScaler().fit(y.reshape(-1, 1))
y_norm2 = scaler_y2.transform(y.reshape(-1, 1)).ravel()

start_train = time.time()
m2 = XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8, random_state=42)
m2.fit(X_rbf2, y_norm2)
train_time = time.time() - start_train

start_pred = time.time()
y_pred = scaler_y2.inverse_transform(m2.predict(X_rbf2).reshape(-1, 1)).ravel()
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

# === Model 3: XGB + Poly2 + RBF ===
scaler3 = StandardScaler().fit(X)
X_scaled3 = scaler3.transform(X)
poly3 = PolynomialFeatures(degree=2, include_bias=False)
X_poly3 = poly3.fit_transform(X_scaled3)

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

# === Model 4: LightGBM ===
start_train = time.time()
m4 = LGBMRegressor(n_estimators=300, learning_rate=0.08, max_depth=6, random_state=42)
m4.fit(X, y)
train_time = time.time() - start_train

start_pred = time.time()
y_pred = m4.predict(X)
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

# === Print Results ===
df_results = pd.DataFrame(results)
print("\nModel Comparison Results:")
print(df_results.to_string(index=False))

# === Save to CSV ===
df_results.to_csv("/home/student/Documents/MAS500/test/model_comparison_results.csv", index=False)

# === Plot MAE Comparison ===
plt.figure(figsize=(10, 6))
plt.bar(df_results["Model"], df_results["MAE"], color='skyblue')
plt.ylabel("Mean Absolute Error")
plt.title("Model Comparison: MAE")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# === Visualize prediction at py = 120 ===
df_subset = df[df["py"] == 120].copy()
X_subset = df_subset[["x (m)", "y (m)"]].values
y_actual = np.abs(df_subset["z (m)"].values)
x_vals = df_subset["x (m)"].values

# Predict again using separate preprocessing for accuracy
# XGB + Poly2
X_sub_scaled1 = scaler1.transform(X_subset)
X_poly_sub1 = poly1.transform(X_sub_scaled1)
y_pred1 = np.abs(scaler_y1.inverse_transform(m1.predict(X_poly_sub1).reshape(-1, 1)).ravel())

# XGB + RBF
X_sub_scaled2 = scaler2.transform(X_subset)
dist_sub2 = cdist(X_sub_scaled2, rbf_centers2, "sqeuclidean")
X_rbf_sub2 = np.exp(-dist_sub2 / (2 * sigma ** 2))
y_pred2 = np.abs(scaler_y2.inverse_transform(m2.predict(X_rbf_sub2).reshape(-1, 1)).ravel())

# XGB + Poly2 + RBF
X_sub_scaled3 = scaler3.transform(X_subset)
X_poly_sub3 = poly3.transform(X_sub_scaled3)
dist_sub3 = cdist(X_sub_scaled3, rbf_centers3, "sqeuclidean")
X_rbf_sub3 = np.exp(-dist_sub3 / (2 * sigma ** 2))
X_combined_sub3 = np.hstack([X_poly_sub3, X_rbf_sub3])
y_pred3 = np.abs(scaler_y3.inverse_transform(m3.predict(X_combined_sub3).reshape(-1, 1)).ravel())

# LightGBM
y_pred4 = np.abs(m4.predict(X_subset))

# === Plot
plt.figure(figsize=(12, 7))
plt.plot(x_vals, y_actual, label="Actual Z", color="black", linewidth=4)
plt.plot(x_vals, y_pred1, label="XGB + Poly2", linestyle='--', linewidth=4)
plt.plot(x_vals, y_pred2, label="XGB + RBF", linestyle='--', linewidth=4)
plt.plot(x_vals, y_pred3, label="XGB + Poly2 + RBF", linestyle='--', linewidth=4)
plt.plot(x_vals, y_pred4, label="LightGBM", linestyle='--', linewidth=4)

# Set font size for all text elements
plt.xlabel("x (m)", fontsize=24)
plt.ylabel("|z| (m)", fontsize=24)
plt.title("Absolute Z Prediction Comparison at py = 120", fontsize=24)
plt.legend(fontsize=24)

# Remove the grid and set text size for ticks
plt.grid(False)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.tight_layout()
plt.show()
