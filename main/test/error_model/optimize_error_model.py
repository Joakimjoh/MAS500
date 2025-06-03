import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error
from xgboost import XGBRegressor
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist

# === Load and clean data ===
df = pd.read_csv("/home/student/Documents/MAS500/test/raw_test_data.csv")
df["x (m)"] = df["x (m)"].apply(lambda s: float(s.strip("[]")))
df["y (m)"] = df["y (m)"].apply(lambda s: float(s.strip("[]")))
df["z (m)"] = df["z (m)"].apply(lambda s: float(s.strip("[]")))

X = df[["x (m)", "y (m)"]].values.astype(np.float32)
y = df["z (m)"].values.astype(np.float32)

# === Preprocess input features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled).astype(np.float32)

xq = np.quantile(X_scaled[:, 0], [0.01, 0.99])
yq = np.quantile(X_scaled[:, 1], [0.01, 0.99])
x_centers = np.linspace(xq[0], xq[1], 5)
y_centers = np.linspace(yq[0], yq[1], 5)
xc, yc = np.meshgrid(x_centers, y_centers)
rbf_centers = np.column_stack([xc.ravel(), yc.ravel()]).astype(np.float32)

sigma = 0.3
dist = euclidean_distances(X_scaled, rbf_centers) ** 2
X_rbf = np.exp(-dist / (2 * sigma ** 2)).astype(np.float32)

X_combined = np.hstack([X_poly, X_rbf])

scaler_y = StandardScaler()
y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# === Hyperparameter search ===
param_grid = {
    "n_estimators": [10, 20, 30, 40, 50],
    "max_depth":  [8, 10, 12, 14, 16],
    "learning_rate": [0.12, 0.14, 0.16, 0.18]
}

results = []
best_model = None
best_score = float("inf")

for ne in param_grid["n_estimators"]:
    for md in param_grid["max_depth"]:
        for lr in param_grid["learning_rate"]:
            model = XGBRegressor(
                n_estimators=ne,
                learning_rate=lr,
                max_depth=md,
                subsample=0.7,
                colsample_bytree=0.7,
                tree_method="hist",
                enable_categorical=False,
                verbosity=0,
                random_state=42,
                n_jobs=-1
            )
            start_train = time.time()
            model.fit(X_combined, y_norm)
            train_time = time.time() - start_train

            start_pred = time.time()
            y_pred = scaler_y.inverse_transform(model.predict(X_combined).reshape(-1, 1)).ravel()
            pred_time = time.time() - start_pred

            mae = mean_absolute_error(y, y_pred)
            rmse = mean_squared_error(y, y_pred, squared=False)
            r2 = r2_score(y, y_pred)
            max_err = max_error(y, y_pred)

            if max_err < 0.0008 and pred_time < 0.02:
                results.append({
                    "Model": f"XGB + Poly2 + RBF (ne={ne}, md={md}, lr={lr})",
                    "n_estimators": ne,
                    "max_depth": md,
                    "learning_rate": lr,
                    "Train Time (s)": train_time,
                    "Pred Time (s)": pred_time,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "Max Error": max_err
                })

                if mae < best_score:
                    best_score = mae
                    best_model = model

# === Save all valid results ===
df_results = pd.DataFrame(results)
print("\nFiltered Model Comparison Results:")
print(df_results.to_string(index=False))
df_results.to_csv("/home/student/Documents/MAS500/test/model_opimization_comparison.csv", index=False)

# === Visualize prediction at py = 120 (only best model) ===
if best_model:
    df_subset = df[df["py"] == 120].copy()
    X_subset = df_subset[["x (m)", "y (m)"]].values.astype(np.float32)
    y_actual = np.abs(df_subset["z (m)"].values)
    x_vals = df_subset["x (m)"].values

    X_sub_scaled = scaler.transform(X_subset)
    X_poly_sub = poly.transform(X_sub_scaled).astype(np.float32)
    dist_sub = cdist(X_sub_scaled, rbf_centers, "sqeuclidean")
    X_rbf_sub = np.exp(-dist_sub / (2 * sigma ** 2)).astype(np.float32)
    X_combined_sub = np.hstack([X_poly_sub, X_rbf_sub])
    y_pred_final = np.abs(scaler_y.inverse_transform(best_model.predict(X_combined_sub).reshape(-1, 1)).ravel())

    plt.figure(figsize=(12, 7))
    plt.plot(x_vals, y_actual, label="Actual Z", color="black", linewidth=2)
    plt.plot(x_vals, y_pred_final, label=df_results.iloc[0]['Model'], linestyle='--')
    plt.xlabel("x (m)")
    plt.ylabel("|z| (m)")
    plt.title("Absolute Z Prediction Comparison at py = 120")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No model met both max error and pred time constraints.")
