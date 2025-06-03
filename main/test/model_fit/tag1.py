import numpy as np
import csv
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBRegressor
from scipy.spatial.distance import cdist

from camera import Camera

class Tag:
    def __init__(self, camera: Camera, side: str) -> None:
        self.camera = camera
        self.side = side
        self.orientation = camera.get_orientation(side)

        while self.orientation is None:
            camera.frame.text = "Error: AprilTag(s) not found. Press Enter to try again"
            while camera.key != 13:
                pass
            self.orientation = camera.get_orientation(side)

        camera.frame.text = f"Training error model for {side} side."

        self.load_or_train_model()

    def get_region_data(self):
        if not os.path.exists("region.csv"):
            self.camera.create_sample_region()

        region_data = []
        with open("region.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                x, y, depth = int(row[0]), int(row[1]), float(row[2])
                region_data.append((x, y, depth))
        return region_data

    def load_or_train_model(self):
        model_path = f"error_model_{self.side}.joblib"

        if os.path.exists(model_path):
            data = joblib.load(model_path)
            self.model = data['model']
            self.scaler_xy = data['scaler_xy']
            self.scaler_z = data['scaler_z']
            self.poly = data['poly']
            self.rbf_centers = data['rbf_centers']
            self.rbf_sigma = data['rbf_sigma']
            return

        # === Load region data ===
        region_data = self.get_region_data()
        x_values, y_values, z_values = [], [], []

        for x, y, depth in region_data:
            if depth > 0:
                point = self.camera.pixel_to_coordsystem(self, (x, y, depth))
                x_values.append(point[0])
                y_values.append(point[1])
                z_values.append(point[2])

        if len(x_values) < 10:
            raise ValueError("Not enough region data to train model.")

        # === Preprocess ===
        XY = np.column_stack((x_values, y_values))
        self.scaler_xy = StandardScaler().fit(XY)
        XY_scaled = self.scaler_xy.transform(XY)

        self.scaler_z = StandardScaler().fit(np.array(z_values).reshape(-1, 1))
        z_norm = self.scaler_z.transform(np.array(z_values).reshape(-1, 1)).ravel()

        # === Polynomial features ===
        degree = 4
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = self.poly.fit_transform(XY_scaled)

        # === RBF features ===
        self.rbf_sigma = 0.3
        xq = np.quantile(XY_scaled[:, 0], [0.01, 0.99])
        yq = np.quantile(XY_scaled[:, 1], [0.01, 0.99])
        x_centers = np.linspace(xq[0], xq[1], 7)
        y_centers = np.linspace(yq[0], yq[1], 7)
        xc, yc = np.meshgrid(x_centers, y_centers)
        self.rbf_centers = np.vstack([xc.ravel(), yc.ravel()]).T

        def rbf_features(X):
            dist = cdist(X, self.rbf_centers, 'sqeuclidean')
            return np.exp(-dist / (2 * self.rbf_sigma ** 2))

        X_rbf = rbf_features(XY_scaled)
        X_full = np.hstack([X_poly, X_rbf])

        # === Train model ===
        self.model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        # Split train/test for early stopping
        X_train, X_val, z_train, z_val = train_test_split(X_full, z_norm, test_size=0.2, random_state=42)

        self.model.fit(X_train, z_train, eval_set=[(X_val, z_val)], verbose=False)

        # === Save model ===
        joblib.dump({
            'model': self.model,
            'scaler_xy': self.scaler_xy,
            'scaler_z': self.scaler_z,
            'poly': self.poly,
            'rbf_centers': self.rbf_centers,
            'rbf_sigma': self.rbf_sigma
        }, model_path)

    def get_linear_reg_error(self, point_xy):
        input_xy = self.scaler_xy.transform(np.array(point_xy).reshape(1, -1))
        poly_feat = self.poly.transform(input_xy)
        dist = cdist(input_xy, self.rbf_centers, 'sqeuclidean')
        rbf_feat = np.exp(-dist / (2 * self.rbf_sigma ** 2))
        X_input = np.hstack([poly_feat, rbf_feat])
        z_norm = self.model.predict(X_input).reshape(-1, 1)
        z_pred = self.scaler_z.inverse_transform(z_norm).ravel()[0]
        return z_pred

    def adjust_error(self, point):
        if point is None or any(np.isnan(point)) or any(np.isinf(point)):
            return point
        error = self.get_linear_reg_error((point[0], point[1]))
        point[2] -= error
        return point

    def reverse_adjust_error(self, point):
        error = self.get_linear_reg_error((point[0], point[1]))
        point[2] += error
        return point
