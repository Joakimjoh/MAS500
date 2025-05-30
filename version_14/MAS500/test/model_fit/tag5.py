from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import joblib
import csv
import os

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

        self.load_or_train_model()

    def get_region_data(self):
        region_data = []

        if not os.path.exists("region.csv"):
            self.camera.create_sample_region()

        with open("region.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                x, y, depth = int(row[0]), int(row[1]), float(row[2])
                region_data.append((x, y, depth))

        return region_data

    def get_linear_reg_error(self, point_xy):
        input_array = np.array(point_xy).reshape(1, -1)

        if np.any(np.isnan(input_array)) or np.any(np.isinf(input_array)):
            return 0.0

        input_scaled = self.scaler_xy.transform(input_array)
        input_poly = self.poly.transform(input_scaled)
        z_pred = self.model.predict(input_poly)

        return z_pred[0]

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

    def load_or_train_model(self):
        model_path = f"error_model_{self.side}.joblib"

        if os.path.exists(model_path):
            data = joblib.load(model_path)
            self.model = data['model']
            self.scaler_xy = data['scaler_xy']
            self.poly = data['poly']
            return

        region_data = self.get_region_data()
        x_values, y_values, z_values = [], [], []

        for x, y, depth in region_data:
            if depth > 0:
                point = self.camera.pixel_to_coordsystem(self, (x, y, depth))
                x_values.append(point[0])
                y_values.append(point[1])
                z_values.append(point[2])

        if not x_values:
            raise ValueError("No valid region data for training error model.")

        X = np.column_stack((x_values, y_values))

        # === Preprocessing ===
        self.scaler_xy = StandardScaler().fit(X)
        X_scaled = self.scaler_xy.transform(X)

        self.poly = PolynomialFeatures(degree=3, include_bias=False)
        X_poly = self.poly.fit_transform(X_scaled)

        # === Model training ===
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_poly, z_values)

        # === Save model ===
        joblib.dump({
            'model': self.model,
            'scaler_xy': self.scaler_xy,
            'poly': self.poly
        }, model_path)
