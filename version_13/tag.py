"""Third-Party Libraries"""
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
import csv
import os

"""Internal Modules"""
from camera import Camera

class Tag:
    """Handles RealSense camera initialization and continuous frame fetching."""
    def __init__(self,
        camera: Camera,
        side: str
    ) -> None:
        self.camera = camera
        self.side = side
        self.orientation = camera.get_orientation(side)

        while self.orientation is None:
            camera.frame.text = f"Error: AprilTag(s) not found. Press Enter to try again"
            while camera.key != 13:
                pass
            self.orientation = camera.get_orientation(side)
            
        self.model = self.get_error_model()

    # Function to read data from the CSV files
    def get_region_data(self):
        region_data = []

        if not os.path.exists("region.csv"):
            self.camera.create_sample_region()

        # Read region.csv
        with open("region.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                x, y, depth = int(row[0]), int(row[1]), float(row[2])
                region_data.append((x, y, depth))

        return region_data

    def get_linear_reg_error(self, point_xy):
        input_array = np.array(point_xy).reshape(1, -1)

        # 🔐 Sanitize input
        if np.any(np.isnan(input_array)) or np.any(np.isinf(input_array)):
            print(f"⚠️ Skipping regression prediction due to invalid input: {input_array}")
            return 0  # or None or float("nan")

        z_pred = self.model.predict(input_array)
        return z_pred[0]  # or return full if needed

    
    def adjust_error(self, point):
        if point is None or any(np.isnan(point)) or any(np.isinf(point)):
            return point  # or return None
        error = self.get_linear_reg_error((point[0], point[1]))
        point[2] -= error.item()
        return point
    
    def reverse_adjust_error(self, point):
        """Reverse the Z-axis error adjustment by adding the predicted depth error back."""
        error = self.get_linear_reg_error((point[0], point[1]))
        point[2] += error.item()
        return point

    # LINEAR REGRESSION FITTING
    def linear_reg(self, x_values, y_values, z_values):
        # Combine x and y into a single array of shape (n_samples, n_features)
        x_values = np.array(x_values).flatten()
        y_values = np.array(y_values).flatten()
        z_values = np.array(z_values).flatten()
        X = np.column_stack((x_values, y_values))

        # Create and fit the model
        model = LinearRegression()
        model.fit(X, z_values)

        return model

    def get_error_model(self):
        model_path = f"error_model_{self.side}.joblib"

        if os.path.exists(model_path):
            return joblib.load(model_path)

        # Read CSV data
        region_data = self.get_region_data()
        x_values, y_values, z_values = [], [], []

        for x, y, depth in region_data:
            if depth > 0:
                point_tag = self.camera.pixel_to_coordsystem(self, (x, y, depth))
                x_values.append(point_tag[0])
                y_values.append(point_tag[1])
                z_values.append(point_tag[2])

        if not x_values:
            raise ValueError("No valid region data for training error model.")

        X = np.column_stack((x_values, y_values))
        model = LinearRegression()
        model.fit(X, z_values)

        # Save model for reuse
        joblib.dump(model, model_path)

        return model