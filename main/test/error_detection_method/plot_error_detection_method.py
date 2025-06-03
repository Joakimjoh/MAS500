import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load and clean data ===
def clean_dataframe(path):
    df = pd.read_csv(path)
    for axis in ["x (m)", "y (m)", "z (m)"]:
        df[axis] = df[axis].apply(lambda s: abs(float(s.strip("[]"))) if isinstance(s, str) else abs(s))
    return df

df_apriltag = clean_dataframe("/home/student/Documents/MAS500/depth_data_apriltag.csv")
df_solvepnp = clean_dataframe("/home/student/Documents/MAS500/depth_data_solvepnp.csv")
df_pupil = clean_dataframe("/home/student/Documents/MAS500/depth_data_apriltag_pupil.csv")

# === Plot helper ===
def plot_comparison_cross_section(df1, df2, df3, fixed_axis, fixed_value, variable_axis, label1, label2, label3, title_suffix):
    df1_section = df1[df1[fixed_axis] == fixed_value].copy()
    df2_section = df2[df2[fixed_axis] == fixed_value].copy()
    df3_section = df3[df3[fixed_axis] == fixed_value].copy()

    if df1_section.empty or df2_section.empty or df3_section.empty:
        print(f"No data found at {fixed_axis} = {fixed_value}")
        return

    x1, z1 = df1_section[variable_axis].values, df1_section["z (m)"].values
    x2, z2 = df2_section[variable_axis].values, df2_section["z (m)"].values
    x3, z3 = df3_section[variable_axis].values, df3_section["z (m)"].values

    # Sort for smooth line plotting
    idx1 = np.argsort(x1)
    idx2 = np.argsort(x2)
    idx3 = np.argsort(x3)

    x1, z1 = x1[idx1], z1[idx1]
    x2, z2 = x2[idx2], z2[idx2]
    x3, z3 = x3[idx3], z3[idx3]

    plt.figure(figsize=(12, 7))
    plt.plot(x1, z1, label=label1, linewidth=4, color="blue")
    plt.plot(x2, z2, label=label2, linewidth=4, color="green")
    plt.plot(x3, z3, label=label3, linewidth=4, color="red")

    # Increase font sizes for all text elements
    plt.xlabel(variable_axis, fontsize=24)
    plt.ylabel("z (m)", fontsize=24)
    plt.title(f"Absolute Z Comparison at {fixed_axis} = {fixed_value} ({title_suffix})", fontsize=24)
    plt.legend(fontsize=24)
    plt.grid(False)  # Disable grid
    plt.tight_layout()

    # Increase tick label sizes
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.show()

# === Visualize cross-sections ===
plot_comparison_cross_section(
    df_apriltag, df_solvepnp, df_pupil,
    fixed_axis="py", fixed_value=120,
    variable_axis="x (m)",
    label1="AprilTag", label2="SolvePnP", label3="AprilTag Pupil",
    title_suffix="X-Slice"
)

plot_comparison_cross_section(
    df_apriltag, df_solvepnp, df_pupil,
    fixed_axis="px", fixed_value=160,
    variable_axis="y (m)",
    label1="AprilTag", label2="SolvePnP", label3="AprilTag Pupil",
    title_suffix="Y-Slice"
)