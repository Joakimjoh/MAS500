import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load and clean data ===
def clean_dataframe(path):
    df = pd.read_csv(path)
    for axis in ["x (m)", "y (m)", "z (m)"]:
        df[axis] = df[axis].apply(lambda s: abs(float(s.strip("[]"))) if isinstance(s, str) else abs(s))
    return df

# Load all the datasets into separate variables
df_factory = clean_dataframe("/home/student/Documents/MAS500/test/factory_checkerboard/factory.csv")
df_apriltag_5 = clean_dataframe("/home/student/Documents/MAS500/test/factory_checkerboard/checkerboard5.csv")
df_apriltag_10 = clean_dataframe("/home/student/Documents/MAS500/test/factory_checkerboard/checkerboard10.csv")
df_apriltag_20 = clean_dataframe("/home/student/Documents/MAS500/test/factory_checkerboard/checkerboard20.csv")
df_apriltag_30 = clean_dataframe("/home/student/Documents/MAS500/test/factory_checkerboard/checkerboard30.csv")
df_apriltag_40 = clean_dataframe("/home/student/Documents/MAS500/test/factory_checkerboard/checkerboard40.csv")

# === Plot helper ===
def plot_comparison_at_fixed_values(df1, df2, df3, df4, df5, df6, fixed_axis, fixed_value, variable_axis, label1, label2, label3, label4, label5, label6, title_suffix):
    # Filter data at the fixed values of px or py
    df1_section = df1[df1[fixed_axis] == fixed_value].copy()
    df2_section = df2[df2[fixed_axis] == fixed_value].copy()
    df3_section = df3[df3[fixed_axis] == fixed_value].copy()
    df4_section = df4[df4[fixed_axis] == fixed_value].copy()
    df5_section = df5[df5[fixed_axis] == fixed_value].copy()
    df6_section = df6[df6[fixed_axis] == fixed_value].copy()

    # Check if any section is empty
    if df1_section.empty or df2_section.empty or df3_section.empty or df4_section.empty or df5_section.empty or df6_section.empty:
        print(f"No data found at {fixed_axis} = {fixed_value}")
        return

    # Get the x and z values for each section
    x1, z1 = df1_section[variable_axis].values, df1_section["z (m)"].values
    x2, z2 = df2_section[variable_axis].values, df2_section["z (m)"].values
    x3, z3 = df3_section[variable_axis].values, df3_section["z (m)"].values
    x4, z4 = df4_section[variable_axis].values, df4_section["z (m)"].values
    x5, z5 = df5_section[variable_axis].values, df5_section["z (m)"].values
    x6, z6 = df6_section[variable_axis].values, df6_section["z (m)"].values

    # Sort for smooth line plotting
    idx1 = np.argsort(x1)
    idx2 = np.argsort(x2)
    idx3 = np.argsort(x3)
    idx4 = np.argsort(x4)
    idx5 = np.argsort(x5)
    idx6 = np.argsort(x6)

    x1, z1 = x1[idx1], z1[idx1]
    x2, z2 = x2[idx2], z2[idx2]
    x3, z3 = x3[idx3], z3[idx3]
    x4, z4 = x4[idx4], z4[idx4]
    x5, z5 = x5[idx5], z5[idx5]
    x6, z6 = x6[idx6], z6[idx6]

    # Plot all the data
    plt.figure(figsize=(12, 7))
    plt.plot(x1, z1, label=label1, linewidth=4, color="blue")
    plt.plot(x2, z2, label=label2, linewidth=4, color="green")
    plt.plot(x3, z3, label=label3, linewidth=4, color="red")
    plt.plot(x4, z4, label=label4, linewidth=4, color="purple")
    plt.plot(x5, z5, label=label5, linewidth=4, color="orange")
    plt.plot(x6, z6, label=label6, linewidth=4, color="brown")

    # Increase font sizes for all text elements
    plt.xlabel(variable_axis, fontsize=24)
    plt.ylabel("z (m)", fontsize=24)
    plt.title(f"Comparison of Absolute Z at {fixed_axis} = {fixed_value} ({title_suffix})", fontsize=24)
    plt.legend(fontsize=24)
    plt.grid(False)
    plt.tight_layout()

    # Increase tick label sizes
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.show()

# === Visualize comparison at fixed px = 120 and py = 160 ===
plot_comparison_at_fixed_values(
    df_factory, df_apriltag_5, df_apriltag_10, df_apriltag_20, df_apriltag_30, df_apriltag_40,
    fixed_axis="py", fixed_value=120,
    variable_axis="x (m)", label1="Factory", label2="Checkerboard (5)", label3="Checkerboard (10)", label4="Checkerboard (20)", label5="Checkerboard (30)", label6="Checkerboard (40)",
    title_suffix="X-Slice"
)

plot_comparison_at_fixed_values(
    df_factory, df_apriltag_5, df_apriltag_10, df_apriltag_20, df_apriltag_30, df_apriltag_40,
    fixed_axis="px", fixed_value=160,
    variable_axis="y (m)", label1="Factory", label2="Checkerboard (5)", label3="Checkerboard (10)", label4="Checkerboard (20)", label5="Checkerboard (30)", label6="Checkerboard (40)",
    title_suffix="Y-Slice"
)
