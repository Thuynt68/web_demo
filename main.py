import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# Load metadata file
def load_metadata(file_path):
    return pd.read_csv(file_path)

# Load speed data file (last 18 rows only)
def load_speed_data(file_path):
    nrows = sum(1 for _ in open(file_path))  # Count the number of rows
    skip_rows = max(0, nrows - 18)          # Load only the last 18 rows
    return pd.read_csv(file_path, skiprows=range(1, skip_rows))

# Load metadata and speed data
meta_file = "data/PEMS-BAY-META.csv"
speed_file = "data/PEMS-BAY.csv"
meta_data = load_metadata(meta_file)
speed_data = load_speed_data(speed_file)

# Split the speed data into 12 actual and 6 predicted points
actual_speeds = speed_data.iloc[:12, :]  # First 12 rows for actual speeds
predicted_speeds = speed_data.iloc[11:, :]  # Last 6 rows for predicted speeds

# Extract the latest actual speed for each sensor
latest_speeds = actual_speeds.iloc[-1]  # Use the last row of the actual data

# Add the speed column to metadata
def map_speed(sensor_id):
    return latest_speeds[str(sensor_id)]

meta_data["Speed"] = meta_data["sensor_id"].map(map_speed)

# App title
st.title("Traffic Speed Prediction")

# Map sensor names to IDs
sensor_name_map = {row["Name"]: row["sensor_id"] for _, row in meta_data.iterrows()}
all_sensor_names = ["All"] + list(sensor_name_map.keys())

# Dropdown to select sensor
selected_sensor_name = st.selectbox("Select a sensor:", all_sensor_names)

# Function to determine marker color based on speed
def get_marker_color(speed):
    if speed > 70:
        return "green"
    elif speed > 40:
        return "orange"
    else:
        return "red"

# Generate a detailed speed chart
def generate_speed_chart(sensor_id):
    # Extract data for the selected sensor
    speeds_actual = actual_speeds[str(sensor_id)].tolist()  # 12 actual points
    speeds_predicted = predicted_speeds[str(sensor_id)].tolist()  # 6 predicted points

    time_continuous1 = list(range(-55, 5, 5))
    time_continuous2 = list(range(0, 35, 5))

    plt.figure(figsize=(10, 6))
    
    plt.plot(time_continuous2, speeds_predicted, label="Speed Predicted", color="red", marker="o")
    plt.plot(time_continuous1, speeds_actual, label="Speed Actual", color="blue", marker="o")

    plt.title(f"Speed Data for Sensor {sensor_id}", fontsize=16)
    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Speed (km/h)", fontsize=12)
    plt.grid(True, alpha=0.5)

    # Annotate each data point with speed
    for x, y in zip(time_continuous2, speeds_predicted):
        plt.text(x+ 0.005, y + 0.005, f"{y:.1f}", fontsize=10, ha="center", va="bottom", color="red")

    for x, y in zip(time_continuous1, speeds_actual):
        plt.text(x+ 0.005, y + 0.005, f"{y:.1f}", fontsize=10, ha="center", va="bottom", color="blue")

    
    plt.legend(fontsize=12)
    st.pyplot(plt)

# Create map
m = folium.Map(location=[meta_data["Latitude"].mean(), meta_data["Longitude"].mean()], zoom_start=10)

if selected_sensor_name == "All":
    # Display all sensors on the map
    for _, row in meta_data.iterrows():
        color = get_marker_color(row["Speed"])
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            tooltip=f"Sensor {row['Name']}: {row['Speed']:.2f} km/h"
        ).add_to(m)
else:
    # Display selected sensor info
    selected_sensor_id = sensor_name_map[selected_sensor_name]
    sensor_info = meta_data[meta_data["sensor_id"] == selected_sensor_id]
    latitude = sensor_info.iloc[0]["Latitude"]
    longitude = sensor_info.iloc[0]["Longitude"]
    speed = sensor_info.iloc[0]["Speed"]

    st.subheader(f"Sensor Name: {selected_sensor_name}")
    st.write(f"Sensor ID: {selected_sensor_id}")
    st.write(f"Latitude: {latitude}")
    st.write(f"Longitude: {longitude}")
    st.write(f"Speed: {speed:.2f} km/h")

    folium.Marker(
        location=[latitude, longitude],
        popup=f"Sensor {selected_sensor_name}: {speed:.2f} km/h",
        icon=folium.Icon(color="blue"),
    ).add_to(m)

    generate_speed_chart(selected_sensor_id)

# Display the map
st_folium(m, width=700, height=500)
