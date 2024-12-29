import streamlit as st
import pandas as pd
import folium
import json
import h5py
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

from streamlit_folium import st_folium
from spektral.layers import GCNConv
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Flatten, Reshape, Permute
from os.path import join

class GCN_LSTM(tf.keras.Model):
    def __init__(self, num_sensors, adj_matrix):
        super(GCN_LSTM, self).__init__()
        self.adj_matrix = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)
        self.gcn1 = GCNConv(gcn_hidden, activation="relu")
        self.gcn2 = GCNConv(seq_len, activation="relu")
        self.lstm1 = LSTM(lstm_hidden, activation='tanh', input_shape=(seq_len, num_sensors), 
                          return_sequences=True)
        self.lstm2 = LSTM(lstm_hidden, activation='tanh', return_sequences=True)

        self.reshape1 = Reshape((-1, seq_len, 1))
        self.reshape2 = Reshape((seq_len, -1))
        self.gcn_permute = Permute((2, 1, 3))
        self.lstm_permute = Permute((2, 1))
        self.dropout = Dropout(drop)
        self.dense = Dense(num_sensors)
        self.out_dense = Dense(pre_len)

    def call(self, inputs):
        x = self.gcn1([inputs, self.adj_matrix])
        x = self.dropout(x)
        x = self.gcn2([x, self.adj_matrix])
        x = self.dropout(x)
        x = self.reshape1(x)
        x = self.gcn_permute(x)
        x = self.reshape2(x)
        x = self.lstm1(x)
        x = self.dropout(x)
        x = self.lstm2(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.lstm_permute(x)
        x = self.out_dense(x)
        return x

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
with open('data/adj_mx_bay.pkl', 'rb') as file:
    sensor_, sensor_map, sensor_dist_adj = pickle.load(file, encoding='latin1')
sensor = sensor_dist_adj.shape[0]

#Load model
parameter_path = 'parameters/gcnlstm/'
file_name = 'gcnlstm1'
with open(join(parameter_path, file_name, f"{file_name}.json"), "r") as file:
    parameters = json.load(file)  
seq_len = parameters['seq_len']
pre_len = parameters['pre_len']
gcn_hidden = parameters['gcn_hidden_units']
lstm_hidden = parameters['lstm_hidden_units']
drop = parameters['drop_rate']
model_path = 'models/gcnlstm/'
gcn_lstm_model = GCN_LSTM(sensor, adj_matrix=sensor_dist_adj)
gcn_lstm_model.load_weights(join(model_path, file_name, f"{file_name}"))

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

# Filter by street
distinct_streets = meta_data["Fwy"].unique()
selected_street = st.selectbox("Filter by street:", ["None"] + list(distinct_streets))
if selected_street != "None":
    meta_data = meta_data[meta_data["Fwy"] == selected_street]

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

    # time_actual = [f"{55 - i * 5} mins ago" for i in range(12)]
    # time_predicted = ["Now"] + [f" {i * 5} minutes" for i in range(0, 6)]

    time_actual = list(range(-55, 5, 5))
    time_predicted = list(range(0, 35, 5))

    plt.figure(figsize=(10, 6))
    
    # plt.plot(time_predicted, speeds_predicted, label="Speed Predicted", color="red", marker="o")
    # plt.plot(time_actual[::-1], speeds_actual[::-1], label="Speed Actual", color="blue", marker="o")

    plt.plot(time_predicted, speeds_predicted, label="Speed Predicted", color="red", marker="o")
    plt.plot(time_actual, speeds_actual, label="Speed Actual", color="blue", marker="o")
    plt.title(f"Speed Data for Sensor {sensor_id}", fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Speed (km/h)", fontsize=12)
    plt.grid(True, alpha=0.5)

    max_speed = max(max(speeds_actual), max(speeds_predicted))

    for x, y in zip(time_predicted, speeds_predicted):
        plt.text(x+0.005, y + 0.005, f"{y:.1f}", fontsize=10, ha="center", va="bottom", color="red")

    for x, y in zip(time_actual[::-1], speeds_actual[::-1]):
        plt.text(x+0.005, y + 0.005, f"{y:.1f}", fontsize=10, ha="center", va="bottom", color="blue")

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
