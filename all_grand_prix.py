# ----------------------------
# COMPLETE F1 FASTEST LAP MODEL
# ----------------------------

import fastf1
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Enable FastF1 caching
fastf1.Cache.enable_cache("cache_folder")

# ----------------------------
# COMPLETE DRIVER LIST (Abbreviations)
# ----------------------------
ALL_DRIVERS = [
    "NOR", "PIA", "VER", "RUS", "TSU",
    "ALB", "LEC", "HAM", "GAS", "SAI",
    "STR", "ALO", "HAD", "ANT", "BOR",
    "HUL", "OCO", "LAW", "COL", "BEA"
]

# ----------------------------
# FUNCTION: Get fastest lap for a track + session + 107% fix
# ----------------------------
def get_fastest_laps(year, round_no, track_name, session_type):
    session = fastf1.get_session(year, round_no, session_type)
    session.load()

    laps = session.laps

    # Get fastest lap per driver
    fastest = laps.groupby("Driver")["LapTime"].min().reset_index()
    fastest["LapTime_s"] = fastest["LapTime"].dt.total_seconds()
    fastest.drop(columns=["LapTime"], inplace=True)

    # Use full driver list to ensure everyone is included
    full_df = pd.DataFrame({"Driver": ALL_DRIVERS})

    # Merge with lap times
    full_df = full_df.merge(fastest, on="Driver", how="left")

    # Apply 107% rule for missing times
    min_time = full_df["LapTime_s"].min()
    full_df["LapTime_s"].fillna(min_time * 1.07, inplace=True)

    # Add Track & Session columns
    full_df["Track"] = track_name
    full_df["Session"] = session_type   # Q or R

    return full_df

# ----------------------------
# LOAD MULTI-TRACK TRAINING DATA (Monaco + Spain)
# ----------------------------
training_rows = []

# Monaco (Round 9)
training_rows.append(get_fastest_laps(2025, 9, "Monaco", "Q"))
training_rows.append(get_fastest_laps(2025, 9, "Monaco", "R"))

# Spain (Round 10)
training_rows.append(get_fastest_laps(2025, 10, "Spain", "Q"))
training_rows.append(get_fastest_laps(2025, 10, "Spain", "R"))

# Combine everything
training_data = pd.concat(training_rows, ignore_index=True)

# Format times to 3 decimals
training_data["LapTime_s"] = training_data["LapTime_s"].round(3)

print("\n================ TRAINING DATA (LONG FORMAT) ================")
print(training_data)

# ----------------------------
# PREPARE TRAINING SET (Quali → Race for each track)
# ----------------------------

# Separate qualifying and race data
quali = training_data[training_data["Session"] == "Q"]
race  = training_data[training_data["Session"] == "R"]

# Merge quali & race by Driver + Track
dataset = pd.merge(
    quali,
    race,
    on=["Driver", "Track"],
    suffixes=("_quali", "_race")
)

# Encode Track as numeric column
track_encoder = LabelEncoder()
dataset["Track_ID"] = track_encoder.fit_transform(dataset["Track"])

# Training X and y
X_train = dataset[["LapTime_s_quali", "Track_ID"]]
y_train = dataset["LapTime_s_race"]

print("\n================ MODEL TRAINING DATA ================")
print(dataset)

# ----------------------------
# TRAIN MODEL
# ----------------------------
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# ----------------------------
# PREDICT RACE FASTEST LAP FOR CANADA (Round 11)
# ----------------------------

# Canada track ID (new unseen track)
canada_track_id = len(track_encoder.classes_)

# Canada qualifying dataset using abbreviations
qualifying_2025 = pd.DataFrame({
    "Driver": [
        "NOR", "PIA", "VER", "RUS", "TSU",
        "ALB", "LEC", "HAM", "GAS", "SAI",
        "STR", "ALO", "HAD", "ANT", "BOR",
        "HUL", "OCO", "LAW", "COL", "BEA"
    ],
    "QualifyingTime_s": [
        71.625, 71.120, 71.059, 70.899, 72.102,
        71.907, 71.682, 71.526, 72.667, 72.398,
        72.517, 71.586, 71.867, 71.391, 72.385,
        72.183, 73.201, 72.525, 72.142, 72.340
    ]
})

# Build prediction feature set
X_pred = pd.DataFrame({
    "LapTime_s_quali": qualifying_2025["QualifyingTime_s"],
    "Track_ID": canada_track_id
})

# Predict race times
qualifying_2025["Predicted_Race_s"] = model.predict(X_pred)

# Sort by predicted race time
qualifying_2025_sorted = qualifying_2025.sort_values(
    "Predicted_Race_s"
).reset_index(drop=True)

print("\n================ PREDICTED RACE FASTEST LAPS (CANADA) ================")
print(qualifying_2025_sorted)

# ----------------------------
# MODEL PERFORMANCE (Using Actual Canada Race Times)
# ----------------------------
y_actual = [
    74.229, 74.255, 74.287, 74.119, 75.358,
    76.197, 74.261, 74.805, 74.993, 74.389,
    74.902, 75.024, 76.292, 74.455, 75.414,
    75.372, 74.593, 76.320, 76.076, 75.397
]

r2 = r2_score(y_actual, qualifying_2025["Predicted_Race_s"])
mae = mean_absolute_error(y_actual, qualifying_2025["Predicted_Race_s"])

print("\n================ MODEL PERFORMANCE ================")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f} seconds")