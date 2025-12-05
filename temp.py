# ----------------------------
# COMPLETE F1 FASTEST LAP MODEL
# ----------------------------

import fastf1
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
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

# Australia (Round 1)
training_rows.append(get_fastest_laps(2025, 1, "Australia", "Q"))
training_rows.append(get_fastest_laps(2025, 1, "Australia", "R"))

# China (Round 2)
training_rows.append(get_fastest_laps(2025, 2, "China", "Q"))
training_rows.append(get_fastest_laps(2025, 2, "China", "R"))

# Japan (Round 3)
training_rows.append(get_fastest_laps(2025, 3, "Japan", "Q"))
training_rows.append(get_fastest_laps(2025, 3, "Japan", "R"))

# Bahrain (Round 4)
training_rows.append(get_fastest_laps(2025, 4, "Bahrain", "Q"))
training_rows.append(get_fastest_laps(2025, 4, "Bahrain", "R"))

# Saudi Arabia (Round 5)
training_rows.append(get_fastest_laps(2025, 5, "Saudi", "Q"))
training_rows.append(get_fastest_laps(2025, 5, "Saudi", "R"))

# Miami (Round 6)
training_rows.append(get_fastest_laps(2025, 6, "Miami", "Q"))
training_rows.append(get_fastest_laps(2025, 6, "Miami", "R"))

# Emilia Romagna (Round 7)
training_rows.append(get_fastest_laps(2025, 7, "Emilia Romagna", "Q"))
training_rows.append(get_fastest_laps(2025, 7, "Emilia Romagna", "R"))

# Monaco (Round 8)
training_rows.append(get_fastest_laps(2025, 8, "Monaco", "Q"))
training_rows.append(get_fastest_laps(2025, 8, "Monaco", "R"))

# Spain (Round 9)
training_rows.append(get_fastest_laps(2025, 9, "Spain", "Q"))
training_rows.append(get_fastest_laps(2025, 9, "Spain", "R"))

# Canada (Round 10)
training_rows.append(get_fastest_laps(2025, 10, "Canada", "Q"))
training_rows.append(get_fastest_laps(2025, 10, "Canada", "R"))

# Austria (Round 11)
training_rows.append(get_fastest_laps(2025, 11, "Austria", "Q"))
training_rows.append(get_fastest_laps(2025, 11, "Austria", "R"))

# Britain (Round 12)
training_rows.append(get_fastest_laps(2025, 12, "Britain", "Q"))
training_rows.append(get_fastest_laps(2025, 12, "Britain", "R"))

# Belgium (Round 13)
training_rows.append(get_fastest_laps(2025, 13, "Belgium", "Q"))
training_rows.append(get_fastest_laps(2025, 13, "Belgium", "R"))

# Hungary (Round 14)
training_rows.append(get_fastest_laps(2025, 14, "Hungary", "Q"))
training_rows.append(get_fastest_laps(2025, 14, "Hungary", "R"))

# Dutch (Round 15)
training_rows.append(get_fastest_laps(2025, 15, "Dutch", "Q"))
training_rows.append(get_fastest_laps(2025, 15, "Dutch", "R"))

# Italy (Round 16)
training_rows.append(get_fastest_laps(2025, 16, "Italy", "Q"))
training_rows.append(get_fastest_laps(2025, 16, "Italy", "R"))

# Azerbaijan (Round 17)
training_rows.append(get_fastest_laps(2025, 17, "Azerbaijan", "Q"))
training_rows.append(get_fastest_laps(2025, 17, "Azerbaijan", "R"))

# Singapore (Round 18)
training_rows.append(get_fastest_laps(2025, 18, "Singapore", "Q"))
training_rows.append(get_fastest_laps(2025, 18, "Singapore", "R"))

# United States (Round 19)
training_rows.append(get_fastest_laps(2025, 19, "United States", "Q"))
training_rows.append(get_fastest_laps(2025, 19, "United States", "R"))

# Mexico (Round 20)
training_rows.append(get_fastest_laps(2025, 20, "Mexico", "Q"))
training_rows.append(get_fastest_laps(2025, 20, "Mexico", "R"))

# Sao Paulo (Round 21)
training_rows.append(get_fastest_laps(2025, 21, "Sao Paulo", "Q"))
training_rows.append(get_fastest_laps(2025, 21, "Sao Paulo", "R"))

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
# Add the future track(s) to encoder before fitting
future_tracks = ["Las Vegas"]  # Add more if needed

all_tracks_for_encoder = list(training_data["Track"].unique()) + future_tracks

track_encoder = LabelEncoder()
track_encoder.fit(all_tracks_for_encoder)

# Encode training data
dataset["Track_ID"] = track_encoder.transform(dataset["Track"])

# Encode prediction track
Vegas_track_id = track_encoder.transform(["Las Vegas"])[0]

# Training X and y
X_train = dataset[["LapTime_s_quali", "Track_ID"]]
y_train = dataset["LapTime_s_race"]

print("\n================ MODEL TRAINING DATA ================")
print(dataset)

# ----------------------------
# TRAIN MODEL
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# PREDICT RACE FASTEST LAP FOR Las Vegas (Round 22)
# ----------------------------

# Canada qualifying dataset using abbreviations
qualifying_2025 = pd.DataFrame({
    "Driver": [
        "NOR", "PIA", "VER", "RUS", "TSU",
        "ALB", "LEC", "HAM", "GAS", "SAI",
        "STR", "ALO", "HAD", "ANT", "BOR",
        "HUL", "OCO", "LAW", "COL", "BEA"
    ],
    "QualifyingTime_s": [
        107.934, 108.961, 108.257, 108.803, 116.798,
        116.220, 109.872, 117.115, 111.540, 108.296,
        112.850, 109.466, 109.554, 116.314, 116.674,
        152.781, 112.987, 109.062, 113.683, 113.094
    ]
})

# Build prediction feature set
X_pred = pd.DataFrame({
    "LapTime_s_quali": qualifying_2025["QualifyingTime_s"],
    "Track_ID": Vegas_track_id
})

# Predict race times
qualifying_2025["Predicted_Race_s"] = model.predict(X_pred)
# Round predicted race times to 3 decimals
qualifying_2025["Predicted_Race_s"] = qualifying_2025["Predicted_Race_s"].round(3)

# Sort by predicted race time
qualifying_2025_sorted = qualifying_2025.sort_values("Predicted_Race_s").reset_index(drop=True)

print("\n================ PREDICTED RACE FASTEST LAPS (Las Vegas) ================")
print(qualifying_2025_sorted)

# ----------------------------
# MODEL PERFORMANCE (Using Actual Las Vegas Race Times)
# ----------------------------
y_actual = [
    93.965, 94.086, 93.365, 94.592, 94.967,
    95.184, 94.304, 94.553, 95.674, 94.496,
    99.900, 95.629, 94.620, 93.998, 99.900,
    94.592, 94.557, 94.837, 95.780, 94.591
]

r2 = r2_score(y_actual, qualifying_2025["Predicted_Race_s"])
mae = mean_absolute_error(y_actual, qualifying_2025["Predicted_Race_s"])

print("\n================ MODEL PERFORMANCE ================")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f} seconds")