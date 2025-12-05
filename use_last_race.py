import fastf1
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Enable FastF1 caching
fastf1.Cache.enable_cache("cache_folder")

# Function to get each driver's fastest lap
def get_fastest_laps(year, round_no, session_type):
    session = fastf1.get_session(year, round_no, session_type)
    session.load()

    laps = session.laps[["Driver", "LapTime"]].copy()
    laps.dropna(subset=["LapTime"], inplace=True)
    laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()

    fastest = laps.groupby("Driver")["LapTime (s)"].min().reset_index()
    return fastest

# ------------------------------------------------------
# 1. Load 2025 Spanish GP (Round 10) Qualifying + Race
# ------------------------------------------------------
fastest_quali_2025 = get_fastest_laps(2025, 10, "Q")
fastest_race_2025  = get_fastest_laps(2025, 10, "R")

# Merge by driver
data = pd.merge(
    fastest_quali_2025,
    fastest_race_2025,
    on="Driver",
    suffixes=("_quali", "_race")
)

# Training data
X_train = data[["LapTime (s)_quali"]]
y_train = data["LapTime (s)_race"]

# ------------------------------------------------------
# 2. Train the Gradient Boosting model
# ------------------------------------------------------
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# ------------------------------------------------------
# 3. Predict for Canadian(Round 11) qualifying times
# ------------------------------------------------------
qualifying_2025 = pd.DataFrame({
    "Driver": [
        "Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell", "Yuki Tsunoda",
        "Alexander Albon", "Charles Leclerc", "Lewis Hamilton", "Pierre Gasly", "Carlos Sainz",
        "Lance Stroll", "Fernando Alonso", "Isack Hadjar", "Kimi Antonelli", "Gabriel Bortoleto",
        "Nico Hulkenberg", "Esteban Ocon", "Liam Lawson", "Franco Colapinto", "Oliver Bearman"
    ],
    "QualifyingTime (s)": [
        71.625, 71.120, 71.059, 70.899, 72.102, 
        71.907, 71.682, 71.526, 72.667, 72.398, 
        72.517, 71.586, 71.867, 71.391, 72.385, 
        72.183, 73.201, 72.525, 72.142, 72.340
    ]
})

y_actual = [74.229,74.255,74.287,74.119,75.358,76.197,74.261,74.805,74.993,74.389,74.902,75.024,76.292,74.455,75.414,75.372,74.593,76.320,76.076,75.397]
# Make sure feature names match training
X_pred = qualifying_2025[["QualifyingTime (s)"]].copy()
X_pred.columns = ["LapTime (s)_quali"]  # <-- MUST match training column

# Predict race times
y_pred = model.predict(X_pred)
qualifying_2025["Predicted Race Time (s)"] = y_pred

qualifying_2025_sorted = qualifying_2025.sort_values("Predicted Race Time (s)").reset_index(drop=True)
print("\nPREDICTED RACE TIMES:")
print(qualifying_2025_sorted[["Driver", "QualifyingTime (s)", "Predicted Race Time (s)"]])

r2 = r2_score(y_pred, y_actual)
mae = mean_absolute_error(y_pred, y_actual)

print("\nMODEL PERFORMANCE:")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f} seconds")