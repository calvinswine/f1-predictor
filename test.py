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
    laps = session.laps

    fastest_laps = []
    for driver in session.drivers:
        drv_laps = laps.pick_driver(driver)
        if len(drv_laps) > 0:
            fastest = drv_laps.pick_fastest()
            # Convert to seconds float
            lap_time = fastest['LapTime'].total_seconds()
            fastest_laps.append([fastest['Driver'], lap_time])

    return pd.DataFrame(fastest_laps, columns=['Driver', f'LapTime_{session_type.lower()}'])

# ------------------------------------------------------
# 1. Load TRAINING DATA (Monaco R9 + Spain R10)
# ------------------------------------------------------

# --- FETCH MONACO DATA (ROUND 9) ---
monaco_quali = get_fastest_laps(2024, 9, "Q")
monaco_race  = get_fastest_laps(2024, 9, "R")

# merge Monaco
monaco = monaco_quali.merge(monaco_race, on="Driver", suffixes=("_quali_monaco", "_race_monaco"))
data_r9 = pd.merge( monaco_quali, monaco_race, on="Driver", suffixes=("_quali", "_race") )

# --- FETCH SPAIN DATA (ROUND 10) ---
spain_quali = get_fastest_laps(2024, 10, "Q")
spain_race  = get_fastest_laps(2024, 10, "R")

# merge Spain
spain = spain_quali.merge(spain_race, on="Driver", suffixes=("_quali_spain", "_race_spain"))
data_r10 = pd.merge( spain_quali, spain_race, on="Driver", suffixes=("_quali", "_race") )

# --- FINAL MERGE OF MONACO + SPAIN ---
training_data = (
    monaco
    .merge(spain, on="Driver")
    .rename(columns={
        "LapTime_q_quali_monaco": "Monaco_LapTime_quali",
        "LapTime_r_race_monaco": "Monaco_LapTime_race",
        "LapTime_q_quali_spain": "Spain_LapTime_quali",
        "LapTime_r_race_spain": "Spain_LapTime_race"
    })
)

# Print clean formatted dataset
print("\nTRAINING DATASET (Final Format):")
print(training_data[[
    "Driver",
    "Monaco_LapTime_quali",
    "Monaco_LapTime_race",
    "Spain_LapTime_quali",
    "Spain_LapTime_race"
]])

training_data = pd.concat([data_r9, data_r10], ignore_index=True)
# Training inputs and targets
X_train = training_data[["LapTime (s)_quali"]]
y_train = training_data["LapTime (s)_race"]


# ------------------------------------------------------
# 2. Train the Gradient Boosting model
# ------------------------------------------------------
model = GradientBoostingRegressor()
model.fit(X_train, y_train)


# ------------------------------------------------------
# 3. Predict for Canada (Round 11) qualifying times
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

# Ensure column name matches training feature
X_pred = qualifying_2025[["QualifyingTime (s)"]].copy()
X_pred.columns = ["LapTime (s)_quali"]

# Predict race fastest laps
y_pred = model.predict(X_pred)
qualifying_2025["Predicted Race Time (s)"] = y_pred

# Sort results from fastest to slowest predicted race time
qualifying_2025_sorted = qualifying_2025.sort_values(
    "Predicted Race Time (s)"
).reset_index(drop=True)

print("\nPREDICTED RACE TIMES (Canada 2025):")
print(qualifying_2025_sorted[["Driver", "QualifyingTime (s)", "Predicted Race Time (s)"]])


# ------------------------------------------------------
# 4. Model Performance (using your actual Canadian times)
# ------------------------------------------------------
y_actual = [
    74.229, 74.255, 74.287, 74.119, 75.358,
    76.197, 74.261, 74.805, 74.993, 74.389,
    74.902, 75.024, 76.292, 74.455, 75.414,
    75.372, 74.593, 76.320, 76.076, 75.397
]

# Correct R² calculation: r2_score(y_true, y_pred)
r2 = r2_score(y_actual, y_pred)
mae = mean_absolute_error(y_actual, y_pred)

print("\nMODEL PERFORMANCE:")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f} seconds")
