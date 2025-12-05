import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
fastf1.Cache.enable_cache("cache_folder")

# Load FastF1 2024 Azerbaijan GP race session
session_2024 = fastf1.get_session(2024, 16, "R")
session_2024.load()

# Extract lap times
laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()

# Get each driver’s fastest lap
fastest_laps = laps_2024.groupby("Driver")["LapTime (s)"].min().reset_index().sort_values(by="Driver", key=lambda s: s.astype(str))

# Sort in increasing order of lap time (fastest first)
#fastest_laps = fastest_laps.sort_values("LapTime (s)")

# Print in format: NOR   lt
for _, row in fastest_laps.iterrows():
    print(f"{row['Driver']:3}   {row['LapTime (s)']:.3f}")