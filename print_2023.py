import fastf1
import pandas as pd
import numpy as np

# Enable FastF1 caching
fastf1.Cache.enable_cache("cache_folder")

# Load FastF1 2023 Azerbaijan GP race session
session_2023 = fastf1.get_session(2023, 4, "R")  # 2023, round 4 = Azerbaijan GP
session_2023.load()

# Extract lap times
laps_2023 = session_2023.laps[["Driver", "LapTime"]].copy()
laps_2023.dropna(subset=["LapTime"], inplace=True)
laps_2023["LapTime (s)"] = laps_2023["LapTime"].dt.total_seconds()

# Get each driver’s fastest lap, sorted alphabetically
fastest_laps = laps_2023.groupby("Driver")["LapTime (s)"]\
    .min().reset_index()\
    .sort_values(by="Driver", key=lambda s: s.astype(str))

# Print in format: NOR   lt
for _, row in fastest_laps.iterrows():
    print(f"{row['Driver']:3}   {row['LapTime (s)']:.3f}")