import fastf1
import pandas as pd
fastf1.Cache.enable_cache("cache_folder")
session_2023=fastf1.get_session(2023,"Canada","R")
session_2023.load()
session_2022=fastf1.get_session(2022,"Canada","R")
session_2022.load()
laps_2023 = session_2023.laps[["Driver","LapTime"]].copy()
laps_2022 = session_2022.laps[["Driver","LapTime"]].copy()
laps_2023.dropna(inplace=True)
laps_2022.dropna(inplace=True)
laps_2023["LapTime (s)"] = laps_2023["LapTime"].dt.total_seconds()
laps_2022["LapTime (s)"] = laps_2022["LapTime"].dt.total_seconds()
avg_lap_2023= laps_2023.groupby("Driver")["LapTime (s)"].mean().reset_index()
avg_lap_2022= laps_2022.groupby("Driver")["LapTime (s)"].mean().reset_index()
merged_data = pd.merge(avg_lap_2023,avg_lap_2022,on="Driver",suffixes=('_2023','_2022'))
merged_data["LapTimeDiff (s)"] = merged_data["LapTime (s)_2023"] - merged_data["LapTime (s)_2022"]
merged_data["PerformanceChange (%)"] = (merged_data["LapTimeDiff (s)"]/merged_data["LapTime (s)_2022"])*100
merged_data["WetPerformanceScore"] = 1+(merged_data["PerformanceChange (%)"]/100)
print("\nDriver Wet Performance Scores (2023 vs 2022) : ")
print(merged_data[["Driver", "LapTime (s)_2023", "LapTime (s)_2022", "WetPerformanceScore"]])