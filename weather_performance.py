import fastf1
import pandas as pd

fastf1.Cache.enable_cache("cache_folder")

session_q = fastf1.get_session(2025, "Australia", "Q")
session_q.load()

session_r = fastf1.get_session(2025, "Australia", "R")
session_r.load()

laps_q = session_q.laps[["Driver", "LapTime"]].copy()
laps_r = session_r.laps[["Driver", "LapTime"]].copy()


laps_q["LapTime (s)"] = laps_q["LapTime"].dt.total_seconds()
laps_r["LapTime (s)"] = laps_r["LapTime"].dt.total_seconds()

avg_q = laps_q.groupby("Driver")["LapTime (s)"].mean().reset_index()
avg_r = laps_r.groupby("Driver")["LapTime (s)"].mean().reset_index()

merged = pd.merge(avg_q, avg_r, on="Driver", suffixes=('_Q', '_R'))

merged["LapTimeDiff (s)"] = merged["LapTime (s)_R"] - merged["LapTime (s)_Q"]

merged["PerformanceChange (%)"] = (merged["LapTimeDiff (s)"] / merged["LapTime (s)_Q"]) * 100

merged["WetPerformanceScore"] = 1+(merged["PerformanceChange (%)"] / 100)

print("\nDriver Performance Change (Race vs Qualifying) — Australia 2025")
print(merged[["Driver", "LapTime (s)_Q", "LapTime (s)_R", "WetPerformanceScore"]])
