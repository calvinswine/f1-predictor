import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache("cache_folder")

session_2024 = fastf1.get_session(2024,"Abu Dhabi","R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver","LapTime","Sector1Time","Sector2Time","Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

for col in ["LapTime","Sector1Time","Sector2Time","Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

#Aggregate Sector data by driver
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"]+
    sector_times_2024["Sector2Time (s)"]+
    sector_times_2024["Sector3Time (s)"]
)

#2025 Bahrain GP quali data (we need to change this before Jeddah)
qualifying_2025 = pd.DataFrame({
    "Driver":["VER","NOR","PIA","LEC","RUS","HAM","GAS","ALO","TSU","SAI","HUL","OCO","STR"],
    "QualifyingTime (s)": [79.651, 79.495, 79.387, 80.561, 79.662,
                          80.907, 80.477, 80.418, 80.761, 80.287,
                          80.353, 80.864, 81.058]
})

driver_points = {
    "VER": 396,"PIA": 392,"LEC": 230,"RUS": 309,"HAM": 152,
    "GAS": 22, "ALO": 48, "TSU": 33, "SAI": 64, "HUL": 49,
    "OCO": 32, "STR": 32, "NOR": 408
}

driver_wet_performance = {
    "VER": 0.975196, "HAM": 0.976464, "LEC": 0.975862, "NOR": 0.978179, "ALO": 0.972655,
    "RUS": 0.968678, "SAI": 0.978754, "TSU": 0.996338, "OCO": 0.981810, "GAS": 0.978832, "STR": 0.979857
}

qualifying_2025["WetPerformanceFactor"] = qualifying_2025["Driver"].map(driver_wet_performance)

#Changes depending on the race day
rain_probability = 0
temperature = 20

if rain_probability>=0.75:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]*qualifying_2025["WetPerformanceFactor"]
else:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# Points scored by teams in constructor times
team_points = {
    "McLaren" : 800,"Mercedes": 459,"Red Bull":426,"Williams":137,"Ferrari":382,
    "Haas":73,"Aston Martin":80,"Kick Sauber":68,"Racing Bulls":92,"Alpine":22
}

max_points = max(team_points.values())
team_performace_score = {team: points/max_points for team,points in team_points.items()}

driver_to_team = {
    "VER":"Red Bull","NOR":"McLaren","PIA":"McLaren","LEC":"Ferrari","RUS":"Mercedes",
    "HAM":"Ferrari","GAS":"Alpine","ALO":"Aston Martin","TSU":"Red Bull",
    "SAI":"Williams","HUL":"Kick Sauber","OCO":"Haas","STR":"Aston Martin"
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performace_score)
qualifying_2025["DriverPoints"] = qualifying_2025["Driver"].map(driver_points)

merged_data = qualifying_2025.merge(sector_times_2024[["Driver","TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature
last_year_winner = "NOR"
merged_data["LastYearWinner"] = (merged_data["Driver"] == last_year_winner).astype(int)

merged_data["QualifyingTime"] = merged_data["QualifyingTime"] ** 2

X = merged_data[[
    "QualifyingTime (s)","RainProbability","Temperature","TeamPerformanceScore","TotalSectorTime (s)",
    "DriverPoints","LastYearWinner"
]].fillna(0)

y=laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

clean_data=merged_data.copy()
clean_data["LapTime (s)"] = y.values
clean_data = clean_data.dropna(subset=["LapTime (s)"])

X = clean_data[[
    "QualifyingTime","RainProbability","Temperature","TeamPerformanceScore",
    "TotalSectorTime (s)","DriverPoints","LastYearWinner"
]].fillna(0)
y = clean_data["LapTime (s)"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=38)
model =GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,random_state=38)
model.fit(X_train,y_train)
clean_data["PredictedRaceTime (s)"] = model.predict(X)

final_results = clean_data.sort_values("PredictedRaceTime (s)")
print("Predicted 2025 Abu Dhabi GP Winner:\n")
print(final_results[["Driver","PredictedRaceTime (s)"]])

y_pred = model.predict(X_test)
print(f"\nModel Error (MAE) : {mean_absolute_error(y_test,y_pred):.2f} seconds")

plt.figure(figsize=(12,8))
plt.scatter(final_results["TeamPerformanceScore"],
            final_results["PredictedRaceTime (s)"],
            c = final_results["QualifyingTime"])

for i, driver in enumerate(final_results["Driver"]):
    plt.annotate(
        driver,
        (final_results["TeamPerformanceScore"].iloc[i],
         final_results["PredictedRaceTime (s)"].iloc[i]),
        xytext=(5, 5),
        textcoords='offset points'
    )


plt.colorbar(label="Qualifying Time")
plt.xlabel("Team Performance Score")
plt.ylabel("Predicted Race Time (s)")
plt.title("Effect of Team Performance on Predicted Race Results")
plt.tight_layout()
plt.show()

feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features,feature_importance,color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Race Time Prediction")
plt.tight_layout()
plt.show()