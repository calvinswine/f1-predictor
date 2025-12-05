import fastf1
import pandas as pd

# Enable caching
fastf1.Cache.enable_cache("cache_folder")

# Get the 2025 F1 schedule
schedule = fastf1.get_event_schedule(2025)

# Print in the format: round_no.      name
print("Round_No.      Name")
for _, row in schedule.iterrows():
    print(f"{row['RoundNumber']:<13} {row['EventName']}")