import fastf1
fastf1.Cache.enable_cache('f1_cache')
session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load()
if hasattr(session, 'pit_stops'):
    print("Pit stops attribute exists?") # actually fastf1 has no session.pit_stops.
    # actually wait, it does: https://docs.fastf1.dev/events.html
    # let's just print dir(session)
print(dir(session))
# Let's see if we can calc pit loss from laps
# In-lap time + Out-lap time - 2 * average lap time?
import pandas as pd

laps = session.laps
in_laps = laps[~laps['PitInTime'].isna()][['Driver', 'Stint', 'LapNumber', 'PitInTime']]
out_laps = laps[~laps['PitOutTime'].isna()][['Driver', 'Stint', 'LapNumber', 'PitOutTime']]

# Out lap Stint is usually In lap Stint + 1
in_laps['NextStint'] = in_laps['Stint'] + 1
pit_stops = pd.merge(in_laps, out_laps, left_on=['Driver', 'NextStint'], right_on=['Driver', 'Stint'], suffixes=('_in', '_out'))
pit_stops['PitLaneTime'] = (pit_stops['PitOutTime'] - pit_stops['PitInTime']).dt.total_seconds()
print("Median Pit Lane Time in seconds:", pit_stops['PitLaneTime'].median())
print("Sample Pit Stops:", pit_stops.head())

