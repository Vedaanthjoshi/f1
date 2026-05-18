import pandas as pd
import numpy as np

def calculate_pit_loss(session):
    """
    Calculates the median time spent driving down the pit lane for all pit stops 
    in a session to establish the 'pit_loss_delta' accurately.
    """
    laps = session.laps
    
    # In Lap: Timestamp entering the pitlane
    in_laps = laps[~laps['PitInTime'].isna()][['Driver', 'Stint', 'LapNumber', 'PitInTime']]
    # Out Lap: Timestamp exiting the pitlane
    out_laps = laps[~laps['PitOutTime'].isna()][['Driver', 'Stint', 'LapNumber', 'PitOutTime']]
    
    # We match the out-lap (Stint=N) to its originating in-lap (Stint=N-1)
    in_laps['NextStint'] = in_laps['Stint'] + 1
    pit_stops = pd.merge(in_laps, out_laps, 
                         left_on=['Driver', 'NextStint'], 
                         right_on=['Driver', 'Stint'], 
                         suffixes=('_in', '_out'))
                         
    pit_stops['PitLaneTime'] = (pit_stops['PitOutTime'] - pit_stops['PitInTime']).dt.total_seconds()
    
    # F1 Median Pit Loss (Usually ~24-26 seconds for most tracks)
    median_pit_loss = pit_stops['PitLaneTime'].median()
    return median_pit_loss

def simulate_undercut_window(model, current_lap, current_state, track_env, pit_loss_delta, next_compound_encoded=3):
    """
    Executes the PRD Phase 4 simulation logic to forecast 15 laps ahead.
    Runs a "Stay Out" curve vs "Pit Now" curve to find the Crossover point.
    
    current_state dict requires: tyre_age, compound_encoded, baseline_pace
    track_env dataframe requires mapping for: track_evolution, TrackTemp for future laps
    """
    sim_horizon = 15
    stay_out_lap_times = []
    pit_now_lap_times = []
    feature_columns = ['tyre_age', 'compound_encoded', 'baseline_pace', 'track_evolution', 'TrackTemp']
    
    # Track environment dictionaries (to easily lookup future laps)
    evo_map = track_env.set_index('LapNumber')['track_evolution'].to_dict()
    temp_map = track_env.set_index('LapNumber')['TrackTemp'].to_dict()
    
    for n in range(sim_horizon):
        target_lap = current_lap + n
        
        # Fallback to last known conditions if simulating past the race end
        evo = evo_map.get(target_lap, list(evo_map.values())[-1])
        temp = temp_map.get(target_lap, list(temp_map.values())[-1])
        
        # 1. Stay Out Prediction
        # Features: tyre_age, compound_encoded, baseline_pace, track_evolution, TrackTemp
        stay_out_features = pd.DataFrame([{
            'tyre_age': current_state['tyre_age'] + n,
            'compound_encoded': current_state['compound_encoded'],
            'baseline_pace': current_state['baseline_pace'],
            'track_evolution': evo,
            'TrackTemp': temp
        }], columns=feature_columns)
        stay_out_degradation = model.predict(stay_out_features)[0]
        stay_out_predicted_time = current_state['baseline_pace'] + stay_out_degradation
        
        # Un-correct for fuel! (Since we want actual predicted lap times)
        # fuel_mass = 110 - (1.7 * lap_number), correction = mass * 0.03
        fuel_correction = (110.0 - (1.7 * target_lap)) * 0.03
        stay_out_actual = stay_out_predicted_time + fuel_correction
        stay_out_lap_times.append(stay_out_actual)
        
        # 2. Pit Now Prediction
        pit_features = pd.DataFrame([{
            'tyre_age': 1 + n,
            'compound_encoded': next_compound_encoded,
            'baseline_pace': current_state['baseline_pace'],
            'track_evolution': evo,
            'TrackTemp': temp
        }], columns=feature_columns)
        pit_degradation = model.predict(pit_features)[0]
        pit_predicted_time = current_state['baseline_pace'] + pit_degradation
        pit_actual = pit_predicted_time + fuel_correction
        
        # If it's the very first lap of the new stint, add the physical Pit Stop Loss time!
        if n == 0:
            pit_actual += pit_loss_delta
            
        pit_now_lap_times.append(pit_actual)

    # 3. Find Crossover Lap
    # An undercut is a cumulative race-time decision: the pit option must recover
    # the pit loss and become faster across the forecast horizon.
    stay_out_curve = pd.Series(stay_out_lap_times).cumsum().tolist()
    pit_now_curve = pd.Series(pit_now_lap_times).cumsum().tolist()
    crossover_relative = None
    for n in range(1, sim_horizon):  # Start from lap 1 since lap 0 includes pit loss
        if pit_now_curve[n] < stay_out_curve[n]:
            crossover_relative = n
            break

    crossover_lap = (current_lap + crossover_relative) if crossover_relative else None
    
    # Check if window is open (<= current lap + 3) as per PRD
    window_open = False
    if crossover_relative and crossover_relative <= 3:
        window_open = True
        
    return {
        'stay_out_curve': stay_out_curve,
        'pit_now_curve': pit_now_curve,
        'stay_out_lap_times': stay_out_lap_times,
        'pit_now_lap_times': pit_now_lap_times,
        'crossover_lap': crossover_lap,
        'window_open': window_open
    }

if __name__ == "__main__":
    from data_pipeline import main as run_pipeline
    from feature_engineering import apply_feature_engineering
    from ml_model import prepare_ml_data, train_degradation_model
    from fastf1 import get_session
    import fastf1
    
    # 1. Grab model and data
    clean_laps = run_pipeline()
    engineered_data = apply_feature_engineering(clean_laps)
    ml_data = prepare_ml_data(engineered_data)
    model, features = train_degradation_model(ml_data)
    
    # 2. Get Pit Loss Delta directly from session
    session = fastf1.get_session(2023, 'Bahrain', 'R')
    session.load(weather=False, messages=False) # Laps already loaded
    
    print("\n--- Starting Phase 4 (Simulation Logic) ---")
    pit_loss = calculate_pit_loss(session)
    print(f"Calculated dynamic Pit Loss Delta: {pit_loss:.3f} seconds")
    
    # 3. Simulate scenarios!
    # Pick a random lap where Verstappen is on old softs (Lap 12)
    ver_lap_12 = ml_data[(ml_data['Driver'] == 'VER') & (ml_data['LapNumber'] == 12)].iloc[0]
    
    current_state = {
        'tyre_age': ver_lap_12['tyre_age'],
        'compound_encoded': ver_lap_12['compound_encoded'],
        'baseline_pace': ver_lap_12['baseline_pace']
    }
    
    # environmental lookup df
    track_env = ml_data[['LapNumber', 'track_evolution', 'TrackTemp']].drop_duplicates('LapNumber')
    
    print(f"\nSimulating for VER going into Lap 12 on {current_state['tyre_age']} lap old tyres...")
    sim_result = simulate_undercut_window(model, 
                                          current_lap=12, 
                                          current_state=current_state, 
                                          track_env=track_env, 
                                          pit_loss_delta=pit_loss,
                                          next_compound_encoded=3) # Simulating pitting for Hards
                                          
    print("\n--- Simulation Results ---")
    for lap_offset in range(5):
        lap = 12 + lap_offset
        st_out = sim_result['stay_out_curve'][lap_offset]
        p_now = sim_result['pit_now_curve'][lap_offset]
        print(f"Lap {lap}: Stay Out Pace = {st_out:.2f}s | Pit Now Pace = {p_now:.2f}s")
        
    print(f"\nCrossover Lap Detected: {sim_result['crossover_lap']}")
    print(f"Undercut Window Open? : {'YES' if sim_result['window_open'] else 'NO'}")
