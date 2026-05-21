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

def simulate_undercut_window(model, current_lap, current_state, track_env, pit_loss_delta, next_compound_encoded=3, feature_cols=None):
    """
    Executes the simulation logic to forecast 15 laps ahead.
    Calculates the Net Undercut Advantage (time gained against a rival who stays out).
    """
    if feature_cols is None:
        feature_cols = ['tyre_age', 'compound_encoded', 'baseline_pace', 'track_evolution', 'TrackTemp', 'Track', 'Driver']
        
    sim_horizon = 15
    stay_out_lap_times = []
    pit_now_lap_times = []
    
    evo_map = track_env.set_index('LapNumber')['track_evolution'].to_dict()
    temp_map = track_env.set_index('LapNumber')['TrackTemp'].to_dict()
    
    for n in range(sim_horizon):
        target_lap = current_lap + n
        
        evo = evo_map.get(target_lap, list(evo_map.values())[-1])
        temp = temp_map.get(target_lap, list(temp_map.values())[-1])
        
        # 1. Stay Out Prediction
        stay_out_df = pd.DataFrame([{
            'tyre_age': current_state['tyre_age'] + n,
            'compound_encoded': current_state['compound_encoded'],
            'baseline_pace': current_state['baseline_pace'],
            'track_evolution': evo,
            'TrackTemp': temp,
            'Track': current_state.get('Track', 'Bahrain'),
            'Driver': current_state.get('Driver', 'VER')
        }], columns=feature_cols)
        
        # Cast categorical features if using XGBoost global model
        if 'Track' in stay_out_df.columns:
            stay_out_df['Track'] = stay_out_df['Track'].astype('category')
            stay_out_df['Driver'] = stay_out_df['Driver'].astype('category')
            
        stay_out_degradation = model.predict(stay_out_df)[0]
        stay_out_predicted_time = current_state['baseline_pace'] + stay_out_degradation
        
        fuel_correction = (110.0 - (1.7 * target_lap)) * 0.03
        stay_out_actual = stay_out_predicted_time + fuel_correction
        stay_out_lap_times.append(stay_out_actual)
        
        # 2. Pit Now Prediction
        pit_df = pd.DataFrame([{
            'tyre_age': 1 + n,
            'compound_encoded': next_compound_encoded,
            'baseline_pace': current_state['baseline_pace'],
            'track_evolution': evo,
            'TrackTemp': temp,
            'Track': current_state.get('Track', 'Bahrain'),
            'Driver': current_state.get('Driver', 'VER')
        }], columns=feature_cols)
        
        if 'Track' in pit_df.columns:
            pit_df['Track'] = pit_df['Track'].astype('category')
            pit_df['Driver'] = pit_df['Driver'].astype('category')
            
        pit_degradation = model.predict(pit_df)[0]
        pit_predicted_time = current_state['baseline_pace'] + pit_degradation
        pit_actual = pit_predicted_time + fuel_correction
        pit_now_lap_times.append(pit_actual)

    # Calculate Undercut Advantage
    # How much time do you gain each lap over a rival who stays out?
    lap_advantage = [stay_out_lap_times[i] - pit_now_lap_times[i] for i in range(sim_horizon)]
    
    # Cumulative advantage (if rival pits n laps later, this is how many seconds you jumped them by)
    net_undercut_advantage = pd.Series(lap_advantage).cumsum().tolist()
    
    # Window is open if you can gain more than 1.5 seconds over the next 3 laps
    window_open = net_undercut_advantage[2] >= 1.5 if len(net_undercut_advantage) >= 3 else False
    
    # Crossover Lap: The lap where your OLD tyres become slower than FRESH tyres (Advantage becomes positive)
    crossover_relative = None
    for n in range(sim_horizon):
        if lap_advantage[n] > 0:
            crossover_relative = n
            break

    crossover_lap = (current_lap + crossover_relative) if crossover_relative is not None else None
    
    # For UI compatibility with previous version, we also return the raw cumulative curves
    # (But now the app can plot the 'net_undercut_advantage' for a much better view!)
    stay_out_curve = pd.Series(stay_out_lap_times).cumsum().tolist()
    pit_now_curve = pd.Series([t + (pit_loss_delta if i==0 else 0) for i, t in enumerate(pit_now_lap_times)]).cumsum().tolist()

    return {
        'stay_out_curve': stay_out_curve,
        'pit_now_curve': pit_now_curve,
        'stay_out_lap_times': stay_out_lap_times,
        'pit_now_lap_times': pit_now_lap_times,
        'net_undercut_advantage': net_undercut_advantage,
        'crossover_lap': crossover_lap,
        'window_open': window_open
    }

if __name__ == "__main__":
    import joblib
    import os
    import fastf1
    from data_pipeline import load_race_session, clean_lap_data, merge_weather_data
    from feature_engineering import apply_feature_engineering
    from ml_model import prepare_ml_data
    
    model_path = os.path.join('models', 'global_degradation_model.joblib')
    if not os.path.exists(model_path):
        print("Global model not found. Run train_global_model.py first.")
        exit()
        
    print("Loading global model...")
    saved_obj = joblib.load(model_path)
    model = saved_obj['model']
    features = saved_obj['features']
    
    session = load_race_session(2023, 'Bahrain')
    clean_laps = clean_lap_data(session.laps)
    final_laps = merge_weather_data(clean_laps, session.weather_data)
    engineered_data = apply_feature_engineering(final_laps)
    ml_data = prepare_ml_data(engineered_data)
    
    pit_loss = calculate_pit_loss(session)
    print(f"Calculated dynamic Pit Loss Delta: {pit_loss:.3f} seconds")
    
    ver_lap_12 = ml_data[(ml_data['Driver'] == 'VER') & (ml_data['LapNumber'] == 12)].iloc[0]
    current_state = {
        'tyre_age': ver_lap_12['tyre_age'],
        'compound_encoded': ver_lap_12['compound_encoded'],
        'baseline_pace': ver_lap_12['baseline_pace'],
        'Track': 'Bahrain',
        'Driver': 'VER'
    }
    
    track_env = ml_data[['LapNumber', 'track_evolution', 'TrackTemp']].drop_duplicates('LapNumber')
    
    sim_result = simulate_undercut_window(model, 12, current_state, track_env, pit_loss, 3, feature_cols=features)
                                          
    print("\n--- Simulation Results ---")
    for lap_offset in range(5):
        lap = 12 + lap_offset
        st_out = sim_result['stay_out_lap_times'][lap_offset]
        p_now = sim_result['pit_now_lap_times'][lap_offset]
        adv = sim_result['net_undercut_advantage'][lap_offset]
        print(f"Lap {lap}: Stay Out Pace = {st_out:.2f}s | Pit Now Pace = {p_now:.2f}s | Net Advantage = {adv:.2f}s")
        
    print(f"\nCrossover Lap Detected: {sim_result['crossover_lap']}")
    print(f"Undercut Window Open? : {'YES' if sim_result['window_open'] else 'NO'}")
