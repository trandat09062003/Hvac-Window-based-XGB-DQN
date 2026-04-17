import os
import sys

# Ensure EnergyPlus environment is set before importing Sinergym functionality
if 'EPLUS_PATH' not in os.environ:
    os.environ['EPLUS_PATH'] = '/usr/local/EnergyPlus-25-2-0'
if os.environ['EPLUS_PATH'] not in sys.path:
    sys.path.insert(0, os.environ['EPLUS_PATH'])

import gymnasium as gym
import sinergym
import pandas as pd
import numpy as np
import argparse


def generate_data(env_name, episodes, max_steps, weather_file=None):
    print(f"Initializing Environment: {env_name}")
    if weather_file:
        weather_file_abs = os.path.abspath(weather_file)
        print(f"Using custom weather file (absolute): {weather_file_abs}")
        # Sinergym expects 'weather_files' as a list or a single string depending on version
        # Based on error log, the keyword is 'weather_files'
        env = gym.make(env_name, weather_files=[weather_file_abs])
    else:
        env = gym.make(env_name)
    
    # Identify observation mapping from Sinergym
    obs_names = list(env.unwrapped.observation_variables)
    
    transition_data = []

    for ep in range(episodes):
        print(f"Starting Episode {ep + 1}/{episodes}")
        obs, info = env.reset()
        terminated = False
        truncated = False
        step = 0
        
        while not (terminated or truncated) and step < max_steps:
            # 1. Capture State
            state_dict = {f"State_{name}": val for name, val in zip(obs_names, obs)}
            
            # 2. Select Action (Using random sampling for robust state exploration, critical for offline RL)
            # Sinergym continuous action space: [heating_setpoint, cooling_setpoint]
            action = env.action_space.sample()
            
            # 3. Step Environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # 4. Capture Next State
            next_state_dict = {f"NextState_{name}": val for name, val in zip(obs_names, next_obs)}
            
            # 5. Assemble Transition Row
            row = {
                "Episode": ep + 1,
                "Step": step,
                "Action_Heating_Setpoint": action[0],
                "Action_Cooling_Setpoint": action[1],
                "Reward": reward,
                "Terminated": terminated
            }
            # Merge state and next_state dictionaries
            row.update(state_dict)
            row.update(next_state_dict)
            
            transition_data.append(row)
            obs = next_obs
            step += 1
            
            if step % 200 == 0:
                print(f"  ... Running Step {step}/{max_steps}")
                
    env.close()
    
    # Write to CSV
    output_filename = "Sinergym_Transition_Data.csv"
    df = pd.DataFrame(transition_data)
    df.to_csv(output_filename, index=False)
    print(f"\nData generation complete! Saved {len(df)} transition rows to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sequence data for offline RL")
    parser.add_argument('--env', type=str, default='Eplus-5zone-mixed-continuous-v1', help='Sinergym Env Name')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--weather', type=str, default=None, help='Path to custom EPW weather file')
    args = parser.parse_args()
    
    generate_data(args.env, args.episodes, args.max_steps, args.weather)
