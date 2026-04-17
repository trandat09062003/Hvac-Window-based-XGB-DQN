import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import gymnasium as gym
import sinergym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

# Ensure EnergyPlus environment
if 'EPLUS_PATH' not in os.environ:
    os.environ['EPLUS_PATH'] = '/usr/local/EnergyPlus-25-2-0'
if os.environ['EPLUS_PATH'] not in sys.path:
    sys.path.insert(0, os.environ['EPLUS_PATH'])

# Re-import training components
from train_rl import DQN, DQNAgent, map_action

def calculate_pmv(temp, rh):
    """Simplified PMV calculation for demo purposes (Fanger model proxy).
    Assumes fixed metabolic rate (1.2 met) and clothing (1.0 clo in winter, 0.5 in summer).
    This is a rough approximation based on ASHRAE-55.
    """
    # comfort range is roughly 22-24C.
    pmv = (temp - 23.0) * 0.5 + (rh - 50.0) * 0.01
    return np.clip(pmv, -3, 3)

def evaluate(episodes, weather_file, model_path, max_steps=1000):
    env_name = 'Eplus-5zone-mixed-continuous-v1'
    weather_file_abs = os.path.abspath(weather_file)
    env = gym.make(env_name, weather_files=[weather_file_abs])
    
    obs_names = list(env.unwrapped.observation_variables)
    indices = {
        'month': obs_names.index('month'),
        'hour': obs_names.index('hour'),
        'o_temp': obs_names.index('outdoor_temperature'),
        'o_rh': obs_names.index('outdoor_humidity'),
        'i_temp': obs_names.index('air_temperature'),
        'i_rh': obs_names.index('air_humidity'),
        'occ': obs_names.index('people_occupant'),
        'power': obs_names.index('HVAC_electricity_demand_rate')
    }
    
    state_dim = 8
    num_actions = 24
    agent = DQNAgent(state_dim, num_actions)
    if os.path.exists(model_path):
        agent.model.build((None, state_dim))
        agent.model.load_weights(model_path)
        print(f"Loaded model from {model_path}")
    
    # Storage for comparison
    results = {
        'RL': {'energy': [], 'indoor_temp': [], 'outdoor_temp': [], 'pmv': [], 'reward': []},
        'Baseline': {'energy': [], 'indoor_temp': [], 'outdoor_temp': [], 'pmv': [], 'reward': []}
    }

    # Run RL Agent
    print("\nRunning Evaluation: RL Agent...")
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            state = np.array([obs[indices[k]] for k in indices.keys()])
            # for evaluation we set epsilon to 0
            old_eps = agent.epsilon
            agent.epsilon = 0
            action_idx = agent.act(state)
            agent.epsilon = old_eps
            
            action = map_action(action_idx)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))
            
            results['RL']['energy'].append(obs[indices['power']])
            results['RL']['indoor_temp'].append(obs[indices['i_temp']])
            results['RL']['outdoor_temp'].append(obs[indices['o_temp']])
            results['RL']['pmv'].append(calculate_pmv(obs[indices['i_temp']], obs[indices['i_rh']]))
            results['RL']['reward'].append(reward)
            done = terminated or truncated
            step += 1

    # Run Baseline (Constant Setpoint 21C / 25C)
    print("Running Evaluation: Baseline (Constant 21C/25C)...")
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            action = [21.0, 25.0] # Constant baseline
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))
            
            results['Baseline']['energy'].append(obs[indices['power']])
            results['Baseline']['indoor_temp'].append(obs[indices['i_temp']])
            results['Baseline']['outdoor_temp'].append(obs[indices['o_temp']])
            results['Baseline']['pmv'].append(calculate_pmv(obs[indices['i_temp']], obs[indices['i_rh']]))
            results['Baseline']['reward'].append(reward)
            done = terminated or truncated
            step += 1

    env.close()

    # Calculate metrics
    rl_energy = np.mean(results['RL']['energy'])
    bs_energy = np.mean(results['Baseline']['energy'])
    rl_pmv_abs = np.mean(np.abs(results['RL']['pmv']))
    bs_pmv_abs = np.mean(np.abs(results['Baseline']['pmv']))

    print("\n" + "="*40)
    print("EVALUATION RESULTS (Vietnam Climate)")
    print("="*40)
    print(f"Avg RL Power Demand:       {rl_energy:.2f} W")
    print(f"Avg Baseline Power Demand: {bs_energy:.2f} W")
    print(f"Energy Saving:             {(bs_energy - rl_energy)/bs_energy*100:.2f} %")
    print("-"*40)
    print(f"Avg RL Comfort (PMV abs):  {rl_pmv_abs:.2f} (lower is better)")
    print(f"Avg Baseline PMV abs:      {bs_pmv_abs:.2f}")
    print("="*40)

    # Plotting
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(results['RL']['indoor_temp'][:200], label='RL Indoor Temp', color='blue')
    plt.plot(results['Baseline']['indoor_temp'][:200], label='Baseline Indoor Temp', color='red', linestyle='--')
    plt.plot(results['RL']['outdoor_temp'][:200], label='Outdoor Temp', color='gray', alpha=0.5)
    plt.axhline(22, color='green', linestyle=':', alpha=0.5, label='Comfort Range')
    plt.axhline(26, color='green', linestyle=':', alpha=0.5)
    plt.title('Indoor Temperature Comparison (First 200 steps)')
    plt.legend()
    plt.ylabel('Temp (C)')

    plt.subplot(3, 1, 2)
    plt.plot(results['RL']['energy'][:200], label='RL Power', color='blue')
    plt.plot(results['Baseline']['energy'][:200], label='Baseline Power', color='red', linestyle='--')
    plt.title('Power Demand Comparison')
    plt.legend()
    plt.ylabel('Power (W)')

    plt.subplot(3, 1, 3)
    plt.plot(results['RL']['pmv'][:200], label='RL PMV', color='blue')
    plt.plot(results['Baseline']['pmv'][:200], label='Baseline PMV', color='red', linestyle='--')
    plt.title('Comfort Index (PMV) Comparison')
    plt.axhline(0, color='black', alpha=0.3)
    plt.legend()
    plt.ylabel('PMV')

    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    print("\nComparison plots saved to 'performance_comparison.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weather', type=str, required=True)
    parser.add_argument('--model', type=str, default='models/dqn_hvac_vietnam.weights.h5')
    args = parser.parse_args()
    
    evaluate(1, args.weather, args.model)
