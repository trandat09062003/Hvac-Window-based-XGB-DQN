import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import gymnasium as gym
import sinergym
import random
from collections import deque
import argparse

# Ensure EnergyPlus environment
if 'EPLUS_PATH' not in os.environ:
    os.environ['EPLUS_PATH'] = '/usr/local/EnergyPlus-25-2-0'
if os.environ['EPLUS_PATH'] not in sys.path:
    sys.path.insert(0, os.environ['EPLUS_PATH'])

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# DQN Model
class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

class DQNAgent:
    def __init__(self, state_dim, num_actions, learning_rate=0.001, gamma=0.9, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.model = DQN(num_actions)
        self.target_model = DQN(num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.update_target_network()

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.num_actions)
        state = np.array([state], dtype=np.float32)
        q_values = self.model(state)
        return np.argmax(q_values[0])

    def train(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        next_q_values = self.target_model(next_states)
        max_next_q = tf.reduce_max(next_q_values, axis=1)
        target_q = rewards + (1 - dones) * self.gamma * max_next_q

        with tf.GradientTape() as tape:
            current_q_values = self.model(states)
            masks = tf.one_hot(actions, self.num_actions)
            current_q = tf.reduce_sum(current_q_values * masks, axis=1)
            loss = self.loss_fn(target_q, current_q)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def map_action(action_idx):
    """Map discrete action index to Sinergym continuous action [heating, cooling]."""
    # Action 0: Off-like (15, 30)
    if action_idx == 0:
        return [15.0, 30.0]
    # Simple linear mapping for demo: 
    # heating: 19 + idx, cooling: 21 + idx (to avoid overlap)
    # We'll adapt to stay within Sinergym bounds: [12, 23.25] and [23.25, 30]
    heating = 15.0 + (action_idx % 8)  # 15 to 22
    cooling = 24.0 + (action_idx // 3) # 24 to 30
    return [min(heating, 23.0), max(cooling, 24.0)]

def train(episodes, max_steps, weather_file):
    env_name = 'Eplus-5zone-mixed-continuous-v1'
    weather_file_abs = os.path.abspath(weather_file)
    env = gym.make(env_name, weather_files=[weather_file_abs])
    
    # State: [month, hour, outdoor_temp, outdoor_humidity, indoor_temp, indoor_humidity, occupants, power]
    # We need to map Sinergym obs to this simplified vector
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
    replay_buffer = ReplayBuffer(10000)
    batch_size = 32
    
    total_rewards = []

    for ep in range(episodes):
        obs, info = env.reset()
        state = np.array([obs[indices[k]] for k in indices.keys()])
        ep_reward = 0
        
        for step in range(max_steps):
            action_idx = agent.act(state)
            action = map_action(action_idx)
            # action must be a numpy array for Sinergym and clipped to bounds
            action = np.array(action, dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = np.array([next_obs[indices[k]] for k in indices.keys()])
            done = terminated or truncated
            
            replay_buffer.push(state, action_idx, reward, next_state, done)
            agent.train(replay_buffer, batch_size)
            
            state = next_state
            ep_reward += reward
            
            if done:
                break
        
        agent.update_target_network()
        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}/{episodes}, Reward: {ep_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()
    
    # Save model
    os.makedirs('models', exist_ok=True)
    agent.model.save_weights('models/dqn_hvac_vietnam.weights.h5')
    print("Model saved to models/dqn_hvac_vietnam.weights.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--weather', type=str, required=True)
    args = parser.parse_args()
    
    train(args.episodes, args.steps, args.weather)
