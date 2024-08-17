import streamlit as st
import gymnasium as gym
import gym_anytrading

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

st.title('Stock Prediction Model')

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Display the trading environment settings
    st.sidebar.header("Trading Environment Settings")
    frame_bound_start = st.sidebar.number_input("Evaluation Frame Bound Start", min_value=0, max_value=len(df)-1, value=5)
    frame_bound_end = st.sidebar.number_input("Evaluation Frame Bound End", min_value=0, max_value=len(df)-1, value=100)
    
    window_size = 5  # Fixed window size

    if st.sidebar.button("Run Model"):
        # Fixed training frame bound
        training_frame_bound_start = 5
        training_frame_bound_end = 100
        
        # Create training environment
        env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=(training_frame_bound_start, training_frame_bound_end), window_size=window_size)
        env = DummyVecEnv([env_maker])
        
        # Train the model
        model = DQN('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=100000)
        
        # Create evaluation environment with user-provided frame bounds
        env = gym.make('stocks-v0', df=df, frame_bound=(frame_bound_start, frame_bound_end), window_size=window_size)
        obs, info = env.reset()
        while True:
            obs = obs[np.newaxis, ...]
            action, _states = model.predict(obs)    
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                st.write("Prediction Info:")
                st.write(info)
                break
        
        fig, ax = plt.subplots(figsize=(15, 6))
        env.render_all()
        st.pyplot(fig)
