import streamlit as st
import gymnasium as gym
import gym_anytrading

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

st.title('Stock Prediction Model using RL')

# Use st.cache_resource to train the model only once
@st.cache_resource
def train_model(df):
    st.write("Training model... this will only run once.")
    # Fixed training frame bound
    training_frame_bound_start = 5
    training_frame_bound_end = 150
    window_size = 5
    
    # Create training environment
    env_maker = lambda: gym.make(
        'stocks-v0', 
        df=df, 
        frame_bound=(training_frame_bound_start, training_frame_bound_end), 
        window_size=window_size
    )
    env = DummyVecEnv([env_maker])
    
    # Train the model
    model = DQN('MlpPolicy', env, verbose=0) # Set verbose to 0 for a cleaner UI
    model.learn(total_timesteps=100000)
    return model

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    # Ensure 'Date' column exists before processing
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        except Exception as e:
            st.error(f"Error processing the 'Date' column: {e}")
            st.stop()
    else:
        st.error("CSV file must contain a 'Date' column.")
        st.stop()

    # Display the trading environment settings
    st.sidebar.header("Evaluation Settings")
    frame_bound_start = st.sidebar.number_input(
        "Evaluation Start Point", 
        min_value=0, 
        max_value=len(df)-1, 
        value=100
    )
    frame_bound_end = st.sidebar.number_input(
        "Evaluation End Point", 
        min_value=0, 
        max_value=len(df)-1, 
        value=len(df)-1
    )
    
    window_size = 5  # Fixed window size

    if st.sidebar.button("Run Evaluation"):
        # Train or load the cached model
        model = train_model(df)
        
        # Create evaluation environment with user-provided frame bounds
        st.write(f"Evaluating model from index {frame_bound_start} to {frame_bound_end}...")
        env = gym.make(
            'stocks-v0', 
            df=df, 
            frame_bound=(frame_bound_start, frame_bound_end), 
            window_size=window_size
        )
        obs, info = env.reset()

        while True:
            obs = obs[np.newaxis, ...]
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                st.write("### Evaluation Results")
                st.json(info) # Use st.json for nicely formatted dictionary output
                break
        
        # --- FIX IS HERE ---
        st.write("### Trading Visualization")
        plt.figure(figsize=(15, 6))
        plt.cla()  # Clear the plot to prevent overlapping visuals on re-runs
        env.unwrapped.render_all()
        st.pyplot(plt.gcf())  # gcf() gets the current figure that render_all() drew on