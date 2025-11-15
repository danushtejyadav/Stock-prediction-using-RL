import streamlit as st
import gymnasium as gym
import gym_anytrading
import yfinance as yf

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

st.title('Stock Prediction Demo (RL)')

# --- MODIFICATION: ADD A WARNING ---
st.warning(
    "**Demonstration Only:** To run on a free server, model training is "
    "limited to 2,000 timesteps. These predictions are **not** suitable "
    "for real-world trading and are for technical demonstration purposes only."
)

# Use st.cache_resource to train the model only once per dataset
@st.cache_resource
def train_model(df):
    st.write("Training model... (This will run once per dataset & cache)")
    window_size = 5
    
    split_point = int(len(df) * 0.8)
    training_frame_bound_start = window_size
    training_frame_bound_end = split_point

    if training_frame_bound_end <= training_frame_bound_start:
        st.error("Not enough data to train. Please select a longer period.")
        return None

    # Create training environment
    env_maker = lambda: gym.make(
        'stocks-v0', 
        df=df, 
        frame_bound=(training_frame_bound_start, training_frame_bound_end), 
        window_size=window_size
    )
    env = DummyVecEnv([env_maker])
    
    # --- MODIFICATION: REDUCE TIMESTEPS ---
    # We change 100,000 to 2,000 to prevent a server crash on the free tier.
    # This is the "appropriate measure" to handle the memory/CPU limitation.
    model = DQN('MlpPolicy', env, verbose=0) 
    model.learn(total_timesteps=2000) # Reduced from 100,000
    
    st.write("Model training complete.")
    return model

# --- Sidebar for data fetching ---
st.sidebar.header("Data Source")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
period = st.sidebar.selectbox("Period", ["1y", "2y", "5y", "max"])

if st.sidebar.button("Fetch Data & Train Model"):
    try:
        # Fetch data using yfinance
        data = yf.Ticker(ticker)
        df = data.history(period=period)
        
        if df.empty:
            st.error("Could not fetch data. Check the ticker or try again.")
        else:
            st.write(f"Fetched {len(df)} data points for {ticker}.")
            st.write("Data Preview:")
            st.write(df.head())

            # Train or load the cached model
            model = train_model(df)
            
            if model is not None:
                # Store the model and data in session state to use in evaluation
                st.session_state.model = model
                st.session_state.df = df
                st.session_state.data_fetched = True
                st.sidebar.success("Model trained successfully!")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- Evaluation section only appears after model is trained ---
if 'data_fetched' in st.session_state and st.session_state.data_fetched:
    
    # Load data and model from session state
    df = st.session_state.df
    model = st.session_state.model

    # Set default evaluation frame (last 20%)
    split_point = int(len(df) * 0.8)

    st.sidebar.header("Evaluation Settings")
    frame_bound_start = st.sidebar.number_input(
        "Evaluation Start Point", 
        min_value=0, 
        max_value=len(df)-1, 
        value=split_point # Default to start of test set
    )
    frame_bound_end = st.sidebar.number_input(
        "Evaluation End Point", 
        min_value=0, 
        max_value=len(df)-1, 
        value=len(df)-1 # Default to end of data
    )
    
    window_size = 5  # Fixed window size

    if st.sidebar.button("Run Evaluation"):
        st.write(f"Evaluating model from index {frame_bound_start} to {frame_bound_end}...")
        
        # Create evaluation environment
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
        
        st.write("### Trading Visualization")
        plt.figure(figsize=(15, 6))
        plt.cla()  
        env.unwrapped.render_all()
        st.pyplot(plt.gcf())