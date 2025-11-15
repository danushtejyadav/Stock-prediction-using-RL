import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
from gymnasium import spaces
from huggingface_hub import HfApi
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# -------------------- CONFIG --------------------
MODEL_PREFIX = "dqn_"  # file name pattern: dqn_<TICKER>.zip
CACHE_DIR = Path("./models_cache")
CACHE_DIR.mkdir(exist_ok=True)


# -------------------- SIMPLE PRICE ENV --------------------
class PriceEnv(gym.Env):
        metadata = {"render.modes": ["human"]}

        def __init__(self, df):
            super().__init__()
            self.df = df.reset_index(drop=True)
            # observation: [price, return, ma5, ma20]
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
            # action: 0=hold,1=buy,2=sell
            self.action_space = gym.spaces.Discrete(3)
            self.ptr = 0
            self.position = 0  # -1 short, 0 flat, 1 long
            self.cash = 0.0

        def reset(self, *, seed=None, options=None):
            """
            Gymnasium-compatible reset:
            - accepts seed and options (stable-baselines3 / DummyVecEnv will pass seed=)
            - returns (obs, info)
            """
            # optional: use seed to make env deterministic
            if seed is not None:
                np.random.seed(seed)

            self.ptr = 20  # start after rolling windows
            self.position = 0
            self.cash = 0.0
            obs = self._obs()
            info = {}
            return obs, info

        def _obs(self):
            row = self.df.iloc[self.ptr]
            return np.array([row.Close, row["return"], row.ma_5, row.ma_20], dtype=np.float32)

        def step(self, action):
            # simple reward: change in price * position
            prev_price = self.df.iloc[self.ptr].Close
            # apply action: change position
            if action == 1:
                self.position = 1
            elif action == 2:
                self.position = -1
            # else hold
            self.ptr += 1
            done = self.ptr >= len(self.df) - 1
            cur_price = self.df.iloc[self.ptr].Close
            reward = (cur_price - prev_price) * self.position
            info = {}
            # Gymnasium expects (obs, reward, terminated, truncated, info)
            # We don't implement truncated here, so set truncated=False.
            terminated = bool(done)
            truncated = False
            return self._obs(), float(reward), terminated, truncated, info

        def render(self, mode="human"):
            pass



def make_simple_env_from_df(df: pd.DataFrame, feature_cols=None) -> PriceEnv:
    return PriceEnv(df=df, feature_cols=feature_cols)


# ---------- Replace these helpers in train_local.py with the block below ----------

def prepare_df_for_training(ticker: str, period: str = "2y", feature_cols=None) -> pd.DataFrame:
    """
    Fetch price history and produce the default features.
    feature_cols can be used later if you want to customize which columns are used.
    """
    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        raise ValueError("No data fetched for ticker")
    df = df[["Close"]].copy()
    df["return"] = df["Close"].pct_change().fillna(0)
    # use bfill to avoid FutureWarning
    df["ma_5"] = df["Close"].rolling(5).mean().bfill()
    df["ma_20"] = df["Close"].rolling(20).mean().bfill()
    df = df.dropna().reset_index(drop=True)
    if feature_cols is None:
        feature_cols = ["Close", "return", "ma_5", "ma_20"]
    # Ensure ordered columns exist
    for c in feature_cols:
        if c not in df.columns:
            raise KeyError(f"Feature column {c} missing from dataframe")
    # keep only feature columns (and Close for rendering if needed)
    return df[feature_cols]


def make_simple_env_from_df(df: pd.DataFrame, feature_cols=None):
    """
    Return a PriceEnv instance that uses the given df and feature_cols.
    The PriceEnv implements Gymnasium API: reset(..., seed=...), step -> (obs, reward, terminated, truncated, info)
    """

    class PriceEnv(gym.Env):
        metadata = {"render.modes": ["human"]}

        def __init__(self, df, feature_cols=None):
            super().__init__()
            self.df = df.reset_index(drop=True)
            if feature_cols is None:
                # default ordering, if user passed full df this will match
                feature_cols_local = list(self.df.columns)
            else:
                feature_cols_local = feature_cols
            self.feature_cols = feature_cols_local
            # observation space dimension is length of feature_cols
            obs_dim = len(self.feature_cols)
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            # action: 0=hold,1=buy,2=sell
            self.action_space = gym.spaces.Discrete(3)
            self.ptr = 0
            self.position = 0  # -1 short, 0 flat, 1 long
            self.cash = 0.0

        def reset(self, *, seed=None, options=None):
            """
            Gymnasium-compatible reset: accepts seed, options and returns (obs, info)
            """
            if seed is not None:
                # affect any randomization you might add later
                np.random.seed(seed)
            # start after rolling windows - ensure we have enough history
            # choose start index such that we can compute rolling windows; keep it simple
            self.ptr = 20 if len(self.df) > 40 else 0
            self.position = 0
            self.cash = 0.0
            obs = self._obs()
            info = {}
            return obs, info

        def _obs(self):
            row = self.df.iloc[self.ptr]
            vals = [row[c] for c in self.feature_cols]
            return np.asarray(vals, dtype=np.float32)

        def step(self, action):
            prev_price = float(self.df.iloc[self.ptr]["Close"]) if "Close" in self.df.columns else float(self.df.iloc[self.ptr][0])
            # apply action: change position
            if int(action) == 1:
                self.position = 1
            elif int(action) == 2:
                self.position = -1
            # advance pointer
            self.ptr += 1
            done = self.ptr >= len(self.df) - 1
            cur_price = float(self.df.iloc[self.ptr]["Close"]) if "Close" in self.df.columns else float(self.df.iloc[self.ptr][0])
            reward = (cur_price - prev_price) * self.position
            info = {}
            terminated = bool(done)
            truncated = False
            return self._obs(), float(reward), terminated, truncated, info

        def render(self, mode="human"):
            # optional simple render
            print(f"ptr={self.ptr}, pos={self.position}")

    # instantiate with the provided feature_cols or default to df.columns
    return PriceEnv(df=df, feature_cols=feature_cols)

# ---------- End replacement block ----------


def train_and_save(ticker: str, timesteps: int, save_path: str, period: str = "2y"):
    """
    Train a DQN agent on the price data and save the model and metadata.
    """
    df = prepare_df_for_training(ticker, period)
    # features used for observation (must match env.feature_cols)
    feature_cols = ["Close", "return", "ma_5", "ma_20"]

    env = make_simple_env_from_df(df, feature_cols=feature_cols)
    env = DummyVecEnv([lambda: env])

    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    model_file = Path(save_path) / f"{MODEL_PREFIX}{ticker.upper()}.zip"
    model.save(str(model_file))

    # write metadata
    meta = {"ticker": ticker.upper(), "timesteps": timesteps, "period": period, "features": feature_cols}
    with open(Path(save_path) / f"meta_{ticker.upper()}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model to {model_file}")
    return model_file


# -------------------- HUGGINGFACE PUSH --------------------
def push_to_hf(repo_id: str, path_to_file: str, token: str = None):
    """
    Upload file to a HF repo using HfApi.upload_file.
    Note: For large models use git-lfs or huggingface-cli with proper LFS support.
    """
    api = HfApi()
    with open(path_to_file, "rb") as f:
        res = api.upload_file(
            path_or_fileobj=f,
            path_in_repo=Path(path_to_file).name,
            repo_id=repo_id,
            token=token,
        )
    return res


# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--save-path", default=str(CACHE_DIR))
    parser.add_argument("--period", default="2y")
    parser.add_argument("--push-hf", action="store_true", help="Upload model to HF Hub repo")
    parser.add_argument("--hf-repo", default="", help="HF repo id like username/repo")
    args = parser.parse_args()

    model_file = train_and_save(args.ticker, args.timesteps, args.save_path, period=args.period)

    if args.push_hf:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("HF_TOKEN not set in environment. Skipping push.")
        elif not args.hf_repo:
            print("--hf-repo required to push to HF Hub. Skipping push.")
        else:
            print("Uploading to HF Hub...")
            res = push_to_hf(args.hf_repo, str(model_file), token=hf_token)
            print("Upload response:", res)
