import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from huggingface_hub import hf_hub_download
from pathlib import Path
import json
import tempfile
import os

# -------------------- CONFIG --------------------
DEFAULT_HF_REPO = ""  # set to your HF repo like "username/stock-models" or leave empty
MODEL_PREFIX = "dqn_"  # file name pattern: dqn_<TICKER>.zip
CACHE_DIR = Path("./models_cache")
CACHE_DIR.mkdir(exist_ok=True)


# -------------------- HELPERS --------------------
def load_model_local(ticker: str) -> DQN:
    """
    Try to load a model from the local cache directory.
    """
    p = CACHE_DIR / f"{MODEL_PREFIX}{ticker.upper()}.zip"
    if p.exists():
        return DQN.load(str(p))
    raise FileNotFoundError("Local model not found")


def download_model_from_hf(ticker: str, hf_repo: str) -> DQN:
    """
    Download the model file from HF Hub and save it to the local cache.
    hf_repo must be like 'username/repo_name'
    """
    if not hf_repo:
        raise FileNotFoundError("HF repo not configured")
    filename = f"{MODEL_PREFIX}{ticker.upper()}.zip"
    try:
        path = hf_hub_download(repo_id=hf_repo, filename=filename)
        dest = CACHE_DIR / filename
        # move or replace if path != dest
        if Path(path) != dest:
            Path(path).rename(dest)
        return DQN.load(str(dest))
    except Exception as e:
        raise FileNotFoundError(f"Could not download model from HF Hub: {e}")


@st.cache_resource
def load_model_for_ticker(ticker: str, hf_repo: str = "") -> DQN:
    """
    Return a DQN model for the given ticker. Try local first, then HF if provided.
    st.cache_resource ensures the loaded model is cached across runs.
    """
    ticker = ticker.upper()
    try:
        return load_model_local(ticker)
    except FileNotFoundError:
        if hf_repo:
            return download_model_from_hf(ticker, hf_repo)
        raise


@st.cache_data
def prepare_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance and compute simple features.
    Returns a DataFrame with:
      - open, high, low, close, volume
      - pct_change (close returns)
      - sma_short (5), sma_long (20)
      - volatility (rolling std of returns)
    NOTE: you should adapt feature engineering to match what your model expects.
    """
    ticker = ticker.upper()
    df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    # ensure standard columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )

    # basic features
    df["return"] = df["close"].pct_change().fillna(0)
    df["sma_short"] = df["close"].rolling(window=5, min_periods=1).mean()
    df["sma_long"] = df["close"].rolling(window=20, min_periods=1).mean()
    df["volatility"] = df["return"].rolling(window=10, min_periods=1).std().fillna(0)

    # Normalize or scale features if your model was trained on scaled data.
    # Here we perform a simple z-score using rolling mean/std to avoid lookahead:
    df["ret_z"] = (df["return"] - df["return"].rolling(window=20, min_periods=1).mean()) / (
        df["return"].rolling(window=20, min_periods=1).std().replace(0, 1)
    )
    df = df.fillna(0)

    return df


def create_observation_from_row(row: pd.Series, feature_order: list) -> np.ndarray:
    """
    Convert a DataFrame row into a 1D numpy observation in the order expected by the model.
    IMPORTANT: Ensure this ordering and the features exactly match the features used when training the model.
    """
    obs = np.array([row[f] for f in feature_order], dtype=np.float32)
    return obs


def predict_action(model: DQN, obs: np.ndarray, deterministic: bool = True):
    """
    Use a loaded stable-baselines3 DQN model to predict an action.
    The obs must be shaped appropriately for model.predict: typically a 1D array works.
    """
    # model.predict expects a single observation or batch. stable-baselines3 will accept a 1D array.
    action, _states = model.predict(obs, deterministic=deterministic)
    return action


# -------------------- STREAMLIT APP --------------------
st.set_page_config(page_title="DQN Stock Model Loader", layout="wide")

st.title("DQN model loader & quick predictor")
st.markdown(
    "Load a trained `DQN` model (local or from Hugging Face), fetch recent price data, "
    "and run a quick single-step prediction. Make sure feature-engineering matches training."
)

with st.sidebar:
    ticker = st.text_input("Ticker (e.g. AAPL)", value="AAPL").upper()
    period = st.selectbox("History period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    interval = st.selectbox("Interval", options=["1d", "1h", "30m"], index=0)
    hf_repo = st.text_input("HF repo (optional)", value=DEFAULT_HF_REPO)
    load_from_hf = st.checkbox("Try HF if local model missing", value=bool(DEFAULT_HF_REPO))
    st.markdown("---")
    st.write("Model naming convention: `dqn_<TICKER>.zip` (case-insensitive)")
    st.write(f"Cache dir: `{CACHE_DIR.resolve()}`")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Market Data (features)")
    try:
        df = prepare_data(ticker, period=period, interval=interval)
        st.dataframe(df.tail(10))
    except Exception as e:
        st.error(f"Error fetching/preparing data: {e}")
        st.stop()

with col2:
    st.header("Model")
    model_loaded = None
    try:
        if st.button("Load model"):
            # Attempt local first, then HF if checkbox enabled
            try:
                model_loaded = load_model_for_ticker(ticker, hf_repo if load_from_hf else "")
                st.success("Model loaded successfully (local or HF).")
            except FileNotFoundError as e_local:
                st.warning(f"Local model not found. {e_local}")
                if load_from_hf and hf_repo:
                    try:
                        model_loaded = download_model_from_hf(ticker, hf_repo)
                        st.success("Model downloaded from HF and loaded.")
                    except Exception as e_hf:
                        st.error(f"Failed to download/load from HF: {e_hf}")
                else:
                    st.info("Either enable HF fallback or place a model file in the cache directory.")
    except Exception as e:
        st.error(f"Unexpected error while loading model: {e}")

    # If model has been loaded earlier in the session, cache_resource will keep it.
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = None
    if model_loaded is not None:
        st.session_state.model_loaded = model_loaded

    if st.session_state.model_loaded is not None:
        st.write("✅ Model present in session.")
        st.write(st.session_state.model_loaded)
    else:
        st.write("No model in session yet.")

st.markdown("---")
st.header("Quick single-step prediction")

# Define the feature order expected by your model during training.
# MUST match training-time order. Update this to match your model.
feature_order = ["close", "return", "sma_short", "sma_long", "volatility", "ret_z"]

if st.button("Run prediction on latest data"):
    if st.session_state.model_loaded is None:
        st.error("No model loaded. Please load a model first.")
    else:
        latest_row = df.iloc[-1]
        obs = create_observation_from_row(latest_row, feature_order)
        # If your model expects a different shape (e.g. [1, n]), wrap with np.expand_dims(obs, 0)
        try:
            action = predict_action(st.session_state.model_loaded, obs, deterministic=True)
            st.success(f"Predicted action: {action}")
            st.json({"ticker": ticker, "action": int(action), "features_used": feature_order})
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.caption(
                "If prediction fails, check that the observation shape & feature order exactly match what the model "
                "was trained on. DQN models generally expect the same features and scaling as during training."
            )

st.markdown("---")
st.write("Notes:")
st.write(
    "- This app provides a quick way to load models and run single-step predictions. "
    "It **does not** recreate the training environment (gym env) used for training. "
)
st.write(
    "- Make sure the feature engineering, scaling, and feature order here mirror the training pipeline. "
    "Mismatch will lead to incorrect predictions or runtime errors."
)
