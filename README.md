# Stock Trading Demo with Reinforcement Learning

This project is a Streamlit web application that demonstrates how a Reinforcement Learning (RL) agent can be trained to make simple stock trading decisions. It uses a Deep Q-Network (DQN) model from `stable_baselines3` to learn a trading policy within the `gym-anytrading` environment. Historical stock data is fetched live using `yfinance`.

## ⚠️ Important Warning

This application is for **educational and demonstration purposes only**. The model is trained on a very limited number of timesteps (2,000) to allow it to run on free, resource-constrained servers (like Streamlit Community Cloud).

**These predictions are NOT suitable for real-world trading.** Real-world financial models require extensive training (often on millions of timesteps), rigorous backtesting, feature engineering, and validation.

[Try it out!](https://stock-prediction-using-rl-naa4qg9elcmoka97ffwwzf.streamlit.app/)

## 🚀 Features

* **Interactive UI:** Built with Streamlit, allowing easy user interaction.
* **Live Data:** Fetches historical stock data from `yfinance` using any valid ticker.
* **RL Model:** Trains a Deep Q-Network (DQN) agent from `stable_baselines3`.
* **Efficient Training:** Uses `st.cache_resource` to train the model only once per dataset, avoiding retraining on every widget interaction.
* **Evaluation & Visualization:** Evaluates the trained agent on a test dataset (the last 20% of the data by default) and visualizes the trading decisions (buy/sell signals) using `matplotlib`.

## 🛠️ Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://your-repository-url/stock-rl-demo.git
    cd stock-rl-demo
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required packages:**
    Create a `requirements.txt` file with the following content:
    ```plaintext
    streamlit
    gymnasium
    gym_anytrading
    yfinance
    stable_baselines3
    torch
    numpy
    pandas
    matplotlib
    ```
    Then, install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `stable_baselines3` requires PyTorch, which is included as `torch`.)*

## 🏃‍♂️ How to Run

From your terminal, navigate to the project directory and run the Streamlit app:

```bash
streamlit run stock_app.py
