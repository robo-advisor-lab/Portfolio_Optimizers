# Quantitative Portfolio Optimizer

## Overview
The Quantitative Portfolio Optimizer is an advanced trading bot designed to dynamically identify and invest in the best-performing tokens based on quantitative analysis and reinforcement learning. The system leverages a dynamic SQL token classifier, reinforcement learning (RL) models, and on-chain execution to manage a portfolio that adapts to market conditions.

The optimizer is composed of three main components:

1. **SQL Token Classifier**: Dynamically identifies tokens based on multiple quantitative metrics.
2. **Reinforcement Learning Model**: Trains a PPO model to optimize portfolio allocation and rebalancing frequencies.
3. **On-Chain Execution**: Executes portfolio trades on-chain for the backtest duration, ensuring that the strategy adapts to the market dynamics.

## How it Works

### 1. Dynamic SQL Token Classifier

The system utilizes a powerful SQL query engine integrated with the Flipside API to classify tokens using the following metrics:
- **Returns Thresholding**: Identifies tokens with above-average returns.
- **Price Trends**: Filters tokens where the 7-day rolling average is above the 30-day rolling average, indicating a bullish trend.
- **Volume Analysis**: Includes only tokens with above-average trading volume compared to other tokens on Uniswap V3.
- **Meme Coin Filter**: Excludes high-volatility meme coins unless explicitly included.
- **Data Requirement**: Ensures that each token has at least 6 months of historical data to provide enough context for model training.
### Advanced Filtering:
- **Trending Tokens**: Optionally includes tokens trending upwards by comparing 7d and 30d rolling averages.
- **Volume Thresholding**: Filters tokens based on a volume threshold calculated relative to the average market volume.

### 2. Reinforcement Learning Model

Once the tokens are classified, the system moves on to the RL model training phase:

- **Model Architecture**: Utilizes Proximal Policy Optimization (PPO) from `Stable-Baselines3` in an OpenAI Gymnasium environment.
- **Custom Gym Environment**: The `Portfolio` environment simulates real-world trading conditions, including fees, slippage, and liquidity constraints.
- **Training and Testing Split**: Historical prices are split into a 66% training set and a 33% testing set to evaluate model generalization.
- **Parameter Grid Search**: Conducts grid search for optimal portfolio rebalancing frequency using normalized returns as the performance metric.
- **Sortino Ratio Calculation**: Calculates Sortino ratios to assess risk-adjusted returns, providing a more accurate evaluation for downside volatility.

### Rebalancing Frequency Optimization:

- **Grid Search Strategy**: Explores multiple rebalancing frequencies (e.g., 24h, 48h, 168h) to identify the most profitable configuration.
- **Selection Criteria**: Chooses the frequency with the highest normalized return, which is then cached for future use.

### 3. On-Chain Execution

The second app handles live on-chain execution for the backtest duration:

- **On-Chain Portfolio Management**: The portfolio is executed on the blockchain (currently on Arbitrum), matching the backtest duration.
- **Dynamic Strategy Duration**: The duration equals the backtest period, which is the time the model was trained on. This provides an estimate of the strategy's expected performance horizon.
- **Automated Portfolio Rebalancing**: The system automatically rebalances the portfolio at the selected frequency.
- **Portfolio Rotation**: Once the duration ends, the system:
    - Triggers the SQL classifier to obtain the latest token set.
    - Re-trains the RL model on the latest data.
    - Caches the new model and waits for the main app to sense the update.
    - Sells existing assets into WETH and trades into the new portfolio composition.

### 4. Automated Quantitative Research

The system automates quantitative research via:

- **Classifier Scripts**: Re-runs the SQL classifier at the end of each portfolio duration to ensure relevance to the latest market trends.
- **Trend Detection and Market Adaptation**: Ensures the portfolio dynamically adapts to market shifts, leveraging the most recent price, volume, and trend data.
- **Portfolio Expected Return**: The Portfolio Expected Return calculation utilizes the Capital Asset Pricing Model (CAPM) to estimate the annualized expected return for the AI-managed portfolio. This calculation compares the portfolio's performance to the market benchmark (DPI - DeFi Pulse Index) and adjusts for risk using the portfolio's beta.  

## Key Features
### Dynamic Token Classification

- Real-time classification using the latest market data.
- Volume and price trend filters for robust token selection.
- Automated exclusion of meme coins and low-liquidity tokens.

### Reinforcement Learning (RL) Model

- Utilizes PPO for stability and policy gradient efficiency.
- Custom Gym environment tailored to financial simulations.
- Parameter grid search for optimal rebalancing frequency.
- Sortino ratio for downside risk-adjusted performance evaluation.

### On-Chain Integration

- Runs on-chain for the backtest duration using Web3.py and Uniswap SDK.
- Trades executed using Uniswap V3 pools on Arbitrum.
- Portfolio dynamically adjusts to market conditions.

### Caching and Efficiency

- Utilizes `diskcache.Cache` for fast caching of model parameters and classifier results.
- Asynchronous design with `asyncio` and `httpx` for non-blocking HTTP requests.

## Architecture and Flow

### 1. Classifier Endpoint (`classifier_endpoint.py`)

- Constructs a dynamic SQL query using parameters for network, number of days to measure cumulative return, volume threshold, and backtest period.
- Queries data from the Flipside API and filters tokens with above-average returns, trending patterns, and high volume.
- The RL model is then trained on the portfolio and saved to be loaded by the main backend script.

### 2. Live Backend (`live_backend_v2.py`)

- Manages model versions and execution using asynchronous tasks.
- Retrieves cached classifier results and trains the RL model.
- Initializes the portfolio environment and runs the model using the selected rebalancing frequency.
- Deploys the trained model on-chain and monitors its performance.

## Technology Stack

- **Flask**: Backend framework for API and web server.
- **Pandas & NumPy**: Data manipulation and numerical calculations.
- **Stable-Baselines3**: RL library for PPO implementation.
- **OpenAI Gymnasium**: Custom environment for financial simulations.
- **Web3.py**: Blockchain integration for on-chain execution.
- **Uniswap-Python SDK**: Token swaps on Uniswap V3.
- **Flipside API**: Real-time market data.
- **Plotly**: Data visualization for portfolio performance.

## Future Roadmap

### Smart Contract Integration

- **Deposits and Withdrawals**: Smart contract to support proportional gains based on deposits, enabling exposure to the portfolio via portfolio share tokens.
- **On-Chain Rebalancing**: Fully on-chain execution and rebalancing using Solidity smart contracts.

### Advanced Classifiers and Model Enhancements

- **Sharpe Ratio Classifier**: Additional classifiers for Sharpe ratio and other risk metrics.
- **LLM Integration**: Use of language models (LLMs) to dynamically tune SQL classifier parameters.
- **RL Param Optimization**: Incorporation of advanced hyperparameter tuning using Optuna or Ray Tune.

### Token Liquidity and Market Integration

- **Liquidity on Uniswap**: Supporting liquidity for portfolio share tokens on Uniswap V3.
- **Third Market Integration**: Enabling secondary market trading of portfolio tokens.


## Deployed Contracts

- **Main Model**: Currently deployed on **Arbitrum** at the following address: [0x75baD5ae9f46e8AEBf61e4A7179cEf3A0CeD6766](https://arbiscan.io/address/0x75baD5ae9f46e8AEBf61e4A7179cEf3A0CeD6766)

## Getting Started

1. **Prerequisites**
    - Python 3.9+
2. **Installation**

```bash
pip install -r requirements.txt

```

1. **Configuration**
    - Set up environment variables for Flipside API key, wallet private key, and Web3 RPC URLs.
2. **Run the Classifier Endpoint**

```bash
python classifier_endpoint.py

```

3. **Start Live Backend**

```bash
python live_backend_v2.py

```


## Support and Community

For issues or questions, please contact:

- **Email**: [general@optimizerfinance.com](mailto:general@optimizerfinance.com)
- **Github**: https://github.com/robo-advisor-lab
- **Gitbook**: [Optimizer Finance Litepaper](https://robo-advisor-labs.gitbook.io/optimizer-finance-litepaper)
