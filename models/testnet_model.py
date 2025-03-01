import gymnasium as gym
from gymnasium import spaces
# from stable_baselines3 import PPO
# from scipy.optimize import minimize, Bounds, LinearConstraint
import plotly.graph_objs as go
import pandas as pd
import requests
import numpy as np
import yfinance as yf
import matplotlib

import random
import plotly.io as pio
# import cvxpy as cp
# import matplotlib.pyplot as plt
# from prophet import Prophet
from sklearn.metrics import r2_score, mean_absolute_error
# from stable_baselines3.common.vec_env import DummyVecEnv
# import torch
from flipside import Flipside
from dune_client.client import DuneClient

import os
from dotenv import load_dotenv

import datetime as dt
from datetime import timedelta
import pytz  # Import pytz if using timezones

from sklearn.linear_model import LinearRegression
import tensorflow as tf
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import matplotlib.pyplot as plt

from plotly.subplots import make_subplots

from python_scripts.utils import mvo,calculate_log_returns,calculate_portfolio_returns
    
def normalize_portfolio(portfolio):
    total = np.sum(portfolio)
    if total == 0:
        # If the total is zero, avoid division by zero and return an equally distributed portfolio
        return np.ones_like(portfolio) / len(portfolio)
    return portfolio / total

class Portfolio(gym.Env):
    def __init__(self, df, seed, risk_free_annual, compositions=None, rebalance_frequency=24):
        super(Portfolio, self).__init__()
        self.df = df
        self.current_step = 0
        self.total_assets = len(df.drop(columns='DXY', errors='ignore').columns)  # Number of assets
        self.seed(seed)
        self.prev_prices = None

        # Convert annual risk-free rate to hourly rate
        self.target_return = risk_free_annual  # Convert annual to hourly rate
        self.returns_log = pd.DataFrame(columns=['Return'])

        # Define action and observation space
        self.action_space = spaces.Box(low=0, high=1, shape=(self.total_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(df.columns),), dtype=np.float32)

        # Initialize portfolio and logs
        if compositions is None:
            self.portfolio = np.full(self.total_assets, 1.0 / self.total_assets)
        else:
            self.portfolio = compositions

        self.rebalance_frequency = rebalance_frequency
        self.steps_since_last_rebalance = 0

        # Logs
        self.states_log = []
        self.rewards_log = []
        self.actions_log = []
        self.sortino_ratios_log = []

        # Asset price columns
        self.price_columns = df.drop(columns='DXY', errors='ignore').columns.tolist()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.steps_since_last_rebalance = 0
        self.states_log = []
        self.rewards_log = []
        self.actions_log = []
        self.sortino_ratios_log = []

        obs = self._get_observation()
        return obs.astype(np.float32), {}

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _get_observation(self):
        if self.current_step > len(self.df):
            raise IndexError(f"Current step {self.current_step} is out of bounds for the dataframe.")
        obs = self.df.iloc[self.current_step].values
        if np.isnan(obs).any():
            raise ValueError("Observation contains NaN values.")
        return obs.astype(np.float32)

    def step(self, action):
        print(f"\n--- Step {self.current_step} ---")
        self.steps_since_last_rebalance += 1

        # Clip actions to ensure valid values
        action = np.clip(action, 0, 1)
        print(f"Action after clipping: {action}")

        if np.isnan(action).any():
            raise ValueError("Action contains NaN values.")

        # Get current asset prices
        prices = self.df.iloc[self.current_step][self.price_columns].values
        print(f"Current prices: {prices}")

        print(F'self.df: {self.df}')

        if self.prev_prices is None:
            self.prev_prices = prices

        print(f'Portfolio: {self.portfolio}')

        print(f'self.df: {self.df}')

        # Calculate log returns
        # price_returns = np.log(prices / self.prev_prices)
        print(f"self.df.iloc[:self.current_step][self.price_columns]: {self.df.iloc[:self.current_step + 1 ][self.price_columns]}")

        price_returns = calculate_log_returns(self.df.iloc[:self.current_step + 1 ][self.price_columns])

        print(f'price_returns:{price_returns}')
        print(f'self.portfolio:{self.portfolio}')
        
        # Calculate portfolio return as weighted sum of log returns
        portfolio_return = calculate_portfolio_returns(self.portfolio.iloc[:self.current_step + 1 ], price_returns)
        print(f'portfolio_return: {portfolio_return}')
        # Convert portfolio_return to DataFrame before appending

        # Append to returns_log
        self.returns_log = portfolio_return.reset_index().rename(columns={'index': 'Date', 0: 'Return'})
        # Rebalance portfolio if needed
        if self.steps_since_last_rebalance >= self.rebalance_frequency or self.current_step == 0:
            # import pdb; pdb.set_trace()
            print("Rebalancing portfolio...")
            self.steps_since_last_rebalance = 0

            # Handle single-asset or multiple-asset portfolios
            if len(action) == 1:
                # If there is only one asset, allocate 100% to it
                action_percentages = np.array([1.0])
            else:
                total_action = np.sum(action)
                if total_action > 0:
                    action_percentages = action / total_action
                else:
                    # If all actions are zero, default to an equally distributed portfolio
                    action_percentages = np.ones_like(action) / len(action)

            # Ensure the action percentages sum to 1 (handle floating-point errors)
            action_percentages = normalize_portfolio(action_percentages)

            print(f"Action percentages: {action_percentages}")

            self.actions_log.append((action_percentages, self.df.index[self.current_step]))

        # Calculate Sortino Ratio
        print(f'self.target_return: {self.target_return}')
        
        # self.portfolio_composition_log.append((self.portfolio.copy(), self.df.index[self.current_step]))

        # Calculate optimized weights and distance penalty using MVO
        print(f"full_data: {self.df.drop(columns='DXY')}")
        print(f"data with step condition: {self.df.drop(columns='DXY').iloc[:self.current_step + 1 ]}")
        print(f"data_at_mvo:{self.df.drop(columns='DXY'), self.portfolio, self.target_return}")
        optimized_weights, returns, sortino_ratio = mvo(self.df.drop(columns='DXY').iloc[:self.current_step + 1 ], self.portfolio.iloc[:self.current_step + 1 ], self.target_return)
        print(
            f'portfolio: {self.portfolio}, optimized_portfolio: {optimized_weights}, sortino_ratio: {sortino_ratio}'
        )

        print(F'sortino_ratio: {sortino_ratio}')

        if sortino_ratio is None:
            sortino_ratio = 0

        print(F'sortino_ratio: {sortino_ratio}')

        print(f'self.portfolio.iloc[-1]: {self.portfolio.iloc[-1]}')

        current_weights = self.portfolio.iloc[-1] if optimized_weights is not None else None
        max_distance = sum(abs(1 - value) for value in optimized_weights) if optimized_weights is not None else 0
        distance_penalty = sum(abs(current_weights[i] - optimized_weights[i]) for i in range(len(optimized_weights))) / max_distance if max_distance != 0 else 0

        print(f'portfolio_return: {portfolio_return}')
        # Reward calculation
        reward = portfolio_return.iloc[-1] * (sortino_ratio - distance_penalty)
        print(f"Reward: {reward}")

        print(f'self.df.iloc[self.current_step]: {self.df.iloc[self.current_step]}')
        print(f'self.df: {self.df}')

        # Update logs and state
        self.prev_prices = self.df.iloc[self.current_step][self.price_columns].values
        self.rewards_log.append((reward, self.df.index[self.current_step]))
        self.sortino_ratios_log.append((sortino_ratio, self.df.index[self.current_step]))
        self.current_step += 1

        done = self.current_step >= len(self.df) - 1

        if done:
            print(f'done: {done}')
            return None, reward, done, False, {}
        
        state = self._get_observation()

        print(f'done: {done}')
        return state.astype(np.float32), reward, done, False, {}

    # Functions to get logs
    def get_states_df(self):
        states, dates = zip(*self.states_log) if self.states_log else ([], [])
        return pd.DataFrame(states, columns=self.df.columns).assign(Date=dates)
    
    def get_actions_df(self):
        actions, dates = zip(*self.actions_log) if self.actions_log else ([], [])
        # Use asset names with "_weight" suffix for column names
        asset_names = [col.split('_Price')[0] for col in self.price_columns]
        column_names = [f'{asset}_weight' for asset in asset_names]
        return pd.DataFrame(actions, columns=column_names).assign(Date=dates)

    def get_rewards_df(self):
        rewards, dates = zip(*self.rewards_log) if self.rewards_log else ([], [])
        return pd.DataFrame(rewards, columns=['Reward']).assign(Date=dates)
    
    def get_returns_df(self):
        return self.returns_log.copy()

    def get_sortino_ratios_df(self):
        returns, dates = zip(*self.sortino_ratios_log) if self.sortino_ratios_log else ([], [])
        return pd.DataFrame(returns, columns=['Portfolio Hourly Sortino Ratios']).assign(Date=dates)