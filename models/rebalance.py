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
import datetime as dt
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

from python_scripts.utils import mvo

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
        self.returns_log = []

        # Define action and observation space
        self.action_space = spaces.Box(low=0, high=1, shape=(self.total_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(df.columns),), dtype=np.float32)

        # Initialize portfolio and logs
        if compositions is None:
            self.portfolio = np.full(self.total_assets, 1.0 / self.total_assets)
        else:
            self.portfolio = compositions.iloc[-1].values

        print(f'self.portfolio: {self.portfolio}')

        self.rebalance_frequency = rebalance_frequency
        self.steps_since_last_rebalance = 0

        # Logs
        self.states_log = []
        self.rewards_log = []
        self.actions_log = []
        self.portfolio_composition_log = []
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
        self.portfolio_composition_log = []
        self.sortino_ratios_log = []

        obs = self._get_observation()
        return obs.astype(np.float32), {}

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _get_observation(self):
        if self.current_step >= len(self.df):
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
        current_prices = self.df.iloc[self.current_step][self.price_columns].values
        print(f"Current prices: {current_prices}")

        if self.prev_prices is None:
            self.prev_prices = current_prices

        print(f'Portfolio: {self.portfolio}')

        # Calculate log returns
        log_returns = np.log(current_prices / self.prev_prices)
        print(f"Log returns: {log_returns}")

        # Calculate portfolio return as weighted sum of log returns
        portfolio_return = np.sum(log_returns * self.portfolio)
        self.returns_log.append((portfolio_return, self.df.index[self.current_step]))
        print(f'Portfolio return: {portfolio_return}')

        if np.sum(action) == 0:
            print("Action is zero. Defaulting to the last portfolio composition.")
            action = self.portfolio.copy()

        # Rebalance portfolio if needed
        if self.steps_since_last_rebalance >= self.rebalance_frequency or self.current_step == 0:
            print("Rebalancing portfolio...")
            self.steps_since_last_rebalance = 0

            total_action = np.sum(action)
            action_percentages = action / total_action if total_action > 0 else np.zeros_like(action)
            print(f"Action percentages: {action_percentages}")

            # Update portfolio based on desired allocations
            self.portfolio = action_percentages
            print(f"Updated portfolio: {self.portfolio}")

            self.actions_log.append((action_percentages, self.df.index[self.current_step]))
        else: 
            new_portfolio_comp = self.portfolio * np.exp(log_returns)
            self.portfolio = normalize_portfolio(new_portfolio_comp)

        self.portfolio_composition_log.append((self.portfolio.copy(), self.df.index[self.current_step]))

        print(F'self.portfolio_composition_log: {self.portfolio_composition_log}')

        comp_history =  self.get_portfolio_composition_df()
        comp_history['Date'] = pd.to_datetime(comp_history['Date'])
        comp_history.set_index('Date',inplace=True)

        print(f'comp_history: {comp_history}')

        comp_history.columns = comp_history.columns.str.replace('_Price',' comp')

        print(f'comp_history: {comp_history}')

        # Calculate optimized weights and distance penalty using MVO
        print(f"data_at_mvo:{self.df.iloc[:self.current_step + 1], comp_history, self.target_return}")
        optimized_weights, returns, sortino_ratio = mvo(self.df.iloc[:self.current_step + 1][self.price_columns], comp_history, self.target_return)
        print(
            f'portfolio: {self.portfolio}, optimized_portfolio: {optimized_weights}, sortino_ratio: {sortino_ratio}'
        )

        if sortino_ratio is None:
            sortino_ratio = 0

        current_weights = self.portfolio if optimized_weights is not None else None
        max_distance = sum(abs(1 - value) for value in optimized_weights) if optimized_weights is not None else 0
        distance_penalty = sum(abs(current_weights[i] - optimized_weights[i]) for i in range(len(optimized_weights))) / max_distance if max_distance != 0 else 0

        # Reward calculation
        reward = portfolio_return * (sortino_ratio - distance_penalty)
        print(f"Reward: {reward}")

        # Update logs and state
        self.prev_prices = current_prices
        self.rewards_log.append((reward, self.df.index[self.current_step]))
        self.sortino_ratios_log.append((sortino_ratio, self.df.index[self.current_step]))
        self.current_step += 1

        done = self.current_step >= len(self.df) - 1
        state = self._get_observation()
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
        print(f'actions: {actions}, asset_names: {asset_names}, column_names: {column_names}, dates: {dates}')
        return pd.DataFrame(actions, columns=column_names).assign(Date=dates)

    def get_rewards_df(self):
        rewards, dates = zip(*self.rewards_log) if self.rewards_log else ([], [])
        return pd.DataFrame(rewards, columns=['Reward']).assign(Date=dates)

    def get_portfolio_composition_df(self):
        compositions, dates = zip(*self.portfolio_composition_log) if self.portfolio_composition_log else ([], [])
        return pd.DataFrame(compositions, columns=self.price_columns).assign(Date=dates)
    
    def get_returns_df(self):
        returns, dates = zip(*self.returns_log) if self.returns_log else ([], [])
        return pd.DataFrame(returns, columns=['Return']).assign(Date=dates)
    
    def get_sortino_ratios_df(self):
        returns, dates = zip(*self.sortino_ratios_log) if self.sortino_ratios_log else ([], [])
        return pd.DataFrame(returns, columns=['Portfolio Hourly Sortino Ratios']).assign(Date=dates)