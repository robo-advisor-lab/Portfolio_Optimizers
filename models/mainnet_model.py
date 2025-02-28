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

class Portfolio(gym.Env):
    def __init__(self, historical_data, rebalance_frequency,start_date,end_date, seed, compositions):
        super(Portfolio, self).__init__()
        self.start_date = start_date
        self.end_date = end_date
        self.historical_data = historical_data[(historical_data.index >= self.start_date) & (historical_data.index <= self.end_date)]
        self.current_step = 0
        self.total_assets = len(self.historical_data.columns)  # Number of assets
        self.seed(seed)
        self.prev_prices = None
        # Define action and observation space
        self.action_space = spaces.Box(low=0, high=1, shape=(self.total_assets,), dtype=np.float32)
        self.target_return = 0  # Target return for Sortino calculation (0 for downside deviation)
        self.returns_log = []

        self.portfolio = compositions.values

        self.last_rebalance_time = self.start_date
        
        # Define observation space based on the number of features in the dataframe
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.historical_data.columns),), dtype=np.float32)
        
        # Initialize portfolio and logs
        self.compositions = compositions

        self.prev_portfolio_value = 0
        self.rebalance_frequency = rebalance_frequency
        self.steps_since_last_rebalance = 0  # Initialize the counter for rebalancing

        # Logs
        self.states_log = []
        self.rewards_log = []
        self.actions_log = []
        self.portfolio_values_log = []
        self.portfolio_composition_log = []
        self.sortino_ratios = []  

        # Get the columns related to asset prices
        self.price_columns = [col for col in self.historical_data.columns if col.endswith('_Price')]

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_portfolio_value = 0
        self.steps_since_last_rebalance = 0  # Reset rebalance counter
        self.states_log = []
        self.rewards_log = []
        self.actions_log = []
        self.portfolio_values_log = []
        self.portfolio_composition_log = []
        self.last_rebalance_time = self.start_date
        # self.portfolio = self.compositions.iloc[0].values
        
        obs = self._get_observation()
        print(f"Reset environment. Initial observation: {obs}")
        return obs.astype(np.float32), {}
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        seed = int(seed % (2**32 - 1))
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    #This is where we feed the current prices to the environment/model
    def _get_observation(self):
        # Ensure current_step does not go out-of-bounds
        if self.current_step >= len(self.historical_data):
            raise IndexError(f"Current step {self.current_step} is out of bounds for the dataframe.")
        
        obs = self.historical_data.iloc[self.current_step].values #Prices
        if np.isnan(obs).any():
            raise ValueError("Observation contains NaN values.")
        return obs.astype(np.float32)

    #Step function; where we increment through the dataframe
    def step(self, action): #Recieves the action made in model.predict(state) when we go live.  In training this is obscured with the built-in learn method
        print(f"\n--- Step {self.current_step} ---")

        if self.current_step >= len(self.historical_data):
            done = True
            print(f"Step {self.current_step}: Done (out of bounds)")
            return self._get_observation(), 0, done, False, {}
        
        # Increment the counter for rebalancing
        self.steps_since_last_rebalance += 1
        print(f"Steps since last rebalance: {self.steps_since_last_rebalance}")

        # Clip actions to ensure valid values
        action = np.clip(action, 0, 1)
        print(f"Action after clipping: {action}")

        if np.isnan(action).any():
            raise ValueError("Action contains NaN values.")
 
        # Get current asset prices
        current_prices = self.historical_data.iloc[self.current_step][self.price_columns].values
        print(f"Current prices: {current_prices}")

        if self.prev_prices is None:
            self.prev_prices = current_prices

        print(f'Portfolio: {self.portfolio, len(self.portfolio)}')

        print(f"Prev prices: {self.prev_prices, len(self.prev_prices)}")
        
        # Calculate total portfolio value
        portfolio_value_before = np.sum(self.portfolio * self.prev_prices) 
        print(f"Portfolio value before action: {portfolio_value_before}")

        current_date = self.historical_data.index[self.current_step]
        print(f'current_date: {current_date}')
        current_date = current_date.replace(tzinfo=None)

        if isinstance(self.last_rebalance_time, str):
            self.last_rebalance_time = dt.datetime.fromisoformat(self.last_rebalance_time).replace(tzinfo=None)

        print(f'self.last_rebalance_time: {self.last_rebalance_time}')

        print(f'self.last_rebalance_time: {self.last_rebalance_time}')

        self.last_rebalance_time = self.last_rebalance_time.replace(tzinfo=None)

        time_since_last_rebalance = (current_date - self.last_rebalance_time).total_seconds() / 3600
        print(f"Step {self.current_step}: Current date: {current_date}, Last rebalance time: {self.last_rebalance_time}, Time since last rebalance: {time_since_last_rebalance} hours")
        
        print(f'time_since_last_rebalance: {time_since_last_rebalance}')
        print(f'self.rebalance_frequency: {self.rebalance_frequency}')
        import pdb; pdb.set_trace()
        if time_since_last_rebalance >= self.rebalance_frequency or self.current_step == 0:
            if np.all(action == 0):
                print("Action is zero, maintaining current portfolio allocation.")
                portfolio_value_after = np.sum(self.portfolio * current_prices)
                current_value_per_asset = self.portfolio * self.prev_prices
                action_percentages = current_value_per_asset / portfolio_value_before
                print(f"Current portfolio allocation percentages: {action_percentages}")
                self.actions_log.append((action_percentages, self.historical_data.index[self.current_step]))
                print(f"Actions log updated")
            else:
                self.rebalance_frequency
                print("Rebalancing portfolio...")

                # Reset the counter after performing the rebalance
                self.last_rebalance_time = current_date
                print("Steps since last rebalance reset to 0")

                # Calculate action as percentage of portfolio
                total_value = portfolio_value_before
                print(f"Total portfolio value: {total_value}")

                total_action = np.sum(action)
                print(f"Total action: {total_action}")

                action_percentages = action / total_action if total_action > 0 else np.zeros_like(action)
                print(f"Action percentages: {action_percentages}")

                # Calculate desired allocations based on the current portfolio value
                desired_allocations = total_value * action_percentages
                print(f"Desired allocations (in value): {desired_allocations}")

                # Calculate updated portfolio in units
                self.portfolio = desired_allocations / current_prices
                print(f"Updated portfolio (in units): {self.portfolio}")

                # Ensure no negative values in portfolio
                self.portfolio = np.maximum(self.portfolio, 0)

                self.actions_log.append((action_percentages, self.historical_data.index[self.current_step]))
                print(f"Actions log updated")

            # Calculate portfolio value after rebalancing
            
        portfolio_value_after = np.sum(self.portfolio * current_prices) 
        print(f"Portfolio value after action: {portfolio_value_after}")

        portfolio_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before if portfolio_value_before else 0
        self.returns_log.append(portfolio_return)  # Log returns

        print(f'portfolio_return: {portfolio_return}')

        # Calculate Sortino Ratio
        negative_returns = [r for r in self.returns_log if r < self.target_return]
        downside_deviation = np.std(negative_returns) if negative_returns else 0

        if downside_deviation > 0:
            sortino_ratio = (np.mean(self.returns_log) - self.target_return) / downside_deviation
        else:
            sortino_ratio = 0  # Avoid division by zero

        print(f'sortino_ratio: {sortino_ratio}')

        reward = sortino_ratio  # Set Sortino ratio as the reward
        print(f"Reward: {reward}")

        # Move to the next step
        state = self._get_observation()
        print(f"State: {state}")
        self.states_log.append((state, self.historical_data.index[self.current_step]))
        print(f"State log updated")

        self.prev_prices = current_prices

        # Update logs
        self.prev_portfolio_value = portfolio_value_after
        print(f"Previous portfolio value updated: {self.prev_portfolio_value}")

        self.rewards_log.append((reward, self.historical_data.index[self.current_step]))
        print(f"Rewards log updated")

        self.portfolio_values_log.append((portfolio_value_after, self.historical_data.index[self.current_step]))
        print(f"Portfolio values log updated")

        self.portfolio_composition_log.append((self.portfolio.copy(), self.historical_data.index[self.current_step]))
        print(f"Portfolio composition log updated")

        self.current_step += 1
        done = self.current_step >= len(self.historical_data) 
        print(f"Done: {done}")

        return (state.astype(np.float32), reward, done, False, {}) if state is not None else (None, reward, done, False, {})

    #These functions are called to get current state, reward, actions, values, comp, from the environemtn, each step
    def get_states_df(self):
        states, dates = zip(*self.states_log) if self.states_log else ([], [])
        return pd.DataFrame(states, columns=self.historical_data.columns).assign(Date=dates)

    def get_rewards_df(self):
        rewards, dates = zip(*self.rewards_log) if self.rewards_log else ([], [])
        return pd.DataFrame(rewards, columns=['Reward']).assign(Date=dates)

    def get_actions_df(self):
        actions, dates = zip(*self.actions_log) if self.actions_log else ([], [])
        # Use asset names with "_weight" suffix for column names
        asset_names = [col.split('_Price')[0] for col in self.price_columns]
        column_names = [f'{asset}_weight' for asset in asset_names]
        return pd.DataFrame(actions, columns=column_names).assign(Date=dates)

    def get_portfolio_values_df(self):
        portfolio_values, dates = zip(*self.portfolio_values_log) if self.portfolio_values_log else ([], [])
        return pd.DataFrame(portfolio_values, columns=['Portfolio_Value']).assign(Date=dates)

    def get_portfolio_composition_df(self):
        # Extract portfolio compositions, cash, and dates
        compositions, dates = zip(*self.portfolio_composition_log) if self.portfolio_composition_log else ([], [])

        # Convert portfolio compositions to a DataFrame
        compositions_df = pd.DataFrame(compositions, columns=[f'Asset_{i}' for i in range(len(compositions[0]))])

        # Convert dates to a DataFrame
        dates_df = pd.DataFrame(dates, columns=['Date'])

        # Combine compositions_df and dates_df
        portfolio_composition_df = pd.concat([dates_df, compositions_df], axis=1)

        return portfolio_composition_df
    
    def get_portfolio_value(self):
        # Get the current prices for all assets
        current_prices = self.historical_data.iloc[self.current_step][self.price_columns].values
        
        # Calculate the total portfolio value
        portfolio_value = np.sum(self.portfolio * current_prices)
        
        return portfolio_value