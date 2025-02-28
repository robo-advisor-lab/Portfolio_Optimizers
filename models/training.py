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

from python_scripts.utils import flipside_api_results, set_random_seed, to_time, clean_prices, calculate_cumulative_return, calculate_cagr, calculate_beta, fetch_and_process_tbill_data, pull_data, data_cleaning, set_global_seed
from sql_scripts.queries import lst_portfolio_prices,eth_btc_prices,yield_portfolio_prices
from models.rebalance import Portfolio

def train_model(function=None,risk_free_annual=None, prices=None, seed=20, name='interest_bearing_model',api=False, rebalance_frequency=2,
                training_percentage=0.75,path=None,start_date=None,dxy=None,
                    dropna=True,ffill=False,ent_coef=0.01,clip_range=0.3):

    load_dotenv()
    flipside_api_key = os.getenv("FLIPSIDE_API_KEY")
    dune_api_key = os.getenv('DUNE_API_KEY')
    dune = DuneClient(dune_api_key)

    print(f'rebalance_frequency: {rebalance_frequency}')

    if function is None and prices is None:
        raise KeyError('Need to pass either SQL function or prices df')

    elif prices is None and function is not None:
        if path is None:
            raise KeyError('Need to pass a save path for pull_data function')
        else:
            data_set = pull_data(function=function,start_date=start_date, path=path, api=api,model_name=name)
    else:

        data_set = prices.copy()

    prices_df = data_cleaning(data_set, dropna=dropna,ffill=ffill)
    prices_df.index = prices_df.index.tz_localize(None)

    print(f'prices_df.index: {prices_df.index}')
    # print(f'dxy.index: {dxy.index}')

    # import pdb; pdb.set_trace()

    if dxy is not None:
        prices_df = pd.merge(prices_df,dxy,left_index=True,right_index=True,how='left').ffill()

    print(f'prices_df: {prices_df}')
    print(f'prices_df.columns: {prices_df.columns}')
    # breakpoint()

    prices_df.dropna(inplace=True)
    
    max_date = prices_df.index.max()
    min_date = prices_df.index.min()

    # Set training percentage (0.75 for 75% training and 25% testing)

    # Calculate the total hours in the dataset
    total_hours = (max_date - min_date).total_seconds() / 3600

    # Calculate hours for training period and the training end date
    train_hours = total_hours * training_percentage
    train_end_date = min_date + timedelta(hours=train_hours)
    test_start_date = train_end_date + timedelta(hours=1)

    print(f"Training period end date: {train_end_date}")
    print(f"Testing period start date: {test_start_date}")

    train_data = prices_df[prices_df.index <= train_end_date]
    test_data = prices_df[prices_df.index > train_end_date]

    print(f'train_data index: {train_data}')

    # #Here we train

    env = Portfolio(train_data , seed=seed, rebalance_frequency=rebalance_frequency,risk_free_annual=risk_free_annual)
    set_global_seed(env, seed)
    # env.seed(seed)
    # env.action_space.seed(seed)
    # env = DummyVecEnv([lambda: Portfolio(filtered_combined_1, hold_cash=False, seed=seed)])

    # Create PPO agent
    model = PPO('MlpPolicy', env, seed=seed, verbose=1,ent_coef=ent_coef,clip_range=clip_range)

    # Train the model
    model.learn(total_timesteps=train_hours)

    # Save the model
    print(f'saving under AI_Models/{name}.zip')
    model.save(f'AI_Models/{name}')

    print(f'prices_df:{prices_df.columns}')

    print(f'returning from train_model')

    return test_data, train_end_date, test_start_date 

# if __name__ == "__main__":
#     train_model(
#         seed=20, 
#         name='interest_bearing_model', 
#         api=False, 
#         rebalance_frequency=2, 
#         training_percentage=0.75, 
#         path='data/interest_bearing_prices.csv', 
#         start_date='2024-03-19'
#     )