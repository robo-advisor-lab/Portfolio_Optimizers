# %% [markdown]
# # Imports

# %%
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
# import cvxpy as cp
# import matplotlib.pyplot as plt
import datetime as dt
# from prophet import Prophet
from sklearn.metrics import r2_score, mean_absolute_error
# from stable_baselines3.common.vec_env import DummyVecEnv
# import torch
from flipside import Flipside

import os
from dotenv import load_dotenv

import datetime as dt
from datetime import timedelta
import pytz  # Import pytz if using timezones

from sklearn.linear_model import LinearRegression
import tensorflow as tf
import torch
import sys

# %%
from python_scripts.utils import flipside_api_results, set_random_seed, to_time, clean_prices, calculate_cumulative_return, calculate_cagr, calculate_beta, fetch_and_process_tbill_data,pull_data,data_cleaning
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../sql_scripts')))

print(os.getcwd())

from sql_scripts.queries import lst_portfolio_prices,eth_btc_prices,dao_advisor_portfolio, yield_portfolio_prices

load_dotenv()
flipside_api_key = os.getenv("FLIPSIDE_API_KEY")

def data_processing(api=False, training_percentage=0.75, prices=None, function=None, start_date=None,
                    name='interest_bearing_model',path=f'data/interest_bearing_prices.csv',data_start_date=None,
                    dropna=True,ffill=False):
    print(f'function:{function},start_date:{start_date},path:{path},api:{api},model_name: {name},training_percentage:{training_percentage}')
    if function is None and prices is None:
        raise KeyError('Need to pass either SQL function or prices df')

    elif prices is None and function is not None:

        data_set = pull_data(function=function,start_date=start_date, path=path, api=api,model_name=name)
    else:

        data_set = prices.copy()
    
    print(f'data_set: {data_set}')

    prices_df = data_cleaning(data_set[f'portfolio'], dropna=dropna,ffill=ffill)

    if data_start_date is not None:
        test_data = prices_df[prices_df.index > data_start_date]
        train_data = None
    else:
    
        max_date = prices_df.index.max()
        min_date = prices_df.index.min()

        # Set training percentage (0.75 for 75% training and 25% testing)

        # Calculate the total hours in the dataset
        total_hours = (max_date - min_date).total_seconds() / 3600

        # Calculate hours for training period and the training end date
        train_hours = total_hours * training_percentage
        train_end_date = min_date + timedelta(hours=train_hours)

        print(f"Training period end date: {train_end_date}")
        print(f"Testing period start date: {train_end_date + timedelta(hours=1)}")

        train_data = prices_df[prices_df.index <= train_end_date]
        test_data = prices_df[prices_df.index > train_end_date]

    return test_data, train_data, prices_df

    