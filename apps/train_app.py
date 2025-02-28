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

# %%
from diskcache import Cache

# %%

# %%

base_cache_dir = r'E:\Projects\portfolio_optimizers\classifier_optimizer'

cache = Cache(os.path.join(base_cache_dir, 'test_model_cache'))

# %%
from python_scripts.utils import flipside_api_results, set_random_seed, to_time, clean_prices, calculate_cumulative_return, calculate_cagr, calculate_beta, fetch_and_process_tbill_data,data_cleaning,prepare_data_for_simulation,normalize_asset_returns,calculate_sortino_ratio,pull_data
from sql_scripts.queries import lst_portfolio_prices,eth_btc_prices,dao_advisor_portfolio, yield_portfolio_prices, token_prices, all_yield_portfolio_prices,arb_stables_portfolio_prices, all_arb_stables_prices
from models.training import train_model
from python_scripts.apis import token_classifier_portfolio
import streamlit as st

# %%
files = os.listdir()
print(files)

# %%
from chart_builder.scripts.visualization_pipeline import visualization_pipeline
from chart_builder.scripts.utils import main as chartBuilder

# %%
load_dotenv()
flipside_api_key = os.getenv("FLIPSIDE_API_KEY")

# %% [markdown]
# # Network is the name any of the data tables available from Flipside which are generally EVM compatible
# 
# - ethereum
# - gnosis
# - base
# - avalanche
# - arbitrum
# - optimism

# %%
token_prices

# %%
@st.cache_data(ttl=timedelta(days=7))
def fetch_and_process_tbill_data(api_url, data_key, date_column, value_column, date_format='datetime'):
    api_key = os.getenv("FRED_API_KEY")
    api_url_with_key = f"{api_url}&api_key={api_key}"

    response = requests.get(api_url_with_key)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data[data_key])
        
        if date_format == 'datetime':
            df[date_column] = pd.to_datetime(df[date_column])
        
        df.set_index(date_column, inplace=True)
        df[value_column] = df[value_column].astype(float)
        return df
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure
    
three_month_tbill_historical_api = "https://api.stlouisfed.org/fred/series/observations?series_id=TB3MS&file_type=json"

try:
    three_month_tbill = fetch_and_process_tbill_data(three_month_tbill_historical_api, "observations", "date", "value")
    three_month_tbill['decimal'] = three_month_tbill['value'] / 100
    current_risk_free = three_month_tbill['decimal'].iloc[-1]
    print(f"3-month T-bill data fetched: {three_month_tbill.tail()}")
except Exception as e:
    print(f"Error in fetching tbill data: {e}")


# %%
#Global Parameters 
model_name = 'v0.1'#What the model files will be saved under; used programmatically in testing scripts
train_model_bool = True #This controls if you actually train a model under the model_name
function = None #Price feed function; leave None if want to use classifier

#Classifier Parameters
classifier_visualizations = True #If passing a function this must be False
network = 'arbitrum'
days = 7 # How many days back we calculate returns, averages, sums; if use_cached_data True, this doesn't make a difference
use_cached_data = False #True means we use saved data, false pulls fresh data.  If making new model, must be False
volume_threshold = 0.001 #Classifier filter; values less than one enable tokens w/ lower than average volume to stay in dataset.  Keep at 1 for "Default"
backtest_period = 4380 #How many days back we pull price feed data; set to Days in Hours; current default is 6 Months (for train/test)
start_date = None
# 4380
#Prices Visualization Parameters
window = 720 #Days in hours
normalized_value = 100

#Asset Filters (Pandas)
sortino_ratio_filter = True
sortino_threshold = 1 #x times more than average; 1 is default with higher number meaning higher ratio requirement
top = None

#Model Training Parameters
seed = 20 # Random number generator; each seed will have slightly result
rebalance_frequency = 24 # Hours between rebalance
training_percentage = 0.75 #% of Dataset it will train on
dropna = True
ffill = False
ent_coef = 0.01
clip_range = 0.3
risk_free_annual = current_risk_free if current_risk_free is not None else 0.0425

# %%
if function is not None: #Don't have classifier data to visualize if we are directly providing price feed function (Not using the classifier)
    classifier_visualizations = False

# %%
if start_date is None:
    start_date_raw = dt.datetime.now() - timedelta(hours=backtest_period)
    start_date = start_date_raw.strftime('%Y-%m-%d %H:00:00')
    

# %%
print(f'Start Date: {start_date}') #Start date used for prices function, if applicable

# %%
if model_name is None:
    model_name = f'{network}_classifier'
    print(f'model_name: {model_name}')

# %%
def prices_data_func(network,
                     api_key,use_cached_data,name,days=None,
                     function=None,start_date=None,
                     backtest_period=None,ffill=False,dropna=True,volume_threshold=1):
    
    if start_date is None and backtest_period is None:
        raise KeyError("Provide either a start date or backtest_period")
    
    if backtest_period is None:
        backtest_period = (pd.to_datetime(dt.datetime.now()) - pd.to_datetime(start_date)).days * 24

    if function is None:

        data = token_classifier_portfolio(
            network=network,
            days=days,
            name=name,
            api_key = flipside_api_key,
            use_cached_data=use_cached_data,
            volume_threshold=volume_threshold,
            backtest_period=backtest_period,
            prices_only=False
        )

        prices_df = data_cleaning(data['portfolio'],ffill=ffill,dropna=dropna)
        print(f'prices: {prices_df}') 
    else: 
        data = pull_data(function=function,start_date=start_date, path=f'data/{model_name}.csv', api=not use_cached_data,model_name=model_name)
        prices_df = data_cleaning(data['portfolio'])

    prices_df.columns = prices_df.columns.str.replace('_Price','')

    return data, prices_df

today_utc = dt.datetime.now(dt.timezone.utc) 
formatted_today_utc = today_utc.strftime('%Y-%m-%d %H:00:00')

# %%
data, prices_df = prices_data_func(
    network=network, 
    days=days,
    name=model_name,
    api_key=flipside_api_key,
    use_cached_data=use_cached_data,
    function=function,
    start_date=start_date,
    backtest_period=backtest_period,
    volume_threshold=volume_threshold,
    ffill=ffill,
    dropna=dropna)

# %%
prices_returns = prices_df.pct_change().dropna()
result = calculate_sortino_ratio(prices_returns,current_risk_free,window)
rolling_sortinos = result[0].dropna()
sortino_ratios_ranked = result[1].sort_values(ascending = False)

sortino_ratio_df = sortino_ratios_ranked.to_frame(
    'Sortino Ratio'
)
sortino_ratio_df.reset_index(inplace=True)

price_data = data['portfolio']

print(f'data: {data}')
print(f'data type: {type(data)}')


if 'classifier' in data:
    classifier_data = data['classifier']

# %%
sortino_ratio_df

# %%
if not sortino_ratio_filter and top is not None:
    sorted_by_return = classifier_data.sort_values(by='sixty_d_return',ascending=False).head(top)
    filtered_assets = sorted_by_return['symbol'].head(top).to_list()

    price_data = price_data[price_data['symbol'].isin(filtered_assets)]
    classifier_data = classifier_data[classifier_data['symbol'].isin(filtered_assets)]
    sortino_ratio_df = sortino_ratio_df[sortino_ratio_df['index'].isin(filtered_assets)]
    prices_df = prices_df[filtered_assets]
    rolling_sortinos = rolling_sortinos[filtered_assets]

# %%
usdc_data = {
    'ethereum': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
    'arbitrum': '0xaf88d065e77c8cC2239327C5EDb3A432268e5831',
    'optimism': '0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85',
    'gnosis': '0x2a22f9c3b484c3629090FeED35F17Ff8F88f76F0'
}

# %%
if sortino_ratio_filter:
    
    mean_ratio = sortino_ratio_df['Sortino Ratio'].mean() * sortino_threshold
    
    if top is not None:
        filtered_assets = sortino_ratio_df[sortino_ratio_df['Sortino Ratio']>=mean_ratio]['index'].sort_values(ascending=False).head(top).to_list()
    else:
        filtered_assets = sortino_ratio_df[sortino_ratio_df['Sortino Ratio']>=mean_ratio][['Sortino Ratio','index']].sort_values(by='Sortino Ratio', ascending=False)
        filtered_assets = filtered_assets['index'].head(top).to_list()

    price_data = price_data[price_data['symbol'].isin(filtered_assets)]

    if 'classifier' in data:
        classifier_data = classifier_data[classifier_data['symbol'].isin(filtered_assets)]
        
    sortino_ratio_df = sortino_ratio_df[sortino_ratio_df['index'].isin(filtered_assets)]
    prices_df = prices_df[filtered_assets]
    rolling_sortinos = rolling_sortinos[filtered_assets]

prices_df = prepare_data_for_simulation(prices_df, formatted_today_utc, formatted_today_utc)

# %%
price_data

# %%
if classifier_visualizations:
    classifier_fig1 = visualization_pipeline(
        df=classifier_data,
        groupby='symbol',
        num_col='sixty_d_return',
        chart_type='ranked bar',
        title=f'{days} Day Return by Symbol',
        ticksuffix=dict(y1='%',y2=None),
        to_percent=True,
        text=True

    )
    chartBuilder(
        classifier_fig1,
        title_xy=dict(x=0.5,y=0.8),
        title=f'{network.upper()} Tokens Classifier <br> {days}D Return',
        add_the_date=False,
        dt_index=False,
        save=False
    )

    # prices_df = prepare_data_for_simulation(prices_df, formatted_today_utc, formatted_today_utc)

    classifier_fig4 = visualization_pipeline(
        df=classifier_data,
        groupby='symbol',
        num_col='volume',
        chart_type='ranked bar',
        title=f'{network} Classifier Volume',
        tickprefix=dict(y1="$",y2=None),
        text=True,
        text_font_size=16

    )

    chartBuilder(
        classifier_fig4,
        title_xy=dict(x=0.5,y=0.82),
        title=f'{network.upper()} Tokens Classifier <br> {days} Volume by Token',
        dt_index=False,
        add_the_date=False,
        save=False
    )

    classifier_fig5 = visualization_pipeline(
        df=classifier_data,
        groupby='symbol',
        num_col='average_order',
        chart_type='ranked bar',
        title=f'{network} Classifier Avg Order',
        tickprefix=dict(y1="$",y2=None),
        text=True,
        descending=False,
        orientation='v',
        text_font_size=16,
        textposition='inside',
        margin=dict(t=75,r=0,l=0,b=0)

    )

    chartBuilder(
        classifier_fig5,
        title_xy=dict(x=0.2,y=0.95),
        title=f'{network.upper()} Tokens Classifier <br> {days} Avg Order Size by Token',
        dt_index=False,
        add_the_date=False,
        save=False
    )


# %%
normalized_prices = normalize_asset_returns(prices_df,prices_df.index.min(),prices_df.index.max(),normalized_value)

classifier_fig2 = visualization_pipeline(
    df=normalized_prices,
    cols_to_plot='All',
    chart_type='line',
    title=f'{model_name.upper()} Tokens Normalized',
    tickprefix=dict(y1="$",y2=None),
    show_legend=True

)
chartBuilder(
    classifier_fig2,
    title_xy=dict(x=0.3,y=0.82),
    title=f'{model_name.upper()} Tokens <br> Normalized Returns',
    dt_index=True,
    save=False
)

# %%
classifier_fig3 = visualization_pipeline(
    df=rolling_sortinos,
    cols_to_plot='All',
    chart_type='line',
    title=f'{model_name} Tokens {window * 24} Rolling D Sortino Ratios',
    tickprefix=dict(y1=None,y2=None),
    show_legend=True,
    buffer=0

)

chartBuilder(
    classifier_fig3,
    title_xy=dict(x=0.4,y=0.82),
    title=f'{model_name.upper()} Tokens {int(window / 24)} Rolling D <br> Sortino Ratios',
    dt_index=True,
    save=False
)

# %% [markdown]
# - 120 - 237.6

# %%
sortino_fig1 = visualization_pipeline(
        df=sortino_ratio_df,
        groupby='index',
        num_col='Sortino Ratio',
        chart_type='ranked bar',
        barmode='relative',
        title=f'{model_name} Tokens Sortino Ratios Over {days}D',
        ticksuffix=dict(y1=None,y2=None),
        to_percent=False,
        text=False

    )
chartBuilder(
    sortino_fig1,
    title_xy=dict(x=0.5,y=0.8),
    title=f'{model_name} Tokens Classifier <br> {days}D Sortino Ratios',
    dt_index=False,
    save=False,
    add_the_date=False
)

# %%
data

# %%
model_name

# %%
price_data

# %%
if train_model_bool:
    # Train the 
    #Try 6 Months worth of data for train/test before deploying
    test_data, train_end_date, test_start_date  = train_model(
        seed=seed, 
        prices=price_data,
        name=model_name, 
        api=False, 
        rebalance_frequency=rebalance_frequency, 
        training_percentage=training_percentage, 
        risk_free_annual=risk_free_annual,
        dropna=dropna,
        ffill=ffill,
        ent_coef=ent_coef,
        clip_range=clip_range
    )

    

# %%
if train_model_bool:
    print(f'train_end_date: {train_end_date} \n test_start_date: {test_start_date}')
    print(f'test_data: {test_data}')

# %%
f'{model_name} Test Start'

# %%
# classifier_data

# %%
train_dict = {
    "name":model_name,
    "rebalance_frequency":rebalance_frequency,
    "seed":seed,
    "function":function if function is not None else None,
    "network":network,
    'days':days,
    'backtest_period':backtest_period
}

# %%
prices_df.columns.to_list()

# %%
if function is None:
    cache.set(f'{model_name} Classifier',classifier_data)

# %%
if train_model_bool:
    cache.set(f'{model_name} Test Start', test_start_date)
    cache.set(f'{model_name} Params',train_dict)


# %%
if top is not None:
    cache.set(f'{model_name} Assets', filtered_assets)
else:
    cache.set(f'{model_name} Assets', prices_df.columns.to_list())

# %%
train_dict = cache.get(f'{model_name} Params')
train_dict

# %%
cache.get(f'{model_name} Assets')

# %%
cache.get(f'{model_name} Classifier')

# %%


