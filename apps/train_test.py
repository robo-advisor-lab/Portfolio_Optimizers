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
import sys

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(project_root)

# os.chdir('..')
# files = os.listdir()
# print(files)

# %%
base_cache_dir = r'E:\Projects\portfolio_optimizers\classifier_optimizer'

cache = Cache(os.path.join(base_cache_dir, 'test_model_cache'))

# %%
from python_scripts.utils import flipside_api_results, set_random_seed, to_time, clean_prices, calculate_cumulative_return, calculate_cagr, calculate_beta, fetch_and_process_tbill_data,data_cleaning,prepare_data_for_simulation,normalize_asset_returns,calculate_sortino_ratio,pull_data,data_cleaning,calculate_log_returns, to_percent, mvo
from sql_scripts.queries import lst_portfolio_prices,eth_btc_prices,dao_advisor_portfolio, yield_portfolio_prices, token_prices, all_yield_portfolio_prices,arb_stables_portfolio_prices, all_arb_stables_prices
from models.training import train_model
from python_scripts.apis import token_classifier_portfolio
from models.rebalance import Portfolio
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

font_family = "Cardo"

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
    
@st.cache_data(ttl=timedelta(days=1))
def fetch_and_process_dxy_data(api_url, data_key, date_column, value_column, date_format='datetime'):
    api_key = os.getenv("FRED_API_KEY")
    api_url_with_key = f"{api_url}&api_key={api_key}"

    response = requests.get(api_url_with_key)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data[data_key])
        
        if date_format == 'datetime':
            df[date_column] = pd.to_datetime(df[date_column])
        
        df.set_index(date_column, inplace=True)
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

dxy_historical_api = "https://api.stlouisfed.org/fred/series/observations?series_id=DTWEXBGS&file_type=json"

try:
    dxy_historical = fetch_and_process_dxy_data(dxy_historical_api, "observations", "date", "value")

    dxy_historical['value'] = dxy_historical['value'].replace(".",np.nan).ffill().bfill()
    hourly_dxy = dxy_historical[['value']].resample('H').ffill()
    hourly_dxy.rename(columns={'value':'DXY'},inplace=True)
    hourly_dxy['DXY'] = hourly_dxy['DXY'].astype(float)
except Exception as e:
    print(f"Error in fetching DXY data: {e}")

# hourly_dxy = None

def prices_data_func(network,
                     api_key,use_cached_data,name,days=None,
                     function=None,start_date=None,
                     backtest_period=None,ffill=False,dropna=True,volume_threshold=1,prices_only=False):
    
    if start_date is None and backtest_period is None and function is None:
        raise KeyError("Provide either a start date or backtest_period")
    
    if backtest_period is None and start_date is not None:
        backtest_period = (pd.to_datetime(dt.datetime.now(dt.timezone.utc)) - pd.to_datetime(start_date).tz_localize('UTC')).days * 24

    if function is None:

        data = token_classifier_portfolio(
            network=network,
            days=days,
            name=name,
            api_key = flipside_api_key,
            use_cached_data=use_cached_data,
            volume_threshold=volume_threshold,
            backtest_period=backtest_period,
            prices_only=prices_only
        )

        prices_df = data_cleaning(data['portfolio'],ffill=ffill,dropna=dropna)
        print(f'prices: {prices_df}') 
    else: 
        data = pull_data(function=function,start_date=start_date, path=f'data/{name}.csv', api=not use_cached_data,model_name=name)
        prices_df = data_cleaning(data['portfolio'])

    prices_df.columns = prices_df.columns.str.replace('_Price','')

    return data, prices_df


# %%

def train_script(network,model_name=None,train_model_bool=False,function=None,classifier_visualizations=True,days=60,use_cached_data=True,volume_threshold=1,backtest_period=4380,
                start_date = None,window=720,normalized_value=100, sortino_ratio_filter=False,sortino_threshold=1,top=None,seed=20,rebalance_frequency=24,training_percentage=0.75,
                dropna=True,ffill=False,ent_coef=0.01,clip_range=0.3,visualizations=False):
    
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
            show=visualizations,
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
            show=visualizations,
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
            show=visualizations,
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
        show=visualizations,
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

    # import pdb; pdb.set_trace()

    if not rolling_sortinos.empty:

        chartBuilder(
            classifier_fig3,
            title_xy=dict(x=0.4,y=0.82),
            title=f'{model_name.upper()} Tokens {int(window / 24)} Rolling D <br> Sortino Ratios',
            dt_index=True,
            show=visualizations,
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
        show=visualizations,
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
            dxy = hourly_dxy,
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


def main(model,rebalance_frequency,data,prices_df,normalized_value=100,seed=20,visualizations=False):

    model = PPO.load(f"AI_Models/{model}")

    env = Portfolio(prices_df, seed=seed, rebalance_frequency=rebalance_frequency,risk_free_annual=current_risk_free)
   
    # Function to run simulation for each environment and model
    def run_simulation():
        

        states = []
        rewards = []
        actions = []
        portfolio_values = []

        # Reset the environment to get the initial state
        state, _ = env.reset(seed=seed)
        done = False

        while not done:
            action, _states = model.predict(state)
            action = action / np.sum(action) if np.sum(action) > 0 else action  # Prevent division by zero

            next_state, reward, done, truncated, info = env.step(action)

            # Store the results
            states.append(next_state.flatten())
            rewards.append(reward)
            actions.append(action.flatten())

            # Update the state
            state = next_state

        # Access the logged data as DataFrames
        states_df = env.get_states_df()
        rewards_df = env.get_rewards_df()
        actions_df = env.get_actions_df()
        portfolio_values_df = env.get_returns_df()
        composition = env.get_portfolio_composition_df()
        sortino_ratios = env.get_sortino_ratios_df()

        return states_df, rewards_df, actions_df, portfolio_values_df,composition,sortino_ratios
    
    states_df, rewards_df, actions_df, portfolio_values_df,composition,sortino_ratios = run_simulation()
    actions_df.to_csv(r'E:\Projects\portfolio_optimizers\classifier_optimizer\data\actions_df.csv')
    composition.to_csv(r'E:\Projects\portfolio_optimizers\classifier_optimizer\data\test_app_comp.csv')
    results = {
        "states_df": states_df,
        "rewards_df": rewards_df,
        "actions_df": actions_df,
        "Hourly Returns": portfolio_values_df,
        "composition":composition,
        'sortino_ratios':sortino_ratios
    }

    # Optionally, you can plot the rewards and portfolio values here for each model
    plt.plot(rewards_df['Date'], rewards_df['Reward'])
    plt.xlabel('Date')
    plt.ylabel('Reward')
    plt.title(f'Rewards over Time for {model}')
    # plt.savefig(f"plots/rewards_plot_{rebalance_frequency}.png")
    # plt.close() 

    # Access the stored DataFrames for each model
    dao_advisor_results = results["Hourly Returns"]
    dao_advisor_comp = results["composition"]

    print(f'dao_advisor_comp: {dao_advisor_comp}')
    print(f'dao_advisor_results: {dao_advisor_results}')
    print(f'prices_df: {prices_df}')

    # import pdb; pdb.set_trace()

    prices_df = prices_df.drop(columns='DXY', errors='ignore')

    dao_advisor_comp.set_index('Date',inplace=True)
    dao_advisor_comp.columns = prices_df.columns.str.replace('_Price', ' comp', regex=False)
    # dao_advisor_portfolio_values_df = dao_advisor_results["portfolio_values_df"]
    dao_advisor_results.set_index('Date', inplace=True)
    dao_advisor_portfolio_values_df = dao_advisor_results 

    optimized_weights, returns, sortino_ratio = mvo(prices_df,dao_advisor_comp,current_risk_free)

    price_returns = calculate_log_returns(prices_df)

    normalized_portfolio_value = normalized_value * (1 + returns).cumprod()
    normalized_prices = normalized_value * (1 + price_returns).cumprod()

    print(f'returns: {returns}')
    print(f'normalized_portfolio_value: {normalized_portfolio_value}')
    print(f'normalized_prices: {normalized_prices}')

    # dao_advisor_portfolio_return = calculate_cumulative_return(dao_advisor_portfolio_values_df)
    # dao_advisor_normalized = normalize_asset_returns(dao_advisor_portfolio_values_df, dao_advisor_portfolio_values_df.index.min(),dao_advisor_portfolio_values_df.index.max(), normalized_value)

    #Prices performance for comparison
    # dao_advisor_prices_normalized = normalize_asset_returns(prices_df, dao_advisor_portfolio_values_df.index.min(),dao_advisor_portfolio_values_df.index.max(), normalized_value)
    # dao_advisor_prices_return = calculate_cumulative_return(dao_advisor_prices_normalized)
    # dao_advisor_normalized.rename(columns={'Portfolio_Value':f'{model} Portfolio Value'},inplace=True)

    norm_df = normalized_portfolio_value.to_frame('Normalized Return')
    print(f'norm_df: {norm_df}')

    model_fig1 = visualization_pipeline(
        df=norm_df,
        title='model_normalized',
        chart_type = 'line',
        cols_to_plot='All',
        tickprefix=dict(y1='$',y2=None),
        show_legend=True,
        decimals=True,
        tickformat=dict(x='%b %d <br> %y',y1=None,y2=None),
        legend_placement=dict(x=0.05,y=0.8),
        font_family=font_family
    )

    chartBuilder(
        fig = model_fig1,
        show=visualizations,
        save=False,
        title='Normalized Model Performance',
        subtitle=f'{rebalance_frequency} Portfolio'
    )

    combined_comparison = pd.merge(
        normalized_portfolio_value,
        normalized_prices,
        left_index=True,
        right_index=True,
        how='inner'

    )

    model_fig2 = visualization_pipeline(
        df=combined_comparison,
        title='normalized comparison',
        chart_type = 'line',
        cols_to_plot='All',
        tickprefix=dict(y1='$',y2=None),
        show_legend=True,
        decimals=True,
        tickformat=dict(x='%b %d <br> %y',y1=None,y2=None),
        legend_placement=dict(x=0.05,y=0.8),
        font_family=font_family
    )

    chartBuilder(
        fig = model_fig2,
        show=visualizations,
        save=False,
        title=f'Normalized Performance Comparison {rebalance_frequency}',
        subtitle=None
    )

    # dao_advisor_returns_df = calculate_excess_return(dao_advisor_portfolio_return, dao_advisor_prices_return)
    # for col in dao_advisor_returns_df:
    #     print(f'{model} excess return over {col}: {dao_advisor_returns_df[col].values[0]}')

    print(f'seed: {seed}')
    print(f'rebalance_frequency:{rebalance_frequency}')
    # print(f'{dao_advisor_normalized.index.min().strftime("%d/%m/%Y")} through {dao_advisor_normalized.index.max().strftime("%d/%m/%Y")}')
    print(f'normalized value:{normalized_value}')
    # print(f'{model} Cumulative Return: {dao_advisor_portfolio_return.values[0][0]*100:.2f}%')
    # print(f'average excess return: {dao_advisor_returns_df.mean(axis=1).values[0]}')
    # dao_advisor_comp.columns = [f"{col} Comp" for col in dao_advisor_comp.columns if col]
    prices_df.columns = prices_df.columns.str.replace(' Comp','_Price')

    dao_advisor_combined_analysis = pd.merge(
        dao_advisor_comp,
        prices_df,
        left_index=True,
        right_index=True,
        how='inner'
    )

    print(f'dao_advisor_combined_analysis: {dao_advisor_combined_analysis}')

    dao_advisor_combined_analysis_comp = to_percent(dao_advisor_combined_analysis)

    print(f'dao_advisor_combined_analysis_comp: {dao_advisor_combined_analysis_comp}')

    model_fig3 = visualization_pipeline(
        df=dao_advisor_comp,
        title='dao_advisor_combined_analysis_comp',
        chart_type='line',
        area=True,
        show_legend=True,
        to_percent=True,
        legend_placement=dict(x=0.1,y=1.3),
        cols_to_plot='All',
        ticksuffix=dict(y1='%',y2=None),
        margin=dict(t=150,b=0,l=0,r=0),
        font_family=font_family
    )

    chartBuilder(
        fig=model_fig3,
        title='Portfolio Composition Over Time',
        date_xy=dict(x=0.1,y=1.4),
        show=visualizations,
        save=False
    )

    print(f'sortino_ratio: {sortino_ratio}')

    results['Normalized Return'] = norm_df

    return results

if __name__ == "__main__":
    
    #Global Parameters 
    model_name = 'test1'#What the model files will be saved under; used programmatically in testing scripts
    train_model_bool = True #This controls if you actually train a model under the model_name
    function = None #Price feed function; leave None if want to use classifier

    #Classifier Parameters
    classifier_visualizations = True #If passing a function this must be False
    network = 'ethereum'
    days = 90 # How many days back we calculate returns, averages, sums; if use_cached_data True, this doesn't make a difference
    use_cached_data = True #True means we use saved data, false pulls fresh data.  If making new model, must be False
    volume_threshold = 0.1 #Classifier filter; values less than one enable tokens w/ lower than average volume to stay in dataset.  Keep at 1 for "Default"
    backtest_period = 2150 #How many days back we pull price feed data; set to Days in Hours; current default is 6 Months (for train/test)
    start_date = None
    # 4380
    #Prices Visualization Parameters
    window = 72 #Days in hours
    normalized_value = 100

    #Asset Filters (Pandas)
    sortino_ratio_filter = True
    sortino_threshold = 1 #x times more than average; 1 is default with higher number meaning higher ratio requirement
    top = 5

    #Model Training Parameters
    seed = 20 # Random number generator; each seed will have slightly result
    rebalance_frequency = 24 # Hours between rebalance
    training_percentage = 0.75 #% of Dataset it will train on
    dropna = True
    ffill = False
    ent_coef = 0.01
    clip_range = 0.3

    train_script(network=network,model_name=model_name,train_model_bool=train_model_bool,function=function,classifier_visualizations=classifier_visualizations,days=days,use_cached_data=use_cached_data,volume_threshold=volume_threshold,backtest_period=backtest_period,
                start_date = start_date,window=window,normalized_value=normalized_value, sortino_ratio_filter=sortino_ratio_filter,sortino_threshold=sortino_threshold,top=top,seed=seed,rebalance_frequency=rebalance_frequency,training_percentage=training_percentage,
                dropna=dropna,ffill=ffill,ent_coef=ent_coef,clip_range=clip_range,visualizations=True)

    params = cache.get(f'{model_name} Params')
    network = params['network']
    
    print(f'model_name: {model_name} \nnetwork: {network}')

    test_start_date = cache.get(f'{model_name} Test Start') # First hour after end of training dataset; ensures no overlap
    filtered_assets = cache.get(f'{model_name} Assets')

    print(f'assets: {filtered_assets}')
    
    function = params['function']
    seed = params['seed']
    
    rebalance_frequency = params['rebalance_frequency']

    print(f'rebalance_frequency: {rebalance_frequency}')

    test_start_date = str(test_start_date)
    print(f'test_start_date: {test_start_date}')

    data, prices_df = prices_data_func(
                    network=network, 
                    name=model_name,
                    api_key=flipside_api_key,
                    use_cached_data=True,
                    function=function,
                    start_date=test_start_date,
                    prices_only=True
                    )
    
    print(f'data: {data}')
    print(f'prices_df: {prices_df}')
    prices_df.index = prices_df.index.tz_localize(None)
    prices_df = prices_df[prices_df.index>=test_start_date]
    prices_df = prices_df[filtered_assets]
    prices_df.columns = [f"{col}_Price" for col in prices_df.columns]
    if hourly_dxy is not None:
        prices_df = pd.merge(prices_df,hourly_dxy,left_index=True,right_index=True,how='left').ffill()
    # data['portfolio'] = data['portfolio'][data['portfolio'].index>=test_start_date]

    main(seed=seed,
         rebalance_frequency=rebalance_frequency,
         data=data['portfolio'],
         prices_df=prices_df,
         model=model_name,
         normalized_value=normalized_value,
         visualizations=True
         )
