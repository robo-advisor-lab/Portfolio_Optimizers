
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
from plotly.utils import PlotlyJSONEncoder

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

import asyncio

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import matplotlib.pyplot as plt

from plotly.subplots import make_subplots

from diskcache import Cache

from flask import Flask, render_template, request, jsonify
import logging
import json

# %%
base_cache_dir = r'E:\Projects\portfolio_optimizers\classifier_optimizer'
cache = Cache(os.path.join(base_cache_dir, 'test_model_cache'))
global_classifier_cache = Cache(os.path.join(base_cache_dir, 'global_classifier_cache'))

# %%
from python_scripts.utils import flipside_api_results, set_random_seed, to_time, clean_prices, calculate_cumulative_return, calculate_cagr, calculate_beta, fetch_and_process_tbill_data,data_cleaning,prepare_data_for_simulation,normalize_asset_returns,calculate_sortino_ratio,pull_data,mvo,calculate_log_returns,normalize_log_returns
from sql_scripts.queries import lst_portfolio_prices,eth_btc_prices,dao_advisor_portfolio, yield_portfolio_prices, token_prices, arb_stables_portfolio_prices
from models.training import train_model
from python_scripts.apis import token_classifier_portfolio
import streamlit as st
from models.rebalance import Portfolio

from chart_builder.scripts.visualization_pipeline import visualization_pipeline
from chart_builder.scripts.utils import main as chartBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# %%
load_dotenv()
flipside_api_key = os.getenv("FLIPSIDE_API_KEY")
COINGEKCO_KEY = os.getenv("COINGEKCO_KEY")

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
def fetch_and_process_dxy_data(api_url, data_key, date_column, value_column, start_date=None, end_date=None, date_format='datetime'):
    api_key = os.getenv("FRED_API_KEY")
    
    # Add date range parameters to the API URL if specified
    if start_date and end_date:
        api_url += f"&observation_start={start_date}&observation_end={end_date}"
    
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

def calculate_expected_return():
    print(f'at calculate expected return')
    url = "https://api.coingecko.com/api/v3/coins/defipulse-index/market_chart?vs_currency=usd&days=365"

    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": COINGEKCO_KEY
    }

    response = requests.get(url, headers=headers)
    response_text = response.text
    data = json.loads(response_text)
    df_prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df_prices["timestamp"] = pd.to_datetime(df_prices["timestamp"], unit='ms')  # Convert to datetime
    df_prices.set_index('timestamp', inplace=True)
    model_name = global_classifier_cache.get('current_model_name')
    df = pd.read_csv(f'E:/Projects/portfolio_optimizers/classifier_optimizer/results/{model_name}/norm_returns.csv')
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
    df.set_index('Unnamed: 0', inplace=True)
    daily_df = df.resample('D').last()
    daily_df.index = pd.to_datetime(daily_df.index.strftime('%Y-%m-%d'))
    df_prices.rename(columns={'price': 'DPI Price'}, inplace=True)
    daily_df.rename(columns={"Return": "Portfolio Return"}, inplace=True)
    analysis_df = pd.merge(
        df_prices,
        daily_df,
        left_index=True,
        right_index=True,
        how='inner'
    )

    # ✅ Dynamically Calculate Test Period in Months
    start_date = analysis_df.index.min()
    end_date = analysis_df.index.max()
    test_period_days = (end_date - start_date).days
    test_period_months = test_period_days / 30  # Approximate as 30 days per month
    print(f"Test Period: {test_period_months:.2f} months")

    # Calculate CAGR
    dpi_cagr = calculate_cagr(analysis_df['DPI Price'])
    dpi_cumulative_risk_premium = dpi_cagr - current_risk_free
    portfolio_cagr = calculate_cagr(analysis_df['Portfolio Return'])
    portfolio_beta = calculate_beta(analysis_df, 'DPI Price', 'Portfolio Return')
    portfolio_expected_return = current_risk_free + (portfolio_beta * dpi_cumulative_risk_premium)
    
    # ✅ Annualize the Expected Return
    annualized_expected_return = (1 + portfolio_expected_return) ** (12 / test_period_months) - 1
    print(f'Annualized Portfolio Expected Return: {annualized_expected_return * 100:.2f}%')
    
    return annualized_expected_return

# %%
def prices_data_func(network,
                     api_key,use_cached_data,name,days=None,
                     function=None,start_date=None,
                     backtest_period=None,ffill=False,dropna=True,volume_threshold=1,prices_only=False):
    
    if start_date is None and backtest_period is None and function is None:
        raise KeyError("Provide either a start date or backtest_period")
    
    if backtest_period is None and start_date is not None:
        backtest_period = (pd.to_datetime(dt.datetime.now(dt.timezone.utc)) - pd.to_datetime(start_date)).days * 24

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

async def classifier_results(network, model_name, days=60, use_cached_data=True, volume_threshold=1, backtest_period=720,
                start_date=None, window=320, normalized_value=100, sortino_ratio_filter=False, sortino_threshold=1, top=None, seed=20, rebalance_frequency=24, training_percentage=0.75,
                dropna=True, ffill=False, ent_coef=0.01, clip_range=0.3, visualizations=False):
    global classifier_dict
    try:
        if start_date is None:
            start_date_raw = dt.datetime.now() - timedelta(hours=backtest_period)
            start_date = start_date_raw.strftime('%Y-%m-%d %H:00:00')

        print(f'Start Date: {start_date}')

        today_utc = dt.datetime.now(dt.timezone.utc) 
        formatted_today_utc = today_utc.strftime('%Y-%m-%d %H:00:00')

        # ✅ Fetch price data
        data, prices_df = prices_data_func(
            network=network, 
            days=days,
            name=model_name,
            api_key=flipside_api_key,
            use_cached_data=use_cached_data,
            function=None,
            start_date=start_date,
            backtest_period=backtest_period,
            volume_threshold=volume_threshold,
            ffill=ffill,
            dropna=dropna
        )

        prices_returns = prices_df.pct_change().dropna()
        result = calculate_sortino_ratio(prices_returns, current_risk_free, window)
        rolling_sortinos = result[0].dropna()
        sortino_ratios_ranked = result[1].sort_values(ascending=False)

        sortino_ratio_df = sortino_ratios_ranked.to_frame('Sortino Ratio')
        sortino_ratio_df.reset_index(inplace=True)

        price_data = data['portfolio']
        
        # ✅ Ensure classifier data exists
        if 'classifier' not in data:
            print(f"❌ No classifier data found for {model_name}")
            return False  # Exit early if no classifier data

        classifier_data = data['classifier']

        # ✅ Apply sorting/filtering logic
        if sortino_ratio_filter:
            if top is None:
                top=20
            mean_ratio = sortino_ratio_df['Sortino Ratio'].mean() * sortino_threshold
            filtered_assets = sortino_ratio_df[sortino_ratio_df['Sortino Ratio'] >= mean_ratio]['index'].head(top).tolist()
            price_data = price_data[price_data['symbol'].isin(filtered_assets)]
            prices_df = prices_df[filtered_assets]
            rolling_sortinos = rolling_sortinos[filtered_assets]
            sortino_ratio_df = sortino_ratio_df[sortino_ratio_df['index'].isin(filtered_assets)]
        
        if not sortino_ratio_filter and top is not None:
            sorted_by_return = classifier_data.sort_values(by='sixty_d_return',ascending=False).head(top)
            filtered_assets = sorted_by_return['symbol'].head(top).to_list()

            price_data = price_data[price_data['symbol'].isin(filtered_assets)]
            classifier_data = classifier_data[classifier_data['symbol'].isin(filtered_assets)]
            sortino_ratio_df = sortino_ratio_df[sortino_ratio_df['index'].isin(filtered_assets)]
            prices_df = prices_df[filtered_assets]
            rolling_sortinos = rolling_sortinos[filtered_assets]

        classifier_dict = {
            "data": data,
            "prices": prices_df,
            "price data for model": price_data,
            "assets": filtered_assets,
            "sortino ratio": sortino_ratio_filter,
            "top": top,
            "run date": formatted_today_utc,
            "model": model_name,
            "rolling sortinos": rolling_sortinos,
            "classifier data": classifier_data,
            "sortino_ratio_df": sortino_ratio_df,
            "network": network,
            "backtest_period": backtest_period,
            'days': days,
            'window': window,
            'normalized_value': normalized_value
        }

        print(f'params: {classifier_dict}')
        # breakpoint()

        global_classifier_cache.set('current_model_name', model_name)
        cache.set(f'{model_name} Params', classifier_dict)

        print(f'✅ Classification completed. Data saved to cache.')

        # ✅ Optional visualization
        vizualizations(model_name)

        print(f'vizualizations saved')

        # ✅ **Save results in cache**
        model_classifier_result = cache.get(f"{model_name} Params")
        print(f'params: {model_classifier_result}')
        # breakpoint()
        if not model_classifier_result:
            print(f"❌ No classifier results found for {model_name}")
            cache.set(f"{model_name}_status", "error")
            return
        
        print(f'at rebalance frequencies')

        # ✅ **Optimize rebalance frequency**
        rebalance_frequencies = [24, 48, 72, 96, 120, 360]  # days in hours
        results, best_result = await asyncio.to_thread(
            grid_search_rebalance_frequency,
            model_name=model_name,
            rebalance_frequencies=rebalance_frequencies,
            normalized_value=100,
            seed=20,
            risk_free_annual=current_risk_free  # Ensure this variable is defined
        )

        best_rebalance_frequency = best_result["rebalance_frequency"]
        train_dict = cache.get(f"{model_name} Params")

        if train_dict:
            train_dict["rebalance_frequency"] = best_rebalance_frequency
            cache.set(f"{model_name} Params", train_dict)

        print(f"✅ Classifier {model_name} completed. Best rebalance frequency: {best_rebalance_frequency} hours")
        cache.set(f"{model_name}_status", "done")

        # breakpoint()

        print('at save_results')

        save_results(model_name, classifier_dict, results, best_result, best_results_dict['Hourly Returns'],best_results_dict['Normalized Return'],best_results_dict['sortino_ratios'],best_results_dict['composition'],best_results_dict['actions_df'],best_results_dict['rewards_df'])

        print(f'at calculate expected return')

        portfolio_expected_return = calculate_expected_return()
        print(f'portfolio_expected_return: {portfolio_expected_return}')

        cache.set(f"{model_name} Expected Return",portfolio_expected_return)

        return True  # ✅ Return success

    except Exception as e:
        print(f"❌ Error in classifier_results: {e}")
        return False  # ✅ Return failure


def vizualizations(model_name,visualizations=False,classifier_visualizations=True):
    print(f'model_name {model_name}')
    classifier_dict = cache.get(f'{model_name} Params')
    days = classifier_dict['days']
    classifier_data = classifier_dict['classifier data']
    network = classifier_dict['network']
    prices_df = classifier_dict['prices']
    rolling_sortinos = classifier_dict['rolling sortinos']
    window = classifier_dict['window']
    sortino_ratio_df = classifier_dict['sortino_ratio_df']
    normalized_value = classifier_dict['normalized_value']

    print(f'classifier_dict: {classifier_dict}')
    # breakpoint()

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
            title=f'{model_name} Tokens Classifier <br> {days}D Return',
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
            title=f'{model_name} Volume',
            tickprefix=dict(y1="$",y2=None),
            text=True,
            text_font_size=16

        )

        chartBuilder(
            classifier_fig4,
            title_xy=dict(x=0.5,y=0.82),
            title=f'{model_name} Tokens <br> {days} Volume by Token',
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
            title=f'{model_name} Tokens Avg Order',
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
            title=f'{model_name} Tokens Classifier <br> {days} Avg Order Size by Token',
            dt_index=False,
            show=visualizations,
            add_the_date=False,
            save=False
        )


    # %%
    normalized_prices = normalize_asset_returns(prices_df,prices_df.index.min(),prices_df.index.max(),normalized_value)

    # breakpoint()

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
        # breakpoint()

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

    graph_json_1 = json.dumps(classifier_fig5.return_fig(), cls=PlotlyJSONEncoder)
    graph_json_2 = json.dumps(classifier_fig1.return_fig(), cls=PlotlyJSONEncoder)
    graph_json_3 = json.dumps(classifier_fig2.return_fig(), cls=PlotlyJSONEncoder)
    graph_json_4 = json.dumps(classifier_fig3.return_fig(), cls=PlotlyJSONEncoder)
    graph_json_5 = json.dumps(sortino_fig1.return_fig(), cls=PlotlyJSONEncoder)

    cached_data = {"graph_1": graph_json_1, "graph_2": graph_json_2,"graph_3":graph_json_3,"graph_4":graph_json_4,"graph_5":graph_json_5}

    cache.set(f'{model_name} classifier charts', cached_data)

    return jsonify(cached_data)

def save_results(model_name, classifier_dict, results, best_result, main_results, norm_returns,sortino_ratios,composition,actions_df,rewards_df):
    """Saves the results of the model training process to CSV files, ensuring serialization and error handling."""
    
    try:
        # Create directory for the model if it doesn't exist
        model_dir = f"results/{model_name}"
        os.makedirs(model_dir, exist_ok=True)

        def save_to_csv(data, filename):
            """Helper function to save data to CSV format."""
            filepath = f"{model_dir}/{filename}.csv"
            
            try:
                if isinstance(data, pd.DataFrame):
                    data.to_csv(filepath, index=True)
                elif isinstance(data, dict):
                    # Convert dict to DataFrame properly
                    df = pd.DataFrame(list(data.items()), columns=['Key', 'Value'])
                    df.to_csv(filepath, index=False)
                elif isinstance(data, list):
                    # Convert list to DataFrame properly
                    df = pd.DataFrame(data) if data and isinstance(data[0], (list, dict)) else pd.DataFrame({filename: data})
                    df.to_csv(filepath, index=False)
                elif isinstance(data, (int, float, str, bool)):
                    pd.DataFrame([data], columns=[filename]).to_csv(filepath, index=False)
                else:
                    print(f"⚠️ Unsupported data type for {filename}, skipping save.")
                    return

                print(f"✅ Saved {filename}.csv")

            except Exception as e:
                print(f"⚠️ Failed to save {filename}: {e}")

        # Save each data structure separately
        save_to_csv(classifier_dict, "classifier_dict")
        save_to_csv(results, "grid_search_results")
        save_to_csv(best_result, "best_result")
        save_to_csv(main_results, "main_results")
        save_to_csv(norm_returns, "norm_returns")
        save_to_csv(sortino_ratios, "sortino_ratios")
        save_to_csv(composition, "composition")
        save_to_csv(actions_df, "actions_df")
        save_to_csv(rewards_df, "rewards_df")

        print(f"✅ All results successfully saved in {model_dir}")

    except Exception as e:
        print(f"⚠️ Error saving results for model {model_name}: {e}")
    
def main(model,rebalance_frequency,prices_df,filtered_assets,normalized_value=100,seed=20,risk_free_annual=0.0425):

    print(f'at main')

    # import pdb; pdb.set_trace() 

    # prices_df = pd.merge(prices_df,hourly_dxy,left_index=True,right_index=True,how='left').ffill()

    model = PPO.load(f"AI_Models/{model}")
    print(f'filtered_assets: {filtered_assets}')
    print(f'prices assets: {prices_df.columns.unique()}')

    # breakpoint()

    env = Portfolio(prices_df, seed=seed, rebalance_frequency=rebalance_frequency,risk_free_annual=risk_free_annual)
   
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
    actions_df.to_csv('data/actions_df.csv')
    composition.to_csv('data/test_app_comp.csv')
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

    prices_df = prices_df.drop(columns='DXY', errors='ignore')

    dao_advisor_comp.set_index('Date',inplace=True)
    dao_advisor_comp.columns = prices_df.columns.str.replace('_Price', ' comp', regex=False)
    # dao_advisor_portfolio_values_df = dao_advisor_results["portfolio_values_df"]
    dao_advisor_results.set_index('Date', inplace=True)
    dao_advisor_portfolio_values_df = dao_advisor_results 

    optimized_weights, returns, sortino_ratio = mvo(prices_df,dao_advisor_comp,current_risk_free)

    price_returns = calculate_log_returns(prices_df)

    normalized_portfolio_value = normalize_log_returns(dao_advisor_results, dao_advisor_results.index.min(),dao_advisor_results.index.max(), 100)

    # normalized_portfolio_value = normalized_value * (1 + returns).cumprod()
    # normalized_prices = normalized_value * (1 + price_returns).cumprod()

    print(f'returns: {returns}')
    print(f'normalized_portfolio_value: {normalized_portfolio_value}')
    # print(f'normalized_prices: {normalized_prices}')
    # norm_df = normalized_portfolio_value.to_frame('Normalized Return')
    results['Normalized Return'] = normalized_portfolio_value

    return results

# %%
def grid_search_rebalance_frequency(model_name, rebalance_frequencies, normalized_value=100, 
                                    seed=20, training_percentage=0.66, dropna=True, ffill=False, ent_coef=0.01, clip_range=0.3,
                                    risk_free_annual=0.0425):
    global results, best_result, best_results_dict
    """
    Perform grid search over rebalance frequencies to find the best one based on normalized portfolio value.
    """
    results = []
    results_mapping = {}

    print(f'getting dxy')

    params = cache.get(f'{model_name} Params')

    prices_df = params['prices']

    # breakpoint()

    dxy_start_date = (prices_df.index.min() - timedelta(hours=360)).strftime('%Y-%m-%d %H:00:00') #pull dxy data starting 15 days from earliest price data dt

    dxy_historical = fetch_and_process_dxy_data(dxy_historical_api, "observations", "date", "value",start_date=dxy_start_date,end_date=None)

    dxy_historical['value'] = dxy_historical['value'].replace(".",np.nan).ffill().bfill()
    hourly_dxy = dxy_historical[['value']].resample('H').ffill()
    hourly_dxy.rename(columns={'value':'DXY'},inplace=True)
    hourly_dxy['DXY'] = hourly_dxy['DXY'].astype(float)
    print(hourly_dxy.memory_usage(deep=True), flush=True)
    print(hourly_dxy.info(), flush=True)

    # import pdb; pdb.set_trace()

    print(hourly_dxy)

    # print(f'test_data: {test_data}')
    # print(f'test_data: {hourly_dxy}')

    print(f'got dxy, merging test_data', flush=True)
    
    for freq in rebalance_frequencies:
        print(f"Testing rebalance frequency: {freq} hours")
        print(f'model name: {model_name}')

        params = cache.get(f'{model_name} Params')
        filtered_assets = cache.get(f'{model_name} Assets')
        data = params['data']
        prices_df = params['prices']
        price_data = params['price data for model']

        print(f'hourly_dxy: {hourly_dxy}')

        # hourly_dxy.index = pd.to_datetime(hourly_dxy)

        print(f'prices_df.index: {prices_df.index}')

        # import pdb; pdb.set_trace()

        hourly_dxy = prepare_data_for_simulation(hourly_dxy, prices_df.index.min().strftime('%Y-%m-%d %H:00:00'), prices_df.index.max().strftime('%Y-%m-%d %H:00:00'))

        print(f'price_data: {price_data}')
        print(f'hourly_dxy: {hourly_dxy}')

        # breakpoint()
        
        # Train the model with the current rebalance frequency
        test_data, train_end_date, test_start_date = train_model(
            seed=seed, 
            prices=price_data,
            name=model_name, 
            api=False, 
            dxy = hourly_dxy,
            rebalance_frequency=freq, 
            training_percentage=training_percentage, 
            risk_free_annual=risk_free_annual,
            dropna=dropna,
            ffill=ffill,
            ent_coef=ent_coef,
            clip_range=clip_range
        )

        cache.set(f'{model_name} Test Start', test_start_date)
        
        # Retrieve parameters and test data
        # prices_df = prices_df[prices_df.index >= test_start_date]
        # prices_df = prices_df[filtered_assets]
        # prices_df.columns = [f"{col}_Price" for col in prices_df.columns]
        print(f'test_data in gridsearch: {test_data}')
        # import pdb; pdb.set_trace()
        # Run the simulation and get the normalized portfolio value
        # prices_df = prices_df[prices_df.index>=test_start_date]
        # prices_df = prices_df[filtered_assets]
        # prices_df.columns = [f"{col}_Price" for col in prices_df.columns]

        print(f'calling main with rebalance_frequency: {freq}, dxy: {hourly_dxy}, test_data: {test_data}, model_name: {model_name}, filtered_assets: {filtered_assets}, normalized_value: {normalized_value}, risk_free_annual: {risk_free_annual}')

        # import pdb; pdb.set_trace() 
        
        results_dict = main(seed=seed, rebalance_frequency=freq,
                            prices_df=test_data, model=model_name, filtered_assets=filtered_assets, normalized_value=normalized_value,risk_free_annual=risk_free_annual)
        
        print(f'results_dict: {results_dict}')

        results_mapping[freq] = results_dict
        
        norm_value_df = results_dict['Normalized Return']
        print(f'norm_value_df: {norm_value_df}')
        final_norm_value = norm_value_df.iloc[-1]["Return"]
        print(f'final norm val: {final_norm_value}')
        results.append({"rebalance_frequency": freq, "final_norm_value": final_norm_value})
        print(f'results: {results}')
    
    # Find the best rebalance frequency
    best_result = max(results, key=lambda x: x["final_norm_value"])
    print(f'best_result: {best_result}')
    print(f"Best rebalance frequency: {best_result['rebalance_frequency']} hours with final normalized value: {best_result['final_norm_value']}")
    best_rebalance_frequency=best_result['rebalance_frequency']
    best_results_dict = results_mapping[best_rebalance_frequency]
    train_dict = cache.get(f'{model_name} Params')

    # Update the rebalance_frequency in the train_dict
    if train_dict:
        train_dict['rebalance_frequency'] = best_rebalance_frequency

        # Write the updated train_dict back to the cache
        cache.set(f'{model_name} Params', train_dict)
        print(f"Updated cache for {model_name} with best rebalance frequency: {best_rebalance_frequency} hours")
    else:
        print(f"No cache entry found for {model_name} Params.")
    
    print('at save_results')
    
    try:
        save_results(model_name, classifier_dict, results, best_result, best_results_dict['Hourly Returns'],best_results_dict['Normalized Return'],best_results_dict['sortino_ratios'],best_results_dict['composition'],best_results_dict['actions_df'],best_results_dict['rewards_df'])
    except Exception as e:
        print(f"⚠️ Failed to save results: {e}")

    portfolio_expected_return = calculate_expected_return()
    print(f'portfolio_expected_return: {portfolio_expected_return}')

    cache.set(f"{model_name} Expected Return",portfolio_expected_return)
        
    cache.set(f"{model_name}_status",'done')
    return results, best_result

async def run_classifier(model_name, network):
    """Runs the classifier asynchronously in a background task."""
    try:
        print(f"🚀 Running classifier for {model_name} on {network}...")

        # ✅ Run the async function directly (no `asyncio.to_thread()` needed)
        success = await classifier_results(
            network=network,
            model_name=model_name,
            days=60,
            use_cached_data=False,
            volume_threshold=0.005,
            backtest_period=8640, #6 Months for train/test
            start_date=None,
            window=720,
            normalized_value=100,
            sortino_ratio_filter=False,
            sortino_threshold=1,
            top=10,
            dropna=True,
            ffill=False
        )

        if not success:
            print(f"❌ Classifier failed for {model_name}")
            cache.set(f"{model_name}_status", "error")
            return
                
        print(f"✅ Classifier completed successfully for {model_name}")

    except Exception as e:
        print(f"❌ Error in run_classifier: {e}")
        cache.set(f"{model_name}_status", "error")

def create_app():
    app = Flask(__name__)

    @app.route('/')
    def home():
        return render_template('classifier_index.html')

    @app.route('/classifier-portfolio', methods=['POST'])
    async def get_classifier_portfolio():
        """Trigger the classifier but return a response immediately to prevent timeouts."""
        try:
            # ✅ Ensure JSON request
            data = request.get_json()
            model_name = data.get("model_name")
            network = data.get("network")

            if not model_name or not network:
                return jsonify({"error": "Both 'model_name' and 'network' are required"}), 400

            print(f"🔄 Classifier started for {model_name} on {network}")
            cache.set(f"{model_name}_status", "processing")

            # ✅ **Run classifier asynchronously in the background**
            asyncio.create_task(run_classifier(model_name, network))

            # ✅ **Return an immediate response to prevent HTTP timeout**
            return jsonify({"status": "processing", "message": f"Classifier {model_name} started"}), 200

        except Exception as e:
            print(f"❌ Error starting classifier: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/status', methods=['GET'])
    def status():
        """Check the status of the classifier process."""
        model_name = request.args.get('model_name')
        if not model_name:
            return jsonify({"error": "Model name is required"}), 400
        
        # breakpoint()
        
        print(f'model_name: {model_name}')

        # Check the explicit status flag
        status = cache.get(f'{model_name}_status', 'unknown')
        print(f'status: {status}')
        if status == 'done':
            return jsonify({"status": "done", "model_name": model_name}), 200
        elif status == 'processing':
            return jsonify({"status": "processing", "model_name": model_name}), 200
        elif status == 'error':
            return jsonify({"status": "error", "model_name": model_name}), 500
        else:
            return jsonify({"status": "unknown", "model_name": model_name}), 404
        
    @app.route('/cached-data')
    def get_cached_data():
        global model_name
        model_name = global_classifier_cache.get('current_model_name')
        cached_data = cache.get(f'{model_name} classifier charts')
        # print(f'cached_data: {cached_data}')
        if cached_data:
            return jsonify(cached_data)
        else:
            return jsonify({"error": "No cached data available"}), 404
        
    @app.route('/portfolio_expected_return')
    def expected_return_endpoint():
        model_name = global_classifier_cache.get('current_model_name')
        portfolio_expected_return = cache.get(f'{model_name} Expected Return')

        print(f'model_name: {model_name}')
        print(f'portfolio_expected_return: {portfolio_expected_return}')

        # Package them inside a dict and jsonify
        response = {
            "model_name": model_name,
            "portfolio_expected_return": portfolio_expected_return
        }

        return jsonify(response)
        
    @app.route('/clear-cache', methods=['POST'])
    def clear_cache():
        # Parse the JSON request
        global_classifier_cache.clear()
        cache.clear()
        return jsonify({"status": "Cache cleared successfully"})

    return app

# %%
if __name__ == "__main__":
    logger.info("Starting Flask app...")
    app = create_app()
    print('Starting Flask app...')
    app.run(debug=True, use_reloader=False, port=5025)
    # Define parameters
    # model_name = 'v0.2'
    # function = None
    # rebalance_frequencies = [24, 48, 72, 96, 120, 360]  # List of rebalance frequencies in hours
    # network = 'arbitrum'
    # days = 7
    # seed = 20
    # normalized_value = 100
    # volume_threshold = 0.01
    # backtest_period = 4380


