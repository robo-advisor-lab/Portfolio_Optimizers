
import pandas as pd
import requests
import numpy as np
import yfinance as yf
from collections import OrderedDict

import random
import plotly.io as pio
from flask import Flask, render_template, request, jsonify
from web3 import Web3, EthereumTesterProvider
from web3.exceptions import TransactionNotFound,TimeExhausted
import httpx

import asyncio
import datetime as dt
import pickle

import os
from dotenv import load_dotenv

import datetime as dt
from datetime import timedelta

from uniswap import Uniswap
import math

from flask_cors import CORS

from stable_baselines3 import PPO

import streamlit as st

from plotly.utils import PlotlyJSONEncoder

pio.templates["custom"] = pio.templates["plotly"]
pio.templates["custom"].layout.font.family = "Cardo"

font_family = "Cardo"

# Set the default template
pio.templates.default = "custom"

# %%
from eth_account import Account
from eth_account.signers.local import LocalAccount
from web3 import Web3, EthereumTesterProvider
from web3.exceptions import TransactionNotFound,TimeExhausted

from python_scripts.utils import (flipside_api_results, set_random_seed, to_time, clean_prices, 
                                  calculate_cumulative_return, calculate_cagr, calculate_beta, 
                                  data_cleaning,prepare_data_for_simulation,
                                  pull_data,mvo,composition_difference_exceeds_threshold,calculate_compositions,
                                  set_global_seed,calculate_log_returns,normalize_log_returns,normalize_asset_returns,
                                  calculate_portfolio_returns)
from python_scripts.plots import model_visualizations
from sql_scripts.queries import lst_portfolio_prices,eth_btc_prices,dao_advisor_portfolio, yield_portfolio_prices,token_prices,model_flows
from models.testnet_model import Portfolio
from python_scripts.apis import token_classifier_portfolio,fetch_and_process_tbill_data
from python_scripts.web3_utils import get_token_decimals,get_balance,convert_to_usd, network_func, rebalance_portfolio
from python_scripts.plots import create_interactive_sml

# %%
from diskcache import Cache

# %%
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import time

import logging
import json

from pyngrok import ngrok, conf, installer
import ssl
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

# # Set the path to the ngrok executable installed by Chocolatey
# ngrok_path = "C:\\ProgramData\\chocolatey\\bin\\ngrok.exe"

# # Update the pyngrok configuration with the ngrok path
# pyngrok_config = conf.PyngrokConfig(ngrok_path=ngrok_path)

# # Check if ngrok is installed at the specified path, if not, install it using the custom SSL context
# if not os.path.exists(pyngrok_config.ngrok_path):
#     installer.install_ngrok(pyngrok_config.ngrok_path, context=context)

# # Configure ngrok with custom SSL context
# conf.set_default(pyngrok_config)
# conf.get_default().ssl_context = context

# ngrok_token = os.getenv('ngrok_token')

# # Set your ngrok auth token
# ngrok.set_auth_token(ngrok_token)

# Start ngrok
# public_url = ngrok.connect(5000, pyngrok_config=pyngrok_config, hostname="www.optimizerfinance.com").public_url
# print("ngrok public URL:", public_url)

# Price Data API
flipside_api_key = os.getenv("FLIPSIDE_API_KEY")
COINGECKO_API_KEY = os.getenv('COINGEKCO_KEY')

#Blockchain RPCs
# GNOSIS_GATEWAY = os.getenv('GNOSIS_GATEWAY')
# ARBITRUM_GATEWAY = os.getenv('ARBITRUM_GATEWAY')
# OPTIMISM_GATEWAY = os.getenv('OPTIMISM_GATEWAY')
# ETHEREUM_GATEWAY = os.getenv('ETHEREUM_GATEWAY')

#Address Credentials
ACCOUNT_ADDRESS = os.getenv('MODEL_ADDRESS')
PRIVATE_KEY = os.getenv('MODEL_KEY')
WETH_ADDRESS = os.getenv('WETH_ADDRESS')

# os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f'Current working directory: {os.getcwd()}')
os.chdir('classifier_optimizer')

global global_cache

live_results_path = r'E:\Projects\portfolio_optimizers\classifier_optimizer\live_results'

base_cache_dir = r'E:\Projects\portfolio_optimizers\classifier_optimizer'
global_cache = Cache(os.path.join(base_cache_dir, 'global_cache')) 

classifier_cache = Cache(os.path.join(base_cache_dir, 'test_model_cache'))
global_classifier_cache = Cache(os.path.join(base_cache_dir, 'global_classifier_cache'))

# original_directory = os.getcwd()

# # Change to 'classifier_optimizer'
# new_directory = os.path.abspath(os.path.join(original_directory, 'classifier_optimizer'))
# os.chdir(new_directory)
# print(f'Directory: {os.getcwd()}')

# Change back to the original directory
# os.chdir(original_directory)

def prices_data_func(network,
                     api_key,use_cached_data,name,days=None,
                     function=None,start_date=None,
                     backtest_period=None,filtered_assets=None):
    
    if start_date is None and backtest_period is None:
        raise KeyError("Provide either a start date or backtest_period")
    
    print(f"backtest days: {(pd.to_datetime(dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:00:00')) - pd.to_datetime(start_date)).days}")
    
    if backtest_period is None:
        backtest_period = (pd.to_datetime(dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:00:00')) - pd.to_datetime(start_date)).days * 24
        if backtest_period < 1:
            backtest_period = 1

    if function is None:
        print(f'name: {name}')

        data = token_classifier_portfolio(
            network=network,
            days=days,
            name=name,
            api_key = api_key,
            use_cached_data=use_cached_data,
            start_date = start_date,
            prices_only=True
        )

        prices_df = data_cleaning(data['portfolio'])
        prices_df
    else: 
        data = pull_data(function=function,start_date=start_date, path=f'data/{name}.csv', api=not use_cached_data,model_name=name)
        prices_df = data_cleaning(data['portfolio'])
        prices_df = prices_df[prices_df.index >= start_date].dropna()
        prices_df

    # prices_df.columns = prices_df.columns.str.replace('_Price','')
    # filtered_assets_with_price = [f"{asset}_Price" for asset in filtered_assets]


    return data, prices_df

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
    
dxy_historical_api = "https://api.stlouisfed.org/fred/series/observations?series_id=DTWEXBGS&file_type=json"

async def initialize_model_cache():
    global cache, params, model_name, chain, rebalancing_frequency, w3, gateway, factory, router, version, params_cache, last_rebalance_time, first_run_bool, uniswap, global_contracts, global_decimals

    base_cache_dir = r'E:\Projects\portfolio_optimizers\classifier_optimizer'

    # Ensure global cache is initialized
    if 'cache' not in globals() or cache is None:
        cache = Cache(os.path.join(base_cache_dir, 'global_cache'))  # Global fallback cache
        print(f"Initialized global cache at {cache.directory}")

    # Check if a new version is cached, otherwise use a default
    new_version = cache.get('new_version', None)
    # breakpoint()
    model_name = new_version if new_version else 'v01'

    # Create a separate cache for the selected model version
    cache = Cache(os.path.join(base_cache_dir, f'live_{model_name}_cache'))
    print(f"Initialized cache for model: {model_name} at {cache.directory}")

    # Load parameters for the selected model version
    params_cache = Cache(os.path.join(base_cache_dir, 'test_model_cache'))  # Parameters cache
    params = params_cache.get(f'{model_name} Params', {})

    print(f'Params Cache Directory: {params_cache.directory}')
    print(f'Live Cache Directory: {cache.directory}')

    # if params['backtest_period'] is None:
    #     backtest_period = 0

    w3, gateway, factory, router, version = network_func('arbitrum')

    account = Account.from_key(PRIVATE_KEY)
    w3.eth.default_account = account.address

    uniswap = Uniswap(address=ACCOUNT_ADDRESS, private_key=PRIVATE_KEY, version=version, provider=gateway,router_contract_addr=router,factory_contract_addr=factory)

    # print(f'historical_data: {historical_data}')

    if not params:
        
        print(f"Parameters for model '{model_name}' not found in cache. Triggering classifier...")
        print(f'selling all assets to WETH')
        global_contracts = global_cache.get('global_contracts', {})
        global_decimals = global_cache.get('global_decimals', {})

        # Add WETH to the global cache if not already present
        if 'WETH' not in global_contracts:
            global_contracts['WETH'] = WETH_ADDRESS
            print("Added WETH to global contracts.")

        if 'WETH' not in global_decimals:
            global_decimals['WETH'] = 18
            print("Added WETH decimals to global decimals.")

        if global_contracts and global_decimals:
            print(f"Selling all assets to WETH using global contracts...")
            new_compositions = {
                'WETH': 1.0  # 100% allocation to WETH
            }

            # sell_all_assets_to_weth(uniswap, global_contracts, global_decimals, ACCOUNT_ADDRESS)
            rebalance_portfolio(
                uniswap, 
                global_contracts, 
                global_decimals, 
                new_compositions, 
                ACCOUNT_ADDRESS
            )

        first_run_bool = True
        
        # Trigger classifier and wait for it to complete
        
        await trigger_new_portfolio(
            api_url=os.getenv('CLASSIFIER_URL'),
            network='arbitrum',  # Replace with the correct network
            current_version=model_name,
            backtest_period=0,  # Adjust the backtest period if necessary
            start_date=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            current_date=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        )

        # print(f"Waiting for the classifier to complete for model '{model_name}'...")
        # timeout = 24 * 60 * 60  # 24-hour timeout
        # start_time = dt.datetime.now()

        # while True:
        #     await asyncio.sleep(300)  # Wait 5 minutes before checking again

        #     classifier_status = await is_classifier_done(os.getenv('CLASSIFIER_URL'), model_name)

        #     if classifier_status:
        #         print(f"✅ Classifier completed for model '{model_name}'.")
        #         break  # Exit loop when done

        #     # If classifier is still processing, we continue waiting
        #     print(f"🔄 Classifier {model_name} is still processing...")

        #     # Check if timeout exceeded
        #     elapsed_time = (dt.datetime.now() - start_time).total_seconds()
        #     if elapsed_time > timeout:
        #         print("❌ Classifier did not complete within timeout. Resetting classifier status.")
        #         global_cache.set('classifier_running', False)  # Reset only after full timeout
        #         raise TimeoutError(f"Classifier did not complete within 24 hours.")

        # # Wait for the classifier to complete
        # print(f"Waiting for the classifier to complete for model '{model_name}'...")
        # timeout = 24 * 60 * 60  # 24 hours in seconds
        # start_time = dt.datetime.now()

        # while True:
        #     await asyncio.sleep(300)  # Wait for 5 minutes between checks
        #     if await is_classifier_done(os.getenv('CLASSIFIER_URL'), model_name):
        #         print(f"Classifier completed for model '{model_name}'.")
        #         break

        #     # Check if timeout exceeded
        #     elapsed_time = (dt.datetime.now() - start_time).total_seconds()
        #     if elapsed_time > timeout:
        #         raise TimeoutError(f"Classifier did not complete within the timeout period of {timeout // 3600} hours.")

        # Retry loading parameters from the cache
        params = params_cache.get(f'{model_name} Params', {})
        if not params:
            raise ValueError(f"Parameters for model '{model_name}' not found in cache even after classifier completion.")

    # Extract key parameters
    chain = params['network']
    classifier_data = params['classifier data']
    backtest_period = params['days'] * 24

    print(f'backtest_period: {backtest_period}')

    print(f'model_name: {model_name}')
    # breakpoint()

    # Load other data from the cache
    historical_data = cache.get(f'{model_name} historical_data', pd.DataFrame())
    historical_port_values = cache.get(f'{model_name} historical_port_values', pd.DataFrame())
    oracle_prices = cache.get(f'{model_name} oracle_prices', pd.DataFrame())
    last_rebalance_time = cache.get(f'{model_name} last_rebalance_time', None)
    model_actions = cache.get(f'{model_name} actions', pd.DataFrame())

    return {
        "model_name": model_name,
        "params": params,
        "historical_data": historical_data,
        "historical_port_values": historical_port_values,
        "oracle_prices": oracle_prices,
        "last_rebalance_time": last_rebalance_time,
        "model_actions": model_actions,
        "backtest_period": backtest_period,
        "classifier_data": classifier_data,
        "version":model_name
    }

erc20_abi = [
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    }
]

three_month_tbill_historical_api = "https://api.stlouisfed.org/fred/series/observations?series_id=TB3MS&file_type=json"

try: 
    three_month_tbill = fetch_and_process_tbill_data(api_url=three_month_tbill_historical_api, 
                                                     data_key="observations",
                                                       date_column="date", 
                                                       value_column="value")
    three_month_tbill['decimal'] = three_month_tbill['value'] / 100
    current_risk_free = three_month_tbill['decimal'].iloc[-1]
    print(f"3-month T-bill data fetched: {three_month_tbill.tail()}")
except Exception as e:
    print(f"Error in fetching tbill data: {e}")

scheduler = BackgroundScheduler()

# if last_rebalance_time != None:
#     print(f'last rebalance time: {last_rebalance_time}')

def shutdown_scheduler(exception=None):
        if scheduler.running:
            scheduler.shutdown()
            logger.info("Scheduler shut down.")

def backup_and_clear_cache():
    # Save cache to a file
    with open('cache_backup.pkl', 'wb') as f:
        pickle.dump(dict(cache), f)
    
    cache.clear()
    print("Cache cleared and backup saved.")

async def wait_for_classifier_completion(api_url, model_name, max_wait_hours=6, check_interval_minutes=30):
    """Wait up to 6 hours for classifier completion, checking every 30 min."""
    max_attempts = (max_wait_hours * 60) // check_interval_minutes

    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(f"{api_url}/status", params={"model_name": model_name})
                
                if response.status_code == 200 and response.json().get("status") == "done":
                    print(f"✅ Classifier completed for {model_name}.")
                    return True
                
                print(f"⌛ Waiting for classifier completion... (Attempt {attempt+1}/{max_attempts})")

        except Exception as e:
            print(f"⚠️ Error checking classifier status: {e}")

        await asyncio.sleep(check_interval_minutes * 60)  # Wait before next check

    print(f"❌ Classifier did not complete within {max_wait_hours} hours.")
    return False

async def send_classifier_request(api_url, payload):
    """Send a classifier request once and wait up to 6 hours for completion."""
    
    print(f"📡 Sending classifier request to {api_url}/classifier-portfolio with payload: {payload}")

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{api_url}/classifier-portfolio", json=payload)

        print(f"🔄 Response: {response.status_code}, {response.text}")

    except Exception as e:
        print(f"⚠️ Classifier trigger request failed due to error: {e}")

    # ✅ **Now we always enter the wait loop**
    print(f"⏳ Waiting up to 6 hours for classifier '{payload['model_name']}' to complete...")

    max_wait_time = 6 * 60 * 60  # 6 hours in seconds
    check_interval = 300  # Check every 5 minutes
    elapsed_time = 0

    while elapsed_time < max_wait_time:
        await asyncio.sleep(check_interval)
        elapsed_time += check_interval

        # Periodically check classifier status
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{api_url}/status", params={"model_name": payload["model_name"]})

            if response.status_code == 200:
                status = response.json().get("status")

                if status == "done":
                    print(f"✅ Classifier completed successfully after {elapsed_time // 60} minutes.")
                    return True  # ✅ Done, return success
                
                print(f"🔄 Still processing... ({elapsed_time // 60} minutes elapsed)")

        except Exception as e:
            print(f"⚠️ Failed to check classifier status: {e}")

    # ❌ **If classifier doesn't finish in 6 hours, return False**
    print(f"❌ Classifier did not complete in 6 hours. Selling Assets to WETH.")

    # Ensure Uniswap & global contracts are initialized before selling
    w3, gateway, factory, router, version = network_func('arbitrum')
    uniswap = Uniswap(
        address=ACCOUNT_ADDRESS,
        private_key=PRIVATE_KEY,
        version=version,
        provider=gateway,
        router_contract_addr=router,
        factory_contract_addr=factory
    )

    global_contracts = global_cache.get('global_contracts', {})
    global_decimals = global_cache.get('global_decimals', {})

    # Ensure WETH is in global contracts
    global_contracts.setdefault('WETH', WETH_ADDRESS)
    global_decimals.setdefault('WETH', 18)

    # ✅ **Sell all assets to WETH**
    new_compositions = {'WETH': 1.0}
    rebalance_portfolio(uniswap, global_contracts, global_decimals, new_compositions, ACCOUNT_ADDRESS)

    print("✅ Sold all assets to WETH due to classifier timeout.")
    return False  # Timeout

async def wait_for_classifier_api(api_url, model_name, max_attempts=10, wait_seconds=10):
    """Wait for the classifier API to be available before sending a request."""
    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{api_url}/status", params={"model_name": model_name})

                if response.status_code in [200, 404]:  # API is responsive
                    print(f"✅ Classifier API is up (Attempt {attempt+1})")
                    return True

        except Exception as e:
            print(f"⏳ Waiting for classifier API (Attempt {attempt+1}/{max_attempts})... Error: {e}")

        await asyncio.sleep(wait_seconds)

    print("❌ Classifier API did not start in time.")
    return False


async def trigger_new_portfolio(api_url, network, current_version, backtest_period, start_date, current_date):
    global first_run_bool

    print(f"API URL: {api_url}")
    
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    if isinstance(current_date, str):
        current_date = dt.datetime.strptime(current_date, "%Y-%m-%d %H:%M:%S")

    remaining_hours = backtest_period - (current_date - start_date).total_seconds() / 3600
    print(f"Remaining hours: {remaining_hours}, Trigger threshold: 12 hours")

    if remaining_hours <= 12:
        new_version = current_version if first_run_bool else increment_version(current_version)
        first_run_bool = False
        print(f"New Version: {new_version}")

        cache.set('new_version', new_version)

        payload = {"model_name": new_version, "network": network}

        print(f'Payload: {payload}')

        # Ensure API is available before sending request
        api_ready = await wait_for_classifier_api(api_url, new_version)
        if not api_ready:
            print(f"Classifier API is not available. Skipping request for {new_version}.")
            return current_version  

        # ✅ **Trigger classifier once**
        classifier_started = await send_classifier_request(api_url, payload)
        if not classifier_started:
            print(f"❌ Failed to start classifier for {new_version}. Skipping.")
            return current_version  

        # ✅ **Wait for up to 6 hours**
        classifier_done = await wait_for_classifier_completion(api_url, new_version, max_wait_hours=6)

        if classifier_done:
            print(f"✅ Classifier completed successfully for {new_version}.")
            return new_version
        else:
            print(f"❌ Classifier did not complete within 6 hours. Selling all assets to WETH.")

            # Ensure Uniswap & global contracts are initialized
            w3, gateway, factory, router, version = network_func('arbitrum')
            uniswap = Uniswap(
                address=ACCOUNT_ADDRESS,
                private_key=PRIVATE_KEY,
                version=version,
                provider=gateway,
                router_contract_addr=router,
                factory_contract_addr=factory
            )

            global_contracts = global_cache.get('global_contracts', {})
            global_decimals = global_cache.get('global_decimals', {})

            # Ensure WETH is in global contracts
            global_contracts.setdefault('WETH', WETH_ADDRESS)
            global_decimals.setdefault('WETH', 18)

            # ✅ **Sell all assets to WETH**
            new_compositions = {'WETH': 1.0}
            rebalance_portfolio(uniswap, global_contracts, global_decimals, new_compositions, ACCOUNT_ADDRESS)

            print("✅ Sold all assets to WETH due to classifier timeout.")

        return new_version
    else:
        print(f"Not triggering. Remaining Hours: {remaining_hours:.2f}")
        return current_version


def should_trigger_classifier(start_date, backtest_period, current_date):
    global remaining_time
    
    start_datetime = dt.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    current_datetime = dt.datetime.strptime(current_date, "%Y-%m-%d %H:%M:%S")
    backtest_end = start_datetime + timedelta(hours=backtest_period)

    # Calculate remaining time in hours
    remaining_time = (backtest_end - current_datetime).total_seconds() / 3600
    
    # 🛑 **Check if results already exist**
    cached_results = cache.get(f'{model_name} classifier_results')
    if cached_results is not None:
        print(f"Skipping classifier run: Cached results already exist for {model_name}")
        return False, remaining_time

    print(f'remaining_time <= 12: {remaining_time <= 12}')
    print(f'remaining_time: {remaining_time}')

    return remaining_time <= 12, remaining_time


async def is_classifier_done(api_url, model_name):
    """Check if the classifier process has completed."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{api_url}/status", params={"model_name": model_name})
            if response.status_code == 200:
                status = response.json().get("status")
                if status == "done":
                    print(f"Classifier {model_name} has completed successfully.")
                    return True
                elif status == "processing":
                    print(f"Classifier {model_name} is still processing.")
                    return False
                elif status == "error":
                    print(f"Classifier {model_name} encountered an error.")
                    return False
                else:
                    print(f"Classifier {model_name} status is unknown.")
                    return False
            elif response.status_code == 404:
                print(f"Classifier {model_name} status is not found.")
                return False
            else:
                print(f"Unexpected response while checking classifier status: {response.status_code}, {response.text}")
                return False
        except httpx.RequestError as e:
            print(f"Error while checking classifier status: {e}")
            return False


def should_rebalance(current_time, actions_df, rebalancing_frequency):
    global last_rebalance_time

    current_time = pd.to_datetime(current_time).replace(tzinfo=None, minute=0, second=0, microsecond=0)

    # Print debug information
    print(f"last rebal time: {last_rebalance_time}")
    print(f"current time: {current_time}")
    print(f"actions df: {actions_df}")
    print(f"rebalancing frequency: {rebalancing_frequency}")

    # Calculate next rebalance date for rebalancing_frequency = 1 (hourly rebalancing)
    if rebalancing_frequency == 1:
        next_rebalance_date = (last_rebalance_time or current_time) + pd.Timedelta(hours=1)
        print(f"Next rebalance date (hourly): {next_rebalance_date}")

        if last_rebalance_time is None or (current_time - last_rebalance_time).total_seconds() >= 3600:
            last_rebalance_time = current_time
            cache.set('last_rebalance_time', last_rebalance_time)
            print(f"last_rebalance_time updated to: {last_rebalance_time}")
            return True, next_rebalance_date
        else:
            print("Rebalancing is not required at this time.")
            return False, next_rebalance_date
    else:
        # Sort actions_df by date
        actions_df = actions_df.sort_values(by='Date')
        print(f"actions df: {actions_df}")

        # Get last rebalance time from actions_df
        last_rebalance_time_from_actions = pd.to_datetime(actions_df['Date'].iloc[-1])
        last_rebalance_time_from_actions = last_rebalance_time_from_actions.replace(tzinfo=None, minute=0, second=0, microsecond=0)
        print(f"Last rebalance time from actions: {last_rebalance_time_from_actions}")

        # Calculate hours since last rebalance
        hours_since_last_rebalance = (current_time - last_rebalance_time_from_actions).total_seconds() / 3600
        print(f"Hours since last rebalance: {hours_since_last_rebalance}")

        # Calculate next rebalance date
        next_rebalance_date = last_rebalance_time_from_actions + pd.Timedelta(hours=rebalancing_frequency)
        print(f"Next rebalance date: {next_rebalance_date}")

        if hours_since_last_rebalance >= rebalancing_frequency:
            print("Rebalancing required based on frequency.")
            if last_rebalance_time is None or (current_time - last_rebalance_time).total_seconds() >= rebalancing_frequency * 3600:
                last_rebalance_time = current_time
                cache.set('last_rebalance_time', last_rebalance_time)
                print(f"last_rebalance_time updated to: {last_rebalance_time}")
                return True, next_rebalance_date
            else:
                print("Rebalancing is not required at this time.")
                return False, next_rebalance_date
        else:
            print("Rebalancing is not required at this time.")
            return False, next_rebalance_date

def update_historical_data(live_comp):
    new_data = pd.DataFrame([live_comp])
    historical_data = cache.get(f'{model_name} historical_data', pd.DataFrame())
    historical_data = pd.concat([historical_data, new_data]).reset_index(drop=True)
    historical_data.drop_duplicates(subset='date', keep='last', inplace=True)
    cache.set(f'{model_name} historical_data', historical_data)

def update_portfolio_data(values):
    print(f'values: {values}')
    values = pd.DataFrame([values])
    historical_port_values = cache.get(f'{model_name} historical_port_values')
    historical_port_values = pd.concat([historical_port_values, values]).reset_index(drop=True)
    print(f'historical_port_values: {historical_port_values}')
    # breakpoint()
    historical_port_values.drop_duplicates(subset='date', keep='last', inplace=True)
    cache.set(f'{model_name} historical_port_values', historical_port_values)

def update_global_cache(token_contracts, token_decimals):
    global global_contracts, global_decimals
    """
    Update the global cache with new token contracts and decimals.

    Args:
        token_contracts (dict): A dictionary of token symbols to addresses.
        token_decimals (dict): A dictionary of token symbols to decimals.
    """
    # Retrieve existing global data
    global_contracts = global_cache.get('global_contracts', {})
    global_decimals = global_cache.get('global_decimals', {})
    
    # Update the global cache
    global_contracts.update(token_contracts)
    global_decimals.update(token_decimals)

    # Save the updated data back to the global cache
    global_cache.set('global_contracts', global_contracts)
    global_cache.set('global_decimals', global_decimals)

    print(f"Updated global contracts: {global_contracts}")
    print(f"Updated global decimals: {global_decimals}")

def update_price_data(values):
    print(f'values at update price: {values}')

    # Ensure the 'hour' column exists by resetting index if necessary
    if isinstance(values.index, pd.DatetimeIndex):
        values = values.reset_index().rename(columns={'index': 'hour'})
    
    if 'hour' not in values.columns:
        raise ValueError("The provided DataFrame must have a 'hour' column.")

    oracle_prices = cache.get(f'{model_name} oracle_prices', pd.DataFrame())
    # breakpoint()

    # Concatenate the new values with the existing oracle_prices
    oracle_prices = pd.concat([oracle_prices, values]).drop_duplicates(subset='hour', keep='last').reset_index(drop=True)

    print(f'model_name: {model_name}')

    # breakpoint()
    
    # Cache the updated oracle_prices
    cache.set(f'{model_name} oracle_prices', oracle_prices)

    print(f'Updated oracle_prices:\n{oracle_prices}')

def update_weighted_returns(df):
    weighted_returns = cache.get(f'{model_name} weighted returns',pd.DataFrame())
    weighted_returns = (
        pd.concat([weighted_returns, df])
        .reset_index()  # Temporarily move index to columns
        .drop_duplicates(subset='index', keep='last')  # Use 'index' column to drop duplicates
        .set_index('index')  # Restore index
    )
    print(f'weighted returns: {weighted_returns}')
    cache.set(f'{model_name} weighted returns',weighted_returns)
    weighted_returns.to_csv(os.path.join(live_results_path, f'{model_name}.csv'))

def update_model_actions(actions):
    new_data = pd.DataFrame(actions)
    print(f'new data: {new_data}')
    model_actions = cache.get(f'{model_name} actions', pd.DataFrame())
    print(f'model actions before update: {model_actions}')
    model_actions = pd.concat([model_actions, new_data]).reset_index(drop=True)
    model_actions.drop_duplicates(subset='Date', keep='last', inplace=True)
    cache.set(f'{model_name} actions', model_actions)

def increment_version(version: str) -> str:
    # Extract the prefix and numeric part
    prefix = version[:1]  # Assuming 'v' is always the prefix
    num = version[1:]  # Extract the numeric part

    # Increment the numeric part and ensure it stays zero-padded
    new_num = int(num) + 1

    return f"{prefix}{new_num:02}"

def get_token_price(token='0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'):
    url = f"https://api.coingecko.com/api/v3/simple/token_price/ethereum?contract_addresses={token}&vs_currencies=usd"

    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": COINGECKO_API_KEY
    }

    response = requests.get(url, headers=headers)

    

    eth_data = response.json()

    eth_df = pd.DataFrame(eth_data)
    eth_usd = eth_df[f'{token}'].values[0]

    print(eth_usd)

    return eth_usd

async def sell_all_assets_to_weth():
    """Sell all assets to WETH if classifier fails after 6 hours."""
    print("⚠️ Classifier did not finish. Selling all assets to WETH.")

    # Ensure Uniswap & global contracts are initialized
    w3, gateway, factory, router, version = network_func('arbitrum')
    uniswap = Uniswap(
        address=ACCOUNT_ADDRESS,
        private_key=PRIVATE_KEY,
        version=version,
        provider=gateway,
        router_contract_addr=router,
        factory_contract_addr=factory
    )

    global_contracts = global_cache.get('global_contracts', {})
    global_decimals = global_cache.get('global_decimals', {})

    # Ensure WETH is in global contracts
    global_contracts.setdefault('WETH', WETH_ADDRESS)
    global_decimals.setdefault('WETH', 18)

    # ✅ **Sell all assets to WETH**
    new_compositions = {'WETH': 1.0}
    rebalance_portfolio(uniswap, global_contracts, global_decimals, new_compositions, ACCOUNT_ADDRESS)

    print("✅ Sold all assets to WETH due to classifier timeout.")

classifier_api_url = os.getenv('CLASSIFIER_URL')

def save_model_data_to_csv(model_name, cache):
    """Save old model's historical data before clearing it."""
    try:
        # Save historical data
        historical_data = cache.get(f'{model_name} historical_data', pd.DataFrame())
        if isinstance(historical_data, pd.DataFrame) and not historical_data.empty:
            historical_data.to_csv(f"cache_storage/{model_name}_historical_data.csv", index=False)
            print(f"Saved {model_name} historical data to CSV.")

        # Save historical portfolio values
        historical_port_values = cache.get(f'{model_name} historical_port_values', pd.DataFrame())
        if isinstance(historical_port_values, pd.DataFrame) and not historical_port_values.empty:
            historical_port_values.to_csv(f"cache_storage/{model_name}_portfolio_values.csv", index=False)
            print(f"Saved {model_name} portfolio values to CSV.")

        # Save oracle prices
        oracle_prices = cache.get(f'{model_name} oracle_prices', pd.DataFrame())
        if isinstance(oracle_prices, pd.DataFrame) and not oracle_prices.empty:
            oracle_prices.to_csv(f"cache_storage/{model_name}_oracle_prices.csv", index=False)
            print(f"Saved {model_name} oracle prices to CSV.")

        # Save model actions
        model_actions = cache.get(f'{model_name} actions', pd.DataFrame())
        if isinstance(model_actions, pd.DataFrame) and not model_actions.empty:
            model_actions.to_csv(f"cache_storage/{model_name}_actions.csv", index=False)
            print(f"Saved {model_name} actions to CSV.")

        weighted_returns = cache.get(f'{model_name} weighted returns',pd.DataFrame())
        if isinstance(weighted_returns, pd.DataFrame) and not weighted_returns.empty:
            weighted_returns.to_csv(f"cache_storage/{model_name}_weighted_returns.csv")
            print(f"Saved {model_name} returns to CSV.")

        # 🔥 **Clear old model data after saving**
        cache.delete(f'{model_name} historical_data')
        cache.delete(f'{model_name} historical_port_values')
        cache.delete(f'{model_name} oracle_prices')
        cache.delete(f'{model_name} actions')
        cache.delete(f'{model_name} weighted returns')

    except Exception as e:
        print(f"Error saving model data: {e}")

def create_app():
    app = Flask(__name__)

    # with app.app_context():
    #     try:
    #         print("Initializing and running the model...")
    #         asyncio.run(run_model())
    #         print("Model initialization complete.")
    #     except Exception as e:
    #         logger.error(f"Error during model initialization: {e}")

    def fetch_and_cache_data():
        with app.app_context():
            logger.info("Scheduled task running.")
            print("Scheduled task running.")
            asyncio.run(run_model())

    scheduler.add_job(
        fetch_and_cache_data, 
        trigger=CronTrigger(minute='0'),  # Ensures it runs at the top of every hour
        id=f'{model_name} data_fetch_job',             # A unique identifier for this job
        replace_existing=True            # Ensures the job is replaced if it's already running when you restart the app
    )

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/clear-cache', methods=['POST'])
    def clear_cache():
        # Use silent=True to avoid errors if no JSON is provided
        request_data = request.get_json(silent=True) or {}

        # Check which cache to clear
        if request_data.get('params', False):
            print('Clearing the params cache...')
            params_cache.clear()
            cache.clear()

        elif request_data.get('global', False): 
            print('Clearing the global cache...')
            global_cache.clear()
            cache.clear()

        else:
            print('Clearing the main cache...')
            cache.clear()

        return jsonify({"status": "cache cleared"}), 200

    @app.route('/new-portfolio', methods=['POST'])
    async def new_portfolio():
        global backtest_period
        await trigger_new_portfolio(
            api_url=os.getenv('CLASSIFIER_URL'),
            network='arbitrum',  # Replace with the correct network
            current_version=model_name,
            backtest_period=0,  # Adjust the backtest period if necessary
            start_date=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            current_date=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        )

        return jsonify({"status":"triggered classifier and model"})

    @app.route('/run-model')
    async def run_model(seed=20, start_date=None):
        global today, prices_df, current_risk_free, TOKEN_CONTRACTS, TOKEN_DECIMALS
        global params_cache, cache, model_name, model_data, backtest_period, global_contracts, global_decimals

        # Fetch cached historical data
        historical_data = cache.get(f'{model_name} historical_data', pd.DataFrame())
        print(f'Historical Data:\n{historical_data}')

        # Determine start date
        if start_date is None:
            start_date = (
                pd.to_datetime(historical_data['date'].min())
                if not historical_data.empty else dt.datetime.now(dt.timezone.utc)
            )
        start_date = start_date.strftime('%Y-%m-%d %H:00:00')

        print('Start Date:', start_date)

        # Check if the classifier should be triggered
        should_trigger, remaining_time = should_trigger_classifier(
            start_date=start_date,
            backtest_period=backtest_period,
            current_date=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        )

        if should_trigger:
            print(f"Triggering classifier. Remaining time: {remaining_time:.2f} hours.")
            
            # Ensure the classifier is triggered BEFORE checking for status
            print(f"Triggering classifier...")
            await trigger_new_portfolio(
                classifier_api_url,
                chain,
                model_name,
                backtest_period=backtest_period,
                start_date=start_date,
                current_date=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            )

            print(cache.get('new_version'))

            # breakpoint()

            new_version = cache.get('new_version')

            # print(f"Waiting for the classifier to complete for model '{model_name}'...")
            # timeout = 24 * 60 * 60  # 24-hour timeout
            # start_time = dt.datetime.now()

            # while True:
            #     await asyncio.sleep(300)  # Wait 5 minutes before checking again

            #     classifier_status = await is_classifier_done(os.getenv('CLASSIFIER_URL'), model_name)

            #     if classifier_status:
            #         print(f"✅ Classifier completed for model '{model_name}'.")
            #         break  # Exit loop when done

            #     # If classifier is still processing, we continue waiting
            #     print(f"🔄 Classifier {model_name} is still processing...")

            #     # Check if timeout exceeded
            #     elapsed_time = (dt.datetime.now() - start_time).total_seconds()
            #     if elapsed_time > timeout:
            #         print("❌ Classifier did not complete within timeout. Resetting classifier status.")
            #         global_cache.set('classifier_running', False)  # Reset only after full timeout
            #         raise TimeoutError(f"Classifier did not complete within 24 hours.")

            # Now wait for classifier to complete
            # classifier_done = await wait_for_classifier_completion(classifier_api_url, new_version)

            # if classifier_done:
            #     print('Classifier is done')

            #     
            #     print(f'New Version: {new_version}')

            #     # Reinitialize Uniswap and Contracts
            #     w3, gateway, factory, router, version = network_func('arbitrum')
            #     account = Account.from_key(PRIVATE_KEY)
            #     w3.eth.default_account = account.address

            #     uniswap = Uniswap(
            #         address=ACCOUNT_ADDRESS,
            #         private_key=PRIVATE_KEY,
            #         version=version,
            #         provider=gateway,
            #         router_contract_addr=router,
            #         factory_contract_addr=factory
            #     )

            #     # **🔹 Ensure we have the latest contract addresses before selling**
            #     global_contracts = global_cache.get('global_contracts', {})
            #     global_decimals = global_cache.get('global_decimals', {})

            #     # **Ensure WETH is included**
            #     global_contracts.setdefault('WETH', WETH_ADDRESS)
            #     global_decimals.setdefault('WETH', 18)

            #     # **🔹 Sell all assets to WETH before rebalancing**
            #     print(f"Selling all assets to WETH before transitioning to {new_version}...")
            #     sell_all_assets_to_weth(uniswap, global_contracts, global_decimals, ACCOUNT_ADDRESS)

            #     # **🔹 Rebalance the portfolio using WETH as the base**
            #     print(f"Rebalancing into new portfolio: {new_version}")
            #     rebalance_portfolio(uniswap, global_contracts, global_decimals, {'WETH': 1.0}, ACCOUNT_ADDRESS)

            # Save old model data before clearing
            save_model_data_to_csv(model_name, cache)

            # Reset model data
            model_name = new_version
            model_data = await initialize_model_cache()

        w3, gateway, factory, router, version = network_func(chain)

        params = params_cache.get(f'{model_name} Params')

        # breakpoint() 

        account = Account.from_key(PRIVATE_KEY)
        w3.eth.default_account = account.address

        # breakpoint()

        classifier_data = params['classifier data']
        portfolio = classifier_data[['symbol','token_address']]

        print(f'connected to: {account} on {chain}')

        # Update TOKEN_CONTRACTS and ensure WETH is included
        TOKEN_CONTRACTS = {
            row['symbol']: row['token_address'] for _, row in portfolio.iterrows()
        }
        TOKEN_CONTRACTS['WETH'] = WETH_ADDRESS
        TOKEN_DECIMALS = get_token_decimals(TOKEN_CONTRACTS, w3)
        global_cache.set('global_contracts', TOKEN_CONTRACTS)
        global_cache.set('global_decimals',TOKEN_DECIMALS)
        print(f"Portfolio updated with new version: {model_name}")
        print(f"Updated TOKEN_CONTRACTS: {TOKEN_CONTRACTS}")

        # breakpoint()

        cached_data = cache.get(f'{model_name} latest_data')

        # print(cached_data['results'])

        data_start_date = dt.datetime.now(dt.timezone.utc) - timedelta(hours=5)
        data_start_date = data_start_date.strftime('%Y-%m-%d %H:00:00')

        today_utc = dt.datetime.now(dt.timezone.utc) 
        formatted_today_utc = today_utc.strftime('%Y-%m-%d %H:00:00')

        data_version = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H-00-00')
        data_version_comp = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:00:00') 

        if cached_data and 'results' in cached_data and cached_data['results'].get('last run (UTC)') is not None:
            if cached_data['results']['last run (UTC)'] == data_version_comp:
                print(f"cached_data['results']['last run (UTC)']: {cached_data['results']['last run (UTC)']}")
                print(f"data_version_comp: {data_version_comp}")
                print("Using cached data")
                return jsonify(cached_data)

        end_date = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:00:00') 

        print(list(params_cache.iterkeys()))
        
        params = params_cache.get(f'{model_name} Params')
        network = params['network']
        
        print(f'model_name: {model_name} \nnetwork: {network}')

        test_start_date = params_cache.get(f'{model_name} Test Start') # First hour after end of training dataset; ensures no overlap
        # filtered_assets = params['assets']

        print(f'params: {params}')

        # breakpoint()
        
        rebalance_frequency = params['rebalance_frequency']

        test_start_date = str(test_start_date)
        print(f'test_start_date: {test_start_date}')

        diff_tolerance_threshold = 0.05 #percent for diff between new portfolio action and current portfolio

        start_date = str(data_start_date)

        # start_date = dt.datetime.strptime(start_date, '%Y-%m-%d %H:00:00') 
        print(f'sql start date: {start_date}')

        print(f'model_name: {model_name}')

        data, prices_df = prices_data_func(
                            network=network, 
                            name=model_name,
                            api_key=flipside_api_key,
                            use_cached_data=False,
                            function=None,
                            start_date=start_date,
                            # filtered_assets=filtered_assets
                            )
        # filtered_columns = [col for col in prices_df.columns if col.endswith('_Price') and col.replace('_Price', '') in filtered_assets]
        
        # prices_df = prices_df[filtered_columns]

        # print(f'filtered_columns: {filtered_assets}')

        print("Filtered prices_df:")
        print(prices_df)

        dxy_start_date = (prices_df.index.min() - timedelta(hours=360)).strftime('%Y-%m-%d %H:00:00') #pull dxy data starting 15 days from earliest price data dt

        try:
            dxy_historical = fetch_and_process_dxy_data(dxy_historical_api, "observations", "date", "value",start_date=dxy_start_date)

            dxy_historical['value'] = dxy_historical['value'].replace(".",np.nan).ffill().bfill()
            hourly_dxy = dxy_historical[['value']].resample('H').ffill()
            hourly_dxy.rename(columns={'value':'DXY'},inplace=True)
            hourly_dxy['DXY'] = hourly_dxy['DXY'].astype(float)
        except Exception as e:
            print(f"Error in fetching DXY data: {e}")
        
        # hourly_dxy.index = pd.to_datetime(hourly_dxy.index.strftime('%Y-%m-%d %H:00:00'))

        hourly_dxy = prepare_data_for_simulation(hourly_dxy, prices_df.index.min().strftime('%Y-%m-%d %H:00:00'), end_date)

        hourly_dxy.index = hourly_dxy.index.tz_localize('UTC')

        # breakpoint()

        prices_df = pd.merge(prices_df,hourly_dxy,left_index=True,right_index=True,how='left').ffill()

        prices_df = prepare_data_for_simulation(prices_df, start_date, end_date)

        # breakpoint()

        model_balances = get_balance(TOKEN_CONTRACTS,TOKEN_DECIMALS,ACCOUNT_ADDRESS,w3)

        latest_prices = {
            token: float(prices_df[f"{token}_Price"].iloc[-1])
            for token in TOKEN_CONTRACTS.keys()
            if f"{token}_Price" in prices_df.columns
        }

        latest_prices['WETH'] = get_token_price()

        update_price_data(prices_df)

        model_balances_usd = convert_to_usd(model_balances,latest_prices,TOKEN_CONTRACTS)
        portfolio_balance = sum(model_balances_usd.values())

        comp_dict = {
            f"{token} comp": balance_usd / portfolio_balance
            for token, balance_usd in model_balances_usd.items()
            # if token != "WETH"  # Skip WETH
        }

        comp_dict["date"] = formatted_today_utc

        update_historical_data(comp_dict)

        historical_data = cache.get(f'{model_name} historical_data')

        portfolio_dict = {
            "Portfolio Value": portfolio_balance,
            "date": formatted_today_utc
        }

        update_portfolio_data(portfolio_dict)

        hist_comp = cache.get(f'{model_name} historical_data')
        print(f'hist_comp: {hist_comp}')
        hist_comp.set_index('date', inplace=True)
        hist_comp.index = pd.to_datetime(hist_comp.index)
        print(f'hist comp for env {hist_comp}')

        print(f'model_name: {model_name}')

        oracle_prices = cache.get(f'{model_name} oracle_prices')
        print(f'oracle_prices: {oracle_prices}')

        oracle_prices_copy = oracle_prices.copy().set_index('hour')
        oracle_prices_copy.index = pd.to_datetime(oracle_prices_copy.index)
        # filtered_columns = [col for col in oracle_prices_copy.columns if col.endswith('_Price') and col.replace('_Price', '') in filtered_assets]

        # print(f'filtered_assets: {filtered_assets}')

        # oracle_prices_copy = oracle_prices_copy[filtered_columns]
        # breakpoint()

        uniswap = Uniswap(address=ACCOUNT_ADDRESS, private_key=PRIVATE_KEY, version=version, provider=gateway,router_contract_addr=router,factory_contract_addr=factory)

        def run_sim(model,prices_df, hist_comp, seed, rebalancing_frequency):
            print(f'prices before sim: {prices_df}')
            prices_df.dropna(inplace=True)

            model = PPO.load(f"AI_Models/{model}")

            # breakpoint()
            
            env = Portfolio(df=prices_df, compositions=hist_comp, seed=seed, rebalance_frequency=rebalancing_frequency,
                            risk_free_annual=current_risk_free)

            set_global_seed(env,seed)

            states = []
            rewards = []
            actions = []
            portfolio_values = []
            compositions = []
            dates = []

            # Reset the environment to get the initial state
            state, _ = env.reset(seed=seed)  # Get the initial state
            done = False

            while not done:
                # Use the model to predict the action
                action, _states = model.predict(state)
                
                # Take a step in the environment
                next_state, reward, done, truncated, info = env.step(action)
                
                # Break the loop if done to avoid processing a None state
                if done:
                    print("Episode done. Exiting the loop.")
                    break
                
                # Normalize the action to ensure it sums to 1
                # action = action / np.sum(action)
                
                # Store the results
                if next_state is not None:
                    states.append(next_state.flatten())  # Ensure the state is flattened
                rewards.append(reward)
                actions.append(action.flatten())  # Ensure the action is flattened
                compositions.append(env.portfolio)  # Store the portfolio composition
                print(f'Action: {action}')

                # Update the state
                state = next_state

                # Print debug information
                print(f"Step: {env.current_step}")
                print(f"State: {next_state}")
                print(f'Action: {action}')
                print(f"Reward: {reward}")
                print(f"Done: {done}")
                print(f"Info: {info}")

            states_df = env.get_states_df()
            rewards_df = env.get_rewards_df()
            actions_df = env.get_actions_df()
            returns_df = env.get_returns_df()
            sortino_ratios = env.get_sortino_ratios_df()


            return states_df, rewards_df, actions_df, returns_df, sortino_ratios,rebalancing_frequency,sortino_ratios
        
        prices_df = prices_df.drop(columns='DXY', errors='ignore')
        
        hist_comp.index = pd.to_datetime(hist_comp.index)
        # filtered_columns = [col for col in hist_comp.columns if col.endswith(' comp') and col.replace(' comp', '') in filtered_assets]
        print(f'hist_comp: {hist_comp}')
        # print(f'hist_comp: {filtered_columns}')
        # breakpoint()
        # hist_comp = hist_comp[filtered_columns]

        
        # breakpoint()

        hist_comp = hist_comp.drop(columns=["WETH comp"], errors="ignore")
        
        states_df, rewards_df, actions_df, portfolio_values_df, composition,rebalancing_frequency,sortino_ratios = run_sim(
        model_name,oracle_prices_copy[oracle_prices_copy.index>=hist_comp.index.min()],hist_comp,seed,rebalance_frequency
        )

        rewards_df.to_csv(r'E:\Projects\portfolio_optimizers\classifier_optimizer\data\rewards.csv')

        print(f'actions_df: {actions_df}')

        update_model_actions(actions_df.to_dict())

        model_actions = cache.get(f'{model_name} actions')

        current_time = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:00:00')
        current_time = pd.to_datetime(current_time)

        print(f'model_actions: {model_actions}')

        # breakpoint()

        # Create the dictionary with compositions
        new_compositions = {
            token: float(model_actions.iloc[-1].get(f"{token}_weight", 0)) for token in TOKEN_CONTRACTS
        }

        # Explicitly add 'WETH' with a composition of 0 if not already included
        if 'WETH' not in new_compositions:
            new_compositions['WETH'] = 0.0

        print(f'new_compositions: {new_compositions}')

        # Convert the dictionary to a DataFrame
        target_comp_df = pd.DataFrame([new_compositions])

        # Reshape the DataFrame for further processing
        reshaped_df = target_comp_df.melt(
            var_name="token",        # Name of the new column for tokens
            value_name="composition" # Name of the new column for compositions
        )

        print("Reshaped DataFrame:")
        print(reshaped_df)

        latest_comp = hist_comp.iloc[-1]

        latest_comp.index = latest_comp.index.str.replace('comp','')
        latest_comp = latest_comp.to_dict()

        should_rebal, next_rebalance_date = should_rebalance(formatted_today_utc, actions_df, rebalance_frequency)

        print(f'at trigger for rebalance')
        print(f'latest_comp: {latest_comp}')
        print(f'new_compositions: {new_compositions}')
        print(f'rebalancing_frequency: {rebalancing_frequency}')
        print(f'should_rebal: {should_rebal}')
        print(f'next_rebalance_date: {next_rebalance_date}')

        if composition_difference_exceeds_threshold(latest_comp, new_compositions, diff_tolerance_threshold):
            print("Composition difference exceeds the threshold. Triggering immediate rebalance.")
            rebalance_portfolio(
                uniswap, 
                TOKEN_CONTRACTS, 
                TOKEN_DECIMALS, 
                new_compositions, 
                ACCOUNT_ADDRESS
            )
        elif should_rebal:
            print("Time-based rebalance condition met.")
            if composition_difference_exceeds_threshold(latest_comp, new_compositions, diff_tolerance_threshold):
                print("Composition difference exceeds the threshold. Triggering immediate rebalance.")
                rebalance_portfolio(
                    uniswap, 
                    TOKEN_CONTRACTS, 
                    TOKEN_DECIMALS, 
                    new_compositions, 
                    ACCOUNT_ADDRESS
                )
            else:
                print(f"No immediate composition difference. Next rebalance scheduled for {next_rebalance_date}.")
        else:
            print(f"No rebalance required at this time. Next rebalance scheduled for {next_rebalance_date}.")
  
        price_returns = calculate_log_returns(oracle_prices_copy[oracle_prices_copy.index>=hist_comp.index.min()])

        print(F'oracle_prices_copy[oracle_prices_copy.index>=hist_comp.index.min()]: {oracle_prices_copy[oracle_prices_copy.index>=hist_comp.index.min()]}')
        print(F'hist_comp: {hist_comp}')

        oracle_prices_copy = oracle_prices_copy.drop(columns='DXY', errors='ignore')

        optimized_weights, returns, sortino_ratio = mvo(oracle_prices_copy[oracle_prices_copy.index>=hist_comp.index.min()], hist_comp, current_risk_free)

        print(F'sortino_ratio:{sortino_ratio} \noptimized_weights:{optimized_weights}')

        model_balances = get_balance(TOKEN_CONTRACTS,TOKEN_DECIMALS,ACCOUNT_ADDRESS,w3)

        model_balances_usd = convert_to_usd(model_balances,latest_prices,TOKEN_CONTRACTS)
        portfolio_balance = sum(model_balances_usd.values())

        portfolio_dict = {
            "Portfolio Value": portfolio_balance,
            "date": end_date
        }

        # Save new portfolio to CSV

        update_portfolio_data(portfolio_dict)

        comp_dict_data = calculate_compositions(model_balances_usd, portfolio_balance)

        comp_dict = {**comp_dict_data, 'date': formatted_today_utc}

        update_historical_data(comp_dict)

        historical_data = cache.get(f'{model_name} historical_data')

        # portfolio_values_df.set_index('Date', inplace=True)
        # portfolio_return = calculate_cumulative_return(portfolio_values_df, 'Portfolio_Value')

        # prices_df.index = pd.to_datetime(prices_df.index)
        print(f'price index: {prices_df.index}')

        # Assuming 'portfolio_values_df' is your DataFrame with a timezone-aware index
        print(f'portfolio index: {portfolio_values_df.index}')
        # portfolio_values_df.index = portfolio_values_df.index.tz_localize(None)

        print(f'returns in live_backend: {returns}')

        hist_returns = calculate_portfolio_returns(hist_comp, price_returns)
        update_weighted_returns(hist_returns.to_frame('Return'))
        print(f'hist_returns in live_backend: {hist_returns}')
        viz_port_values = hist_returns.to_frame('Return')
        # viz_port_values.set_index('Date',inplace=True)

        print(f'viz_port_values index: {viz_port_values.index}')

        print(F'viz_port_values: {viz_port_values}')

        # viz_port_values.set_index('index',inplace=True)

        norm_port = normalize_log_returns(viz_port_values, viz_port_values.index.min(),viz_port_values.index.max(), 100)

        norm_prices = normalize_asset_returns(oracle_prices_copy, viz_port_values.index.min(),viz_port_values.index.max(), 100)

        norm_port.rename(columns={'Return':f'{model_name} Portfolio Value'},inplace=True)

        nom_comp = pd.merge(
            norm_port, norm_prices, left_index=True, right_index=True, how='left'
        )

        nom_comp.to_csv('data/nom_comp.csv')

        try:
            # First merge operation
            analysis_df = pd.merge(
                hist_comp,
                oracle_prices_copy,
                left_index=True,
                right_index=True,
                how='inner'
            )
            logger.info("Merged hist_comp with oracle_prices_copy successfully. Rows: %d, Columns: %d", *analysis_df.shape)

            # Second merge operation
            analysis_df = analysis_df.merge(viz_port_values, left_index=True, right_index=True, how='inner')
            logger.info("Merged analysis_df with viz_port_values successfully. Rows: %d, Columns: %d", *analysis_df.shape)

            # Save DataFrame to CSV
            output_path = 'data/analysis_results.csv'
            analysis_df.to_csv(output_path)
            logger.info("Saved analysis results to CSV at '%s'", output_path)

        except Exception as e:
            logger.error("An error occurred during processing: %s", str(e))

        print(f'historical_data: {historical_data}')
        print(f'nom_comp: {nom_comp}')

        plot_historical_data = historical_data.copy()
        plot_historical_data.set_index('date',inplace=True)
        plot_historical_data.index = pd.to_datetime(plot_historical_data.index)
        plot_historical_data = plot_historical_data.resample('H').ffill().bfill()
        print(f'plot_historical_data: {plot_historical_data.index}')

        print(f'portfolio: {classifier_data["token_address"].unique()}')
        print(f'ACCOUNT_ADDRESS: {ACCOUNT_ADDRESS}')
        print(f'chain: {chain}')

        # flows_data = model_flows(classifier_data['token_address'].unique(),ACCOUNT_ADDRESS,chain)

        # flows_data_df = flipside_api_results(flows_data,flipside_api_key)
        # print(f'flows_data_df: {flows_data_df}')
        # if flows_data_df.empty:
        #     print(f'flows_data_df is empty')
        # else:
        #     flows_data_df.set_index('dt',inplace=True)
        #     flows_data_df.index = pd.to_datetime(flows_data_df.index).strftime('%Y-%m-%d')
        #     daily_flows = flows_data_df.groupby([flows_data_df.index,'symbol','transaction_type'])[['amount_usd']].sum().reset_index().set_index('dt')
        #     daily_flows.index = pd.to_datetime(daily_flows.index)

        print(f'model_name: {model_name}')
        historical_port_values = cache.get(f'{model_name} historical_port_values')
        # breakpoint()
        
        model_fig1, model_fig2, model_fig3, model_fig4, model_fig5, model_fig6 = model_visualizations(nom_comp, plot_historical_data,historical_port_values,sortino_ratios,reshaped_df,model_actions,model_balances_usd,formatted_today_utc, model_name, font_family)

        oracle_prices_copy.to_csv('data/oracle_prices.csv')

        model_name = global_classifier_cache.get('current_model_name')
        portfolio_expected_return = classifier_cache.get(f'{model_name} Expected Return')

        results = {
            'start date (UTC)': hist_comp.index.min().strftime('%Y-%m-%d %H:00:00'),
            'last run (UTC)': formatted_today_utc,
            'next rebalance (UTC)':next_rebalance_date.strftime('%Y-%m-%d %H:00:00'),
            'address':ACCOUNT_ADDRESS,
            'portfolio balance': f"${portfolio_balance:,.2f}",
            'sortino ratio':sortino_ratio,
            'rebalance frequency (hours)':rebalancing_frequency,
            'chain':chain,
            'Hours Till Next Portfolio':remaining_time,
            'Portfolio Expected Return':f"{portfolio_expected_return*100:.2f}%"
        }

        ordered_results = OrderedDict()
        for key in results:
            ordered_results[key] = results[key]

        graph_json_1 = json.dumps(model_fig1.return_fig(), cls=PlotlyJSONEncoder)
        graph_json_2 = json.dumps(model_fig2.return_fig(), cls=PlotlyJSONEncoder)
        graph_json_3 = json.dumps(model_fig3.return_fig(), cls=PlotlyJSONEncoder)
        graph_json_4 = json.dumps(model_fig4.return_fig(), cls=PlotlyJSONEncoder)
        graph_json_5 = json.dumps(model_fig5.return_fig(), cls=PlotlyJSONEncoder)
        graph_json_6 = json.dumps(model_fig6.return_fig(), cls=PlotlyJSONEncoder)
        # graph_json_7 = json.dumps(flows_fig_1.return_fig(), cls=PlotlyJSONEncoder)
        # graph_json_8 = json.dumps(flows_fig_2.return_fig(), cls=PlotlyJSONEncoder)

        cached_data = {"results": ordered_results, "graph_1": graph_json_1, "graph_2": graph_json_2,"graph_3":graph_json_3,"graph_4":graph_json_4,"graph_5":graph_json_5,'graph_6':graph_json_6}

        cache.set(f'{model_name} latest_data', cached_data)

        return jsonify(cached_data)

    @app.route('/cached-data')
    def get_cached_data():
        cached_data = cache.get(f'{model_name} latest_data')
        if cached_data:
            return jsonify(cached_data)
        else:
            return jsonify({"error": "No cached data available"}), 404
            
    return app

global model_data, model_name, params, backtest_period, start_date, first_run_bool

if __name__ == "__main__":
    logger.info("Starting Flask app...")

    # Initialize the model and cache asynchronously
    logger.info("Initializing model and cache...")
    model_data = asyncio.run(initialize_model_cache())
    first_run_bool = False

    # Access initialized data
    model_name = model_data['model_name']
    params = model_data['params']
    historical_data = model_data['historical_data']
    backtest_period = model_data['backtest_period'] 

    print(f'historical_data: {historical_data}')

    logger.info(f"Running model: {model_name}")
    logger.info(f"Backtest period: {backtest_period} hours")
    logger.info(f"Historical data: {historical_data.head() if not historical_data.empty else 'No data available'}")

    # Start the Flask app and scheduler
    app = create_app()
    print("Starting Flask app...")
    scheduler.start()
    logger.info("Scheduler started.")
    app.run(debug=True, use_reloader=False, port=5013)

    logger.info("Flask app has stopped.")
    print("Flask app has stopped.")




