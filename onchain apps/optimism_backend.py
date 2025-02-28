
import pandas as pd
import requests
import numpy as np
import yfinance as yf

import random
import plotly.io as pio
from flask import Flask, render_template, request, jsonify
from web3 import Web3, EthereumTesterProvider
from web3.exceptions import TransactionNotFound,TimeExhausted

import asyncio
import datetime as dt

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
from sql_scripts.queries import lst_portfolio_prices,eth_btc_prices,dao_advisor_portfolio, yield_portfolio_prices,token_prices,model_flows
from models.testnet_model import Portfolio
from python_scripts.apis import token_classifier_portfolio,fetch_and_process_tbill_data
from python_scripts.web3_utils import get_token_decimals,get_balance,convert_to_usd
from python_scripts.plots import create_interactive_sml

# %%
from diskcache import Cache

# %%
from chart_builder.scripts.visualization_pipeline import visualization_pipeline
from chart_builder.scripts.utils import main as chartBuilder

# %%
from uniswap import Uniswap

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

#Blockchain RPCs
GNOSIS_GATEWAY = os.getenv('GNOSIS_GATEWAY')
ARBITRUM_GATEWAY = os.getenv('ARBITRUM_GATEWAY')
OPTIMISM_GATEWAY = os.getenv('OPTIMISM_GATEWAY')
ETHEREUM_GATEWAY = os.getenv('ETHEREUM_GATEWAY')

#Address Credentials
ACCOUNT_ADDRESS = os.getenv('MODEL_ADDRESS')
PRIVATE_KEY = os.getenv('MODEL_KEY')

# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# print(f'Current working directory: {os.getcwd()}')

# original_directory = os.getcwd()

# # Change to 'classifier_optimizer'
# new_directory = os.path.abspath(os.path.join(original_directory, 'classifier_optimizer'))
# os.chdir(new_directory)
# print(f'Directory: {os.getcwd()}')

# Change back to the original directory
# os.chdir(original_directory)

global cache, params, model_name, chain, rebalancing_frequency

# cache_dir = os.path.join(os.getcwd(), 'test_model_cache')
# print(f'Cache directory contents: {os.listdir(cache_dir)}')

current_directory = os.getcwd()
print(f'Current working directory: {current_directory}')

model_name = 'optimism_classifier'

params_cache = Cache('test_model_cache')

print(f'Cache directory: {params_cache.directory}')

cache = Cache(f'live_{model_name}_cache')

print(f'Cache directory: {params_cache.directory}')

params = params_cache.get(f'{model_name} Params')

print(f'params: {params}')

chain = params['network']

historical_data = cache.get(f'{model_name} historical_data', pd.DataFrame())
historical_port_values = cache.get(f'{model_name} historical_port_values', pd.DataFrame())
oracle_prices = cache.get(f'{model_name} oracle_prices',pd.DataFrame())
last_rebalance_time = cache.get(f'{model_name} last_rebalance_time', None)
model_actions = cache.get(f'{model_name} actions', pd.DataFrame()) 

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

if last_rebalance_time != None:
    print(f'last rebalance time: {last_rebalance_time}')

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
    filtered_assets_with_price = [f"{asset}_Price" for asset in filtered_assets]


    return data, prices_df[filtered_assets_with_price]

def shutdown_scheduler(exception=None):
        if scheduler.running:
            scheduler.shutdown()
            logger.info("Scheduler shut down.")
    
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
    global historical_data
    new_data = pd.DataFrame([live_comp])
    historical_data = pd.concat([historical_data, new_data]).reset_index(drop=True)
    historical_data.drop_duplicates(subset='date', keep='last', inplace=True)
    cache.set(f'{model_name} historical_data', historical_data)

def update_portfolio_data(values):
    global historical_port_values
    print(f'values: {values}')
    values = pd.DataFrame([values])
    historical_port_values = pd.concat([historical_port_values, values]).reset_index(drop=True)
    historical_port_values.drop_duplicates(subset='date', keep='last', inplace=True)
    cache.set(f'{model_name} historical_port_values', historical_port_values)

def update_price_data(values):
    global oracle_prices

    # Ensure the 'hour' column exists by resetting index if necessary
    if isinstance(values.index, pd.DatetimeIndex):
        values = values.reset_index().rename(columns={'index': 'hour'})
    
    if 'hour' not in values.columns:
        raise ValueError("The provided DataFrame must have a 'hour' column.")

    # Concatenate the new values with the existing oracle_prices
    oracle_prices = pd.concat([oracle_prices, values]).drop_duplicates(subset='hour', keep='last').reset_index(drop=True)
    
    # Cache the updated oracle_prices
    cache.set(f'{model_name} oracle_prices', oracle_prices)

    print(f'Updated oracle_prices:\n{oracle_prices}')

def update_model_actions(actions):
    global model_actions
    print(f'model actions before update: {model_actions}')
    new_data = pd.DataFrame(actions)
    print(f'new data: {new_data}')
    model_actions = pd.concat([model_actions, new_data]).reset_index(drop=True)
    model_actions.drop_duplicates(subset='Date', keep='last', inplace=True)
    cache.set(f'{model_name} actions', model_actions)

def network(chain):
        if chain == 'gnosis':
            primary_gateway = GNOSIS_GATEWAY  # Replace with your Infura URL
            backup_gateway = 'https://lb.nodies.app/v1/406d8dcc043f4cb3959ed7d6673d311a'  # Your backup gateway
        elif chain == 'arbitrum':
            primary_gateway = ARBITRUM_GATEWAY  # Replace with your Infura URL
            backup_gateway = ARBITRUM_GATEWAY
        elif chain == 'optimism':
            primary_gateway = OPTIMISM_GATEWAY  # Replace with your Infura URL
            backup_gateway = OPTIMISM_GATEWAY
        elif chain == 'ethereum':
            primary_gateway = ETHEREUM_GATEWAY  # Replace with your Infura URL
            backup_gateway = ETHEREUM_GATEWAY

        print(f'Gateway: {primary_gateway}')

        for gateway in [primary_gateway, backup_gateway]:
            w3 = Web3(Web3.HTTPProvider(gateway))
            if w3.is_connected():
                try:
                    latest_block = w3.eth.get_block('latest')['number']  # Only try this if connected
                    print(f"Connected to {chain} via {gateway}: {latest_block} block")
                    return w3, gateway
                except Exception as e:
                    print(f"Connected to {gateway} but failed to fetch latest block. Error: {e}")
            else:
                print(f"Failed to connect to {chain} via {gateway}. Trying next gateway...")

        raise ConnectionError(f"Failed to connect to {chain} network using both primary and backup gateways.")

def rebalance_portfolio(
    uniswap, 
    token_contracts, 
    token_decimals, 
    target_compositions, 
    account_address, 
):
    """
    Rebalances the portfolio by selling all tokens into WETH and then buying target allocations using WETH.

    Parameters:
    - uniswap: Initialized Uniswap class instance.
    - token_contracts: Dict of token addresses.
    - token_decimals: Dict of token decimals.
    - target_compositions: Dict of target compositions as fractions summing to 1.
    - account_address: ETH wallet address.
    - web3: Initialized Web3 instance.
    """

    # WETH address and checksum
    WETH_ADDRESS = '0x4200000000000000000000000000000000000006'
    checksum_weth_address = Web3.to_checksum_address(WETH_ADDRESS)

    # Step 1: Convert Token Addresses to Checksum Format
    checksum_addresses = {token: Web3.to_checksum_address(address) for token, address in token_contracts.items()}

    # Step 2: Sell All Current Token Holdings into WETH
    for token, address in checksum_addresses.items():
        try:
            balance_wei = uniswap.get_token_balance(address)
            balance = balance_wei / 10**token_decimals[token]
            
            # Adjust the balance to avoid precision issues (round down to 6 decimal places)
            adjusted_balance = math.floor(balance * 10**8) / 10**8
            
            if adjusted_balance > 0:
                amount_to_sell = int(adjusted_balance * 10**token_decimals[token])
                print(f"Selling {adjusted_balance:.6f} {token} for WETH")
                uniswap.make_trade(
                    checksum_addresses[token],
                    checksum_weth_address,  # WETH as output token
                    amount_to_sell
                )
                wait_time = random.randint(15, 30)
                print(f"Waiting {wait_time} seconds before the next call...")
                time.sleep(wait_time)
        except Exception as e:
            print(f"Error selling {token}: {e}")

    # Step 3: Get Current WETH Balance
    weth_balance_wei = uniswap.get_token_balance(checksum_weth_address)
    weth_balance = weth_balance_wei / 10**18
    print(f"Total WETH balance after selling: {weth_balance:.6f} WETH")

    # Step 4: Buy Target Tokens Based on Target Compositions
    for token, target_weight in target_compositions.items():
        if target_weight > 0:
            weth_to_spend = weth_balance * target_weight
            
            # Adjust the WETH amount to avoid precision issues (round down to 6 decimal places)
            adjusted_weth_to_spend = math.floor(weth_to_spend * 10**8) / 10**8

            if adjusted_weth_to_spend <= 0:
                continue

            try:
                print(f"Buying {token} with {adjusted_weth_to_spend:.6f} WETH")

                uniswap.make_trade(
                    checksum_weth_address,        # WETH as input token
                    checksum_addresses[token],    # Target token
                    int(adjusted_weth_to_spend * 10**18),  # Convert WETH amount to wei
                    fee=3000                      # Assuming 0.3% fee pool for Uniswap V3
                )

                wait_time = random.randint(15, 30)
                print(f"Waiting {wait_time} seconds before the next call...")
                time.sleep(wait_time)

            except Exception as e:
                print(f"Error buying {token}: {e}")

    # Step 5: Log the Rebalancing Info
    final_weth_balance = uniswap.get_token_balance(checksum_weth_address) / 10**18
    print(f"Final WETH balance: {final_weth_balance:.6f} WETH")

    rebal_info = {
        "account_address": account_address,
        "initial_weth_balance": weth_balance,
        "final_weth_balance": final_weth_balance,
        "purchases": target_compositions,
    }

    # Save rebalancing info to CSV
    rebal_df = pd.DataFrame([rebal_info])
    rebal_df.to_csv('data/live_rebal_results.csv', index=False)
    print("Rebalancing info saved to 'data/live_rebal_results.csv'.")


w3, gateway = network(chain)

account = Account.from_key(PRIVATE_KEY)
w3.eth.default_account = account.address

print(f'connected to: {account} on {chain}')

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
        return render_template('optimism_classifier.html')

    @app.route('/clear-cache', methods=['POST'])
    def clear_cache():
        print('Clearing the cache...')
        cache.clear()
        return jsonify({"status": "Cache cleared successfully"})

    @app.route('/run-model')
    async def run_model(seed=20):
        global today, prices_df, Current_risk_free,TOKEN_CONTRACTS,TOKEN_DECIMALS,params_cache,cache,model_name

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

        # model_name = 'arbitrum_classifier'
        
        normalized_value = 100
        
        params = params_cache.get(f'{model_name} Params')
        network = params['network']

        if network == 'gnosis':
            factory = '0xA818b4F111Ccac7AA31D0BCc0806d64F2E0737D7'
            router = '0x1C232F01118CB8B424793ae03F870aa7D0ac7f77'
            version = 2
        elif network == 'arbitrum':
            factory = '0x1F98431c8aD98523631AE4a59f267346ea31F984'
            router = '0x5E325eDA8064b456f4781070C0738d849c824258'
            version = 3
        elif network == 'ethereum':
            factory = None
            router = None
            version = 3
        elif network == 'optimism':
            factory = '0x1F98431c8aD98523631AE4a59f267346ea31F984'
            router = '0xCb1355ff08Ab38bBCE60111F1bb2B784bE25D7e8'
            version = 3
        else:
            print(f'pass correct network')
        
        print(f'model_name: {model_name} \nnetwork: {network}')

        test_start_date = params_cache.get(f'{model_name} Test Start') # First hour after end of training dataset; ensures no overlap
        filtered_assets = params_cache.get(f'{model_name} Assets')
        classifier_data = params_cache.get(f'{model_name} Classifier')

        days = params['days']
        function = params['function']
        seed = params['seed']

        rebalance_frequency = params['rebalance_frequency']

        test_start_date = str(test_start_date)
        print(f'test_start_date: {test_start_date}')

        diff_tolerance_threshold = 0.05 #percent for diff between new portfolio action and current portfolio

        start_date = str(data_start_date)

        # start_date = dt.datetime.strptime(start_date, '%Y-%m-%d %H:00:00') 
        print(f'sql start date: {start_date}')

        data, prices_df = prices_data_func(
                            network=network, 
                            name=model_name,
                            api_key=flipside_api_key,
                            use_cached_data=False,
                            function=function,
                            start_date=start_date,
                            filtered_assets=filtered_assets
                            )
        
        prices_df = prepare_data_for_simulation(prices_df, start_date, end_date)

        portfolio = classifier_data[['symbol','token_address']]

        TOKEN_CONTRACTS = {
            row['symbol']: row['token_address'] for _, row in portfolio.iterrows()
        }

        TOKEN_DECIMALS = get_token_decimals(TOKEN_CONTRACTS,w3)

        model_balances = get_balance(TOKEN_CONTRACTS,TOKEN_DECIMALS,ACCOUNT_ADDRESS,w3)

        latest_prices = {
            token: float(prices_df[f"{token}_Price"].iloc[-1])
            for token in TOKEN_CONTRACTS.keys()
            if f"{token}_Price" in prices_df.columns
        }

        update_price_data(prices_df)

        model_balances_usd = convert_to_usd(model_balances,latest_prices,TOKEN_CONTRACTS)
        portfolio_balance = sum(model_balances_usd.values())

        comp_dict = {
            f"{token} comp": balance_usd / portfolio_balance
            for token, balance_usd in model_balances_usd.items()
        }

        comp_dict["date"] = formatted_today_utc

        update_historical_data(comp_dict)

        portfolio_dict = {
            "Portfolio Value": portfolio_balance,
            "date": formatted_today_utc
        }

        update_portfolio_data(portfolio_dict)

        hist_comp = historical_data.copy()
        hist_comp.set_index('date', inplace=True)
        hist_comp.index = pd.to_datetime(hist_comp.index)
        print(f'hist comp for env {hist_comp}')

        oracle_prices_copy = oracle_prices.copy().set_index('hour')
        oracle_prices_copy.index = pd.to_datetime(oracle_prices_copy.index)

        uniswap = Uniswap(address=ACCOUNT_ADDRESS, private_key=PRIVATE_KEY, version=version, provider=gateway,router_contract_addr=router,factory_contract_addr=factory)

        def run_sim(model,prices_df, hist_comp, seed, rebalancing_frequency):
            print(f'prices before sim: {prices_df}')

            model = PPO.load(f"AI_Models/{model}")
            
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
        
        hist_comp.index = pd.to_datetime(hist_comp.index)
        
        states_df, rewards_df, actions_df, portfolio_values_df, composition,rebalancing_frequency,sortino_ratios = run_sim(
        model_name,oracle_prices_copy[oracle_prices_copy.index>=hist_comp.index.min()],hist_comp,seed,rebalance_frequency
        )

        update_model_actions(actions_df.to_dict())

        current_time = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:00:00')
        current_time = pd.to_datetime(current_time)

        new_compositions = {
            token: float(model_actions.iloc[-1][f"{token}_weight"]) for token in TOKEN_CONTRACTS
                }
        target_comp_df = pd.DataFrame([new_compositions])

        reshaped_df = target_comp_df.melt(
            var_name="token",        # Name of the new column for tokens
            value_name="composition" # Name of the new column for compositions
        )
                
        latest_comp = hist_comp.iloc[-1]

        latest_comp.index = latest_comp.index.str.replace('comp','')
        latest_comp = latest_comp.to_dict()

        should_rebal, next_rebalance_date = should_rebalance(formatted_today_utc, actions_df, rebalance_frequency)

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

        # portfolio_values_df.set_index('Date', inplace=True)
        # portfolio_return = calculate_cumulative_return(portfolio_values_df, 'Portfolio_Value')

        # prices_df.index = pd.to_datetime(prices_df.index)
        print(f'price index: {prices_df.index}')

        # Assuming 'portfolio_values_df' is your DataFrame with a timezone-aware index
        print(f'portfolio index: {portfolio_values_df.index}')
        # portfolio_values_df.index = portfolio_values_df.index.tz_localize(None)

        print(f'returns in live_backend: {returns}')

        hist_returns = calculate_portfolio_returns(hist_comp, price_returns)
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

        flows_data = model_flows(classifier_data['token_address'].unique(),ACCOUNT_ADDRESS,chain)

        flows_data_df = flipside_api_results(flows_data,flipside_api_key)
        print(f'flows_data_df: {flows_data_df}')
        # Check if the query returned data
        if flows_data_df.empty:
            print("No data returned from the query. Skipping processing.")
        else:
            try:
                # Proceed with data processing
                flows_data_df.set_index('dt', inplace=True)
                flows_data_df.index = pd.to_datetime(flows_data_df.index).strftime('%Y-%m-%d')
                daily_flows = flows_data_df.groupby(
                    [flows_data_df.index, 'symbol', 'transaction_type']
                )[['amount_usd']].sum().reset_index().set_index('dt')
                daily_flows.index = pd.to_datetime(daily_flows.index)

                # Visualization 1
                flows_fig_1 = visualization_pipeline(
                    df=daily_flows,
                    title='flows_data_df_1',
                    chart_type='bar',
                    groupby='symbol',
                    num_col='amount_usd',
                    barmode='relative',
                    show_legend=True,
                    tickprefix=dict(y1='$', y2=None),
                    buffer=1,
                    legend_placement=dict(x=0.1, y=0.8)
                )

                chartBuilder(
                    fig=flows_fig_1,
                    title='Flows by Token',
                    dt_index=True,
                    add_the_date=True,
                    show=False,
                    save=False
                )

                # Visualization 2
                flows_fig_2 = visualization_pipeline(
                    df=daily_flows,
                    title='flows_data_df_1',
                    chart_type='bar',
                    groupby='transaction_type',
                    num_col='amount_usd',
                    barmode='relative',
                    tickprefix=dict(y1='$', y2=None),
                    buffer=1,
                    show_legend=True,
                    text=True,
                    textposition='auto'
                )

                chartBuilder(
                    fig=flows_fig_2,
                    title='Flows by Type',
                    dt_index=True,
                    groupby='True',
                    add_the_date=True,
                    show=False,
                    save=False
                )
            except KeyError as e:
                print(f"Error processing data: {e}")

        model_fig1 = visualization_pipeline(
            df=nom_comp,
            title='model_normalized',
            chart_type = 'line',
            cols_to_plot='All',
            tickprefix=dict(y1='$',y2=None),
            show_legend=True,
            decimals=True,
            sort_list = False,
            tickformat=dict(x='%b %d <br> %y',y1=None,y2=None),
            legend_placement=dict(x=0.05,y=0.8),
            font_family=font_family
        )

        chartBuilder(
            fig = model_fig1,
            show=False,
            save=False,
            title='Normalized Model Performance',
            subtitle=f'{model_name} Portfolio',
        )

        model_fig2 = visualization_pipeline(
            df=plot_historical_data,
            title='model_historical',
            chart_type='bar',
            to_percent=True,
            show_legend=True,
            sort_list = False,
            legend_placement=dict(x=0.1,y=1.3),
            cols_to_plot='All',
            buffer=1,
            ticksuffix=dict(y1='%',y2=None),
            margin=dict(t=150,b=0,l=0,r=0),
            font_family=font_family
        )

        chartBuilder(
            fig=model_fig2,
            title='Portfolio Composition Over Time',
            date_xy=dict(x=0.1,y=1.4),
            show=False,
            save=False
        )

        aum_df = historical_port_values.copy()
        aum_df.set_index('date',inplace=True)
        aum_df.index = pd.to_datetime(aum_df.index)

        print(f'aum_df: {aum_df}')

        model_fig3 = visualization_pipeline(
            df=aum_df,
            title='viz_port_values',
            chart_type='line',
            area=True,
            show_legend=True,
            sort_list = False,
            legend_placement=dict(x=0.1,y=1.3),
            cols_to_plot='All',
            tickprefix=dict(y1='$',y2=None),
            margin=dict(t=150,b=0,l=0,r=0),
            font_family=font_family
        )

        chartBuilder(
            fig=model_fig3,
            title='AUM Over Time',
            date_xy=dict(x=0.1,y=1.4),
            show=False,
            save=False
        )

        latest_comp_data = plot_historical_data.iloc[-1].to_frame('Composition').reset_index()
        latest_comp_data = pd.DataFrame([model_balances_usd]).iloc[0].to_frame('Balance USD').reset_index()

        model_fig4 = visualization_pipeline(
            df=latest_comp_data,
            title='viz_port_values',
            chart_type='pie',
            groupby='index',
            num_col='Balance USD',
            show_legend=False,
            sort_list = False,
            line_width=0,
            legend_placement=dict(x=0.1,y=1.3),
            margin=dict(t=150,b=0,l=0,r=0),
            annotation_prefix='$',
            font_family=font_family,
            annotations=True
        )

        chartBuilder(
            fig=model_fig4,
            title='Latest Composition',
            subtitle=f'{formatted_today_utc}',
            dt_index=False,
            add_the_date=False,
            show=False,
            save=False
        )

        sortino_ratio_ts = sortino_ratios.set_index('Date')
        sortino_ratio_ts.index = pd.to_datetime(sortino_ratio_ts.index)

        model_fig5 = visualization_pipeline(
            df=sortino_ratio_ts,
            title='sortinos',
            chart_type='line',
            cols_to_plot='All',
            show_legend=True,
            sort_list = False,
            legend_placement=dict(x=0.1,y=1.3),
            # annotation_prefix=dict(y1='$',y2=None),
            margin=dict(t=150,b=0,l=0,r=0),
            font_family=font_family
        )

        chartBuilder(
            fig=model_fig5,
            title='Portfolio Sortino Ratios',
            dt_index=True,
            add_the_date=True,
            show=False,
            save=False
        )

        model_fig6 = visualization_pipeline(
            df=reshaped_df,
            groupby='token',
            num_col='composition',
            title='target_comp',
            chart_type='pie',
            show_legend=False,
            sort_list = False,
            line_width=0,
            legend_placement=dict(x=0.1,y=1.3),
            margin=dict(t=150,b=0,l=0,r=0),
            annotation_prefix='$',
            annotations=False,
            font_family=font_family
        )

        chartBuilder(
            fig=model_fig6,
            title='Target Composition',
            subtitle=f'Last Model Action: {actions_df["Date"].iloc[-1]}',
            dt_index=False,
            add_the_date=False,
            show=False,
            save=False
        )

        # flows_fig_1 = visualization_pipeline(
        #     df=daily_flows,
        #     title='flows_data_df_1',
        #     chart_type='bar',
        #     groupby='symbol',
        #     num_col='amount_usd',
        #     barmode='relative',
        #     show_legend=True,
        #     tickprefix=dict(y1='$',y2=None),
        #     buffer=1,
        #     legend_placement=dict(x=0.1,y=0.8)
        # )

        # chartBuilder(
        #     fig=flows_fig_1,
        #     title='Flows by Token',
        #     dt_index=True,
        #     add_the_date=True,
        #     show=False,
        #     save=False
        # )

        # flows_fig_2 = visualization_pipeline(
        #     df=daily_flows,
        #     title='flows_data_df_1',
        #     chart_type='bar',
        #     groupby='transaction_type',
        #     num_col='amount_usd',
        #     barmode='relative',
        #     tickprefix=dict(y1='$',y2=None),
        #     buffer=1,
        #     show_legend=True,
        #     text=True,
        #     textposition='auto'

        # )

        # chartBuilder(
        #     fig=flows_fig_2,
        #     title='Flows by Type',
        #     dt_index=True,
        #     groupby='True',
        #     add_the_date=True,
        #     show=False,
        #     save=False
        # )

        oracle_prices_copy.to_csv('data/oracle_prices.csv')

        results = {
            'start date (UTC)': hist_comp.index.min().strftime('%Y-%m-%d %H:00:00'),
            'last run (UTC)': formatted_today_utc,
            'next rebalance (UTC)':next_rebalance_date.strftime('%Y-%m-%d %H:00:00'),
            'address':ACCOUNT_ADDRESS,
            # 'sUSDE Balance': f"{new_holdings['SUSDE']:,.2f}",
            # 'sDAI Balance': f"{new_holdings['SDAI']:,.2f}",
            # 'cDAI Balance': f"{new_holdings['CDAI']:,.2f}",
            'portfolio balance': f"${portfolio_balance:,.2f}",
            'sortino ratio':sortino_ratio,
            'rebalance frequency (hours)':rebalancing_frequency,
            'chain':chain
        }

        graph_json_1 = json.dumps(model_fig1.return_fig(), cls=PlotlyJSONEncoder)
        graph_json_2 = json.dumps(model_fig2.return_fig(), cls=PlotlyJSONEncoder)
        graph_json_3 = json.dumps(model_fig3.return_fig(), cls=PlotlyJSONEncoder)
        graph_json_4 = json.dumps(model_fig4.return_fig(), cls=PlotlyJSONEncoder)
        graph_json_5 = json.dumps(model_fig5.return_fig(), cls=PlotlyJSONEncoder)
        graph_json_6 = json.dumps(model_fig6.return_fig(), cls=PlotlyJSONEncoder)

        if flows_data_df.empty:
            print("No data returned from the query. Skipping processing.")
            graph_json_7 = None
            graph_json_8 = None
        else:
            graph_json_7 = json.dumps(flows_fig_1.return_fig(), cls=PlotlyJSONEncoder)
            graph_json_8 = json.dumps(flows_fig_2.return_fig(), cls=PlotlyJSONEncoder)

        cached_data = {"results": results, "graph_1": graph_json_1, "graph_2": graph_json_2,"graph_3":graph_json_3,"graph_4":graph_json_4,"graph_5":graph_json_5,'graph_6':graph_json_6, 'graph_7':graph_json_7, 'graph_8':graph_json_8}

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

if __name__ == "__main__":
    logger.info("Starting Flask app...")
    app = create_app()
    print('Starting Flask app...')
    scheduler.start()
    logger.info("Scheduler started.")
    app.run(debug=True, use_reloader=False, port=5011)
    # Since app.run() is blocking, the following line will not execute until the app stops:
    logger.info("Flask app has stopped.")
    print('Flask app has stopped.')


# %%

