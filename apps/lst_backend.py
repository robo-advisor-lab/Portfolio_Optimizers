import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import plotly.graph_objs as go
import pandas as pd
import requests
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import random
import time
import traceback
from web3.exceptions import ContractLogicError
import aiohttp

import math

import datetime as dt
from plotly.subplots import make_subplots
from web3.exceptions import TransactionNotFound,TimeExhausted

from chart_builder.scripts.visualization_pipeline import visualization_pipeline
from chart_builder.scripts.utils import main as chartBuilder

from flipside import Flipside
from plotly.utils import PlotlyJSONEncoder
import json

import os
from dotenv import load_dotenv

import datetime as dt
from datetime import timedelta
import pytz  
import asyncio

from sklearn.linear_model import LinearRegression
import tensorflow as tf
import torch

from python_scripts.utils import set_random_seed, calculate_cumulative_return, calculate_cagr, calculate_beta, fetch_and_process_tbill_data, flipside_api_results, set_global_seed, normalize_asset_returns, prepare_data_for_simulation,data_cleaning
# from python_scripts.data_processing import data_processing
from models.testnet_model import Portfolio
from sql_scripts.queries import lst_portfolio_prices, yield_portfolio_prices

from flask import Flask, render_template, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

import logging
from diskcache import Cache

import plotly.io as pio

from pyngrok import ngrok, conf, installer
import ssl

from eth_account import Account
from eth_account.signers.local import LocalAccount
from web3 import Web3, EthereumTesterProvider

pio.templates["custom"] = pio.templates["plotly"]
pio.templates["custom"].layout.font.family = "Cardo"

font_family = "Cardo"

# Set the default template
pio.templates.default = "custom"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

# Set the path to the ngrok executable installed by Chocolatey
ngrok_path = "C:\\ProgramData\\chocolatey\\bin\\ngrok.exe"

# Update the pyngrok configuration with the ngrok path
pyngrok_config = conf.PyngrokConfig(ngrok_path=ngrok_path)

# Check if ngrok is installed at the specified path, if not, install it using the custom SSL context
if not os.path.exists(pyngrok_config.ngrok_path):
    installer.install_ngrok(pyngrok_config.ngrok_path, context=context)

# Configure ngrok with custom SSL context
conf.set_default(pyngrok_config)
conf.get_default().ssl_context = context

ngrok_token = os.getenv('ngrok_token')

# Set your ngrok auth token
ngrok.set_auth_token(ngrok_token)

# Start ngrok
# public_url = ngrok.connect(5000, pyngrok_config=pyngrok_config, hostname="www.optimizerfinance.com").public_url
# print("ngrok public URL:", public_url)

#to connect account to sepolia
PRIVATE_KEY = os.getenv('LST_PORTFOLIO_KEY')
ACCOUNT_ADDRESS = os.getenv('LST_PORTFOLIO_ADDRESS')
GATEWAY_URL = os.getenv('GATEWAY_URL')
SEPOLIA_GATEWAY = os.getenv('SEPOLIA_GATEWAY')

#DEX App Address
FUND_ACCOUNT_ADDRESS = os.getenv('FUND_ACCOUNT_ADDRESS')

#the addresses for each token
SUSDE_CONTRACT_ADDRESS = os.getenv('SUSDE_CONTRACT_ADDRESS')
CDAI_CONTRACT_ADDRESS = os.getenv('CDAI_CONTRACT_ADDRESS')
SDAI_CONTRACT_ADDRESS = os.getenv('SDAI_CONTRACT_ADDRESS')

AETH_CONTRACT_ADDRESS = os.getenv('AETH_CONTRACT_ADDRESS')
CBETH_CONTRACT_ADDRESS = os.getenv('CBETH_CONTRACT_ADDRESS')
ETHX_CONTRACT_ADDRESS = os.getenv('ETHX_CONTRACT_ADDRESS')
EZETH_CONTRACT_ADDRESS = os.getenv('EZETH_CONTRACT_ADDRESS')
RETH_CONTRACT_ADDRESS = os.getenv('RETH_CONTRACT_ADDRESS')
WSTETH_CONTRACT_ADDRESS = os.getenv('WSTETH_CONTRACT_ADDRESS')
SFRXETH_CONTRACT_ADDRESS = os.getenv('SFRXETH_CONTRACT_ADDRESS')
METH_CONTRACT_ADDRESS = os.getenv('METH_CONTRACT_ADDRESS')
MSOL_CONTRACT_ADDRESS = os.getenv('MSOL_CONTRACT_ADDRESS')
RSETH_CONTRACT_ADDRESS = os.getenv('RSETH_CONTRACT_ADDRESS')
OSTETH_CONTRACT_ADDRESS = os.getenv('OSTETH_CONTRACT_ADDRESS')
LBTC_CONTRACT_ADDRESS = os.getenv('LBTC_CONTRACT_ADDRESS')
WEETH_CONTRACT_ADDRESS = os.getenv('WEETH_CONTRACT_ADDRESS')
SWETH_CONTRACT_ADDRESS = os.getenv('SWETH_CONTRACT_ADDRESS')
LSTETH_CONTRACT_ADDRESS = os.getenv('LSTETH_CONTRACT_ADDRESS')
OETH_CONTRACT_ADDRESS = os.getenv('OETH_CONTRACT_ADDRESS')

TOKEN_CONTRACTS = {
    "aeth": os.getenv('AETH_CONTRACT_ADDRESS'),
    "cbeth": os.getenv('CBETH_CONTRACT_ADDRESS'),
    "ethx": os.getenv('ETHX_CONTRACT_ADDRESS'),
    "ezeth": os.getenv('EZETH_CONTRACT_ADDRESS'),
    "reth": os.getenv('RETH_CONTRACT_ADDRESS'),
    "wsteth": os.getenv('WSTETH_CONTRACT_ADDRESS'),
    "sfrxeth": os.getenv('SFRXETH_CONTRACT_ADDRESS'),
    "meth": os.getenv('METH_CONTRACT_ADDRESS'),
    "rseth": os.getenv('RSETH_CONTRACT_ADDRESS'),
    "oseth": os.getenv('OSTETH_CONTRACT_ADDRESS'),
    "lbtc": os.getenv('LBTC_CONTRACT_ADDRESS'),
    "weeth": os.getenv('WEETH_CONTRACT_ADDRESS'),
    "sweth": os.getenv('SWETH_CONTRACT_ADDRESS'),
    "lseth": os.getenv('LSTETH_CONTRACT_ADDRESS'),
    # "oeth": os.getenv('OETH_CONTRACT_ADDRESS')
}

#the api key to access the dex app
API_KEY = os.getenv("APIKEY")

# ERC20 ABI for the transfer function
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

print(f'sepolia_gateway: {SEPOLIA_GATEWAY}')

def network(chain='sepolia'):
    primary_gateway = GATEWAY_URL  # Replace with your Infura URL
    backup_gateway = SEPOLIA_GATEWAY  # Your backup gateway

    for gateway in [primary_gateway, backup_gateway]:
        w3 = Web3(Web3.HTTPProvider(gateway))
        if w3.is_connected():
            try:
                latest_block = w3.eth.get_block('latest')['number']  # Only try this if connected
                print(f"Connected to {chain} via {gateway}: {latest_block} block")
                return w3
            except Exception as e:
                print(f"Connected to {gateway} but failed to fetch latest block. Error: {e}")
        else:
            print(f"Failed to connect to {chain} via {gateway}. Trying next gateway...")

    raise ConnectionError(f"Failed to connect to {chain} network using both primary and backup gateways.")

chain = 'sepolia'

w3 = network(chain)

account = Account.from_key(PRIVATE_KEY)
w3.eth.default_account = account.address

print(f"Connected account: {account.address}")

cache = Cache('lst_cache_dir')
# historical_data = pd.DataFrame()
# historical_port_values = pd.DataFrame()
# model_actions = pd.DataFrame()
# last_rebalance_time = None

historical_data = cache.get('lst_historical_data', pd.DataFrame())
historical_port_values = cache.get('lst_historical_port_values', pd.DataFrame())
model_actions = cache.get('lst_model_actions', pd.DataFrame())
last_rebalance_time = cache.get('lst_last_rebalance_time', None)

scheduler = BackgroundScheduler()

if last_rebalance_time != None:
    print(f'last rebalance time: {last_rebalance_time}')

def update_historical_data(live_comp):
    global historical_data
    new_data = pd.DataFrame([live_comp])
    historical_data = pd.concat([historical_data, new_data]).reset_index(drop=True)
    historical_data.drop_duplicates(subset='date', keep='last', inplace=True)
    cache.set('historical_data', historical_data)
    print(f"cache:{cache.get('historical_data')}")

def update_portfolio_data(values):
    global historical_port_values
    print(f'values: {values}')
    # new_data = pd.DataFrame([values])
    historical_port_values = pd.concat([historical_port_values, values]).reset_index(drop=True)
    historical_port_values.drop_duplicates(subset='date', keep='last', inplace=True)
    cache.set('historical_port_values', historical_port_values)
    print(f"cache:{cache.get('historical_port_values')}")

def update_model_actions(actions):
    global model_actions
    print(f'model actions before update: {model_actions}')
    new_data = pd.DataFrame(actions)
    print(f'new data: {new_data}')
    model_actions = pd.concat([model_actions, new_data]).reset_index(drop=True)
    model_actions.drop_duplicates(subset='Date', keep='last', inplace=True)
    print(f'model actions after update: {model_actions}')
    cache.set('model_actions', model_actions)

def convert_to_usd(balances, prices):
    """
    Convert token balances to their USD equivalent using token prices.

    Parameters:
    - balances (dict): Dictionary of token balances.
    - prices (dict): Dictionary of token prices.

    Returns:
    - dict: Dictionary of token balances converted to USD.
    """
    # Convert token keys to upper case for consistency
    balances = {token.upper(): balance for token, balance in balances.items()}

    print(f'balances: {balances.keys()}')
    print(f'TOKEN_CONTRACTS.keys(): {TOKEN_CONTRACTS.keys()}')

    for token in TOKEN_CONTRACTS.keys():
        if f"{token}_PRICE" not in prices:
            print(f"Missing price for token: {token}")

    usd_balances = {
        token: balances[token] * prices[f"{token}_PRICE"]
        for token in TOKEN_CONTRACTS.keys()
        if f"{token}_PRICE" in prices
    }
    return usd_balances

def should_rebalance(current_time, actions_df, rebalancing_frequency):
    global last_rebalance_time
    
    print(f'last rebal time: {last_rebalance_time}')
    print(f'current time {current_time}')
    print(f'actions df {actions_df}')
    print(f'rebalancing frequency {rebalancing_frequency}')

    # Ensure current_time is tz-naive and truncate to date and hour
    # current_time = pd.to_datetime(current_time).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    print(f'should rebal current time: {current_time}')

    if rebalancing_frequency == 1:
        # Check if last_rebalance_time is None (first rebalance) or if more than an hour has passed
        if last_rebalance_time is None or (current_time - last_rebalance_time).total_seconds() >= 3600:
            last_rebalance_time = current_time
            cache.set('last_rebalance_time', last_rebalance_time)
            print(f'last_rebalance_time updated to: {last_rebalance_time}')
            return True
        else:
            print("Rebalancing is not required at this time.")
            return False
    else:
        actions_df = actions_df.sort_values(by='Date')
        print(f'actions df: {actions_df}')

        last_rebalance_time_from_actions = pd.to_datetime(actions_df['Date'].iloc[-1])

        # Ensure last_rebalance_time_from_actions is tz-naive and truncate to date and hour
        last_rebalance_time_from_actions = last_rebalance_time_from_actions.replace(tzinfo=None, minute=0, second=0, microsecond=0)
        print(f'should rebal last rebal time: {last_rebalance_time_from_actions}')

        # Calculate hours since last rebalance
        hours_since_last_rebalance = (current_time - last_rebalance_time_from_actions).total_seconds() / 3600
        print(f'hours since last rebal: {hours_since_last_rebalance}')
        
        if hours_since_last_rebalance == 0:
            print('initial rebalance')
            if last_rebalance_time is None or (current_time - last_rebalance_time).total_seconds() >= 3600:
                last_rebalance_time = current_time
                cache.set('last_rebalance_time', last_rebalance_time)
                print(f'last_rebalance_time updated to: {last_rebalance_time}')
                return True
            else:
                print("Rebalancing is not required at this time.")
                return False
        elif hours_since_last_rebalance >= rebalancing_frequency:
            print("Rebalancing required based on frequency.")
            if last_rebalance_time is None or (current_time - last_rebalance_time).total_seconds() >= rebalancing_frequency * 3600:
                last_rebalance_time = current_time
                cache.set('last_rebalance_time', last_rebalance_time)
                print(f'last_rebalance_time updated to: {last_rebalance_time}')
                return True
            else:
                print("Rebalancing is not required at this time.")
                return False
        else:
            print("Rebalancing is not required at this time.")
            return False

def shutdown_scheduler(exception=None):
        if scheduler.running:
            scheduler.shutdown()
            logger.info("Scheduler shut down.") 

def get_contract_address(token):
    """Get contract address based on token name."""
    if token == 'aeth':
        return AETH_CONTRACT_ADDRESS
    elif token == 'cbeth':
        return CBETH_CONTRACT_ADDRESS
    elif token == 'ethx':
        return ETHX_CONTRACT_ADDRESS
    elif token == 'ezeth':
        return EZETH_CONTRACT_ADDRESS
    elif token == 'reth':
        return RETH_CONTRACT_ADDRESS
    elif token == 'wsteth':
        return WSTETH_CONTRACT_ADDRESS
    elif token == 'sfrxeth':
        return SFRXETH_CONTRACT_ADDRESS
    elif token == 'meth':
        return METH_CONTRACT_ADDRESS
    elif token == 'rseth':
        return RSETH_CONTRACT_ADDRESS
    elif token == 'oseth':
        return OSTETH_CONTRACT_ADDRESS
    elif token == 'lbtc':
        return LBTC_CONTRACT_ADDRESS
    elif token == 'weeth':
        return WEETH_CONTRACT_ADDRESS
    elif token == 'sweth':
        return SWETH_CONTRACT_ADDRESS
    elif token == 'lseth':
        return LSTETH_CONTRACT_ADDRESS
    elif token == 'oeth':
        return OETH_CONTRACT_ADDRESS
    else:
        raise ValueError(f"Unknown token: {token}")
    
def get_balance():
    """Fetch token balances using Web3."""
    try:
        # ERC20 ABI for balanceOf function
        erc20_abi = [
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            }
        ]

        # Fetch balances programmatically
        balances = {}
        for token, address in TOKEN_CONTRACTS.items():
            if address:
                contract = w3.eth.contract(address=Web3.to_checksum_address(address), abi=erc20_abi)
                balance_wei = contract.functions.balanceOf(ACCOUNT_ADDRESS).call()
                balances[token] = balance_wei / 10**18
            else:
                print(f"Contract address for {token} is not set.")

        # Print and return balances
        print(f"Balances for account {ACCOUNT_ADDRESS}: {balances}")
        return balances

    except Exception as e:
        print(f"Error fetching balances: {e}")
        return None
    
async def transfer_tokens_from_fund(web3, private_key, token, amount, recipient_address):
    print(f"Starting transfer from fund function...")
    print(f"Token: {token}, Amount: {amount}, Recipient: {recipient_address}")

    recipient_address = web3.to_checksum_address(recipient_address)
    fund_account_address = web3.to_checksum_address(FUND_ACCOUNT_ADDRESS)
    contract_address = web3.to_checksum_address(get_contract_address(token))

    contract = web3.eth.contract(address=contract_address, abi=erc20_abi)
    amount_in_smallest_unit = int(amount * 10**18)

    retry_count = 0
    max_retries = 5
    gas_multiplier = 1.2
    gas_price = max(w3.eth.gas_price, w3.to_wei('20', 'gwei'))
    nonce = w3.eth.get_transaction_count(fund_account_address, 'pending')

    while retry_count < max_retries:
        try:
            txn = contract.functions.transfer(
                Web3.to_checksum_address(recipient_address),
                amount_in_smallest_unit
            ).build_transaction({
                'chainId': w3.eth.chain_id,
                'gas': 200000,
                'gasPrice': int(gas_price),
                'nonce': nonce,
            })

            signed_txn = web3.eth.account.sign_transaction(txn, private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)
            print(f"Transaction sent! Hash: {tx_hash.hex()}")

            receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            if receipt['status'] == 1:
                print(f"Transaction successful! Hash: {tx_hash.hex()}")
                return receipt
            else:
                print(f"Transaction failed on-chain. Hash: {tx_hash.hex()}")
                break
        except Exception as e:
            print(f"Error: {e}. Retrying with a higher gas price.")
            if "replacement transaction underpriced" in str(e):
                gas_price *= gas_multiplier
            elif "already known" in str(e):
                print(f"Transaction already known: {tx_hash.hex()}. Waiting for confirmation.")
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
                return receipt

        retry_count += 1
        await asyncio.sleep(5)

    raise Exception("Failed to complete the transfer after retries.")

async def wait_for_transaction(tx_hash, retries=30, delay=60):
    """Wait for transaction confirmation with detailed debugging."""
    print(f"Waiting for transaction receipt: {tx_hash.hex()}")
    for attempt in range(retries):
        try:
            receipt = w3.eth.get_transaction_receipt(tx_hash)
            print(f"Transaction receipt: {receipt}")
            if receipt and receipt['status'] == 1:
                print(f"Transaction {tx_hash.hex()} confirmed successfully.")
                return receipt
            elif receipt and receipt['status'] == 0:
                raise Exception(f"Transaction {tx_hash.hex()} failed on-chain.")
        except TransactionNotFound:
            print(f"Transaction {tx_hash.hex()} not yet confirmed. Attempt {attempt + 1}/{retries}. Retrying...")
        except TimeExhausted:
            print("Transaction confirmation timed out.")
            break
        await asyncio.sleep(delay)
    raise Exception(f"Transaction {tx_hash.hex()} failed to confirm within {retries} attempts.")
    
async def send_balances_to_fund(web3, contract_abi, private_key, initial_holdings, target_balances, prices, new_compositions):
    print(f"Starting send back balance function...")
    print("Current balances:", initial_holdings)
    print("Target balances:", target_balances)

    processed_tokens = set()
    needs_rebalance = False  # Track if we need to send a rebalance request

    for token, target_balance in target_balances.items():
        if token in processed_tokens:
            print(f"Token {token} already processed. Skipping...")
            continue

        processed_tokens.add(token)

        current_balance = initial_holdings.get(token, 0)
        print(f"Token: {token}, current balance: {current_balance}")
        amount_to_adjust = current_balance - target_balance
        amount_to_adjust = math.floor(amount_to_adjust * 10**6) / 10**6
        print(f"Token: {token}, clipped amount to adjust: {amount_to_adjust}")

        if math.isclose(amount_to_adjust, 0, abs_tol=1e-6):
            print(f"Skipping token {token} with negligible adjustment: {amount_to_adjust}")
            continue

        try:
            if amount_to_adjust > 0:
                # Handle excess tokens individually
                print(f"Sending back {amount_to_adjust} of {token} to the fund.")
                await transfer_tokens(token=token, amount=amount_to_adjust, recipient_address=FUND_ACCOUNT_ADDRESS)
            elif amount_to_adjust < 0:
                # Mark that we need a rebalance request
                needs_rebalance = True
        except Exception as e:
            print(f"Error processing token {token}: {e}")
            traceback.print_exc()

    # If any token requires rebalance, send a single request for all tokens
    if needs_rebalance:
        try:
            await send_rebalance_request(
                web3=web3,
                token=None,  # Not used
                amount_to_send=None,  # Not used
                recipient_address=ACCOUNT_ADDRESS,
                prices=prices,
                initial_holdings=initial_holdings,
                new_compositions=new_compositions,
            )
        except Exception as e:
            print(f"Error during rebalance request: {e}")
            traceback.print_exc()

    print("Completed sending balances to fund.")

def calculate_compositions(balances_usd, total_balance):
    """
    Calculate the composition of each token in the portfolio.

    Parameters:
    - balances_usd (dict): Dictionary of token balances in USD.
    - total_balance (float): Total portfolio balance in USD.

    Returns:
    - dict: Dictionary of token compositions.
    """
    if total_balance == 0:
        print("Warning: Total portfolio balance is zero. Setting all compositions to 0.")
        return {f"{token} comp": 0.0 for token in balances_usd}

    # Calculate composition for each token
    compositions = {f"{token} comp": balance_usd / total_balance for token, balance_usd in balances_usd.items()}
    return compositions

async def send_rebalance_request(web3, token, amount_to_send, recipient_address, prices, initial_holdings, new_compositions):
    """Send rebalance request to the DEX app's endpoint."""
    url = 'http://127.0.0.1:5001/rebalance'

    rebalance_data = {
        'recipient_address': web3.to_checksum_address(recipient_address),
        'prices': prices,
        'initial_holdings': initial_holdings,
        'new_compositions': new_compositions
    }

    print(f"Sending rebalance request to {url} with data: {rebalance_data}")

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
            async with session.post(url, json=rebalance_data) as response:
                if response.status == 200:
                    print(f"Rebalance response: {await response.json()}")
                else:
                    print(f"Error: Received status {response.status} with response: {await response.text()}")
    except asyncio.TimeoutError:
        print(f"Timeout error when sending rebalance request for token: {token}")
        # Optionally retry or escalate
    except aiohttp.ClientError as e:
        print(f"Error sending rebalance request: {e}, data: {rebalance_data}")

async def transfer_tokens(token, amount, recipient_address):
    """Transfer tokens using Web3 with detailed debugging."""
    print(f"Starting transfer: token={token}, amount={amount}, recipient={recipient_address}")
    contract_address = get_contract_address(token)
    amount_wei = int(amount * 10**18)  # Adjust for token decimals

    # ERC20 ABI
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

    contract = w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=erc20_abi)
    nonce = w3.eth.get_transaction_count(account.address, 'pending')  # Handle pending transactions properly

    # Fetch initial gas price and gas limit estimation
    gas_price = max(w3.eth.gas_price, w3.to_wei('20', 'gwei'))
    try:
        estimated_gas = contract.functions.transfer(
            Web3.to_checksum_address(recipient_address), amount_wei
        ).estimate_gas({'from': account.address})
    except Exception as e:
        print(f"Gas estimation failed: {e}")
        return

    # Build the transaction
    tx = contract.functions.transfer(
        Web3.to_checksum_address(recipient_address),
        amount_wei
    ).build_transaction({
        'chainId': w3.eth.chain_id,
        'gas': estimated_gas,
        'gasPrice': gas_price,
        'nonce': nonce,
    })

    # Print transaction details
    print(f"Transaction details: {tx}")

    # Sign and send the transaction
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
    print(f"Signed transaction hash: {signed_tx.hash.hex()}")

    try:
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        print(f"Transaction sent: {tx_hash.hex()}")
    except Exception as e:
        print(f"Error sending transaction: {e}")
        return

    # Wait for transaction confirmation
    try:
        receipt = await wait_for_transaction(tx_hash)
        print(f"Transaction confirmed: {receipt}")
        return receipt
    except Exception as e:
        print(f"Transaction confirmation failed: {e}")
        return

print(f'historical Port vals: {historical_port_values}')
print(f'historical comp: {historical_data}')

def create_app():
    app = Flask(__name__)

    def fetch_and_cache_data():
        with app.app_context():
            logger.info("Scheduled task running.")
            print("Scheduled task running.")
            asyncio.run(run_model())

    scheduler.add_job(
        fetch_and_cache_data, 
        trigger=CronTrigger(minute='0'),  # Ensures it runs at the top of every hour
        id='data_fetch_job',             # A unique identifier for this job
        replace_existing=True            # Ensures the job is replaced if it's already running when you restart the app
    )

    @app.route('/')
    def home():
        return render_template('lst_index.html')

    @app.route('/clear-cache', methods=['POST'])
    def clear_cache():
        print('Clearing the cache...')
        cache.clear()
        return jsonify({"status": "Cache cleared successfully"})
    
    global is_running
    
    is_running = False
    print(f'is_running: {is_running}')

    @app.route('/run-model')
    async def run_model():
        # print(f'is_running: {is_running}')
        # Receive latest composition and update DF; front-fill/back-fill if necessary (only triggered once per 24 hours)
        # data = request.get_json()  # Ensure you're extracting data correctly
        # live_comp = data['initial_holdings']
        # print(f'rebalance initial holdings {initial_holdings}')

        global today, days_left, prices_df, three_month_tbill, current_risk_free,is_running,TOKEN_CONTRACTS
        if is_running:
            print("Model is already running, skipping...")
            return jsonify({"status": "already running"})
        
        is_running = True

        # print(f"run_model started at {dt.datetime.now()} by {request.remote_addr}")
        # print(f"run_model completed at {dt.datetime.now()}")

        try:

            cached_data = cache.get('latest_data')   

            today_utc = dt.datetime.now(dt.timezone.utc) 
            formatted_today_utc = today_utc.strftime('%Y-%m-%d %H:00:00')

            data_version = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H-00-00')
            data_version_comp = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:00:00') 

            if cached_data and 'results' in cached_data and cached_data['results'].get('current date') is not None:
                if cached_data['results']['current date'] == data_version_comp:
                    print(f"cached_data['results']['current date']: {cached_data['results']['current date']}")
                    print(f"data_version_comp: {data_version_comp}")
                    print("Using cached data")
                    return jsonify(cached_data)

            seed=20

            print(f'today: {formatted_today_utc}')

            flipside_api_key = os.getenv("FLIPSIDE_API_KEY")

            print(f'historical_port_values: {historical_port_values}')

            if 'date' in historical_port_values.columns and not historical_port_values['date'].empty:
                start_date = pd.to_datetime(historical_port_values['date'].min()).strftime('%Y-%m-%d %H:00:00')
            else:
                start_date = formatted_today_utc  # Ensure this is in '%Y-%m-%d %H:%M:%S' format

            original_start = start_date 

            # Pass start_date as a datetime object
            start_date = dt.datetime.strptime(start_date, '%Y-%m-%d %H:00:00') 
            # - timedelta(hours=12)
            print(f'histortical port values: {historical_port_values}')
            print(f'sql start date: {start_date}')

            original_balances = get_balance()
            print(f'original_balances: {original_balances}')

            prices_query = lst_portfolio_prices(start_date)
            print(f'prices_query: {prices_query}')

            prices_df = flipside_api_results(prices_query, flipside_api_key)
            prices_df = data_cleaning(prices_df,dropna=False,ffill=True)
            prices_df = prepare_data_for_simulation(prices_df, start_date, formatted_today_utc)
            print(f'prices df: {prices_df}')
            prices_df = prices_df.bfill().ffill()

            prices_df.to_csv('data/latest_live_prices.csv')
            prices_df.set_index('hour', inplace=True)

            # Fetch prices dynamically based on TOKEN_CONTRACTS
            TOKEN_CONTRACTS = {token.upper(): address for token, address in TOKEN_CONTRACTS.items()}

            # Create the prices dictionary with token names in upper case
            prices = {
                f"{token.upper()}_PRICE": float(prices_df[f"{token.upper()}_Price"].iloc[-1])
                for token in TOKEN_CONTRACTS.keys()
                if f"{token.upper()}_Price" in prices_df.columns
            }

            # Fetch original holdings dynamically based on TOKEN_CONTRACTS
            original_holdings = {
                token: float(original_balances[token.lower()])
                for token in TOKEN_CONTRACTS.keys() if token.lower() in original_balances
            }

            print(f'initial prices for USD conversion: {prices}')
            print(f'initial balances used for USD conversion: {original_holdings}')

            # Convert balances to USD
            balances_in_usd = convert_to_usd(original_holdings, prices)
            initial_portfolio_balance = sum(balances_in_usd.values())

            # Calculate compositions dynamically
            print(f'balances_in_usd.items(): {balances_in_usd.items()}')
            
            missing_prices = [token for token in TOKEN_CONTRACTS.keys() if f"{token.upper()}_Price" not in prices_df.columns]
            print(f"Tokens missing price data: {missing_prices}")

            comp_dict = {
                f"{token} comp": balance_usd / initial_portfolio_balance
                for token, balance_usd in balances_in_usd.items()
            }

            comp_dict["date"] = formatted_today_utc

            print(f'Composition dictionary: {comp_dict}')

            end_date = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:00:00') 
            print(f'end date {end_date}')

            print(f'old historical data {historical_data}')
            update_historical_data(comp_dict)
            print(f'updated historical data {historical_data}')

            print(f"Initial portfolio balance in USD: {initial_portfolio_balance}")
            print(f"Initial holdings for rebalancing: {original_holdings}")

            hist_comp = historical_data.copy()
            hist_comp.set_index('date', inplace=True)
            print(f'hist comp for env {hist_comp}')

            hist_comp.to_csv('historical_comp.csv')
            prices_df.to_csv('prices_df.csv')

            def run_sim():
                global prices_df

                print(f'prices before sim: {prices_df}')

                rebalancing_frequency = 1

                model = PPO.load("AI_Models/hourly_lst_model")
                # model = PPO.load("AI_Models/interest_bearing_model_2")
                prices_df = prices_df.ffill().bfill()
                env = Portfolio(prices_df, compositions=hist_comp, seed=seed, rebalance_frequency=rebalancing_frequency,
                                start_date=start_date,end_date=end_date)

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
                    portfolio_values.append(env.get_portfolio_value())
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
                    print(f"Portfolio Value: {env.get_portfolio_value()}")

                states_df = env.get_states_df()
                rewards_df = env.get_rewards_df()
                actions_df = env.get_actions_df()
                portfolio_values_df = env.get_portfolio_values_df()
                composition = env.get_portfolio_composition_df()
            
                return states_df, rewards_df, actions_df, portfolio_values_df, composition, rebalancing_frequency
            
            states_df, rewards_df, actions_df, portfolio_values_df, composition, rebalancing_frequency = run_sim()
            update_model_actions(actions_df.to_dict())

            current_time = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:00:00')
            current_time = pd.to_datetime(current_time)

            print(f'last rebal time before should_rebalance: {last_rebalance_time}')

            if should_rebalance(current_time, model_actions, rebalancing_frequency):
                print(f'last rebal time after rebalance: {last_rebalance_time}')
                print('actions df', actions_df)

                # Get initial balances dynamically
                initial_balances = get_balance()
                initial_holdings = {
                    token.upper(): float(initial_balances[token.upper()]) for token in TOKEN_CONTRACTS
                }

                print(f'initial holdings: {initial_holdings}')

                # Get new compositions dynamically from model actions
                new_compositions = {
                    token.upper(): float(model_actions.iloc[-1][f"{token}_weight"]) for token in TOKEN_CONTRACTS
                }

                print(f'new compositions: {new_compositions}')

                # Calculate total portfolio value
                total_value = sum(
                    initial_holdings[token.upper()] * prices[f"{token}_PRICE"] for token in TOKEN_CONTRACTS
                )

                # Calculate target balances dynamically
                target_balances = {
                    token.upper(): total_value * new_compositions.get(token.upper(), 0) / prices[f"{token}_PRICE"]
                    for token in TOKEN_CONTRACTS
                }

                print(f'total value: {total_value}')
                print(f'target balances: {target_balances}')

                # Create rebalancing info dictionary dynamically
                rebal_info = {
                    "new compositions": new_compositions,
                    "prices": prices,
                    "initial holdings": initial_holdings,
                    "account address": ACCOUNT_ADDRESS,
                    "target balances": target_balances,
                    **{
                        f"{token} bal usd": initial_holdings[token.lower()] * prices[f"{token}_PRICE"]
                        for token in TOKEN_CONTRACTS
                    },
                    "portfolio balance": total_value
                }

                # Save rebalancing info to CSV
                rebal_df = pd.DataFrame([rebal_info])
                rebal_df.to_csv('data/live_rebal_results.csv')

                print(f'send to fund balance {total_value}')
                await send_balances_to_fund(w3, erc20_abi, PRIVATE_KEY, initial_holdings, target_balances, prices, new_compositions)

            else:
                print("Rebalancing is not required at this time.")
        finally:
            # Reset the running flag
            is_running = False

        # Save rewards to CSV
        rewards_df.to_csv('data/live_rewards.csv')

        # Get new balances dynamically
        new_balances = get_balance()
        print(f'initial prices for usd conversion: {prices}')
        print(f'new balances used for usd conversion: {new_balances}')
        new_holdings = {
            token: float(new_balances[token.upper()]) for token in TOKEN_CONTRACTS
        }

        

        # Convert new balances to USD dynamically
        new_balances_usd = {
            token: new_holdings[token.upper()] * prices[f"{token.upper()}_PRICE"]
            for token in TOKEN_CONTRACTS
        }

        # Calculate the new portfolio balance
        new_portfolio_balance = sum(new_balances_usd.values())

        # Create portfolio summary
        portfolio_dict = {
            "Portfolio Value": new_portfolio_balance,
            "date": end_date
        }

        # Save new portfolio to CSV
        new_portfolio = pd.DataFrame([portfolio_dict])

        print(f"cache:{cache.get('historical_port_values')}")
        update_portfolio_data(new_portfolio)
        print(f"cache:{cache.get('historical_port_values')}")

        portfolio_values_df.set_index('Date', inplace=True)
        # portfolio_return = calculate_cumulative_return(portfolio_values_df, 'Portfolio_Value')

        # prices_df.index = pd.to_datetime(prices_df.index)
        print(f'price index: {prices_df.index}')

        # Assuming 'portfolio_values_df' is your DataFrame with a timezone-aware index
        print(f'portfolio index: {portfolio_values_df.index}')
        portfolio_values_df.index = pd.to_datetime(portfolio_values_df.index)
        # portfolio_values_df.index = portfolio_values_df.index.tz_localize(None)

        print(f'portfolio_values_df: {portfolio_values_df}')
        print(f'first portfolio_values_df: {portfolio_values_df.values[0]}')
        viz_port_values = portfolio_values_df.copy()
        # viz_port_values.set_index('Date',inplace=True)
        viz_port_values.index = pd.to_datetime(viz_port_values.index)

        print(F'viz_port_values: {viz_port_values}')

        norm_port = normalize_asset_returns(viz_port_values, viz_port_values.index.min(),viz_port_values.index.max(), 100)
        norm_prices = normalize_asset_returns(prices_df, viz_port_values.index.min(),viz_port_values.index.max(), 100)

        print(f'norm_port: {norm_port}')
        print(f'norm_port: {norm_prices}')
        norm_port.rename(columns={'Portfolio_Value':'Interest Bearing Portfolio Value'},inplace=True)

        nom_comp = pd.merge(
            norm_port, norm_prices, left_index=True, right_index=True, how='left'
        )

        comp_dict_data = calculate_compositions(new_balances_usd, new_portfolio_balance)

        print(F'comp_dict_data: {comp_dict_data}')

        comp_dict = {**comp_dict_data, 'date': formatted_today_utc}

        update_historical_data(comp_dict)

        print(f'historical_data: {historical_data}')
        print(f'norm_port: {norm_port}')

        plot_historical_data = historical_data.copy()
        plot_historical_data.set_index('date',inplace=True)
        plot_historical_data.index = pd.to_datetime(plot_historical_data.index)
        plot_historical_data = plot_historical_data.resample('H').ffill().bfill()
        print(f'plot_historical_data: {plot_historical_data.index}')

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
            subtitle='Interest Bearing Portfolio',
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

        graph_json_1 = json.dumps(model_fig1.return_fig(), cls=PlotlyJSONEncoder)
        graph_json_2 = json.dumps(model_fig2.return_fig(), cls=PlotlyJSONEncoder)
        graph_json_3 = json.dumps(model_fig3.return_fig(), cls=PlotlyJSONEncoder)

        print(f'portfolio_values_df.iloc[-1]: {portfolio_values_df.iloc[-1]}')

        today = dt.datetime.now().strftime('%Y-%m-%d %H:00:00') 
        portfolio_value = float(portfolio_values_df.iloc[-1])

        print(f'portfolio_values_df:{portfolio_values_df}')

        print(f'initial_portfolio_balance:{initial_portfolio_balance}')
        print(f'new_portfolio_balance:{new_portfolio_balance}')

        print(f'norm_port_index: {norm_port.index}')
        print(f'norm_prices_index: {norm_prices.index}')

        results = {
            'start date': original_start,
            'current date': end_date,
            'today': formatted_today_utc,
            'address':ACCOUNT_ADDRESS,
            # 'sUSDE Balance': f"{new_holdings['SUSDE']:,.2f}",
            # 'sDAI Balance': f"{new_holdings['SDAI']:,.2f}",
            # 'cDAI Balance': f"{new_holdings['CDAI']:,.2f}",
            'portfolio balance': f"${new_portfolio_balance:,.2f}",
            'rebalance frequency':rebalancing_frequency,
            'chain':chain
        }

        cached_data = {"results": results, "graph_1": graph_json_1, "graph_2": graph_json_2,"graph_3":graph_json_3}

        cache.set('latest_data', cached_data)

        return jsonify(cached_data)
    
    # with app.app_context():
    #     fetch_and_cache_data()

    @app.route('/cached-data')
    def get_cached_data():
        cached_data = cache.get('latest_data')
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
    app.run(debug=True, use_reloader=False, port=5002)
    # Since app.run() is blocking, the following line will not execute until the app stops:
    logger.info("Flask app has stopped.")
    print('Flask app has stopped.')














        




