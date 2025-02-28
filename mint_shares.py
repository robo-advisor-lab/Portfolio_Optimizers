from flask import Flask, render_template, request, jsonify
from web3 import Web3, EthereumTesterProvider
from web3.exceptions import TransactionNotFound,TimeExhausted
import httpx

import asyncio
import datetime as dt
import pickle
import pandas as pd
import requests

import os
from dotenv import load_dotenv
from eth_account import Account
import json
from diskcache import Cache


import datetime as dt
from datetime import timedelta
from python_scripts.web3_utils import get_token_decimals,get_balance,convert_to_usd, network_func, rebalance_portfolio

load_dotenv()

ACCOUNT_ADDRESS = os.getenv('ACCOUNT_ADDRESS')
PRIVATE_KEY = os.getenv('ACCOUNT_KEY')
WETH_ADDRESS = os.getenv('WETH_ADDRESS')
VAULT_ADDRESS = os.getenv('VAULT_ADDRESS')
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')
ETHERSCAN_KEY = os.getenv('ETHERSCAN_KEY')
ARBISCAN_KEY = os.getenv('ARBISCAN_KEY')

with open(r'E:\Projects\portfolio_optimizers\classifier_optimizer\ABIS\erc20_abi.json', "r") as file:
    ERC20_ABI = json.load(file)  # Use name as key
    
print(ERC20_ABI)

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

def approve_token(token_address, spender, amount, w3, account_address, private_key):
    token_contract = w3.eth.contract(address=token_address, abi=ERC20_ABI)  # Provide ERC20 ABI
    nonce = w3.eth.get_transaction_count(account_address)

    approve_txn = token_contract.functions.approve(spender, amount).build_transaction({
        "chainId": 42161,
        "gasPrice": w3.to_wei("1", "gwei"),
        "nonce": nonce
    })

    try:
        gas_estimate = w3.eth.estimate_gas({
            "to": token_address,
            "from": account_address,
            "data": approve_txn["data"],
        })
        print(f"Gas estimate: {gas_estimate}")
        approve_txn["gas"] = gas_estimate
    except Exception as e:
        print(f"Gas estimation failed: {e}")
        return None

    signed_txn = w3.eth.account.sign_transaction(approve_txn, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
    print(f"Approval Transaction Sent: {tx_hash.hex()}")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"Approval Transaction Mined: {receipt.transactionHash.hex()}")

def get_withdrawal_logs_by_topic(contract_address, start_block, end_block, topic0, etherscan_api_key):
    """
    Fetch logs for a contract address filtered by event topics using Arbiscan API.

    Parameters:
        contract_address (str): The contract address to fetch logs for.
        start_block (int): The starting block number.
        end_block (int): The ending block number.
        topic0 (str): The event signature hash (topic0).
        etherscan_api_key (str): Your Arbiscan API key.

    Returns:
        list: A list of logs fetched from the API.
    """
    base_url = "https://api.arbiscan.io/api"  # Change for different networks
    logs = []
    page = 1
    offset = 1000  # Max logs per page

    while True:
        url = (
            f"{base_url}?module=logs&action=getLogs"
            f"&fromBlock={start_block}"
            f"&toBlock={end_block}"
            f"&address={contract_address}"
            f"&topic0={topic0}"
            f"&page={page}"
            f"&offset={offset}"
            f"&apikey={etherscan_api_key}"
        )

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if data["status"] != "1":  # Arbiscan returns "1" for success
                print(f"No more logs or error: {data.get('message', 'Unknown error')}")
                break

            logs.extend(data["result"])
            print(f"Fetched {len(data['result'])} logs from page {page}.")

            if len(data["result"]) < offset:
                break  # Stop if fewer logs than `offset` are returned

            page += 1  # Move to the next page

        except requests.RequestException as e:
            print(f"Error fetching logs: {e}")
            break

    return logs

# def get_withdrawal_logs(contract_address, start_block, end_block, etherscan_api_key):
#     """
#     Fetch logs for a contract address using Arbiscan API.

#     Parameters:
#         contract_address (str): The contract address to fetch logs for.
#         start_block (int): The starting block number.
#         end_block (int): The ending block number.
#         etherscan_api_key (str): Your Arbiscan API key.

#     Returns:
#         list: A list of logs fetched from the API.
#     """
#     base_url = "https://api.arbiscan.io/api"  # Change to appropriate network if needed
#     logs = []
#     page = 1
#     offset = 1000  # Max logs per page

#     while True:
#         url = (
#             f"{base_url}?module=logs&action=getLogs"
#             f"&address={contract_address}"
#             f"&fromBlock={start_block}"
#             f"&toBlock={end_block}"
#             f"&page={page}"
#             f"&offset={offset}"
#             f"&apikey={etherscan_api_key}"
#         )

#         try:
#             response = requests.get(url)
#             response.raise_for_status()
#             data = response.json()

#             if data["status"] != "1":  # Arbiscan returns "1" for success
#                 print(f"No more logs or error: {data.get('message', 'Unknown error')}")
#                 break

#             logs.extend(data["result"])
#             print(f"Fetched {len(data['result'])} logs from page {page}.")

#             if len(data["result"]) < offset:
#                 break  # Stop if fewer logs than `offset` are returned

#             page += 1  # Move to the next page

#         except requests.RequestException as e:
#             print(f"Error fetching logs: {e}")
#             break

#     return logs

def parse_withdrawal_log(log):
    """
    Parse a `WithdrawalRequest` log entry.

    Parameters:
        log (dict): A raw log entry from Etherscan.

    Returns:
        dict: Parsed log data.
    """
    tx_hash = log["transactionHash"]

    # Extract requester's address from topic[1]
    requester = "0x" + log["topics"][1][-40:]  # Address is right-padded

    # Decode the `amount` field from log data (strip 0x)
    data = log["data"][2:]
    amount = int(data[0:64], 16) / 1e18  # Convert from Wei to ETH

    return {
        "transaction_hash": tx_hash,
        "requester": requester,
        "withdraw_amount_eth": amount,
    }


def main():

    w3, gateway, factory, router, version = network_func('arbitrum')

    account = Account.from_key(PRIVATE_KEY)
    w3.eth.default_account = account.address

    abi_path = r'E:\Projects\portfolio_optimizers\classifier_optimizer\Vault-Contracts\artifacts'
    abi_paths = []  # Assuming GAS_ACCOUNTANT_ABI_PATH is predefined

    for file in os.listdir(abi_path):
        if file.endswith('.json') and "metadata" in file:  # Exclude metadata files
            abi_paths.append(os.path.join(abi_path, file))  # Add full path

    print(abi_paths)  # Debug: Check the final list

    abis = {}

    for path in abi_paths:
        filename = os.path.basename(path)  # Extract filename (e.g., "YieldVault.json")
        name = os.path.splitext(filename)[0]  # Remove .json extension (e.g., "YieldVault")

        with open(path, "r") as file:
            metadata = json.load(file)  # Load the full metadata JSON
            abis[name] = metadata["output"]["abi"]  # Extract only the ABI list

    print(abis)  # Debug output

    # breakpoint()

    token0_contract = w3.eth.contract(address=WETH_ADDRESS, abi=ERC20_ABI)

    weth_allowance = token0_contract.functions.allowance(ACCOUNT_ADDRESS, VAULT_ADDRESS).call()

    if (weth_allowance < int(MAX_UINT256*0.8)):

        approve_token(
            token_address=WETH_ADDRESS,
            spender=VAULT_ADDRESS,  # Address of the Uniswap Position Manager
            amount=MAX_UINT256,  # Use a very high value to avoid re-approvals (e.g., `2**256 - 1`)
            w3=w3,
            account_address=ACCOUNT_ADDRESS,
            private_key=PRIVATE_KEY
        )

    print(f'abi keys: {abis.keys()}')

    print(abis["WETHVault_metadata"])

    base_cache_dir = r'E:\Projects\portfolio_optimizers\classifier_optimizer'

    global_classifier_cache = Cache(os.path.join(base_cache_dir, 'global_classifier_cache'))
    model_name = global_classifier_cache.get('current_model_name')

    cache = Cache(os.path.join(base_cache_dir, f'live_{model_name}_cache'))

    historical_port_values = cache.get(f'{model_name} historical_port_values')

    latest_usd_val = historical_port_values.iloc[-1].values[0]

    eth_usd = get_token_price()

    weth_port_val = latest_usd_val / eth_usd

    print(f'weth_port_val: {weth_port_val}')

    print(f'historical_port_values: {latest_usd_val}')
    
    # breakpoint()

    mint_shares_contract = w3.eth.contract(address=VAULT_ADDRESS, abi=abis['WETHVault_metadata'])

    whitelist_bool = mint_shares_contract.functions.whitelist(ACCOUNT_ADDRESS).call()

    print(f"{ACCOUNT_ADDRESS} {'is' if whitelist_bool else 'is not'} whitelisted")

    if not whitelist_bool:
        print(F'whitelisting {ACCOUNT_ADDRESS}')

        transaction = mint_shares_contract.functions.addToWhitelist(ACCOUNT_ADDRESS).build_transaction({
            "from": ACCOUNT_ADDRESS,
            "nonce": w3.eth.get_transaction_count(ACCOUNT_ADDRESS),
            "gasPrice": w3.to_wei("5", "gwei"),
        })

        # Estimate gas before sending
        gas_estimate = w3.eth.estimate_gas(transaction)
        print(f"Estimated Gas: {gas_estimate}")

        # Update transaction with correct gas estimate
        transaction["gas"] = gas_estimate

        # Sign transaction
        signed_tx = w3.eth.account.sign_transaction(transaction, PRIVATE_KEY)

        # Send transaction
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)

        # Wait for transaction confirmation
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"✅ Portfolio value updated! Tx Hash: {receipt.transactionHash.hex()}")

    # Convert portfolio value to 18 decimal precision
    new_portfolio_value = int(weth_port_val * 1e18)

    # Build the transaction (without setting gas limit)
    transaction = mint_shares_contract.functions.setPortfolioValue(new_portfolio_value).build_transaction({
        "from": ACCOUNT_ADDRESS,
        "nonce": w3.eth.get_transaction_count(ACCOUNT_ADDRESS),
        "gasPrice": w3.to_wei("5", "gwei"),
    })

    # Estimate gas before sending
    gas_estimate = w3.eth.estimate_gas(transaction)
    print(f"Estimated Gas: {gas_estimate}")

    # Update transaction with correct gas estimate
    transaction["gas"] = gas_estimate

    # Sign transaction
    signed_tx = w3.eth.account.sign_transaction(transaction, PRIVATE_KEY)

    # Send transaction
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)

    # Wait for transaction confirmation
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"✅ Portfolio value updated! Tx Hash: {receipt.transactionHash.hex()}")

    symbol = mint_shares_contract.functions.symbol().call()
    print(f"Vault Symbol: {symbol}")

    deposit_amount = token0_contract.functions.balanceOf(ACCOUNT_ADDRESS).call()

    deposit_amount = int(deposit_amount * 0.05)

    # Step 2: Estimate Gas for Deposit
    gas_estimate = mint_shares_contract.functions.deposit(deposit_amount).estimate_gas({
        "from": ACCOUNT_ADDRESS,
    })
    print(f"Estimated gas for deposit: {gas_estimate}")

    # Step 3: Execute Deposit Transaction
    deposit_txn = mint_shares_contract.functions.deposit(deposit_amount).build_transaction({
        "from": ACCOUNT_ADDRESS,
        "nonce": w3.eth.get_transaction_count(ACCOUNT_ADDRESS),
        "gas": gas_estimate,
        "gasPrice": w3.to_wei("5", "gwei"),
    })

    # Sign and send deposit transaction
    signed_deposit_txn = w3.eth.account.sign_transaction(deposit_txn, PRIVATE_KEY)
    deposit_tx_hash = w3.eth.send_raw_transaction(signed_deposit_txn.rawTransaction)
    receipt = w3.eth.wait_for_transaction_receipt(deposit_tx_hash)

    print(f"✅ Deposit successful! Transaction Hash: {deposit_tx_hash.hex()}")

    start_block = w3.eth.block_number - 10000000
    end_block = w3.eth.block_number

    withdrawal_event_signature = "WithdrawalRequest(address,uint256)"
    withdrawal_topic0 = Web3.keccak(text=withdrawal_event_signature).hex()
    print(f"Event topic0: {withdrawal_topic0}")

    # Fetch logs
    logs = get_withdrawal_logs_by_topic(VAULT_ADDRESS, start_block, end_block, withdrawal_topic0, ARBISCAN_KEY)

    # Decode logs
    decoded_logs = [parse_withdrawal_log(log) for log in logs]

    # Print results
    for log in decoded_logs:
        print(log)

    total_supply = mint_shares_contract.functions.totalSupply().call()
    print(f"Total Supply Normalized: {total_supply / 1e18}")

    portfolio_value = mint_shares_contract.functions.portfolioValue().call()
    print(f"Portfolio Value: {portfolio_value / 1e18}")

    price_per_token = (portfolio_value / 1e18) / (total_supply / 1e18)

    print(f'price per share in eth via python: {price_per_token}')
    print(f'price per share in usd via python: {price_per_token*eth_usd}')


    price_per_share_raw = mint_shares_contract.functions.pricePerShare().call()
    price_per_share_eth = price_per_share_raw / 1e18
    price_per_share_us = price_per_share_eth * eth_usd
    print(f"Price per share via smart contract: {price_per_share_eth} ETH")
    print(f"Price per share: {price_per_share_us} USD")


if __name__ == "__main__":
    MAX_UINT256 = 2**256 - 1
    main()