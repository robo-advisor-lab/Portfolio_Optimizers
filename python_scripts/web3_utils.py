from eth_account import Account
from eth_account.signers.local import LocalAccount
from web3 import Web3, EthereumTesterProvider
from web3.exceptions import TransactionNotFound,TimeExhausted

from uniswap import Uniswap
import math
import pandas as pd 

import os
from dotenv import load_dotenv

import random
import time

load_dotenv()

GNOSIS_GATEWAY = os.getenv('GNOSIS_GATEWAY')
ARBITRUM_GATEWAY = os.getenv('ARBITRUM_GATEWAY')
OPTIMISM_GATEWAY = os.getenv('OPTIMISM_GATEWAY')
ETHEREUM_GATEWAY = os.getenv('ETHEREUM_GATEWAY')

ARBITRUM_BACKUP = os.getenv('ARBITRUM_BACKUP')
OPTIMISM_BACKUP = os.getenv('OPTIMISM_BACKUP')
ETHEREUM_BACKUP = os.getenv('ETHEREUM_BACKUP')

os.chdir('..')

def get_token_decimals(TOKEN_CONTRACTS,w3):
        
        """Fetch decimals for each token contract using Web3."""
        try:
            # ERC20 ABI for decimals function
            decimals_abi = [
                {
                    "constant": True,
                    "inputs": [],
                    "name": "decimals",
                    "outputs": [{"name": "", "type": "uint8"}],
                    "type": "function"
                }
            ]

            # Dictionary to store decimals for each token
            token_decimals = {}

            for token, address in TOKEN_CONTRACTS.items():
                if address:
                    try:
                        contract = w3.eth.contract(address=Web3.to_checksum_address(address), abi=decimals_abi)
                        decimals = contract.functions.decimals().call()
                        token_decimals[token] = decimals
                    except Exception as e:
                        print(f"Error fetching decimals for {token}: {e}")
                        token_decimals[token] = None
                else:
                    print(f"Contract address for {token} is not set.")
                    token_decimals[token] = None

            return token_decimals

        except Exception as e:
            print(f"Error: {e}")
            return None

def get_balance(TOKEN_CONTRACTS, TOKEN_DECIMALS, ACCOUNT_ADDRESS,w3):
    """Fetch token balances using Web3 with provided decimal adjustments."""
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

        # Fetch balances using the provided decimals
        balances = {}
        for token, address in TOKEN_CONTRACTS.items():
            decimals = TOKEN_DECIMALS.get(token)

            if address and decimals is not None:
                try:
                    contract = w3.eth.contract(address=Web3.to_checksum_address(address), abi=erc20_abi)
                    balance_wei = contract.functions.balanceOf(ACCOUNT_ADDRESS).call()
                    balances[token] = balance_wei / 10**decimals
                except Exception as e:
                    print(f"Error fetching balance for {token}: {e}")
                    balances[token] = None
            else:
                print(f"Skipping {token} due to missing address or decimals.")
                balances[token] = None

        # Print and return balances
        print(f"Balances for account {ACCOUNT_ADDRESS}: {balances}")
        return balances

    except Exception as e:
        print(f"Error fetching balances: {e}")
        return None

def convert_to_usd(balances, prices,TOKEN_CONTRACTS):
    """
    Convert token balances to their USD equivalent using token prices.

    Parameters:
    - balances (dict): Dictionary of token balances.
    - prices (dict): Dictionary of token prices.

    Returns:
    - dict: Dictionary of token balances converted to USD.
    """
    # Convert token keys to upper case for consistency
    print(f'balances: {balances.keys()}')
    print(f'TOKEN_CONTRACTS.keys(): {TOKEN_CONTRACTS.keys()}')

    for token in TOKEN_CONTRACTS.keys():
        if f"{token}" not in prices:
            print(f"Missing price for token: {token}")

    usd_balances = {
        token: balances[token] * prices[f"{token}"]
        for token in TOKEN_CONTRACTS.keys()
        if f"{token}" in prices
    }
    return usd_balances

def network_func(chain):
    if chain == 'gnosis':
        primary_gateway = GNOSIS_GATEWAY  # Replace with your Infura URL
        backup_gateway = 'https://lb.nodies.app/v1/406d8dcc043f4cb3959ed7d6673d311a'  # Your backup gateway
    elif chain == 'arbitrum':
        primary_gateway = ARBITRUM_GATEWAY  # Replace with your Infura URL
        backup_gateway = ARBITRUM_BACKUP
    elif chain == 'optimism':
        primary_gateway = OPTIMISM_GATEWAY  # Replace with your Infura URL
        backup_gateway = OPTIMISM_BACKUP
    elif chain == 'ethereum':
        primary_gateway = ETHEREUM_GATEWAY  # Replace with your Infura URL
        backup_gateway = ETHEREUM_BACKUP

    print(f'Gateway: {primary_gateway}')

    if chain == 'gnosis':
        factory = '0xA818b4F111Ccac7AA31D0BCc0806d64F2E0737D7'
        router = '0x1C232F01118CB8B424793ae03F870aa7D0ac7f77'
        version = 2
    elif chain == 'arbitrum':
        factory = '0x1F98431c8aD98523631AE4a59f267346ea31F984'
        router = '0x5E325eDA8064b456f4781070C0738d849c824258'
        version = 3
    elif chain == 'ethereum':
        factory = None
        router = None
        version = 2
    elif chain == 'optimism':
        factory = '0x1F98431c8aD98523631AE4a59f267346ea31F984'
        router = '0xCb1355ff08Ab38bBCE60111F1bb2B784bE25D7e8'
        version = 2
    else:
        print(f'pass correct network')

    for gateway in [primary_gateway, backup_gateway]:
        w3 = Web3(Web3.HTTPProvider(gateway))
        if w3.is_connected():
            try:
                latest_block = w3.eth.get_block('latest')['number']  # Only try this if connected
                print(f"Connected to {chain} via {gateway}: {latest_block} block")
                return w3, gateway, factory, router, version
            except Exception as e:
                print(f"Connected to {gateway} but failed to fetch latest block. Error: {e}")
        else:
            print(f"Failed to connect to {chain} via {gateway}. Trying next gateway...")

    raise ConnectionError(f"Failed to connect to {chain} network using both primary and backup gateways.")

def get_best_fee_tier(uniswap, token_in, token_out, fee_tiers):
    max_liquidity = 0
    best_fee = None

    for fee in fee_tiers:
        try:
            # Get the pool contract instance
            pool = uniswap.get_pool_instance(token_in, token_out, fee)
            
            # Check if pool exists
            if pool.address == "0x0000000000000000000000000000000000000000":
                print(f"⚠️ Pool not found for Fee Tier {fee/10000:.2%}")
                continue
            
            # Get the state of the pool
            pool_state = uniswap.get_pool_state(pool)
            liquidity = int(pool_state['liquidity'])

            print(f"🔎 Liquidity for Fee Tier {fee/10000:.2%}: {liquidity}")

            # Check if this pool has the highest liquidity
            if liquidity > max_liquidity:
                max_liquidity = liquidity
                best_fee = fee

        except Exception as e:
            print(f"⚠️ Error getting liquidity for Fee Tier {fee/10000:.2%}: {e}")

    return best_fee if best_fee else 3000

def is_sufficient_liquidity(uniswap, token_in, token_out, amount_in, fee, liquidity_buffer=1.10):
    try:
        pool = uniswap.get_pool_instance(token_in, token_out, fee)
        
        if pool.address == "0x0000000000000000000000000000000000000000":
            print(f"⚠️ Pool not found for Fee Tier {fee/10000:.2%}")
            return False

        pool_state = uniswap.get_pool_state(pool)
        liquidity = int(pool_state['liquidity'])
        
        required_liquidity = amount_in * liquidity_buffer
        print(f"🔎 Checking liquidity: Required = {required_liquidity}, Available = {liquidity}")

        return liquidity > required_liquidity
        
    except Exception as e:
        print(f"⚠️ Error checking liquidity for Fee Tier {fee/10000:.2%}: {e}")
        return False

def rebalance_portfolio(
    uniswap, 
    token_contracts, 
    token_decimals, 
    target_compositions, 
    account_address, 
    fee_tiers=[3000, 500, 10000],  # Uniswap Fee Tiers
    slippage_values=[0.01, 0.03, 0.05, 0.10],  # Dynamic slippage: 1%, 3%, 5%, 10%
    liquidity_buffer=1.10,  # 🔥 Ensure at least 10% more liquidity than needed
    MIN_BALANCE_THRESHOLD=1e-06,  # Minimum balance required to trade
    MIN_TRADE_AMOUNT=1e-06,  # Minimum trade amount to execute buy orders
    SAFETY_BUFFER=0.99  # 🔥 Apply 99% buffer to avoid "insufficient balance" errors
):
    """
    Rebalances the portfolio by:
      1. Selling excess tokens for WETH.
      2. Buying target tokens with WETH according to target compositions.

    Args:
        uniswap: Uniswap instance for making trades.
        token_contracts (dict): Token name -> contract address.
        token_decimals (dict): Token name -> decimals.
        target_compositions (dict): Target composition of each token.
        account_address (str): Ethereum account address.
        fee_tiers (list): Uniswap Fee Tiers to consider.
        slippage_values (list): Dynamic slippage tolerance levels.
        liquidity_buffer (float): Extra liquidity buffer for safety.
        MIN_BALANCE_THRESHOLD (float): Minimum balance to consider for trading.
        MIN_TRADE_AMOUNT (float): Minimum trade amount for buy orders.
        SAFETY_BUFFER (float): Safety buffer for avoiding balance errors.
    """

    # ✅ 0. Initial Setup
    WETH_ADDRESS = os.getenv('WETH_ADDRESS')
    checksum_weth_address = Web3.to_checksum_address(WETH_ADDRESS)
    checksum_addresses = {token: Web3.to_checksum_address(address) for token, address in token_contracts.items()}

    # 1️⃣ **Step 1: Sell Excess Tokens for WETH**
    print("\n🔹 Starting rebalance: Selling excess tokens for WETH\n")
    
    for token, address in checksum_addresses.items():
        if address == checksum_weth_address:
            print(f"⚠️ Skipping sell for {token} (WETH is the base token).")
            continue
        
        balance_wei = uniswap.get_token_balance(address)
        balance = balance_wei / 10**token_decimals[token]

        print(f"🔹 {token} balance: {balance:.6f}")

        if balance < MIN_BALANCE_THRESHOLD:
            print(f"⚠️ Skipping {token} due to low balance ({balance:.6f}).")
            continue

        target_balance = target_compositions.get(token, 0.0) * balance
        excess_balance = max(balance - target_balance, 0)

        if excess_balance > 0:
            adjusted_excess_balance = excess_balance * SAFETY_BUFFER  # 🔥 Apply buffer to avoid errors
            amount_to_sell = int(adjusted_excess_balance * 10**token_decimals[token])

            print(f"🔄 Selling {adjusted_excess_balance:.6f} {token} for WETH (Applying 99% buffer)")

            trade_success = False
            
            # ✅ 1. Select the Best Fee Tier with Highest Liquidity
            best_fee = get_best_fee_tier(uniswap, address, checksum_weth_address, fee_tiers)
            if best_fee is None:
                print(f"⚠️ No suitable liquidity found for {token}. Skipping trade.")
                continue

            print(f"✅ Selected Fee Tier for {token}: {best_fee/10000:.2%}")

            # 🔎 2. Pre-Trade Liquidity Check with Buffer
            if not is_sufficient_liquidity(uniswap, address, checksum_weth_address, amount_to_sell, best_fee, liquidity_buffer):
                print(f"⚠️ Insufficient liquidity for {token}. Skipping trade.")
                continue

            # ✅ 3. Execute Trade with Selected Fee and Slippage
            for slippage in slippage_values:
                try:
                    uniswap.make_trade(
                        address, 
                        checksum_weth_address, 
                        amount_to_sell, 
                        fee=best_fee,
                        slippage=slippage
                    )
                    print(f"✅ Trade succeeded for {token} (Fee: {best_fee/10000:.2%}, Slippage: {slippage*100:.2f}%)")
                    trade_success = True
                    break  # Break slippage loop if successful
                except Exception as e:
                    print(f"❌ Trade failed for {token} (Fee: {best_fee/10000:.2%}, Slippage: {slippage*100:.2f}%): {e}")

            if not trade_success:
                print(f"⚠️ Failed to sell {token} after multiple attempts.")

    print(f"\n✅ Sell phase complete.\n")

    # 2️⃣ **Step 2: Buy Target Tokens Using WETH**
    print("\n🔹 Buying target tokens with WETH\n")

    weth_balance_wei = uniswap.get_token_balance(checksum_weth_address)
    weth_balance = weth_balance_wei / 10**18  # Assuming WETH has 18 decimals
    print(f"🔹 WETH balance after selling: {weth_balance:.6f}")

    if weth_balance < MIN_TRADE_AMOUNT:
        print(f"⚠️ WETH balance too low to buy target tokens ({weth_balance:.6f}). Skipping buy phase.")
        return

    for token, target_alloc in target_compositions.items():
        if target_alloc == 0:
            print(f"⚠️ Skipping {token} as target allocation is 0.")
            continue
        
        if token == "WETH":
            print(f"⚠️ Skipping WETH as it's the base token.")
            continue
        
        target_amount_in_weth = target_alloc * weth_balance
        amount_to_buy = int(target_amount_in_weth * 10**18)

        print(f"🔄 Buying {target_alloc:.6f} of {token} using {target_amount_in_weth:.6f} WETH")

        trade_success = False

        # ✅ 1. Select the Best Fee Tier with Highest Liquidity
        best_fee = get_best_fee_tier(uniswap, checksum_weth_address, checksum_addresses[token], fee_tiers)
        if best_fee is None:
            print(f"⚠️ No suitable liquidity found for {token}. Skipping buy.")
            continue

        print(f"✅ Selected Fee Tier for {token}: {best_fee/10000:.2%}")

        # 🔎 2. Pre-Trade Liquidity Check with Buffer
        if not is_sufficient_liquidity(uniswap, checksum_weth_address, checksum_addresses[token], amount_to_buy, best_fee, liquidity_buffer):
            print(f"⚠️ Insufficient liquidity for {token}. Skipping buy.")
            continue

        # ✅ 3. Execute Trade with Selected Fee and Slippage
        for slippage in slippage_values:
            try:
                uniswap.make_trade(
                    checksum_weth_address,
                    checksum_addresses[token],
                    amount_to_buy,
                    fee=best_fee,
                    slippage=slippage
                )
                print(f"✅ Buy succeeded for {token} (Fee: {best_fee/10000:.2%}, Slippage: {slippage*100:.2f}%)")
                trade_success = True
                break  # Break slippage loop if successful
            except Exception as e:
                print(f"❌ Buy failed for {token} (Fee: {best_fee/10000:.2%}, Slippage: {slippage*100:.2f}%): {e}")

        if not trade_success:
            print(f"⚠️ Failed to buy {token} after multiple attempts.")
    
    print("\n✅ Buy phase complete.\n")






