import os
import json
import asyncio
import traceback
from flask import Flask, request, jsonify
from eth_account import Account
from web3 import Web3
from web3.exceptions import TransactionNotFound, TimeExhausted
from dotenv import load_dotenv
import pandas as pd
from functools import wraps

# Load environment variables
load_dotenv()

API_KEY = os.getenv("APIKEY")
PRIVATE_KEY = os.getenv('FUND_ACCOUNT_PRIVATE_KEY')
ACCOUNT_ADDRESS = os.getenv('FUND_ACCOUNT_ADDRESS')
GATEWAY_URL = os.getenv('GATEWAY_URL')
SEPOLIA_GATEWAY = os.getenv('SEPOLIA_GATEWAY')

#Compounding Stablecoin Tokens
SUSDE_CONTRACT_ADDRESS = os.getenv('SUSDE_CONTRACT_ADDRESS')
CDAI_CONTRACT_ADDRESS = os.getenv('CDAI_CONTRACT_ADDRESS')
SDAI_CONTRACT_ADDRESS = os.getenv('SDAI_CONTRACT_ADDRESS')

#LST Tokens
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

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Extract API key from the request headers
        api_key = request.headers.get('x-api-key')
        if not api_key or api_key != API_KEY:
            return jsonify({"status": "error", "message": "Unauthorized. Invalid or missing API key."}), 401
        return f(*args, **kwargs)
    return decorated_function

# Connect to Sepolia network
def network(chain='sepolia'):
    primary_gateway = GATEWAY_URL  # Infura URL
    backup_gateway = SEPOLIA_GATEWAY  # Backup gateway

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

w3 = network('sepolia')

# Setup account
account = Account.from_key(PRIVATE_KEY)
w3.eth.default_account = account.address

print(f"Connected account: {account.address}")

def get_contract_address(token):
    """Get contract address based on token name."""
    if token == 'sdai':
        return SDAI_CONTRACT_ADDRESS
    elif token == 'susde':
        return SUSDE_CONTRACT_ADDRESS
    elif token == 'cdai':
        return CDAI_CONTRACT_ADDRESS
    elif token == 'aeth':
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
    elif token == 'msol':
        return MSOL_CONTRACT_ADDRESS
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

# def get_contract_address(token):
#     """Get contract address based on token name."""
#     if token == 'sdai':
#         return SDAI_CONTRACT_ADDRESS
#     elif token == 'susde':
#         return SUSDE_CONTRACT_ADDRESS
#     elif token == 'cdai':
#         return CDAI_CONTRACT_ADDRESS
#     else:
#         raise ValueError(f"Unknown token: {token}")

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

async def rebalance_fund_account(prices, initial_holdings, new_compositions, recipient_address):
    """Rebalance fund account by transferring tokens."""
    print(f"Starting rebalance fund account...")
    print(f"Initial holdings: {initial_holdings}")
    print(f"Prices: {prices}")
    print(f"New compositions: {new_compositions}")

    normalized_prices = {key.upper(): value for key, value in prices.items()}
    print(f"Normalized prices: {normalized_prices}")

    total_value = sum(
        initial_holdings[token] * normalized_prices.get(f"{token.upper()}_PRICE", 0)
        for token in initial_holdings
    )
    print(f"Total portfolio value: {total_value}")

    target_balances = {
        token: (total_value * new_compositions.get(token, 0)) / normalized_prices.get(f"{token.upper()}_PRICE", 1)
        for token in initial_holdings
    }
    print(f"Target balances: {target_balances}")

    differences = {token: target_balances[token] - initial_holdings[token] for token in initial_holdings}
    print(f"Differences: {differences}")

    for token, difference in differences.items():
        print(f"Processing difference for {token}: {difference}")
        if difference > 0:
            print(f"Requesting {difference} of {token} from the fund to {recipient_address}...")
            await transfer_tokens(token, difference, recipient_address)
        elif difference < 0:
            print(f"Skipping adjustment for {token}: {difference} (negative balance).")
        else:
            print(f"No adjustment needed for {token}.")

# Flask App
app = Flask(__name__)

@app.route('/rebalance', methods=['POST'])
async def rebalance():
    """Rebalance endpoint."""
    print(f"Received request: {request.data}")
    try:
        if not request.json:
            return jsonify({"status": "error", "message": "Invalid JSON payload"}), 400

        required_keys = ['recipient_address', 'prices', 'initial_holdings', 'new_compositions']
        for key in required_keys:
            if key not in request.json:
                return jsonify({"status": "error", f"message": "Missing key: {key}"}), 400

        data = request.json
        print(f"Parsed JSON: {data}")

        recipient_address = data['recipient_address']
        prices = data['prices']
        initial_holdings = data['initial_holdings']
        new_compositions = data['new_compositions']

        print(f"Recipient address: {recipient_address}")
        print(f"Prices: {prices}")
        print(f"Initial holdings: {initial_holdings}")
        print(f"New compositions: {new_compositions}")

        # Run the rebalance logic asynchronously
        await rebalance_fund_account(prices, initial_holdings, new_compositions, recipient_address)

        return jsonify({"status": "success"})
    except KeyError as e:
        print(f"Missing key in input data: {e}")
        return jsonify({"status": "error", "message": f"Missing key: {str(e)}"}), 400
    except Exception as e:
        print(f"Error in rebalance: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the DEX App!"})

if __name__ == "__main__":
    app.run(port=5001)
