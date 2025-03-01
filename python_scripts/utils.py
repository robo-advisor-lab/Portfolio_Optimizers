import pandas as pd
from flipside import Flipside
import random
import numpy as np
import torch 
import tensorflow
from sklearn.linear_model import LinearRegression
import requests
import os
import datetime as dt
from dotenv import load_dotenv
import json
import time
import streamlit as st
from datetime import timedelta
from scipy.optimize import minimize, Bounds, LinearConstraint
# from python_scripts.apis import token_classifier_portfolio

load_dotenv()
flipside_api_key = os.getenv("FLIPSIDE_API_KEY")

# os.chdir('..')

@st.cache_data(ttl=timedelta(hours=1))
def flipside_api_results(query, api_key, attempts=10, delay=30):
    import requests
    import time
    import pandas as pd

    url = "https://api-v2.flipsidecrypto.xyz/json-rpc"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }

    # Step 1: Create the query
    payload = {
        "jsonrpc": "2.0",
        "method": "createQueryRun",
        "params": [
            {
                "resultTTLHours": 1,
                "maxAgeMinutes": 0,
                "sql": query,
                "tags": {"source": "python-script", "env": "production"},
                "dataSource": "snowflake-default",
                "dataProvider": "flipside"
            }
        ],
        "id": 1
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print(f"Query creation failed. Status: {response.status_code}, Response: {response.text}")
        raise Exception("Failed to create query.")

    try:
        response_data = response.json()
    except json.JSONDecodeError as e:
        print(f"Error decoding query creation response: {e}. Response text: {response.text}")
        raise

    query_run_id = response_data.get('result', {}).get('queryRun', {}).get('id')
    if not query_run_id:
        print(f"Query creation response: {response_data}")
        raise KeyError("Failed to retrieve query run ID.")

    # Step 2: Poll for query completion
    for attempt in range(attempts):
        status_payload = {
            "jsonrpc": "2.0",
            "method": "getQueryRunResults",
            "params": [
                {
                    "queryRunId": query_run_id,
                    "format": "json",
                    "page": {"number": 1, "size": 10000}
                }
            ],
            "id": 1
        }

        response = requests.post(url, headers=headers, json=status_payload)
        if response.status_code != 200:
            print(f"Polling error. Status: {response.status_code}, Response: {response.text}")
            time.sleep(delay)
            continue

        try:
            resp_json = response.json()
        except json.JSONDecodeError as e:
            print(f"Error decoding polling response: {e}. Response text: {response.text}")
            time.sleep(delay)
            continue

        if 'result' in resp_json and 'rows' in resp_json['result']:
            all_rows = []
            page_number = 1

            while True:
                status_payload["params"][0]["page"]["number"] = page_number
                response = requests.post(url, headers=headers, json=status_payload)
                resp_json = response.json()

                if 'result' in resp_json and 'rows' in resp_json['result']:
                    rows = resp_json['result']['rows']
                    if not rows:
                        break  # No more rows to fetch
                    all_rows.extend(rows)
                    page_number += 1
                else:
                    break

            return pd.DataFrame(all_rows)

        if 'error' in resp_json and 'not yet completed' in resp_json['error'].get('message', '').lower():
            print(f"Query not completed. Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            print(f"Unexpected polling error: {resp_json}")
            raise Exception(f"Polling error: {resp_json}")

    raise TimeoutError(f"Query did not complete after {attempts} attempts.")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensorflow.random.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def to_time(df):
    time_cols = ['date','dt','hour','time','day','month','year','week','timestamp','date(utc)','block_timestamp']
    for col in df.columns:
        if col.lower() in time_cols and col.lower() != 'timestamp':
            df[col] = pd.to_datetime(df[col])
            df.set_index(col, inplace=True)
        elif col.lower() == 'timestamp':
            df[col] = pd.to_datetime(df[col], unit='ms')
            df.set_index(col, inplace=True)
    print(df.index)
    return df 

def clean_prices(prices_df):
    print('cleaning prices')
    # Pivot the dataframe
    prices_df = prices_df.drop_duplicates(subset=['hour', 'symbol'])
    prices_df_pivot = prices_df.pivot(
        index='hour',
        columns='symbol',
        values='price'
    )
    prices_df_pivot = prices_df_pivot.reset_index()

    # Rename the columns by combining 'symbol' with a suffix
    prices_df_pivot.columns = ['hour'] + [f'{col}_Price' for col in prices_df_pivot.columns[1:]]
    
    print(f'cleaned prices: {prices_df_pivot}')
    return prices_df_pivot

def calculate_cumulative_return(portfolio_values_df):
    """
    Calculate the cumulative return for each column in the portfolio.
    
    Parameters:
    portfolio_values_df (pd.DataFrame): DataFrame with columns representing portfolio values
    
    Returns:
    pd.DataFrame: DataFrame with cumulative returns for each column
    """
    cumulative_returns = {}

    for col in portfolio_values_df.columns:
        initial_value = portfolio_values_df[col].iloc[0]
        final_value = portfolio_values_df[col].iloc[-1]
        cumulative_return = (final_value / initial_value) - 1
        cumulative_returns[col] = cumulative_return

    # Convert the dictionary to a DataFrame
    cumulative_returns_df = pd.DataFrame(cumulative_returns, index=['Cumulative_Return'])
    
    return cumulative_returns_df

def calculate_cagr(history):
    print(f'cagr history: {history}')
    #print(f'cagr history {history}')
    initial_value = history.iloc[0]
    #print(f'cagr initial value {initial_value}')
    final_value = history.iloc[-1]
    #print(f'cagr final value {final_value}')
    number_of_hours = (history.index[-1] - history.index[0]).total_seconds() / 3600
    #print(f'cagr number of hours {number_of_hours}')
    number_of_years = number_of_hours / (365.25 * 24)  # Convert hours to years
    #print(f'cagr number of years {number_of_years}')

    if number_of_years == 0:
        return 0

    cagr = (final_value / initial_value) ** (1 / number_of_years) - 1
    cagr_percentage = cagr * 100
    return cagr

def calculate_cagr_for_all_columns(df):
    """
    Calculate the CAGR for each column in a DataFrame.

    Parameters:
    - df: Pandas DataFrame with datetime index and columns containing historical data.

    Returns:
    - Pandas Series with CAGR for each column.
    """
    cagr_results = {}
    for column in df.columns:
        cagr_results[column] = calculate_cagr(df[column].dropna())
    
    return pd.Series(cagr_results, name="CAGR")

def calculate_beta(data, columnx, columny):
    X = data[f'{columnx}'].pct_change().dropna().values.reshape(-1, 1)  
    Y = data[f'{columny}'].pct_change().dropna().values
  
    # Check if X and Y are not empty
    if X.shape[0] == 0 or Y.shape[0] == 0:
        print("Input arrays X and Y must have at least one sample each.")
        return 0

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, Y)

    # Output the beta
    beta = model.coef_[0]
    return beta

def fetch_and_process_tbill_data(api_url, api_key, data_key, date_column, value_column, date_format='datetime'):
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
    
def set_global_seed(env, seed=20):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    env.seed(seed)
    env.action_space.seed(seed)

def normalize_asset_returns(price_timeseries, start_date, end_date, normalize_value=1e4):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the data based on the start date and end date
    filtered_data = price_timeseries[(price_timeseries.index >= start_date) & (price_timeseries.index <= end_date)].copy()
    
    if filtered_data.empty:
        print("Filtered data is empty after applying the date filter.")
        return pd.DataFrame()

    # Initialize a dictionary to store normalized values for each asset (column)
    normalized_values = {col: [normalize_value] for col in filtered_data.columns}
    dates = [filtered_data.index[0]]  # Use the original start date for labeling

    # Loop through the filtered data and calculate normalized returns for each asset
    for i in range(1, len(filtered_data)):
        for col in filtered_data.columns:
            prev_price = filtered_data[col].iloc[i-1]
            current_price = filtered_data[col].iloc[i]
            
            # Calculate log returns
            price_ratio = current_price / prev_price
            log_return = np.log(price_ratio)

            # Update the normalized value using the exponential of the log return
            normalized_values[col].append(normalized_values[col][-1] * np.exp(log_return))

        dates.append(filtered_data.index[i])

    # Create a DataFrame with normalized values for each asset
    normalized_returns_df = pd.DataFrame(normalized_values, index=dates)

    return normalized_returns_df

# def calculate_log_returns(prices):
#     return np.log(prices / prices.shift(1)).fillna(0)

def prepare_data_for_simulation(price_timeseries, start_date, end_date):
    """
    Ensure price_timeseries has entries for start_date and end_date.
    If not, fill in these dates using the last available data.
    """
    # Ensure 'ds' is in datetime format
    # price_timeseries['hour'] = pd.to_datetime(price_timeseries['hour'])
    
    # Set the index to 'ds' for easier manipulation
    # if price_timeseries.index.name != 'hour':
    #     price_timeseries.set_index('hour', inplace=True)

    print(f'price index: {price_timeseries.index}')

    price_timeseries.index = price_timeseries.index.tz_localize(None)
    
    # Check if start_date and end_date exist in the data
    required_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    all_dates = price_timeseries.index.union(required_dates)
    
    # Reindex the dataframe to ensure all dates from start to end are present
    price_timeseries = price_timeseries.reindex(all_dates)
    
    # Forward fill to handle NaN values if any dates were missing
    price_timeseries.fillna(method='ffill', inplace=True)

    # Reset index if necessary or keep the datetime index based on further needs
    # price_timeseries.reset_index(inplace=True, drop=False)
    # price_timeseries.rename(columns={'index': 'hour'}, inplace=True)
    # price_timeseries.set_index('hour',inplace=True)
    
    return price_timeseries

def pull_data(function,path,model_name, api=False,start_date=None):
    print(f'function:{function},start_date:{start_date},path:{path},api:{api},model_name: {model_name}')

    if api:
        print(f'api True')
        # Parse dates into datetime format for consistency
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        
        # Use formatted date strings as needed in the dao_advisor_portfolio and lst_portfolio_prices functions
        prices = function(start_date.strftime('%Y-%m-%d %H:%M:%S'))
        
        prices_df = flipside_api_results(prices, flipside_api_key)

        prices_df.to_csv(path)
    else:
        print(f'api False')
        prices_df = pd.read_csv(path)

    dataset = {
        f'portfolio': prices_df
    }

    return dataset

def data_cleaning(df,dropna=True,ffill=False):
    clean_df = clean_prices(df)
    clean_df = to_time(clean_df)
    if dropna == True:
        # clean_df = clean_df.dropna(axis=1, how='any')
        clean_df = clean_df.dropna()
    if ffill == True:
        clean_df = clean_df.resample('h').ffill().bfill()

    if '__row_index' in clean_df.columns:
        clean_df.drop(columns=['__row_index'], inplace=True)

    return clean_df

def set_global_seed(env, seed):
    print(f'seed: {seed}')
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    env.seed(seed)
    env.action_space.seed(seed)

def to_percent(df):
    df_copy = df.copy()

    # Identify the asset names based on the column suffix (e.g., 'Comp' and 'Price')
    assets = [col.replace(' Comp', '') for col in df.columns if ' Comp' in col]
    print(assets)

    # Calculate the value (Comp * Price) for each asset
    for asset in assets:
        comp_col = f"{asset} Comp"
        price_col = f"{asset}_Price"
        value_col = f"{asset}_Value"
        df_copy[value_col] = df_copy[comp_col] * df_copy[price_col]

    # Calculate the total portfolio value as the sum of all asset values
    df_copy['Total_Value'] = df_copy[[f"{asset}_Value" for asset in assets]].sum(axis=1)

    # Calculate the percentage composition for each asset
    for asset in assets:
        percentage_col = f"{asset}_Percentage"
        df_copy[percentage_col] = (df_copy[f"{asset}_Value"] / df_copy['Total_Value']) * 100

    # Display the resulting DataFrame with percentage columns
    percentage_cols = [f"{asset}_Percentage" for asset in assets]
    print(f'percentage_cols: {percentage_cols}')
    print(df_copy[percentage_cols])

    return df_copy[percentage_cols]

def calculate_excess_return(portfolio_return_df, asset_returns_df):
    """
    Calculate the excess return of the portfolio over each individual asset.
    
    Parameters:
    portfolio_return_df (pd.DataFrame): DataFrame with portfolio cumulative return (one row, one column)
    asset_returns_df (pd.DataFrame): DataFrame with cumulative returns for individual assets
    
    Returns:
    pd.DataFrame: DataFrame with excess returns for each asset
    """
    # Extract the portfolio cumulative return value
    portfolio_cumulative_return = portfolio_return_df.iloc[0, 0]
    print(f'portfolio_cumulative_return: {portfolio_cumulative_return}')
    print(f'asset_returns_df: {asset_returns_df}')
    
    # Calculate the excess return by subtracting asset cumulative returns from portfolio return
    excess_returns = portfolio_cumulative_return - asset_returns_df
    
    # Return the excess returns as a DataFrame
    return excess_returns
    
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

def calculate_sortino_ratio(df, risk_free, window_size):
    def sortino_ratio(returns, risk_free_rate):
        returns = pd.Series(returns)
        daily_risk_free_rate = (1 + risk_free_rate) ** (1/365) - 1

        excess_returns = returns - daily_risk_free_rate
        downside_returns = np.where(excess_returns < 0, excess_returns**2, 0)
        daily_downside_deviation = np.sqrt(downside_returns.mean())

        if np.isnan(daily_downside_deviation):
            daily_downside_deviation = 0.0

        active_days = returns.notna().sum()
        annual_factor = 365 / active_days if active_days != 0 else 0
        compounding_return = (1 + excess_returns).prod() ** annual_factor - 1 if active_days != 0 else 0
        annual_downside_deviation = daily_downside_deviation * np.sqrt(365)

        sortino = compounding_return / annual_downside_deviation if annual_downside_deviation != 0 else 0.0
        
        if np.isinf(sortino) or sortino > 1000:
            sortino = 0.0
            print("Unusual Sortino ratio detected, setting to 0.0")
            
        return sortino

    # Calculate rolling Sortino ratios for each column
    rolling_sortino = df.rolling(window=window_size,min_periods=1).apply(lambda x: sortino_ratio(x, risk_free))

    # Calculate all-time Sortino ratios for each column
    all_time_sortino = df.apply(lambda col: sortino_ratio(col.dropna(), risk_free))

    return rolling_sortino, all_time_sortino
import re
def calculate_portfolio_returns(historical_data, price_timeseries):
    """
    Calculate the weighted daily return of the portfolio using price timeseries.

    Parameters:
    - historical_data: DataFrame with portfolio compositions for each asset (columns ending with '_comp').
    - price_timeseries: DataFrame with asset prices (columns ending with '_Price').

    Returns:
    - portfolio_returns: Series with weighted portfolio returns.
    """

    # Extract token names from the column names
    tokens_comp = [re.sub(r'\s*Comp', '', col, flags=re.IGNORECASE) for col in historical_data.columns]
    tokens_price = [col.replace("_Price", "") for col in price_timeseries.columns]

    print(f'tokens_price: {tokens_price}')
    print(f'tokens_comp: {tokens_comp}')

    # Find the common tokens between the two DataFrames
    common_tokens = set(tokens_comp) & set(tokens_price)

    print(f'common_tokens: {common_tokens}')

    # Filter and sort the columns to match in the same order
    comp_columns = [f"{token} comp" for token in common_tokens]
    price_columns = [f"{token}_Price" for token in common_tokens]

    print(f'comp_columns: {comp_columns}')
    print(f'price_columns: {price_columns}')

    # Align the data on the common index and selected columns
    common_index = historical_data.index.intersection(price_timeseries.index)
    print(f'common_index: {common_index}')

    compositions = historical_data.loc[common_index, comp_columns]
    prices = price_timeseries.loc[common_index, price_columns]

    print(f'compositions: {compositions}')
    print(f'prices: {prices}')

    # Calculate weighted returns
    weighted_log_returns = (compositions.values * prices.values).sum(axis=1)
    print(f'weighted_log_returns: {weighted_log_returns}')

    # Convert to a Series with the correct index
    portfolio_returns = pd.Series(weighted_log_returns, index=common_index, name='Portfolio_Return')
    print(f'portfolio_returns: {portfolio_returns}')

    return portfolio_returns

# Example usage

def calculate_log_returns(prices):
    """
    Calculate log returns for a DataFrame of prices.

    Parameters:
    - prices: DataFrame of token prices with datetime index.

    Returns:
    - DataFrame of log returns.
    """
    # Replace zeros with NaN to avoid division errors
    prices = prices.replace(0, np.nan)

    # Forward-fill NaN values to maintain continuity
    prices = prices.ffill()

    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1))

    # Fill remaining NaN (e.g., first row) with 0
    log_returns = log_returns.fillna(0)

    return log_returns

def mvo(data, portfolio, risk_free_rate):
    print(f"\n🔍 --- Running MVO --- 🔍")
    print(f"Input portfolio:\n{portfolio}")
    print(f"Input data:\n{data}")

    # Convert the annual risk-free rate to an hourly rate
    hourly_risk_free_rate = (1 + risk_free_rate) ** (1 / 8760) - 1
    print(f"Converted hourly risk-free rate: {hourly_risk_free_rate}\n")

    if len(data) < 2:
        print("⚠️ Insufficient data for MVO. Using last known weights.")
        return portfolio.iloc[-1].values, pd.Series(0, index=data.index, name='Portfolio_Return'), 0

    try:
        # Compute log returns
        price_returns = np.log(data / data.shift(1)).dropna()
        print(f"\n📊 Calculated price_returns:\n{price_returns}")

        if price_returns.empty or (price_returns == 0).all().all():
            print("❌ No valid return data found. Using last known weights.")
            return portfolio.iloc[-1].values, pd.Series(0, index=data.index, name='Portfolio_Return'), 0

        valid_returns_mask = (price_returns != 0).any(axis=1)
        filtered_returns = price_returns.loc[valid_returns_mask]

        # Copy portfolio before filtering
        filtered_portfolio = portfolio.copy()
        filtered_portfolio = filtered_portfolio.reindex(filtered_returns.index).dropna()

        if filtered_portfolio.empty:
            print("❌ No valid portfolio data after filtering. Using last known weights.")
            return portfolio.iloc[-1].values, pd.Series(0, index=data.index, name='Portfolio_Return'), 0

        # Compute portfolio returns
        returns = calculate_portfolio_returns(portfolio, price_returns)
        print(f"\n📊 Computed portfolio_returns:\n{returns}")

        if returns.empty or returns.isnull().all():
            print("❌ No valid returns after calculation. Using last known weights.")
            return portfolio.iloc[-1].values, pd.Series(0, index=data.index, name='Portfolio_Return'), 0

        # Excess returns over risk-free rate
        excess_returns = returns - hourly_risk_free_rate
        excess_returns.fillna(0, inplace=True)
        print(f"\n📊 Excess returns:\n{excess_returns}")

        # Calculate downside deviation
        downside_returns = np.where(excess_returns < 0, excess_returns**2, 0)
        hourly_downside_deviation = np.sqrt(np.mean(downside_returns))

        # Adjust annual factor for hourly data
        active_hours = returns.notna().sum()
        annual_factor = 8760 / active_hours if active_hours != 0 else 0

        # Compounding return adjusted for hourly data
        compounding_return = np.exp(np.log(1 + excess_returns).sum() * annual_factor / active_hours) - 1

        # Annualized downside deviation adjusted for hourly data
        annual_downside_deviation = max(hourly_downside_deviation * np.sqrt(8760), 1e-6)

        # Calculate Sortino ratio
        sortino_ratio = compounding_return / annual_downside_deviation if annual_downside_deviation != 0 else 0
        sortino_ratio = min(sortino_ratio, 100)

        def sortino_ratio_objective(weights):
            # print("\n⚡ Debugging Inside Objective Function ⚡")
            # print("weights.shape:", weights.shape)
            # print("filtered_returns.shape:", filtered_returns.shape)

            portfolio_returns = np.dot(filtered_returns, weights)

            # print("portfolio_returns.shape:", portfolio_returns.shape)  # Should match filtered_returns index length

            excess_portfolio_returns = portfolio_returns - hourly_risk_free_rate
            excess_portfolio_returns = np.clip(excess_portfolio_returns, -1, 1)
            downside_portfolio_returns = np.where(excess_portfolio_returns < 0, excess_portfolio_returns**2, 0)
            
            portfolio_downside_deviation = np.sqrt(np.mean(downside_portfolio_returns))
            portfolio_annual_downside_deviation = max(portfolio_downside_deviation * np.sqrt(8760), 1e-6)

            # annual_portfolio_return = (1 + excess_portfolio_returns).prod() ** (8760 / len(excess_portfolio_returns)) - 1

            annual_portfolio_return = np.exp(np.mean(np.log1p(excess_portfolio_returns))) ** (8760 / len(excess_portfolio_returns)) - 1

            # print("annual_portfolio_return:", annual_portfolio_return)
            # print("portfolio_annual_downside_deviation:", portfolio_annual_downside_deviation)

            if np.isnan(annual_portfolio_return) or np.isnan(portfolio_annual_downside_deviation):
                return np.inf  # Return a high value if NaN is encountered

            return -annual_portfolio_return / portfolio_annual_downside_deviation

        print("\n🔍 Debugging Data Shapes Before Optimization 🔍")
        print("filtered_returns.shape:", filtered_returns.shape)  # Number of rows × columns
        print("filtered_returns.columns:", filtered_returns.columns.tolist())  # Check the assets included

        num_assets = len(portfolio.columns)
        print("Expected num_assets from portfolio:", num_assets)

        epsilon = 1e-8  # Small buffer to avoid boundary issues
        bounds = Bounds([epsilon] * num_assets, [1.0 - epsilon] * num_assets)
        # Weights between 0 and 1

        print(f'fun_constraint: {np.sum(np.ones(num_assets) / num_assets) - 1}')

        # Constraint: Sum of weights should equal 1
        weight_constraint = {
            'type': 'eq', 
            'fun': lambda weights: np.sum(weights) - 1
        }

        # Initial weights
        initial_weights = np.ones(num_assets) / num_assets
        initial_weights /= np.sum(initial_weights)
        print("Sum of initial weights:", np.sum(initial_weights))  # Should be 1

        solvers = ["SLSQP"]
        result = None

        print("\n🔍 Debugging Before Optimizer Call 🔍")
        print("weights.shape (initial):", initial_weights.shape)
        print("weights (initial):", initial_weights)

        # import pdb; pdb.set_trace()

        for solver in solvers:
            try:
                print(f"\n🚀 Running optimizer: {solver}")
                result = minimize(
                    sortino_ratio_objective,
                    initial_weights,
                    method=solver,
                    bounds=bounds,
                    constraints=[weight_constraint],  # Added constraint here
                    options={'maxiter': 5000}
                )

                print(f"\n🔎 Optimizer result object: {result}")

                if result.success:
                    break
            except Exception as e:
                print(f"⚠️ {solver} optimization failed: {e}")

        print(f"\n🔎 Final optimizer result: {result}")

        if result and result.success:
            optimized_weights = result.x
            print(f"\n✅ Optimization successful: {optimized_weights}")
            print(f"Sum of optimized weights: {np.sum(optimized_weights)}")  # Check if sum is 1
            return optimized_weights, returns, sortino_ratio  # Sortino ratio is not recalculated here
        else:
            print("\n⚠️ Optimization failed: Using last known weights")
            return portfolio.iloc[-1].values, returns, sortino_ratio

    except Exception as e:
        print(f"\n❌ Error during MVO computation: {e}")
        return portfolio.iloc[-1].values, pd.Series(0, index=data.index, name='Portfolio_Return'), 0
        
def composition_difference_exceeds_threshold(latest_comp, new_compositions, threshold=0.05):
    """
    Check if the difference between the latest and new compositions exceeds the threshold.

    Parameters:
    - latest_comp: Dict of the current portfolio composition.
    - new_compositions: Dict of the target portfolio composition.
    - threshold: The tolerance threshold (default 5%).

    Returns:
    - True if the difference exceeds the threshold for any token, False otherwise.
    """
    print(f'new_compositions: {new_compositions}')

    # Normalize latest_comp keys by stripping spaces
    latest_comp_normalized = {token.strip(): value for token, value in latest_comp.items()}
    print(f'latest_comp (normalized): {latest_comp_normalized}')

    for token in new_compositions:
        latest_value = latest_comp_normalized.get(token, 0)
        new_value = new_compositions.get(token, 0)
        
        # Calculate the absolute difference
        difference = abs(latest_value - new_value)
        
        print(f"{token}: Latest: {latest_value:.6f}, New: {new_value:.6f}, Difference: {difference:.6f}")
        
        # Check if the difference exceeds the threshold
        if difference > threshold:
            return True

    return False

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

# def calculate_log_returns(prices):
#     return np.log(prices / prices.shift(1)).fillna(0)

def normalize_log_returns(log_returns_timeseries, start_date, end_date, normalize_value=1e4):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the data based on the start date and end date
    filtered_data = log_returns_timeseries[(log_returns_timeseries.index >= start_date) & 
                                           (log_returns_timeseries.index <= end_date)].copy()
    
    if filtered_data.empty:
        print("Filtered data is empty after applying the date filter.")
        return pd.DataFrame()

    # Initialize a dictionary to store normalized values for each asset (column)
    normalized_values = {col: [normalize_value] for col in filtered_data.columns}
    dates = [filtered_data.index[0]]  # Use the original start date for labeling

    # Loop through the filtered data and calculate normalized returns for each asset
    for i in range(len(filtered_data)):
        for col in filtered_data.columns:
            log_return = filtered_data[col].iloc[i]
            normalized_values[col].append(normalized_values[col][-1] * np.exp(log_return))

        dates.append(filtered_data.index[i])

    # Create a DataFrame with normalized values for each asset, excluding the initial date
    normalized_returns_df = pd.DataFrame({col: vals[1:] for col, vals in normalized_values.items()}, index=dates[1:])

    return normalized_returns_df


