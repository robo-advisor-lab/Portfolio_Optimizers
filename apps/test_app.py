from models.rebalance import Portfolio
from models.training import train_model
from python_scripts.data_processing import data_processing
from python_scripts.utils import normalize_asset_returns, calculate_cumulative_return,calculate_excess_return,to_percent,flipside_api_results,data_cleaning,calculate_log_returns
from stable_baselines3 import PPO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sql_scripts.queries import lst_portfolio_prices,eth_btc_prices,dao_advisor_portfolio, yield_portfolio_prices, all_arb_stables_prices
from python_scripts.apis import token_classifier_portfolio
import plotly.io as pio
import os 
from dotenv import load_dotenv
from chart_builder.scripts.visualization_pipeline import visualization_pipeline
from chart_builder.scripts.utils import main as chartBuilder

from python_scripts.utils import pull_data, mvo

import streamlit as st
import requests

import datetime as dt
from datetime import timedelta

from diskcache import Cache

pio.templates["custom"] = pio.templates["plotly"]
pio.templates["custom"].layout.font.family = "Cardo"

font_family = "Cardo"

# Set the default template
pio.templates.default = "custom"

load_dotenv()
flipside_api_key = os.getenv("FLIPSIDE_API_KEY")

import os

cache = Cache('test_model_cache')

print(f'test_model_cache:')
print(os.path.abspath('test_model_cache'))

print(list(cache))

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

def prices_data_func(network,
                     api_key,use_cached_data,name,days=None,
                     function=None,start_date=None,
                     backtest_period=None):
    
    if start_date is None and backtest_period is None:
        raise KeyError("Provide either a start date or backtest_period")
    
    if backtest_period is None:
        backtest_period = (pd.to_datetime(dt.datetime.now(dt.timezone.utc)) - pd.to_datetime(start_date)).days * 24

    if function is None:

        data = token_classifier_portfolio(
            network=network,
            days=days,
            name=name,
            api_key = api_key,
            use_cached_data=use_cached_data,
            backtest_period=backtest_period,
            prices_only=True
        )

        prices_df = data_cleaning(data['portfolio'])
    else: 
        data = pull_data(function=function,start_date=start_date, path=f'data/{name}.csv', api=not use_cached_data,model_name=name)
        prices_df = data_cleaning(data['portfolio'])
        prices_df = prices_df[prices_df.index >= start_date].dropna()

    prices_df.columns = prices_df.columns.str.replace('_Price','')

    return data, prices_df

def main(model,rebalance_frequency,data,prices_df,normalized_value=100,seed=20):

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
    plt.title(f'Rewards over Time for {model_name}')
    plt.show()

    # Access the stored DataFrames for each model
    dao_advisor_results = results["Hourly Returns"]
    dao_advisor_comp = results["composition"]

    print(f'dao_advisor_comp: {dao_advisor_comp}')
    print(f'dao_advisor_results: {dao_advisor_results}')

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
        show=True,
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
        show=True,
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
        show=True,
        save=False
    )

    print(f'sortino_ratio: {sortino_ratio}')

if __name__ == "__main__":

    #yield_optimizer = LST portfolio

    model_name = 'v0.1'
    use_cached_data = True
    normalized_value = 100
    
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
                    use_cached_data=use_cached_data,
                    function=function,
                    start_date=test_start_date,
                    )
    
    print(f'data: {data}')
    print(f'prices_df: {prices_df}')
    prices_df = prices_df[prices_df.index>=test_start_date]
    prices_df = prices_df[filtered_assets]
    prices_df.columns = [f"{col}_Price" for col in prices_df.columns]
    # data['portfolio'] = data['portfolio'][data['portfolio'].index>=test_start_date]

    main(seed=seed,
         rebalance_frequency=rebalance_frequency,
         data=data['portfolio'],
         prices_df=prices_df,
         model=model_name,
         normalized_value=normalized_value
         )
