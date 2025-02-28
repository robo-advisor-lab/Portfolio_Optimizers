from models.rebalance import Portfolio
from models.training import train_model
from python_scripts.data_processing import data_processing
from python_scripts.utils import normalize_asset_returns, calculate_cumulative_return,calculate_excess_return,to_percent,flipside_api_results,data_cleaning
from stable_baselines3 import PPO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sql_scripts.queries import lst_portfolio_prices,eth_btc_prices,dao_advisor_portfolio, yield_portfolio_prices

import plotly.io as pio
import os 
from dotenv import load_dotenv
from chart_builder.scripts.visualization_pipeline import visualization_pipeline
from chart_builder.scripts.utils import main as chartBuilder

import datetime as dt
from datetime import timedelta


pio.templates["custom"] = pio.templates["plotly"]
pio.templates["custom"].layout.font.family = "Cardo"

font_family = "Cardo"

# Set the default template
pio.templates.default = "custom"

load_dotenv()
flipside_api_key = os.getenv("FLIPSIDE_API_KEY")

# def pull_data(start_date,function,path,model_name, api=False):
#     print(f'function:{function},start_date:{start_date},path:{path},api:{api},model_name: {model_name}')
#     # Parse dates into datetime format for consistency
#     start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
    
#     # Use formatted date strings as needed in the dao_advisor_portfolio and lst_portfolio_prices functions
#     prices = function(start_date.strftime('%Y-%m-%d %H:%M:%S'))

#     if api:
#         print(f'api True')
#         prices_df = flipside_api_results(prices, flipside_api_key)

#         prices_df.to_csv(path)
#     else:
#         prices_df = pd.read_csv(path)

#     dataset = {
#         f'{model_name}_prices': prices_df
#     }

#     return dataset

# def data_processing(name,path,api,function,start_date,training_percentage=0.75):
#     print(f'function:{function},start_date:{start_date},path:{path},api:{api},model_name: {name},training_percentage:{training_percentage}')
#     data_set = pull_data(function=function,start_date=start_date, path=path, api=api,model_name=name)

#     prices_df = data_cleaning(data_set[f'{name}_prices'], dropna=False,ffill=True)
    
#     max_date = prices_df.index.max()
#     min_date = prices_df.index.min()

#     # Set training percentage (0.75 for 75% training and 25% testing)

#     # Calculate the total hours in the dataset
#     total_hours = (max_date - min_date).total_seconds() / 3600

#     # Calculate hours for training period and the training end date
#     train_hours = total_hours * training_percentage
#     train_end_date = min_date + timedelta(hours=train_hours)

#     print(f"Training period end date: {train_end_date}")
#     print(f"Testing period start date: {train_end_date + timedelta(hours=1)}")

#     train_data = prices_df[prices_df.index <= train_end_date]
#     test_data = prices_df[prices_df.index >= train_end_date]  

#     return test_data, train_data, prices_df

def main(model,path,rebalance_frequency,start_date,
         function=yield_portfolio_prices,training_percentage=0.75,api=False,normalized_value=100,
         seed=20,data_start_date=None):
    print(f'function:{function},start_date:{start_date},path:{path},api:{api},model_name: {model},training_percentage:{training_percentage}')

    test_data, train_data, dao_advisor_prices_df = data_processing(api=api,function=function,
                                                                   start_date=start_date,name=model,
                                                                   training_percentage=training_percentage,path=path,
                                                                   data_start_date=data_start_date)

    models = {
    "dao_advisor_model": PPO.load(f"AI_Models/{model}")
    }

    # Corresponding DataFrames for each environment
    dfs = {
        "dao_advisor_df": test_data
    }

    # Initialize environments for each model
    environments = {
        "dao_advisor_env": Portfolio(dfs["dao_advisor_df"], seed=seed, rebalance_frequency=rebalance_frequency)
    }

    # Function to run simulation for each environment and model
    def run_simulation(model_name, env_name):
        model = models[model_name]
        env = environments[env_name]

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
            portfolio_values.append(env.get_portfolio_value())

            # Update the state
            state = next_state

        # Access the logged data as DataFrames
        states_df = env.get_states_df()
        rewards_df = env.get_rewards_df()
        actions_df = env.get_actions_df()
        portfolio_values_df = env.get_portfolio_values_df()
        composition = env.get_portfolio_composition_df()

        return states_df, rewards_df, actions_df, portfolio_values_df,composition

    results = {}

    # Run simulations for all models and environments
    for model_name, env_name in zip(models.keys(), environments.keys()):
        states_df, rewards_df, actions_df, portfolio_values_df,composition = run_simulation(model_name, env_name)
        print(f"Results for {model_name}:")
        print(states_df.head(), rewards_df.describe(), actions_df.describe(), portfolio_values_df.head())
        # Store results in the dictionary for each model
        results[model_name] = {
            "states_df": states_df,
            "rewards_df": rewards_df,
            "actions_df": actions_df,
            "portfolio_values_df": portfolio_values_df,
            "composition":composition
        }

        # Optionally, you can plot the rewards and portfolio values here for each model
        plt.plot(rewards_df['Date'], rewards_df['Reward'])
        plt.xlabel('Date')
        plt.ylabel('Reward')
        plt.title(f'Rewards over Time for {model_name}')
        plt.show()

        plt.plot(portfolio_values_df['Date'], portfolio_values_df['Portfolio_Value'])
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.title(f'Portfolio Value over Time for {model_name}')
        plt.show()

    # Access the stored DataFrames for each model
    dao_advisor_results = results["dao_advisor_model"]
    dao_advisor_comp = dao_advisor_results["composition"]

    dao_advisor_comp.set_index('Date',inplace=True)
    dao_advisor_comp.columns = dao_advisor_prices_df.columns.str.replace('_Price', ' Comp', regex=False)
    dao_advisor_portfolio_values_df = dao_advisor_results["portfolio_values_df"]
    dao_advisor_portfolio_values_df.set_index('Date', inplace=True)
    dao_advisor_portfolio_return = calculate_cumulative_return(dao_advisor_portfolio_values_df)
    dao_advisor_normalized = normalize_asset_returns(dao_advisor_portfolio_values_df, dao_advisor_portfolio_values_df.index.min(),dao_advisor_portfolio_values_df.index.max(), normalized_value)

    #Prices performance for comparison
    dao_advisor_prices_normalized = normalize_asset_returns(dao_advisor_prices_df, dao_advisor_portfolio_values_df.index.min(),dao_advisor_portfolio_values_df.index.max(), normalized_value)
    dao_advisor_prices_return = calculate_cumulative_return(dao_advisor_prices_normalized)
    dao_advisor_normalized.rename(columns={'Portfolio_Value':'Interest Bearing Portfolio Value'},inplace=True)



    model_fig1 = visualization_pipeline(
        df=dao_advisor_normalized,
        title='model_normalized',
        chart_type = 'line',
        cols_to_plot=['Interest Bearing Portfolio Value'],
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
        subtitle='Interest Bearing Portfolio'
    )

    combined_comparison = pd.merge(
        dao_advisor_normalized,
        dao_advisor_prices_normalized,
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
        title='Normalized Performance Comparison',
        subtitle=None
    )

    dao_advisor_returns_df = calculate_excess_return(dao_advisor_portfolio_return, dao_advisor_prices_return)
    for col in dao_advisor_returns_df:
        print(f'dao advisor excess return over {col}: {dao_advisor_returns_df[col].values[0]}')

    print(f'seed: {seed}')
    print(f'rebalance_frequency:{rebalance_frequency}')
    print(f'{dao_advisor_normalized.index.min().strftime("%d/%m/%Y")} through {dao_advisor_normalized.index.max().strftime("%d/%m/%Y")}')
    print(f'normalized value:{normalized_value}')
    print(f'dao advisor Cumulative Return: {dao_advisor_portfolio_return.values[0][0]*100:.2f}%')
    print(f'average excess return: {dao_advisor_returns_df.mean(axis=1).values[0]}')

    dao_advisor_combined_analysis = pd.merge(
        dao_advisor_comp,
        dao_advisor_prices_df,
        left_index=True,
        right_index=True,
        how='inner'
    )

    dao_advisor_combined_analysis_comp = to_percent(dao_advisor_combined_analysis)

    model_fig3 = visualization_pipeline(
        df=dao_advisor_combined_analysis_comp[['CDAI_Percentage','SDAI_Percentage','SUSDE_Percentage']],
        title='dao_advisor_combined_analysis_comp',
        chart_type='line',
        area=True,
        show_legend=True,
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

if __name__ == "__main__":
    main(seed=20,rebalance_frequency=730,api=False,start_date='2024-09-29 22:00:00',
         model='interest_bearing_model_2',
         path='data/interest_bearing_prices.csv',
         normalized_value=100,
         training_percentage=0.75,
         data_start_date='2024-09-29 22:00:00')
# model training ended 2024-09-16 10:00:00?