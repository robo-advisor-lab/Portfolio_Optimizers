# %%
import pandas as pd
import requests
import numpy as np
import yfinance as yf
import random
import plotly.io as pio

import datetime as dt
import plotly.graph_objects as go

import os
from dotenv import load_dotenv

import datetime as dt
from datetime import timedelta

from stable_baselines3 import PPO
import re

import streamlit as st

from diskcache import Cache

from python_scripts.plots import plot_continuous_return_with_versions

pio.templates["custom"] = pio.templates["plotly"]
pio.templates["custom"].layout.font.family = "Cardo"

font_family = "Cardo"

# Set the default template
pio.templates.default = "custom"

# %%
def normalize_log_returns(log_returns_df, start_date, end_date, normalize_value=1e4):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the data based on the start date and end date
    filtered_data = log_returns_df[(log_returns_df.index >= start_date) & 
                                   (log_returns_df.index <= end_date)].copy()
    
    if filtered_data.empty:
        print("Filtered data is empty after applying the date filter.")
        return pd.DataFrame()

    # Initialize normalized values
    normalized_values = [normalize_value]
    dates = [filtered_data.index[0]]  # Start date
    versions = [filtered_data['version'].iloc[0]]  # Track the first version

    # Compute normalized returns
    for timestamp, log_return, version in zip(filtered_data.index, filtered_data['Return'], filtered_data['version']):
        normalized_values.append(normalized_values[-1] * np.exp(log_return))
        dates.append(timestamp)
        versions.append(version)  # Track the version used at each timestamp

    # Create DataFrame
    normalized_returns_df = pd.DataFrame({
        'Normalized_Return': normalized_values[1:],  # Exclude initial value
        'version': versions[1:]  # Exclude initial version
    }, index=dates[1:])

    return normalized_returns_df

# %%
def main():

    file_list = os.listdir('E:/Projects/portfolio_optimizers/classifier_optimizer/cache_storage')

    model_names = sorted(set(re.findall(r'v\d{2}', ' '.join(file_list))))

    print(model_names)

    # %%
    model_data = pd.DataFrame()

    for model_name in model_names:
        values = pd.read_csv(f'E:/Projects/portfolio_optimizers/classifier_optimizer/cache_storage/{model_name}_weighted_returns.csv')
        values['version'] = model_name

        model_data = pd.concat([model_data,values])

    model_data


    # %%
    model_data['index'] = pd.to_datetime(model_data['index'])
    model_data.set_index('index',inplace=True)

    # %%
    filled_data = model_data.resample('h').agg({
        "Return":'last',
        "version":'last'
    })

    # %%
    norm_model_returns = normalize_log_returns(filled_data, filled_data.index.min(), filled_data.index.max(),100)

    # %%
    fig = plot_continuous_return_with_versions(norm_model_returns)

    return fig

# %%



