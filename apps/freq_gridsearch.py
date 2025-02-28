from train_test import train_script, main, prices_data_func

from diskcache import Cache

from dotenv import load_dotenv

import os

from python_scripts.utils import flipside_api_results, set_random_seed, to_time, clean_prices, calculate_cumulative_return, calculate_cagr, calculate_beta, fetch_and_process_tbill_data,data_cleaning,prepare_data_for_simulation,normalize_asset_returns,calculate_sortino_ratio,pull_data,data_cleaning,calculate_log_returns, to_percent, mvo
from sql_scripts.queries import lst_portfolio_prices,eth_btc_prices,dao_advisor_portfolio, yield_portfolio_prices, token_prices, all_yield_portfolio_prices,arb_stables_portfolio_prices, all_arb_stables_prices
from models.training import train_model
from python_scripts.apis import token_classifier_portfolio

cache = Cache('test_model_cache')

load_dotenv()
flipside_api_key = os.getenv("FLIPSIDE_API_KEY")

def grid_search_rebalance_frequency(network, model_name, function, rebalance_frequencies, train_model_bool=True,
                                    classifier_visualizations=True, days=60, use_cached_data=True, volume_threshold=1,
                                    backtest_period=4380, start_date=None, window=720, normalized_value=100, 
                                    sortino_ratio_filter=False, sortino_threshold=1, top=None, seed=20, 
                                    training_percentage=0.75, dropna=True, ffill=False, ent_coef=0.01, clip_range=0.3):
    """
    Perform grid search over rebalance frequencies to find the best one based on normalized portfolio value.
    """
    results = []

    for freq in rebalance_frequencies:
        print(f"Testing rebalance frequency: {freq} hours")
        print(f'model name: {model_name}')
        
        # Train the model with the current rebalance frequency
        train_script(network=network, model_name=model_name, train_model_bool=train_model_bool, function=function,
                     classifier_visualizations=classifier_visualizations, days=days, use_cached_data=use_cached_data,
                     volume_threshold=volume_threshold, backtest_period=backtest_period, start_date=start_date,
                     window=window, normalized_value=normalized_value, sortino_ratio_filter=sortino_ratio_filter,
                     sortino_threshold=sortino_threshold, top=top, seed=seed, rebalance_frequency=freq,
                     training_percentage=training_percentage, dropna=dropna, ffill=ffill, ent_coef=ent_coef,
                     clip_range=clip_range)
        
        # Retrieve parameters and test data
        params = cache.get(f'{model_name} Params')
        test_start_date = cache.get(f'{model_name} Test Start')
        filtered_assets = cache.get(f'{model_name} Assets')
        
        data, prices_df = prices_data_func(
            network=params['network'],
            name=model_name,
            api_key=flipside_api_key,
            use_cached_data=True,
            function=params['function'],
            start_date=str(test_start_date),
            prices_only=True
        )
        
        prices_df = prices_df[prices_df.index >= test_start_date]
        prices_df = prices_df[filtered_assets]
        prices_df.columns = [f"{col}_Price" for col in prices_df.columns]

        # Run the simulation and get the normalized portfolio value
        results_dict = main(seed=seed, rebalance_frequency=freq, data=data['portfolio'],
                            prices_df=prices_df, model=model_name, normalized_value=normalized_value)
        
        print(f'results_dict: {results_dict}')
        
        norm_value_df = results_dict['Normalized Return']
        final_norm_value = norm_value_df.iloc[-1]["Normalized Return"]
        results.append({"rebalance_frequency": freq, "final_norm_value": final_norm_value})
    
    # Find the best rebalance frequency
    best_result = max(results, key=lambda x: x["final_norm_value"])
    print(f"Best rebalance frequency: {best_result['rebalance_frequency']} hours with final normalized value: {best_result['final_norm_value']}")
    return results, best_result

if __name__ == "__main__":
    # Define parameters
    model_name = 'v0.1'
    function = None
    rebalance_frequencies = [24, 48, 72, 96, 120, 360]  # List of rebalance frequencies in hours
    network = 'arbitrum'
    days = 7
    seed = 20
    normalized_value = 100
    volume_threshold = 0.01
    backtest_period = 4380
    
    # Perform grid search
    results, best_result = grid_search_rebalance_frequency(
        network=network,
        model_name=model_name,
        function=function,
        rebalance_frequencies=rebalance_frequencies,
        train_model_bool=True,
        days=days,
        use_cached_data=True,
        volume_threshold=volume_threshold,
        backtest_period=backtest_period,
        normalized_value=normalized_value,
        seed=seed
    )
    
    # Print all results
    for result in results:
        print(f"Rebalance Frequency: {result['rebalance_frequency']} hours, Final Normalized Value: {result['final_norm_value']}")
    
    print(f"Optimal rebalance frequency: {best_result['rebalance_frequency']} hours")

    # Update cache with the best rebalance frequency
    best_rebalance_frequency = best_result['rebalance_frequency']

    # Retrieve the current parameters from the cache
    train_dict = cache.get(f'{model_name} Params')

    # Update the rebalance_frequency in the train_dict
    if train_dict:
        train_dict['rebalance_frequency'] = best_rebalance_frequency
        
        # Write the updated train_dict back to the cache
        cache.set(f'{model_name} Params', train_dict)
        print(f"Updated cache for {model_name} with best rebalance frequency: {best_rebalance_frequency} hours")
    else:
        print(f"No cache entry found for {model_name} Params.")
