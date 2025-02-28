import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import pandas as pd

from chart_builder.scripts.visualization_pipeline import visualization_pipeline
from chart_builder.scripts.utils import main as chartBuilder

def create_interactive_sml(risk_free_rate, market_risk_premium, betas, returns):
    """
    Creates an interactive Security Market Line (SML) plot with dynamic inputs.
    
    Parameters:
    - risk_free_rate (float): The risk-free rate.
    - market_risk_premium (float): The market risk premium.
    - betas (dict): Dictionary of asset betas with names as keys and beta values as values.
    - returns (dict): Dictionary of actual returns with names as keys and return values as values.
    
    Example:
    betas = {
        'RL': 0.5,
        'MVO': 0.7,
        'Historical': 0.6,
        'Defi': 1.0,
        'Non-Defi': 0.8
    }
    returns = {
        'RL': 0.08,
        'MVO': 0.10,
        'Historical': 0.09,
        'Defi': 0.12,
        'Non-Defi': 0.07
    }
    """
    def generate_shades(base_color, light_factor=1.3, dark_factor=0.7):
        rgb = mcolors.to_rgb(base_color)
        lighter_shade = mcolors.to_hex(tuple(min(1, c * light_factor) for c in rgb))
        darker_shade = mcolors.to_hex(tuple(max(0, c * dark_factor) for c in rgb))
        return lighter_shade, darker_shade

    # List of base colors to assign dynamically
    base_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # Collect all beta values and filter out None
    beta_values = [beta for beta in betas.values() if beta is not None]
    
    # Determine the beta range
    max_beta = max(beta_values) if beta_values else 6
    min_beta = min(beta_values) if min(beta_values) < 0 else 0
    beta_range = np.linspace(min_beta, np.absolute(max_beta) * 1.1, 100)  # Slightly extend the range

    # Calculate expected returns for the SML line
    expected_returns = risk_free_rate + beta_range * market_risk_premium

    # Create the SML line
    sml_line = go.Scatter(
        x=beta_range,
        y=expected_returns*100,
        mode='lines',
        name='SML',
        line=dict(color='black')
    )

    # Plot points for expected and actual returns dynamically
    data = [sml_line]

    for i, (name, beta) in enumerate(betas.items()):
        if beta is not None:
            # Get base color and generate shades
            base_color = base_colors[i % len(base_colors)]
            lighter_shade, darker_shade = generate_shades(base_color)

            # Expected return based on SML (darker shade)
            expected_return = risk_free_rate + beta * market_risk_premium
            data.append(go.Scatter(
                x=[beta],
                y=[expected_return * 100],  # Convert to percentage
                mode='markers',
                marker=dict(size=10, color=darker_shade),
                name=f'{name} Expected ({expected_return:.2%})'
            ))

            # Actual return (lighter shade)
            actual_return = returns.get(name)
            if actual_return is not None:
                data.append(go.Scatter(
                    x=[beta],
                    y=[actual_return * 100],  # Convert to percentage
                    mode='markers',
                    marker=dict(size=10, color=lighter_shade),
                    name=f'{name} Actual ({actual_return:.2%})'
                ))

    # Risk-Free Rate line
    risk_free_line = go.Scatter(
        x=[min(beta_range), max(beta_range)],
        y=[risk_free_rate*100, risk_free_rate*100],
        mode='lines',
        line=dict(dash='dash', color='green'),
        name='Risk-Free Rate'
    )
    
    data.append(risk_free_line)

    # Layout settings
    layout = go.Layout(
        title='Security Market Line',
        xaxis=dict(title='Beta (Systematic Risk)'),
        yaxis=dict(title='CAGR',ticksuffix='%'),
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font = dict(size=18,color='black')
    )

    # Combine all the plots
    fig = go.Figure(data=data, layout=layout)
    return fig

def model_visualizations(nom_comp, plot_historical_data,historical_port_values,sortino_ratios,reshaped_df,actions_df,model_balances_usd,formatted_today_utc, model_name, font_family):
    print(F'model_name: {model_name}')

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
    print(f'latest_comp_data b4: {latest_comp_data}') 
    latest_comp_data = pd.DataFrame([model_balances_usd]).iloc[0].to_frame('Balance USD').reset_index()
    latest_comp_data = latest_comp_data[latest_comp_data['Balance USD'] > 1e-5]

    print(f'latest_comp_data: {latest_comp_data}')

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
        show=True,
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

    # breakpoint()

    chartBuilder(
        fig=model_fig6,
        title='Target Composition',
        subtitle=f'Last Model Action: {actions_df["Date"].iloc[-1]}',
        dt_index=False,
        add_the_date=False,
        show=False,
        save=False
    )

    # if flows_data_df.empty():
    #     print('no flows')
    # else:

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
    return model_fig1, model_fig2, model_fig3, model_fig4, model_fig5, model_fig6

def plot_continuous_return_with_versions(df, title="Portfolio Return Over Time"):
    # Ensure data is sorted by time
    df = df.sort_index()

    # Forward-fill missing values for continuity
    df['Normalized_Return'] = df['Normalized_Return'].ffill()

    # Create figure
    fig = go.Figure()

    # Get unique versions in order of appearance
    unique_versions = df['version'].unique()

    # Identify version change points
    version_change_points = df['version'] != df['version'].shift(1)
    version_change_times = df.index[version_change_points]

    # Loop through each version and plot it as a segment
    for idx, version in enumerate(unique_versions):
        # Select rows for this version
        version_df = df[df['version'] == version].copy()

        # Ensure continuity: Include the last value of the previous version
        if idx > 0:
            prev_version_df = df[df['version'] == unique_versions[idx - 1]]
            if not prev_version_df.empty:
                last_old_value = prev_version_df.iloc[-1]  # Get last row of previous version
                version_df = pd.concat([last_old_value.to_frame().T, version_df])  # Prepend last value

        # Define color for this version
        color = f"hsl({idx * 50}, 70%, 50%)"  # Generate distinct colors

        # Plot version as part of a single continuous line
        fig.add_trace(go.Scatter(
            x=version_df.index,
            y=version_df['Normalized_Return'],
            mode='lines',  # Only lines to keep smooth transitions
            line=dict(color=color, width=4),
            name=version
        ))

    # Add vertical dashed lines at version change points
    for version_time, version in zip(version_change_times, df['version'][version_change_points]):
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=version_time, x1=version_time,
                y0=df['Normalized_Return'].min(), y1=df['Normalized_Return'].max(),
                line=dict(color="black", width=2, dash="dash")  # Dashed vertical line
            )
        )

        # Add version annotation near the top of the plot
        fig.add_annotation(
            x=version_time, 
            y=df['Normalized_Return'].max(),
            text=f"Version {version}",
            showarrow=False,
            font=dict(size=14, color="black"),
            yshift=10
        )

    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Normalized Return",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        legend_title="Model Versions",
        template="plotly_white"
    )

    return fig

