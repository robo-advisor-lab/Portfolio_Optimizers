import datetime as dt

def lst_portfolio_prices(today):

  beginning = f"'{today}'"
  print('beginning', beginning)
  
  prices_query =f"""

WITH addresses AS (
    SELECT column1 AS token_address 
    FROM (VALUES
        (LOWER('0xae78736cd615f374d3085123a210448e74fc6393')),--reth
        (LOWER('0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0')),--wsteth
        (LOWER('0xac3e018457b222d93114458476f3e3416abbe38f')),--sfrxeth
        (LOWER('0xd5F7838F5C461fefF7FE49ea5ebaF7728bB0ADfa')),--meth,
        (LOWER('0xbf5495Efe5DB9ce00f80364C8B423567e58d2110')),--ezeth,
        (LOWER('0xA1290d69c65A6Fe4DF752f95823fae25cB99e5A7')),--rseth,
        (LOWER('0xBe9895146f7AF43049ca1c1AE358B0541Ea49704')),--cbeth,
        (LOWER('0xf1C9acDc66974dFB6dEcB12aA385b9cD01190E38')),--oseth,
        (LOWER('0x8236a87084f8B84306f72007F36F2618A5634494')),--lbtc
        (LOWER('0xCd5fE23C85820F7B72D0926FC9b05b43E359b7ee')),--weeth
        (LOWER('0xf951e335afb289353dc249e82926178eac7ded78')),--sweth
        (LOWER('0xa35b1b31ce002fbf2058d22f30f95d405200a15b')),--ethx
        (LOWER('0x8c1bed5b9a0928467c9b1341da1d7bd5e10b6549')),--LSETH
        (LOWER('0xe95a203b1a91a908f9b9ce46459d101078c2c3cb'))--ANKRETH 



    ) AS tokens(column1)
)

select hour,
       symbol,
       price
from ethereum.price.ez_prices_hourly
where token_address in (select token_address from addresses)
and hour >= date_trunc('hour', to_timestamp({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
order by hour desc, symbol 


"""
  return prices_query

def eth_btc_prices(today):

  beginning = f"'{today}'"
  print('beginning', beginning)
  
  prices_query =f"""

WITH addresses AS (
    SELECT column1 AS token_address 
    FROM (VALUES
        (LOWER('0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599')),
        (LOWER('0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'))
    ) AS tokens(column1)
)

select hour,
       symbol,
       price
from ethereum.price.ez_prices_hourly
where token_address in (select token_address from addresses)
and hour >= date_trunc('hour', to_timestamp({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
order by hour desc, symbol 


"""
  return prices_query

def dao_advisor_portfolio(today):
    beginning = f"'{today}'"
    print('beginning', beginning)
    
    prices_query =f"""

  WITH addresses AS (
    SELECT column1 AS token_address 
    FROM (VALUES
        (LOWER('0x1494CA1F11D487c2bBe4543E90080AeBa4BA3C2b')),--dpi,
        (LOWER('0x45804880De22913dAFE09f4980848ECE6EcbAf78')),--paxg,
        (LOWER('0xdab396cCF3d84Cf2D07C4454e10C8A6F5b008D2b')),--gfi,
        (LOWER('0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984')),--UNI
        (LOWER('0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2')),--MKR
        (LOWER('0xc221b7E65FfC80DE234bbB6667aBDd46593D34F0')),--CFG
        (LOWER('0xD33526068D116cE69F19A9ee46F0bd304F21A51f')),--RPL
        (LOWER('0x320623b8E4fF03373931769A31Fc52A4E78B5d70')),--RSR
        (LOWER('0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9')),--AAVE
        (LOWER('0x3432B6A60D23Ca0dFCa7761B7ab56459D9C964D0')),--FRAX
        (LOWER('0xB50721BCf8d664c30412Cfbc6cf7a15145234ad1')),--ARB
        (LOWER('0xC18360217D8F7Ab5e7c516566761Ea12Ce7F9D72')),--ENS
        (LOWER('0xba100000625a3754423978a60c9317c58a424e3D')),--BAL
        (LOWER('0x4F9254C83EB525f9FCf346490bbb3ed28a81C667')),--CELR
        (LOWER('0xc5102fE9359FD9a28f877a67E36B0F050d81a3CC')),--HOP
        (LOWER('0x33349B282065b0284d756F0577FB39c158F935e6')),--MPL
        (LOWER('0xAf5191B0De278C7286d6C7CC6ab6BB8A73bA2Cd6')),--STG
        (LOWER('0x408e41876cCCDC0F92210600ef50372656052a38')),--REN
        (LOWER('0x83F20F44975D03b1b09e64809B757c47f942BEeA')),--SDAI
        (LOWER('0xFC4B8ED459e00e5400be803A9BB3954234FD50e3')),--aWBTC
        (LOWER('0xccf4429db6322d5c611ee964527d42e5d685dd6a')),--CWBTC
        (LOWER('0x467719ad09025fcc6cf6f8311755809d45a5e5f3')),--AXL
        (LOWER('0x44108f0223a3c3028f5fe7aec7f9bb2e66bef82f')),--ACX
        (LOWER('0x0f2d719407fdbeff09d87557abb7232601fd9f29')),--SYN
        (LOWER('0xa11bd36801d8fa4448f0ac4ea7a62e3634ce8c7c'))--ABR

    ) AS tokens(column1)
)

  select hour,
        symbol,
        price
  from ethereum.price.ez_prices_hourly
  where token_address in (select token_address from addresses)
  and hour >= date_trunc('hour', to_timestamp({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
  order by hour desc, symbol 


"""
    return prices_query

def yield_portfolio_prices(today):

  beginning = f"'{today}'"
  print('beginning', beginning)
  
  prices_query =f"""

WITH addresses AS (
    SELECT column1 AS token_address 
    FROM (VALUES
        (LOWER('0x83F20F44975D03b1b09e64809B757c47f942BEeA')), 
        (LOWER('0x9D39A5DE30e57443BfF2A8307A4256c8797A3497')), 
        (LOWER('0x5d3a536e4d6dbd6114cc1ead35777bab948e3643'))

    ) AS tokens(column1)
)

select hour,
       symbol,
       price
from ethereum.price.ez_prices_hourly
where token_address in (select token_address from addresses)
and hour >= date_trunc('hour', to_timestamp({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
order by hour desc, symbol 


"""
  return prices_query

def token_prices(token_addresses, network, start_date):
    start_date = dt.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    """
    Generate a SQL query to get historical price data for given token addresses from a specific start date.

    Parameters:
    - token_addresses (list): List of token addresses.
    - start_date (str): Start date in 'YYYY-MM-DD' format.

    Returns:
    - str: The SQL query string.
    """
    # Format the addresses into the SQL VALUES clause
    addresses_clause = ", ".join(f"(LOWER('{address}'))" for address in token_addresses)

    beginning = f"'{start_date.strftime('%Y-%m-%d %H:%M:%S')}'"
    print('Beginning:', beginning)
    
    prices_query = f"""
    WITH addresses AS (
        SELECT column1 AS token_address 
        FROM (VALUES
            {addresses_clause}
        ) AS tokens(column1)
    )

    SELECT 
        hour,
        symbol,
        price
    FROM 
        {network}.price.ez_prices_hourly
    WHERE 
        token_address IN (SELECT token_address FROM addresses)
        AND hour >= DATE_TRUNC('hour', TO_TIMESTAMP({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
    ORDER BY 
        hour DESC, symbol
    """

    return prices_query



# ),
def token_classifier(network,days,volume_threshold,backtest_period, trending_inc=True):
  
    hours_back = 8640 if backtest_period < 8640 else backtest_period #Ensures token data for least 1 year
    print(f'hours_back: {hours_back}')

    if trending_inc:
   
        query = f"""
        
        WITH 
        v3_assets as (

        select distinct token_address from (

        select distinct TOKEN_OUT as TOKEN_ADDRESS from {network}.defi.ez_dex_swaps where platform = 'uniswap-v3'
        UNION ALL
        select distinct TOKEN_IN as TOKEN_ADDRESS from {network}.defi.ez_dex_swaps where platform = 'uniswap-v3'
        )

        ), 
        rolling_averages AS (
            SELECT
                symbol,
                token_address,
                hour,
                price,
                AVG(price) OVER (
                    PARTITION BY symbol, token_address
                    ORDER BY hour
                    ROWS BETWEEN 168 PRECEDING AND CURRENT ROW
                ) AS rolling_7d_avg,
                AVG(price) OVER (
                    PARTITION BY symbol, token_address
                    ORDER BY hour
                    ROWS BETWEEN 720 PRECEDING AND CURRENT ROW
                ) AS rolling_30d_avg
            FROM
                {network}.price.ez_prices_hourly
            WHERE
                hour >= DATEADD(DAY, -{days}, CURRENT_DATE) -- Ensure a 60-day window
            AND token_address in (select distinct token_address from v3_assets)
        ),
        first_appearance AS (
            SELECT
                symbol,
                token_address,
                MIN(hour) AS first_hour
            FROM
                {network}.price.ez_prices_hourly
            where token_address in (select distinct token_address from v3_assets)
            GROUP BY
                symbol, token_address
        ),
        latest_price AS (
            SELECT
                symbol,
                token_address,
                MAX(hour) AS latest_hour
            FROM
                rolling_averages
            WHERE
                hour >= CURRENT_DATE
            GROUP BY
                symbol, token_address
        ),
        latest_price_details AS (
            SELECT
                lp.symbol,
                lp.token_address,
                lp.latest_hour,
                ra.price AS latest_price,
                ra.rolling_7d_avg,
                ra.rolling_30d_avg
            FROM
                latest_price lp
            JOIN
                rolling_averages ra
            ON
                lp.symbol = ra.symbol AND lp.token_address = ra.token_address AND lp.latest_hour = ra.hour
        ),
        price_60_days_ago AS (
            SELECT
                symbol,
                token_address,
                MIN(hour) AS sixty_days_hour
            FROM
                {network}.price.ez_prices_hourly
            WHERE
                hour >= DATEADD(DAY, -{days}, CURRENT_DATE)
            AND token_address in (select distinct token_address from v3_assets)
            GROUP BY
                symbol, token_address
        ),
        price_60_days_details AS (
            SELECT
                p60.symbol,
                p60.token_address,
                p60.sixty_days_hour,
                eph.price AS sixty_day_price
            FROM
                price_60_days_ago p60
            JOIN
                {network}.price.ez_prices_hourly eph
            ON
                p60.symbol = eph.symbol AND p60.token_address = eph.token_address AND p60.sixty_days_hour = eph.hour
        ),
        token_volumes AS (
            SELECT
                symbol_out,
                token_out,
                AVG(amount_out_usd) AS avg_vol,
                SUM(amount_out_usd) AS sum_vol
            FROM
                {network}.defi.ez_dex_swaps
            WHERE
                DATE_TRUNC('day', block_timestamp) >= DATEADD(DAY, -{days}, CURRENT_DATE)
            AND token_out in (select distinct token_address from v3_assets)
            AND PLATFORM = 'uniswap-v3'
            GROUP BY
                symbol_out, token_out
        ),
        dataset_avg AS (
            SELECT
                AVG(sum_vol) AS overall_avg_vol
            FROM
                token_volumes
        ),
        combined_agg AS (
            SELECT
                tv.symbol_out,
                tv.token_out,
                tv.avg_vol,
                tv.sum_vol
            FROM
                token_volumes tv
            CROSS JOIN
                dataset_avg da
            WHERE
                tv.sum_vol > (da.overall_avg_vol * {volume_threshold})
        )
        SELECT
            lpd.symbol,
            lpd.token_address,
            lpd.latest_hour,
            lpd.latest_price,
            p60d.sixty_day_price,
            (lpd.latest_price - p60d.sixty_day_price) / p60d.sixty_day_price AS sixty_d_return,
            lpd.rolling_7d_avg,
            lpd.rolling_30d_avg,
            tkv.sum_vol as Volume,
            tkv.avg_vol as Average_Order
        FROM
            latest_price_details lpd
        JOIN
            price_60_days_details p60d
        ON
            lpd.symbol = p60d.symbol AND lpd.token_address = p60d.token_address
        JOIN
            first_appearance fa
        ON
            lpd.symbol = fa.symbol AND lpd.token_address = fa.token_address
        JOIN 
            token_volumes tkv
        ON 
            tkv.symbol_out = lpd.symbol AND tkv.token_out = lpd.token_address
        WHERE
            lpd.token_address IN (SELECT DISTINCT token_out FROM combined_agg)
            AND sixty_d_return > 0.001
            AND lpd.rolling_7d_avg > lpd.rolling_30d_avg -- Uncomment to filter tokens trending up
            AND fa.first_hour <= DATEADD(HOUR, -{hours_back}, CURRENT_DATE) -- Ensure data goes back at least 1 year
            AND p60d.sixty_day_price > 0.0001
        ORDER BY
            sixty_d_return DESC;

        """
    else:
       query = f"""
        
        WITH 
        v3_assets as (

        select distinct token_address from (

        select distinct TOKEN_OUT as TOKEN_ADDRESS from {network}.defi.ez_dex_swaps where platform = 'uniswap-v3'
        UNION ALL
        select distinct TOKEN_IN as TOKEN_ADDRESS from {network}.defi.ez_dex_swaps where platform = 'uniswap-v3'
        )

        ), 
        rolling_averages AS (
            SELECT
                symbol,
                token_address,
                hour,
                price,
                AVG(price) OVER (
                    PARTITION BY symbol, token_address
                    ORDER BY hour
                    ROWS BETWEEN 168 PRECEDING AND CURRENT ROW
                ) AS rolling_7d_avg,
                AVG(price) OVER (
                    PARTITION BY symbol, token_address
                    ORDER BY hour
                    ROWS BETWEEN 720 PRECEDING AND CURRENT ROW
                ) AS rolling_30d_avg
            FROM
                {network}.price.ez_prices_hourly
            WHERE
                hour >= DATEADD(DAY, -{days}, CURRENT_DATE) -- Ensure a 60-day window
            AND token_address in (select distinct token_address from v3_assets)
        ),
        first_appearance AS (
            SELECT
                symbol,
                token_address,
                MIN(hour) AS first_hour
            FROM
                {network}.price.ez_prices_hourly
            where token_address in (select distinct token_address from v3_assets)
            GROUP BY
                symbol, token_address
        ),
        latest_price AS (
            SELECT
                symbol,
                token_address,
                MAX(hour) AS latest_hour
            FROM
                rolling_averages
            WHERE
                hour >= CURRENT_DATE
            GROUP BY
                symbol, token_address
        ),
        latest_price_details AS (
            SELECT
                lp.symbol,
                lp.token_address,
                lp.latest_hour,
                ra.price AS latest_price,
                ra.rolling_7d_avg,
                ra.rolling_30d_avg
            FROM
                latest_price lp
            JOIN
                rolling_averages ra
            ON
                lp.symbol = ra.symbol AND lp.token_address = ra.token_address AND lp.latest_hour = ra.hour
        ),
        price_60_days_ago AS (
            SELECT
                symbol,
                token_address,
                MIN(hour) AS sixty_days_hour
            FROM
                {network}.price.ez_prices_hourly
            WHERE
                hour >= DATEADD(DAY, -{days}, CURRENT_DATE)
            AND token_address in (select distinct token_address from v3_assets)
            GROUP BY
                symbol, token_address
        ),
        price_60_days_details AS (
            SELECT
                p60.symbol,
                p60.token_address,
                p60.sixty_days_hour,
                eph.price AS sixty_day_price
            FROM
                price_60_days_ago p60
            JOIN
                {network}.price.ez_prices_hourly eph
            ON
                p60.symbol = eph.symbol AND p60.token_address = eph.token_address AND p60.sixty_days_hour = eph.hour
        ),
        token_volumes AS (
            SELECT
                symbol_out,
                token_out,
                AVG(amount_out_usd) AS avg_vol,
                SUM(amount_out_usd) AS sum_vol
            FROM
                {network}.defi.ez_dex_swaps
            WHERE
                DATE_TRUNC('day', block_timestamp) >= DATEADD(DAY, -{days}, CURRENT_DATE)
            AND token_out in (select distinct token_address from v3_assets)
            AND PLATFORM = 'uniswap-v3'
            GROUP BY
                symbol_out, token_out
        ),
        dataset_avg AS (
            SELECT
                AVG(sum_vol) AS overall_avg_vol
            FROM
                token_volumes
        ),
        combined_agg AS (
            SELECT
                tv.symbol_out,
                tv.token_out,
                tv.avg_vol,
                tv.sum_vol
            FROM
                token_volumes tv
            CROSS JOIN
                dataset_avg da
            WHERE
                tv.sum_vol > (da.overall_avg_vol * {volume_threshold})
        )
        SELECT
            lpd.symbol,
            lpd.token_address,
            lpd.latest_hour,
            lpd.latest_price,
            p60d.sixty_day_price,
            (lpd.latest_price - p60d.sixty_day_price) / p60d.sixty_day_price AS sixty_d_return,
            lpd.rolling_7d_avg,
            lpd.rolling_30d_avg,
            tkv.sum_vol as Volume,
            tkv.avg_vol as Average_Order
        FROM
            latest_price_details lpd
        JOIN
            price_60_days_details p60d
        ON
            lpd.symbol = p60d.symbol AND lpd.token_address = p60d.token_address
        JOIN
            first_appearance fa
        ON
            lpd.symbol = fa.symbol AND lpd.token_address = fa.token_address
        JOIN 
            token_volumes tkv
        ON 
            tkv.symbol_out = lpd.symbol AND tkv.token_out = lpd.token_address
        WHERE
            lpd.token_address IN (SELECT DISTINCT token_out FROM combined_agg)
            AND sixty_d_return > 0.001
            AND fa.first_hour <= DATEADD(HOUR, -{hours_back}, CURRENT_DATE) -- Ensure data goes back at least 1 year
            AND p60d.sixty_day_price > 0.0001
        ORDER BY
            sixty_d_return DESC;

        """
       
    return query

def model_flows(token_addresses, model_address, network):
   addresses_clause = ", ".join(f"(LOWER('{address}'))" for address in token_addresses)
   query = f"""

WITH portfolio AS (
        SELECT column1 AS token_address 
        FROM (VALUES
            {addresses_clause}
        ) AS tokens(column1)
    ),
inflows as (
  select
    date_trunc('hour',block_timestamp) as dt,
    SYMBOL,
    amount_usd,
    'inflow' as transaction_type
  from
    {network}.core.ez_token_transfers
  where
    to_address = lower('{model_address}')
    AND amount_usd is not NULL
    AND TX_HASH NOT IN (
      SELECT
        DISTINCT TX_HASH
      FROM
        {network}.defi.ez_dex_swaps
    )
    AND CONTRACT_ADDRESS IN (
      select
        distinct token_address
      from
        portfolio
    )
),
outflows as (
  select
    date_trunc('hour',block_timestamp) as dt,
    SYMBOL,
    amount_usd * -1 as amount_usd,
    -- Use negative values for outflows
    'outflow' as transaction_type
  from
    {network}.core.ez_token_transfers
  where
    from_address = lower('{model_address}')
    AND amount_usd is not NULL
    AND TX_HASH NOT IN (
      SELECT
        DISTINCT TX_HASH
      FROM
        {network}.defi.ez_dex_swaps
    )
    AND CONTRACT_ADDRESS IN (
      select
        distinct token_address
      from
        portfolio
    )
)
select
  *
from
  inflows
union
all
select
  *
from
  outflows
order by
  dt;


"""
   return query


def all_yield_portfolio_prices(today):

  beginning = f"'{today}'"
  print('beginning', beginning)
  
  prices_query =f"""

WITH addresses AS (
    SELECT column1 AS token_address 
    FROM (VALUES
        (LOWER('0xae78736cd615f374d3085123a210448e74fc6393')),
        (LOWER('0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0')),
        (LOWER('0xac3e018457b222d93114458476f3e3416abbe38f')),
        (LOWER('0x83F20F44975D03b1b09e64809B757c47f942BEeA')),
        (LOWER('0x9D39A5DE30e57443BfF2A8307A4256c8797A3497')),
        (LOWER('0xA1290d69c65A6Fe4DF752f95823fae25cB99e5A7')),
        (LOWER('0xBe9895146f7AF43049ca1c1AE358B0541Ea49704')),
        (LOWER('0xf1C9acDc66974dFB6dEcB12aA385b9cD01190E38')),
        (LOWER('0x8236a87084f8B84306f72007F36F2618A5634494')),
        (LOWER('0xCd5fE23C85820F7B72D0926FC9b05b43E359b7ee')),
        (LOWER('0xf951e335afb289353dc249e82926178eac7ded78')),
        (LOWER('0xa35b1b31ce002fbf2058d22f30f95d405200a15b')),
        (LOWER('0x8c1bed5b9a0928467c9b1341da1d7bd5e10b6549')),
        (LOWER('0xe95a203b1a91a908f9b9ce46459d101078c2c3cb')),
        (LOWER('0x856c4efb76c1d1ae02e20ceb03a2a6a08b0b8dc3'))

    ) AS tokens(column1)
)

select hour,
       symbol,
       price
from ethereum.price.ez_prices_hourly
where token_address in (select token_address from addresses)
and hour >= date_trunc('hour', to_timestamp({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
order by hour desc, symbol 


"""
  return prices_query

def arb_stables_portfolio_prices(today):

  beginning = f"'{today}'"
  print('beginning', beginning)
  
  prices_query =f"""

WITH addresses AS (
    SELECT column1 AS token_address 
    FROM (VALUES
        (LOWER('0xd3443ee1e91af28e5fb858fbd0d72a63ba8046e0')),
        (LOWER('0x7788a3538c5fc7f9c7c8a74eac4c898fc8d87d92'))

    ) AS tokens(column1)
)

select hour,
       symbol,
       price
from arbitrum.price.ez_prices_hourly
where token_address in (select token_address from addresses)
and hour >= date_trunc('hour', to_timestamp({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
order by hour desc, symbol 


"""
  return prices_query

def all_arb_stables_prices(start_date):
    query = """
    WITH arb_addresses AS (
        SELECT
            column1 AS token_address
        FROM
            (
            VALUES
                (
                LOWER('0xd3443ee1e91af28e5fb858fbd0d72a63ba8046e0')
                ),
                (
                LOWER('0x7788A3538C5fc7F9c7C8A74EAC4c898fC8d87d92')
                )
            ) AS tokens(column1)
    ),
    eth_addresses AS (
        SELECT
            column1 AS token_address
        FROM
            (
            VALUES
                (
                LOWER('0x9D39A5DE30e57443BfF2A8307A4256c8797A3497')
                )
            ) AS tokens(column1)
    ),
    arb_prices AS (
        SELECT
            hour,
            symbol,
            price
        FROM
            arbitrum.price.ez_prices_hourly
        WHERE
            token_address IN (
            SELECT
                token_address
            FROM
                arb_addresses
            )
        ORDER BY
            hour DESC,
            symbol
    ),
    eth_prices AS (
        SELECT
            hour,
            symbol,
            price
        FROM
            ethereum.price.ez_prices_hourly
        WHERE
            token_address IN (
            SELECT
                token_address
            FROM
                eth_addresses
            )
        ORDER BY
            hour DESC,
            symbol
    ),
    combined_prices AS (
        SELECT
            *
        FROM
            arb_prices
        UNION
        ALL
        SELECT
            *
        FROM
            eth_prices
    ),
    earliest_data AS (
        SELECT
            MIN(hour) AS earliest
        FROM
            combined_prices
        GROUP BY
            symbol
    )
    SELECT
        hour, symbol, price 
    FROM
        combined_prices
    WHERE
        hour >= (
            SELECT
                MAX(earliest)
            FROM
                earliest_data
        )
    ORDER BY
        hour DESC
    """

    return query
