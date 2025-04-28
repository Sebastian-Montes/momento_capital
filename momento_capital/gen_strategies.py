from .strategies import arjun, dinorah


def gen_arjun(
    gen,
    portfolio_id,
    start_date,
    end_date,
    target_rebalance_date,
    filtered_holdings_df,
    filtered_etfs_df,
    interval_keyed_historical_holdings,
    sector_keyed_holdings,
    benchmark_series,
):
    return arjun(
        start_date=start_date,
        end_date=end_date,
        target_rebalance_date=target_rebalance_date,
        filtered_holdings_df=filtered_holdings_df,
        filtered_etfs_df=filtered_etfs_df,
        interval_keyed_historical_holdings=interval_keyed_historical_holdings,
        sector_keyed_holdings=sector_keyed_holdings,
        benchmark_series=benchmark_series,
        etfs_largest_window=max(gen[3], gen[4], gen[5]),
        holdings_largest_window=max(gen[6], gen[7], gen[8], gen[9]),
        portfolio_id=portfolio_id,
        freq=int(gen[0]),
        etfs_vol_n_top=int(gen[1]),
        holdings_vol_n_top=int(gen[2]),
        etfs_volatility_window_size=int(gen[3]),
        etfs_long_sma_window_size=int(gen[4]),
        etfs_short_sma_window_size=int(gen[5]),
        holdings_volatility_window_size=int(gen[6]),
        holdings_long_sma_window_size=int(gen[7]),
        holdings_short_sma_window_size=int(gen[8]),
        bb_window_size=int(gen[9]),
        bb_factor=float(gen[10]),
    )


def gen_dinorah(
    gen,
    portfolio_id,
    start_date,
    end_date,
    target_rebalance_date,
    filtered_holdings_df,
    filtered_etfs_df,
    interval_keyed_historical_holdings,
    sector_keyed_holdings, benchmark_series
):
    return dinorah(
        etfs_df=filtered_etfs_df,
        holdings_df=filtered_holdings_df,
        start_date=start_date,
        end_date=end_date,
        target_rebalance_date=target_rebalance_date,
        interval_keyed_historical_holdings=interval_keyed_historical_holdings,
        sector_keyed_holdings=sector_keyed_holdings,
        portfolio_id=portfolio_id, benchmark_series=benchmark_series,
        freq=int(gen[0]),
        etfs_vol_window_size=int(gen[1]),
        etfs_vol_threshold=float(gen[2]),
        holdings_rsi_window_size=int(gen[3]),
        holdings_rsi_upper_limit=int(gen[4]),
        holdings_rsi_lower_limit=int(gen[5]),
        holdings_vol_window_size=int(gen[6]),
        holdings_vol_n_top=int(gen[7]),
        manager_sma_window_size=int(gen[8]),
    )
