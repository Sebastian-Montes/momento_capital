from .strategies import arjun


def gen_arjun(
    start_date,
    end_date,
    target_rebalance_day,
    filtered_holdings_df,
    filtered_etfs_df,
    interval_keyed_historical_holdings,
    sector_keyed_holdings,
    # holdings_largest_window,
    # etfs_largest_window,
    gen,
):
    return arjun(
        start_date=start_date,
        end_date=end_date,
        target_rebalance_day=target_rebalance_day,
        filtered_holdings_df=filtered_holdings_df,
        filtered_etfs_df=filtered_etfs_df,
        interval_keyed_historical_holdings=interval_keyed_historical_holdings,
        sector_keyed_holdings=sector_keyed_holdings,
        etfs_largest_window=max(gen[3], gen[4], gen[5]),
        holdings_largest_window=max(gen[6], gen[7], gen[8], gen[9]),
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
