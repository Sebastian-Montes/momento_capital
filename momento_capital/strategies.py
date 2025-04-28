from .portfolio import PortfolioEvaluator, PortfolioSimulator, TrailingStopBollinger, TrailingStopSMA
from signals.utilities import extract_common_detailed_signal, process_data, clean_signal
from signals.extractor import EtfSignalExtractor, HoldingsSignalExtractor
import pandas as pd


def arjun(
    start_date,
    end_date,
    filtered_holdings_df,
    filtered_etfs_df,
    benchmark_series,
    interval_keyed_historical_holdings,
    sector_keyed_holdings,
    # benchmark_series,
    holdings_largest_window,
    etfs_largest_window,
    freq,
    etfs_vol_n_top,
    holdings_vol_n_top,
    etfs_volatility_window_size,
    etfs_long_sma_window_size,
    etfs_short_sma_window_size,
    holdings_volatility_window_size,
    holdings_long_sma_window_size,
    holdings_short_sma_window_size,
    bb_window_size,
    bb_factor,
    target_rebalance_date=None,
    portfolio_id="_id_",
):

    filtered_holdings_df, holdings_first_valid_date = process_data(
        filtered_df=filtered_holdings_df,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        target_rebalance_date=target_rebalance_date,
        max_window=holdings_largest_window,
    )

    filtered_etfs_df, etfs_first_valid_date = process_data(
        filtered_df=filtered_etfs_df,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        target_rebalance_date=target_rebalance_date,
        max_window=etfs_largest_window,
    )

    dates = [
        date.strftime("%Y-%m-%d")
        for date in filtered_holdings_df.index
        if date > pd.to_datetime(holdings_first_valid_date)
    ]

    etf_extractor = EtfSignalExtractor(
        etfs_df=filtered_etfs_df,
        freq=freq,
        interval_keyed_historical_holdings=interval_keyed_historical_holdings,
        sector_keyed_holdings=sector_keyed_holdings,
        start_date=etfs_first_valid_date,
    )

    etfs_volatility_signal = etf_extractor.top_most_volatile(
        n_top=etfs_vol_n_top,
        volatility_windows=[etfs_volatility_window_size],
        returns_period=1,
        returns_method="logarithmic",
        ddof=0,
    )
    etfs_sma_signal = etf_extractor.conditioned_price_vs_sma(
        sma_windows=[etfs_long_sma_window_size, etfs_short_sma_window_size],
        condition=lambda price, indicator: price >= indicator,
    )

    etfs_signal = extract_common_detailed_signal(
        [etfs_volatility_signal, etfs_sma_signal]
    )

    holdings_extractor = HoldingsSignalExtractor(
        holdings_df=filtered_holdings_df,
        freq=freq,
        detailed_sector_signal=etfs_signal,
        sector_keyed_holdings=sector_keyed_holdings,
        interval_keyed_historical_holdings=interval_keyed_historical_holdings,
        start_date=holdings_first_valid_date,
    )
    holdings_volatility_signal = holdings_extractor.top_least_volatile(
        n_top=holdings_vol_n_top,
        volatility_windows=[holdings_volatility_window_size],
        returns_method="logarithmic",
    )

    holdings_sma_signal_1 = holdings_extractor.conditioned_price_vs_sma(
        sma_windows=[holdings_long_sma_window_size],
        condition=lambda price, sma: price < sma,
        base_signal=holdings_volatility_signal,
    )
    holdings_sma_signal_2 = holdings_extractor.conditioned_price_vs_sma(
        sma_windows=[holdings_short_sma_window_size],
        condition=lambda price, sma: price >= sma,
        base_signal=holdings_sma_signal_1,
    )

    holdings_signal = {k: list(v.keys()) for k, v in holdings_sma_signal_2.items()}
    cleaned_signal = clean_signal(holdings_signal)
    if all(len(s) == 0 for s in cleaned_signal.values()):
        return [{"date": date, "value": 100000} for date in dates]

    manager = TrailingStopBollinger(
        df=filtered_holdings_df, window_size=bb_window_size, bollinger_factor=bb_factor
    )
    evaluator = PortfolioEvaluator(benchmark_series=benchmark_series)
    portfolio = PortfolioSimulator(
        initial_cash=100000,
        target_weight=1,
        df=filtered_holdings_df,
        id_structure="11111",
        manager=manager,
        evaluator=evaluator,
        seed=1,
        verbose=0,
        portfolio_id=portfolio_id,
    )
    portfolio.simulate(cleaned_signal)

    return portfolio

def dinorah(
    etfs_df,
    holdings_df,
    start_date,
    end_date,
    target_rebalance_date,
    interval_keyed_historical_holdings,
    sector_keyed_holdings,
    portfolio_id,
    freq,
    etfs_vol_window_size,
    etfs_vol_threshold,
    holdings_rsi_window_size,
    holdings_rsi_upper_limit,
    holdings_rsi_lower_limit,
    holdings_vol_window_size,
    holdings_vol_n_top, 
    manager_sma_window_size
):

    filtered_etfs_df, etfs_first_valid_date = process_data(
        filtered_df=etfs_df,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        target_rebalance_date=target_rebalance_date,
        max_window=etfs_vol_window_size
    )

    filtered_holdings_df, holdings_first_valid_date = process_data(
        filtered_df=holdings_df,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        target_rebalance_date=target_rebalance_date,
        max_window=max(holdings_rsi_window_size, holdings_vol_window_size),
    )

    dates = [
        date.strftime("%Y-%m-%d")
        for date in filtered_holdings_df.index
        if date > pd.to_datetime(holdings_first_valid_date)
    ]

    etf_extractor = EtfSignalExtractor(
        etfs_df=filtered_etfs_df,
        freq=freq,
        interval_keyed_historical_holdings=interval_keyed_historical_holdings,
        sector_keyed_holdings=sector_keyed_holdings,
        start_date=etfs_first_valid_date,
    )

    etfs_vol_signal = etf_extractor.conditioned_volatility(
        volatility_windows=[etfs_vol_window_size],
        condition=lambda vol: vol <= etfs_vol_threshold,
        returns_method="percentage",
        returns_period=1,
    )

    holding_extractor = HoldingsSignalExtractor(
        holdings_df=filtered_holdings_df,
        freq=freq,
        interval_keyed_historical_holdings=interval_keyed_historical_holdings,
        sector_keyed_holdings=sector_keyed_holdings,
        detailed_sector_signal=etfs_vol_signal,
        start_date=holdings_first_valid_date,
    )

    holding_rsi_signal = holding_extractor.conditioned_rsi(
        rsi_windows=[holdings_rsi_window_size],
        rsi_condition=lambda rsi: (rsi >= holdings_rsi_lower_limit)
        and (rsi <= holdings_rsi_upper_limit),
    )
    holdings_vol_signal = holding_extractor.top_least_volatile(
        n_top=holdings_vol_n_top,
        volatility_windows=[holdings_vol_window_size],
        returns_period=1,
        returns_method="percentage",
        base_signal=holding_rsi_signal,
    )

    holding_signal = {k: list(v.keys()) for k, v in holdings_vol_signal.items()}
    cleaned_signal = clean_signal(holding_signal)
    if all(len(s) == 0 for s in cleaned_signal.values()):
        return [{"date": date, "value": 100000} for date in dates]

    manager = TrailingStopSMA(period=manager_sma_window_size, df=filtered_holdings_df)
    portfolio = PortfolioSimulator(
        initial_cash=100000,
        target_weight=1,
        df=filtered_holdings_df,
        id_structure="11111",
        manager=manager,
        evaluator=None,
        seed=1,
        verbose=0,
        portfolio_id=portfolio_id,
    )
    portfolio.simulate(cleaned_signal)
    return portfolio