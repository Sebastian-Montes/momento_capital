from .portfolio import PortfolioEvaluator, PortfolioSimulator, TrailingStopBollinger
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

    # equity_records = portfolio.value
    # if equity_records[0]["date"] != holdings_first_valid_date:
    #     equity_records = [
    #         {"date": date, "value": 100000}
    #         for date in dates
    #         if date < equity_records[0]["date"]
    #     ] + equity_records

    # logs_records = portfolio.history
    # trades_records = portfolio.trades
    # holdings_records = portfolio.holdings
    # metrics_records = portfolio.metrics.to_dict(orient="records")
    # trade_metrics_records = portfolio.trade_metrics.to_dict(orient="records")

    # return {
    #     "equity": equity_records,
    #     "logs": logs_records,
    #     "trades": trades_records,
    #     "holdings": holdings_records,
    #     # "metrics": metrics_records,
    #     # "trade_metrics": trade_metrics_records,
    # }
    # return equity_records


def dinorah(
    etfs_df,
    holdings_df,
    start_date,
    end_date,
    target_rebalance_date,
    interval_keyed_historical_holdings,
    sector_keyed_holdings,
    benchmark_series,
    portfolio_id,
    freq=15,
    etfs_rsi_window_size=20,
    etfs_rsi_lower_limit=20,
    etfs_rsi_upper_limit=50,
    etfs_sma_window_size_1=10,
    etfs_sma_window_size_2=20,
    holdings_sma_window_size=15,
    holdings_rm_window_size=15,
    holdings_rm_n_top=5,
    bb_window_size=20,
    bb_factor=2.2,
):

    filtered_etfs_df, etfs_first_valid_date = process_data(
        filtered_df=etfs_df,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        target_rebalance_date=target_rebalance_date,
        max_window=max(
            etfs_rsi_window_size, etfs_sma_window_size_1, etfs_sma_window_size_2
        ),
    )

    filtered_holdings_df, holdings_first_valid_date = process_data(
        filtered_df=holdings_df,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        target_rebalance_date=target_rebalance_date,
        max_window=max(holdings_sma_window_size, holdings_rm_window_size),
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

    etfs_rsi_signal = etf_extractor.conditioned_rsi(
        rsi_windows=[etfs_rsi_window_size],
        rsi_condition=lambda rsi: (rsi < etfs_rsi_upper_limit)
        and (rsi > etfs_rsi_lower_limit),
    )

    etfs_sma_signal = etf_extractor.conditioned_price_vs_sma(
        sma_windows=[etfs_sma_window_size_1, etfs_sma_window_size_2],
        condition=lambda price, sma: price < sma,
    )

    holding_extractor = HoldingsSignalExtractor(
        holdings_df=filtered_holdings_df,
        freq=freq,
        interval_keyed_historical_holdings=interval_keyed_historical_holdings,
        sector_keyed_holdings=sector_keyed_holdings,
        detailed_sector_signal=extract_common_detailed_signal(
            signals=[etfs_rsi_signal, etfs_sma_signal]
        ),
        start_date=holdings_first_valid_date,
    )

    holdings_sma_signal = holding_extractor.conditioned_price_vs_sma(
        sma_windows=[holdings_sma_window_size],
        condition=lambda price, sma: price > sma,
    )

    holdings_rm_signal = holding_extractor.top_most_relative_momentum(
        n_top=holdings_rm_n_top,
        relative_momentum_windows=[holdings_rm_window_size],
        base_signal=holdings_sma_signal,
    )

    holding_signal = {k: list(v.keys()) for k, v in holdings_rm_signal.items()}
    cleaned_signal = clean_signal(holding_signal)
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

    # equity_records = portfolio.value
    # if equity_records[0]["date"] != holdings_first_valid_date:
    #     equity_records = [
    #         {"date": date, "value": 100000}
    #         for date in dates
    #         if date < equity_records[0]["date"]
    #     ] + equity_records
    # return equity_records
