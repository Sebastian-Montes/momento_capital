from .portfolio import (
    PortfolioEvaluator,
    PortfolioSimulator,
    TrailingStopBollinger,
    TrailingStopSMA,
)
from .transformers import (
    calculate_relative_volatility_on_prices,
    calculate_simple_moving_average,
)
from .utilities import (
    find_current_filtered_active_holdings,
    process_data,
    apply_function_to_data,
    find_active_etfs,
)
from signals.utilities import extract_common_detailed_signal, clean_signal
import pandas as pd
from godolib.fast_transformers import calculate_rsi


def dinorah(
    etfs_df,
    holdings_df,
    start_date,
    end_date,
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
    manager_sma_window_size,
    benchmark_series,
):

    filtered_etfs_df, etfs_first_valid_date = process_data(
        filtered_df=etfs_df,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        max_window=etfs_vol_window_size,
    )

    filtered_holdings_df, holdings_first_valid_date = process_data(
        filtered_df=holdings_df,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        max_window=max(
            holdings_rsi_window_size, holdings_vol_window_size, manager_sma_window_size
        ),
    )

    dates = [
        date.strftime("%Y-%m-%d")
        for date in filtered_holdings_df.index
        if date > pd.to_datetime(holdings_first_valid_date)
    ]

    etfs_vol_dates = [date.strftime("%Y-%m-%d") for date in filtered_etfs_df.index]
    if etfs_first_valid_date not in etfs_vol_dates:
        raise ValueError("start date not in index")
    etfs_vol_dfs = [
        apply_function_to_data(
            df=filtered_etfs_df,
            function=calculate_relative_volatility_on_prices,
            returns_period=1,
            returns_method="percentage",
            ddof=0,
            window_size=etfs_vol_window_size,
        )
    ]
    etfs_vol_first_valid_date = max(
        vol_df.index[0].strftime("%Y-%m-%d") for vol_df in etfs_vol_dfs
    )
    if etfs_vol_first_valid_date >= etfs_first_valid_date:
        raise ValueError(
            f"not enough data to satisfy given start date ({etfs_first_valid_date}). Transformed data first valid date: {etfs_vol_first_valid_date})"
        )
    etfs_vol_dates = [
        date
        for date in etfs_vol_dates
        if pd.to_datetime(date) >= pd.to_datetime(etfs_first_valid_date)
    ]
    etfs_vol_dfs = [
        vol_df.loc[vol_df.index >= etfs_first_valid_date] for vol_df in etfs_vol_dfs
    ]

    etfs_vol_signal = {}

    etfs_vol_condition = lambda vol: vol <= etfs_vol_threshold

    for date_idx, date in enumerate(etfs_vol_dates):
        if date_idx % freq == 0:
            current_active_etfs = find_active_etfs(
                str_date=date,
                interval_keyed_historical_holdings=interval_keyed_historical_holdings,
                sector_keyed_holdings=sector_keyed_holdings,
            )
            current_active_etfs = [
                etf for etf in current_active_etfs if etf in filtered_etfs_df
            ]
            values = [
                df[current_active_etfs].loc[date].to_dict() for df in etfs_vol_dfs
            ]
            etfs_vol_signal[date] = [
                etf
                for etf in current_active_etfs
                if all(etfs_vol_condition(value[etf]) for value in values)
            ]

    etfs_vol_signal = {
        date: {
            etf: {
                "price": float(filtered_etfs_df.loc[date, etf]),
                **{
                    f"volatility_{window}": float(
                        etfs_vol_dfs[window_idx].loc[date, etf]
                    )
                    for window_idx, window in enumerate([etfs_vol_window_size])
                },
            }
            for etf in date_signal
        }
        for date, date_signal in etfs_vol_signal.items()
    }

    holdings_rsi_dates = [
        date.strftime("%Y-%m-%d") for date in filtered_holdings_df.index
    ]
    if holdings_first_valid_date not in holdings_rsi_dates:
        raise ValueError("start date not in index")
    holdings_rsi_dfs = [
        apply_function_to_data(
            df=filtered_holdings_df,
            function=calculate_rsi,
            window_size=holdings_rsi_window_size,
        )
    ]
    holdings_rsi_first_valid_date = max(
        rsi_df.index[0].strftime("%Y-%m-%d") for rsi_df in holdings_rsi_dfs
    )
    if holdings_rsi_first_valid_date > holdings_first_valid_date:
        raise ValueError(
            f"not enough data to satisfy given start date ({holdings_first_valid_date}). Transformed data first valid date: {holdings_rsi_first_valid_date})"
        )
    holdings_rsi_dates = [
        date
        for date in holdings_rsi_dates
        if pd.to_datetime(date) >= pd.to_datetime(holdings_first_valid_date)
    ]
    holdings_rsi_dfs = [
        rsi_df.loc[rsi_df.index >= holdings_first_valid_date]
        for rsi_df in holdings_rsi_dfs
    ]

    holding_rsi_signal = {}
    holdings_rsi_condition = lambda rsi: (rsi >= holdings_rsi_lower_limit) and (
        rsi <= holdings_rsi_upper_limit
    )

    for date_idx, date in enumerate(holdings_rsi_dates):
        if date_idx % freq == 0:
            current_active_holdings = find_current_filtered_active_holdings(
                date=date,
                interval_keyed_historical_holdings=interval_keyed_historical_holdings,
                holdings_df=filtered_holdings_df,
                sector_keyed_holdings=sector_keyed_holdings,
                sector_signal={
                    date: list(date_signal.keys())
                    for date, date_signal in etfs_vol_signal.items()
                },
            )
            holding_rsi_signal[date] = [
                holding
                for holding in current_active_holdings
                if all(
                    holdings_rsi_condition(
                        df[current_active_holdings].loc[date].to_dict()[holding]
                    )
                    for df in holdings_rsi_dfs
                )
            ]

    holding_rsi_signal = {
        date: {
            holding: {
                "price": float(filtered_holdings_df.loc[date, holding]),
                **{
                    f"rsi_{window}": float(
                        holdings_rsi_dfs[window_idx].loc[date, holding]
                    )
                    for window_idx, window in enumerate([holdings_rsi_window_size])
                },
            }
            for holding in date_signal
        }
        for date, date_signal in holding_rsi_signal.items()
    }

    holdings_vol_dates = [
        date.strftime("%Y-%m-%d") for date in filtered_holdings_df.index
    ]
    if holdings_first_valid_date not in holdings_vol_dates:
        raise ValueError("start date not in index")
    holdings_vol_dfs = [
        apply_function_to_data(
            df=filtered_holdings_df,
            function=calculate_relative_volatility_on_prices,
            returns_period=1,
            window_size=holdings_vol_window_size,
            returns_method="percentage",
            ddof=0,
        )
    ]

    holdings_vol_first_valid_date = max(
        vol_df.index[0].strftime("%Y-%m-%d") for vol_df in holdings_vol_dfs
    )
    if holdings_vol_first_valid_date > holdings_first_valid_date:
        raise ValueError(
            f"not enough data to satisfy given start date ({holdings_first_valid_date}). Transformed data first valid date: {holdings_vol_first_valid_date})"
        )
    holdings_vol_dates = [
        date
        for date in holdings_vol_dates
        if pd.to_datetime(date) >= pd.to_datetime(holdings_first_valid_date)
    ]
    holdings_vol_dfs = [
        vol_df.loc[vol_df.index >= holdings_first_valid_date]
        for vol_df in holdings_vol_dfs
    ]

    holdings_vol_signal = {}

    for date_idx, date in enumerate(holdings_vol_dates):
        if date_idx % freq == 0:
            current_active_holdings = list(holding_rsi_signal[date].keys())
            holdings_vol_signal[date] = [
                holding
                for holding in current_active_holdings
                if all(
                    holding
                    in list(
                        dict(
                            sorted(
                                df[current_active_holdings].loc[date].to_dict().items(),
                                key=lambda items: items[1],
                                reverse=False,
                            )
                        ).keys()
                    )[:holdings_vol_n_top]
                    for df in holdings_vol_dfs
                )
            ]

    holdings_vol_signal = {
        date: {
            holding: {
                "price": float(filtered_holdings_df.loc[date, holding]),
                **{
                    f"volatility_{window}": float(
                        holdings_vol_dfs[window_idx].loc[date, holding]
                    )
                    for window_idx, window in enumerate([holdings_vol_window_size])
                },
            }
            for holding in date_signal
        }
        for date, date_signal in holdings_vol_signal.items()
    }

    holding_signal = {k: list(v.keys()) for k, v in holdings_vol_signal.items()}
    cleaned_signal = clean_signal(holding_signal)
    if all(len(s) == 0 for s in cleaned_signal.values()):
        return [{"date": date, "value": 100000} for date in dates]

    manager = TrailingStopSMA(period=manager_sma_window_size, df=filtered_holdings_df)
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


def arjun(
    start_date,
    end_date,
    filtered_holdings_df,
    filtered_etfs_df,
    interval_keyed_historical_holdings,
    sector_keyed_holdings,
    benchmark_series,
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
    portfolio_id="_id_",
):

    filtered_holdings_df, holdings_first_valid_date = process_data(
        filtered_df=filtered_holdings_df,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        max_window=holdings_largest_window,
    )
    filtered_etfs_df, etfs_first_valid_date = process_data(
        filtered_df=filtered_etfs_df,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        max_window=etfs_largest_window,
    )

    dates = [
        date.strftime("%Y-%m-%d")
        for date in filtered_holdings_df.index
        if date > pd.to_datetime(holdings_first_valid_date)
    ]

    etfs_dates = [date.strftime("%Y-%m-%d") for date in filtered_etfs_df.index]
    etfs_vol_dfs = [
        apply_function_to_data(
            df=filtered_etfs_df,
            function=calculate_relative_volatility_on_prices,
            returns_period=1,
            returns_method="percentage",
            ddof=0,
            window_size=etfs_volatility_window_size,
        )
    ]
    etfs_vol_first_valid_date = max(
        vol_df.index[0].strftime("%Y-%m-%d") for vol_df in etfs_vol_dfs
    )
    if etfs_vol_first_valid_date >= etfs_first_valid_date:
        raise ValueError(
            f"not enough data to satisfy given start date ({etfs_first_valid_date}). Transformed data first valid date: {etfs_vol_first_valid_date})"
        )
    etfs_vol_dates = [
        date
        for date in etfs_dates
        if pd.to_datetime(date) >= pd.to_datetime(etfs_first_valid_date)
    ]
    etfs_vol_dfs = [
        vol_df.loc[vol_df.index >= etfs_first_valid_date] for vol_df in etfs_vol_dfs
    ]

    etfs_volatility_signal = {}
    for date_idx, date in enumerate(etfs_vol_dates):
        if date_idx % freq == 0:
            current_active_etfs = find_active_etfs(
                str_date=date,
                interval_keyed_historical_holdings=interval_keyed_historical_holdings,
                sector_keyed_holdings=sector_keyed_holdings,
            )
            current_active_etfs = [
                etf for etf in current_active_etfs if etf in filtered_etfs_df
            ]
            etfs_volatility_signal[date] = [
                etf
                for etf in current_active_etfs
                if all(
                    etf
                    in list(
                        dict(
                            sorted(
                                df[current_active_etfs].loc[date].to_dict().items(),
                                key=lambda items: items[1],
                                reverse=True,
                            )
                        ).keys()
                    )[:etfs_vol_n_top]
                    for df in etfs_vol_dfs
                )
            ]
    etfs_volatility_signal = {
        date: {
            etf: {
                "price": float(filtered_etfs_df.loc[date, etf]),
                **{
                    f"volatility_{window}": float(
                        etfs_vol_dfs[window_idx].loc[date, etf]
                    )
                    for window_idx, window in enumerate([etfs_volatility_window_size])
                },
            }
            for etf in date_signal
        }
        for date, date_signal in etfs_volatility_signal.items()
    }

    sma_windows = [etfs_long_sma_window_size, etfs_short_sma_window_size]

    etfs_sma_dfs = [
        apply_function_to_data(
            df=filtered_etfs_df,
            function=calculate_simple_moving_average,
            window_size=window,
        )
        for window in sma_windows
    ]
    etfs_sma_first_valid_date = max(
        sma_df.index[0].strftime("%Y-%m-%d") for sma_df in etfs_sma_dfs
    )
    if etfs_sma_first_valid_date >= etfs_first_valid_date:
        raise ValueError(
            f"not enough data to satisfy given start date ({etfs_first_valid_date}). Transformed data first valid date: {etfs_sma_first_valid_date})"
        )
    etfs_sma_dates = [
        date
        for date in etfs_dates
        if pd.to_datetime(date) >= pd.to_datetime(etfs_first_valid_date)
    ]
    etfs_sma_dfs = [
        sma_df.loc[sma_df.index >= etfs_first_valid_date] for sma_df in etfs_sma_dfs
    ]

    etfs_sma_condition = lambda price, indicator: price >= indicator

    etfs_sma_signal = {}
    for date_idx, date in enumerate(etfs_sma_dates):
        if date_idx % freq == 0:
            current_active_etfs = find_active_etfs(
                str_date=date,
                interval_keyed_historical_holdings=interval_keyed_historical_holdings,
                sector_keyed_holdings=sector_keyed_holdings,
            )
            current_active_etfs = [
                etf for etf in current_active_etfs if etf in filtered_etfs_df
            ]
            etfs_sma_signal[date] = [
                etf
                for etf in current_active_etfs
                if all(
                    etfs_sma_condition(
                        filtered_etfs_df[etf].loc[date], indicator[etf].loc[date]
                    )
                    for indicator in etfs_sma_dfs
                )
            ]

    etfs_sma_signal = {
        date: {
            etf: {
                "price": float(filtered_etfs_df.loc[date, etf]),
                **{
                    f"sma_{window}": float(etfs_sma_dfs[window_idx].loc[date, etf])
                    for window_idx, window in enumerate(sma_windows)
                },
            }
            for etf in date_signal
        }
        for date, date_signal in etfs_sma_signal.items()
    }

    etfs_signal = extract_common_detailed_signal(
        [etfs_volatility_signal, etfs_sma_signal]
    )

    holdings_volatility_windows = [holdings_volatility_window_size]

    holdings_vol_dates = [
        date.strftime("%Y-%m-%d") for date in filtered_holdings_df.index
    ]
    if holdings_first_valid_date not in holdings_vol_dates:
        raise ValueError("start date not in index")
    holdings_vol_dfs = [
        apply_function_to_data(
            df=filtered_holdings_df,
            function=calculate_relative_volatility_on_prices,
            returns_period=1,
            window_size=holdings_volatility_window_size,
            returns_method="percentage",
            ddof=0,
        )
    ]

    holdings_vol_first_valid_date = max(
        vol_df.index[0].strftime("%Y-%m-%d") for vol_df in holdings_vol_dfs
    )
    if holdings_vol_first_valid_date > holdings_first_valid_date:
        raise ValueError(
            f"not enough data to satisfy given start date ({holdings_first_valid_date}). Transformed data first valid date: {holdings_vol_first_valid_date})"
        )
    holdings_vol_dates = [
        date
        for date in holdings_vol_dates
        if pd.to_datetime(date) >= pd.to_datetime(holdings_first_valid_date)
    ]
    holdings_vol_dfs = [
        vol_df.loc[vol_df.index >= holdings_first_valid_date]
        for vol_df in holdings_vol_dfs
    ]

    holdings_volatility_signal = {}

    for date_idx, date in enumerate(holdings_vol_dates):
        if date_idx % freq == 0:

            current_active_holdings = find_current_filtered_active_holdings(
                date=date,
                interval_keyed_historical_holdings=interval_keyed_historical_holdings,
                holdings_df=filtered_holdings_df,
                sector_keyed_holdings=sector_keyed_holdings,
                sector_signal={
                    date: list(date_signal.keys())
                    for date, date_signal in etfs_signal.items()
                },
            )

            holdings_volatility_signal[date] = [
                holding
                for holding in current_active_holdings
                if all(
                    holding
                    in list(
                        dict(
                            sorted(
                                df[current_active_holdings].loc[date].to_dict().items(),
                                key=lambda items: items[1],
                                reverse=False,
                            )
                        ).keys()
                    )[:holdings_vol_n_top]
                    for df in holdings_vol_dfs
                )
            ]
    holdings_volatility_signal = {
        date: {
            holding: {
                "price": float(filtered_holdings_df.loc[date, holding]),
                **{
                    f"volatility_{window}": float(
                        holdings_vol_dfs[window_idx].loc[date, holding]
                    )
                    for window_idx, window in enumerate(holdings_volatility_windows)
                },
            }
            for holding in date_signal
        }
        for date, date_signal in holdings_volatility_signal.items()
    }

    holdings_long_sma_dates = [
        date.strftime("%Y-%m-%d") for date in filtered_holdings_df.index
    ]
    if holdings_first_valid_date not in holdings_long_sma_dates:
        raise ValueError("start date not in index")
    holdings_long_sma_dfs = [
        apply_function_to_data(
            df=filtered_holdings_df,
            function=calculate_simple_moving_average,
            window_size=holdings_long_sma_window_size,
        )
    ]

    holdings_long_sma_first_valid_date = max(
        sma_df.index[0].strftime("%Y-%m-%d") for sma_df in holdings_long_sma_dfs
    )
    if holdings_long_sma_first_valid_date > holdings_first_valid_date:
        raise ValueError(
            f"not enough data to satisfy given start date ({holdings_first_valid_date}). Transformed data first valid date: {holdings_long_sma_first_valid_date})"
        )
    holdings_long_sma_dates = [
        date
        for date in holdings_long_sma_dates
        if pd.to_datetime(date) >= pd.to_datetime(holdings_first_valid_date)
    ]

    holdings_long_sma_signal = {}

    holdings_long_sma_condition = lambda price, sma: price < sma

    for date_idx, date in enumerate(holdings_long_sma_dates):
        if date_idx % freq == 0:
            current_active_holdings = list(holdings_volatility_signal[date].keys())
            holdings_long_sma_signal[date] = [
                holding
                for holding in current_active_holdings
                if all(
                    holdings_long_sma_condition(
                        filtered_holdings_df[holding].loc[date],
                        indicator[holding].loc[date],
                    )
                    for indicator in holdings_long_sma_dfs
                )
            ]

    holdings_long_sma_signal = {
        date: {
            holding: {
                "price": float(filtered_holdings_df.loc[date, holding]),
                **{
                    f"sma_{window}": float(
                        holdings_long_sma_dfs[window_idx].loc[date, holding]
                    )
                    for window_idx, window in enumerate([holdings_long_sma_window_size])
                },
            }
            for holding in date_signal
        }
        for date, date_signal in holdings_long_sma_signal.items()
    }

    holdings_short_sma_dates = [
        date.strftime("%Y-%m-%d") for date in filtered_holdings_df.index
    ]
    if holdings_first_valid_date not in holdings_short_sma_dates:
        raise ValueError("start date not in index")
    holdings_short_sma_dfs = [
        apply_function_to_data(
            df=filtered_holdings_df,
            function=calculate_simple_moving_average,
            window_size=holdings_short_sma_window_size,
        )
    ]

    holdings_short_sma_first_valid_date = max(
        sma_df.index[0].strftime("%Y-%m-%d") for sma_df in holdings_short_sma_dfs
    )
    if holdings_short_sma_first_valid_date > holdings_first_valid_date:
        raise ValueError(
            f"not enough data to satisfy given start date ({holdings_first_valid_date}). Transformed data first valid date: {holdings_short_sma_first_valid_date})"
        )
    holdings_short_sma_dates = [
        date
        for date in holdings_short_sma_dates
        if pd.to_datetime(date) >= pd.to_datetime(holdings_first_valid_date)
    ]

    holdings_short_sma_signal = {}

    holdings_short_sma_condition = lambda price, sma: price >= sma

    for date_idx, date in enumerate(holdings_short_sma_dates):
        if date_idx % freq == 0:
            current_active_holdings = list(holdings_long_sma_signal[date].keys())
            holdings_short_sma_signal[date] = [
                holding
                for holding in current_active_holdings
                if all(
                    holdings_short_sma_condition(
                        filtered_holdings_df[holding].loc[date],
                        indicator[holding].loc[date],
                    )
                    for indicator in holdings_short_sma_dfs
                )
            ]

    holdings_short_sma_signal = {
        date: {
            holding: {
                "price": float(filtered_holdings_df.loc[date, holding]),
                **{
                    f"sma_{window}": float(
                        holdings_short_sma_dfs[window_idx].loc[date, holding]
                    )
                    for window_idx, window in enumerate([holdings_long_sma_window_size])
                },
            }
            for holding in date_signal
        }
        for date, date_signal in holdings_short_sma_signal.items()
    }

    holdings_signal = {k: list(v.keys()) for k, v in holdings_short_sma_signal.items()}
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
