from .portfolio import (
    PortfolioEvaluator,
    PortfolioSimulator,
    TrailingStopBollinger,
    TrailingStopSMA,
)
from .transformers import (
    calculate_relative_volatility_on_prices,
    calculate_simple_moving_average,
    calculate_rsi,
)
from .utilities import (
    find_current_filtered_active_holdings,
    process_data,
    apply_function_to_data,
    find_active_etfs,
    clean_signal,
    extract_common_detailed_signal,
)
import pandas as pd
from .checkster_utilities import *

from pandas.tseries.offsets import DateOffset


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


def checkster(
    df_s3,
    df,
    benchmark_series,
    filtered_historical_holdings,
    filtered_sector_holdings,
    holdings_adjusted_close,
    start_date,
    end_date,  # ← NEW  (pass a "YYYY‑MM‑DD" string or leave None)
    stocks_per_etf,
    marked_dates_per_year,
    rebalance_freq,
    n_top,
    roc_stddev_period,
):

    def calculate_roc(df, period):
        return df["adjusted_close"].pct_change(periods=period) * 100

    def calculate_stddev(df, period):
        returns = df["adjusted_close"].pct_change()
        return returns.rolling(window=period).std(ddof=1) * 100

    def calculate_roc_stddev_ratio(df, period):
        df = df.copy()
        df["ROC"] = calculate_roc(df, period)
        df["STDDEV"] = calculate_stddev(df, period)
        df["ROC_STDDEV_Ratio"] = df["ROC"] / df["STDDEV"]
        return df

    def rank_top_10_by_etf(df, etf_column, metric, k=50):
        """
        Return the top‑`k` rows (by `metric`) for every ETF on every date.
        The wider depth gives your back‑fill logic more candidates to reach
        stocks_per_etf * n_top after filtering inactive tickers.

        Parameters
        ----------
        df : pandas.DataFrame
            Long‑form price/metric data.
        etf_column : str
            Column holding ETF identifiers (e.g. 'ETF 1').
        metric : str
            Column to rank on (e.g. 'ROC_STDDEV_Ratio').
        k : int, default 30
            How many rows per ETF‑date to keep.
        """
        ranked_data = []

        for etf in df[etf_column].unique():
            etf_data = df[df[etf_column] == etf]
            for date, daily_data in etf_data.groupby("date"):
                daily_top = daily_data.nlargest(k, metric)
                if not daily_top.empty:
                    daily_top = daily_top[["ticker", metric, "date"]]
                    daily_top["Rank"] = range(1, len(daily_top) + 1)
                    daily_top["ETF"] = etf
                    ranked_data.append(daily_top)

        return pd.concat(ranked_data, axis=0)

    def pivot_ranked_data_by_etf(df, etf):
        etf_data = df[df["ETF"] == etf]
        return etf_data.pivot(index="Rank", columns="date", values="ticker")

    def get_fixed_marked_dates(df, marked_dates_per_year):
        df.index = pd.to_datetime(df.index)
        df_sorted = df.sort_index()
        unique_years = df_sorted.index.year.unique()
        marked_dates = []

        for year in unique_years:
            start_date = pd.Timestamp(f"{year}-01-01")
            interval = 12 // marked_dates_per_year

            for i in range(marked_dates_per_year):
                boundary = start_date + DateOffset(months=i * interval)
                valid_subset = df_sorted.index[df_sorted.index >= boundary]
                if len(valid_subset) > 0:
                    first_valid_after_boundary = valid_subset[0]
                    marked_dates.append(first_valid_after_boundary)

        return pd.to_datetime(sorted(set(marked_dates)))

    # Main logic of checkster
    marked_dates = get_fixed_marked_dates(df, marked_dates_per_year)
    df.index = pd.to_datetime(df.index)
    # dates = [
    #     d
    #     for d in df.index.strftime("%Y-%m-%d").tolist()
    #     if d >= start_date and (end_date is None or d <= end_date)
    # ]
    # Support warm-up logic to allow building active holdings from earlier data ----------------
    true_start_date = start_date
    warmup_start_date = "2022-01-01"

    all_dates = df.index.strftime("%Y-%m-%d").tolist()
    dates = [
        d
        for d in all_dates
        if d >= warmup_start_date and (end_date is None or d <= end_date)
    ]

    active_holdings_list = [
        extract_active_holdings(
            date, filtered_historical_holdings, filtered_sector_holdings
        )
        for date in dates
    ]

    active_holdings_cache = dict(zip(dates, active_holdings_list))
    # ------------------------------------------------------------------------------------
    # === NEW forward‑filled cache =============================================
    active_holdings_cache = {}
    last_valid = None

    for date in dates:  # dates is already sorted chronologically
        holdings = extract_active_holdings(
            date, filtered_historical_holdings, filtered_sector_holdings
        )

        # if extract_active_holdings() returns [] or {}, reuse last_valid
        if not holdings:  # [] or empty dict / None
            holdings = last_valid
        else:
            last_valid = holdings  # update anchor when we get real data

        active_holdings_cache[date] = holdings or {}  # guarantee a dict
    # ===========================================================================

    rebalance_dates = [dates[i] for i in range(len(dates)) if i % rebalance_freq == 0]
    combined_rebalance_dates = sorted(
        set(rebalance_dates).union(md.strftime("%Y-%m-%d") for md in marked_dates)
    )

    def process_dates_chunk(
        dates_chunk, df, symbols, marked_dates, active_holdings_cache, n_top
    ):
        local_etfs_return_cache = {}
        local_top_etfs = {}

        for date in dates_chunk:
            current_date = pd.to_datetime(date)

            if date not in active_holdings_cache:
                print(
                    f"Warning: {date} not found in active_holdings_cache. Skipping..."
                )
                continue

            if current_date in marked_dates:
                past_marked_dates = marked_dates[marked_dates < current_date]
            else:
                past_marked_dates = marked_dates[marked_dates <= current_date]

            if past_marked_dates.empty:
                print(f"❌ ERROR: No valid past rebalance date for {date}. Skipping...")
                continue

            marked_date = past_marked_dates.max()

            if marked_date not in df.index:
                print(
                    f"❌ ERROR: Marked date {marked_date} missing in DataFrame. Skipping..."
                )
                continue

            if df.loc[marked_date, symbols].isnull().any():
                print(
                    f"⚠️ Warning: NaN in ETF prices on {marked_date}. Skipping return."
                )
                continue

            if date not in local_etfs_return_cache:
                local_etfs_return_cache[date] = (
                    df.loc[date, symbols] / df.loc[marked_date, symbols]
                ) - 1

            sorted_returns = local_etfs_return_cache[date].sort_values(ascending=False)[
                :n_top
            ]
            local_top_etfs[date] = sorted_returns.index.tolist()

        return local_etfs_return_cache, local_top_etfs

    import math

    symbols = [
        "XLK",
        "XLC",
        "XLV",
        "XLF",
        "XLP",
        "XLI",
        "XLE",
        "XLY",
        "XLB",
        "XLU",
        "XLRE",
    ]

    # --- sin paralelizacion ------------------
    etfs_return_cache = {}
    top_etfs = {}

    # we can reuse the same chunk logic if you like, but a single call works too
    partial_return_cache, partial_top_etfs = process_dates_chunk(
        dates,  # <- just pass the full list
        df,
        symbols,
        marked_dates,
        active_holdings_cache,
        n_top,
    )

    etfs_return_cache.update(partial_return_cache)
    top_etfs.update(partial_top_etfs)
    # ---------------------------------------------------------------------------

    # print("Marked dates:\n", marked_dates)

    top_ranked_sectors = {
        date: etfs
        for date, etfs in top_etfs.items()
        if date in combined_rebalance_dates
    }
    top_ranked_sectors = pd.DataFrame(top_ranked_sectors)

    # Now do your parallel on df_s3
    df_s3 = df_s3[df_s3["ETF 1"].isin(symbols)].copy()
    df_s3["date"] = pd.to_datetime(df_s3["date"])
    df_s3["adjusted_close"] = pd.to_numeric(df_s3["adjusted_close"], errors="coerce")
    df_s3.dropna(subset=["adjusted_close", "ticker", "date"], inplace=True)

    results = [
        calculate_roc_stddev_ratio(df_s3[df_s3["ticker"] == ticker], roc_stddev_period)
        for ticker in df_s3["ticker"].unique()
    ]

    df_s3 = pd.concat(results, ignore_index=True)
    df_s3.dropna(subset=["ROC_STDDEV_Ratio"], inplace=True)

    ranked_data = rank_top_10_by_etf(df_s3, "ETF 1", "ROC_STDDEV_Ratio")
    pivoted_data = {etf: pivot_ranked_data_by_etf(ranked_data, etf) for etf in symbols}

    # ------------------ build fixed‑length daily lists -------------------------
    # target_per_day = stocks_per_etf * n_top
    # top_stocks_data = {}

    # for date in top_ranked_sectors.columns:
    target_per_day = stocks_per_etf * n_top
    top_stocks_data = {}

    for date in top_ranked_sectors.columns:
        if date < true_start_date:
            continue  # skip pre-simulation signals

        date_str = str(date)
        daily_picks = []
        # ---------------------------------------
        actives_today = active_holdings_cache.get(date_str, {}) or {}

        # Pass 1: normal top‑rank loop, filtered by active set
        for etf in top_ranked_sectors[date]:
            if etf not in pivoted_data or date not in pivoted_data[etf].columns:
                continue

            col = pivoted_data[etf][date].dropna()
            active_col = [t for t in col if t in actives_today.get(etf, [])]

            daily_picks.extend(active_col[:stocks_per_etf])

        # Pass 2: if we came up short, keep walking down the same ETF columns
        if len(daily_picks) < target_per_day:
            for etf in top_ranked_sectors[date]:
                if etf not in pivoted_data or date not in pivoted_data[etf].columns:
                    continue

                col = pivoted_data[etf][date].dropna()
                active_col = [t for t in col if t in actives_today.get(etf, [])]

                needed = target_per_day - len(daily_picks)
                daily_picks.extend(active_col[stocks_per_etf : stocks_per_etf + needed])

                if len(daily_picks) >= target_per_day:
                    break

        # Fail‑safe: still short? raise an explicit error so you know
        if len(daily_picks) < target_per_day:
            raise ValueError(
                f"Could only find {len(daily_picks)} active tickers on {date_str}; "
                "check holdings data or widen rank search."
            )

        top_stocks_data[date_str] = daily_picks[:target_per_day]  # exact length
    # --------------------------------------------------------------------------

    for date, ticker_list in top_stocks_data.items():
        cleaned_list = []
        for t in ticker_list:
            # We check coverage in df_s3 for (ticker == t & date == date).
            # If that row is missing or has a NaN 'adjusted_close', skip it.
            row_mask = (df_s3["ticker"] == t) & (df_s3["date"] == pd.to_datetime(date))

            # df_s3 might have 1 row or 0 rows for that combination
            this_row = df_s3.loc[row_mask]

            if not this_row.empty:
                # check if 'adjusted_close' is non-NaN
                if this_row["adjusted_close"].notna().all():
                    # it has coverage => keep it
                    cleaned_list.append(t)
                else:
                    print(f"Skipping {t} on {date} (NaN coverage).")
            else:
                print(f"Skipping {t} on {date} (no row in df_s3).")

        # Overwrite the old list with the cleaned version
        top_stocks_data[date] = cleaned_list

    # top_stocks_df = pd.DataFrame.from_dict(top_stocks_data, orient="index").transpose()
    # top_stocks_df.index = range(1, (stocks_per_etf * n_top) + 1)
    # top_stocks_df.index.name = "Rank"

    # final_dictionary = {
    #    date: list(stocks.values()) for date, stocks in top_stocks_df.to_dict().items()
    # }

    final_dictionary = {date: lst for date, lst in top_stocks_data.items()}

    # Build simulation_df from holdings_adjusted_close using the original price data.
    simulation_df = holdings_adjusted_close.copy()
    simulation_df["Date"] = pd.to_datetime(simulation_df["Date"], errors="coerce")
    simulation_df.set_index("Date", inplace=True)
    simulation_df.sort_index(inplace=True)

    # Gather all tickers that are signaled (i.e. appear in final_dictionary)
    all_portfolio_tickers = set()
    for date_str, tlist in final_dictionary.items():
        all_portfolio_tickers.update(tlist)

    # Restrict simulation_df to only these columns (tickers)
    columns_to_keep = sorted(all_portfolio_tickers.intersection(simulation_df.columns))
    simulation_df = simulation_df[columns_to_keep]

    # Drop rows that are entirely NaN (e.g. holidays or inactive days)
    simulation_df.dropna(how="all", inplace=True)

    evaluator = PortfolioEvaluator(benchmark_series=benchmark_series)
    portfolio = PortfolioSimulator(
        initial_cash=100000,
        target_weight=1,
        df=simulation_df,
        id_structure="1111",
        manager=None,
        evaluator=evaluator,
        seed=1,
        verbose=0,
        portfolio_id="prueba",
    )

    portfolio.simulate(signals=final_dictionary)

    return portfolio
