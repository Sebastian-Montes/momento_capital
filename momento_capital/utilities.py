import pandas as pd


def apply_function_by_groups(df, func):
    groups = split_df_by_nan_config(df)
    result = [func(group) for group in groups]
    return pd.concat(result, axis=1)


def split_df_by_nan_config(df):

    # Create a dictionary to store the NaN pattern for each column
    nan_config = {}
    for column_idx, column in enumerate(df.columns):
        nan_config[column] = df.isna().iloc[:, column_idx].tolist()

    # Group columns by their NaN patterns
    grouped = {}
    for key, value in nan_config.items():
        value_tuple = tuple(value)  # Convert the NaN pattern to a tuple for hashing
        if value_tuple not in grouped:
            grouped[value_tuple] = []
        grouped[value_tuple].append(key)

    # Extract groups of columns and drop NaN values
    grouped_columns = list(grouped.values())
    groups = [df[current_group].dropna().copy() for current_group in grouped_columns]
    groups = [group for group in groups if not group.empty]

    return groups


def func_by_groups(group, func, *args, **kwargs):
    applied_func_array = func(array=group.values, *args, **kwargs)
    if len(applied_func_array) == 0:
        return pd.DataFrame()
    return pd.DataFrame(
        applied_func_array,
        index=group.index[-applied_func_array.shape[0] :],
        columns=group.columns,
    )


def find_current_filtered_active_holdings(
    date,
    interval_keyed_historical_holdings,
    holdings_df,
    sector_keyed_holdings,
    sector_signal,
):
    current_active_holdings = find_active_holdings(
        str_date=date,
        interval_keyed_historical_holdings=interval_keyed_historical_holdings,
    )
    current_active_holdings = [
        holding for holding in current_active_holdings if holding in holdings_df
    ]  ### Se filtra la lista para mantener unicamente los holdings que no se eliminaron al aplicar clean_df
    current_active_holdings = [
        holding
        for holding in current_active_holdings
        if any(
            holding in sector_keyed_holdings[sector] for sector in sector_signal[date]
        )
    ]
    return current_active_holdings


def find_active_holdings(str_date, interval_keyed_historical_holdings):
    active_interval = find_active_interval(
        str_date=str_date,
        interval_keyed_historical_holdings=interval_keyed_historical_holdings,
    )
    active_holdings = interval_keyed_historical_holdings[active_interval]
    return active_holdings


def find_active_interval(str_date, interval_keyed_historical_holdings):
    for interval in interval_keyed_historical_holdings.keys():
        str_interval_start_date, str_interval_end_date = interval.split("/")
        if str_interval_end_date == "--":
            break
        elif (
            pd.to_datetime(str_interval_start_date)
            <= pd.to_datetime(str_date)
            <= pd.to_datetime(str_interval_end_date)
        ):
            break
    return interval


def get_next_closest_date(target_date, date_list):
    target_date = pd.to_datetime(target_date)
    date_series = pd.to_datetime(pd.Series(date_list))

    # Filtrar solo las fechas posteriores o iguales
    next_dates = date_series[date_series >= target_date]

    if next_dates.empty:
        return None  # No hay fecha posterior

    # Retornar la fecha mínima de las posteriores
    return next_dates.min().strftime("%Y-%m-%d")


def apply_function_to_data(df, function, *args, **kwargs):
    if int(df.isna().sum().sum()) > 0:
        transformed_df = apply_function_by_groups(
            df=df,
            func=lambda group: func_by_groups(
                group=group, func=function, *args, **kwargs
            ),
        )
    else:
        transformed_array = function(array=df.values, *args, **kwargs)
        transformed_df = pd.DataFrame(
            data=transformed_array,
            index=df.index[-transformed_array.shape[0] :],
            columns=df.columns,
        )
    return transformed_df


def find_active_etfs(
    str_date, interval_keyed_historical_holdings, sector_keyed_holdings
):
    active_holdings = find_active_holdings(
        str_date=str_date,
        interval_keyed_historical_holdings=interval_keyed_historical_holdings,
    )
    active_etfs = [
        etf
        for etf in sector_keyed_holdings
        if any(holding in sector_keyed_holdings[etf] for holding in active_holdings)
    ]
    return active_etfs


def find_active_holdings(str_date, interval_keyed_historical_holdings):
    active_interval = find_active_interval(
        str_date=str_date,
        interval_keyed_historical_holdings=interval_keyed_historical_holdings,
    )
    active_holdings = interval_keyed_historical_holdings[active_interval]
    return active_holdings


def find_active_interval(str_date, interval_keyed_historical_holdings):
    for interval in interval_keyed_historical_holdings.keys():
        str_interval_start_date, str_interval_end_date = interval.split("/")
        if str_interval_end_date == "--":
            break
        elif (
            pd.to_datetime(str_interval_start_date)
            <= pd.to_datetime(str_date)
            <= pd.to_datetime(str_interval_end_date)
        ):
            break
    return interval


def check_for_signal_keys(signals):
    reference_keys = set(signals[0].keys())
    condition = all(set(d.keys()) == reference_keys for d in signals)
    if not condition:
        raise ValueError("Keys must be the same over all dictionaries")


def extract_common_detailed_signal(signals):
    check_for_signal_keys(signals)
    common_signals = {}
    for date in signals[0].keys():
        date_unique_holdings = list(
            {holding for signal in signals for holding in signal[date].keys()}
        )
        common_signals[date] = [
            holding
            for holding in date_unique_holdings
            if all(holding in signal[date] for signal in signals)
        ]
    joined_signals = {
        date: {
            holding: {
                k: v for signal in signals for k, v in signal[date][holding].items()
            }
            for holding in date_signal
        }
        for date, date_signal in common_signals.items()
    }
    return joined_signals


def clean_signal(signal):
    """
    Filters a signal dictionary to include only entries from the first non-empty list of assets onward.

    Parameters:
    ----------
    signal : dict
        A dictionary where:
        - Keys are dates represented as strings (e.g., 'YYYY-MM-DD').
        - Values are lists of assets associated with the corresponding dates.

    Returns:
    -------
    dict
        A filtered dictionary that includes only the entries where the date is
        greater than or equal to the first date with a non-empty list of assets.

    Example:
    --------
    >>> signal = {
    ...     "2025-01-01": [],
    ...     "2025-01-02": [],
    ...     "2025-01-03": ["AAPL", "TSLA"],
    ...     "2025-01-04": ["GOOGL"],
    ... }
    >>> clean_signal(signal)
    {'2025-01-03': ['AAPL', 'TSLA'], '2025-01-04': ['GOOGL']}

    Notes:
    ------
    - If all values in the dictionary are empty lists, the function will return an empty dictionary.
    - Dates are converted to pandas datetime objects for robust comparison.
    """
    if all(len(s) == 0 for s in signal.values()):
        print("Empty signal")
        return signal
    # if len([v for vs in signal.values() for v in vs]) == 0:
    #     raise ValueError("Empty signal")
    # Identify the first date with a non-empty list of assets
    for date, assets in signal.items():
        if len(assets) != 0:
            starting_date = date  # Save the starting date
            break

    # Filter the dictionary to include only entries from the starting date onward
    cleaned_signal = {
        date: assets
        for date, assets in signal.items()
        if pd.to_datetime(date) >= pd.to_datetime(starting_date)
    }

    return cleaned_signal


def forward_fill_until_last_value(df):
    """
    Aplica forward-fill en un DataFrame, pero solo hasta el último valor no nulo en cada columna.
    Los NaNs al final (sin valores posteriores) no son rellenados.
    """
    result = df.copy()
    for col in result.columns:
        # Crear máscara que detecta si hay valores válidos en adelante (incluyendo el actual)
        mask = result[col][::-1].notna().cumsum()[::-1] > 0

        # Aplicar ffill solo a los NaNs
        result[col] = result[col].where(~result[col].isna(), result[col].ffill())

        # Anular los valores rellenados más allá del último dato no-nulo
        result[col] = result[col].where(mask)

    return result


def preprocess_data(filtered_df, start_date, end_date, max_window):
    filtered_df = filtered_df.loc[filtered_df.index <= end_date].copy()

    filtered_df_dates = [date.strftime("%Y-%m-%d") for date in filtered_df.index]

    first_valid_date = get_next_closest_date(
        target_date=start_date, date_list=filtered_df_dates
    )
    not_enough_columns = [
        c
        for c in (filtered_df.notna().sum() <= max_window)
        .loc[filtered_df.notna().sum() <= max_window]
        .index.tolist()
    ]
    filtered_df.drop(not_enough_columns, axis=1, inplace=True)
    filtered_df.dropna(how="all", axis=1, inplace=True)
    return filtered_df, first_valid_date


def process_data(df, start_date, max_window):
    processed_df = df.copy()
    indices = df.index.astype(str).tolist()
    if indices.index(start_date) < max_window + 1:
        raise ValueError(
            f"not enough past dates for given start date and max window. \ndates before starting date: {indices.index(start_date)} max window: {max_window}"
        )
    processed_df = processed_df.loc[
        processed_df.index >= indices[indices.index(start_date) - max_window + 1]
    ]
    processed_df.dropna(how="all", axis=1, inplace=True)
    not_enough_columns = [
        col
        for col in processed_df.columns.tolist()
        if processed_df[col].notna().sum() <= max_window
    ]
    processed_df.drop(columns=not_enough_columns, inplace=True)
    return processed_df


def remove_almost_full_nan_rows(df, significance_percectage=0.95):
    nans_per_row = {
        idx: len(df.loc[idx].isna().loc[df.loc[idx].isna()])
        for idx in df.index.tolist()
    }
    rows_to_remove = [
        idx
        for idx, nan_count in nans_per_row.items()
        if nan_count >= df.shape[1] * significance_percectage
    ]
    filtered_df = df.copy()
    filtered_df.drop(rows_to_remove, inplace=True)
    return filtered_df


def get_next_closest_date(target_date, date_list):
    target_date = pd.to_datetime(target_date)
    date_series = pd.to_datetime(pd.Series(date_list))
    next_dates = date_series[date_series >= target_date]

    if next_dates.empty:
        return None
    return next_dates.min().strftime("%Y-%m-%d")


def filter_historical_holdings(
    historical_holdings, sectors_holdings, since, nearest_to_months
):
    if not any(isinstance(i, int) for i in nearest_to_months):
        raise ValueError("List elements of nearest_to_months must be integers")
    if any(i > 12 or i < 1 for i in nearest_to_months):
        raise ValueError("List elements of nearest_to_months must be between 1 and 12")

    filtered_spy_holdings = {}
    for interval, holds in historical_holdings.items():
        start_str, end_str = interval.split("/")
        end_dt = pd.Timestamp.today() if end_str == "--" else pd.to_datetime(end_str)
        if end_dt >= pd.to_datetime(since):  # keep intervals that overlap ‘since’
            filtered_spy_holdings[interval] = holds

    unique_rebalance_dates = pd.to_datetime(
        [interval[:10] for interval in filtered_spy_holdings.keys()]
    )
    last_date = max(unique_rebalance_dates)
    active_years = {pd.to_datetime(date).year for date in unique_rebalance_dates}
    target_dates = [
        f"{year}-{month_idx:02d}-01"
        for month_idx in nearest_to_months
        for year in active_years
    ]

    target_dates = [
        date.strftime("%Y-%m-%d")
        for date in pd.to_datetime(target_dates).sort_values()
        if date <= last_date
    ]

    filtered_dates = [
        min(
            [
                date
                for date in unique_rebalance_dates
                if date >= pd.to_datetime(target_date)
            ]
        )
        for target_date in target_dates
    ]
    filtered_dates = [date.strftime("%Y-%m-%d") for date in filtered_dates]

    filtered_historical_holdings = {
        interval: holdings
        for interval, holdings in historical_holdings.items()
        if interval[:10] in filtered_dates
    }

    unique_filtered_holdings = []
    for holdings in filtered_historical_holdings.values():
        for holding in holdings:
            if holding not in unique_filtered_holdings:
                unique_filtered_holdings.append(holding)

    filtered_sector_holdings = {
        sector: [
            holding
            for holding in sectors_holdings[sector]
            if holding in unique_filtered_holdings
        ]
        for sector in sectors_holdings.keys()
    }

    return filtered_historical_holdings, filtered_sector_holdings


def generate_window_dates_not_overlapping(start_date, end_date, n_windows):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_windows + 1)
    dates = dates.strftime("%Y-%m-%d").tolist()
    date_ranges = {
        f"window_{i+1}": {
            "start_date": dates[i],
            "end_date": (pd.to_datetime(dates[i + 1]) - pd.DateOffset(days=1)).strftime(
                "%Y-%m-%d"
            ),
        }
        for i in range(n_windows)
    }
    return date_ranges


def split_train_test_dates(start_date_str, end_date_str, test_proportion):
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    total_days = (end_date - start_date).days + 1
    test_days = int(round(total_days * test_proportion))
    if test_days < 1 and test_proportion > 0:
        test_days = 1
    if test_days > total_days:
        test_days = total_days
    test_start_date = end_date - pd.Timedelta(days=test_days - 1)
    train_end_date = test_start_date - pd.Timedelta(days=1)
    if test_proportion == 0:
        train_end_date = end_date
        test_start_date = None
    result = {
        "train": {
            "start_date": str(start_date.date()),
            "end_date": str(train_end_date.date()) if train_end_date else None,
        },
        "test": {
            "start_date": str(test_start_date.date()) if test_start_date else None,
            "end_date": str(end_date.date()) if test_start_date else None,
        },
    }

    return result


def spot_on(df, ending_string):

    filtered_df = df.copy()
    target_columns = [column for column in df if column.endswith(ending_string)]
    filtered_df = filtered_df[target_columns]
    filtered_df.columns = [column.removesuffix(ending_string) for column in filtered_df]
    return filtered_df


def format_datetime_df(df):
    formatted_df = df.copy()
    if ("date" in df.columns) and ("Date" not in df.columns):
        formatted_df["date"] = pd.to_datetime(formatted_df["date"])
        formatted_df.set_index("date", inplace=True)
        return formatted_df
    elif ("Date" in df.columns) and ("date" not in df.columns):
        formatted_df["Date"] = pd.to_datetime(formatted_df["Date"])
        formatted_df.set_index("Date", inplace=True)
        return formatted_df
    else:
        raise ValueError("No date columns")
