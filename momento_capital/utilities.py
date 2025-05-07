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


def process_data(filtered_df, start_date, end_date, freq, max_window):
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
    return filtered_df, first_valid_date


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
