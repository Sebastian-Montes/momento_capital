import numpy as np


def max_drawdown(equity_curve):
    equity_curve = np.asarray(equity_curve)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = np.min(drawdowns)
    return max_dd


def calculate_returns(array, period):
    if not isinstance(array, np.ndarray):
        raise ValueError("Input object must be an array")
    if array.ndim != 2:
        raise ValueError("Array must be bidimensional")
    if np.isnan(array).any():
        raise ValueError("Array contains NaNs")
    transformed_array = (array[period:] / array[:-period]) - 1
    return np.array(transformed_array)


def calculate_log_returns(array, period):
    if not isinstance(array, np.ndarray):
        raise ValueError("Input object must be an array")
    if array.ndim != 2:
        raise ValueError("Array must be bidimensional")
    if np.isnan(array).any():
        raise ValueError("Array contains NaNs")
    log_returns_array = np.diff(np.log(array), axis=0)
    return log_returns_array


def calculate_relative_volatility_on_prices(
    array,
    returns_period,
    window_size,
    returns_method="percentage",
    ddof=0,
):
    if not isinstance(array, np.ndarray):
        raise ValueError("Input object must be an array")
    if array.ndim != 2:
        raise ValueError("Array must be bidimensional")
    if np.isnan(array).any():
        raise ValueError("Array contains NaNs")
    if returns_method == "percentage":
        returns_array = calculate_returns(array=array, period=returns_period)
    elif returns_method == "logarithmic":
        returns_array = calculate_log_returns(array=array, period=returns_period)
    else:
        raise ValueError("returns_method must be either 'percentage' or 'logarithmic'.")
    returns_windows = np.lib.stride_tricks.sliding_window_view(
        x=returns_array, window_shape=window_size, axis=0
    )
    volatility_array = np.std(returns_windows, axis=2, ddof=ddof)
    return volatility_array


def calculate_simple_moving_average(array, window_size):

    if not isinstance(array, np.ndarray):
        raise ValueError("Input object must be an array")
    if array.ndim != 2:
        raise ValueError("Array must be bidimensional")
    if np.isnan(array).any():
        raise ValueError("Array contains NaNs")
    transformed_values = np.mean(
        np.lib.stride_tricks.sliding_window_view(
            x=array, axis=0, window_shape=window_size
        ),
        axis=2,
    )
    return transformed_values


def standardize(array, mean, std):
    if not isinstance(array, np.ndarray):
        raise ValueError("Input object must be an array")
    if array.ndim != 2:
        raise ValueError("Array must be bidimensional")
    if np.isnan(array).any():
        raise ValueError("Array contains NaNs")
    standardized_array = (array - mean) / std
    return standardized_array


def calculate_lower_bb(array, window_size, bollinger_factor):
    sma_array = calculate_simple_moving_average(array=array, window_size=window_size)
    rolling_std_array = np.std(
        np.lib.stride_tricks.sliding_window_view(
            x=array, axis=0, window_shape=window_size
        ),
        axis=2,
    )
    lower_bollinger_band_array = sma_array - bollinger_factor * rolling_std_array
    return lower_bollinger_band_array

