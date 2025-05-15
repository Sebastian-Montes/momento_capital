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


def calculate_rsi(array, window_size):
    if not isinstance(array, np.ndarray):
        raise ValueError("Input object must be an array")
    if array.ndim != 2:
        raise ValueError("Array must be bidimensional")
    if np.isnan(array).any():
        raise ValueError("Array contains NaNs")
    delta = np.diff(array, axis=0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(array)
    avg_loss = np.zeros_like(array)
    avg_gain[window_size] = np.mean(gain[:window_size], axis=0)
    avg_loss[window_size] = np.mean(loss[:window_size], axis=0)
    for i in range(window_size + 1, array.shape[0]):
        avg_gain[i] = (avg_gain[i - 1] * (window_size - 1) + gain[i - 1]) / window_size
        avg_loss[i] = (avg_loss[i - 1] * (window_size - 1) + loss[i - 1]) / window_size
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi[window_size:]

    return rsi


def calculate_rolling_corr(A, B, window_size):
    if A.shape[0] != B.shape[0]:
        raise ValueError("size at axis 0 must be the same for input A and B")
    unfolded_A = np.lib.stride_tricks.sliding_window_view(
        A, window_shape=window_size, axis=0
    )
    unfolded_B = np.lib.stride_tricks.sliding_window_view(
        B, window_shape=window_size, axis=0
    )
    corr_tensor = np.zeros((B.shape[1], A.shape[1], unfolded_A.shape[0]))
    for window in range(unfolded_A.shape[0]):
        corr_tensor[..., window] = calculate_pearson_correlation(
            A=unfolded_A[window].T, B=unfolded_B[window].T
        )
    return corr_tensor


def calculate_pearson_correlation(A, B):
    A_mean = np.mean(A, axis=0)
    B_mean = np.mean(B, axis=0)
    A_mean = np.repeat(np.expand_dims(A_mean, axis=1), repeats=A.shape[0], axis=1).T
    B_mean = np.repeat(np.expand_dims(B_mean, axis=1), repeats=B.shape[0], axis=1).T
    displaced_A = A - A_mean
    displaced_B = B - B_mean
    displaced_A = np.repeat(
        np.expand_dims(displaced_A, axis=1), repeats=B.shape[1], axis=1
    )
    displaced_B = np.repeat(
        np.expand_dims(displaced_B, axis=2), repeats=A.shape[1], axis=2
    )
    numerator = np.sum(displaced_A * displaced_B, axis=0)
    denominator = np.sqrt(np.sum(displaced_A**2, axis=0)) * np.sqrt(
        np.sum(displaced_B**2, axis=0)
    )
    return numerator / denominator
