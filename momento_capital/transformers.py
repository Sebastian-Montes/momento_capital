import numpy as np


def max_drawdown(equity_curve):
    equity_curve = np.asarray(equity_curve)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = np.min(drawdowns)
    return max_dd
