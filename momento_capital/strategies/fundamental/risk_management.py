import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statistics
from .utilities import (
    compute_metric_growth,
    compute_qoq_std,
    compute_yy_qq_ratio,
    filtro_para_miembros_de_spy,
    processed_metric_with_quarter_start_date_based_on_filing_date,
    genera_watchlist,
    target_range_zscore,
    compute_rolling_zscore_fixed_window,
    compute_rolling_zscore_expanding,
)


class EarningSurprise_Percentage:
    def __init__(
        self,
        methodology: str,
        zscore_fixed_window_size: int,
        lowerzscore_limit: float,
        upperzscore_limit: float,
        time_period_on_watchlist_days: int,
        # New parameters for pre-processed data
        eps_data: dict = None,
        report_date_per_ticker_per_quarter: dict = None,
        tickers_to_check: list = None,
        quarter_start_date_EPS_data: pd.DataFrame = None,
        # New optional parameter for pruning
        simulation_start_date: str = None,
    ):

        self.methodology = methodology
        self.zscore_fixed_window_size = zscore_fixed_window_size  # in years
        self.lowerzscore_limit = lowerzscore_limit
        self.upperzscore_limit = upperzscore_limit
        self.time_period_on_watchlist_days = time_period_on_watchlist_days
        # Data containers from pre-processed data
        self.eps_data = eps_data
        self.tickers_to_check = tickers_to_check
        self.report_date_per_ticker_per_quarter = report_date_per_ticker_per_quarter
        self.quarter_start_date_EPS_data = quarter_start_date_EPS_data
        self.simulation_start_date = simulation_start_date

        # Add cache attributes
        self.watchlist = self.generate_watchlist_for_each_date_based_on_report_date()

    def target_range_zscore(self):

        if self.methodology == "fw":
            # Get sorted dates for windowing
            zscores = dict(
                sorted(
                    compute_rolling_zscore_fixed_window(
                        self.quarter_start_date_EPS_data, self.zscore_fixed_window_size
                    ).items()
                )
            )
        else:
            zscores = dict(
                sorted(
                    compute_rolling_zscore_expanding(
                        self.quarter_start_date_EPS_data
                    ).items()
                )
            )

        tickers_with_target_range_zscore_on_current_date = {}

        for date, ticker_dict in zscores.items():
            outliers = []
            for ticker, zscore in ticker_dict.items():
                if (
                    pd.notna(zscore)
                    and self.lowerzscore_limit <= zscore <= self.upperzscore_limit
                ):
                    outliers.append(ticker)
            if outliers:  # only include if there are tickers above the threshold
                tickers_with_target_range_zscore_on_current_date[date] = outliers

        return tickers_with_target_range_zscore_on_current_date

    def generate_watchlist_for_each_date_based_on_report_date(self):
        """
        Maps tickers from dict1 to their closest future dates from dict2,
        and keeps them in the result for a rolling window of specified days.

        Args:
            target_range_zscore (dict): A dictionary with quarter start dates as keys and lists of tickers as values.
                Example: {'2001-10-01': ['FDX', 'JWN', 'NTAP']}
            report_date_per_ticker_per_quarter (dict): A dictionary with tickers as keys and nested dictionaries of quarterly reporting dates.
                Example: {'VTRS': {2025: {'Q1': '2025-02-26', 'Q2': '2025-05-07'}, 2024: {...}}}
            time_period_on_watchlist_days (int): Number of days to keep a ticker in the result after its closest future date.
                Default is 252 (approximately 1 trading year).

        Returns:
            dict: A dictionary with daily dates as keys and lists of tickers as values, where each ticker
                appears from its closest future date through the specified window size.
        """
        target_range_zscore = self.target_range_zscore()

        # Create a flattened dictionary for quick lookup: {ticker: [sorted_dates]}
        ticker_report_dates = {}
        for ticker, years_data in self.report_date_per_ticker_per_quarter.items():
            dates = []
            for year, quarters in years_data.items():
                for quarter, date in quarters.items():
                    if date:  # Only add non-None dates
                        dates.append(date)
            # Sort the dates for each ticker
            ticker_report_dates[ticker] = sorted(dates)

        # De las fechas (quarter_start) en target_range_zscore, encuentra la fecha de reporte más cercana en el futuro -> ticker_report_dates
        ticker_closest_report_date = {}
        for quarter_start, tickers in target_range_zscore.items():
            for ticker in tickers:
                if ticker in ticker_report_dates:
                    future_dates = [
                        date
                        for date in ticker_report_dates[ticker]
                        if date > quarter_start
                    ]
                    if future_dates:
                        closest_future_date = future_dates[0]

                        # Store the mapping of ticker to its closest future date
                        if ticker not in ticker_closest_report_date:
                            ticker_closest_report_date[ticker] = []
                        ticker_closest_report_date[ticker].append(closest_future_date)

        # Find the earliest and latest dates to create our daily range
        all_dates = []
        for dates in ticker_report_dates.values():
            all_dates.extend(dates)

        # Also include the original quarter start dates
        # all_dates.extend(target_range_zscore.keys())

        dt_dates = [datetime.strptime(date, "%Y-%m-%d") for date in all_dates if date]
        earliest_date = min(dt_dates)
        latest_date = max(dt_dates) + timedelta(
            days=self.time_period_on_watchlist_days
        )  # Add window size to ensure all tickers complete their window

        # Genera las fechas diarias desde la minima hasta la maxima fecha
        current_date = earliest_date
        all_daily_dates = []
        while current_date <= latest_date:
            all_daily_dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)

        # Create the result dictionary with all daily dates
        result = {date: [] for date in all_daily_dates}

        # For each ticker and its closest future dates
        for ticker, closest_dates in ticker_closest_report_date.items():
            for closest_date in closest_dates:
                # Convert to datetime for easier date arithmetic
                dt_closest = datetime.strptime(closest_date, "%Y-%m-%d")

                # Add ticker to each date in its window
                for i in range(self.time_period_on_watchlist_days):
                    current_date = dt_closest + timedelta(days=i)
                    current_date_str = current_date.strftime("%Y-%m-%d")

                    # Only add if the date is in our result (in case it falls beyond latest_date)
                    if (
                        current_date_str in result
                        and ticker not in result[current_date_str]
                    ):
                        result[current_date_str].append(ticker)

        result = {date: sorted(tickers) for date, tickers in result.items()}

        self.watchlist = result

        return result

    def evaluate_risk(self, simulator, date):

        positions = list(simulator.positions.keys())

        if not positions:
            return {}, {}  # No positions to evaluate

        decision_dict = {}
        details_dict = {}

        # --- Get Relevant Signal Details --- Assume valid simulation_start_date and successful call
        relevant_signals = self.watchlist  # Uses default periods_prior=2

        # --- Evaluate Each Position ---
        for asset in positions:
            decision_dict[asset] = asset in relevant_signals[date]
            details_dict[asset] = (
                f"Hold: Ticker '{asset}' found in relevant signals."
                if decision_dict[asset]
                else f"Sell: Ticker '{asset}' not found in relevant signals."
            )

        return decision_dict, details_dict

    def evaluate_prospects(self, simulator, prospects, date):

        relevant_signals = self.watchlist

        decision_dict = {}
        details_dict = {}

        # --- Evaluate Each Prospect ---
        for asset in prospects:
            decision_dict[asset] = asset in relevant_signals[date]
            details_dict[asset] = (
                f"Buy: Ticker '{asset}' found in relevant signals."
                if decision_dict[asset]
                else f"Skip: Ticker '{asset}' not found in relevant signals."
            )

        return decision_dict, details_dict


class fundamentals_vs_spy:
    """
    A class for screening stocks based on fundamental metrics and Z-score analysis.

    This class analyzes financial data from companies using different methodologies
    to identify potential investment opportunities based on fundamental metrics.
    """

    def __init__(
        self,
        metric: str,
        window_size: int,
        methodology: str,
        lowerzscore_limit: float,
        upperzscore_limit: float,
        # New parameters for pre-processed data
        income_statements_or_cashflow_statements: dict = None,
        tickers_to_check: list = None,
        quarter_report_dates: dict = None,
        spy_holdings_by_quarter_start_date: dict = None,
        # New optional parameter for pruning
        simulation_start_date: str = None,
    ):  # Add new optional parameter

        self.metric = metric
        self.window_size = window_size  # in years
        self.quarterly_window_size = window_size * 4  # in quarters
        self.methodology = methodology
        self.lowerzscore_limit = lowerzscore_limit
        self.upperzscore_limit = upperzscore_limit
        self.simulation_start_date = simulation_start_date

        # Data containers from pre-processed data
        self.income_statements_or_cashflow_statements = (
            income_statements_or_cashflow_statements
        )
        self.tickers_to_check = tickers_to_check
        self.quarter_report_dates = quarter_report_dates
        self.spy_holdings_by_quarter_start_date = spy_holdings_by_quarter_start_date
        # Processed data containers
        self.watchlist = self.genera_watchlist()

    def compute_metric_growth(self):
        growth = compute_metric_growth(
            self.income_statements_or_cashflow_statements,
            self.metric,
            self.tickers_to_check,
            self.quarterly_window_size,
        )
        return growth

    def sequential_growth(self):

        seq_growth = compute_metric_growth(
            self.income_statements_or_cashflow_statements,
            self.metric,
            self.tickers_to_check,
            1,
        )
        return seq_growth

    def compute_qoq_std(self):
        seq_growth = self.sequential_growth()
        qoq_std = compute_qoq_std(seq_growth, self.quarterly_window_size)
        return qoq_std

    def compute_yy_qq_ratio(self):

        growth = self.compute_metric_growth()
        std = self.compute_qoq_std()
        yy_qq_ratio = compute_yy_qq_ratio(growth, std)

        return yy_qq_ratio

    def processed_metric_with_quarter_start_date_based_on_filing_date(self):
        # El quarter start date es basado en el filing date

        if self.methodology == "ROCSTD":
            processed_metric = self.compute_yy_qq_ratio()
        else:
            processed_metric = self.compute_metric_growth()

        processed_metric_with_quarter_start_date_based_on_filing_date1 = (
            processed_metric_with_quarter_start_date_based_on_filing_date(
                processed_metric, self.income_statements_or_cashflow_statements
            )
        )

        return processed_metric_with_quarter_start_date_based_on_filing_date1

    def filtro_para_miembros_de_spy(self):
        processed_metric_with_quarter_start_date_based_on_filing_date = (
            self.processed_metric_with_quarter_start_date_based_on_filing_date()
        )

        filtro_para_miembros_de_spy1 = filtro_para_miembros_de_spy(
            processed_metric_with_quarter_start_date_based_on_filing_date,
            self.spy_holdings_by_quarter_start_date,
        )

        return filtro_para_miembros_de_spy1

    def compute_z_scores(self):
        """
        Computes Z-score for each quarter based on YoY growth values.

        Args:
            df (pd.DataFrame): DataFrame with YoY growth values.

        Returns:
            pd.DataFrame: DataFrame with additional columns for mean, std deviation, and Z-scores.
        """
        df = self.filtro_para_miembros_de_spy()
        df = pd.DataFrame.from_dict(df, orient="columns")
        df = df.sort_index()
        # Compute mean and standard deviation for each row (quarter)
        df["Mean"] = df.mean(axis=1)
        df["StdDev"] = df.std(axis=1)

        # Compute Z-score for each ticker (excluding mean and std columns)
        z_scores = df.iloc[:, :-2].sub(df["Mean"], axis=0).div(df["StdDev"], axis=0)

        # Replace invalid Z-scores (when StdDev is NaN)
        z_scores = z_scores.where(df["StdDev"].notna())

        z_scores = z_scores.to_dict()
        return z_scores

    def target_range_zscore(self):

        target_range_zscore1 = target_range_zscore(
            self.compute_z_scores(), self.lowerzscore_limit, self.upperzscore_limit
        )

        return target_range_zscore1

    def genera_watchlist(self):

        watchlist = genera_watchlist(self.target_range_zscore())

        return watchlist

    def evaluate_risk(self, simulator, date):

        positions = list(simulator.positions.keys())

        if not positions:
            return {}, {}  # No positions to evaluate

        decision_dict = {}
        details_dict = {}

        # --- Get Relevant Signal Details --- Assume valid simulation_start_date and successful call
        relevant_signals = self.watchlist

        # --- Evaluate Each Position ---
        for asset in positions:
            decision_dict[asset] = asset in relevant_signals[date]
            details_dict[asset] = (
                f"Hold: Ticker '{asset}' found in relevant signals."
                if decision_dict[asset]
                else f"Sell: Ticker '{asset}' not found in relevant signals."
            )

        return decision_dict, details_dict

    def evaluate_prospects(self, simulator, prospects, date):

        relevant_signals = self.watchlist

        decision_dict = {}
        details_dict = {}

        # --- Evaluate Each Prospect ---
        for asset in prospects:
            decision_dict[asset] = asset in relevant_signals[date]
            details_dict[asset] = (
                f"Buy: Ticker '{asset}' found in relevant signals."
                if decision_dict[asset]
                else f"Skip: Ticker '{asset}' not found in relevant signals."
            )

        return decision_dict, details_dict


class fundamentals_vs_sector:
    def __init__(
        self,
        metric: str,
        window_size: int,
        methodology: str,
        lowerzscore_limit: float,
        upperzscore_limit: float,
        # New parameters for pre-processed data
        income_statements_or_cashflow_statements: dict = None,
        tickers_to_check: list = None,
        quarter_report_dates: dict = None,
        spy_holdings_by_quarter_start_date: dict = None,
        etfs_holdings: dict = None,
        # New optional parameter for pruning
        simulation_start_date: str = None,
    ):  # Add new optional parameter

        self.metric = metric
        self.window_size = window_size  # in years
        self.quarterly_window_size = window_size * 4  # in quarters
        self.methodology = methodology
        self.lowerzscore_limit = lowerzscore_limit
        self.upperzscore_limit = upperzscore_limit
        self.simulation_start_date = simulation_start_date

        # Data containers from pre-processed data
        self.income_statements_or_cashflow_statements = (
            income_statements_or_cashflow_statements
        )
        self.tickers_to_check = tickers_to_check
        self.quarter_report_dates = quarter_report_dates
        self.spy_holdings_by_quarter_start_date = spy_holdings_by_quarter_start_date
        self.etfs_holdings = etfs_holdings
        # Processed data containers
        self.watchlist = self.genera_watchlist()

    def compute_metric_growth(self):
        growth = compute_metric_growth(
            self.income_statements_or_cashflow_statements,
            self.metric,
            self.tickers_to_check,
            self.quarterly_window_size,
        )
        return growth

    def sequential_growth(self):

        seq_growth = compute_metric_growth(
            self.income_statements_or_cashflow_statements,
            self.metric,
            self.tickers_to_check,
            1,
        )
        return seq_growth

    def compute_qoq_std(self):
        seq_growth = self.sequential_growth()
        qoq_std = compute_qoq_std(seq_growth, self.quarterly_window_size)
        return qoq_std

    def compute_yy_qq_ratio(self):

        growth = self.compute_metric_growth()
        std = self.compute_qoq_std()
        yy_qq_ratio = compute_yy_qq_ratio(growth, std)

        return yy_qq_ratio

    def processed_metric_with_quarter_start_date_based_on_filing_date(self):
        # El quarter start date es basado en el filing date

        if self.methodology == "ROCSTD":
            processed_metric = self.compute_yy_qq_ratio()
        else:
            processed_metric = self.compute_metric_growth()

        processed_metric_with_quarter_start_date_based_on_filing_date1 = (
            processed_metric_with_quarter_start_date_based_on_filing_date(
                processed_metric, self.income_statements_or_cashflow_statements
            )
        )

        return processed_metric_with_quarter_start_date_based_on_filing_date1

    def filtro_para_miembros_de_spy(self):
        processed_metric_with_quarter_start_date_based_on_filing_date = (
            self.processed_metric_with_quarter_start_date_based_on_filing_date()
        )

        filtro_para_miembros_de_spy1 = filtro_para_miembros_de_spy(
            processed_metric_with_quarter_start_date_based_on_filing_date,
            self.spy_holdings_by_quarter_start_date,
        )

        return filtro_para_miembros_de_spy1

    def compute_sector_z_scores(self):

        growth_for_spy_holdings_by_date = self.filtro_para_miembros_de_spy()

        # Inicializar la estructura de salida
        sector_holdings_date_data = {}

        # Para cada sector y sus holdings
        for sector, holdings in self.etfs_holdings.items():
            # Recopilamos los datos por fecha para este sector
            for ticker in holdings:
                if (
                    ticker in growth_for_spy_holdings_by_date
                ):  # Solo procesamos si existe en prueba_func
                    if ticker not in sector_holdings_date_data:
                        sector_holdings_date_data[ticker] = {}

            # Para cada fecha, calculamos los z-scores
            all_dates = set()
            for ticker in holdings:
                if ticker in growth_for_spy_holdings_by_date:
                    all_dates.update(growth_for_spy_holdings_by_date[ticker].keys())

            for date in all_dates:
                # Recogemos los valores de crecimiento disponibles para este sector en esta fecha
                sector_growths = {}
                for ticker in holdings:
                    if (
                        ticker in growth_for_spy_holdings_by_date
                        and date in growth_for_spy_holdings_by_date[ticker]
                    ):
                        sector_growths[ticker] = float(
                            growth_for_spy_holdings_by_date[ticker][date]
                            if growth_for_spy_holdings_by_date[ticker][date] is not None
                            else 0.0
                        )

                # Calculamos z-score solo si hay al menos 2 valores
                if len(sector_growths) >= 2:
                    mean = statistics.mean(sector_growths.values())
                    std = statistics.stdev(sector_growths.values())

                    # Asignamos los z-scores a cada ticker
                    for ticker, growth in sector_growths.items():
                        zscore = (growth - mean) / std if std != 0 else 0
                        sector_holdings_date_data[ticker][date] = zscore

        # sector_holdings_date_data ahora tiene la estructura {ticker: {fecha: valor}}
        return sector_holdings_date_data

    def target_range_zscore(self):

        target_range_zscore1 = target_range_zscore(
            self.compute_sector_z_scores(),
            self.lowerzscore_limit,
            self.upperzscore_limit,
        )

        return target_range_zscore1

    def genera_watchlist(self):

        watchlist = genera_watchlist(self.target_range_zscore())

        return watchlist

    def evaluate_risk(self, simulator, date):

        positions = list(simulator.positions.keys())

        if not positions:
            return {}, {}  # No positions to evaluate

        decision_dict = {}
        details_dict = {}

        # --- Get Relevant Signal Details --- Assume valid simulation_start_date and successful call
        relevant_signals = self.watchlist

        # --- Evaluate Each Position ---
        for asset in positions:
            decision_dict[asset] = asset in relevant_signals[date]
            details_dict[asset] = (
                f"Hold: Ticker '{asset}' found in relevant signals."
                if decision_dict[asset]
                else f"Sell: Ticker '{asset}' not found in relevant signals."
            )

        return decision_dict, details_dict

    def evaluate_prospects(self, simulator, prospects, date):

        relevant_signals = self.watchlist

        decision_dict = {}
        details_dict = {}

        # --- Evaluate Each Prospect ---
        for asset in prospects:
            decision_dict[asset] = asset in relevant_signals[date]
            details_dict[asset] = (
                f"Buy: Ticker '{asset}' found in relevant signals."
                if decision_dict[asset]
                else f"Skip: Ticker '{asset}' not found in relevant signals."
            )

        return decision_dict, details_dict
