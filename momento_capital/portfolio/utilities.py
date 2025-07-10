import pandas as pd
import numpy as np
import random
import string
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, date
import math
from typing import Dict, List, Optional, Any, Union, Tuple
from io import StringIO
from momento_capital.transformers import (
    calculate_simple_moving_average,
    calculate_relative_volatility_on_prices,
    calculate_lower_bb,
)
from momento_capital.utilities import (
    apply_function_by_groups,
    func_by_groups,
    apply_function_to_data,
)


class SectorFundamentals:
    def __init__(
        self,
        metric: str,
        window_size: int,
        methodology: str,
        lowerzscore_limit: float,
        upperzscore_limit: float,
        # New parameters for pre-processed data
        fin_statements: dict = None,
        etfs_holdings: dict = None,
        tickers_to_check: list = None,
        quarterly_data: dict = None,
        filtered_ticker_data: dict = None,
        # New optional parameter for pruning
        simulation_start_date: str = None,
    ):

        self.metric = metric
        self.window_size = window_size  # in years
        self.quarterly_window = window_size * 4  # in quarters
        self.methodology = methodology
        self.lowerzscore_limit = lowerzscore_limit
        self.upperzscore_limit = upperzscore_limit
        # Data containers from pre-processed data
        self.fin_statements = fin_statements
        self.etfs_holdings = etfs_holdings
        self.tickers_to_check = tickers_to_check
        self.quarterly_data = quarterly_data
        self.filtered_ticker_data = filtered_ticker_data
        self.simulation_start_date = simulation_start_date

        # Add cache attributes
        self._cached_signal_details = {}  # Dictionary to cache results by periods_prior
        self._cached_sector_z_scores = {}  # Cache for z-scores
        self._cached_growth_data = None  # Cache for growth data

    def _calculate_metric_growth(self):
        """
        Computes growth over self.window_size years for self.metric using instance data.
        Adapts logic from compute_growth_fixed_v4.

        Returns:
            dict: Dictionary structured as { 'YYYY-MM-DD': {'TICKER1': growth, ...}, ... }
        """
        # Check if we have cached results
        if self._cached_growth_data and len(self._cached_growth_data) > 0:
            return self._cached_growth_data

        growth_data_by_date = {}

        # Process each quarter in the filtered holdings data
        for quarter_start_str, tickers in self.filtered_ticker_data.items():
            # Parse the quarter start date
            quarter_start_date = datetime.strptime(quarter_start_str, "%Y-%m-%d")

            year, month = quarter_start_date.year, quarter_start_date.month

            # Determine which quarter this belongs to
            if month <= 3:
                quarter = "Q1"
            elif month <= 6:
                quarter = "Q2"
            elif month <= 9:
                quarter = "Q3"
            else:
                quarter = "Q4"

            # Past year to compare against
            past_year = year - self.window_size

            # Initialize dictionary for this quarter date if not already present
            if quarter_start_str not in growth_data_by_date:
                growth_data_by_date[quarter_start_str] = {}

            # Process each ticker for this quarter
            for ticker in tickers:
                # Data gathering
                quarterly_dates_ticker = self.quarterly_data.get(ticker, {})
                report_date_current = quarterly_dates_ticker.get(year, {}).get(quarter)
                report_date_past = quarterly_dates_ticker.get(past_year, {}).get(
                    quarter
                )

                # Skip if dates are missing (implicit check by subsequent .get)
                if not report_date_current or not report_date_past:
                    continue

                ticker_fin_data = self.fin_statements.get(ticker, {})
                current_data = ticker_fin_data.get(report_date_current, {})
                past_data = ticker_fin_data.get(report_date_past, {})

                current_value_str = current_data.get(self.metric)
                past_value_str = past_data.get(self.metric)

                # Skip if values are missing (implicit check by subsequent float conversion)
                if current_value_str is None or past_value_str is None:
                    continue

                # Convert values
                current_value = float(current_value_str)
                past_value = float(past_value_str)

                # Calculate Growth
                if past_value == 0:
                    growth = None
                else:
                    growth = (current_value - past_value) / abs(past_value)

                # Store the computed growth
                growth_data_by_date[quarter_start_str][ticker] = growth

        # Cache the calculated data
        self._cached_growth_data = growth_data_by_date
        return growth_data_by_date

    def get_sector_growth_data(self):
        """
        Calculates metric growth and processes it for each sector, returning a dictionary of DataFrames.

        Returns:
            dict: A dictionary where keys are sector names and values are DataFrames
                  with growth data for the specified metric.
        """
        # Check if growth data is cached
        if (
            hasattr(self, "_cached_sector_growth_data")
            and self._cached_sector_growth_data is not None
        ):
            return self._cached_sector_growth_data

        sector_growth_data = {}

        # 1. Calculate the growth data
        growth_dict = self._calculate_metric_growth()

        # 2. Convert the growth dictionary to a DataFrame
        df_growth = pd.DataFrame.from_dict(growth_dict, orient="index")
        df_growth.index = pd.to_datetime(
            df_growth.index
        )  # Convert index to DatetimeIndex

        # 3. Process each sector
        for sector, tickers in self.etfs_holdings.items():
            # Find tickers present in both the sector list and the DataFrame columns
            common_tickers = list(set(tickers).intersection(df_growth.columns))

            # Skip if no common tickers
            if not common_tickers:
                continue

            # Extract relevant data for the sector
            df_sector = df_growth[common_tickers].copy()
            df_sector.dropna(
                how="all", inplace=True
            )  # Drop rows where all values are NaN

            # Store the sector DataFrame if it's not empty
            if not df_sector.empty:
                sector_growth_data[sector] = df_sector

        # Cache the result
        self._cached_sector_growth_data = sector_growth_data
        return sector_growth_data  # Returns a dictionary of sector DataFrames

    def compute_sector_z_scores(self):
        """
        Compute Z-scores for each sector based on their tickers' growth data.
        Calls get_sector_growth_data() internally.

        Returns:
            dict: Dictionary where keys are sector names and values are DataFrames
                  with computed Z-scores for that sector.
        """
        # Check cache first
        if (
            hasattr(self, "_cached_sector_z_scores_result")
            and self._cached_sector_z_scores_result is not None
        ):
            return self._cached_sector_z_scores_result

        sector_z_scores = {}

        # Call internal method to get growth data
        sector_growth_data = self.get_sector_growth_data()

        # Process each sector DataFrame provided
        for sector, df_sector in sector_growth_data.items():
            # Skip empty DataFrames implicitly by subsequent operations

            df_sector_numeric = df_sector.apply(pd.to_numeric, errors="coerce").copy()

            sector_mean = df_sector_numeric.mean(axis=1, skipna=True)
            sector_std = df_sector_numeric.std(axis=1, skipna=True)

            z_scores = df_sector_numeric.sub(sector_mean, axis=0).div(
                sector_std, axis=0
            )
            z_scores.replace([np.inf, -np.inf], np.nan, inplace=True)

            if not z_scores.empty:
                sector_z_scores[sector] = z_scores

        # Cache the result
        self._cached_sector_z_scores_result = sector_z_scores
        return sector_z_scores

    # --- Methods for Historical Filtering and Signal Details ---

    def get_historical_z_scores(self, periods_prior=2):
        """
        Filters sector Z-score dataframes to include history starting from
        a specified number of observation periods before the target date's period.
        Calls compute_sector_z_scores() internally.

        Parameters:
            periods_prior (int): Number of periods to go back.

        Returns:
            dict: Filtered dictionary of sector Z-score DataFrames.
        """
        # Check cache for this periods_prior value
        cache_key = f"historical_z_scores_{periods_prior}"
        if cache_key in self._cached_sector_z_scores:
            return self._cached_sector_z_scores[cache_key]

        target_date_str = self.simulation_start_date
        filtered_sector_z_scores = {}

        # Call internal method to get Z-scores
        sector_z_scores = self.compute_sector_z_scores()

        target_ts = pd.Timestamp(target_date_str)

        for sector, df in sector_z_scores.items():
            # Skip empty DFs implicitly
            if df.empty:
                filtered_sector_z_scores[sector] = pd.DataFrame(
                    index=pd.to_datetime([]), columns=df.columns
                ).astype(df.dtypes)
                continue

            # Assume index is DatetimeIndex
            df_sorted = df.sort_index()
            ref_loc = df_sorted.index.get_indexer([target_ts], method="ffill")[0]

            if ref_loc == -1:
                start_loc = -1
            else:
                start_loc = ref_loc - periods_prior

            if start_loc >= 0:
                start_date = df_sorted.index[start_loc]
                filtered_sector_z_scores[sector] = df_sorted.loc[start_date:].copy()
            else:
                filtered_sector_z_scores[sector] = pd.DataFrame(
                    index=pd.to_datetime([]), columns=df_sorted.columns
                ).astype(df_sorted.dtypes)

        # Cache result for this periods_prior
        self._cached_sector_z_scores[cache_key] = filtered_sector_z_scores
        return filtered_sector_z_scores

    def _get_quarter_start_date(self, year: Any, quarter_str: str):
        """Calculates the quarter start date string (YYYY-MM-DD). (Internal helper)"""
        QUARTER_MONTH_MAP: Dict[str, int] = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}
        month = QUARTER_MONTH_MAP.get(quarter_str)
        # Assume year is int and month is valid
        return date(year, month, 1).strftime("%Y-%m-%d")

    def _get_z_score(
        self,
        sector_z_scores: Dict[str, pd.DataFrame],
        sector: Optional[str],
        ticker: str,
        quarter_start_date: Optional[str],
    ) -> Optional[float]:
        """Looks up the Z-score for a given ticker, sector, and date. (Internal helper)"""
        # Assume inputs are valid and sector_df exists and is DataFrame
        sector_df = sector_z_scores.get(sector)

        # Assume index is datetime
        lookup_date = pd.Timestamp(quarter_start_date)
        if lookup_date in sector_df.index and ticker in sector_df.columns:
            raw_z = sector_df.loc[lookup_date, ticker]
            if raw_z is not None and pd.notna(raw_z):
                z_score = float(raw_z)
                return None if np.isnan(z_score) else z_score
        return None

    def get_signal_details(
        self, periods_prior: int = 2
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Computes detailed signal dict using filtered Z-scores and instance data.
        Calls get_historical_z_scores() internally.

        Args:
            periods_prior (int): The number of periods prior to the target date for filtering.

        Returns:
            dict: Ticker -> Report Date -> {"filing_date":..., "quarter_start_date":..., "z_score":...}
        """
        # Check cache first for this periods_prior value
        if periods_prior in self._cached_signal_details:
            return self._cached_signal_details[periods_prior]

        # --- Get Filtered Z-Scores Internally ---
        filtered_sector_z_scores = self.get_historical_z_scores(periods_prior)

        # --- Use Instance Data ---
        quarterly_data_fixed = self.quarterly_data
        spy_incst_q = self.fin_statements
        etfs_holdings = self.etfs_holdings
        sector_z_scores = filtered_sector_z_scores  # Use the filtered dict

        # --- Precompute ticker_to_sector mapping ---
        ticker_to_sector: Dict[str, str] = {}
        for sector, tickers_list in etfs_holdings.items():
            for ticker in tickers_list:
                ticker_to_sector[ticker] = sector

        results: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # --- Main Loop ---
        for ticker, yearly_data in quarterly_data_fixed.items():
            sector = ticker_to_sector.get(ticker)
            sector_df = sector_z_scores.get(sector)  # Look in the filtered results

            # Skip if sector_df is None or empty (implicit check)
            if sector_df is None or sector_df.empty:
                continue

            # Assume index is datetime

            for year, quarterly_reports in yearly_data.items():
                for quarter_str, report_date_str in quarterly_reports.items():
                    # Assume report_date_str is valid string

                    quarter_start_date_str = self._get_quarter_start_date(
                        year, quarter_str
                    )
                    # Assume quarter_start_date_str is valid

                    quarter_start_ts = pd.Timestamp(quarter_start_date_str)

                    # IMPORTANT Check: Ensure the quarter_start_ts is within the filtered Z-score index
                    if quarter_start_ts not in sector_df.index:
                        continue

                    filing_info = spy_incst_q.get(ticker, {}).get(report_date_str, {})
                    # Assume filing_info is dict or None if not found
                    filing_date_str = (
                        filing_info.get("filing_date")
                        if isinstance(filing_info, dict)
                        else None
                    )

                    # Use internal helper with the filtered Z-scores
                    z_score = self._get_z_score(
                        sector_z_scores, sector, ticker, quarter_start_date_str
                    )

                    if ticker not in results:
                        results[ticker] = {}

                    # Store result only if Z-score is found
                    if z_score is not None:
                        results[ticker][report_date_str] = {
                            "filing_date": filing_date_str,
                            "quarter_start_date": quarter_start_date_str,
                            "z_score": z_score,
                        }

        # Cache the result for this periods_prior value
        self._cached_signal_details[periods_prior] = results
        return results

    # --- Risk Evaluation Method ---
    def evaluate_risk(self, simulator, date):
        """
        Evaluates risk for current positions based on the latest signal Z-score
        within the historical window [date - periods_prior, date]. Assumes valid inputs.

        Args:
            simulator: The portfolio simulator instance with a 'positions' attribute (dict).
            date: The evaluation date string ('YYYY-MM-DD').

        Returns:
            A tuple containing (decision_dict, details_dict).
            decision_dict: Ticker -> bool (True=Keep, False=Sell).
            details_dict: Ticker -> str (Reason for decision).
        """
        # --- Get Current Positions --- Assume valid simulator and positions
        positions = list(simulator.positions.keys())

        if not positions:
            return {}, {}  # No positions to evaluate

        decision_dict: Dict[str, bool] = {}
        details_dict: Dict[str, str] = {}

        # --- Get Relevant Signal Details --- Assume valid simulation_start_date and successful call
        relevant_signals = self.get_signal_details()  # Uses default periods_prior=2

        # --- Convert Evaluation Date --- Assume valid date format
        date_ts = pd.Timestamp(date)

        # --- Evaluate Each Position ---
        for asset in positions:
            decision_dict[asset] = False  # Default to Sell
            details_dict[asset] = (
                "Sell: Default - No keep condition met."  # Default detail
            )

            # Check if asset has signals
            if asset not in relevant_signals:
                details_dict[asset] = (
                    f"Sell: Ticker '{asset}' not found in relevant signals."
                )
                continue

            asset_signal_data = relevant_signals[asset]
            # Assume asset_signal_data is a dict

            # Find signals with valid filing date <= evaluation date WITHIN the retrieved signals
            valid_filed_signals = []
            for report_date, signal_info in asset_signal_data.items():
                # Assume signal_info is dict and has 'filing_date'
                filing_date_str = signal_info.get("filing_date")
                if filing_date_str:  # Check if filing_date exists and is not empty/None
                    filing_date_ts = pd.Timestamp(
                        filing_date_str
                    )  # Assume valid format
                    if filing_date_ts <= date_ts:
                        valid_filed_signals.append((filing_date_ts, signal_info))

            if not valid_filed_signals:
                details_dict[asset] = (
                    f"Sell: No signal for '{asset}' with filing date <= {date}."
                )
                continue

            # Get the latest signal based on filing date
            valid_filed_signals.sort(key=lambda item: item[0], reverse=True)
            latest_filing_date_ts, latest_signal_info = valid_filed_signals[0]
            latest_filing_date_str = latest_filing_date_ts.strftime("%Y-%m-%d")

            z_score = latest_signal_info.get("z_score")

            # Evaluate Z-score - Assume z_score is float or None/NaN
            if z_score is not None and pd.notna(z_score):
                z_score_float = float(z_score)  # Assume conversion works
                if self.lowerzscore_limit <= z_score_float <= self.upperzscore_limit:
                    decision_dict[asset] = True
                    details_dict[asset] = (
                        f"Keep: Z={z_score_float:.2f} (filing: {latest_filing_date_str}) in range [{self.lowerzscore_limit:.2f}, {self.upperzscore_limit:.2f}]."
                    )
                else:
                    details_dict[asset] = (
                        f"Sell: Z={z_score_float:.2f} (filing: {latest_filing_date_str}) out of range [{self.lowerzscore_limit:.2f}, {self.upperzscore_limit:.2f}]."
                    )
            else:
                details_dict[asset] = (
                    f"Sell: Z-score missing/invalid in latest signal (filing: {latest_filing_date_str})."
                )

        return decision_dict, details_dict

    # --- Prospect Evaluation Method (Optimized) ---
    def evaluate_prospects(
        self, simulator, prospects: List[str], date: str
    ) -> Tuple[Dict[str, bool], Dict[str, str]]:
        """
        Evaluates potential new investments (prospects) based on the latest signal Z-score
        filed on or before the evaluation date.

        Args:
            simulator: The portfolio simulator instance (passed but not used in this implementation).
            prospects: A list of tickers representing potential investments.
            date: The evaluation date string ('YYYY-MM-DD').

        Returns:
            A tuple containing (decision_dict, details_dict).
            decision_dict: Ticker -> bool (True=Buy, False=Skip).
            details_dict: Ticker -> str (Reason for decision).
        """
        decision_dict: Dict[str, bool] = {}
        details_dict: Dict[str, str] = {}

        # --- Get Relevant Signal Details ---
        # This will now use the cached results if available
        relevant_signals = self.get_signal_details()

        # --- Convert Evaluation Date ---
        date_ts = pd.Timestamp(date)

        # --- Evaluate Each Prospect ---
        for asset in prospects:
            # Check if asset exists in the keys of relevant_signals before accessing
            if asset in relevant_signals:
                asset_signal_data = relevant_signals[asset]

                # Find signals with filing date <= evaluation date
                valid_filed_signals = []
                for report_date, signal_info in asset_signal_data.items():
                    filing_date_str = signal_info.get("filing_date")
                    z_score_val = signal_info.get("z_score")

                    # Check filing date and z_score existence
                    if filing_date_str and z_score_val is not None:
                        try:
                            filing_date_ts = pd.Timestamp(filing_date_str)
                            if filing_date_ts <= date_ts:
                                valid_filed_signals.append(
                                    (filing_date_ts, signal_info)
                                )
                        except (ValueError, TypeError):
                            pass  # Skip invalid dates

                if valid_filed_signals:
                    # Get the latest signal based on filing date
                    valid_filed_signals.sort(key=lambda item: item[0], reverse=True)
                    latest_filing_date_ts, latest_signal_info = valid_filed_signals[0]
                    latest_filing_date_str = latest_filing_date_ts.strftime("%Y-%m-%d")

                    z_score = latest_signal_info.get("z_score")

                    try:
                        z_score_float = float(z_score)
                        if (
                            self.lowerzscore_limit
                            <= z_score_float
                            <= self.upperzscore_limit
                        ):
                            decision_dict[asset] = True  # Buy the asset
                            details_dict[asset] = (
                                f"Buy: Z={z_score_float:.2f} (filing: {latest_filing_date_str}) in range [{self.lowerzscore_limit:.2f}, {self.upperzscore_limit:.2f}]."
                            )
                        else:
                            decision_dict[asset] = False  # Skip: Z-score out of range
                            details_dict[asset] = (
                                f"Skip: Z={z_score_float:.2f} (filing: {latest_filing_date_str}) out of range [{self.lowerzscore_limit:.2f}, {self.upperzscore_limit:.2f}]."
                            )
                    except (ValueError, TypeError):
                        decision_dict[asset] = False
                        details_dict[asset] = (
                            f"Skip: Invalid Z-score for '{asset}' in latest signal (filing: {latest_filing_date_str})."
                        )
                else:
                    # No valid signal found before the evaluation date
                    decision_dict[asset] = False  # Default to Skip
                    details_dict[asset] = (
                        f"Skip: No valid signal for '{asset}' with filing date <= {date}."
                    )
            else:
                # Asset not found in relevant_signals
                decision_dict[asset] = False
                details_dict[asset] = f"Skip: Prospect '{asset}' not in financial data."

        return decision_dict, details_dict


class Sector_yyqq_Fundamentals:
    def __init__(
        self,
        metric: str,
        window_size: int,
        methodology: str,
        lowerzscore_limit: float,
        upperzscore_limit: float,
        # New parameters for pre-processed data
        fin_statements: dict = None,
        etfs_holdings: dict = None,
        tickers_to_check: list = None,
        quarterly_data: dict = None,
        filtered_ticker_data: dict = None,
        # New optional parameter for pruning
        simulation_start_date: str = None,
    ):

        self.metric = metric
        self.window_size = window_size  # in years
        self.quarterly_window = window_size * 4  # in quarters
        self.methodology = methodology
        self.lowerzscore_limit = lowerzscore_limit
        self.upperzscore_limit = upperzscore_limit
        # Data containers from pre-processed data
        self.fin_statements = fin_statements
        self.etfs_holdings = etfs_holdings
        self.tickers_to_check = tickers_to_check
        self.quarterly_data = quarterly_data
        self.filtered_ticker_data = filtered_ticker_data
        self.simulation_start_date = simulation_start_date

        # Add cache attributes
        self._cached_signal_details = {}  # Dictionary to cache results by periods_prior
        self._cached_sector_z_scores = {}  # Cache for z-scores
        self._cached_growth_data = {}  # Cache for growth data
        self._cached_qoq_growth_data = {}
        self._cached_qoq_std_data = {}
        self._cached_sector_yyqq_data = {}
        self._cached_yy_qq_ratio = {}  # Explicitly initialize this cache too

    def _calculate_metric_growth(self):
        """
        Computes growth over self.window_size years for self.metric using instance data.
        Adapts logic from compute_growth_fixed_v4.

        Returns:
            dict: Dictionary structured as { 'YYYY-MM-DD': {'TICKER1': growth, ...}, ... }
        """
        # Check if we have cached results
        if self._cached_growth_data and len(self._cached_growth_data) > 0:
            return self._cached_growth_data

        growth_data_by_date = {}

        # Process each quarter in the filtered holdings data
        for quarter_start_str, tickers in self.filtered_ticker_data.items():
            # Parse the quarter start date
            quarter_start_date = datetime.strptime(quarter_start_str, "%Y-%m-%d")

            year, month = quarter_start_date.year, quarter_start_date.month

            # Determine which quarter this belongs to
            if month <= 3:
                quarter = "Q1"
            elif month <= 6:
                quarter = "Q2"
            elif month <= 9:
                quarter = "Q3"
            else:
                quarter = "Q4"

            # Past year to compare against
            past_year = year - self.window_size

            # Initialize dictionary for this quarter date if not already present
            if quarter_start_str not in growth_data_by_date:
                growth_data_by_date[quarter_start_str] = {}

            # Process each ticker for this quarter
            for ticker in tickers:
                # Data gathering
                quarterly_dates_ticker = self.quarterly_data.get(ticker, {})
                report_date_current = quarterly_dates_ticker.get(year, {}).get(quarter)
                report_date_past = quarterly_dates_ticker.get(past_year, {}).get(
                    quarter
                )

                # Skip if dates are missing (implicit check by subsequent .get)
                if not report_date_current or not report_date_past:
                    continue

                ticker_fin_data = self.fin_statements.get(ticker, {})
                current_data = ticker_fin_data.get(report_date_current, {})
                past_data = ticker_fin_data.get(report_date_past, {})

                current_value_str = current_data.get(self.metric)
                past_value_str = past_data.get(self.metric)

                # Skip if values are missing (implicit check by subsequent float conversion)
                if current_value_str is None or past_value_str is None:
                    continue

                # Convert values
                current_value = float(current_value_str)
                past_value = float(past_value_str)

                # Calculate Growth
                if past_value == 0:
                    growth = None
                else:
                    growth = (current_value - past_value) / abs(past_value)

                # Store the computed growth
                growth_data_by_date[quarter_start_str][ticker] = growth

        # Cache the calculated data
        self._cached_growth_data = growth_data_by_date
        return growth_data_by_date

    def compute_sequential_growth(self):
        """
        Computes sequential quarter-over-quarter (QoQ) growth for totalRevenue.

        Args:
            income_statements (dict): JSON containing financial data (by ticker by date).
            filtered_quarterly_holdings (dict): Dictionary where keys are quarter start dates
                (`YYYY-MM-01`) and values are lists of tickers that have holdings in that quarter.
            quarterly_dates (dict): Output from `define_quarters_fixed()`, containing actual report dates per quarter.

        Returns:
            dict: A dictionary with quarter dates as keys and, for each quarter, a dictionary of tickers
                with their QoQ growth (or None if not computable) as values.
        """

        growth_data = {}

        for quarter_start_str, tickers in self.filtered_ticker_data.items():
            # Convert quarter start date to datetime
            quarter_start_date = datetime.strptime(quarter_start_str, "%Y-%m-%d")
            year, month = quarter_start_date.year, quarter_start_date.month

            # Determine which quarter this belongs to
            if month <= 3:
                quarter = "Q1"
                prev_quarter, prev_year = "Q4", year - 1
            elif month <= 6:
                quarter = "Q2"
                prev_quarter, prev_year = "Q1", year
            elif month <= 9:
                quarter = "Q3"
                prev_quarter, prev_year = "Q2", year
            else:
                quarter = "Q4"
                prev_quarter, prev_year = "Q3", year

            # Initialize dictionary for this quarter
            growth_data[quarter_start_str] = {}

            for ticker in tickers:
                # Ensure ticker has data in `quarterly_dates`
                if (
                    ticker not in self.quarterly_data
                    or year not in self.quarterly_data[ticker]
                ):
                    continue

                # Get actual report date for this quarter and previous quarter
                report_date_this_quarter = self.quarterly_data[ticker][year].get(
                    quarter
                )
                report_date_prev_quarter = (
                    self.quarterly_data[ticker].get(prev_year, {}).get(prev_quarter)
                )

                # Get totalRevenue for the given dates
                revenue_this_quarter = (
                    self.fin_statements.get(ticker, {})
                    .get(report_date_this_quarter, {})
                    .get(self.metric)
                )
                revenue_prev_quarter = (
                    self.fin_statements.get(ticker, {})
                    .get(report_date_prev_quarter, {})
                    .get(self.metric)
                )

                # Convert revenue to float
                try:
                    revenue_this_quarter = (
                        float(revenue_this_quarter)
                        if revenue_this_quarter is not None
                        else None
                    )
                    revenue_prev_quarter = (
                        float(revenue_prev_quarter)
                        if revenue_prev_quarter is not None
                        else None
                    )
                except ValueError:
                    revenue_this_quarter, revenue_prev_quarter = None, None

                # Compute QoQ growth if data is available
                if revenue_this_quarter is not None and revenue_prev_quarter not in (
                    None,
                    0,
                ):
                    growth = (
                        revenue_this_quarter - revenue_prev_quarter
                    ) / revenue_prev_quarter
                else:
                    growth = None

                # Store the computed growth for this ticker
                growth_data[quarter_start_str][ticker] = growth

        self._cached_qoq_growth_data = growth_data

        return growth_data

    def compute_qoq_std(self):
        """
        Computes the rolling standard deviation of the last `window` observations
        for quarter-over-quarter (QoQ) growth.

        Args:
            qoq_growth_data (dict): Dictionary where keys are quarter start dates and
                                    values are dictionaries of tickers with their QoQ growth.
            window (int, optional): Number of quarters to use for the rolling standard deviation. Default is 4.

        Returns:
            dict: A dictionary with quarter dates as keys and, for each quarter, a dictionary of tickers
                with their rolling standard deviation of QoQ growth (or None if not computable) as values.
        """
        if self._cached_qoq_std_data and len(self._cached_qoq_std_data) > 0:
            return self._cached_qoq_std_data
        std_data = {}
        tickers = set(
            ticker
            for growths in self._cached_qoq_growth_data.values()
            for ticker in growths
        )

        # Convert dictionary to a sorted list of dates
        sorted_dates = sorted(self._cached_qoq_growth_data.keys())

        for quarter_start_str in sorted_dates:
            std_data[quarter_start_str] = {}
            for ticker in tickers:
                # Get the last `window` QoQ growth values for this ticker
                last_values = [
                    self._cached_qoq_growth_data[q][ticker]
                    for q in sorted_dates
                    if q <= quarter_start_str
                    and ticker in self._cached_qoq_growth_data[q]
                ][
                    -self.quarterly_window :
                ]  # Take last `window` values

                # Compute standard deviation if there are at least 2 values
                if len(last_values) >= 2 and all(v is not None for v in last_values):
                    std_data[quarter_start_str][ticker] = np.std(
                        last_values, ddof=1
                    )  # Sample standard deviation
                else:
                    std_data[quarter_start_str][ticker] = None

        self._cached_qoq_std_data = std_data

        return std_data

    def compute_yy_qq_ratio(self):
        """
        Computes the ratio of yearly growth to quarter-over-quarter standard deviation.

        This ratio indicates how significant the yearly growth is compared to the
        quarterly volatility of the metric.

        Returns:
            dict: Dictionary structured as { 'YYYY-MM-DD': {'TICKER1': ratio, ...}, ... }
                where ratio is yearly_growth / qoq_std or None if data is missing/invalid
        """
        # Check for cached results (Check if dict is non-empty)
        if hasattr(self, "_cached_yy_qq_ratio") and self._cached_yy_qq_ratio:
            print("Debug: Returning cached yy_qq_ratio data.")  # Added debug print
            return self._cached_yy_qq_ratio

        print(
            "Debug: Calculating yy_qq_ratio (cache empty or not found)."
        )  # Added debug print
        # Get yearly growth data
        yearly_growth = self._calculate_metric_growth()

        # Ensure sequential growth is computed first
        if not self._cached_qoq_growth_data or len(self._cached_qoq_growth_data) == 0:
            self.compute_sequential_growth()

        # Get QoQ standard deviation data
        qoq_std = self.compute_qoq_std()

        ratio_data = {}

        # Process each date in the data
        for date in yearly_growth:
            if date not in qoq_std:
                continue

            ratio_data[date] = {}

            # Process each ticker for this date
            for ticker in yearly_growth[date]:
                if ticker not in qoq_std[date]:
                    continue

                growth = yearly_growth[date][ticker]
                std = qoq_std[date][ticker]

                # Handle None values and division by zero
                if growth is None or std is None or std == 0:
                    ratio = None
                else:
                    ratio = growth / std

                ratio_data[date][ticker] = ratio

        # Cache the calculated ratios
        self._cached_yy_qq_ratio = ratio_data

        return ratio_data

    def get_sector_yyqq_data(self):
        """
        Calculates metric growth and processes it for each sector, returning a dictionary of DataFrames.

        Returns:
            dict: A dictionary where keys are sector names and values are DataFrames
                with growth data for the specified metric.
        """
        # Check if SECTOR data is cached (Check if dict is non-empty)
        if hasattr(self, "_cached_sector_yyqq_data") and self._cached_sector_yyqq_data:
            print("Debug: Returning cached sector yyqq data.")
            return self._cached_sector_yyqq_data

        # If cache exists but is empty, print a message. If it doesn't exist, this won't trigger.
        if (
            hasattr(self, "_cached_sector_yyqq_data")
            and not self._cached_sector_yyqq_data
        ):
            print(
                "Debug: Cache _cached_sector_yyqq_data exists but is empty. Proceeding with calculation."
            )

        sector_yyqq_data = {}
        print("Debug: Initialized empty sector_yyqq_data.")

        # 1. Calculate the growth data
        print("Debug: Calling self.compute_yy_qq_ratio()...")
        yyqq_dict = self.compute_yy_qq_ratio()
        print("Debug: Returned from self.compute_yy_qq_ratio().")
        print("Debug: Structure of yyqq_dict (first 5 items):")
        # Print a sample to check structure, handle potential errors if not a dict or empty
        try:
            # Check if it's a dict and not empty before slicing
            if isinstance(yyqq_dict, dict) and yyqq_dict:
                print(dict(list(yyqq_dict.items())[:5]))
            elif isinstance(yyqq_dict, dict):
                print("Debug: yyqq_dict is an empty dictionary.")
            else:
                print(f"Debug: yyqq_dict is not a dictionary? Type: {type(yyqq_dict)}")
                print(yyqq_dict)  # Print the actual content if not a dict
        except Exception as e:
            print(f"Debug: Error inspecting yyqq_dict structure: {e}")
            print(f"Debug: yyqq_dict type: {type(yyqq_dict)}")

        # 2. Convert the growth dictionary to a DataFrame
        if not yyqq_dict or not isinstance(yyqq_dict, dict):
            print(
                "Debug: yyqq_dict is empty or not a dict. Returning empty sector data."
            )
            self._cached_sector_yyqq_data = {}  # Ensure cache is empty dict
            return {}

        try:
            print("Debug: Creating DataFrame from yyqq_dict...")
            df_yyqq = pd.DataFrame.from_dict(yyqq_dict, orient="index")
            print("\nDebug: df_yyqq info BEFORE date conversion:")
            # df_yyqq.info() prints to stdout, using print captures it better in some environments
            buffer = StringIO()
            df_yyqq.info(buf=buffer)
            print(buffer.getvalue())
            print("\nDebug: df_yyqq head BEFORE date conversion:")
            print(df_yyqq.head())
            print(f"Debug: df_yyqq shape: {df_yyqq.shape}")
        except Exception as e:
            print(f"Debug: Error creating DataFrame: {e}")
            self._cached_sector_yyqq_data = {}
            return {}

        try:
            print("\nDebug: Converting df_yyqq index to datetime...")
            df_yyqq.index = pd.to_datetime(
                df_yyqq.index
            )  # Convert index to DatetimeIndex
            print("Debug: Index conversion successful.")
            print("\nDebug: df_yyqq head AFTER date conversion:")
            print(df_yyqq.head())
        except Exception as e:
            print(f"Debug: Error converting index to datetime: {e}")
            print("Debug: df_yyqq index values before conversion attempt:")
            print(df_yyqq.index)
            # Decide how to handle error - returning empty dict for safety
            self._cached_sector_yyqq_data = {}
            return {}

        # 3. Process each sector
        print("\nDebug: Starting sector processing loop...")
        for sector, tickers in self.etfs_holdings.items():
            print(f"\n--- Processing Sector: {sector} ---")
            # Limit printing tickers if list is long
            print(f"Debug: Tickers in sector (first 5 shown): {tickers[:5]}...")

            # Find tickers present in both the sector list and the DataFrame columns
            df_columns = list(df_yyqq.columns)
            print(f"Debug: df_yyqq columns (first 10 shown): {df_columns[:10]}...")
            common_tickers = list(set(tickers).intersection(df_columns))
            # Limit printing common tickers if list is long
            print(
                f"Debug: Common tickers found (first 10 shown): {common_tickers[:10]}..."
            )

            # Skip if no common tickers
            if not common_tickers:
                print("Debug: No common tickers found for this sector. Skipping.")
                continue

            # Extract relevant data for the sector
            print(f"Debug: Extracting data for {len(common_tickers)} common tickers...")
            try:
                df_sector = df_yyqq[common_tickers].copy()
                print(
                    f"Debug: df_sector shape for {sector} BEFORE dropna: {df_sector.shape}"
                )
                # print(f"Debug: df_sector head for {sector} BEFORE dropna:") # Optional: uncomment for more detail
                # print(df_sector.head())
            except Exception as e:
                print(f"Debug: Error selecting common tickers for sector {sector}: {e}")
                continue  # Skip to next sector if selection fails

            print(f"Debug: Dropping rows with all NaNs for sector {sector}...")
            try:
                df_sector.dropna(
                    how="all", inplace=True
                )  # Drop rows where all values are NaN
                print(
                    f"Debug: df_sector shape for {sector} AFTER dropna: {df_sector.shape}"
                )
                # print(f"Debug: df_sector head for {sector} AFTER dropna:") # Optional: uncomment for more detail
                # print(df_sector.head())
            except Exception as e:
                print(f"Debug: Error during dropna for sector {sector}: {e}")
                continue  # Skip storing if dropna fails

            # Store the sector DataFrame if it's not empty
            if not df_sector.empty:
                print(
                    f"Debug: Storing data for sector {sector} (Shape: {df_sector.shape})"
                )
                sector_yyqq_data[sector] = df_sector
            else:
                print(
                    f"Debug: df_sector for {sector} is empty after dropna. Not storing."
                )

        # Cache the result
        print("\nDebug: Finished processing all sectors.")
        print(f"Debug: Final sector_yyqq_data keys: {list(sector_yyqq_data.keys())}")
        self._cached_sector_yyqq_data = sector_yyqq_data
        return sector_yyqq_data  # Returns a dictionary of sector DataFrames

    def compute_sector_z_scores(self):
        """
        Compute Z-scores for each sector based on their tickers' growth data.
        Calls get_sector_growth_data() internally.

        Returns:
            dict: Dictionary where keys are sector names and values are DataFrames
                  with computed Z-scores for that sector.
        """
        # Check cache first
        if (
            hasattr(self, "_cached_sector_z_scores_result")
            and self._cached_sector_z_scores_result is not None
        ):
            return self._cached_sector_z_scores_result

        sector_z_scores = {}

        # Call internal method to get growth data
        sector_yyqq_data = self.get_sector_yyqq_data()

        # Process each sector DataFrame provided
        for sector, df_sector in sector_yyqq_data.items():
            # Skip empty DataFrames implicitly by subsequent operations

            df_sector_numeric = df_sector.apply(pd.to_numeric, errors="coerce").copy()

            sector_mean = df_sector_numeric.mean(axis=1, skipna=True)
            sector_std = df_sector_numeric.std(axis=1, skipna=True)

            z_scores = df_sector_numeric.sub(sector_mean, axis=0).div(
                sector_std, axis=0
            )
            z_scores.replace([np.inf, -np.inf], np.nan, inplace=True)

            if not z_scores.empty:
                sector_z_scores[sector] = z_scores

        # Cache the result
        self._cached_sector_z_scores_result = sector_z_scores
        return sector_z_scores

    # --- Methods for Historical Filtering and Signal Details ---

    def get_historical_z_scores(self, periods_prior=2):
        """
        Filters sector Z-score dataframes to include history starting from
        a specified number of observation periods before the target date's period.
        Calls compute_sector_z_scores() internally.

        Parameters:
            periods_prior (int): Number of periods to go back.

        Returns:
            dict: Filtered dictionary of sector Z-score DataFrames.
        """
        # Check cache for this periods_prior value
        cache_key = f"historical_z_scores_{periods_prior}"
        if cache_key in self._cached_sector_z_scores:
            return self._cached_sector_z_scores[cache_key]

        target_date_str = self.simulation_start_date
        filtered_sector_z_scores = {}

        # Call internal method to get Z-scores
        sector_z_scores = self.compute_sector_z_scores()

        target_ts = pd.Timestamp(target_date_str)

        for sector, df in sector_z_scores.items():
            # Skip empty DFs implicitly
            if df.empty:
                filtered_sector_z_scores[sector] = pd.DataFrame(
                    index=pd.to_datetime([]), columns=df.columns
                ).astype(df.dtypes)
                continue

            # Assume index is DatetimeIndex
            df_sorted = df.sort_index()
            ref_loc = df_sorted.index.get_indexer([target_ts], method="ffill")[0]

            if ref_loc == -1:
                start_loc = -1
            else:
                start_loc = ref_loc - periods_prior

            if start_loc >= 0:
                start_date = df_sorted.index[start_loc]
                filtered_sector_z_scores[sector] = df_sorted.loc[start_date:].copy()
            else:
                filtered_sector_z_scores[sector] = pd.DataFrame(
                    index=pd.to_datetime([]), columns=df_sorted.columns
                ).astype(df_sorted.dtypes)

        # Cache result for this periods_prior
        self._cached_sector_z_scores[cache_key] = filtered_sector_z_scores
        return filtered_sector_z_scores

    def _get_quarter_start_date(self, year: Any, quarter_str: str):
        """Calculates the quarter start date string (YYYY-MM-DD). (Internal helper)"""
        QUARTER_MONTH_MAP: Dict[str, int] = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}
        month = QUARTER_MONTH_MAP.get(quarter_str)
        # Assume year is int and month is valid
        return date(year, month, 1).strftime("%Y-%m-%d")

    def _get_z_score(
        self,
        sector_z_scores: Dict[str, pd.DataFrame],
        sector: Optional[str],
        ticker: str,
        quarter_start_date: Optional[str],
    ) -> Optional[float]:
        """Looks up the Z-score for a given ticker, sector, and date. (Internal helper)"""
        # Assume inputs are valid and sector_df exists and is DataFrame
        sector_df = sector_z_scores.get(sector)

        # Assume index is datetime
        lookup_date = pd.Timestamp(quarter_start_date)
        if lookup_date in sector_df.index and ticker in sector_df.columns:
            raw_z = sector_df.loc[lookup_date, ticker]
            if raw_z is not None and pd.notna(raw_z):
                z_score = float(raw_z)
                return None if np.isnan(z_score) else z_score
        return None

    def get_signal_details(
        self, periods_prior: int = 2
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Computes detailed signal dict using filtered Z-scores and instance data.
        Calls get_historical_z_scores() internally.

        Args:
            periods_prior (int): The number of periods prior to the target date for filtering.

        Returns:
            dict: Ticker -> Report Date -> {"filing_date":..., "quarter_start_date":..., "z_score":...}
        """
        # Check cache first for this periods_prior value
        if periods_prior in self._cached_signal_details:
            return self._cached_signal_details[periods_prior]

        # --- Get Filtered Z-Scores Internally ---
        filtered_sector_z_scores = self.get_historical_z_scores(periods_prior)

        # --- Use Instance Data ---
        quarterly_data_fixed = self.quarterly_data
        spy_incst_q = self.fin_statements
        etfs_holdings = self.etfs_holdings
        sector_z_scores = filtered_sector_z_scores  # Use the filtered dict

        # --- Precompute ticker_to_sector mapping ---
        ticker_to_sector: Dict[str, str] = {}
        for sector, tickers_list in etfs_holdings.items():
            for ticker in tickers_list:
                ticker_to_sector[ticker] = sector

        results: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # --- Main Loop ---
        for ticker, yearly_data in quarterly_data_fixed.items():
            sector = ticker_to_sector.get(ticker)
            sector_df = sector_z_scores.get(sector)  # Look in the filtered results

            # Skip if sector_df is None or empty (implicit check)
            if sector_df is None or sector_df.empty:
                continue

            # Assume index is datetime

            for year, quarterly_reports in yearly_data.items():
                for quarter_str, report_date_str in quarterly_reports.items():
                    # Assume report_date_str is valid string

                    quarter_start_date_str = self._get_quarter_start_date(
                        year, quarter_str
                    )
                    # Assume quarter_start_date_str is valid

                    quarter_start_ts = pd.Timestamp(quarter_start_date_str)

                    # IMPORTANT Check: Ensure the quarter_start_ts is within the filtered Z-score index
                    if quarter_start_ts not in sector_df.index:
                        continue

                    filing_info = spy_incst_q.get(ticker, {}).get(report_date_str, {})
                    # Assume filing_info is dict or None if not found
                    filing_date_str = (
                        filing_info.get("filing_date")
                        if isinstance(filing_info, dict)
                        else None
                    )

                    # Use internal helper with the filtered Z-scores
                    z_score = self._get_z_score(
                        sector_z_scores, sector, ticker, quarter_start_date_str
                    )

                    if ticker not in results:
                        results[ticker] = {}

                    # Store result only if Z-score is found
                    if z_score is not None:
                        results[ticker][report_date_str] = {
                            "filing_date": filing_date_str,
                            "quarter_start_date": quarter_start_date_str,
                            "z_score": z_score,
                        }

        # Cache the result for this periods_prior value
        self._cached_signal_details[periods_prior] = results
        return results

    # --- Risk Evaluation Method ---
    def evaluate_risk(self, simulator, date):
        """
        Evaluates risk for current positions based on the latest signal Z-score
        within the historical window [date - periods_prior, date]. Assumes valid inputs.

        Args:
            simulator: The portfolio simulator instance with a 'positions' attribute (dict).
            date: The evaluation date string ('YYYY-MM-DD').

        Returns:
            A tuple containing (decision_dict, details_dict).
            decision_dict: Ticker -> bool (True=Keep, False=Sell).
            details_dict: Ticker -> str (Reason for decision).
        """
        # --- Get Current Positions --- Assume valid simulator and positions
        positions = list(simulator.positions.keys())

        if not positions:
            return {}, {}  # No positions to evaluate

        decision_dict: Dict[str, bool] = {}
        details_dict: Dict[str, str] = {}

        # --- Get Relevant Signal Details --- Assume valid simulation_start_date and successful call
        relevant_signals = self.get_signal_details()  # Uses default periods_prior=2

        # --- Convert Evaluation Date --- Assume valid date format
        date_ts = pd.Timestamp(date)

        # --- Evaluate Each Position ---
        for asset in positions:
            decision_dict[asset] = False  # Default to Sell
            details_dict[asset] = (
                "Sell: Default - No keep condition met."  # Default detail
            )

            # Check if asset has signals
            if asset not in relevant_signals:
                details_dict[asset] = (
                    f"Sell: Ticker '{asset}' not found in relevant signals."
                )
                continue

            asset_signal_data = relevant_signals[asset]
            # Assume asset_signal_data is a dict

            # Find signals with valid filing date <= evaluation date WITHIN the retrieved signals
            valid_filed_signals = []
            for report_date, signal_info in asset_signal_data.items():
                # Assume signal_info is dict and has 'filing_date'
                filing_date_str = signal_info.get("filing_date")
                if filing_date_str:  # Check if filing_date exists and is not empty/None
                    filing_date_ts = pd.Timestamp(
                        filing_date_str
                    )  # Assume valid format
                    if filing_date_ts <= date_ts:
                        valid_filed_signals.append((filing_date_ts, signal_info))

            if not valid_filed_signals:
                details_dict[asset] = (
                    f"Sell: No signal for '{asset}' with filing date <= {date}."
                )
                continue

            # Get the latest signal based on filing date
            valid_filed_signals.sort(key=lambda item: item[0], reverse=True)
            latest_filing_date_ts, latest_signal_info = valid_filed_signals[0]
            latest_filing_date_str = latest_filing_date_ts.strftime("%Y-%m-%d")

            z_score = latest_signal_info.get("z_score")

            # Evaluate Z-score - Assume z_score is float or None/NaN
            if z_score is not None and pd.notna(z_score):
                z_score_float = float(z_score)  # Assume conversion works
                if self.lowerzscore_limit <= z_score_float <= self.upperzscore_limit:
                    decision_dict[asset] = True
                    details_dict[asset] = (
                        f"Keep: Z={z_score_float:.2f} (filing: {latest_filing_date_str}) in range [{self.lowerzscore_limit:.2f}, {self.upperzscore_limit:.2f}]."
                    )
                else:
                    details_dict[asset] = (
                        f"Sell: Z={z_score_float:.2f} (filing: {latest_filing_date_str}) out of range [{self.lowerzscore_limit:.2f}, {self.upperzscore_limit:.2f}]."
                    )
            else:
                details_dict[asset] = (
                    f"Sell: Z-score missing/invalid in latest signal (filing: {latest_filing_date_str})."
                )

        return decision_dict, details_dict

    # --- Prospect Evaluation Method (Optimized) ---
    def evaluate_prospects(
        self, simulator, prospects: List[str], date: str
    ) -> Tuple[Dict[str, bool], Dict[str, str]]:
        """
        Evaluates potential new investments (prospects) based on the latest signal Z-score
        filed on or before the evaluation date.

        Args:
            simulator: The portfolio simulator instance (passed but not used in this implementation).
            prospects: A list of tickers representing potential investments.
            date: The evaluation date string ('YYYY-MM-DD').

        Returns:
            A tuple containing (decision_dict, details_dict).
            decision_dict: Ticker -> bool (True=Buy, False=Skip).
            details_dict: Ticker -> str (Reason for decision).
        """
        decision_dict: Dict[str, bool] = {}
        details_dict: Dict[str, str] = {}

        # --- Get Relevant Signal Details ---
        # This will now use the cached results if available
        relevant_signals = self.get_signal_details()

        # --- Convert Evaluation Date ---
        date_ts = pd.Timestamp(date)

        # --- Evaluate Each Prospect ---
        for asset in prospects:
            # Check if asset exists in the keys of relevant_signals before accessing
            if asset in relevant_signals:
                asset_signal_data = relevant_signals[asset]

                # Find signals with filing date <= evaluation date
                valid_filed_signals = []
                for report_date, signal_info in asset_signal_data.items():
                    filing_date_str = signal_info.get("filing_date")
                    z_score_val = signal_info.get("z_score")

                    # Check filing date and z_score existence
                    if filing_date_str and z_score_val is not None:
                        try:
                            filing_date_ts = pd.Timestamp(filing_date_str)
                            if filing_date_ts <= date_ts:
                                valid_filed_signals.append(
                                    (filing_date_ts, signal_info)
                                )
                        except (ValueError, TypeError):
                            pass  # Skip invalid dates

                if valid_filed_signals:
                    # Get the latest signal based on filing date
                    valid_filed_signals.sort(key=lambda item: item[0], reverse=True)
                    latest_filing_date_ts, latest_signal_info = valid_filed_signals[0]
                    latest_filing_date_str = latest_filing_date_ts.strftime("%Y-%m-%d")

                    z_score = latest_signal_info.get("z_score")

                    try:
                        z_score_float = float(z_score)
                        if (
                            self.lowerzscore_limit
                            <= z_score_float
                            <= self.upperzscore_limit
                        ):
                            decision_dict[asset] = True  # Buy the asset
                            details_dict[asset] = (
                                f"Buy: Z={z_score_float:.2f} (filing: {latest_filing_date_str}) in range [{self.lowerzscore_limit:.2f}, {self.upperzscore_limit:.2f}]."
                            )
                        else:
                            decision_dict[asset] = False  # Skip: Z-score out of range
                            details_dict[asset] = (
                                f"Skip: Z={z_score_float:.2f} (filing: {latest_filing_date_str}) out of range [{self.lowerzscore_limit:.2f}, {self.upperzscore_limit:.2f}]."
                            )
                    except (ValueError, TypeError):
                        decision_dict[asset] = False
                        details_dict[asset] = (
                            f"Skip: Invalid Z-score for '{asset}' in latest signal (filing: {latest_filing_date_str})."
                        )
                else:
                    # No valid signal found before the evaluation date
                    decision_dict[asset] = False  # Default to Skip
                    details_dict[asset] = (
                        f"Skip: No valid signal for '{asset}' with filing date <= {date}."
                    )
            else:
                # Asset not found in relevant_signals
                decision_dict[asset] = False
                details_dict[asset] = f"Skip: Prospect '{asset}' not in financial data."

        return decision_dict, details_dict


class TrailingStopSMA:
    """
    Implements a trailing stop strategy using a simple moving average (SMA).
    Evaluates risk and prospects for portfolio management.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing historical asset prices.
        sma_df (pd.DataFrame): A DataFrame storing the SMA values for each asset.
    """

    def __init__(self, period, df):
        """
        Initializes the TrailingStopSMA class.

        Args:
            period (int): The window size for calculating the simple moving average (SMA).
            df (pd.DataFrame): A DataFrame containing historical asset prices with dates as the index
                and asset names as columns.
        """
        self.df = df
        self.sma_df = apply_function_by_groups(
            df=df,
            func=lambda group: func_by_groups(
                group=group,
                func=calculate_simple_moving_average,
                window_size=period,
            ),
        )

    def evaluate_risk(self, simulator, date):
        """
        Evaluates the risk of the current portfolio positions based on SMA strategy.

        Args:
            simulator (object): An object containing the current portfolio positions (assets held).
                Expected to have an attribute `positions` which is a dictionary of asset names as keys.
            date (str or datetime): The date for which the evaluation is performed.

        returns:
            tuple:
                decision (dict): A dictionary where keys are asset names and values are booleans:
                    - True: Current price >= SMA (hold).
                    - False: Current price < SMA (sell).
                details (dict): A dictionary containing details of each asset's evaluation with keys:
                    - "Price": The current price of the asset.
                    - "SMA": The simple moving average of the asset.
        """
        assets = list(simulator.positions.keys())
        current_price = self.df.loc[date]
        current_sma = self.sma_df.loc[date]
        details = {}
        decision = {}

        for asset in assets:
            details[asset] = {
                "Price": current_price.loc[asset],
                "SMA": current_sma.loc[asset],
            }
            decision[asset] = current_price.loc[asset] >= current_sma.loc[asset]

        return decision, details

    def evaluate_prospects(self, simulator, prospects, date):
        """
        Evaluates potential investment prospects based on SMA strategy.

        Args:
            simulator (object): An object representing the portfolio simulator.
            prospects (list): A list of prospective assets to evaluate.
            date (str or datetime): The date for which the evaluation is performed.

        returns:
            tuple:
                decision (dict): A dictionary where keys are asset names and values are booleans:
                    - True: Current price >= SMA (consider investing).
                    - False: Current price < SMA (do not invest).
                details (dict): A dictionary containing details of each asset's evaluation with keys:
                    - "Price": The current price of the asset.
                    - "SMA": The simple moving average of the asset.
        """
        current_price = self.df.loc[date]
        current_sma = self.sma_df.loc[date]
        details = {}
        decision = {}

        for asset in prospects:
            details[asset] = {
                "Price": current_price.loc[asset],
                "SMA": current_sma.loc[asset],
            }
            decision[asset] = current_price.loc[asset] >= current_sma.loc[asset]

        return decision, details


class TrailingStopNegativeRocVolatilityStd:
    def __init__(self, threshold, df, window_size, z_score_window_size):
        self.threshold = threshold
        self.df = df
        self.window_size = window_size
        self.z_score_window_size = z_score_window_size
        self.max_window = window_size + z_score_window_size
        self.volatility_df = apply_function_to_data(
            df=df,
            function=calculate_relative_volatility_on_prices,
            returns_period=1,
            window_size=window_size,
        )
        self.z_vol_df = apply_function_to_data(
            df=self.volatility_df,
            function=lambda array: (
                array[
                    -len(
                        np.lib.stride_tricks.sliding_window_view(
                            array, window_shape=z_score_window_size, axis=0
                        )
                    ) :
                ]
                - np.mean(
                    np.lib.stride_tricks.sliding_window_view(
                        array, window_shape=z_score_window_size, axis=0
                    ),
                    axis=2,
                )
            )
            / np.std(
                np.lib.stride_tricks.sliding_window_view(
                    array, window_shape=z_score_window_size, axis=0
                ),
                axis=2,
            ),
        )
        self.roc_df = apply_function_to_data(
            df=df, function=lambda array: (array[1:] / array[:-1]) - 1
        )

    def evaluate_risk(self, simulator, date):

        assets = list(simulator.positions.keys())
        current_price = self.df.loc[date]
        current_z_volatility = self.z_vol_df.loc[date]
        details = {}
        decision = {}

        for asset_idx, asset in enumerate(assets):
            last_volatility_window = self.volatility_df.iloc[
                self.volatility_df.index.get_loc(date)
                - self.window_size
                + 1 : self.volatility_df.index.get_loc(date)
                + 1
            ][[asset]]
            volatility_peak_date = last_volatility_window.diff()[asset].idxmax()
            roc_on_max_vol = self.roc_df.loc[volatility_peak_date, asset]
            details[asset] = {
                "Price": current_price.loc[asset],
                "Z_Volatility": current_z_volatility.loc[asset],
                "ROC_on_max_vol": roc_on_max_vol,
            }

            if current_z_volatility.loc[asset] >= self.threshold:
                if roc_on_max_vol < 0:
                    decision[asset] = False
                else:
                    decision[asset] = True
            else:
                decision[asset] = True  # false es vender
        return decision, details

    def evaluate_prospects(self, simulator, prospects, date):

        current_price = self.df.loc[date]
        current_z_volatility = self.z_vol_df.loc[date]
        details = {}
        decision = {}

        for asset_idx, asset in enumerate(prospects):
            last_volatility_window = self.volatility_df.iloc[
                self.volatility_df.index.get_loc(date)
                - self.window_size
                + 1 : self.volatility_df.index.get_loc(date)
                + 1
            ][[asset]]
            volatility_peak_date = last_volatility_window.diff()[asset].idxmax()
            roc_on_max_vol = self.roc_df.loc[volatility_peak_date, asset]
            details[asset] = {
                "Price": current_price.loc[asset],
                "Z_Volatility": current_z_volatility.loc[asset],
                "ROC_on_max_vol": roc_on_max_vol,
            }
            if current_z_volatility.loc[asset] >= self.threshold:
                if roc_on_max_vol < 0:
                    decision[asset] = False
                else:
                    decision[asset] = True
            else:
                decision[asset] = True  # false es vender

        return decision, details


class TrailingStopVolatilityStd:
    def __init__(
        self,
        threshold,
        df,
        window_size,
        z_score_window_size,
    ):
        self.threshold = threshold
        self.df = df
        volatility_df = apply_function_to_data(
            df=df,
            function=calculate_relative_volatility_on_prices,
            returns_period=1,
            window_size=window_size,
        )
        self.z_vol_df = apply_function_to_data(
            df=volatility_df,
            function=lambda array: (
                array[
                    -len(
                        np.lib.stride_tricks.sliding_window_view(
                            array, window_shape=z_score_window_size, axis=0
                        )
                    ) :
                ]
                - np.mean(
                    np.lib.stride_tricks.sliding_window_view(
                        array, window_shape=z_score_window_size, axis=0
                    ),
                    axis=2,
                )
            )
            / np.std(
                np.lib.stride_tricks.sliding_window_view(
                    array, window_shape=z_score_window_size, axis=0
                ),
                axis=2,
            ),
        )
        self.max_window = window_size + z_score_window_size

    def evaluate_risk(self, simulator, date):

        assets = list(simulator.positions.keys())
        current_price = self.df.loc[date]
        current_z_volatility = self.z_vol_df.loc[date]
        details = {}
        decision = {}

        for asset_idx, asset in enumerate(assets):
            details[asset] = {
                "Price": current_price.loc[asset],
                "Z_Volatility": current_z_volatility.loc[asset],
            }
            decision[asset] = current_z_volatility.loc[asset] <= self.threshold

        return decision, details

    def evaluate_prospects(self, simulator, prospects, date):

        current_price = self.df.loc[date]
        current_z_volatility = self.z_vol_df.loc[date]
        details = {}
        decision = {}

        for asset_idx, asset in enumerate(prospects):
            details[asset] = {
                "Price": current_price.loc[asset],
                "Z_Volatility": current_z_volatility.loc[asset],
            }
            decision[asset] = current_z_volatility.loc[asset] <= self.threshold

        return decision, details


class TrailingStopVolatility:
    """
    Implements a trailing stop strategy based on asset volatility.
    Evaluates risk and prospects for portfolio management using a volatility threshold.

    Attributes:
        threshold (float): The multiplier for the standard deviation of volatility to set the trailing stop level.
        df (pd.DataFrame): The input DataFrame containing historical asset prices.
        volatility_df (pd.DataFrame): A DataFrame storing the calculated volatility for each asset.
        stds (np.ndarray): An array of standard deviations of the volatility for each asset.
    """

    def __init__(
        self,
        threshold,
        df,
        returns_period,
        window_size,
        returns_method="percentage",
    ):
        """
        Initializes the TrailingStopVolatility class.

        Args:
            threshold (float): The multiplier for standard deviation of volatility to set the stop level.
            df (pd.DataFrame): A DataFrame containing historical asset prices with dates as the index
                and asset names as columns.
            returns_period (int): The period over which returns are calculated.
            window_size (int): The window size for calculating volatility.
            returns_method (str, optional): The method used to calculate returns (e.g., "percentage" or "log").
                Defaults to "percentage".
        """
        self.threshold = threshold
        self.df = df
        self.volatility_df = apply_function_by_groups(
            df=df,
            func=lambda group: func_by_groups(
                group=group,
                func=calculate_relative_volatility_on_prices,
                returns_period=returns_period,
                window_size=window_size,
                returns_method=returns_method,
            ),
        )
        self.stds = self.volatility_df.std().values

    def evaluate_risk(self, simulator, date):
        """
        Evaluates the risk of current portfolio positions based on volatility threshold.

        Args:
            simulator (object): An object containing the current portfolio positions (assets held).
                Expected to have an attribute `positions` which is a dictionary of asset names as keys.
            date (str or datetime): The date for which the evaluation is performed.

        returns:
            tuple:
                decision (dict): A dictionary where keys are asset names and values are booleans:
                    - True: Current volatility <= threshold (hold).
                    - False: Current volatility > threshold (sell).
                details (dict): A dictionary containing details of each asset's evaluation with keys:
                    - "Price": The current price of the asset.
                    - "Volatility": The current volatility of the asset.
                    - "Volatility STD": The standard deviation of the asset's volatility.
                    - "Volatility STD * threshold": The calculated trailing stop level.
        """
        assets = list(simulator.positions.keys())
        current_price = self.df.loc[date]
        current_volatility = self.volatility_df.loc[date]
        details = {}
        decision = {}

        for asset_idx, asset in enumerate(assets):
            details[asset] = {
                "Price": current_price.loc[asset],
                "Volatility": current_volatility.loc[asset],
                "Volatility STD": self.stds[asset_idx],
                f"Volatility STD * {self.threshold}": self.stds[asset_idx]
                * self.threshold,
            }
            decision[asset] = (
                current_volatility.loc[asset] <= self.stds[asset_idx] * self.threshold
            )

        return decision, details

    def evaluate_prospects(self, simulator, prospects, date):
        """
        Evaluates potential investment prospects based on volatility threshold.

        Args:
            simulator (object): An object representing the portfolio simulator.
            prospects (list): A list of prospective assets to evaluate.
            date (str or datetime): The date for which the evaluation is performed.

        returns:
            tuple:
                decision (dict): A dictionary where keys are asset names and values are booleans:
                    - True: Current volatility <= threshold (consider investing).
                    - False: Current volatility > threshold (do not invest).
                details (dict): A dictionary containing details of each asset's evaluation with keys:
                    - "Price": The current price of the asset.
                    - "Volatility": The current volatility of the asset.
                    - "Volatility STD": The standard deviation of the asset's volatility.
                    - "Volatility STD * threshold": The calculated trailing stop level.
        """
        current_price = self.df.loc[date]
        current_volatility = self.volatility_df.loc[date]
        details = {}
        decision = {}

        for asset_idx, asset in enumerate(prospects):
            details[asset] = {
                "Price": current_price.loc[asset],
                "Volatility": current_volatility.loc[asset],
                "Volatility STD": self.stds[asset_idx],
                f"Volatility STD * {self.threshold}": self.stds[asset_idx]
                * self.threshold,
            }
            decision[asset] = (
                current_volatility.loc[asset] <= self.stds[asset_idx] * self.threshold
            )

        return decision, details


class TrailingStopBollinger:
    """
    Implements a trailing stop strategy using Bollinger Bands.
    Evaluates risk and prospects for portfolio management based on the lower Bollinger Band.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing historical asset prices.
        bollinger_factor (float): The multiplier for the rolling volatility to calculate the Bollinger Bands.
        sma_df (pd.DataFrame): A DataFrame storing the simple moving average (SMA) for each asset.
        roll_vol_df (pd.DataFrame): A DataFrame storing the rolling volatility for each asset.
        lower_bband_df (pd.DataFrame): A DataFrame storing the calculated lower Bollinger Band for each asset.
    """

    def __init__(self, df, window_size, bollinger_factor):
        """
        Initializes the TrailingStopBollinger class.

        Args:
            df (pd.DataFrame): A DataFrame containing historical asset prices with dates as the index
                and asset names as columns.
            window_size (int): The window size for calculating the SMA and rolling volatility.
            bollinger_factor (float): The multiplier for the rolling volatility to calculate the Bollinger Bands.
        """
        self.df = df
        self.lower_bband_df = apply_function_by_groups(
            df=df,
            func=lambda group: func_by_groups(
                group=group,
                func=calculate_lower_bb,
                window_size=window_size,
                bollinger_factor=bollinger_factor,
            ),
        )

    def evaluate_risk(self, simulator, date):
        """
        Evaluates the risk of current portfolio positions based on the lower Bollinger Band.

        Args:
            simulator (object): An object containing the current portfolio positions (assets held).
                Expected to have an attribute `positions` which is a dictionary of asset names as keys.
            date (str or datetime): The date for which the evaluation is performed.

        returns:
            tuple:
                decision (dict): A dictionary where keys are asset names and values are booleans:
                    - True: Current price >= lower Bollinger Band (hold).
                    - False: Current price < lower Bollinger Band (sell).
                details (dict): A dictionary containing details of each asset's evaluation with keys:
                    - "Price": The current price of the asset.
                    - "SMA": The simple moving average of the asset.
                    - "Absolute Volatility": The rolling volatility of the asset.
                    - "Bollinger Lower Band": The calculated lower Bollinger Band of the asset.
        """
        assets = list(simulator.positions.keys())
        current_price = self.df.loc[date]
        current_lower_bband = self.lower_bband_df.loc[date]
        details = {}
        decision = {}

        for asset in assets:
            details[asset] = {
                "Price": current_price.loc[asset],
                f"Bollinger Lower Band": current_lower_bband.loc[asset],
            }
            decision[asset] = current_price.loc[asset] >= current_lower_bband.loc[asset]

        return decision, details

    def evaluate_prospects(self, simulator, prospects, date):
        """
        Evaluates potential investment prospects based on the lower Bollinger Band.

        Args:
            simulator (object): An object representing the portfolio simulator.
            prospects (list): A list of prospective assets to evaluate.
            date (str or datetime): The date for which the evaluation is performed.

        returns:
            tuple:
                decision (dict): A dictionary where keys are asset names and values are booleans:
                    - True: Current price >= lower Bollinger Band (consider investing).
                    - False: Current price < lower Bollinger Band (do not invest).
                details (dict): A dictionary containing details of each asset's evaluation with keys:
                    - "Price": The current price of the asset.
                    - "SMA": The simple moving average of the asset.
                    - "Absolute Volatility": The rolling volatility of the asset.
                    - "Bollinger Lower Band": The calculated lower Bollinger Band of the asset.
        """
        current_price = self.df.loc[date]
        current_lower_bband = self.lower_bband_df.loc[date]

        details = {}
        decision = {}

        for asset in prospects:
            details[asset] = {
                "Price": current_price.loc[asset],
                f"Bollinger Lower Band": current_lower_bband.loc[asset],
            }
            decision[asset] = current_price.loc[asset] >= current_lower_bband.loc[asset]

        return decision, details


class TrailingStopEquitySMA:
    """
    Implements a trailing stop strategy for portfolio equity based on a simple moving average (SMA).
    Evaluates the portfolio's risk by comparing its equity value against the SMA of its historical values.

    Attributes:
        window_size (int): The window size used for calculating the equity SMA.
    """

    def __init__(self, window_size):
        """
        Initializes the TrailingStopEquitySMA class.

        Args:
            window_size (int): The window size for calculating the equity SMA.
        """
        self.window_size = window_size

    def evaluate_risk(self, simulator, date):
        """
        Evaluates the portfolio's risk based on the SMA of its equity.

        Args:
            simulator (object): An object containing the current state of the portfolio.
                Expected to have an attribute `value` which is a DataFrame with columns "date" and "value".
                Also expected to have an attribute `positions` which is a dictionary of assets currently held.
            date (str or datetime): The date for which the evaluation is performed.

        returns:
            tuple:
                decision (dict): A dictionary where keys are asset names and values are booleans:
                    - True: Keep the asset (equity >= SMA).
                    - False: Sell the asset (equity < SMA).
                details (dict): A dictionary containing details of the portfolio's evaluation with keys:
                    - "Portfolio": A sub-dictionary with keys:
                        - "Equity": The current equity value of the portfolio.
                        - "Equity SMA": The calculated SMA of the portfolio's equity.
        """
        # Convert portfolio value to DataFrame and set the index to dates
        equity_df = pd.DataFrame(simulator.value)
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df.set_index("date", inplace=True)

        # Handle the case where there are insufficient data points for the SMA calculation
        if equity_df.shape[0] <= self.window_size:
            details = {
                "Portfolio": {
                    "Equity": equity_df.loc[date, "value"],
                    "Equity SMA": np.nan,
                }
            }
            decision = {asset: True for asset in simulator.positions}
            return decision, details

        # Calculate the SMA for equity values
        equity_sma = calculate_simple_moving_average(
            array=equity_df["value"].values.reshape(-1, 1),
            window_size=self.window_size,
        )
        sma_df = pd.DataFrame(
            equity_sma, index=equity_df.index[-equity_sma.shape[0] :], columns=["SMA"]
        )

        # Gather details of the portfolio's current equity and SMA
        details = {
            "Portfolio": {
                "Equity": equity_df.loc[date, "value"],
                "Equity SMA": sma_df.loc[date, "SMA"],
            }
        }

        # Determine whether to keep or sell assets based on the equity comparison to the SMA
        keeping = equity_df.loc[date, "value"] >= sma_df.loc[date, "SMA"]
        decision = {asset: keeping for asset in simulator.positions}

        return decision, details


class MomentoPolicy:
    def __init__(self, lookback_window, tolerance):
        self.lookback_window = lookback_window
        self.tolerance = tolerance
        self.max_window = lookback_window + 1

    def evaluate_risk(self, simulator, date):
        assets = list(simulator.positions.keys())
        equity_df = pd.DataFrame(simulator.value)
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df.set_index("date", inplace=True)
        if equity_df.shape[0] <= self.lookback_window:
            details = {
                "Portfolio": {
                    "Equity": equity_df.loc[date, "value"],
                }
            }
            decision = {asset: True for asset in assets}
            return decision, details
        window = equity_df.iloc[-self.lookback_window :]["value"].values
        max_value = np.max(window)
        current_value = equity_df.loc[date, "value"]
        percentage_change = (current_value - max_value) / max_value
        details = {
            "Portfolio": {
                f"{self.lookback_window} Days Max": max_value,
                "Current Value": current_value,
                "Percentage Change": percentage_change,
            }
        }
        if percentage_change < -self.tolerance:
            decision = {asset: False for asset in assets}
        else:
            decision = {asset: True for asset in assets}
        return decision, details


class FixedStopLoss:
    """
    Implements a fixed stop-loss strategy based on a predefined threshold.
    Evaluates whether to hold or sell assets by comparing current prices to entry prices adjusted by the threshold.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing historical asset prices.
        threshold (float): The percentage threshold for the stop-loss level.
    """

    def __init__(self, df, threshold):
        """
        Initializes the FixedStopLoss class.

        Args:
            df (pd.DataFrame): A DataFrame containing historical asset prices with dates as the index
                and asset names as columns.
            threshold (float): The stop-loss threshold as a percentage (e.g., 0.05 for 5%).
        """
        self.df = df
        self.threshold = threshold

    def evaluate_risk(self, simulator, date):
        """
        Evaluates the risk of current portfolio positions based on the stop-loss threshold.

        Args:
            simulator (object): An object containing the current portfolio positions and trade history.
                Expected to have attributes `positions` (a dictionary of assets currently held),
                `trades` (a record of trades), and `history` (a log of asset prices and actions).
            date (str or datetime): The date for which the evaluation is performed.

        returns:
            tuple:
                decision (dict): A dictionary where keys are asset names and values are booleans:
                    - True: Current price > entry price * (1 - threshold) (hold).
                    - False: Current price <= entry price * (1 - threshold) (sell).
                details (dict): A dictionary containing details of the evaluation with keys:
                    - "date": The entry date for the asset.
                    - "price": The entry price for the asset.
        """
        assets = list(simulator.positions.keys())
        current_price = self.df.loc[date]
        trades_df = pd.DataFrame(simulator.trades)
        logs_df = pd.DataFrame(simulator.history)

        entry_data = self._read_entry_prices(
            assets=assets, trades_df=trades_df, logs_df=logs_df, date=date
        )

        decision = {}
        details = {}

        for asset in assets:
            details[asset] = entry_data[asset]
            decision[asset] = current_price.loc[asset] > entry_data[asset]["price"] * (
                1 - self.threshold
            )

        return decision, details

    def _read_entry_prices(self, assets, trades_df, logs_df, date):
        """
        Retrieves the entry prices for the given assets from the trade and log history.

        Args:
            assets (list): List of asset names currently held in the portfolio.
            trades_df (pd.DataFrame): A DataFrame containing the trade history with columns "date", "action", and "asset".
            logs_df (pd.DataFrame): A DataFrame containing the log history with columns "date", "asset", and "asset_price".
            date (str or datetime): The date for which the evaluation is performed.

        returns:
            dict: A dictionary where keys are asset names and values are dictionaries with keys:
                - "date": The entry date for the asset.
                - "price": The entry price for the asset.
        """
        dates = trades_df["date"].unique().tolist()[::-1]
        assets_entry_dates = {}

        # Find the most recent entry date for each asset
        for asset in assets:
            for i_date in dates:
                i_date_trade_assets = trades_df.loc[
                    (trades_df["action"] == "Buy") & (trades_df["date"] == i_date)
                ]["asset"].unique()
                if asset not in i_date_trade_assets:
                    continue
                assets_entry_dates[asset] = trades_df.loc[
                    (trades_df["date"] == i_date) & (trades_df["asset"] == asset)
                ]["entry_date"].iloc[0]

        entry_prices = {}

        # Retrieve the entry price for each asset
        for asset in assets:
            entry_prices[asset] = logs_df.loc[
                (logs_df["asset"] == asset)
                & (logs_df["date"] == assets_entry_dates[asset])
            ]["asset_price"].iloc[0]

        return {
            asset: {"date": assets_entry_dates[asset], "price": entry_prices[asset]}
            for asset in assets
        }


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm


class PortfolioEvaluator:
    def __init__(
        self, benchmark_series, risk_free_rate=0.0, confidence_level=0.95, threshold=0
    ):
        self.benchmark_series = benchmark_series
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.threshold = threshold

    def evaluate_trades(self, simulator):
        self.equity_data = pd.DataFrame(simulator.value)
        self.equity_data["date"] = pd.to_datetime(self.equity_data["date"])
        self.equity_data.set_index("date", inplace=True)
        self.equity_data = self.equity_data["value"]
        self.portfolio_id = simulator.portfolio_id
        df_multi = pd.DataFrame(simulator.trades)
        df_logs = pd.DataFrame(simulator.history)
        df_logs["date"] = pd.to_datetime(df_logs["date"])
        df_multi["date"] = pd.to_datetime(df_multi["date"])
        trades_evaluation = self._trade_metrics(df_multi=df_multi, df_logs=df_logs)
        trades_evaluation.insert(0, "portfolio_id", self.portfolio_id)
        return trades_evaluation

    def calculate_metrics(self, simulator):
        self.portfolio_id = simulator.portfolio_id
        self.equity_data = pd.DataFrame(simulator.value)
        self.equity_data["date"] = pd.to_datetime(self.equity_data["date"])
        self.equity_data.set_index("date", inplace=True)
        self.equity_data = self.equity_data["value"]
        equity_data = pd.DataFrame(simulator.value)
        equity_data["date"] = pd.to_datetime(equity_data["date"])
        equity_data.set_index("date", inplace=True)
        metrics_df = self._metrics(df=equity_data)
        pivoted_df = metrics_df.T
        pivoted_df.columns = pivoted_df.iloc[0, :]
        pivoted_df = pivoted_df.iloc[1:]
        pivoted_df["start_date"] = pd.to_datetime(pivoted_df["start_date"])
        pivoted_df["end_date"] = pd.to_datetime(pivoted_df["end_date"])
        pivoted_df.reset_index(inplace=True)
        pivoted_df.drop(columns=["index"], inplace=True)
        return pivoted_df

    def plot_vs_benchmark(self, benchmark_label="spy", single_axis=True):

        series_1 = self.equity_data.copy()
        series_2 = self.benchmark_series.copy()
        if len(series_1) < len(series_2):
            series_2 = series_2.loc[series_2.index >= series_1.index[0]]

        if not single_axis:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor("black")
            ax1.set_facecolor("black")
            ax1.plot(
                series_1.index,
                series_1.values,
                label="Equity Data",
                color="cyan",
                linewidth=2,
                linestyle="-",
            )
            ax1.set_xlabel("Time", fontsize=12, color="white")
            ax1.set_ylabel("Equity Data", color="cyan", fontsize=12)
            ax1.tick_params(axis="y", labelcolor="cyan", labelsize=10)

            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(
                ax1.xaxis.get_majorticklabels(),
                rotation=45,
                ha="right",
                fontsize=10,
                color="white",
            )

            ax1.grid(
                visible=True,
                which="both",
                linestyle="--",
                linewidth=0.5,
                alpha=0.7,
                color="gray",
            )
            ax2 = ax1.twinx()
            ax2.set_facecolor("black")
            ax2.plot(
                series_2.index,
                series_2.values,
                label=benchmark_label,
                color="magenta",
                linewidth=2,
                linestyle="--",
            )
            ax2.set_ylabel(benchmark_label, color="magenta", fontsize=12)
            ax2.tick_params(axis="y", labelcolor="magenta", labelsize=10)

            plt.title(
                "Performance Comparison: Equity vs Benchmark",
                fontsize=14,
                fontweight="bold",
                pad=20,
                color="white",
            )

            # fig.legend(
            #     loc="upper left",
            #     bbox_to_anchor=(0.1, 0.9),
            #     fontsize=10,
            #     frameon=True,
            #     facecolor="black",
            #     edgecolor="white",
            # )
            # Crear la leyenda
            legend = fig.legend(
                loc="upper left",
                bbox_to_anchor=(0.1, 0.9),
                fontsize=10,
                frameon=True,
                facecolor="black",
                edgecolor="white",
            )

            # Cambiar el color del texto de la leyenda a blanco
            for text in legend.get_texts():
                text.set_color("white")

            fig.tight_layout()

            plt.show()
        else:
            series_2_starting_value = series_2.iat[0]
            series_1_starting_value = series_1.iat[0]
            series_1 = series_1 * (series_2_starting_value / series_1_starting_value)

            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")

            ax.plot(
                series_1.index,
                series_1.values,
                label="Equity Data (Scaled)",
                color="cyan",
                linewidth=2,
                linestyle="-",
            )
            ax.plot(
                series_2.index,
                series_2.values,
                label=benchmark_label,
                color="magenta",
                linewidth=2,
                linestyle="--",
            )

            ax.set_xlabel("Time", fontsize=12, color="white")
            ax.set_ylabel("value", fontsize=12, color="white")
            ax.tick_params(axis="x", labelsize=10, color="white")
            ax.tick_params(axis="y", labelsize=10, color="white")

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(
                ax.xaxis.get_majorticklabels(),
                rotation=45,
                ha="right",
                fontsize=10,
                color="white",
            )

            ax.grid(
                visible=True,
                which="both",
                linestyle="--",
                linewidth=0.5,
                alpha=0.7,
                color="gray",
            )

            plt.title(
                "Performance Comparison: Equity vs Benchmark (Single Axis)",
                fontsize=14,
                fontweight="bold",
                pad=20,
                color="white",
            )

            legend = ax.legend(
                loc="upper left",
                fontsize=10,
                frameon=True,
                facecolor="black",
                edgecolor="white",
            )

            # Cambiar el color del texto de la leyenda a blanco
            for text in legend.get_texts():
                text.set_color("white")

            fig.tight_layout()

            plt.show()

    def _calculate_correct_drawdown_time(
        self, df_logs, retornos_por_trade, df_trade_metrics
    ):
        filtered_logs = []
        for _, row in retornos_por_trade.iterrows():
            logs = df_logs.loc[
                (df_logs["asset"] == row["asset"])
                & (df_logs["date"] >= row["fecha_inicio"])
                & (df_logs["date"] <= row["fecha_salida"])
            ].copy()
            logs.loc[:, "trade_id"] = row["trade_id"]
            filtered_logs.append(logs)

        filtered_logs_df = pd.concat(filtered_logs)

        filtered_logs_df = filtered_logs_df.merge(
            df_trade_metrics[["trade_id", "trade_avg_buy_price"]],
            on="trade_id",
            how="left",
        )

        if (
            "asset_price" in filtered_logs_df.columns
            and "trade_avg_buy_price" in filtered_logs_df.columns
        ):
            filtered_logs_df["price_difference"] = (
                filtered_logs_df["asset_price"]
                - filtered_logs_df["trade_avg_buy_price"]
            )
            filtered_logs_df["drawdown"] = filtered_logs_df["price_difference"].clip(
                upper=0
            )
        else:
            raise ValueError(
                "Missing required columns: 'asset_price' or 'trade_avg_buy_price'."
            )

        drawdown_time = (
            filtered_logs_df[filtered_logs_df["drawdown"] < 0]
            .groupby("trade_id")
            .size()
        )
        drawdown_time_df = drawdown_time.reset_index(name="drawdown_time")

        return drawdown_time_df

    def _mean_buy_price(self, trades_df):
        trade_ids = trades_df["trade_id"].unique()
        trade_weighted_avg_buy_price = {}
        for trade_id in trade_ids:
            buy_actions = trades_df.loc[
                (trades_df["trade_id"] == trade_id) & (trades_df["action"] == "Buy")
            ][["amount", "price"]]
            weights = buy_actions.values[:, 0] / np.sum(buy_actions.values[:, 0])
            prices = buy_actions.values[:, 1]
            weighted_average = np.dot(weights, prices)
            trade_weighted_avg_buy_price[trade_id] = weighted_average
        return trade_weighted_avg_buy_price

    def _trade_metrics(self, df_multi, df_logs):
        trades_mean_buy_price = self._mean_buy_price(trades_df=df_multi)
        df_multi["trade_avg_buy_price"] = df_multi["trade_id"].map(
            trades_mean_buy_price
        )

        processed_df = df_multi.groupby("trade_id", group_keys=False).apply(
            self._process_trade_group
        )
        processed_df.reset_index(drop=True, inplace=True)

        date_differences = df_multi.groupby("trade_id")["date"].agg(["min", "max"])
        date_differences["days_difference"] = (
            date_differences["max"] - date_differences["min"]
        ).dt.days
        asset_mapping = df_multi.groupby("trade_id")["asset"].first()
        date_differences["asset"] = asset_mapping
        date_differences.reset_index(inplace=True)

        max_twr_values = processed_df.loc[
            processed_df.groupby("trade_id")["date"].idxmax(),
            ["trade_id", "twrr_acumulado"],
        ]
        retornos_por_trade = date_differences.merge(
            max_twr_values, on="trade_id", how="left"
        )
        retornos_por_trade.rename(
            columns={
                "min": "fecha_inicio",
                "max": "fecha_salida",
                "days_difference": "plazo_dias",
                "twrr_acumulado": "twrr",
            },
            inplace=True,
        )

        amount_summary = []
        for _, row in retornos_por_trade.iterrows():
            filtered_logs = df_logs[
                (df_logs["asset"] == row["asset"])
                & (df_logs["date"] >= row["fecha_inicio"])
                & (df_logs["date"] <= row["fecha_salida"])
            ]
            amount_min = (
                filtered_logs["amount"].min() if not filtered_logs.empty else None
            )
            amount_max = (
                filtered_logs["amount"].max() if not filtered_logs.empty else None
            )
            amount_summary.append(
                {
                    "trade_id": row["trade_id"],
                    "asset": row["asset"],
                    "fecha_inicio": row["fecha_inicio"],
                    "fecha_salida": row["fecha_salida"],
                    "amount_min": amount_min,
                    "amount_max": amount_max,
                }
            )

        amount_summary_df = pd.DataFrame(amount_summary)
        retornos_por_trade["mae"] = round(
            (
                (amount_summary_df["amount_min"] - processed_df["valor_inicial"])
                / processed_df["valor_inicial"]
            )
            * 100,
            2,
        )
        retornos_por_trade["mfe"] = round(
            (
                (amount_summary_df["amount_max"] - processed_df["valor_inicial"])
                / processed_df["valor_inicial"]
            )
            * 100,
            2,
        )
        retornos_por_trade["tpr"] = round(
            (
                (amount_summary_df["amount_max"] - amount_summary_df["amount_min"])
                / amount_summary_df["amount_min"]
            )
            * 100,
            2,
        )
        retornos_por_trade["return_to_tpr"] = round(
            (retornos_por_trade["twrr"] / retornos_por_trade["tpr"]) * 100, 2
        )
        retornos_por_trade["twrr"] = round((retornos_por_trade["twrr"]) * 100, 2)

        retornos_por_trade = retornos_por_trade.merge(
            df_multi[["trade_id", "trade_avg_buy_price"]].drop_duplicates(),
            on="trade_id",
            how="left",
        )

        drawdown_time_df = self._calculate_correct_drawdown_time(
            df_logs, retornos_por_trade, retornos_por_trade
        )
        retornos_por_trade = retornos_por_trade.merge(
            drawdown_time_df, on="trade_id", how="left"
        )
        retornos_por_trade["drawdown_time"] = (
            retornos_por_trade["drawdown_time"].fillna(0).astype(int)
        )

        return retornos_por_trade

    def _process_trade_group(self, group):
        results = []
        for i in range(len(group) - 1):
            row = group.iloc[i]
            next_rows = group[group["date"] > row["date"]].sort_values(by="date")
            if next_rows.empty:
                continue
            next_row = next_rows.iloc[0]
            valor_inicial = row["resulting_amount"] if i > 0 else row["amount"]
            valor_final = (
                next_row["amount"]
                if next_row["resulting_amount"] == 0
                else next_row["resulting_amount"]
            )
            cash_flow = (
                -next_row["amount"]
                if next_row["action"] == "Buy"
                else next_row["amount"] if next_row["resulting_amount"] != 0 else 0
            )
            retorno = (valor_final + cash_flow - valor_inicial) / valor_inicial
            results.append(
                {
                    "date": row["date"],
                    "trade_id": row["trade_id"],
                    "asset": row["asset"],
                    "valor_inicial": valor_inicial,
                    "valor_final": valor_final,
                    "cash_flow": cash_flow,
                    "retorno": retorno,
                }
            )
        result_df = pd.DataFrame(results)
        twrr = []
        for i, row in result_df.iterrows():
            twrr.append(
                (1 + row["retorno"]) if i == 0 else twrr[i - 1] * (1 + row["retorno"])
            )
        result_df["twrr_acumulado"] = [value - 1 for value in twrr]
        return result_df

    def _metrics(self, df):
        returns = df["value"].pct_change().dropna()
        returns = returns.sort_index()
        start_date = df.index[0]
        end_date = df.index[-1]
        initial_value = df["value"].iloc[0]
        final_value = df["value"].iloc[-1]
        days = (end_date - start_date).days
        years = days / 360
        cagr = ((final_value / initial_value) ** (1 / years)) - 1
        cagr_percentage = cagr * 100
        current_year = end_date.year
        max_drawdown = self._maximum_drawdown(df_equity=df)
        value_at_risk_var = self._Value_at_Risk_VaR(df_equity=df)
        sharpe_ratio = self._Sharpe_ratio(df_equity=df)
        sortino_ratio = self._Sortino_ratio(df_equity=df)
        skew = self._Skew(df_equity=df)
        kurtosis = self._Kurtosis(df_equity=df)
        volatility_ann_percent = self.Volatility_Ann_Percent(df_equity=df)
        conditional_var_cvar = self._Conditional_Value_at_Risk_VaR(df_equity=df)
        cumulative_return_percent = (
            (final_value - initial_value) / initial_value
        ) * 100
        self.cumulative_return_percent = (
            cumulative_return_percent  # Use the snake_case variable here
        )
        calmar = cagr_percentage / abs(max_drawdown) if max_drawdown != 0 else None
        treynor_index = self._calculate_treynor_index(df_equity=df)
        beta = self._calculate_beta(df_equity=df)
        recovery_factor = (
            abs(cumulative_return_percent) / abs(max_drawdown)
            if max_drawdown != 0
            else None
        )  # Use snake_case here
        risk_parity = self._calculate_risk_parity(df_equity=df)
        MDD_mean = self._calculate_MDD_mean(df_equity=df)
        MDD_Recovery_time = self._MDD_Recovery_Time(df_equity=df)
        omega = self._calculate_omega_ratio(df_equity=df)
        ulcer_index = self._calculate_ulcer_index(df_equity=df)
        tail_ratio = self._calculate_tail_ratio(df_equity=df)
        gain_pain = self._calculate_gain_to_pain_ratio(df_equity=df)
        ytd_returns = returns.loc[f"{current_year}-01-01":].sum() * 100  # Year to Date
        one_year_returns = (
            returns.loc[f"{current_year - 1}" :f"{current_year - 1}-12-31"].sum() * 100
        )  # Last Year
        two_year_returns = (
            returns.loc[f"{current_year - 2}" :f"{current_year - 1}-12-31"].sum() * 100
        )  # Two Years
        hit_rate = (returns > 0).sum() / len(returns) * 100
        equity_start_date = self.equity_data.index[0].strftime("%Y-%m-%d")

        benchmark_df = self.benchmark_series.copy()  # Keep as DataFrame initially
        benchmark_df = benchmark_df.loc[benchmark_df.index >= equity_start_date]

        benchmark_series = benchmark_df["spy"]

        # Now calculate using the Series - results will be scalar
        benchmark_cumulative_return = (
            (benchmark_series.iloc[-1] / benchmark_series.iloc[0]) - 1
        ) * 100
        self.benchmark_cumulative_return = benchmark_cumulative_return

        metrics = {}
        metrics["portfolio_id"] = self.portfolio_id
        metrics["start_date"] = start_date
        metrics["end_date"] = end_date
        metrics["average_daily_value"] = df["value"].mean()
        metrics["median_daily_value"] = df["value"].median()
        metrics["max_daily_value"] = df["value"].max()
        metrics["min_daily_value"] = df["value"].min()
        metrics["cumulative_return_percent"] = cumulative_return_percent
        metrics["cagr_percent"] = cagr_percentage
        metrics["year_to_date_percent"] = ytd_returns
        metrics["last_year_percent"] = one_year_returns
        metrics["two_years_percent"] = two_year_returns
        metrics["hit_rate_percent"] = hit_rate
        metrics["value_at_risk_var"] = value_at_risk_var
        metrics["sharpe_ratio"] = sharpe_ratio
        metrics["sortino_ratio"] = sortino_ratio
        metrics["max_drawdown_percent"] = max_drawdown
        metrics["volatility_ann_percent"] = volatility_ann_percent
        metrics["conditional_var_cvar"] = conditional_var_cvar
        metrics["calmar_ratio"] = calmar
        metrics["skew"] = skew
        metrics["kurtosis"] = kurtosis
        metrics["recovery_factor"] = recovery_factor
        metrics["sp500_cumulative_return_percent"] = (
            benchmark_cumulative_return  # Use corrected name
        )
        metrics["treynor_index"] = treynor_index
        metrics["beta"] = beta
        metrics["alpha"] = self._calculate_alpha(equity_df=df)
        metrics["risk_parity"] = risk_parity
        metrics["mean_drawdown_depth"] = MDD_mean
        metrics["maximum_drawdown_recovery_time"] = MDD_Recovery_time
        metrics["omega_ratio"] = omega
        metrics["ulcer_index"] = ulcer_index
        metrics["tail_ratio"] = tail_ratio
        metrics["gain_to_pain_ratio"] = gain_pain
        return pd.DataFrame(metrics.items(), columns=["metric", "value"])

    def _calculate_alpha(self, equity_df):

        beta = self._calculate_beta(df_equity=equity_df)

        # Handle NaN values for beta or benchmark return (already snake_case)
        if pd.isna(beta) or pd.isna(self.benchmark_cumulative_return):
            return np.nan
        # Use snake_case attributes
        alpha = self.cumulative_return_percent - (
            self.risk_free_rate
            + beta * (self.benchmark_cumulative_return - self.risk_free_rate)
        )
        return alpha

    def _Conditional_Value_at_Risk_VaR(self, df_equity, horizon=1):
        """
        Compute the parametric Conditional Value at Risk (CVaR),
        also known as Expected Shortfall, from an equity curve.

        Parameters
        ----------
        df_equity : pd.Series or pd.DataFrame
            Portfolio equity curve indexed by date. If a DataFrame is passed,
            the first column will be used.
        horizon : int or float, optional
            Time horizon in the same units as the equity curve (default is 1 day).

        Returns
        -------
        float
            CVaR (Expected Shortfall) expressed as a positive fraction.
        """
        # 1) Extract the equity series
        if isinstance(df_equity, pd.DataFrame):
            equity = df_equity.iloc[:, 0]
        else:
            equity = df_equity.copy()

        # 2) Compute period-by-period simple returns
        returns = equity.pct_change().dropna()

        if returns.empty:
            return np.nan

        # 3) Estimate sample mean and standard deviation of returns
        mu = returns.mean()
        sigma = returns.std(ddof=1)

        # Handle cases where sigma is 0 or NaN
        if sigma == 0 or np.isnan(sigma):
            return np.nan

        # 4) Parametric (Gaussian) CVaR:
        alpha = self.confidence_level
        z = norm.ppf(alpha)
        pdf_z = norm.pdf(z)

        # 5) Scale for multi-period horizon
        scale = np.sqrt(horizon)

        # 6) Compute CVaR
        cvar = ((-mu) + sigma * pdf_z / (1 - alpha)) * scale

        return cvar

    def _Skew(self, df_equity):
        """
        Compute the skewness of the return distribution from an equity curve.

        Parameters
        ----------
        df_equity : pd.Series or pd.DataFrame
            Portfolio equity curve indexed by date. If a DataFrame is passed,
            the first column will be used.

        Returns
        -------
        float
            Sample (Fisher) skewness of the periodic returns.
        """
        # Extract the equity series
        if isinstance(df_equity, pd.DataFrame):
            equity = df_equity.iloc[:, 0]
        else:
            equity = df_equity

        # Compute period-to-period returns
        returns = equity.pct_change().dropna()

        if returns.empty:
            return np.nan

        # Return skewness (Fisher definition, normal = 0)
        return returns.skew()

    def _Kurtosis(self, df_equity):
        """
        Compute the excess kurtosis of the return distribution from an equity curve.

        Parameters
        ----------
        df_equity : pd.Series or pd.DataFrame
            Portfolio equity curve indexed by date. If a DataFrame is passed,
            the first column will be used.

        Returns
        -------
        float
            Sample excess kurtosis of the periodic returns (normal = 0).
        """
        # Extract the equity series
        if isinstance(df_equity, pd.DataFrame):
            equity = df_equity.iloc[:, 0]
        else:
            equity = df_equity

        # Compute period-to-period returns
        returns = equity.pct_change().dropna()

        if returns.empty:
            return np.nan

        # Return excess kurtosis (Fisher definition, normal = 0)
        return returns.kurtosis()

    def Volatility_Ann_Percent(self, df_equity, periods_per_year=252):
        """
        Compute the annualized volatility (as a percentage) from an equity curve.

        Parameters
        ----------
        df_equity : pd.Series or pd.DataFrame
            Portfolio equity curve indexed by date. If a DataFrame is passed,
            the first column will be used.
        periods_per_year : int, optional
            Number of return periods in a year (default 252 for daily data).

        Returns
        -------
        float
            Annualized volatility expressed as a percentage (e.g., 15.23 for 15.23%).
        """
        # Extract series if DataFrame
        if isinstance(df_equity, pd.DataFrame):
            equity = df_equity.iloc[:, 0]
        else:
            equity = df_equity.copy()

        # Calculate period-to-period returns
        returns = equity.pct_change().dropna()

        if returns.empty:
            return np.nan

        # Compute standard deviation of returns
        std_per_period = returns.std(ddof=1)

        # Handle NaN std dev
        if np.isnan(std_per_period):
            return np.nan

        # Annualize volatility
        vol_ann = std_per_period * np.sqrt(periods_per_year)

        # Return as percentage
        return vol_ann * 100

    def _Sortino_ratio(self, df_equity, periods_per_year=252):
        """
        Compute the annualized Sortino ratio from an equity curve.

        Parameters
        ----------
        df_equity : pd.Series or pd.DataFrame
            Portfolio equity curve indexed by date. If a DataFrame is passed,
            the first column will be used.
        periods_per_year : int, optional
            Number of return periods in a year (default 252 for daily data).

        Returns
        -------
        float
            Annualized Sortino ratio, using self.risk_free_rate (annual) as the benchmark.
            Returns np.nan if there are no downside deviations.
        """
        # Extract series if DataFrame
        if isinstance(df_equity, pd.DataFrame):
            equity = df_equity.iloc[:, 0]
        else:
            equity = df_equity.copy()

        # Calculate simple returns
        returns = equity.pct_change().dropna()

        if returns.empty:
            return np.nan

        # Convert annual risk-free rate to per-period
        rf_per_period = self.risk_free_rate / periods_per_year

        # Excess returns over the per-period risk-free rate
        excess_returns = returns - rf_per_period

        # Mean of excess returns
        mean_excess = excess_returns.mean()

        # Downside deviation: root mean square of negative excess returns
        downside_returns = np.minimum(excess_returns, 0)
        downside_std = np.sqrt((downside_returns**2).mean())

        # Avoid division by zero if no downside volatility
        if downside_std == 0 or np.isnan(downside_std):
            return np.nan

        # Annualize Sortino ratio
        sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)

        return sortino

    def _Sharpe_ratio(self, df_equity, periods_per_year=252):
        """
        Compute the annualized Sharpe ratio from an equity curve.

        Parameters
        ----------
        df_equity : pd.Series or pd.DataFrame
            Portfolio equity curve indexed by date. If a DataFrame is passed,
            the first column will be used.
        periods_per_year : int, optional
            Number of return periods in a year (default 252 for daily data).

        Returns
        -------
        float
            Annualized Sharpe ratio, using self.risk_free_rate (annual) as the benchmark.
        """
        # Extract series if DataFrame
        if isinstance(df_equity, pd.DataFrame):
            equity = df_equity.iloc[:, 0]
        else:
            equity = df_equity.copy()

        # Calculate simple returns
        returns = equity.pct_change().dropna()

        if returns.empty:
            return np.nan

        # Convert annual risk-free rate to per-period
        rf_per_period = self.risk_free_rate / periods_per_year

        # Excess returns over the per-period risk-free rate
        excess_returns = returns - rf_per_period

        # Mean and standard deviation of excess returns
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std(ddof=1)

        # Handle division by zero or NaN std dev
        if std_excess == 0 or np.isnan(std_excess):
            return np.nan

        # Annualize Sharpe ratio
        sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)

        return sharpe

    def _Value_at_Risk_VaR(self, df_equity, horizon=1):
        """
        Compute the parametric (Gaussian) Value at Risk (VaR) from an equity curve.

        Parameters
        ----------
        df_equity : pd.Series or pd.DataFrame
            Series or single-column DataFrame of portfolio equity indexed by date.
        horizon : int or float, optional
            Time horizon in the same units as the equity curve (default is 1 day).

        Returns
        -------
        var : float
            Value at Risk (VaR) expressed as a positive number.
        """
        # Extract Series if DataFrame
        if isinstance(df_equity, pd.DataFrame):
            equity = df_equity.iloc[:, 0]
        else:
            equity = df_equity.copy()

        # Calculate simple returns
        returns = equity.pct_change().dropna()

        if returns.empty:
            return np.nan

        # Estimate mean and standard deviation of returns
        mu = returns.mean()
        sigma = returns.std(ddof=1)

        # Handle cases where sigma is 0 or NaN
        if sigma == 0 or np.isnan(sigma):
            # If no volatility, VaR might be considered 0 or negative mean return if mu < 0
            return max(0, -mu * np.sqrt(horizon))

        # Get Z-score for the confidence level
        z = norm.ppf(1 - self.confidence_level)

        # Apply square-root-of-time scaling for multi-day horizon
        scale = np.sqrt(horizon)

        # Compute parametric VaR
        var = -(mu + sigma * z) * scale

        return var

    def _maximum_drawdown(self, df_equity):
        """
        Compute the maximum drawdown of an equity time series.

        Parameters
        ----------
        df_equity : pd.Series or pd.DataFrame
            Equity curve values indexed by datetime. If a DataFrame is provided,
            the first column is used.

        Returns
        -------
        max_dd : float
            Maximum drawdown expressed as a positive fraction (e.g. 0.20 for 20%).
        drawdowns : pd.Series
            Time series of drawdowns at each point (negative values, e.g. -0.05 for a 5% drawdown).
        """
        # Extract Series if DataFrame
        if isinstance(df_equity, pd.DataFrame):
            equity = df_equity.iloc[:, 0]
        else:
            equity = df_equity.copy()

        if equity.empty:
            return np.nan

        # Running maximum of the equity curve
        running_max = equity.cummax()

        # Drawdown series: (current_value − running_max) / running_max
        # Handle potential division by zero if running_max is 0
        drawdowns = (equity - running_max) / running_max.replace(0, np.nan)
        drawdowns.fillna(0, inplace=True)  # Fill NaNs resulting from division by zero

        # Maximum drawdown is the minimum of the drawdown series (most negative)
        max_dd = -drawdowns.min()

        return max_dd

    def _calculate_treynor_index(self, df_equity):

        spy = self.benchmark_series

        # Merge the two DataFrames on the "date" column
        merged_data = pd.merge(spy, df_equity, on="date")

        # Calculate daily returns
        merged_data["return_spy"] = merged_data["spy"].pct_change()
        merged_data["return_equity"] = merged_data["value"].pct_change()

        # Drop rows with NaN values (first row will have NaN returns)
        merged_data = merged_data.dropna()

        if merged_data.empty:
            return np.nan

        beta = self._calculate_beta(
            df_equity
        )  # Recalculate beta based on potentially aligned data

        # Calculate annualized returns
        equity_return_annualized = (1 + merged_data["return_equity"].mean()) ** 252 - 1

        # Calculate excess return of the equity over the risk-free rate
        excess_return = equity_return_annualized - self.risk_free_rate

        # Calculate the Treynor Index
        if beta == 0 or np.isnan(beta):
            return np.nan  # Avoid division by zero or NaN beta
        treynor_index = excess_return / beta

        return treynor_index

    def _calculate_beta(self, df_equity):

        spy = self.benchmark_series

        # Merge the two DataFrames on the "date" column
        merged_data = pd.merge(spy, df_equity, on="date")

        # Calculate daily returns
        merged_data["return_spy"] = merged_data["spy"].pct_change()
        merged_data["return_equity"] = merged_data["value"].pct_change()

        # Drop rows with NaN values (first row will have NaN returns)
        merged_data = merged_data.dropna()

        # Check if there is enough data to calculate covariance/variance
        if len(merged_data) < 2:
            return np.nan

        # Calculate covariance between equity and SPY returns
        covariance = np.cov(
            merged_data["return_equity"], merged_data["return_spy"], ddof=1
        )[
            0, 1
        ]  # Ensure ddof=1 for sample covariance

        variance = np.var(merged_data["return_spy"], ddof=1)

        if variance == 0 or np.isnan(variance):
            return np.nan  # Avoid division by zero or NaN variance

        beta = round(covariance / variance, 2)

        return beta

    def _calculate_risk_parity(self, df_equity):

        spy = self.benchmark_series

        # Merge the two DataFrames on the "date" column
        merged_data = pd.merge(spy, df_equity, on="date")

        # Calculate daily returns
        merged_data["return_spy"] = merged_data["spy"].pct_change()
        merged_data["return_equity"] = merged_data["value"].pct_change()

        # Drop rows with NaN values (first row will have NaN returns)
        merged_data = merged_data.dropna()

        # Check for sufficient data
        if len(merged_data) < 2:
            return np.nan

        # Calculate volatility (standard deviation of returns)
        volatility_spy = np.std(merged_data["return_spy"], ddof=1)  # Use sample std dev
        volatility_equity = np.std(
            merged_data["return_equity"], ddof=1
        )  # Use sample std dev

        # Avoid division by zero or NaN volatility
        if (
            volatility_spy == 0
            or volatility_equity == 0
            or np.isnan(volatility_spy)
            or np.isnan(volatility_equity)
        ):
            return np.nan  # Or handle as appropriate, e.g., assign 0 or 100%

        # Calculate Risk Parity weights
        weight_spy = 1 / volatility_spy
        weight_equity = 1 / volatility_equity

        # Normalize weights so they sum to 1
        total_weight = weight_spy + weight_equity
        weight_spy /= total_weight
        weight_equity /= total_weight

        # Return the weights as a dictionary
        return round(weight_equity, 2) * 100

    def _MDD_Recovery_Time(self, df_equity):
        # Ensure the DataFrame is sorted by date
        df_equity = df_equity.sort_values(by="date")

        # Calculate the cumulative maximum value (peak) up to each point
        df_equity["peak"] = df_equity["value"].cummax()

        # Calculate the drawdown at each point
        # Avoid division by zero if peak is zero
        df_equity["drawdown"] = (df_equity["value"] - df_equity["peak"]) / df_equity[
            "peak"
        ].replace(0, np.nan)
        df_equity["drawdown"].fillna(0, inplace=True)

        # Find the date (index) of the maximum drawdown (minimum drawdown value)
        max_drawdown_date = df_equity["drawdown"].idxmin()

        # Find the date of the previous peak before the maximum drawdown
        previous_peak_date = df_equity.loc[
            df_equity.index < max_drawdown_date, "peak"
        ].idxmax()

        # Find the next peak date after the maximum drawdown
        recovery_data = df_equity[df_equity.index > max_drawdown_date]
        new_peak_date = recovery_data[
            recovery_data["value"] >= df_equity.loc[previous_peak_date, "peak"]
        ].index.min()

        # Calculate the number of recovery days
        if pd.isna(new_peak_date):
            recovery_days = None  # If recovery hasn't happened yet
        else:
            recovery_days = (
                new_peak_date - previous_peak_date
            ).days  # Difference between previous peak and new peak

        return recovery_days

    def _calculate_MDD_mean(self, df_equity):

        column = "value"

        serie = df_equity[column]

        rolling_max = serie.expanding(min_periods=1).max()

        # Calcular drawdown en cada punto
        drawdown = serie / rolling_max - 1

        # Identificar períodos de drawdown (cuando el drawdown es negativo)
        drawdown_periods = drawdown < 0

        # Inicializar lista para almacenar los maximum drawdowns individuales
        max_drawdowns = []

        current_drawdown = 0
        in_drawdown = False

        # Iterar sobre la serie de drawdown para detectar cada episodio de drawdown
        for dd in drawdown:
            if dd < 0:
                current_drawdown = min(current_drawdown, dd)
                in_drawdown = True
            else:
                if in_drawdown:  # Si hubo un drawdown, lo guardamos
                    max_drawdowns.append(current_drawdown)
                current_drawdown = 0
                in_drawdown = False

        # Si la serie termina en drawdown, agregamos el último
        if in_drawdown:
            max_drawdowns.append(current_drawdown)

        # Calcular el maximum drawdown promedio
        average_maximum_drawdown = (
            (sum(max_drawdowns) / len(max_drawdowns)) * 100 if max_drawdowns else 0
        )

        return average_maximum_drawdown

    def _calculate_omega_ratio(self, df_equity):

        # Calculate daily returns
        returns = df_equity["value"].pct_change()

        if returns.empty:
            return np.nan

        # Assign returns safely (without dropping rows)
        df_equity["return"] = returns

        # Now when calculating gains/losses, skip NaN returns
        valid_returns = df_equity["return"].dropna()

        gains = valid_returns[valid_returns > self.threshold].sum()
        losses = -valid_returns[valid_returns <= self.threshold].sum()

        if losses == 0 or np.isnan(losses):
            return np.nan

        omega_ratio = gains / losses
        return omega_ratio

    def _calculate_ulcer_index(self, df_equity):
        column = "value"  # Assuming 'Value' is always the column of interest

        # Calculate the running maximum
        running_max = df_equity[column].cummax()

        # Calculate percentage drawdown
        drawdowns = ((df_equity[column] - running_max) / running_max) * 100

        # Square the drawdowns
        squared_drawdowns = drawdowns**2

        # Calculate the Ulcer Index
        ulcer_index = round(np.sqrt(squared_drawdowns.mean()), 2)

        return ulcer_index

    def _calculate_tail_ratio(self, df_equity):
        column = "value"  # Assuming 'Value' is the column of interest

        # Calculate returns
        df_equity["returns"] = df_equity[column].pct_change()

        # Handle case with insufficient data
        if len(df_equity["returns"]) < 10:  # Need enough data for reliable percentiles
            return np.nan

        # Remove NaN values from the returns column
        returns = df_equity["returns"].dropna()

        # Determine the 95th and 5th percentiles (Common definition for tail ratio)
        positive_tail_threshold = np.percentile(returns, 95)
        negative_tail_threshold = np.percentile(returns, 5)

        # Extract positive and negative tails
        positive_tail = returns[returns > positive_tail_threshold]
        negative_tail = returns[returns < negative_tail_threshold]

        # Calculate average positive and average absolute negative tails
        avg_positive_tail = positive_tail.mean()
        avg_negative_tail = abs(negative_tail.mean())

        # Calculate Tail Ratio
        tail_ratio = round(
            avg_positive_tail / avg_negative_tail if avg_negative_tail != 0 else np.nan,
            2,
        )

        return tail_ratio

    def _calculate_gain_to_pain_ratio(self, df_equity):

        # Calculate daily returns
        column = "value"

        df_equity["returns"] = df_equity[column].pct_change()

        # Remove NaN values
        returns = df_equity["returns"].dropna()

        # Calculate the sum of positive returns (Total Gain)
        sum_positive = returns[returns > 0].sum()
        # Calculate the sum of absolute negative returns (Total Pain)
        sum_negative = abs(returns[returns < 0].sum())

        # Calculate Gain to Pain Ratio
        gain_to_pain_ratio = round(
            sum_positive / sum_negative if sum_negative != 0 else np.nan, 2
        )

        return gain_to_pain_ratio
