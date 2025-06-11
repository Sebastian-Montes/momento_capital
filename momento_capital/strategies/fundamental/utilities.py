import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from collections import defaultdict


# Universal
def tickers_to_check(financial_data_s3):
    tickers_to_check = list(financial_data_s3.keys())
    return tickers_to_check


# ----------------------------------------------------------------------------


# EPS
def EPS_define_quarters_by_report_date(EPS_data, tickers_to_check):
    """
    Assigns each reportDate to the quarter in which it falls (calendar quarter),
    using the reportDate only (not the fiscal date).
    """
    quarter_ranges = {"Q1": (1, 3), "Q2": (4, 6), "Q3": (7, 9), "Q4": (10, 12)}

    quarter_dates = {}

    for ticker in tickers_to_check:
        if ticker not in EPS_data:
            continue

        reports = EPS_data[ticker]
        yearly_quarters = defaultdict(
            lambda: {"Q1": None, "Q2": None, "Q3": None, "Q4": None}
        )

        for entry in reports.values():
            report_date_str = entry.get("reportDate")
            if not report_date_str:
                continue

            report_date = datetime.strptime(report_date_str, "%Y-%m-%d")
            year = report_date.year
            month = report_date.month

            for quarter, (start, end) in quarter_ranges.items():
                if start <= month <= end:
                    if yearly_quarters[year][quarter] is None:
                        yearly_quarters[year][quarter] = report_date_str
                    break

        quarter_dates[ticker] = dict(yearly_quarters)

    return quarter_dates


# EPS
def EPS_quarter_start_report(report_date_per_ticker_per_quarter, EPS_data):
    """
    Transforms quarterly data into a DataFrame with:
    - A 'date' column set to the first day of each quarter
    - Stock symbols as column headers
    - 'surprisePercent' values from eps_data as row values
    """
    transformed_data = []

    for ticker, yearly_data in report_date_per_ticker_per_quarter.items():
        for year, quarters in yearly_data.items():
            for quarter, report_date in quarters.items():
                if report_date is None:
                    continue

                # Determine the first day of the quarter
                quarter_start_month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}[quarter]
                quarter_start_date = datetime(year, quarter_start_month, 1).strftime(
                    "%Y-%m-%d"
                )

                # Extract surprisePercent from eps_data
                fiscal_dates = EPS_data.get(ticker, {})
                surprise_percent = None
                for fiscal_date, eps_info in fiscal_dates.items():
                    if eps_info.get("reportDate") == report_date:
                        surprise_percent = eps_info.get("surprisePercent")
                        break

                transformed_data.append(
                    {
                        "date": quarter_start_date,
                        "ticker": ticker,
                        "surprisePercent": surprise_percent,
                    }
                )

    # Convert to DataFrame
    df = pd.DataFrame(transformed_data)

    # Pivot to have tickers as columns
    df_pivot = df.pivot(index="date", columns="ticker", values="surprisePercent")

    return df_pivot


def compute_rolling_zscore_fixed_window(quarter_start_date_EPS_data, window_size):
    """
    Computes rolling Z-score using a fixed window size for each ticker column.
    """
    df = quarter_start_date_EPS_data
    window = window_size
    zscore_df = df.apply(
        lambda x: (x - x.rolling(window).mean()) / x.rolling(window).std(), axis=0
    )
    zscore_dict = zscore_df.to_dict(orient="index")

    return zscore_dict


def compute_rolling_zscore_expanding(quarter_start_date_EPS_data):
    """
    Computes Z-score using an expanding (cumulative) window from the start up to each time point.
    """
    df = quarter_start_date_EPS_data
    zscore_df = df.apply(
        lambda x: (x - x.expanding().mean()) / x.expanding().std(), axis=0
    )
    zscore_dict = zscore_df.to_dict(orient="index")

    return zscore_dict


# ------------------------------------------------------------------------------------------------
# INCOME STATEMENTS OR CASHFLOW STATEMENTS


def quarter_report_dates(income_statements_or_cashflow_statements, tickers_to_check):
    """
    Extracts the first available report date for each quarter for TSLA, AXON, and PLTR.
    Ensures all four quarters exist for each year, storing None if missing.
    The function strictly assigns dates that fall within the defined quarter ranges.
    """
    quarter_dates = {}
    quarter_ranges = {
        "Q1": (1, 3),  # January - March
        "Q2": (4, 6),  # April - June
        "Q3": (7, 9),  # July - September
        "Q4": (10, 12),  # October - December
    }

    for ticker in tickers_to_check:
        if ticker not in income_statements_or_cashflow_statements:
            continue

        # Extract available report dates
        report_dates = sorted(income_statements_or_cashflow_statements[ticker].keys())

        # Organize reports by year
        yearly_reports = defaultdict(
            lambda: {"Q1": None, "Q2": None, "Q3": None, "Q4": None}
        )

        for date_str in report_dates:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            year = date_obj.year
            month = date_obj.month

            # Assign the first valid date for the correct quarter range
            for quarter, (start_month, end_month) in quarter_ranges.items():
                if (
                    yearly_reports[year][quarter] is None
                    and start_month <= month <= end_month
                ):
                    yearly_reports[year][quarter] = date_str
                    break  # Stop after assigning one quarter

        quarter_dates[ticker] = dict(yearly_reports)

    return quarter_dates


def holdings_since_2000(
    spy_historical_holdings_s3,
    etfs_holdings_s3,
    since="2000-01-01",
    nearest_to_months=[1, 4, 7, 10],
):
    if not any(isinstance(i, int) for i in nearest_to_months):
        raise ValueError("List elements of nearest_to_months must be integers")
    if any(i > 12 or i < 1 for i in nearest_to_months):
        raise ValueError("List elements of nearest_to_months must be between 1 and 12")

    filtered_spy_holdings = {
        interval: active_holdings
        for interval, active_holdings in spy_historical_holdings_s3.items()
        if pd.to_datetime(interval[:10]) >= pd.to_datetime(since)
    }
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
        for interval, holdings in spy_historical_holdings_s3.items()
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
            for holding in etfs_holdings_s3[sector]
            if holding in unique_filtered_holdings
        ]
        for sector in etfs_holdings_s3.keys()
    }

    return filtered_historical_holdings


def spy_holdings_by_quarter_start_date(holdings_since_2000, quarter_report_dates):
    """
    Filters tickers based on whether their start_date and end_date span multiple quarters.
    Ensures tickers are assigned to all relevant quarters within the date range, handling missing end_dates properly.

    Args:
        holdings_since_start_date (dict): JSON data from `holdingsQ.json`, where keys are "start_date/end_date"
                              and values are lists of tickers.
        quarterly_dates (dict): Output from `define_quarters()`, containing valid quarter end dates.

    Returns:
        dict: A dictionary where keys are quarter start dates and values are lists of tickers assigned correctly.
    """
    quarter_ranges = {
        "Q1": (1, 3),  # January - March
        "Q2": (4, 6),  # April - June
        "Q3": (7, 9),  # July - September
        "Q4": (10, 12),  # October - December
    }

    filtered_tickers = {}

    # Extract only valid quarter dates (ignoring dictionaries)
    all_quarter_dates = []
    for ticker_data in quarter_report_dates.values():
        for year_data in ticker_data.values():
            for quarter_date in year_data.values():
                if isinstance(quarter_date, str):  # Ensure it's a valid date string
                    all_quarter_dates.append(quarter_date)

    # Find the latest available quarter date
    latest_available_quarter = max(
        all_quarter_dates, key=lambda d: datetime.strptime(d, "%Y-%m-%d")
    )

    for date_range, tickers in holdings_since_2000.items():
        # Extract start and end dates safely
        dates = date_range.split("/")
        start_date_str = dates[0]
        end_date_str = dates[1] if len(dates) > 1 and dates[1] != "--" else None

        # Convert start_date to datetime
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

        # If end_date is missing, assign it to the latest available quarter
        if end_date_str:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        else:
            end_date = datetime.strptime(latest_available_quarter, "%Y-%m-%d")

        # Iterate through each year in the range
        for year in range(start_date.year, end_date.year + 1):
            for quarter, (start_month, end_month) in quarter_ranges.items():
                # Define quarter start and end dates
                q_start_date = datetime(year, start_month, 1)
                q_end_date = datetime(year, end_month, 30)  # Using 30 for simplicity

                # Check if the quarter falls within the start-end date range
                if not (end_date < q_start_date or start_date > q_end_date):
                    q_start_str = (
                        f"{year}-{str(start_month).zfill(2)}-01"  # Format as YYYY-MM-01
                    )

                    # Store tickers for the identified quarter
                    if q_start_str not in filtered_tickers:
                        filtered_tickers[q_start_str] = set()
                    filtered_tickers[q_start_str].update(tickers)

    # Convert sets to lists for final output
    filtered_tickers = {k: list(v) for k, v in filtered_tickers.items()}

    return filtered_tickers


def to_quarter_start(date_str):
    date = pd.to_datetime(date_str)
    quarter = (date.month - 1) // 3 + 1
    quarter_start_month = (quarter - 1) * 3 + 1
    quarter_start = pd.Timestamp(year=date.year, month=quarter_start_month, day=1)
    return quarter_start.strftime("%Y-%m-%d")


def to_quarter_end(date_str):
    date = pd.to_datetime(date_str)
    quarter = (date.month - 1) // 3 + 1
    # Quarter end months: Q1 → March, Q2 → June, Q3 → September, Q4 → December
    quarter_end_month = quarter * 3
    # Get last day of the quarter-end month
    quarter_end = pd.Timestamp(
        year=date.year, month=quarter_end_month, day=1
    ) + pd.offsets.MonthEnd(0)
    return quarter_end.strftime("%Y-%m-%d")


def compute_metric_growth(
    income_statements_or_cashflow_statements,
    metric,
    tickers_to_check,
    quarterly_window_size,
):
    metric_data = {}  # Dictionary: {ticker: {date: metric_value}}
    for ticker in tickers_to_check:
        if ticker not in income_statements_or_cashflow_statements:
            continue
        ticker_data = income_statements_or_cashflow_statements[ticker]
        for date, report in ticker_data.items():
            metric_value = report.get(metric)
            if ticker not in metric_data:
                metric_data[ticker] = {}
            metric_data[ticker][date] = metric_value

    growth_data = {}
    for ticker, data in metric_data.items():
        dates = sorted(data.keys())
        for i in range(len(dates) - quarterly_window_size):
            start_date = dates[i]
            end_date = dates[i + quarterly_window_size]

            start_value = None
            end_value = None
            growth_value = None
            start_value = (
                float(data[start_date]) if data[start_date] is not None else 0.0
            )
            end_value = float(data[end_date]) if data[end_date] is not None else 0.0
            if start_value == 0.0:
                if end_value == 0.0:
                    growth_value = 0.0
                else:
                    growth_value = None
            else:
                growth_value = (end_value - start_value) / start_value
            if ticker not in growth_data:
                growth_data[ticker] = {}
            growth_data[ticker][end_date] = growth_value

    return growth_data


def compute_qoq_std(sequential_growth, quarterly_window_size):
    qoq_std = {}
    for ticker, data in sequential_growth.items():
        dates = sorted(data.keys())
        stds = {}

        for i in range(len(dates) - quarterly_window_size + 1):
            window_dates = dates[i : i + quarterly_window_size]
            window_values = [data[date] for date in window_dates]
            if all(value is not None for value in window_values):
                std = np.std(
                    window_values, ddof=0
                )  # Use ddof=0 for population std, ddof=1 for sample std
                stds[window_dates[-1]] = std

        qoq_std[ticker] = stds
    return qoq_std


def compute_yy_qq_ratio(metric_growth, qoq_std):

    growth = metric_growth
    std = qoq_std

    yy_qq_ratio = {}

    for ticker in growth.keys() & std.keys():
        yy_qq_ratio[ticker] = {}
        growth_data = growth[ticker]
        std_data = std[ticker]

        for date in growth_data.keys() & std_data.keys():
            growth_value = growth_data[date]
            std_value = std_data[date]
            if (
                growth_value is not None
                and std_value is not None
                and std_value != 0
                and not math.isnan(std_value)
            ):
                yy_qq_ratio[ticker][date] = growth_value / std_value
            else:
                yy_qq_ratio[ticker][date] = None

    sorted_yy_qq = {
        ticker: {k: v for k, v in sorted(inner_dict.items())}
        for ticker, inner_dict in yy_qq_ratio.items()
    }

    processed_metric = sorted_yy_qq

    return processed_metric


def processed_metric_with_quarter_start_date_based_on_filing_date(
    processed_metric, income_statements_or_cashflow_statements
):
    # El quarter start date es basado en el filing date

    methodology_with_quarter_start_date = {}

    for ticker in (
        income_statements_or_cashflow_statements.keys() & processed_metric.keys()
    ):
        financial_data = income_statements_or_cashflow_statements[ticker]
        metric_by_date = processed_metric[ticker]

        for report_date in financial_data:
            filing_date = financial_data[report_date].get("filing_date")
            if filing_date and report_date in metric_by_date:
                quarter_start = to_quarter_start(filing_date)

                if ticker not in methodology_with_quarter_start_date:
                    methodology_with_quarter_start_date[ticker] = {}

                methodology_with_quarter_start_date[ticker][quarter_start] = (
                    metric_by_date[report_date]
                )
    return methodology_with_quarter_start_date


def filtro_para_miembros_de_spy(
    processed_metric_with_quarter_start_date_based_on_filing_date,
    spy_holdings_by_quarter_start_date,
):
    spy_filtered_processed_metric = {}

    for (
        ticker,
        date_metric_dict,
    ) in processed_metric_with_quarter_start_date_based_on_filing_date.items():
        for date, value in date_metric_dict.items():
            # Check if ticker was part of SPY on that quarter start date
            if (
                date in spy_holdings_by_quarter_start_date
                and ticker in spy_holdings_by_quarter_start_date[date]
            ):
                if ticker not in spy_filtered_processed_metric:
                    spy_filtered_processed_metric[ticker] = {}
                spy_filtered_processed_metric[ticker][date] = value

    return spy_filtered_processed_metric


def target_range_zscore(all_zscores, lowerzscore_limit, upperzscore_limit):

    # Step 1: Collect all unique dates
    all_dates = set()
    for ticker in all_zscores:
        all_dates.update(all_zscores[ticker].keys())

    # Step 2: Initialize output with empty lists
    tickers_with_target_range_zscore_on_current_date = {
        date: [] for date in sorted(all_dates)
    }

    # Step 3: Loop through tickers and their z-scores
    for ticker, date_value_dict in all_zscores.items():
        for date, value in date_value_dict.items():
            if pd.notna(value) and lowerzscore_limit <= value <= upperzscore_limit:
                tickers_with_target_range_zscore_on_current_date[date].append(ticker)

    tickers_with_target_range_zscore_on_current_date = {
        k: tickers_with_target_range_zscore_on_current_date[k]
        for k in sorted(tickers_with_target_range_zscore_on_current_date)
    }

    ajuste_de_fechas_to_quarter_end_ = {}
    for date in tickers_with_target_range_zscore_on_current_date.keys():
        new_date = to_quarter_end(date)
        ajuste_de_fechas_to_quarter_end_[new_date] = (
            tickers_with_target_range_zscore_on_current_date[date]
        )

    return ajuste_de_fechas_to_quarter_end_


def genera_watchlist(target_range_zscore):

    data_target_range_zscore = target_range_zscore

    # 1) Ordenamos las fechas del diccionario original
    sorted_dates = sorted(
        data_target_range_zscore.keys(), key=lambda d: datetime.strptime(d, "%Y-%m-%d")
    )

    watchlist = {}

    # 2) Recorremos cada fecha clave y rellenamos el intervalo hasta la siguiente
    for idx, date_str in enumerate(sorted_dates):
        start = datetime.strptime(date_str, "%Y-%m-%d")
        tickers_today = data_target_range_zscore[date_str]

        # a) Calculamos el último día que debe llevar estos tickers
        if idx < len(sorted_dates) - 1:
            next_date = datetime.strptime(sorted_dates[idx + 1], "%Y-%m-%d")
            stop = next_date - timedelta(days=1)  # día anterior al cambio
        else:
            stop = datetime.today()

        current = start
        while current <= stop:
            watchlist[current.strftime("%Y-%m-%d")] = tickers_today.copy()
            current += timedelta(days=1)

    return watchlist
