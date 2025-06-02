import json
import pandas as pd
from io import StringIO


def fetch_s3_json(s3_client, bucket, file_key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=file_key)
        data = json.loads(response["Body"].read().decode("utf-8"))
        print(f"Loaded {file_key} from {bucket}")
        return data
    except Exception as e:
        print(f"Error loading {file_key}: {e}")
        return None


def fetch_s3_csv(s3_client, bucket, file_key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=file_key)
        csv_data = response["Body"].read().decode("utf-8")
        df = pd.read_csv(StringIO(csv_data))
        print(f"Loaded {file_key} from {bucket} into DataFrame")
        return df
    except Exception as e:
        print(f"Error loading {file_key}: {e}")
        return None


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


def get_all_adjusted_close(updated_holdings_data, updated_etfs_data):
    holdings_cols = [
        col for col in updated_holdings_data.columns if col.endswith("_Adjusted_close")
    ]
    etfs_cols = [
        col for col in updated_etfs_data.columns if col.endswith("_Adjusted_close")
    ]

    holdings_data = updated_holdings_data[["Date"] + holdings_cols].dropna(how="all")
    etfs_data = updated_etfs_data[["Date"] + etfs_cols].dropna(how="all")

    # Convert 'Date' to datetime
    holdings_data["Date"] = pd.to_datetime(holdings_data["Date"])
    etfs_data["Date"] = pd.to_datetime(etfs_data["Date"])

    # Find the first available date in 2021 for both datasets
    first_date_2021_holdings = holdings_data.loc[
        holdings_data["Date"].dt.year == 2021, "Date"
    ].min()
    first_date_2021_etfs = etfs_data.loc[
        etfs_data["Date"].dt.year == 2021, "Date"
    ].min()
    first_date_2021 = min(first_date_2021_holdings, first_date_2021_etfs)

    # Filter both datasets to start from this date
    holdings_data = holdings_data[holdings_data["Date"] >= first_date_2021]
    etfs_data = etfs_data[etfs_data["Date"] >= first_date_2021]

    # Rename columns to remove '_Adjusted_close'
    holdings_data = holdings_data.rename(
        columns={col: col.replace("_Adjusted_close", "") for col in holdings_cols}
    )
    etfs_data = etfs_data.rename(
        columns={col: col.replace("_Adjusted_close", "") for col in etfs_cols}
    )

    return holdings_data, etfs_data


def extract_active_holdings(date, spy_historical_holdings, etfs_holdings):
    date = pd.to_datetime(date)
    active_quarter = None
    for quarter, holdings in spy_historical_holdings.items():
        if not isinstance(quarter, str) or "/" not in quarter:
            continue
        start_date, end_date = quarter.split("/")
        if end_date == "--":
            end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
        start_date, end_date = map(pd.to_datetime, [start_date, end_date])
        if start_date <= date <= end_date:
            active_quarter = quarter
            break

    if not active_quarter:
        return []

    active_holdings = spy_historical_holdings[active_quarter]
    sector_holdings = {sector: [] for sector in etfs_holdings.keys()}
    for holding in active_holdings:
        for sector, sector_tickers in etfs_holdings.items():
            if holding in sector_tickers:
                sector_holdings[sector].append(holding)
    return sector_holdings


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
