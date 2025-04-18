import pandas as pd
import numpy as np
from godolib.fast_transformers import (
    calculate_returns,
    calculate_simple_moving_average,
    calculate_relative_volatility_on_prices,
    calculate_lower_bb,
)
from godolib.filing import save_dataframes_to_excel
from godolib.data_handler import apply_function_by_groups, func_by_groups
import random
import string
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import quantstats as qs


class PortfolioSimulator:
    """
    A simulator for managing and evaluating a financial portfolio.

    Attributes:
        initial_cash (float): Initial cash available for investments.
        target_weight (float): Target allocation weight for the portfolio.
        df (DataFrame): A DataFrame containing historical prices of assets.
        id_structure (str, optional): A template for generating unique trade IDs.
        manager (object, optional): An object responsible for risk management. Must implement `check_prospects` and `check_risk` methods.
        evaluator (object, optional): An object for portfolio evaluation. Must implement `calculate_metrics` and optionally `evaluate_trades` methods.
        seed (int, optional): A seed for random number generation to ensure reproducibility.
        verbose (int): Verbosity level (0 for silent, 1 for monitoring outputs).
        portfolio_id (str, optional): A unique identifier for the portfolio.
        liquid_money (float): Current uninvested cash.
        portfolio_value (float): Total value of the portfolio (cash + investments).
        history (list): A log of all portfolio actions and changes.
        balancing_dates (list): dates when rebalancing occurred.
        trades (list): A log of trades, including buy and sell actions.
        positions (dict): Current holdings in the portfolio with allocation and amount details.
        holdings (dict): Historical record of holdings for each date.
    """

    def __init__(
        self,
        initial_cash,
        target_weight,
        df,
        id_structure=None,
        manager=None,
        evaluator=None,
        seed=None,
        verbose=1,
        portfolio_id=None,
    ):
        """
        Initialize the PortfolioSimulator.

        Args:
            initial_cash (float): Initial cash for the portfolio.
            target_weight (float): Target allocation weight for the portfolio.
            df (DataFrame): A DataFrame containing historical prices of assets.
            id_structure (str, optional): Template for generating unique trade IDs.
            manager (object, optional): Risk manager object with `check_prospects` and `check_risk` methods.
            evaluator (object, optional): Portfolio evaluator object with a `calculate_metrics` method. Optionally, `evaluate_trades`.
            seed (int, optional): Seed for random number generation.
            verbose (int): Verbosity level (0 for silent, 1 for monitoring outputs).
            portfolio_id (str, optional): Unique identifier for the portfolio.

        Raises:
            ValueError: If required attributes for `manager` or `evaluator` are missing.
        """
        manager_required_attributes = ["evaluate_risk"]
        evaluator_required_attributes = ["calculate_metrics"]
        if manager:
            missing_attributes = [
                attr
                for attr in manager_required_attributes
                if not hasattr(manager, attr)
            ]
            if missing_attributes:
                raise ValueError(
                    f"Manager object is missing required attribtes: {', '.join(missing_attributes)}"
                )
        if evaluator:
            if not portfolio_id:
                raise ValueError("Evaluator object needs a portfolio_id")
            missing_attributes = [
                attr
                for attr in evaluator_required_attributes
                if not hasattr(evaluator, attr)
            ]
            if missing_attributes:
                raise ValueError(
                    f"Evaluator object is missing required attribtes: {', '.join(missing_attributes)}"
                )
        if verbose not in [0, 1, 2]:
            raise ValueError(
                "Verbose parameter must be 0 (silent),  1 (monitor), 2(management)"
            )
        self.initial_cash = initial_cash
        self.liquid_money = initial_cash
        self.portfolio_value = initial_cash
        self.target_weight = target_weight

        self.manager = manager
        self.df = df
        self.id_structure = id_structure
        self.verbose = verbose
        self.history = []
        self.balancing_dates = []
        self.trades = []
        self.positions = {}
        self.holdings = {}
        self.portfolio_id = portfolio_id
        self.evaluator = evaluator

        if seed is not None:
            random.seed(seed)

    def simulate(self, signals):
        """
        Simulate portfolio management over time based on buy signals.

        Args:
            signals (dict): A dictionary of buy signals where keys are dates and values are lists of assets to buy on those dates.

        Details:
            - Iterates over provided dates, adjusting the portfolio based on risk evaluation and buy signals.
            - If a `manager` is provided, evaluates risk and determines whether to retain or sell assets.
            - Uses `evaluator` to calculate metrics after simulation.

        Raises:
            ValueError: If manager or evaluator does not have required methods but is used in the simulation process.
        """
        initial_date = list(signals.keys())[0]
        if len(signals[initial_date]) == 0:
            raise ValueError("Signal is not cleaned")
        self.dates = [
            date.strftime("%Y-%m-%d")
            for date in self.df.loc[self.df.index >= initial_date].index
        ]
        self.balancing_dates = list(signals.keys())
        self.value = [{"date": initial_date, "value": self.portfolio_value}]

        for date_idx, date in enumerate(self.dates):
            if self.verbose == 2:
                print(date)
            if self.verbose == 1:
                print(
                    f"\n\n\n---------------------------------{date}: {self.portfolio_value}-----------------------------------"
                )
            if date_idx == 0:

                signal_without_delisted = self._approve_signal(
                    holdings=signals[date], date=date
                )

                # if signal_without_delisted != signals[date]:
                #     print (f"154 different")

                self._rebalance(date=date, buy_signals=signal_without_delisted)
                # self._rebalance(date=date, buy_signals=signals[date])
                # if date != self.dates[-1]:
                #     self._check_for_delisted(date=date)
                self._update_history(date=date, rebalance=True)

            else:
                self._update_portfolio_value(date=date)
                self.value.append({"date": date, "value": self.portfolio_value})
                self._refresh_positions(date=date)

                if self.manager:
                    decision, details = self.manager.evaluate_risk(
                        simulator=self, date=date
                    )
                    if self.verbose == 2:
                        if len(decision) == len(details):
                            for key, value in decision.items():
                                print(f"\nDecision")
                                print(key, value)
                                print(f"{details[key]}")
                        else:
                            print("\nDecision:")
                            print(f"{decision}")
                            print(f"\nDetails:")
                            print(f"{details}")

                    sold_assets = []
                    for asset in decision:
                        if not decision[asset]:
                            sold_assets.append(asset)
                            self._sell_(asset=asset, quantity=True, date=date)
                if date != self.dates[-1]:
                    self._check_for_delisted(date=date)

                if date in list(signals.keys()):
                    if (self.manager) and (hasattr(self.manager, "evaluate_prospects")):
                        decision, details = self.manager.evaluate_prospects(
                            simulator=self, prospects=signals[date], date=date
                        )
                        accepted_prospects = [
                            asset for asset in decision if decision[asset]
                        ]
                        if self.verbose == 1:
                            print(f"accepted_prospects: {accepted_prospects}")
                        if date != self.dates[-1]:
                            signal_without_delisted = self._approve_signal(
                                holdings=accepted_prospects, date=date
                            )
                            self._rebalance(
                                date=date, buy_signals=signal_without_delisted
                            )
                        else:
                            self._rebalance(date=date, buy_signals=accepted_prospects)
                        # self._rebalance(date=date, buy_signals=accepted_prospects)
                        self._update_history(date=date, rebalance=True)
                    else:
                        if date != self.dates[-1]:
                            signal_without_delisted = self._approve_signal(
                                holdings=signals[date], date=date
                            )
                            self._rebalance(
                                date=date, buy_signals=signal_without_delisted
                            )
                        else:
                            self._rebalance(date=date, buy_signals=signals[date])
                        # self._rebalance(date=date, buy_signals=signals[date])
                        self._update_history(date=date, rebalance=True)
                else:
                    self._update_history(date=date, rebalance=False)
            self.holdings[date] = list(self.positions.keys())
        if self.id_structure:
            self._assign_ids()
        if self.evaluator:
            self.metrics = self.evaluator.calculate_metrics(simulator=self)
            self.trade_metrics = self.evaluator.evaluate_trades(simulator=self)
        data = []
        history_df = pd.DataFrame(self.history)
        for date, hs in self.holdings.items():
            for h in hs:
                data.append(
                    {
                        "date": date,
                        "holding": h,
                        "allocation": history_df.loc[
                            (history_df["date"] == date) & (history_df["asset"] == h)
                        ]["allocation"].iat[0],
                    }
                )
        self.holdings = data

    def save_to_excel(self, file_path):
        """
        Save the portfolio history, equity, trades, and holdings to an Excel file.

        Args:
            file_path (str): Path to save the Excel file.

        Details:
            - Saves multiple sheets: Logs, Equity, Trades, holdings, and optionally metrics and Trade metrics.
        """
        history = pd.DataFrame(self.history)
        equity = pd.DataFrame(self.value)
        trades = pd.DataFrame(self.trades)
        holdings = pd.DataFrame(self.holdings)
        if self.portfolio_id:
            history.insert(0, "portfolio_id", self.portfolio_id)
            equity.insert(0, "portfolio_id", self.portfolio_id)
            trades.insert(0, "portfolio_id", self.portfolio_id)
            holdings.insert(0, "portfolio_id", self.portfolio_id)
        dataframes = [history, equity, trades, holdings]
        sheet_names = ["Logs", "Equity", "Trades", "Holdings"]
        if self.evaluator:
            dataframes.append(self.metrics)
            dataframes.append(self.trade_metrics)
            sheet_names.append("Metrics")
            sheet_names.append("Trades_Metrics")
        save_dataframes_to_excel(
            dataframes=dataframes,
            sheet_names=sheet_names,
            file_name=f"{file_path}",
        )

    def _approve_signal(self, holdings, date):
        # print(f"Signal before approveal: {holdings}")
        tomorrow_date = self.dates[self.dates.index(date) + 1]
        signal_without_delisted = []
        for holding in holdings:
            if not np.isnan(self.df.loc[tomorrow_date, holding]):
                signal_without_delisted.append(holding)
        # print(f"Signal after approval: {signal_without_delisted}")
        return signal_without_delisted

    def _check_for_delisted(self, date):
        # print(f"Checking for delisted...")
        tomorrow = self.dates[self.dates.index(date) + 1]
        current_positions = list(self.positions.keys())
        for holding in current_positions:
            if np.isnan(self.df.loc[tomorrow, holding]):
                # print(f"Selling {holding}")
                self._sell_(asset=holding, quantity=True, date=date)

    def _assign_ids(self):
        _tuples_ = []
        for row in self.trades:
            current_asset = row["asset"]
            current_entry_date = row["entry_date"]
            if (current_asset, current_entry_date) not in _tuples_:
                _tuples_.append((current_asset, current_entry_date))
        ids = {}
        for i, _ in enumerate(_tuples_):
            ids[i] = self._generate_id()

        for row_idx, row in enumerate(self.trades):
            current_tuple = (row["asset"], row["entry_date"])
            idx = _tuples_.index(current_tuple)
            self.trades[row_idx].update({"trade_id": ids[idx]})

    def _refresh_positions(self, date):
        date_idx = self.dates.index(date)
        last_date_record = pd.DataFrame(self.history).loc[
            (pd.DataFrame(self.history)["date"] == self.dates[date_idx - 1])
            & (pd.DataFrame(self.history)["holding"])
        ]
        assets = last_date_record["asset"].unique().tolist()
        prices_df = self.df.loc[
            [self.dates[date_idx - 1], self.dates[date_idx]], assets
        ]
        try:
            date_return = (
                calculate_returns(array=prices_df.values, period=1).reshape(
                    -1,
                )
                + 1
            )
        except Exception as e:
            self.error = prices_df.copy()
            raise ValueError(
                f"df contains nans on active dates ({self.dates[date_idx - 1], self.dates[date_idx]})"
            )
        self.positions = {}
        for asset_idx, asset in enumerate(last_date_record["asset"].unique()):
            self.positions[asset] = {
                "allocation": last_date_record.loc[last_date_record["asset"] == asset][
                    "amount"
                ].values[0]
                * date_return[asset_idx]
                / self.portfolio_value,
                "amount": last_date_record.loc[last_date_record["asset"] == asset][
                    "amount"
                ].values[0]
                * date_return[asset_idx],
            }

    def _update_history(self, date, rebalance):
        # if date == "2022-01-28":
        #     print(f"------------------DATE: {date}")
        #     print(f"------------------LEN ASSETS: {len(self.positions)}")
        #     print(f"Rebalance day: {rebalance}")
        if rebalance:
            if len(self.history) == 0:
                for asset in self.positions:
                    self.history.append(
                        {
                            "date": date,
                            "asset": asset,
                            "group": 1,
                            "holding": True,
                            "allocation": self.positions[asset]["allocation"],
                            "amount": self.positions[asset]["amount"],
                            "asset_price": self.df.loc[date, asset],
                        }
                    )
            else:
                date_idx = self.dates.index(date)
                last_date_record = pd.DataFrame(self.history).loc[
                    (pd.DataFrame(self.history)["date"] == self.dates[date_idx - 1])
                    & (pd.DataFrame(self.history)["holding"])
                ]
                if self.dates[date_idx - 1] in self.balancing_dates:
                    last_date_record = last_date_record.loc[
                        last_date_record["group"] == 1
                    ]
                assets = last_date_record["asset"].unique().tolist()
                if len(assets) != 0:
                    prices_df = self.df.loc[
                        [self.dates[date_idx - 1], self.dates[date_idx]], assets
                    ]
                    try:
                        date_return = (
                            calculate_returns(array=prices_df.values, period=1).reshape(
                                -1,
                            )
                            + 1
                        )
                    except Exception as e:
                        self.error = prices_df.copy()
                        raise ValueError(
                            f"df contains nans on active dates ({self.dates[date_idx - 1], self.dates[date_idx]})"
                        )
                    for asset_idx, asset in enumerate(assets):
                        self.history.append(
                            {
                                "date": self.dates[date_idx - 1],
                                "asset": asset,
                                "group": 0,
                                "holding": True if asset in self.positions else False,
                                "allocation": last_date_record.loc[
                                    last_date_record["asset"] == asset
                                ]["amount"].values[0]
                                * date_return[asset_idx]
                                / self.portfolio_value,
                                "amount": last_date_record.loc[
                                    last_date_record["asset"] == asset
                                ]["amount"].values[0]
                                * date_return[asset_idx],
                                "asset_price": self.df.loc[date, asset],
                            }
                        )

                    # if len(self.position) == 0:
                    #     self.history.append({"date":date, })
                    if len(self.positions) == 0:
                        self.history.append(
                            {
                                "date": date,
                                "asset": np.nan,
                                "group": np.nan,
                                "holding": np.nan,
                                "allocation": np.nan,
                                "amount": np.nan,
                                "asset_price": np.nan,
                            }
                        )
                    else:
                        for asset in self.positions:
                            self.history.append(
                                {
                                    "date": date,
                                    "asset": asset,
                                    "group": 1,
                                    "holding": True,
                                    "allocation": self.positions[asset]["allocation"],
                                    "amount": self.positions[asset]["amount"],
                                    "asset_price": self.df.loc[date, asset],
                                }
                            )
                else:
                    if len(self.positions) == 0:
                        self.history.append(
                            {
                                "date": date,
                                "asset": np.nan,
                                "group": np.nan,
                                "holding": np.nan,
                                "allocation": np.nan,
                                "amount": np.nan,
                                "asset_price": np.nan,
                            }
                        )
                    else:
                        for asset in self.positions:
                            self.history.append(
                                {
                                    "date": date,
                                    "asset": asset,
                                    "group": 1,
                                    "holding": True,
                                    "allocation": self.positions[asset]["allocation"],
                                    "amount": self.positions[asset]["amount"],
                                    "asset_price": self.df.loc[date, asset],
                                }
                            )
        else:
            date_idx = self.dates.index(date)
            last_date_record = pd.DataFrame(self.history).loc[
                (pd.DataFrame(self.history)["date"] == self.dates[date_idx - 1])
                & (pd.DataFrame(self.history)["holding"])
            ]
            if self.dates[date_idx - 1] in self.balancing_dates:
                last_date_record = last_date_record.loc[last_date_record["group"] == 1]
            last_date_record_assets = last_date_record["asset"].unique().tolist()
            if len(last_date_record_assets) == 0:
                self.history.append(
                    {
                        "date": date,
                        "asset": np.nan,
                        "group": np.nan,
                        "holding": np.nan,
                        "allocation": np.nan,
                        "amount": np.nan,
                        "asset_price": np.nan,
                    }
                )
            else:
                prices_df = self.df.loc[
                    [self.dates[date_idx - 1], self.dates[date_idx]],
                    last_date_record_assets,
                ]
                try:
                    date_return = (
                        calculate_returns(array=prices_df.values, period=1).reshape(
                            -1,
                        )
                        + 1
                    )
                except Exception as e:
                    self.error = prices_df.copy()
                    raise ValueError(
                        f"df contains nans on active dates ({self.dates[date_idx - 1], self.dates[date_idx]})"
                    )
                sold_assets = []
                for asset in last_date_record_assets:
                    if asset not in self.positions:
                        sold_assets.append(asset)
                if len(sold_assets) == 0:
                    for asset_idx, asset in enumerate(last_date_record_assets):
                        self.history.append(
                            {
                                "date": date,
                                "asset": asset,
                                "group": 0,
                                "holding": True,
                                "allocation": last_date_record.loc[
                                    last_date_record["asset"] == asset
                                ]["amount"].values[0]
                                * date_return[asset_idx]
                                / self.portfolio_value,
                                "amount": last_date_record.loc[
                                    last_date_record["asset"] == asset
                                ]["amount"].values[0]
                                * date_return[asset_idx],
                                "asset_price": self.df.loc[self.dates[date_idx], asset],
                            }
                        )
                else:
                    for asset_idx, asset in enumerate(last_date_record_assets):
                        if asset in sold_assets:
                            self.history.append(
                                {
                                    "date": date,
                                    "asset": asset,
                                    "group": 0,
                                    "holding": False,
                                    "allocation": last_date_record.loc[
                                        last_date_record["asset"] == asset
                                    ]["amount"].values[0]
                                    * date_return[asset_idx]
                                    / self.portfolio_value,
                                    "amount": last_date_record.loc[
                                        last_date_record["asset"] == asset
                                    ]["amount"].values[0]
                                    * date_return[asset_idx],
                                    "asset_price": self.df.loc[
                                        self.dates[date_idx], asset
                                    ],
                                }
                            )
                        else:
                            self.history.append(
                                {
                                    "date": date,
                                    "asset": asset,
                                    "group": 0,
                                    "holding": True,
                                    "allocation": last_date_record.loc[
                                        last_date_record["asset"] == asset
                                    ]["amount"].values[0]
                                    * date_return[asset_idx]
                                    / self.portfolio_value,
                                    "amount": last_date_record.loc[
                                        last_date_record["asset"] == asset
                                    ]["amount"].values[0]
                                    * date_return[asset_idx],
                                    "asset_price": self.df.loc[
                                        self.dates[date_idx], asset
                                    ],
                                }
                            )

    def _update_portfolio_value(self, date):
        date_idx = self.dates.index(date)
        last_date_record = pd.DataFrame(self.history).loc[
            (pd.DataFrame(self.history)["date"] == self.dates[date_idx - 1])
            & (pd.DataFrame(self.history)["holding"])
        ]
        assets = last_date_record["asset"].unique().tolist()
        prices_df = self.df.loc[
            [self.dates[date_idx - 1], self.dates[date_idx]], assets
        ]
        try:
            date_return = (
                calculate_returns(array=prices_df.values, period=1).reshape(
                    -1,
                )
                + 1
            )
        except Exception as e:
            self.error = prices_df.copy()
            raise ValueError(
                f"df contains nans on active dates ({self.dates[date_idx - 1], self.dates[date_idx]})"
            )
        amounts = last_date_record["amount"].values
        self.portfolio_value = np.dot(amounts, date_return) + self.liquid_money

    def _rebalance(self, date, buy_signals):
        if len(buy_signals) == 0:
            current_positions = list(self.positions.keys())
            for asset in current_positions:
                self._sell_(asset=asset, quantity=True, date=date)
        else:
            target_weights = self._split_number_into_parts(
                number=self.target_weight, n=len(buy_signals)
            )
            current_positions = list(self.positions.keys())
            keeping_positions = list(set(current_positions) & set(buy_signals))
            keeping_target_weights = target_weights[: len(keeping_positions)]
            selling_positions = list(set(current_positions) - set(buy_signals))
            buying_positions = list(set(buy_signals) - set(current_positions))
            buying_target_weights = target_weights[len(keeping_positions) :]
            if len(selling_positions) != 0:
                for asset_to_sell in selling_positions:
                    self._sell_(asset=asset_to_sell, quantity=True, date=date)
            if len(keeping_positions) != 0:
                keeping_selling_positions = []
                keeping_buying_positions = []
                for asset_to_keep, target_weight in zip(
                    keeping_positions, keeping_target_weights
                ):
                    if self.positions[asset_to_keep]["allocation"] > target_weight:
                        keeping_selling_positions.append(asset_to_keep)
                    else:
                        keeping_buying_positions.append(asset_to_keep)
                keeping_positions = keeping_selling_positions + keeping_buying_positions
                for asset_to_keep, target_weight in zip(
                    keeping_positions, keeping_target_weights
                ):
                    if self.positions[asset_to_keep]["allocation"] > target_weight:
                        self._sell_(
                            asset=asset_to_keep,
                            quantity=(
                                (
                                    self.positions[asset_to_keep]["allocation"]
                                    - target_weight
                                )
                                * self.positions[asset_to_keep]["amount"]
                            )
                            / self.positions[asset_to_keep]["allocation"],
                            date=date,
                        )
                    elif self.positions[asset_to_keep]["allocation"] < target_weight:
                        self._buy_(
                            asset=asset_to_keep,
                            quantity=(
                                (
                                    target_weight
                                    * self.positions[asset_to_keep]["amount"]
                                )
                                / self.positions[asset_to_keep]["allocation"]
                            )
                            - self.positions[asset_to_keep]["amount"],
                            date=date,
                        )
            if len(buying_positions) != 0:
                buying_splits = [
                    self.portfolio_value * target_weight
                    for target_weight in buying_target_weights
                ]
                for asset_to_buy, buying_amount in zip(buying_positions, buying_splits):
                    self._buy_(asset=asset_to_buy, quantity=buying_amount, date=date)

    def _sell_(self, asset, quantity, date):
        if asset not in self.positions:
            raise ValueError(
                f"You can't sell {asset} because it's not in the portfolio."
            )
        if quantity is True:
            self.trades.append(
                {
                    "date": date,
                    "asset": asset,
                    "entry_date": self._calculate_entry_date(asset=asset, date=date),
                    "rebalance_day": True if date in self.balancing_dates else False,
                    "action": "Sell",
                    "amount": self.positions[asset]["amount"],
                    "price": self.df.loc[date, asset],
                    "shares": self.positions[asset]["amount"]
                    / self.df.loc[date, asset],
                    "resulting_amount": 0,
                }
            )
            if self.verbose == 1:
                print(f"Selling {self.positions[asset]['amount']} of {asset}")
            self.liquid_money += self.positions[asset]["amount"]
            del self.positions[asset]
        else:
            if self.positions[asset]["amount"] < quantity:
                raise ValueError(
                    f"You can't sell ${quantity} of {asset}, you only have ${self.positions[asset]['amount']}"
                )
            else:
                self.trades.append(
                    {
                        "date": date,
                        "asset": asset,
                        "entry_date": self._calculate_entry_date(
                            asset=asset, date=date
                        ),
                        "rebalance_day": (
                            True if date in self.balancing_dates else False
                        ),
                        "action": "Sell",
                        "amount": quantity,
                        "price": self.df.loc[date, asset],
                        "shares": quantity / self.df.loc[date, asset],
                        "resulting_amount": self.positions[asset]["amount"] - quantity,
                    }
                )
                self.liquid_money += quantity
                self.positions[asset]["amount"] -= quantity
                self.positions[asset]["allocation"] = (
                    self.positions[asset]["amount"] / self.portfolio_value
                )

    def _buy_(self, asset, quantity, date):
        if self.verbose == 1:
            print(f"Buying {quantity} of {asset}")
        if quantity > self.liquid_money:
            if quantity - self.liquid_money < 0.0001:
                quantity = self.liquid_money
            else:
                raise ValueError(
                    f"Cannot buy {quantity} of {asset} because the liquid money is: {self.liquid_money:.2f}"
                )
        self.trades.append(
            {
                "date": date,
                "asset": asset,
                "entry_date": self._calculate_entry_date(asset=asset, date=date),
                "rebalance_day": True if date in self.balancing_dates else False,
                "action": "Buy",
                "amount": quantity,
                "price": self.df.loc[date, asset],
                "shares": quantity / self.df.loc[date, asset],
                "resulting_amount": (
                    quantity
                    if asset not in self.positions
                    else self.positions[asset]["amount"] + quantity
                ),
            }
        )
        self.liquid_money -= quantity
        if asset in self.positions:
            self.positions[asset]["amount"] += quantity
            self.positions[asset]["allocation"] = (
                self.positions[asset]["amount"] / self.portfolio_value
            )
        else:
            self.positions[asset] = {
                "allocation": quantity / self.portfolio_value,
                "amount": quantity,
            }

    def _generate_id(self):

        possible_replacements = string.digits + string.ascii_lowercase
        modified_string = list(self.id_structure)
        for i, char in enumerate(modified_string):
            if char == "1":
                modified_string[i] = random.choice(possible_replacements)
        return "".join(modified_string)

    def _calculate_entry_date(self, asset, date):
        if date == self.dates[0]:
            return date
        date_idx = self.dates.index(date)
        previous_dates = self.dates[:date_idx]
        reversed_dates = previous_dates[::-1]
        if any(asset in holdings for holdings in self.holdings.values()):
            for date_idx, _date_ in enumerate(reversed_dates):
                date_assets = self.holdings[_date_]
                if (date_idx == 0) & (asset not in date_assets):
                    return date
                if asset not in date_assets:
                    return reversed_dates[date_idx - 1]
                if _date_ == reversed_dates[-1]:
                    return self.dates[0]
        else:
            return date

    def _split_number_into_parts(self, number, n):

        base_part = number / n
        remainder = number - base_part * n
        parts = [base_part] * n
        for i in range(int(remainder * n)):
            parts[i] += 1 / n
        return parts

    def plot_equity(self, figsize=(30, 10)):
        value = pd.DataFrame(self.value)
        value["date"] = pd.to_datetime(value["date"])
        value.set_index("date", inplace=True)
        average_value = value["value"].mean()
        fig, ax = plt.subplots(figsize=figsize, facecolor="black")
        ax.set_facecolor("black")
        ax.plot(
            value.index, value["value"], label="Equity value", color="cyan", linewidth=2
        )
        ax.axhline(
            average_value,
            color="gray",
            linestyle="--",
            linewidth=2,
            label=f"Average value: {average_value:.2f}",
        )
        ax.set_title(
            f"Portfolio Equity Over Time ({self.portfolio_id})",
            fontsize=20,
            fontweight="bold",
            color="white",
        )
        ax.set_xlabel("date", fontsize=14, color="white")
        ax.set_ylabel("Equity value", fontsize=14, color="white")
        ax.grid(True, linestyle="--", alpha=0.6, color="gray")
        fig.autofmt_xdate()
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.legend(fontsize=12, facecolor="black", edgecolor="white", labelcolor="white")

        plt.show()

        ##################################################


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
        start_date = df.index[0]
        end_date = df.index[-1]
        initial_value = df["value"].iloc[0]
        final_value = df["value"].iloc[-1]
        days = (end_date - start_date).days
        years = days / 360
        cagr = ((final_value / initial_value) ** (1 / years)) - 1
        cagr_percentage = cagr * 100
        current_year = end_date.year
        max_drawdown = qs.stats.max_drawdown(returns) * 100
        calmar = cagr_percentage / abs(max_drawdown) if max_drawdown != 0 else None
        treynor_index = self._calculate_treynor_index(df_equity=df)
        beta = self._calculate_beta(df_equity=df)
        risk_parity = self._calculate_risk_parity(df_equity=df)
        MDD_mean = self._calculate_MDD_mean(df_equity=df)
        MDD_Recovery_time = self._MDD_Recovery_Time(df_equity=df)
        omega = self._calculate_omega_ratio(df_equity=df)
        ulcer_index = self._calculate_ulcer_index(df_equity=df)
        tail_ratio = self._calculate_tail_ratio(df_equity=df)
        gain_pain = self._calculate_gain_to_pain_ratio(df_equity=df)
        ytd_returns = returns.loc[f"{current_year}-01-01":].sum() * 100  # Year to date
        one_year_returns = (
            returns.loc[f"{current_year - 1}" :f"{current_year - 1}-12-31"].sum() * 100
        )  # Last Year
        two_year_returns = (
            returns.loc[f"{current_year - 2}" :f"{current_year - 1}-12-31"].sum() * 100
        )  # Two Years
        hit_rate = (returns > 0).sum() / len(returns) * 100
        equity_start_date = self.equity_data.index[0].strftime("%Y-%m-%d")

        benchmark = self.benchmark_series.copy()
        benchmark = benchmark.loc[benchmark.index >= equity_start_date]

        benchmark_cumulative_return = (
            (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
        ) * 100

        metrics = {}
        metrics["portfolio_id"] = self.portfolio_id
        metrics["start_date"] = start_date
        metrics["end_date"] = end_date
        metrics["average_daily_value"] = df["value"].mean()
        metrics["median_daily_value"] = df["value"].median()
        metrics["max_daily_value"] = df["value"].max()
        metrics["min_daily_value"] = df["value"].min()
        metrics["cumulative_return_percent"] = (
            (final_value - initial_value) / initial_value
        ) * 100
        metrics["cagr_percent"] = cagr_percentage
        metrics["year_to_date_percent"] = ytd_returns
        metrics["last_year_percent"] = one_year_returns
        metrics["two_years_percent"] = two_year_returns
        metrics["hit_rate_percent"] = hit_rate
        metrics["value_at_risk_var"] = qs.stats.value_at_risk(returns)
        metrics["conditional_var_cvar"] = qs.stats.expected_shortfall(returns)
        metrics["sharpe_ratio"] = qs.stats.sharpe(returns)
        metrics["sortino_ratio"] = qs.stats.sortino(returns)
        metrics["max_drawdown_percent"] = max_drawdown
        metrics["volatility_ann_percent"] = (
            qs.stats.volatility(returns, annualize=True) * 100
        )
        metrics["calmar_ratio"] = calmar
        metrics["skew"] = qs.stats.skew(returns)
        metrics["kurtosis"] = qs.stats.kurtosis(returns)
        metrics["recovery_factor"] = qs.stats.recovery_factor(returns)
        metrics["sp500_cumulative_return_percent"] = benchmark_cumulative_return
        metrics["treynor_index"] = treynor_index
        metrics["beta"] = beta
        metrics["alpha"] = self._calculate_alpha(equity_df=df, beta=beta)
        metrics["risk_parity"] = risk_parity
        metrics["mean_drawdown_depth"] = MDD_mean
        metrics["maximum_drawdown_recovery_time"] = MDD_Recovery_time
        metrics["omega_ratio"] = omega
        metrics["ulcer_index"] = ulcer_index
        metrics["tail_ratio"] = tail_ratio
        metrics["gain_to_pain_ratio"] = gain_pain
        return pd.DataFrame(metrics.items(), columns=["metric", "value"])

    def _calculate_alpha(self, equity_df, beta):
        starting_date = equity_df.index[0].strftime("%Y-%m-%d")
        ending_date = equity_df.index[-1].strftime("%Y-%m-%d")
        self.benchmark_series = self.benchmark_series.loc[
            (self.benchmark_series.index >= starting_date)
            & (self.benchmark_series.index <= ending_date)
        ]
        equity_return = equity_df.iat[-1, 0] / equity_df.iat[0, 0] * 100
        benchmark_return = (
            self.benchmark_series.iat[-1] / self.benchmark_series.iat[0] * 100
        )
        return equity_return - (
            self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate)
        )

    def _calculate_treynor_index(self, df_equity):

        spy = self.benchmark_series.to_frame()
        spy.index.name = "date"
        spy.reset_index(inplace=True)

        # Merge the two DataFrames on the "date" column
        merged_data = pd.merge(spy, df_equity, on="date")

        # Calculate daily returns
        merged_data["return_spy"] = merged_data["spy"].pct_change()
        merged_data["return_equity"] = merged_data["value"].pct_change()

        # Drop rows with NaN values (first row will have NaN returns)
        merged_data = merged_data.dropna()

        beta = self._calculate_beta(df_equity)

        # Calculate annualized returns
        equity_return_annualized = (1 + merged_data["return_equity"].mean()) ** 252 - 1

        # Calculate excess return of the equity over the risk-free rate
        excess_return = equity_return_annualized - self.risk_free_rate

        # Calculate the Treynor Index
        treynor_index = excess_return / beta

        return treynor_index

    def _calculate_beta(self, df_equity):

        spy = self.benchmark_series.to_frame()
        spy.index.name = "date"
        spy.reset_index(inplace=True)

        # Merge the two DataFrames on the "date" column
        merged_data = pd.merge(spy, df_equity, on="date")

        # Calculate daily returns
        merged_data["return_spy"] = merged_data["spy"].pct_change()
        merged_data["return_equity"] = merged_data["value"].pct_change()

        # Drop rows with NaN values (first row will have NaN returns)
        merged_data = merged_data.dropna()

        # Calculate covariance between equity and spy returns
        covariance = np.cov(merged_data["return_equity"], merged_data["return_spy"])[
            0, 1
        ]

        variance = np.var(merged_data["return_spy"], ddof=1)

        beta = round(covariance / variance, 2)

        return beta

    def _calculate_risk_parity(self, df_equity):

        spy = self.benchmark_series.to_frame()
        spy.index.name = "date"
        spy.reset_index(inplace=True)

        # Merge the two DataFrames on the "date" column
        merged_data = pd.merge(spy, df_equity, on="date")

        # Calculate daily returns
        merged_data["return_spy"] = merged_data["spy"].pct_change()
        merged_data["return_equity"] = merged_data["value"].pct_change()

        # Drop rows with NaN values (first row will have NaN returns)
        merged_data = merged_data.dropna()

        # Calculate volatility (standard deviation of returns)
        volatility_spy = np.std(merged_data["return_spy"])
        volatility_equity = np.std(merged_data["return_equity"])

        # Calculate Risk Parity weights
        weight_spy = 1 / volatility_spy
        weight_equity = 1 / volatility_equity

        # Normalize weights so they sum to 1
        total_weight = weight_spy + weight_equity
        weight_spy /= total_weight
        weight_equity /= total_weight

        # return the weights as a dictionary
        return round(weight_equity, 2) * 100
        # "spy_weight": round(weight_spy,2),
        # "Equity_weight":

    def _MDD_Recovery_Time(self, df_equity):
        # Ensure the DataFrame is sorted by date
        df_equity = df_equity.sort_values(by="date")

        # Calculate the cumulative maximum value (peak) up to each point
        df_equity["peak"] = df_equity["value"].cummax()

        # Calculate the drawdown at each point
        df_equity["drawdown"] = (df_equity["value"] - df_equity["peak"]) / df_equity[
            "peak"
        ]

        # Find the date of the maximum drawdown
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
        df_equity["return"] = df_equity["value"].pct_change().dropna()

        # Calculate gains and losses relative to the threshold
        gains = df_equity["return"][df_equity["return"] > self.threshold].sum()
        losses = abs(df_equity["return"][df_equity["return"] < self.threshold].sum())

        # Handle edge case where there are no losses
        if losses == 0:
            return np.inf  # Infinite Omega Ratio if there are no losses

        # Calculate the Omega Ratio
        omega_ratio = gains / losses

        return omega_ratio

    def _calculate_ulcer_index(self, df_equity):
        column = "value"  # Assuming 'value' is always the column of interest

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
        column = "value"  # Assuming 'value' is the column of interest

        # Calculate returns
        df_equity["returns"] = df_equity[column].pct_change()

        # Remove NaN values from the returns column
        returns = df_equity["returns"].dropna()

        # Determine the 90th and 10th percentiles
        positive_tail_threshold = np.percentile(returns, 90)
        negative_tail_threshold = np.percentile(returns, 10)

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

        # Calculate the sum of positive and negative returns
        sum_positive = returns[returns > 0].sum()
        sum_negative = abs(returns[returns < 0].sum())

        # Calculate Gain to Pain Ratio
        gain_to_pain_ratio = round(
            sum_positive / sum_negative if sum_negative != 0 else np.nan, 2
        )

        return gain_to_pain_ratio
