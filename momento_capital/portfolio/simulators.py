import pandas as pd
import numpy as np
from momento_capital.transformers import (
    calculate_returns,
    calculate_simple_moving_average,
    calculate_relative_volatility_on_prices,
    calculate_lower_bb,
)
from momento_capital.filing import save_dataframes_to_excel
from momento_capital.utilities import apply_function_by_groups, func_by_groups
import random
import string
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
import copy
from functools import partial
import datetime as dt


import copy


class EventPortfolioSimulator1:
    def __init__(self, initial_cash, portfolio_id, verbose=0):
        self.initial_cash = initial_cash
        self.liquid_money = initial_cash
        self.portfolio_value = initial_cash
        self.history = {}
        self.current_holdings = {}
        self.equity = []
        self.trades = []
        self.portfolio_id = portfolio_id
        self.verbose = verbose

    def simulate(
        self, assets_df, holding_state, spy_holdings_mask, allocation_function
    ):
        self.holding_state = holding_state
        signal = self._holding_state_to_signal(
            spy_holdings_mask=spy_holdings_mask, allocation_function=allocation_function
        )
        self.signal = signal

        assets_df = assets_df.loc[assets_df.index >= list(signal.keys())[0]]
        self.assets_df = assets_df.copy()

        df_dates = assets_df.index.astype(str).tolist()
        signal_dates = list(signal.keys())
        if set(df_dates) != set(signal_dates):
            raise ValueError("assets df and signal dates do not match")
        self.dates = df_dates.copy()
        self.today = self.dates[0]
        if len(signal[self.today]["sell"]) > 0:
            raise ValueError("Signal first date include selling instructions")
        self._raise_for_instruction(instruction=signal[self.today])
        buying_assets = signal[self.today]["buy"]
        for i, asset in enumerate(buying_assets):

            amount_to_buy = self.portfolio_value * signal[self.today]["buy"][asset]
            if i == len(buying_assets) - 1 and amount_to_buy > self.liquid_money:
                amount_to_buy = self.liquid_money

            self._buy(
                asset=asset,
                quantity=amount_to_buy,
            )

        self._check_for_delisted()
        self.equity.append(
            {
                "date": self.today,
                "portfolio_value": self.portfolio_value,
                "liquid_money": self.liquid_money,
            }
        )
        self.history[self.today] = self.current_holdings
        for date in self.dates[1:]:
            self.today = date
            last_date = self.dates[self.dates.index(self.today) - 1]
            last_date_records = self.history[last_date]
            holdings_returns = {
                holding: (
                    self.assets_df.loc[self.today, holding]
                    / self.assets_df.loc[last_date, holding]
                )
                - 1
                for holding in last_date_records
            }
            self.current_holdings = {}
            for holding, increase in holdings_returns.items():
                self.portfolio_value += last_date_records[holding]["amount"] * increase
                self.current_holdings[holding] = {
                    "amount": last_date_records[holding]["amount"] * (1 + increase),
                }
            for holding in sorted(holdings_returns.keys()):
                self.current_holdings[holding]["allocation"] = (
                    self.current_holdings[holding]["amount"] / self.portfolio_value
                )
            self._raise_for_instruction(instruction=signal[self.today])
            selling_signal = copy.deepcopy(signal[self.today]["sell"])
            buying_signal = copy.deepcopy(signal[self.today]["buy"])
            repeating_assets = {
                asset: target_allocation
                for asset, target_allocation in buying_signal.items()
                if asset in sorted(selling_signal.keys())
            }
            selling_signal = {
                asset: allocation_to_sell
                for asset, allocation_to_sell in selling_signal.items()
                if asset not in repeating_assets
            }
            buying_signal = {
                asset: allocation_to_buy
                for asset, allocation_to_buy in buying_signal.items()
                if asset not in repeating_assets
            }
            repeating_selling_assets = {
                asset: target_allocation
                for asset, target_allocation in repeating_assets.items()
                if target_allocation < self.current_holdings[asset]["allocation"]
            }
            repeating_buying_assets = {
                asset: target_allocation
                for asset, target_allocation in repeating_assets.items()
                if target_allocation > self.current_holdings[asset]["allocation"]
            }

            if len(selling_signal) > 0:
                for holding, allocation_to_sell in selling_signal.items():
                    if allocation_to_sell == 1:
                        self._sell(asset=holding, quantity="all")
                    else:
                        self._sell(
                            asset=holding,
                            quantity=allocation_to_sell
                            * self.current_holdings[holding]["amount"],
                        )
            if len(repeating_assets) > 0:
                for holding, target_allocation in repeating_selling_assets.items():
                    allocation_diff = (
                        self.current_holdings[holding]["allocation"] - target_allocation
                    )
                    amount_to_sell = allocation_diff * self.portfolio_value
                    self._sell(
                        asset=holding,
                        quantity=amount_to_sell,
                    )
                for i, (holding, target_allocation) in enumerate(
                    repeating_buying_assets.items()
                ):

                    allocation_diff = (
                        target_allocation - self.current_holdings[holding]["allocation"]
                    )
                    amount_to_buy = allocation_diff * self.portfolio_value
                    if (
                        len(buying_signal) == 0
                        and i == len(repeating_buying_assets) - 1
                        and amount_to_buy > self.liquid_money
                    ):
                        amount_to_buy = self.liquid_money

                    self._buy(
                        asset=holding,
                        quantity=amount_to_buy,
                    )

            if len(buying_signal) > 0:  # {}
                for holding, allocation_to_buy in buying_signal.items():
                    quantity_to_buy = allocation_to_buy * self.portfolio_value
                    if (
                        holding == list(sorted(buying_signal.keys()))[-1]
                        and quantity_to_buy > self.liquid_money
                    ):
                        quantity_to_buy = self.liquid_money

                    self._buy(
                        asset=holding,
                        quantity=quantity_to_buy,
                    )

            if self.today != self.dates[-1]:
                self._check_for_delisted()
            self.history[self.today] = self.current_holdings
            self.equity.append(
                {
                    "date": self.today,
                    "portfolio_value": self.portfolio_value,
                    "liquid_money": self.liquid_money,
                }
            )
            self.holdings = self._holdings_from_history()
            self._add_id()

    def _add_id(self):
        self.holdings = [
            {"portfolio_id": self.portfolio_id, **row} for row in self.holdings
        ]
        self.equity = [
            {"portfolio_id": self.portfolio_id, **row} for row in self.equity
        ]
        self.trades = [
            {"portfolio_id": self.portfolio_id, **row} for row in self.trades
        ]

    def _sell(self, asset, quantity):
        if asset not in self.current_holdings:
            raise ValueError(
                f"You can't sell {asset} because it's not in the portfolio."
            )
        if quantity == "all":
            if self.verbose == 2:
                print(f"Selling ${self.current_holdings[asset]['amount']} of {asset}")
            self.trades.append(
                {
                    "date": self.today,
                    "asset": asset,
                    "action": "sell",
                    "amount": self.current_holdings[asset]["amount"],
                    "price": self.assets_df.loc[self.today, asset],
                    "shares": self.current_holdings[asset]["amount"]
                    / self.assets_df.loc[self.today, asset],
                    "price": self.assets_df.loc[self.today, asset],
                    "resulting_amount": 0,
                }
            )
            self.liquid_money = (
                self.liquid_money + self.current_holdings[asset]["amount"]
            )
            del self.current_holdings[asset]

        elif self.current_holdings[asset]["amount"] < quantity:
            raise ValueError(
                f"You can't sell ${quantity} of {asset}, you only have ${self.current_holdings[asset]['amount']}"
            )

        else:
            if self.verbose == 2:
                print(f"Selling ${quantity} of {asset}")

            self.trades.append(
                {
                    "date": self.today,
                    "asset": asset,
                    "action": "sell",
                    "amount": quantity,
                    "price": self.assets_df.loc[self.today, asset],
                    "shares": self.current_holdings[asset]["amount"]
                    / self.assets_df.loc[self.today, asset],
                    "price": self.assets_df.loc[self.today, asset],
                    "resulting_amount": (
                        self.current_holdings[asset]["amount"] - quantity
                    ),
                }
            )

            self.liquid_money = self.liquid_money + quantity
            self.current_holdings[asset]["amount"] -= quantity
            self.current_holdings[asset]["allocation"] = (
                self.current_holdings[asset]["amount"] / self.portfolio_value
            )

    def _buy(self, asset, quantity):
        if quantity > self.liquid_money:
            raise ValueError(
                f"Cannot buy ${quantity} of {asset} because the liquid money is: ${self.liquid_money}"
            )
        self.trades.append(
            {
                "date": self.today,
                "asset": asset,
                "action": "buy",
                "amount": quantity,
                "price": self.assets_df.loc[self.today, asset],
                "shares": quantity / self.assets_df.loc[self.today, asset],
                "price": self.assets_df.loc[self.today, asset],
                "resulting_amount": (
                    quantity
                    if asset not in self.current_holdings
                    else self.current_holdings[asset]["amount"] + quantity
                ),
            }
        )
        self.liquid_money = self.liquid_money - quantity
        if asset in self.current_holdings:
            self.current_holdings[asset]["amount"] += quantity
            self.current_holdings[asset]["allocation"] = (
                self.current_holdings[asset]["amount"] / self.portfolio_value
            )
        else:
            self.current_holdings[asset] = {
                "allocation": quantity / self.portfolio_value,
                "amount": quantity,
            }

    def _raise_for_instruction(self, instruction):
        expected_keys = {"sell", "buy"}
        if set(list(instruction.keys())) != expected_keys:
            raise ValueError("Instruction keys do not match expected ones")
        if (
            self.liquid_money == 0
            and len(instruction["sell"]) == 0
            and len(instruction["buy"]) > 0
        ):
            raise ValueError(
                "Invalid instruction: attempted to buy without available funds and with no assets to sell."
            )
        if len(instruction["sell"]) > 0:
            if any(
                asset not in self.current_holdings
                for asset in instruction["sell"].keys()
            ):
                raise ValueError(
                    "Invalid sell instruction: attempting to sell assets not currently held in the portfolio."
                )
            if any(allocation > 1 for allocation in instruction["sell"].values()):
                raise ValueError(
                    "Invalid sell instruction: one or more asset allocations exceed 1.0 (100%)."
                )
            if any(allocation < 0 for allocation in instruction["sell"].values()):
                raise ValueError(
                    "Invalid sell instruction: asset allocations cannot be negative."
                )
            if len(instruction["sell"].keys()) != len(
                list(set(instruction["sell"].keys()))
            ):
                raise ValueError(
                    "Invalid sell instruction: duplicate assets detected in the list of assets to sell."
                )
        if len(instruction["buy"]) > 0:
            if len(instruction["buy"]) != len(set(instruction["buy"].keys())):
                raise ValueError(
                    "Invalid buy instruction: duplicate assets detected in the list of assets to buy."
                )
            if sum(allocation for allocation in instruction["buy"].values()) > 1:
                raise ValueError(
                    "Invalid buy instruction: total allocation exceeds 100% of available funds."
                )
            for asset in instruction["buy"]:
                if asset not in self.assets_df.columns.tolist():
                    raise ValueError(
                        f"Mismatch on instruction: {asset} asset not in assets df"
                    )

    def plot_equity(self):
        equity_df = pd.DataFrame(self.equity)
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df.set_index("date", inplace=True)
        plt.figure(figsize=(18, 8))
        plt.plot(equity_df["portfolio_value"])

    def _check_for_delisted(self):
        tomorrow_date = self.dates[self.dates.index(self.today) + 1]
        tomorrow_active_assets = (
            self.assets_df.loc[tomorrow_date]
            .notna()
            .loc[self.assets_df.loc[tomorrow_date].notna()]
            .index.tolist()
        )
        holdings = list(self.current_holdings.keys())
        for holding in holdings:
            if holding not in tomorrow_active_assets:
                self._sell(asset=holding, quantity="all")

    def _holdings_from_history(self):
        holdings = []
        for date, date_holdings in self.history.items():
            for holding, holding_data in date_holdings.items():
                holdings.append(
                    {
                        "date": date,
                        "holding": holding,
                        "amount": holding_data["amount"],
                        "allocation": holding_data["allocation"],
                    }
                )
        return holdings

    def holding_state_to_buy_sell_masks(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        This function processes the holding state DataFrame to create buy and sell masks.
        Args:
            holding_state_df (pd.DataFrame): DataFrame indicating whether an asset is held on a given day.
        Returns:
            tuple: A tuple containing two DataFrames:
                - buys_mask: DataFrame indicating buy signals.
                - sells_mask: DataFrame indicating sell signals.
        """

        holding_state_df = self.holding_state

        buys_mask = pd.DataFrame(
            False,
            index=holding_state_df.index,
            columns=holding_state_df.columns,
            dtype=bool,
        )
        sells_mask = pd.DataFrame(
            False,
            index=holding_state_df.index,
            columns=holding_state_df.columns,
            dtype=bool,
        )

        for ticker in holding_state_df.columns:
            holding_indices = np.where(holding_state_df[ticker])[0]
            consecutive_days_in_portfolio = np.split(
                holding_indices, np.where(np.diff(holding_indices) != 1)[0] + 1
            )

            if len(holding_indices) > 0:
                buy_signal_indices = np.array([], dtype=int)
                sell_signal_indices = np.array([], dtype=int)

                for i in range(len(consecutive_days_in_portfolio)):
                    buy_signal_indices = np.append(
                        buy_signal_indices, consecutive_days_in_portfolio[i][0]
                    )
                    if consecutive_days_in_portfolio[i][-1] + 1 < len(
                        holding_state_df.index
                    ):
                        sell_signal_indices = np.append(
                            sell_signal_indices,
                            consecutive_days_in_portfolio[i][-1] + 1,
                        )

                buys_mask[ticker].iloc[buy_signal_indices] = True
                sells_mask[ticker].iloc[sell_signal_indices] = True

        return buys_mask, sells_mask

    def mask_to_signal(self):

        buy_dict = {}
        sell_dict = {}
        for date in self.buy_mask.index:
            buy_dict[date.strftime("%Y-%m-%d")] = self.buy_mask.columns[
                self.buy_mask.loc[date] == True
            ].tolist()
            sell_dict[date.strftime("%Y-%m-%d")] = self.sell_mask.columns[
                self.sell_mask.loc[date] == True
            ].tolist()

        return buy_dict, sell_dict

    def _sell_signal_processor(self) -> dict:
        """This function processes the sell signal dictionary to create a new dictionary where
        a sell signal never occurs if there is no previous buy signal for that date.
        Args:
            sell_signal (dict): A dictionary where keys are dates and values are lists of assets to sell.
            buy_signal (dict): A dictionary where keys are dates and values are lists of assets to buy.
        """

        processed_signal = {}
        assets_that_can_be_sold = []

        for date, sell_call in self.sell_signal.items():
            buy_call = self.buy_signal[date]
            confirmed_sells = []

            for sell in sell_call:
                if sell in assets_that_can_be_sold:
                    confirmed_sells.append(sell)
                    assets_that_can_be_sold.remove(sell)

            processed_signal[date] = confirmed_sells

            assets_that_can_be_sold += buy_call

        return processed_signal

    def _buy_sell_formater(
        self, spy_holdings_mask: pd.DataFrame, allocation_function: partial
    ) -> dict:
        """
        This function processes the buy and sell signals to create a rebalance signal dictionary.
        Args:
            buy_signal (dict): A dictionary where keys are dates and values are lists of assets to buy.
            sell_signal (dict): A dictionary where keys are dates and values are lists of assets to sell.
        Returns:
            dict: A dictionary where keys are dates and values are dictionaries with buy and sell signals.
        """

        buy_signal = self.buy_signal
        sell_signal = self.sell_signal

        if set(buy_signal.keys()) != set(sell_signal.keys()):
            raise ValueError("Buy and sell signals must have the same keys (dates).")

        date_first_buy = next((k for k, v in buy_signal.items() if len(v) > 0), None)
        date_first_buy = dt.datetime.strptime(date_first_buy, "%Y-%m-%d")
        dates_after_buy = [
            key
            for key in buy_signal.keys()
            if dt.datetime.strptime(key, "%Y-%m-%d") >= date_first_buy
        ]
        buy_signal = {key: buy_signal[key] for key in dates_after_buy}

        rebalance_signal = {}
        past_assets = []

        for date, assets_to_buy in buy_signal.items():
            delisted_assets = []
            nan_assets = spy_holdings_mask.loc[date, past_assets].isna()
            spy_belonging_state = spy_holdings_mask.loc[date, past_assets].dropna()

            if any(nan_assets):
                delisted_assets = (
                    spy_holdings_mask[past_assets].loc[date, nan_assets].index.tolist()
                )
                for asset in delisted_assets:
                    past_assets = [
                        holding for holding in past_assets if holding != asset
                    ]

            assets_to_drop = [
                asset for asset in sell_signal[date] if asset in past_assets
            ]

            if not all(spy_belonging_state):
                spy_delisted_assets = list(
                    set(past_assets)
                    - set(
                        spy_belonging_state.index[
                            spy_holdings_mask.loc[date, past_assets]
                        ]
                        .dropna()
                        .tolist()
                    )
                )
                assets_to_drop += spy_delisted_assets
                for asset in spy_delisted_assets:
                    past_assets = [
                        holding for holding in past_assets if holding != asset
                    ]

            current_sell_signal = {}

            for asset in assets_to_drop:
                current_sell_signal[asset] = 1
                past_assets = [holding for holding in past_assets if holding != asset]

            current_assets = past_assets + assets_to_buy

            current_buy_signal = {}
            assets_allocation = []

            if len(assets_to_buy) != 0:
                assets_allocation = allocation_function(
                    date=date, list_of_assets=current_assets
                )

                for asset, allocation in zip(current_assets, assets_allocation):
                    current_buy_signal[asset] = allocation * (current_assets).count(
                        asset
                    )

                one_diference = sum(current_buy_signal.values()) - 1

                current_buy_signal[asset] = current_buy_signal[asset] - one_diference

                for asset in past_assets:
                    if spy_holdings_mask.loc[date, asset]:
                        current_sell_signal[asset] = 1

            rebalance_signal[date] = {
                "buy": dict(sorted(current_buy_signal.items(), key=lambda x: x[0])),
                "sell": dict(sorted(current_sell_signal.items(), key=lambda x: x[0])),
            }

            past_assets += assets_to_buy

        return rebalance_signal

    def _holding_state_to_signal(
        self, spy_holdings_mask: pd.DataFrame, allocation_function: partial
    ) -> dict:
        """
        This function converts the holding state DataFrame into a buy and sell signal dictionary.
        Args:
            spy_holdings_mask (pd.DataFrame): DataFrame indicating whether an asset is held on a given day.
            allocation_function (partial): A partial function to calculate asset allocations.
        Returns:
            dict: A dictionary where keys are dates and values are dictionaries with buy and sell signals.
        """

        self.buy_mask, self.sell_mask = self.holding_state_to_buy_sell_masks()
        self.buy_signal, self.sell_signal = self.mask_to_signal()

        self.sell_signal = self._sell_signal_processor()

        processed_signal = self._buy_sell_formater(
            spy_holdings_mask=spy_holdings_mask, allocation_function=allocation_function
        )

        return processed_signal


class FreqPortfolioSimulator:
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
            series_1 = series_1.squeeze()
            series_2 = series_2.squeeze()

            series_1_starting_value = series_1.iloc[0]  # scalar
            series_2_starting_value = series_2.iloc[0]  # scalar

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
        cagr_percentage = round(cagr * 100, 2)
        current_year = end_date.year
        max_drawdown = round(self._maximum_drawdown(df_equity=df), 2)
        correlation_with_benchmark = round(self.correlation_with_benchmark(), 2)
        value_at_risk_var = round(self._Value_at_Risk_VaR(df_equity=df), 2)
        sharpe_ratio = round(self._Sharpe_ratio(df_equity=df), 2)
        sortino_ratio = round(self._Sortino_ratio(df_equity=df), 2)
        skew = round(self._Skew(df_equity=df), 2)
        kurtosis = round(self._Kurtosis(df_equity=df), 2)
        volatility_ann_percent = round(self.Volatility_Ann_Percent(df_equity=df), 2)
        conditional_var_cvar = round(
            self._Conditional_Value_at_Risk_VaR(df_equity=df), 2
        )
        cumulative_return_percent = round(
            ((final_value - initial_value) / initial_value) * 100, 2
        )
        self.cumulative_return_percent = (
            cumulative_return_percent  # Use the snake_case variable here
        )
        calmar = (
            round(cagr_percentage / abs(max_drawdown), 2) if max_drawdown != 0 else None
        )
        treynor_index = round(self._calculate_treynor_index(df_equity=df), 2)
        beta = round(self._calculate_beta(df_equity=df), 2)
        recovery_factor = (
            round(abs(cumulative_return_percent) / abs(max_drawdown), 2)
            if max_drawdown != 0
            else None
        )
        risk_parity = round(self._calculate_risk_parity(df_equity=df), 2)
        MDD_mean = round(self._calculate_MDD_mean(df_equity=df), 2)
        MDD_Recovery_time = self._MDD_Recovery_Time(df_equity=df)
        omega = round(self._calculate_omega_ratio(df_equity=df), 2)
        ulcer_index = round(self._calculate_ulcer_index(df_equity=df), 2)
        tail_ratio = round(self._calculate_tail_ratio(df_equity=df), 2)
        gain_pain = round(self._calculate_gain_to_pain_ratio(df_equity=df), 2)
        ytd_returns = round(
            returns.loc[f"{current_year}-01-01":].sum() * 100, 2
        )  # Year to Date
        one_year_returns = round(
            returns.loc[f"{current_year - 1}" :f"{current_year - 1}-12-31"].sum() * 100,
            2,
        )  # Last Year
        two_year_returns = round(
            returns.loc[f"{current_year - 2}" :f"{current_year - 1}-12-31"].sum() * 100,
            2,
        )  # Two Years
        hit_rate = round((returns > 0).sum() / len(returns) * 100, 2)
        equity_start_date = self.equity_data.index[0].strftime("%Y-%m-%d")
        equity_end_date = self.equity_data.index[-1].strftime("%Y-%m-%d")

        benchmark_series = self.benchmark_series.copy()  # Keep as DataFrame initially
        benchmark_series = benchmark_series.loc[
            (benchmark_series.index >= equity_start_date)
            & (benchmark_series.index <= equity_end_date)
        ]

        # Now calculate using the Series - results will be scalar
        benchmark_cumulative_return = round(
            ((benchmark_series.iat[-1] / benchmark_series.iat[0]) - 1) * 100, 2
        )
        self.benchmark_cumulative_return = benchmark_cumulative_return

        metrics = {}
        metrics["portfolio_id"] = self.portfolio_id
        metrics["start_date"] = start_date
        metrics["end_date"] = end_date
        metrics["average_daily_value"] = round(df["value"].mean(), 2)
        metrics["median_daily_value"] = round(df["value"].median(), 2)
        metrics["max_daily_value"] = round(df["value"].max(), 2)
        metrics["min_daily_value"] = round(df["value"].min(), 2)
        metrics["correlation_with_benchmark"] = correlation_with_benchmark
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

    def correlation_with_benchmark(self):
        aligned = pd.concat([self.equity_data, self.benchmark_series], axis=1).dropna()
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

        return corr

    def _calculate_alpha(self, equity_df):

        beta = self._calculate_beta(df_equity=equity_df)

        # Handle NaN values for beta or benchmark return (already snake_case)
        if pd.isna(beta) or pd.isna(self.benchmark_cumulative_return):
            return np.nan
        # Use snake_case attributes
        alpha = round(
            self.cumulative_return_percent
            - (
                self.risk_free_rate
                + beta * (self.benchmark_cumulative_return - self.risk_free_rate)
            ),
            2,
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
        max_dd = (-drawdowns.min()) * 100

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
        df_equity["drawdown"] = df_equity["drawdown"].fillna(0)

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
