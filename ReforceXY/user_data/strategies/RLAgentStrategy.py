import logging
from functools import cached_property, reduce

# import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)

ACTION_COLUMN = "&-action"


class RLAgentStrategy(IStrategy):
    """
    RLAgentStrategy
    """

    INTERFACE_VERSION = 3

    minimal_roi = {"0": 0.03}

    process_only_new_candles = True

    stoploss = -0.02
    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.011
    trailing_only_offset_is_reached = True

    use_exit_signal = True

    startup_candle_count: int = 300

    # @cached_property
    # def protections(self):
    #     fit_live_predictions_candles = self.freqai_info.get(
    #         "fit_live_predictions_candles", 100
    #     )
    #     return [
    #         {"method": "CooldownPeriod", "stop_duration_candles": 2},
    #         {
    #             "method": "MaxDrawdown",
    #             "lookback_period_candles": fit_live_predictions_candles,
    #             "trade_limit": self.config.get("max_open_trades"),
    #             "stop_duration_candles": fit_live_predictions_candles,
    #             "max_allowed_drawdown": 0.2,
    #         },
    #         {
    #             "method": "StoplossGuard",
    #             "lookback_period_candles": fit_live_predictions_candles,
    #             "trade_limit": 1,
    #             "stop_duration_candles": fit_live_predictions_candles,
    #             "only_per_pair": True,
    #         },
    #     ]

    @cached_property
    def can_short(self):
        return self.is_short_allowed()

    # def feature_engineering_expand_all(
    #     self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    # ):
    #     dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)

    #     return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ):
        dataframe["%-close_pct_change"] = dataframe.get("close").pct_change()
        dataframe["%-raw_volume"] = dataframe.get("volume")

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ):
        dates = dataframe.get("date")
        dataframe["%-day_of_week"] = (dates.dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dates.dt.hour + 1) / 25

        dataframe["%-raw_close"] = dataframe.get("close")
        dataframe["%-raw_open"] = dataframe.get("open")
        dataframe["%-raw_high"] = dataframe.get("high")
        dataframe["%-raw_low"] = dataframe.get("low")

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs):
        dataframe[ACTION_COLUMN] = 0

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [df.get("do_predict") == 1, df.get(ACTION_COLUMN) == 1]

        df.loc[
            reduce(lambda x, y: x & y, enter_long_conditions),
            ["enter_long", "enter_tag"],
        ] = (1, "long")

        enter_short_conditions = [df.get("do_predict") == 1, df.get(ACTION_COLUMN) == 3]

        df.loc[
            reduce(lambda x, y: x & y, enter_short_conditions),
            ["enter_short", "enter_tag"],
        ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [df.get("do_predict") == 1, df.get(ACTION_COLUMN) == 2]
        df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [df.get("do_predict") == 1, df.get(ACTION_COLUMN) == 4]
        df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df

    def is_short_allowed(self) -> bool:
        trading_mode = self.config.get("trading_mode")
        if trading_mode == "margin" or trading_mode == "futures":
            return True
        elif trading_mode == "spot":
            return False
        else:
            raise ValueError(f"Invalid trading_mode: {trading_mode}")
