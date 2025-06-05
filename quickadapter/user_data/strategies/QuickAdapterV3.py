import json
import logging
from functools import reduce, cached_property
import datetime
import math
from pathlib import Path
import talib.abstract as ta
from pandas import DataFrame, Series, isna
from typing import Optional
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import stoploss_from_absolute
from technical.pivots_points import pivots_points
from freqtrade.persistence import Trade
import numpy as np
import pandas_ta as pta

from Utils import (
    alligator,
    bottom_change_percent,
    get_zl_ma_fn,
    zigzag,
    ewo,
    non_zero_diff,
    price_retracement_percent,
    vwapb,
    top_change_percent,
    get_distance,
    get_gaussian_window,
    get_odd_window,
    derive_gaussian_std_from_window,
    zero_phase_gaussian,
    zlema,
)

logger = logging.getLogger(__name__)

EXTREMA_COLUMN = "&s-extrema"
MINIMA_THRESHOLD_COLUMN = "&s-minima_threshold"
MAXIMA_THRESHOLD_COLUMN = "&s-maxima_threshold"


class QuickAdapterV3(IStrategy):
    """
    The following freqtrade strategy is released to sponsors of the non-profit FreqAI open-source project.
    If you find the FreqAI project useful, please consider supporting it by becoming a sponsor.
    We use sponsor money to help stimulate new features and to pay for running these public
    experiments, with a an objective of helping the community make smarter choices in their
    ML journey.

    This strategy is experimental (as with all strategies released to sponsors). Do *not* expect
    returns. The goal is to demonstrate gratitude to people who support the project and to
    help them find a good starting point for their own creativity.

    If you have questions, please direct them to our discord: https://discord.gg/xE4RMg4QYw

    https://github.com/sponsors/robcaulk
    """

    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "3.3.80"

    timeframe = "5m"

    stoploss = -0.02
    use_custom_stoploss = True

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.011
    trailing_only_offset_is_reached = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "limit",
        "force_exit": "limit",
        "force_entry": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 120,
        "stoploss_on_exchange_limit_ratio": 0.99,
    }

    timeframe_minutes = timeframe_to_minutes(timeframe)
    minimal_roi = {str(timeframe_minutes * 864): -1}

    process_only_new_candles = True

    @cached_property
    def can_short(self) -> bool:
        return self.is_short_allowed()

    @cached_property
    def plot_config(self) -> dict:
        return {
            "main_plot": {},
            "subplots": {
                "accuracy": {
                    "hp_rmse": {"color": "#c28ce3", "type": "line"},
                    "train_rmse": {"color": "#a3087a", "type": "line"},
                },
                "extrema": {
                    EXTREMA_COLUMN: {"color": "#f53580", "type": "line"},
                    MINIMA_THRESHOLD_COLUMN: {"color": "#4ae747", "type": "line"},
                    MAXIMA_THRESHOLD_COLUMN: {"color": "#e6be0b", "type": "line"},
                },
                "min_max": {
                    "maxima": {"color": "#0dd6de", "type": "bar"},
                    "minima": {"color": "#e3970b", "type": "bar"},
                },
            },
        }

    @cached_property
    def protections(self) -> list[dict]:
        fit_live_predictions_candles = self.freqai_info.get(
            "fit_live_predictions_candles", 100
        )
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 2},
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": fit_live_predictions_candles,
                "trade_limit": self.config.get("max_open_trades"),
                "stop_duration_candles": fit_live_predictions_candles,
                "max_allowed_drawdown": 0.2,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": fit_live_predictions_candles,
                "trade_limit": 1,
                "stop_duration_candles": fit_live_predictions_candles,
                "only_per_pair": True,
            },
        ]

    use_exit_signal = True

    @cached_property
    def startup_candle_count(self) -> int:
        # Match the predictions warmup period
        return self.freqai_info.get("fit_live_predictions_candles", 100)

    def bot_start(self, **kwargs) -> None:
        self.pairs = self.config.get("exchange", {}).get("pair_whitelist")
        if not self.pairs:
            raise ValueError(
                "FreqAI strategy requires StaticPairList method defined in pairlists configuration and 'pair_whitelist' defined in exchange section configuration"
            )
        if (
            self.freqai_info.get("identifier") is None
            or self.freqai_info.get("identifier").strip() == ""
        ):
            raise ValueError(
                "FreqAI strategy requires 'identifier' defined in the freqai section configuration"
            )
        self.models_full_path = Path(
            self.config.get("user_data_dir")
            / "models"
            / self.freqai_info.get("identifier")
        )
        self._label_params: dict[str, dict] = {}
        for pair in self.pairs:
            self._label_params[pair] = (
                self.optuna_load_best_params(pair, "label")
                if self.optuna_load_best_params(pair, "label")
                else {
                    "label_period_candles": self.freqai_info["feature_parameters"].get(
                        "label_period_candles", 50
                    ),
                    "label_natr_ratio": float(
                        self.freqai_info["feature_parameters"].get(
                            "label_natr_ratio", 6.0
                        )
                    ),
                }
            )

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ):
        highs = dataframe.get("high")
        lows = dataframe.get("low")
        closes = dataframe.get("close")
        volumes = dataframe.get("volume")

        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-aroonosc-period"] = ta.AROONOSC(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=period)
        dataframe["%-er-period"] = pta.er(closes, length=period)
        dataframe["%-rocr-period"] = ta.ROCR(dataframe, timeperiod=period)
        dataframe["%-trix-period"] = ta.TRIX(dataframe, timeperiod=period)
        dataframe["%-cmf-period"] = pta.cmf(
            highs,
            lows,
            closes,
            volumes,
            length=period,
        )
        dataframe["%-tcp-period"] = top_change_percent(dataframe, period=period)
        dataframe["%-bcp-period"] = bottom_change_percent(dataframe, period=period)
        dataframe["%-prp-period"] = price_retracement_percent(dataframe, period=period)
        dataframe["%-cti-period"] = pta.cti(closes, length=period)
        dataframe["%-chop-period"] = pta.chop(
            highs,
            lows,
            closes,
            length=period,
        )
        dataframe["%-linearreg_angle-period"] = ta.LINEARREG_ANGLE(
            dataframe, timeperiod=period
        )
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-natr-period"] = ta.NATR(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ):
        highs = dataframe.get("high")
        lows = dataframe.get("low")
        opens = dataframe.get("open")
        closes = dataframe.get("close")
        volumes = dataframe.get("volume")

        dataframe["%-close_pct_change"] = closes.pct_change()
        dataframe["%-raw_volume"] = volumes
        dataframe["%-obv"] = ta.OBV(dataframe)
        label_period_candles = self.get_label_period_candles(str(metadata.get("pair")))
        dataframe["%-atr_label_period_candles"] = ta.ATR(
            dataframe, timeperiod=label_period_candles
        )
        dataframe["%-natr_label_period_candles"] = ta.NATR(
            dataframe, timeperiod=label_period_candles
        )
        dataframe["%-ewo"] = ewo(
            dataframe=dataframe,
            pricemode="close",
            mamode="ema",
            zero_lag=True,
            normalize=True,
        )
        psar = ta.SAR(dataframe, acceleration=0.02, maximum=0.2)
        dataframe["%-diff_to_psar"] = closes - psar
        kc = pta.kc(
            highs,
            lows,
            closes,
            length=14,
            scalar=2,
        )
        dataframe["kc_lowerband"] = kc["KCLe_14_2.0"]
        dataframe["kc_middleband"] = kc["KCBe_14_2.0"]
        dataframe["kc_upperband"] = kc["KCUe_14_2.0"]
        dataframe["%-kc_width"] = (
            dataframe["kc_upperband"] - dataframe["kc_lowerband"]
        ) / dataframe["kc_middleband"]
        (
            dataframe["bb_upperband"],
            dataframe["bb_middleband"],
            dataframe["bb_lowerband"],
        ) = ta.BBANDS(
            ta.TYPPRICE(dataframe),
            timeperiod=14,
            nbdevup=2.2,
            nbdevdn=2.2,
        )
        dataframe["%-bb_width"] = (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        ) / dataframe["bb_middleband"]
        dataframe["%-ibs"] = (closes - lows) / non_zero_diff(highs, lows)
        dataframe["jaw"], dataframe["teeth"], dataframe["lips"] = alligator(
            dataframe, pricemode="median", zero_lag=True
        )
        dataframe["%-dist_to_jaw"] = get_distance(closes, dataframe["jaw"])
        dataframe["%-dist_to_teeth"] = get_distance(closes, dataframe["teeth"])
        dataframe["%-dist_to_lips"] = get_distance(closes, dataframe["lips"])
        dataframe["%-spread_jaw_teeth"] = dataframe["jaw"] - dataframe["teeth"]
        dataframe["%-spread_teeth_lips"] = dataframe["teeth"] - dataframe["lips"]
        dataframe["zlema_50"] = zlema(closes, period=50)
        dataframe["zlema_12"] = zlema(closes, period=12)
        dataframe["zlema_26"] = zlema(closes, period=26)
        dataframe["%-distzlema50"] = get_distance(closes, dataframe["zlema_50"])
        dataframe["%-distzlema12"] = get_distance(closes, dataframe["zlema_12"])
        dataframe["%-distzlema26"] = get_distance(closes, dataframe["zlema_26"])
        macd = ta.MACD(dataframe)
        dataframe["%-macd"] = macd["macd"]
        dataframe["%-macdsignal"] = macd["macdsignal"]
        dataframe["%-macdhist"] = macd["macdhist"]
        dataframe["%-dist_to_macdsignal"] = get_distance(
            dataframe["%-macd"], dataframe["%-macdsignal"]
        )
        dataframe["%-dist_to_zerohist"] = get_distance(0, dataframe["%-macdhist"])
        # VWAP bands
        (
            dataframe["vwap_lowerband"],
            dataframe["vwap_middleband"],
            dataframe["vwap_upperband"],
        ) = vwapb(dataframe, 20, 1.0)
        dataframe["%-vwap_width"] = (
            dataframe["vwap_upperband"] - dataframe["vwap_lowerband"]
        ) / dataframe["vwap_middleband"]
        dataframe["%-dist_to_vwap_upperband"] = get_distance(
            closes, dataframe["vwap_upperband"]
        )
        dataframe["%-dist_to_vwap_middleband"] = get_distance(
            closes, dataframe["vwap_middleband"]
        )
        dataframe["%-dist_to_vwap_lowerband"] = get_distance(
            closes, dataframe["vwap_lowerband"]
        )
        dataframe["%-body"] = closes - opens
        dataframe["%-tail"] = (np.minimum(opens, closes) - lows).clip(lower=0)
        dataframe["%-wick"] = (highs - np.maximum(opens, closes)).clip(lower=0)
        pp = pivots_points(dataframe)
        dataframe["r1"] = pp["r1"]
        dataframe["s1"] = pp["s1"]
        dataframe["r2"] = pp["r2"]
        dataframe["s2"] = pp["s2"]
        dataframe["r3"] = pp["r3"]
        dataframe["s3"] = pp["s3"]
        dataframe["%-dist_to_r1"] = get_distance(closes, dataframe["r1"])
        dataframe["%-dist_to_r2"] = get_distance(closes, dataframe["r2"])
        dataframe["%-dist_to_r3"] = get_distance(closes, dataframe["r3"])
        dataframe["%-dist_to_s1"] = get_distance(closes, dataframe["s1"])
        dataframe["%-dist_to_s2"] = get_distance(closes, dataframe["s2"])
        dataframe["%-dist_to_s3"] = get_distance(closes, dataframe["s3"])
        dataframe["%-raw_close"] = closes
        dataframe["%-raw_open"] = opens
        dataframe["%-raw_low"] = lows
        dataframe["%-raw_high"] = highs
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs):
        dates = dataframe.get("date")

        dataframe["%-day_of_week"] = (dates.dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dates.dt.hour + 1) / 25
        return dataframe

    def get_label_period_candles(self, pair: str) -> int:
        label_period_candles = self._label_params.get(pair, {}).get(
            "label_period_candles"
        )
        if label_period_candles and isinstance(label_period_candles, int):
            return label_period_candles
        return self.freqai_info["feature_parameters"].get("label_period_candles", 50)

    def set_label_period_candles(self, pair: str, label_period_candles: int):
        if isinstance(label_period_candles, int) and not np.isnan(label_period_candles):
            self._label_params[pair]["label_period_candles"] = label_period_candles

    def get_label_natr_ratio(self, pair: str) -> float:
        label_natr_ratio = self._label_params.get(pair, {}).get("label_natr_ratio")
        if label_natr_ratio and isinstance(label_natr_ratio, float):
            return label_natr_ratio
        return float(
            self.freqai_info["feature_parameters"].get("label_natr_ratio", 6.0)
        )

    def set_label_natr_ratio(self, pair: str, label_natr_ratio: float):
        if isinstance(label_natr_ratio, float) and not np.isnan(label_natr_ratio):
            self._label_params[pair]["label_natr_ratio"] = label_natr_ratio

    def get_entry_natr_ratio(self, pair: str) -> float:
        return self.get_label_natr_ratio(pair) * 0.0125

    def get_stoploss_natr_ratio(self, pair: str) -> float:
        return self.get_label_natr_ratio(pair) * 0.9

    def get_take_profit_natr_ratio(self, pair: str) -> float:
        return self.get_label_natr_ratio(pair) * 0.7

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs):
        pair = str(metadata.get("pair"))
        label_period_candles = self.get_label_period_candles(pair)
        label_natr_ratio = self.get_label_natr_ratio(pair)
        pivots_indices, _, pivots_directions = zigzag(
            dataframe,
            natr_period=label_period_candles,
            natr_ratio=label_natr_ratio,
        )
        dataframe[EXTREMA_COLUMN] = 0
        if len(pivots_indices) == 0:
            logger.warning(
                f"No extrema to label for pair {pair} with label_period_candles {label_period_candles} and label_natr_ratio {label_natr_ratio:.2f}"
            )
        else:
            for pivot_idx, pivot_dir in zip(pivots_indices, pivots_directions):
                dataframe.at[pivot_idx, EXTREMA_COLUMN] = pivot_dir
            dataframe["minima"] = np.where(dataframe[EXTREMA_COLUMN] == -1, -1, 0)
            dataframe["maxima"] = np.where(dataframe[EXTREMA_COLUMN] == 1, 1, 0)
        dataframe[EXTREMA_COLUMN] = self.smooth_extrema(
            dataframe[EXTREMA_COLUMN],
            self.freqai_info.get("extrema_smoothing_window", 5),
        )
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["DI_catch"] = np.where(
            dataframe.get("DI_values") > dataframe.get("DI_cutoff"),
            0,
            1,
        )

        pair = str(metadata.get("pair"))

        self.set_label_period_candles(
            pair, dataframe.get("label_period_candles").iloc[-1]
        )
        self.set_label_natr_ratio(pair, dataframe.get("label_natr_ratio").iloc[-1])

        dataframe["natr_label_period_candles"] = ta.NATR(
            dataframe, timeperiod=self.get_label_period_candles(pair)
        )

        dataframe["minima_threshold"] = dataframe.get(MINIMA_THRESHOLD_COLUMN)
        dataframe["maxima_threshold"] = dataframe.get(MAXIMA_THRESHOLD_COLUMN)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [
            df.get("do_predict") == 1,
            df.get("DI_catch") == 1,
            df.get(EXTREMA_COLUMN) < df.get("minima_threshold"),
        ]

        df.loc[
            reduce(lambda x, y: x & y, enter_long_conditions),
            ["enter_long", "enter_tag"],
        ] = (1, "long")

        enter_short_conditions = [
            df.get("do_predict") == 1,
            df.get("DI_catch") == 1,
            df.get(EXTREMA_COLUMN) > df.get("maxima_threshold"),
        ]

        df.loc[
            reduce(lambda x, y: x & y, enter_short_conditions),
            ["enter_short", "enter_tag"],
        ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        return df

    def get_trade_entry_date(self, trade: Trade) -> datetime:
        return timeframe_to_prev_date(self.config.get("timeframe"), trade.open_date_utc)

    def get_trade_duration_candles(self, df: DataFrame, trade: Trade) -> Optional[int]:
        """
        Get the number of candles since the trade entry.
        :param df: DataFrame with the current data
        :param trade: Trade object
        :return: Number of candles since the trade entry
        """
        entry_date = self.get_trade_entry_date(trade)
        dates = df.get("date")
        if dates is None or dates.empty:
            return None
        current_date = dates.iloc[-1]
        if isna(current_date):
            return None
        trade_duration_minutes = (current_date - entry_date).total_seconds() / 60.0
        return int(
            trade_duration_minutes / timeframe_to_minutes(self.config.get("timeframe"))
        )

    @staticmethod
    def is_trade_duration_valid(trade_duration: float) -> bool:
        return isinstance(trade_duration, (int, float)) and not (
            isna(trade_duration) or trade_duration <= 0
        )

    def get_trade_natr(
        self, df: DataFrame, pair: str, trade_duration_candles: int
    ) -> Optional[float]:
        if not QuickAdapterV3.is_trade_duration_valid(trade_duration_candles):
            return None
        label_natr = df.get("natr_label_period_candles")
        if label_natr is None or label_natr.empty:
            return None
        trade_natr = np.nan
        if trade_duration_candles >= 2:
            zl_kama = get_zl_ma_fn("kama")
            try:
                trade_kama_natr_values = zl_kama(
                    label_natr, timeperiod=trade_duration_candles
                )
                trade_kama_natr_values = trade_kama_natr_values[
                    ~np.isnan(trade_kama_natr_values)
                ]
                if trade_kama_natr_values.size > 0:
                    trade_natr = trade_kama_natr_values[-1]
            except Exception as e:
                logger.error(
                    f"Failed to calculate KAMA for pair {pair}: {str(e)}", exc_info=True
                )
        if isna(trade_natr):
            trade_natr = zlema(label_natr, period=trade_duration_candles).iloc[-1]
        return trade_natr

    def get_stoploss_distance(
        self, df: DataFrame, trade: Trade, current_rate: float
    ) -> Optional[float]:
        trade_duration_candles = self.get_trade_duration_candles(df, trade)
        if not QuickAdapterV3.is_trade_duration_valid(trade_duration_candles):
            return None
        trade_natr = self.get_trade_natr(df, trade.pair, trade_duration_candles)
        if isna(trade_natr) or trade_natr < 0:
            return None
        return (
            current_rate
            * (trade_natr / 100.0)
            * self.get_stoploss_natr_ratio(trade.pair)
            * (1 / math.log10(3.75 + 0.25 * trade_duration_candles))
        )

    def get_take_profit_distance(self, df: DataFrame, trade: Trade) -> Optional[float]:
        trade_duration_candles = self.get_trade_duration_candles(df, trade)
        if not QuickAdapterV3.is_trade_duration_valid(trade_duration_candles):
            return None
        trade_natr = self.get_trade_natr(df, trade.pair, trade_duration_candles)
        if isna(trade_natr) or trade_natr < 0:
            return None
        return (
            trade.open_rate
            * (trade_natr / 100.0)
            * self.get_take_profit_natr_ratio(trade.pair)
            * math.log10(9.75 + 0.25 * trade_duration_candles)
        )

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[float]:
        df, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.config.get("timeframe")
        )

        if df.empty:
            return None

        stoploss_distance = self.get_stoploss_distance(df, trade, current_rate)
        if isna(stoploss_distance) or stoploss_distance <= 0:
            return None
        sign = 1 if trade.is_short else -1
        return stoploss_from_absolute(
            current_rate + (sign * stoploss_distance),
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        df, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.config.get("timeframe")
        )

        if df.empty:
            return None

        last_candle = df.iloc[-1]
        if last_candle.get("do_predict") == 2:
            return "model_expired"
        if last_candle.get("DI_catch") == 0:
            return "outlier_detected"

        entry_tag = trade.enter_tag

        if (
            entry_tag == "short"
            and last_candle.get("do_predict") == 1
            and last_candle.get(EXTREMA_COLUMN) < last_candle.get("minima_threshold")
        ):
            return "minima_detected_short"
        if (
            entry_tag == "long"
            and last_candle.get("do_predict") == 1
            and last_candle.get(EXTREMA_COLUMN) > last_candle.get("maxima_threshold")
        ):
            return "maxima_detected_long"

        take_profit_distance = self.get_take_profit_distance(df, trade)
        if isna(take_profit_distance) or take_profit_distance <= 0:
            return None
        take_profit_price = (
            trade.open_rate + (-1 if trade.is_short else 1) * take_profit_distance
        )
        trade.set_custom_data(key="take_profit_price", value=take_profit_price)
        logger.info(
            f"Trade {trade.trade_direction} for {pair}: open price {trade.open_rate}, current price {current_rate}, TP price {take_profit_price}"
        )
        if trade.is_short:
            if current_rate <= take_profit_price:
                return "take_profit_short"
        else:
            if current_rate >= take_profit_price:
                return "take_profit_long"

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        if Trade.get_open_trade_count() >= self.config.get("max_open_trades"):
            return False
        max_open_trades_per_side = self.max_open_trades_per_side()
        if max_open_trades_per_side >= 0:
            open_trades = Trade.get_open_trades()
            trades_per_side = sum(1 for trade in open_trades if trade.enter_tag == side)
            if trades_per_side >= max_open_trades_per_side:
                return False

        df, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.config.get("timeframe")
        )
        if df.empty:
            return False
        last_candle = df.iloc[-1]
        last_candle_close = last_candle.get("close")
        last_candle_high = last_candle.get("high")
        last_candle_low = last_candle.get("low")
        last_candle_natr = last_candle.get("natr_label_period_candles")
        if isna(last_candle_natr) or last_candle_natr < 0:
            return False
        lower_bound = 0
        upper_bound = 0
        price_deviation = (last_candle_natr / 100.0) * self.get_entry_natr_ratio(pair)
        if side == "long":
            lower_bound = last_candle_low * (1 - price_deviation)
            upper_bound = last_candle_close * (1 + price_deviation)
        elif side == "short":
            lower_bound = last_candle_close * (1 - price_deviation)
            upper_bound = last_candle_high * (1 + price_deviation)
        if lower_bound < 0:
            logger.info(
                f"User denied {side} entry for {pair}: calculated lower bound {lower_bound} is below zero"
            )
            return False
        if lower_bound <= rate <= upper_bound:
            return True
        else:
            logger.info(
                f"User denied {side} entry for {pair}: rate {rate} outside bounds [{lower_bound}, {upper_bound}]"
            )
        return False

    def max_open_trades_per_side(self) -> int:
        max_open_trades = self.config.get("max_open_trades")
        if max_open_trades < 0:
            return -1
        if self.is_short_allowed():
            if max_open_trades % 2 == 1:
                max_open_trades += 1
            return int(max_open_trades / 2)
        else:
            return max_open_trades

    def is_short_allowed(self) -> bool:
        trading_mode = self.config.get("trading_mode")
        if trading_mode == "margin" or trading_mode == "futures":
            return True
        elif trading_mode == "spot":
            return False
        else:
            raise ValueError(f"Invalid trading_mode: {trading_mode}")

    def smooth_extrema(
        self,
        series: Series,
        window: int,
        std: Optional[float] = None,
    ) -> Series:
        extrema_smoothing = self.freqai_info.get("extrema_smoothing", "gaussian")
        if std is None:
            std = derive_gaussian_std_from_window(window)
        gaussian_window = get_gaussian_window(std, True)
        odd_window = get_odd_window(window)
        smoothing_methods: dict[str, Series] = {
            "gaussian": series.rolling(
                window=gaussian_window,
                win_type="gaussian",
                center=True,
            ).mean(std=std),
            "zero_phase_gaussian": zero_phase_gaussian(
                series=series, window=window, std=std
            ),
            "boxcar": series.rolling(
                window=odd_window, win_type="boxcar", center=True
            ).mean(),
            "triang": series.rolling(
                window=odd_window, win_type="triang", center=True
            ).mean(),
            "smm": series.rolling(window=odd_window, center=True).median(),
            "sma": series.rolling(window=odd_window, center=True).mean(),
            "ewma": series.ewm(span=window).mean(),
            "zlewma": zlema(series, period=window),
        }
        return smoothing_methods.get(
            extrema_smoothing,
            smoothing_methods["gaussian"],
        )

    def optuna_load_best_params(self, pair: str, namespace: str) -> Optional[dict]:
        best_params_path = Path(
            self.models_full_path
            / f"optuna-{namespace}-best-params-{pair.split('/')[0]}.json"
        )
        if best_params_path.is_file():
            with best_params_path.open("r", encoding="utf-8") as read_file:
                return json.load(read_file)
        return None
