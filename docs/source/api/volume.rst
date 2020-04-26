=================
Volume Indicators
=================

Accumulation/Distribution Index (ADI)
#####################################
.. py:function:: fast_ta.volume.ADI(high : np.array, low : np.array, close : np.array, volume : np.array) -> np.array
   
   Compute the `Accumulation/Distribution Index Indicator`_.

   :param np.array high: High Time Series
   :param np.array low: Low Time Series
   :param np.array close: Close Time Series
   :param np.array volume: Volume Time Series
   :return: Accumulation/Distribution Index Indicator
   :rtype: np.array 

.. _Accumulation/Distribution Index Indicator: https://school.stockcharts.com/doku.php?id=technical_indicators:accumulation_distribution_line


Chaikin Money Flow (CMF)
#####################################
.. py:function:: fast_ta.volume.CMF(high : np.array, low : np.array, close : np.array, volume : np.array, n : int = 20) -> np.array
   
   Compute the `Chaikin Money Flow (CMF) Indicator`_.

   :param np.array high: High Time Series
   :param np.array low: Low Time Series
   :param np.array close: Close Time Series
   :param np.array volume: Volume Time Series
   :param int n: Period
   :return: Chaikin Money Flow (CMF) Indicator
   :rtype: np.array 

.. _Chaikin Money Flow (CMF) Indicator: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf


Ease of movement (EoM, EMV)
#####################################
.. py:function:: fast_ta.volume.EMV(high : np.array, low : np.array, volume : np.array, n : int = 14) -> (np.array, np.array)
   
   Compute the `Ease of movement (EoM, EMV) Indicator`_.

   :param np.array high: High Time Series
   :param np.array low: Low Time Series
   :param np.array volume: Volume Time Series
   :param int n: Period
   :return: Ease of movement (EoM, EMV) Indicator, n-Period SMA of EMV Indicator
   :rtype: (np.array, np.array)

.. _Ease of movement (EoM, EMV) Indicator: https://school.stockcharts.com/doku.php?id=technical_indicators:ease_of_movement_emv


Force Index (FI)
#####################################
.. py:function:: fast_ta.volume.FI(close : np.array, volume : np.array, n : int = 13) -> np.array
   
   Compute the `Force Index (FI) Indicator`_.

   :param np.array close: Close Time Series
   :param np.array volume: Volume Time Series
   :param int n: Period
   :return: Force Index (FI) Indicator
   :rtype: np.array

.. _Force Index (FI) Indicator: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index


Money Flow Index (MFI)
#####################################
.. py:function:: fast_ta.volume.MFI(high : np.array, low : np.array, close : np.array, volume : np.array, n : int = 14) -> np.array
   
   Compute the `Money Flow Index (MFI) Indicator`_.

   :param np.array high: High Time Series
   :param np.array low: Low Time Series
   :param np.array close: Close Time Series
   :param np.array volume: Volume Time Series
   :param int n: Period
   :return: Money Flow Index (MFI) Indicator
   :rtype: np.array

.. _Money Flow Index (MFI) Indicator: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi

Negative Volume Index (NVI)
#####################################
.. py:function:: fast_ta.volume.NVI(close : np.array, volume : np.array) -> np.array
   
   Compute the `Negative Volume Index (NVI) Indicator`_.

   :param np.array close: Close Time Series
   :param np.array volume: Volume Time Series
   :return: Negative Volume Index (NVI) Indicator
   :rtype: np.array

.. _Negative Volume Index (NVI) Indicator: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:negative_volume_inde

On-Balance Volume (OBV)
#####################################
.. py:function:: fast_ta.volume.OBV(close : np.array, volume : np.array) -> np.array
   
   Compute the `On-Balance Volume (OBV) Indicator`_.

   :param np.array close: Close Time Series
   :param np.array volume: Volume Time Series
   :return: On-Balance Volume (OBV) Indicator
   :rtype: np.array

.. _On-Balance Volume (OBV) Indicator: https://en.wikipedia.org/wiki/On-balance_volume

Volume-Price Trend (VPT)
#####################################
.. py:function:: fast_ta.volume.VPT(close : np.array, volume : np.array) -> np.array
   
   Compute the `Volume-Price Trend (VPT) Indicator`_.

   :param np.array close: Close Time Series
   :param np.array volume: Volume Time Series
   :return: Volume-Price Trend (VPT) Indicator
   :rtype: np.array

.. _Volume-Price Trend (VPT) Indicator: https://en.wikipedia.org/wiki/Volume%E2%80%93price_trend

Volume Weighted Average Price (VWAP)
#####################################
.. py:function:: fast_ta.volume.VWAP(high : np.array, low : np.array, close : np.array, volume : np.array, n : int = 14) -> np.array
   
   Compute the `Volume Weighted Average Price (VWAP) Indicator`_.

   :param np.array high: High Time Series
   :param np.array low: Low Time Series
   :param np.array close: Close Time Series
   :param np.array volume: Volume Time Series
   :param int n: Period
   :return: Volume Weighted Average Price (VWAP) Indicator
   :rtype: np.array

.. _Volume Weighted Average Price (VWAP) Indicator: https://school.stockcharts.com/doku.php?id=technical_indicators:vwap_intraday
