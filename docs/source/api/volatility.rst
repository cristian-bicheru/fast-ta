=====================
Volatility Indicators
=====================

Average True Range (ATR)
#####################################
.. py:function:: fast_ta.volatility.ATR(high : np.array, low : np.array, close : np.array, n : int = 14) -> np.array
   
   Compute the `Average True Range (ATR) Indicator`_.

   :param np.array high: High Time Series
   :param np.array low: Low Time Series
   :param np.array close: Close Time Series
   :param int n: Period
   :return: Average True Range (ATR) Indicator
   :rtype: np.array 

.. _Average True Range (ATR) Indicator: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr


Bollinger Bands (BOL)
#####################################
.. py:function:: fast_ta.volatility.BOL(close : np.array, n : int = 20, ndev : float = 2.0) -> np.array
   
   Compute the `Bollinger Bands (BOL) Indicator`_.

   :param np.array high: High Time Series
   :param np.array low: Low Time Series
   :param np.array close: Close Time Series
   :param int n: Period
   :param int ndev: Standard Deviation Factor
   :return: Bollinger Bands (BOL) Indicator
   :rtype: np.array 

.. _Bollinger Bands (BOL) Indicator: https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands
