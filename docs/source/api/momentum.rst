===================
Momentum Indicators
===================

Awesome Oscillator
#####################
.. py:function:: fast_ta.momentum.AO(high : np.array, low : np.array, s : int = 5, l : int = 34) -> np.array
   
   Compute the `Awesome Oscillator Indicator`_.

   :param np.array high: High Time Series
   :param np.array low: Low Time Series
   :param int s: Short Period
   :param int l: Long Period
   :return: Awesome Oscillator Indicator
   :rtype: np.array 

.. _Awesome Oscillator Indicator: https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)


KAMA
#####################
.. py:function:: fast_ta.momentum.KAMA(close : np.array, n : int = 10, f : int = 2, s : int = 30) -> np.array
   
   Compute the `KAMA Indicator`_.

   :param np.array close: Close Time Series
   :param int n: Period
   :param int f: Fast EMA Periods
   :param int s: Slow EMA Periods
   :return: KAMA Indicator
   :rtype: np.array 

.. _KAMA Indicator: https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average


ROC
#####################
.. py:function:: fast_ta.momentum.ROC(close : np.array, n : int = 12) -> np.array
   
   Compute the `ROC Indicator`_.

   :param np.array close: Close Time Series
   :param int n: Period
   :return: ROC Indicator
   :rtype: np.array 

.. _ROC Indicator: https://school.stockcharts.com/doku.php?id=technical_indicators:rate_of_change_roc_and_momentum


RSI
#####################
.. py:function:: fast_ta.momentum.RSI(close : np.array, n : int = 14, threads : int = 1) -> np.array
   
   Compute the `Relative Strength Indicator`_.

   :param np.array close: Close Time Series
   :param int n: Period
   :param int threads: Number of Threads to Use During Computation (Experimental)
   :return: Relative Strength Indicator
   :rtype: np.array

.. _Relative Strength Indicator: https://www.investopedia.com/terms/r/rsi.asp


Stochastic Oscillator
#####################
.. py:function:: fast_ta.momentum.StochasticOscillator(high : np.array, low : np.array, close : np.array, n : int = 14, d_n : int = 3) -> np.array
   
   Compute the `Stochastic Oscillator Indicator`_.

   :param np.array high: High Time Series
   :param np.array low: Low Time Series
   :param np.array close: Close Time Series
   :param int n: Period
   :param int d_n: SMA Period
   :return: Stochastic Oscillator Indicator And Signal Line
   :rtype: (np.array, np.array)

.. _Stochastic Oscillator Indicator: https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full


TSI
#####################
.. py:function:: fast_ta.momentum.TSI(close : np.array, r : int = 25, s : int = 13) -> np.array
   
   Compute the `True Strength Indicator`_.

   :param np.array close: Close Time Series
   :param int r: First EMA Period
   :param int s: Second EMA Period
   :return: True Strength Indicator
   :rtype: np.array

.. _True Strength Indicator: https://school.stockcharts.com/doku.php?id=technical_indicators:true_strength_index


Ultimate Oscillator
#####################
.. py:function:: fast_ta.momentum.UltimateOscillator(high : np.array, low : np.array, close : np.array, s : int = 7, m : int = 14, l : int = 28, ws : float = 4, wm : float = 2, wl : float = 1) -> np.array
   
   Compute the `Ultimate Oscillator Indicator`_.

   :param np.array high: High Time Series
   :param np.array low: Low Time Series
   :param np.array close: Close Time Series
   :param int s: Short Period
   :param int m: Medium Period
   :param int l: Long Period
   :param float ws: Short Period Weight
   :param float wm: Medium Period Weight
   :param float wl: Long Period Weight
   :return: Ultimate Oscillator Indicator
   :rtype: np.array

.. _Ultimate Oscillator Indicator: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ultimate_oscillator


Williams %R
#####################
.. py:function:: fast_ta.momentum.WilliamsR(high : np.array, low : np.array, close : np.array, n : int = 14) -> np.array
   
   Compute the `Williams %R Indicator`_.

   :param np.array high: High Time Series
   :param np.array low: Low Time Series
   :param np.array close: Close Time Series
   :param int n: Period
   :return: Williams %R Indicator
   :rtype: np.array

.. _Williams %R Indicator: https://school.stockcharts.com/doku.php?id=technical_indicators:williams_r
