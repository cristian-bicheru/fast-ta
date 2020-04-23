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
.. py:function:: fast_ta.volume.EMV(high : np.array, low : np.array, volume : np.array, n : int = 14) -> np.array
   
   Compute the `Ease of movement (EoM, EMV) Indicator`_.

   :param np.array high: High Time Series
   :param np.array low: Low Time Series
   :param np.array volume: Volume Time Series
   :param int n: Period
   :return: Ease of movement (EoM, EMV) Indicator, n-Period SMA of EMV Indicator
   :rtype: (np.array, np.array)

.. _Ease of movement (EoM, EMV) Indicator: https://school.stockcharts.com/doku.php?id=technical_indicators:ease_of_movement_emv
