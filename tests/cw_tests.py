import pickle
import numpy as np
import pandas
import matplotlib.pyplot as plt
import ta

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import fast_ta

# this data is too large to put in the repo
close_data = pickle.load(open("DAT_ASCII_USDCAD_M1_2019.pickle", "rb"))
close_data = np.array(close_data, dtype=np.double)

def cw_tests():
    def rsi():
        rsi_test = fast_ta.momentum.RSI(close_data, 14)
        rsi_ref = ta.momentum.RSIIndicator(pandas.Series(close_data), n=14).rsi()
        diff = []
        for i in range(len(close_data)):
            _diff = rsi_test[i]-rsi_ref[i]
            if _diff > 1:
                diff.append(_diff)

        plt.hist(diff)
        plt.show()
    rsi()

cw_tests()
