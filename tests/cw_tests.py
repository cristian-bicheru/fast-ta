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


def rsi(data):
    rsi_test = fast_ta.momentum.RSI(data, 14)
    rsi_ref = ta.momentum.RSIIndicator(pandas.Series(data), n=14).rsi()
    diff = []
    for i in range(len(data)):
        _diff = rsi_test[i]-rsi_ref[i]
        if _diff > 1:
            diff.append(_diff)

    return diff


def cw_tests():
    cd_d = np.array(close_data, dtype=np.double)
    diff = rsi(cd_d)
    plt.subplot(1, 2, 1)
    plt.hist(diff)

    cd_f = np.array(close_data, dtype=np.float32)
    diff = rsi(cd_f)
    plt.subplot(1, 2, 2)
    plt.hist(diff)

    plt.savefig('plot.svg')


cw_tests()
