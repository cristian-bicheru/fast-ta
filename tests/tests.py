import numpy as np
import csv
import pickle
import ta
import pandas
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import fast_ta


close_data = pickle.load(open("DAT_ASCII_USDCAD_M1_2019.pickle", "rb"))
close_data = np.array(close_data, dtype=np.double)

def cw_tests():
    def rsi():
        rsi_test = fast_ta.momentum.RSI(close_data, 14)
        rsi_ref = ta.momentum.RSIIndicator(pandas.Series(close_data), n=14).rsi()
        diff = []
        for i in range(len(close_data)):
            diff.append(rsi_test[i]-rsi_ref[i])
        plt.plot(diff)
        plt.show()
    rsi()

def cb_tests():
    with open('AAPL.csv', 'r') as dataset:
        csv_data = list(csv.reader(dataset))[1:]

    close_data = []
    high_data = []
    low_data = []
    open_data = []

    for row in csv_data:
        close_data.append(row[1])
        high_data.append(row[2])
        low_data.append(row[3])
        open_data.append(row[4])

    close_data = np.array(close_data, dtype=np.double)
    high_data = np.array(high_data, dtype=np.double)
    low_data = np.array(low_data, dtype=np.double)
    open_data = np.array(open_data, dtype=np.double)

    def rsi():
        plt.plot(fast_ta.momentum.RSI(close_data, 14))
        plt.plot(ta.momentum.RSIIndicator(pandas.Series(close_data), n=14).rsi())
        #plt.show()
        plt.savefig("plot.svg")

    def ao():
        plt.plot(ta.momentum.AwesomeOscillatorIndicator(pandas.Series(high_data), pandas.Series(low_data)).ao())
        plt.plot(fast_ta.momentum.AO(high_data, low_data, 5, 34))
        plt.show()

    rsi()

cw_tests()
