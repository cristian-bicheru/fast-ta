import numpy as np
import csv
import ta
import pandas
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import fast_ta

with open('tests/AAPL.csv', 'r') as dataset:
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

while 'null' in close_data: close_data.remove('null')
while 'null' in high_data: high_data.remove('null')
while 'null' in low_data: low_data.remove('null')
while 'null' in open_data: open_data.remove('null')

close_data = np.array(close_data, dtype=np.double)
high_data = np.array(high_data, dtype=np.double)
low_data = np.array(low_data, dtype=np.double)
open_data = np.array(open_data, dtype=np.double)


def rsi():
    plt.clf()
    plt.title("RSI "+str(close_data.dtype))
    plt.plot(fast_ta.momentum.RSI(close_data, 14))
    plt.plot(ta.momentum.RSIIndicator(pandas.Series(close_data), n=14).rsi())
    plt.savefig("rsi " + str(close_data.dtype) + ".svg")
    
def ao():
    plt.clf()
    plt.title("AO "+str(high_data.dtype))
    plt.plot(ta.momentum.AwesomeOscillatorIndicator(pandas.Series(high_data), pandas.Series(low_data)).ao())
    plt.plot(fast_ta.momentum.AO(high_data, low_data, 5, 34))
    plt.savefig("AO " + str(high_data.dtype) + ".svg")
    
def kama():
    plt.clf()
    plt.title("KAMA "+str(close_data.dtype))
    plt.plot(fast_ta.momentum.KAMA(close_data, 10, 2, 30))
    plt.plot(list(ta.momentum.KAMAIndicator(pandas.Series(close_data)).kama())[9:])
    plt.savefig("KAMA " + str(close_data.dtype) + ".svg")

def roc():
    plt.clf()
    plt.title("ROC "+str(close_data.dtype))
    plt.plot(fast_ta.momentum.ROC(close_data, 12))
    plt.plot(list(ta.momentum.ROCIndicator(pandas.Series(close_data), n=12).roc())[12:])
    plt.savefig("ROC " + str(close_data.dtype) + ".svg")

def stoch():
    plt.clf()
    plt.title("Stochastic Oscillator "+str(close_data.dtype))
    so = fast_ta.momentum.StochasticOscillator(high_data, low_data, close_data, 14, 3)
    plt.plot(so[0])
    sot = ta.momentum.StochasticOscillator(pandas.Series(high_data), pandas.Series(low_data), pandas.Series(close_data))
    plt.plot(sot.stoch())
    plt.savefig("STOCH " + str(close_data.dtype) + ".svg")

    plt.clf()
    plt.title("Stochastic Oscillator Signal "+str(close_data.dtype))
    plt.plot(so[1])
    plt.plot(sot.stoch_signal())
    plt.savefig("STOCH " + str(close_data.dtype) + ".svg")
    
def run_tests():
    rsi()
    ao()
    kama()
    roc()
    #stoch()

plt.figure(figsize=[25, 5])
run_tests()
close_data = np.array(close_data, dtype=np.float32)
high_data = np.array(high_data, dtype=np.float32)
low_data = np.array(low_data, dtype=np.float32)
open_data = np.array(open_data, dtype=np.float32)
run_tests()
