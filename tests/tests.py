import numpy as np
import csv
import ta
import pandas
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import fast_ta
import argparse

# ARGPARSE CODE
parser = argparse.ArgumentParser(description='Fast-TA lib testing tool.')
parser.add_argument('--show-plots', dest='show_plots', action='store_const',
                     const=True, default=False,
                     help='display interactive matplotlib plots')
parser.add_argument('--save-plots', dest='save_plots', action='store_const',
                     const=True, default=False,
                     help='save matplotlib plots')
parser.add_argument('--use-large-dataset', dest='large_dataset', action='store_const',
                     const=True, default=False,
                     help='use a larger dataset for testing')
args = parser.parse_args()

if not args.save_plots and not args.show_plots:
    parser.print_help()
    exit()
#
if not args.large_dataset:
    with open('tests/AAPL_small.csv', 'r') as dataset:
        csv_data = list(csv.reader(dataset))[1:]

    close_data = []
    high_data = []
    low_data = []
    open_data = []
    volume_data = []

    for row in csv_data:
        close_data.append(row[4])
        high_data.append(row[2])
        low_data.append(row[3])
        open_data.append(row[1])
        volume_data.append(row[6])
else:
    with open('tests/AAPL.csv', 'r') as dataset:
        csv_data = list(csv.reader(dataset))[1:]

    close_data = []
    high_data = []
    low_data = []
    open_data = []
    volume_data = []

    for row in csv_data:
        close_data.append(row[1])
        high_data.append(row[2])
        low_data.append(row[3])
        open_data.append(row[4])
        volume_data.append(row[6])

    while 'null' in close_data: close_data.remove('null')
    while 'null' in high_data: high_data.remove('null')
    while 'null' in low_data: low_data.remove('null')
    while 'null' in open_data: open_data.remove('null')

close_data = np.array(close_data, dtype=np.float64)
high_data = np.array(high_data, dtype=np.float64)
low_data = np.array(low_data, dtype=np.float64)
open_data = np.array(open_data, dtype=np.float64)
volume_data = np.array(volume_data, dtype=np.float64)


def rsi():
    plt.clf()
    plt.title("RSI "+str(close_data.dtype))
    plt.plot(fast_ta.momentum.RSI(close=close_data, n = 14))
    plt.plot(ta.momentum.RSIIndicator(pandas.Series(close_data), n=14).rsi())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/rsi " + str(close_data.dtype) + ".svg")
    
def ao():
    plt.clf()
    plt.title("AO "+str(high_data.dtype))
    plt.plot(ta.momentum.AwesomeOscillatorIndicator(pandas.Series(high_data), pandas.Series(low_data)).ao())
    plt.plot(fast_ta.momentum.AO(high=high_data, low=low_data, s = 5, l = 34))
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/AO " + str(high_data.dtype) + ".svg")
    
def kama():
    plt.clf()
    plt.title("KAMA "+str(close_data.dtype))
    plt.plot(fast_ta.momentum.KAMA(close=close_data, n = 10, f = 2, s = 30))
    plt.plot(list(ta.momentum.KAMAIndicator(pandas.Series(close_data)).kama()))
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/KAMA " + str(close_data.dtype) + ".svg")

def roc():
    plt.clf()
    plt.title("ROC "+str(close_data.dtype))
    plt.plot(fast_ta.momentum.ROC(close = close_data, n = 12))
    plt.plot(list(ta.momentum.ROCIndicator(pandas.Series(close_data), n=12).roc()))
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/ROC " + str(close_data.dtype) + ".svg")

def stoch():
    plt.clf()
    plt.title("Stochastic Oscillator "+str(close_data.dtype))
    so = fast_ta.momentum.StochasticOscillator(high = high_data, low = low_data, close = close_data, n = 14, d_n = 3)
    plt.plot(so[0])
    sot = ta.momentum.StochasticOscillator(pandas.Series(high_data), pandas.Series(low_data), pandas.Series(close_data))
    plt.plot(sot.stoch())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/STOCH " + str(close_data.dtype) + ".svg")

    plt.clf()
    plt.title("Stochastic Oscillator Signal "+str(close_data.dtype))
    plt.plot(so[1])
    plt.plot(sot.stoch_signal())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/STOCH SIGNAL " + str(close_data.dtype) + ".svg")

def tsi():
    plt.clf()
    plt.title("TSI "+str(close_data.dtype))
    plt.plot(fast_ta.momentum.TSI(close=close_data, r = 25, s = 13))
    plt.plot(list(ta.momentum.TSIIndicator(pandas.Series(close_data)).tsi()))
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/TSI " + str(close_data.dtype) + ".svg")

def uo():
    plt.clf()
    plt.title("Ultimate Oscillator (NOTE: TA'S IMPLEMENTATION IS BROKEN, USE TRADINGVIEW TO VALIDATE) "+str(close_data.dtype))
    so = fast_ta.momentum.UltimateOscillator(high=high_data, low=low_data, close=close_data, s = 7, m = 14, l = 28, ws = 4, wm = 2, wl = 1)
    plt.plot(so)
    #sot = ta.momentum.UltimateOscillator(pandas.Series(high_data), pandas.Series(low_data), pandas.Series(close_data))
    #plt.plot(sot.uo())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/UO " + str(close_data.dtype) + ".svg")

def wr():
    plt.clf()
    plt.title("Williams %R "+str(close_data.dtype))
    so = fast_ta.momentum.WilliamsR(high=high_data, low=low_data, close=close_data, n=14)
    plt.plot(so)
    sot = ta.momentum.WilliamsRIndicator(pandas.Series(high_data), pandas.Series(low_data), pandas.Series(close_data))
    plt.plot(sot.wr())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/WR " + str(close_data.dtype) + ".svg")

def adi():
    plt.clf()
    plt.title("Accumulation/Distribution Index (ADI) "+str(close_data.dtype))
    so = fast_ta.volume.ADI(high=high_data, low=low_data, close=close_data, volume=volume_data)
    plt.plot(so)
    sot = ta.volume.AccDistIndexIndicator(pandas.Series(high_data), pandas.Series(low_data), pandas.Series(close_data), pandas.Series(volume_data))
    plt.plot(sot.acc_dist_index())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/ADI " + str(close_data.dtype) + ".svg")
        
def run_tests():
    rsi()
    ao()
    kama()
    roc()
    stoch()
    tsi()
    uo()
    wr()
    adi()

plt.rcParams['figure.figsize'] = (20.0, 10.0)
run_tests()
close_data = np.array(close_data, dtype=np.float32)
high_data = np.array(high_data, dtype=np.float32)
low_data = np.array(low_data, dtype=np.float32)
open_data = np.array(open_data, dtype=np.float32)
volume_data = np.array(volume_data, dtype=np.float32)
run_tests()
