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
        plt.savefig("tests/plots/RSI " + str(close_data.dtype) + ".svg")
    
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
        
def cmf():
    plt.clf()
    plt.title("Chaikin Money Flow (CMF) "+str(close_data.dtype))
    so = fast_ta.volume.CMF(high=high_data, low=low_data, close=close_data, volume=volume_data, n=20)
    plt.plot(so)
    sot = ta.volume.ChaikinMoneyFlowIndicator(pandas.Series(high_data), pandas.Series(low_data), pandas.Series(close_data), pandas.Series(volume_data))
    plt.plot(sot.chaikin_money_flow())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/CMF " + str(close_data.dtype) + ".svg")

def emv():
    plt.clf()
    plt.title("Ease of movement (EoM, EMV) "+str(close_data.dtype))
    so = fast_ta.volume.EMV(high = high_data, low = low_data, volume = volume_data, n = 14)
    plt.plot(so[0])
    sot = ta.volume.EaseOfMovementIndicator(pandas.Series(high_data), pandas.Series(low_data), pandas.Series(volume_data))
    plt.plot(sot.ease_of_movement())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/EMV " + str(close_data.dtype) + ".svg")

    plt.clf()
    plt.title("Ease of movement (EoM, EMV) Signal "+str(close_data.dtype))
    plt.plot(so[1])
    plt.plot(sot.sma_ease_of_movement())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/EMV SIGNAL " + str(close_data.dtype) + ".svg")

def fi():
    plt.clf()
    plt.title("Force Index (FI) "+str(close_data.dtype))
    so = fast_ta.volume.FI(close=close_data, volume=volume_data, n=13)
    plt.plot(so)
    sot = ta.volume.ForceIndexIndicator(pandas.Series(close_data), pandas.Series(volume_data))
    plt.plot(sot.force_index())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/FI " + str(close_data.dtype) + ".svg")
        
def mfi():
    plt.clf()
    plt.title("Money Flow Index (MFI) "+str(close_data.dtype))
    so = fast_ta.volume.MFI(high=high_data, low=low_data, close=close_data, volume=volume_data, n=14)
    plt.plot(so)
    sot = ta.volume.MFIIndicator(pandas.Series(high_data), pandas.Series(low_data), pandas.Series(close_data), pandas.Series(volume_data))
    plt.plot(sot.money_flow_index())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/MFI " + str(close_data.dtype) + ".svg")

def nvi():
    plt.clf()
    plt.title("Negative Volume Index (NVI) "+str(close_data.dtype))
    so = fast_ta.volume.NVI(close=close_data, volume=volume_data)
    plt.plot(so)
    sot = ta.volume.NegativeVolumeIndexIndicator(pandas.Series(close_data), pandas.Series(volume_data))
    plt.plot(sot.negative_volume_index())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/NVI " + str(close_data.dtype) + ".svg")
        
def obv():
    plt.clf()
    plt.title("On-balance volume (OBV) "+str(close_data.dtype))
    so = fast_ta.volume.OBV(close=close_data, volume=volume_data)
    plt.plot(so)
    sot = ta.volume.OnBalanceVolumeIndicator(pandas.Series(close_data), pandas.Series(volume_data))
    plt.plot(sot.on_balance_volume())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/OBV " + str(close_data.dtype) + ".svg")
        
def vpt():
    plt.clf()
    plt.title("Volume-price trend (VPT) (NOTE: TA'S IMPLEMENTATION IS BROKEN, USE TRADINGVIEW TO VALIDATE) "+str(close_data.dtype))
    so = fast_ta.volume.VPT(close=close_data, volume=volume_data)
    plt.plot(so)
    sot = ta.volume.VolumePriceTrendIndicator(pandas.Series(close_data), pandas.Series(volume_data))
    #plt.plot(sot.volume_price_trend())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/VPT " + str(close_data.dtype) + ".svg")
        
def vwap():
    plt.clf()
    plt.title("Volume Weighted Average Price (VWAP) "+str(close_data.dtype))
    so = fast_ta.volume.VWAP(high=high_data, low=low_data, close=close_data, volume=volume_data, n=14)
    plt.plot(so)
    sot = ta.volume.VolumeWeightedAveragePrice(pandas.Series(high_data), pandas.Series(low_data), pandas.Series(close_data), pandas.Series(volume_data))
    plt.plot(sot.volume_weighted_average_price())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/VWAP " + str(close_data.dtype) + ".svg")
        
def atr():
    plt.clf()
    plt.title("Average True Range (ATR) "+str(close_data.dtype))
    so = fast_ta.volatility.ATR(high=high_data, low=low_data, close=close_data, n=14)
    plt.plot(so)
    sot = ta.volatility.AverageTrueRange(pandas.Series(high_data), pandas.Series(low_data), pandas.Series(close_data))
    plt.plot(sot.average_true_range())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/ATR " + str(close_data.dtype) + ".svg")

def bol():
    plt.clf()
    plt.title("Bollinger Bands hband "+str(close_data.dtype))
    so = fast_ta.volatility.BOL(close = close_data, n = 20, ndev = 2)
    plt.plot(so[0])
    sot = ta.volatility.BollingerBands(pandas.Series(close_data))
    plt.plot(sot.bollinger_hband())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/BOL HBAND " + str(close_data.dtype) + ".svg")

    plt.clf()
    plt.title("Bollinger Bands mband "+str(close_data.dtype))
    plt.plot(so[1])
    plt.plot(sot.bollinger_mavg())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/BOL MBAND " + str(close_data.dtype) + ".svg")
    
    plt.clf()
    plt.title("Bollinger Bands lband "+str(close_data.dtype))
    plt.plot(so[2])
    plt.plot(sot.bollinger_lband())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/BOL LBAND " + str(close_data.dtype) + ".svg")

def dc():
    # For testing purposes, we specify the high and low data to be the close data
    # since that is how the ta lib does it, however this should only be done if
    # high/low data does not exist.
    plt.clf()
    plt.title("Donchian Channel hband "+str(close_data.dtype))
    so = fast_ta.volatility.DC(high = close_data, low = close_data, n = 20)
    plt.plot(so[0])
    sot = ta.volatility.DonchianChannel(pandas.Series(close_data))
    plt.plot(sot.donchian_channel_hband())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/DC HBAND " + str(close_data.dtype) + ".svg")
    
    plt.clf()
    plt.title("Donchian Channel lband "+str(close_data.dtype))
    plt.plot(so[2])
    plt.plot(sot.donchian_channel_lband())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/DC LBAND " + str(close_data.dtype) + ".svg")

def kc():
    plt.clf()
    plt.title("Keltner Channel hband "+str(close_data.dtype))
    so = fast_ta.volatility.KC(high=high_data, low=low_data, close=close_data, n1=14, n2=10, num_channels=1)
    plt.plot(so[2])
    sot = ta.volatility.KeltnerChannel(pandas.Series(high_data), pandas.Series(low_data), pandas.Series(close_data), ov=False)
    plt.plot(sot.keltner_channel_hband())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/KC HBAND " + str(close_data.dtype) + ".svg")

    plt.clf()
    plt.title("Keltner Channel mband "+str(close_data.dtype))
    plt.plot(so[1])
    plt.plot(sot.keltner_channel_mband())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/KC MBAND " + str(close_data.dtype) + ".svg")
    
    plt.clf()
    plt.title("Keltner Channel lband "+str(close_data.dtype))
    plt.plot(so[0])
    plt.plot(sot.keltner_channel_lband())
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig("tests/plots/KC LBAND " + str(close_data.dtype) + ".svg")
        
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
    cmf()
    emv()
    fi()
    mfi()
    nvi()
    obv()
    vpt()
    vwap()
    atr()
    bol()
    dc()
    kc()

plt.rcParams['figure.figsize'] = (20.0, 10.0)
run_tests()
close_data = np.array(close_data, dtype=np.float32)
high_data = np.array(high_data, dtype=np.float32)
low_data = np.array(low_data, dtype=np.float32)
open_data = np.array(open_data, dtype=np.float32)
volume_data = np.array(volume_data, dtype=np.float32)
run_tests()
