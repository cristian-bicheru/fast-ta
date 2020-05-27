import numpy as np
import csv
import time
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import fast_ta
import timeit

exec_strings = """fast_ta.volume.ADI(data, data, data, data)
fast_ta.momentum.AO(data, data, 5, 34)
fast_ta.volatility.ATR(data, data, data, 14)
fast_ta.volatility.BOL(data, 20, 2)
fast_ta.volume.CMF(data, data, data, data, 20)
fast_ta.volatility.DC(data, data, 20)
fast_ta.volume.EMV(data, data, data, 14)
fast_ta.volume.FI(data, data, 13)
fast_ta.momentum.KAMA(data, 10, 2, 30)
fast_ta.volatility.KC(data, data, data, 14, 1)
fast_ta.volume.MFI(data, data, data, data, 14)
fast_ta.volume.NVI(data, data)
fast_ta.volume.OBV(data, data)
fast_ta.momentum.ROC(data, 12)
fast_ta.momentum.RSI(data, 14)
fast_ta.momentum.StochasticOscillator(data, data, data, 14, 3)
fast_ta.momentum.TSI(data, 25, 13)
fast_ta.momentum.UltimateOscillator(data, data, data, 7, 14, 28, 4, 2, 1)
fast_ta.volume.VPT(data, data)
fast_ta.volume.VWAP(data, data, data, data, 14)
fast_ta.momentum.WilliamsR(data, data, data, 14)""".split("\n")

num_elements = 10**7
iterations = 10

with open("results/throughput_fast_ta.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Indicator", "fast_ta float32 (Data Points/Second)", "fast_ta float64 (DPS)"])
    
    for execs in exec_strings:
        name = '.'.join(execs.split('.')[1:3]).split('(')[0]
        print("Calculating throughput of", name+'...')
        row = [name]
        for dtype in ["np.float32", "np.float64"]:
            times = timeit.timeit(execs, number=iterations, setup="import fast_ta; import numpy as np; data = fast_ta.core.align(np.array(range("+str(num_elements)+"), dtype="+dtype+"))")
            row.append("{:.3e}".format(num_elements/times*iterations))

        writer.writerow(row)
