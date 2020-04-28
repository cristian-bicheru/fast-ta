import numpy as np
import csv
import ta
import time
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import fast_ta

indicator_names = [x[:-3] for x in os.listdir() if x.endswith('.py') and x != "throughput.py"]
indicators = [__import__(x) for x in indicator_names]

num_elements = 10**6


with open("results/throughput.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Indicator", "fast_ta float32 (Data Points/Second)", "ta float32 (DPS)", "float32 relative speedup", "fast_ta float64 (DPS)", "ta float64 (DPS)", "float64 relative speedup"])
    
    for i in range(len(indicators)):
        print("Calculating throughput of", indicator_names[i]+'...')
        row = [indicator_names[i]]
        for dtype in [np.float32, np.float64]:
            times = indicators[i].bench(num_elements, dtype)
            row.append("{:.3e}".format(num_elements/times[0]))
            row.append("{:.3e}".format(num_elements/times[1]))
            row.append("{:.3e}".format(times[1]/times[0]))

        writer.writerow(row)
