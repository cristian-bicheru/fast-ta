import numpy as np
import csv
import time
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import fast_ta

indicator_names = [x[:-3] for x in os.listdir() if x.endswith('.py') and x != "throughput.py" and x != "throughput_fast_ta.py"]
indicators = [__import__(x) for x in indicator_names]

num_elements = 2**20


with open("results/throughput_fast_ta.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Indicator", "fast_ta float32 (Data Points/Second)", "fast_ta float64 (DPS)"])
    
    for i in range(len(indicators)):
        print("Calculating throughput of", indicator_names[i]+'...')
        row = [indicator_names[i]]
        for dtype in [np.float32, np.float64]:
            time = indicators[i].bench_fast_ta(num_elements, dtype)
            row.append("{:.3e}".format(num_elements/time))

        writer.writerow(row)
