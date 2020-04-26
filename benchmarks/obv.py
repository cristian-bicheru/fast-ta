import numpy as np
import csv
import ta
import pandas
import importlib.util
import matplotlib.pyplot as plt
import time
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import fast_ta
plt.rcParams['figure.figsize'] = (20.0, 10.0)

def bench(n, dtype):
    times = [0,0]
    data = np.array(range(n), dtype=dtype)

    s = time.perf_counter()
    fast_ta.volume.OBV(data, data)
    times[0] = time.perf_counter()-s
    
    data = pandas.Series(data)
    s = time.perf_counter()
    ta.volume.OnBalanceVolumeIndicator(data, data).on_balance_volume()
    times[1] = time.perf_counter()-s
    
    return times

for dtype in [np.float32, np.float64]:
    t_max = 10**4
    t_del = 10
    num_label = 10

    bdata = [bench(x, dtype) for x in range(100, t_max, t_del)]
    bdata = [x[1]/x[0] for x in bdata]

    plt.plot(range(len(bdata)), bdata, label=str(dtype).split("'")[1])

plt.xticks([x*t_max/t_del/num_label for x in range(1, num_label+1)], [x*t_max/t_del/num_label for x in range(1, num_label+1)])
plt.xlabel('n')
plt.ylabel('speedup')
plt.title('On-balance volume (OBV) Speedup')
plt.legend(loc='upper right')
plt.savefig('results/obv.svg')