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


def bench(n):
    times = [0,0]
    data = np.array(range(n), dtype=np.double)

    s = time.perf_counter()
    fast_ta.momentum.KAMA(data, 10, 2, 30)
    times[0] = time.perf_counter()-s
    
    data = pandas.Series(data)
    s = time.perf_counter()
    ta.momentum.KAMAIndicator(data).kama()
    times[1] = time.perf_counter()-s
    
    return times

t_max = 10**4
t_del = 10
num_label = 10

bdata = [bench(x) for x in range(100, t_max, t_del)]
bdata = [x[1]/x[0] for x in bdata]

plt.plot(range(len(bdata)), bdata)
plt.xticks([x*t_max/t_del/num_label for x in range(1, num_label+1)], [x*t_max/t_del/num_label for x in range(1, num_label+1)])
plt.xlabel('n')
plt.ylabel('speedup')
plt.title('KAMA Speedup')
plt.savefig('kama.svg')