import numpy as np
from time import perf_counter

def python_abs(num):
    return abs(num)

def np_abs(num):
    return np.abs(num)

def np_absolute(num):
    return np.absolute(num)



p1 = np.int32(-2)
p2 = np.int32(2)

tic = perf_counter()
for i in range(50000):
    python_abs(p1)
    python_abs(p2)
toc = perf_counter()
print(toc - tic)

tic = perf_counter()
for i in range(50000):
    np_abs(p1)
    np_abs(p2)
toc = perf_counter()
print(toc - tic)

tic = perf_counter()
for i in range(50000):
    np_absolute(p1)
    np_absolute(p2)
toc = perf_counter()
print(toc - tic)