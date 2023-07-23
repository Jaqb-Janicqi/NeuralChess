import numpy as np
from time import perf_counter

Move = np.dtype([
    ("src_square", np.ubyte),
    ("dst_square", np.ubyte),
    ("promo_piece", np.byte),
    ("castle_dir", np.byte)
])

mv = np.array([(2, 0, 0, 0)], dtype=Move)
tic = perf_counter()
for i in range(1000000):
    mv["src_square"] = 2
toc = perf_counter()
print(f"Elapsed time: {toc - tic:0.4f} seconds")

mv = np.array([(2, 0, 0, 0)], dtype=Move)
tic = perf_counter()
for i in range(1000000):
    mv["src_square"] == np.ubyte(2)
toc = perf_counter()
print(f"Elapsed time: {toc - tic:0.4f} seconds")