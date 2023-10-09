import chess
import numpy as np
from time import perf_counter
import sys
import itertools

# Create an empty chess board
# board = chess.Board()

# tic = perf_counter()
# x = 100000
# for i in range(x):
#     board.board_fen()
# toc = perf_counter()
# print((toc - tic) / x)


# arr = np.zeros(1000, dtype=np.float32)
# fen = chess.Board().fen()
# obj_tuple = (fen, arr)
# item_size = sys.getsizeof((fen, arr))
# arr_size = sys.getsizeof(arr)

# def getsize(tpl):
#     return sys.getsizeof(tpl)

# def safe_assume(tpl):
#     return item_size if item_size != 0 else getsize(tpl)

# def assume(tpl):
#     return item_size

# def hybrid_assume(tpl):
#     return arr_size + sys.getsizeof(fen)

# x = 10000000

# tic = perf_counter()
# for i in range(x):
#     assume(obj_tuple)
# toc = perf_counter()
# unsafe_time = (toc - tic) / x
# print("unsafe assume: ", (toc - tic) / x)

# tic = perf_counter()
# for i in range(x):
#     safe_assume(obj_tuple)
# toc = perf_counter()
# safe_time = (toc - tic) / x
# print("safe_assume: ", (toc - tic) / x)


# tic = perf_counter()
# for i in range(x):
#     getsize(obj_tuple)
# toc = perf_counter()
# getsize_time = (toc - tic) / x
# print("true size: ", (toc - tic) / x)

# tic = perf_counter()
# for i in range(x):
#     hybrid_assume(obj_tuple)
# toc = perf_counter()
# hybrid_time = (toc - tic) / x
# print("hybrid assume: ", (toc - tic) / x)

# print(f"unsafe is {getsize_time / unsafe_time} times faster than true size")

tic = perf_counter()
x = 0
for i in range(1000000):
    x += 1

toc = perf_counter()
print((toc - tic) / 1000000)

tic = perf_counter()
x = 0
for i in itertools.count():
    x += 1
    if i == 1000000:
        break

toc = perf_counter()
print((toc - tic) / 1000000)

tic = perf_counter()
x = 0
while x < 1000000:
    x += 1
toc = perf_counter()
print((toc - tic) / 1000000)