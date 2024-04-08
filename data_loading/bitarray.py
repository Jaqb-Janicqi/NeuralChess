import numpy as np


class BitArray():
    def __init__(self, size: int) -> None:
        self.__size = size
        self.__bit_array = np.zeros(
            np.ceil(size/64).astype(np.uint64), dtype=np.uint64)

    def __getitem__(self, index: int) -> bool:
        if index >= self.__size:
            raise IndexError("Index out of range")
        return bool(self.__bit_array[index//64] & np.uint64(1 << (index % 64)))

    def __setitem__(self, index: int, value: bool) -> None:
        if index >= self.__size:
            raise IndexError("Index out of range")
        if value:
            self.__bit_array[index//64] |= np.uint64(1 << (index % 64))
        else:
            self.__bit_array[index//64] &= np.uint64(~(1 << (index % 64)))
