from copy import deepcopy

import numpy as np


move = np.dtype([
    ("src_y", np.ubyte),
    ("src_x", np.ubyte),
    ("dst_y", np.ubyte),
    ("dst_x", np.ubyte),
    ("promo_piece", np.byte),
    ("castle", np.byte)
])


class Move():
    def __init__(self) -> None:
        self.__move: np.ndarray = np.zeros((), dtype=move)

    def __getitem__(self, key) -> np.ndarray:
        return self.__move[key].item()

    def __setitem__(self, key, value) -> None:
        self.__move[key] = value

    def __str__(self) -> str:
        return str(self.__move)

    def copy(self):
        return deepcopy(self.__move)

    @property
    def src_y(self) -> int:
        return self.__move["src_y"]

    @property
    def src_x(self) -> int:
        return self.__move["src_x"]

    @property
    def dst_y(self) -> int:
        return self.__move["dst_y"]

    @property
    def dst_x(self) -> int:
        return self.__move["dst_x"]

    @property
    def promo_piece(self) -> int:
        return self.__move["promo_piece"]

    @property
    def castle(self) -> int:
        return self.__move["castle"]
