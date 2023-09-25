from copy import deepcopy

import numpy as np


move = np.dtype([
    ("src_y", np.ubyte),
    ("src_x", np.ubyte),
    ("dst_y", np.ubyte),
    ("dst_x", np.ubyte),
    ("promo_piece", np.byte)
])


class Move():
    def __init__(self, src_y: int, src_x: int, dst_y: int, dst_x: int, promo_piece: int = 0) -> None:
        self.__move: np.ndarray = np.zeros((), dtype=move)
        self.__move["src_y"] = src_y
        self.__move["src_x"] = src_x
        self.__move["dst_y"] = dst_y
        self.__move["dst_x"] = dst_x
        self.__move["promo_piece"] = promo_piece

    def __getitem__(self, key) -> np.ndarray:
        return self.__move[key].item()

    def __setitem__(self, key, value) -> None:
        self.__move[key] = value

    def __str__(self) -> str:
        return str(self.__move)
    
    def uci_str(self) -> str:
        promo_char = ""
        if self.promo_piece != 0:
            promo_char = "__nbrq"[self.promo_piece]
        return f"{chr(self.src_x + 97)}{self.src_y + 1}{chr(self.dst_x + 97)}{self.dst_y + 1}{promo_char}"

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
