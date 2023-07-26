import numpy as np

Move = np.dtype([
    ("src_y", np.uint8),
    ("src_x", np.uint8),
    ("dst_y", np.uint8),
    ("dst_x", np.uint8),
    ("promo_piece", np.uint8),
    ("special_dir", np.uint8)
])

GameState = np.dtype([
    ("p", np.uint64),
    ("k", np.uint64),
    ("n", np.uint64),
    ("b", np.uint64),
    ("r", np.uint64),
    ("q", np.uint64),
    ("P", np.uint64),
    ("K", np.uint64),
    ("N", np.uint64),
    ("B", np.uint64),
    ("R", np.uint64),
    ("Q", np.uint64),
    ("castle_rights", np.uint64),
    ("en_passant_target", np.uint64),
    ("turn", np.uint8),
    ("winner", np.uint8),
    ("terminal", np.bool_),
    ("black", np.uint64),
    ("white", np.uint64),
    ("occupied", np.uint64),
    ("halfmove", np.uint8),
    ("fullmove", np.uint8),
    ("board", np.int8, (8, 8))
])


def allocate_game_state():
    return np.zeros((), dtype=GameState)


def allocate_move():
    return np.zeros((), dtype=Move)


def allocate_move_arr():
    return np.zeros((218,), dtype=Move)
