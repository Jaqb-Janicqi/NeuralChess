import base64
import math
import numpy as np
import torch
import chess


def bb_to_matrix(bb: np.uint64) -> np.ndarray:
    """Converts a bitboard to a 8x8 matrix."""

    return np.unpackbits(np.frombuffer(bb.tobytes(), dtype=np.uint8)).astype(np.float32).reshape(8, 8)


def decode_from_fen(fen) -> np.ndarray:
    """Returns a 15x8x8 stack of matrices, representing the current state."""

    state = chess.Board(fen)
    white = state.occupied_co[chess.WHITE]
    black = state.occupied_co[chess.BLACK]
    colors = [white, black]
    # get board attributes
    castling_bb = np.uint64(state.castling_rights)
    ep_square_bb = state.ep_square
    if not ep_square_bb:
        ep_square_bb = 0
    # get pieces
    piece_bbs = [
        state.kings,
        state.queens,
        state.rooks,
        state.bishops,
        state.knights,
        state.pawns
    ]
    matrices = []
    # create color matrix
    color_matrix = np.zeros((8, 8), dtype=np.float32) if state.turn else np.ones(
        (8, 8), dtype=np.float32)
    matrices.append(color_matrix)
    # convert pieces
    for color in colors:
        for piece in range(6):
            pieces = piece_bbs[piece] & color
            pieces = bb_to_matrix(np.uint64(pieces))
            matrices.append(pieces)
    # create ep_square matrix
    matrices.append(bb_to_matrix(np.uint64(ep_square_bb)))
    # convert castling
    matrices.append(bb_to_matrix(np.uint64(castling_bb)))
    return np.array(matrices, np.float32)


def get_centipawns(prob):
    return int(111.714640912 * math.tan(1.5620688421 * prob))


def map_centipawns_to_probability(centipawns):
    return math.atan2(centipawns, 111.714640912) / 1.5620688421


def clip_value(value):
    cp = get_centipawns(value)
    if cp > 1500:
        cp = 1500
    elif cp < -1500:
        cp = -1500
    return map_centipawns_to_probability(cp)


def cp_to_value_clip(cp):
    if cp > 1500:
        cp = 1500
    elif cp < -1500:
        cp = -1500
    return map_centipawns_to_probability(cp)


def encode_a85(byte_rep):
    return base64.a85encode(byte_rep).decode()


def decode_a85(byte_rep, dtype=np.uint32):
    return np.frombuffer(base64.a85decode(byte_rep), dtype=dtype)


def decode_position(byte_rep):
    return decode_a85(byte_rep, np.int8).astype(np.float32).reshape(9, 8, 8)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def decode_policy(byte_rep):
    arr = decode_a85(byte_rep, np.uint32).astype(np.float32)
    arr = arr / arr.sum()
    return softmax(arr)
