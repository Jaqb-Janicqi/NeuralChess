from timeit import timeit
from copy import deepcopy

import numba as nb
import numpy as np
from numba import njit, prange
from numpy import int8, int32, uint64

from actionspace import ActionSpace
from move import Move
from state import State


def print_bitboard(bitboard) -> None:
    bitboard = np.binary_repr(bitboard, width=64)
    for i in range(8):
        bitboard_slice = bitboard[i*8:(i+1)*8]
        bitboard_slice = ' '.join(bitboard_slice)
        print(bitboard_slice)
    print()


@njit(nb.uint64(nb.uint64, nb.int8))
def set_bit(bitboard: uint64, bit_num: int8) -> uint64:
    return bitboard | (1 << bit_num)


@njit(nb.uint64(nb.uint64, nb.int8))
def clear_bit(bitboard: uint64, bit_num: int8) -> uint64:
    return bitboard & ~(1 << bit_num)


@njit(nb.bool_(nb.uint64, nb.int8))
def is_set(bitboard: uint64, bit_num: int8) -> bool:
    if bitboard & (1 << bit_num):
        return True
    else:
        return False


@njit(nb.int8(nb.int8, nb.int8))
def square_index(rank: int8, file: int8) -> int8:
    return rank * 8 + file


@njit(nb.types.UniTuple(nb.int8, 2)(nb.int8))
def get_rank_file(square_index: int8) -> tuple[int8, int8]:
    return square_index // 8, square_index % 8


@njit(nb.uint64(nb.int8))
def square_index_binary(bitboard: int8) -> int8:
    return np.log2(bitboard)


@njit()
def idx_of_set_bits(bitboard: uint64) -> list[int8]:
    indices = []
    while bitboard:
        lsb = bitboard & -bitboard
        indices.append(int(np.log2(lsb)))
        bitboard ^= lsb
    return indices


@njit()
def is_in_bounds(y: int8, x: int8) -> bool:
    return 0 <= y < 8 and 0 <= x < 8


@njit(nb.types.Tuple((nb.uint64, nb.int8))(nb.int8, nb.int8, nb.int8, nb.int8))
def generate_ray(rank: int8, file: int8, dst_rank: int8,
                 dst_file: int8) -> tuple[uint64, int8]:
    rank_dir = np.sign(dst_rank - rank) * 8
    file_dir = np.sign(dst_file - file)
    ray_dir = rank_dir + file_dir
    ray_len = max(abs(dst_rank - rank), abs(dst_file - file))
    square = square_index(rank, file)

    ray = 0
    for step in range(1, ray_len + 1):
        ray |= set_bit(ray, square + step * ray_dir)
    return ray, ray_dir


@njit(nb.uint64(nb.uint64, nb.int8, nb.int8))
def get_vision_map(occupied: uint64, y: int8, x: int8) -> int8:
    # calculate maximum distance to edge of board
    y_dist = max(y, 7 - y)
    x_dist = max(x, 7 - x)

    # +1 here instead of in range() to avoid repeated addition millions of times
    ray_len_offset = max(y_dist, x_dist) + 1
    vision_board = 0

    # iterate over all 8 directions
    for y_dir in range(-1, 2):
        for x_dir in range(-1, 2):
            # skip self
            if y_dir == 0 and x_dir == 0:
                continue
            # cast vision ray in direction until edge of board or piece is hit
            for dist in range(1, ray_len_offset):
                y_new = y + y_dir * dist
                x_new = x + x_dir * dist
                if not is_in_bounds(y_new, x_new):
                    break
                tmp_sq = square_index(y_new, x_new)
                vision_board |= set_bit(vision_board, tmp_sq)
                if is_set(occupied, tmp_sq):
                    break
    return vision_board


class Chess():
    def __init__(self) -> None:
        self.__board_size: int = 8
        self.__state: State = self.load_default()
        self.__action_space: ActionSpace = ActionSpace(
            self.__board_size, self.__board_size)
        self.__move_maps = {
            1: np.zeros((8, 8), dtype=uint64),
            -1: np.zeros((8, 8), dtype=uint64),
            2: np.zeros((8, 8), dtype=uint64),
            3: np.zeros((8, 8), dtype=uint64),
            4: np.zeros((8, 8), dtype=uint64),
            5: np.zeros((8, 8), dtype=uint64),
            6: np.zeros((8, 8), dtype=uint64)
        }
        self.__attack_maps = {
            1: np.zeros((8, 8), dtype=uint64),
            -1: np.zeros((8, 8), dtype=uint64),
            2: np.zeros((8, 8), dtype=uint64),
            3: np.zeros((8, 8), dtype=uint64),
            4: np.zeros((8, 8), dtype=uint64),
            5: np.zeros((8, 8), dtype=uint64),
            6: np.zeros((8, 8), dtype=uint64)
        }
        self.__move_functions = {
            1: self.generate_pawn,
            2: self.generate_knight,
            3: self.generate_sliding,
            4: self.generate_sliding,
            5: self.generate_sliding,
            6: self.generate_sliding
        }
        self.__move_range = {
            3: 999,
            4: 999,
            5: 999,
            6: 1
        }
        self.generate_maps()

    def bind_move(self, rank, file) -> tuple[int, int]:
        return max(0, min(rank, self.__board_size - 1)), max(0, min(file, self.__board_size - 1))

    def piece_id(self, piece) -> np.number:
        return abs(piece)

    def piece_color(self, piece) -> np.number:
        return np.sign(piece)

    def int_from_char(char) -> int:
        if char.isdigit():
            return int(ord(char) - 48)
        return int(ord(char) - 97)

    def generate_sliding(self, src_rank: int, src_file: int, piece: int) -> list[int]:
        piece_id = self.piece_id(piece)
        move_range = self.__move_range[piece_id]
        start_rank, start_file = self.bind_move(
            src_rank - move_range, src_file - move_range)
        end_rank, end_file = self.bind_move(
            src_rank + move_range, src_file + move_range)

        directions = []
        if piece_id in [4, 5, 6]:
            directions += [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if piece_id in [3, 5, 6]:
            directions += [(1, 1), (-1, 1), (1, -1), (-1, -1)]

        move_bitboard = 0
        for rank_dir, file_dir in directions:
            rank, file = src_rank + rank_dir, src_file + file_dir
            while start_rank <= rank <= end_rank and start_file <= file <= end_file:
                move_bitboard = set_bit(
                    move_bitboard, square_index(rank, file))
                rank += rank_dir
                file += file_dir
        return move_bitboard

    def generate_knight(self, src_rank: int, src_file: int, piece: int8) -> list[int]:
        move_bitboard = 0
        for file_dir, rank_dir in [(2, 1), (2, -1), (-2, 1), (-2, -1),
                                   (1, 2), (1, -2), (-1, 2), (-1, -2)]:
            rank, file = src_rank + rank_dir, src_file + file_dir
            if is_in_bounds(rank, file):
                move_bitboard = set_bit(
                    move_bitboard, square_index(rank, file))
        return move_bitboard

    def generate_pawn(self, src_rank: int, src_file: int, piece: int8) -> list[int]:
        rank = src_rank + piece
        if not is_in_bounds(rank, src_file):
            return 0
        move_bitboard = 0
        move_bitboard = set_bit(move_bitboard, square_index(rank, src_file))

        for side_dir in [-1, 1]:
            file = src_file + side_dir
            if is_in_bounds(rank, file):
                move_bitboard = set_bit(
                    move_bitboard, square_index(rank, file))

        if (piece == 1 and src_rank == 1) or (piece == -1 and src_rank == 6):
            rank += piece
            if is_in_bounds(rank, src_file):
                move_bitboard = set_bit(
                    move_bitboard, square_index(rank, src_file))
        return move_bitboard

    def generate_moves(self):
        for rank in range(self.__board_size):
            for file in range(self.__board_size):
                for piece_id in self.__move_maps.keys():
                    # asign move maps for each piece
                    self.__move_maps[piece_id][rank, file] = np.uint64(
                        self.__move_functions[self.piece_id(piece_id)](rank, file, piece_id))

        # copy moves for pieces with movement independent of color
        for i in range(2, 7):
            self.__move_maps[-i] = self.__move_maps[i]

    def get_pseudo_legal_moves(self, game_state: State, turn: int8) -> np.ndarray:
        board = game_state.board
        enemy = game_state.pieces_of(-turn)
        empty = game_state.empty

        pseudo = np.zeros((self.__board_size, self.__board_size), dtype=uint64)
        for rank in range(self.__board_size):
            for file in range(self.__board_size):
                # skip empty squares and enemy pieces
                if board[rank, file] == 0 or self.piece_color(board[rank, file]) != turn:
                    continue

                # calculate normal movement for empty squares
                pseudo[rank, file] = self.__move_maps[
                    board[rank, file]][rank, file] & empty
                
                # calculate attack moves
                pseudo[rank, file] |= self.__attack_maps[
                    board[rank, file]][rank, file] & enemy
        return pseudo

    def get_move_map(self, piece_id: int, rank: int, file: int) -> uint64:
        return self.__move_maps[piece_id][rank, file]

    def get_attack_map(self, piece_id: int, rank: int, file: int) -> uint64:
        return self.__attack_maps[piece_id][rank, file]

    def get_seen_squares(self, game_state: State, turn: int8) -> np.ndarray:
        board = game_state.board
        occupied = game_state.occupied
        friendly = game_state.pieces_of(turn)

        # get indices of friendly pieces and possible attacking moves
        friendly_idx = np.vectorize(get_rank_file)(idx_of_set_bits(friendly))
        friendly_ids = np.vectorize(self.piece_id)(board[friendly_idx])
        attack_maps = np.vectorize(self.get_attack_map)(
            friendly_ids, friendly_idx[0], friendly_idx[1])
        # friendly_idx = np.moveaxis(friendly_idx, 0, 1)

        ray_pieces = np.where(friendly_ids > 2, True, False)
        friendly_idx = np.array(friendly_idx)
        friendly_ray_idx = friendly_idx[:, ray_pieces]
        ray_attack_maps = attack_maps[ray_pieces]
        # get vision map of each piece as if it were a queen

        vision_maps = np.vectorize(get_vision_map)(
            occupied, friendly_ray_idx[0], friendly_ray_idx[1]).astype(uint64)

        vision_map = uint64(0)
        for map, attack_map in zip(vision_maps, ray_attack_maps):
            # TODO: attack maps should be used instead of move maps (pawns can't forwards)
            vision_map |= map & attack_map

        # add all possible attacks of non-ray pieces
        for attack_map in attack_maps[~ray_pieces]:
            vision_map |= attack_map
        return vision_map

    def get_masks(self, game_state: State) -> tuple(uint64, uint64, uint64, uint64):
        king = game_state.player_king
        opponent_turn = -game_state['turn']
        opponent_pseudo_moves = self.get_pseudo_legal_moves(
            game_state, opponent_turn)
        seen_mask = self.get_seen_squares(game_state, opponent_turn)

        attackers = np.argwhere(opponent_pseudo_moves & king)
        if not attackers.any():
            # no check, no pins -> all moves are legal
            tmp = np.full(4, -1).astype(uint64)
            tmp[3] = seen_mask
            return tmp

        king_rank, king_file = get_rank_file(square_index_binary(king))
        board = game_state.board
        checkmask = uint64(0)
        pinmask_hv = uint64(0)
        pinmask_diag = uint64(0)

        # calculate pinmasks for attacking pieces
        for attacker in attackers:
            attacker_type = self.piece_id(board[attacker[0], attacker[1]])
            if attacker_type == 2:
                checkmask |= set_bit(
                    checkmask, square_index(attacker[0], attacker[1]))
                continue

            ray, ray_dir_bitrep = generate_ray(
                attacker[0], attacker[1], king_rank, king_file)
            ray &= seen_mask    # TODO this may be needed to not extend pin through multiple pieces

            # extend the ray to behind the king (prevent king from staying in check)
            checkmask |= ray | set_bit(uint64(1), square_index(
                king_rank, king_file) + ray_dir_bitrep)
            
            if attacker[0] == king_rank ^ attacker[1] == king_file:
                pinmask_hv |= ray
            else:
                pinmask_diag |= ray

        if pinmask_hv == 0:
            pinmask_hv = np.array(-1).astype(uint64)
        if pinmask_diag == 0:
            pinmask_diag = np.array(-1).astype(uint64)
        return checkmask, pinmask_hv, pinmask_diag, seen_mask

    def get_legal_moves(self, game_state: State) -> np.ndarray:
        enemy_or_empty = game_state.enemy_or_empty
        pseudo_legal_moves = self.get_pseudo_legal_moves(
            game_state, game_state['turn'])
        checkmask, pinmask_hv, pinmask_diag, enemy_visionmask = self.get_masks(
            game_state)
        pinboard_hv = np.array(list(np.binary_repr(
            pinmask_hv, width=64))).reshape(8, 8)
        pinboard_diag = np.array(list(np.binary_repr(
            pinmask_diag, width=64))).reshape(8, 8)
        no_pin = np.array(-1).astype(uint64)
        board = game_state.board

        legal_moves = np.zeros_like(pseudo_legal_moves)
        for rank in range(self.__board_size):
            for file in range(self.__board_size):
                piece_id = self.piece_id(board[rank, file])
                pinmask = no_pin

                if pinboard_hv[rank, file]:
                    pinmask &= pinmask_hv
                if pinboard_diag[rank, file]:
                    pinmask &= pinmask_diag

                if piece_id == 6:
                    legal_moves[rank, file] = \
                        enemy_or_empty & ~checkmask & ~enemy_visionmask
                else:
                    legal_moves[rank, file] = \
                        pseudo_legal_moves[rank,
                                           file] & enemy_or_empty & checkmask & pinmask
        return legal_moves

    def generate_attacks(self) -> None:
        self.__attack_maps = deepcopy(self.__move_maps)

        vert_line = 0
        for i in range(8):
            vert_line = set_bit(vert_line, uint64(i * 8))

        for file in range(8):
            tmp = uint64(vert_line << file)
            for rank in range(8):
                self.__attack_maps[1][rank, file] &= ~tmp
                self.__attack_maps[-1][rank, file] &= ~tmp

    def remove_sideways_pawn_moves(self) -> None:
        vert_line = 0
        for i in range(8):
            vert_line = set_bit(vert_line, uint64(i * 8))

        for file in range(8):
            tmp = uint64(vert_line << file)
            for rank in range(8):
                self.__attack_maps[1][rank, file] &= tmp
                self.__attack_maps[-1][rank, file] &= tmp

    def generate_maps(self) -> None:
        self.generate_moves()
        self.generate_attacks()
        self.remove_sideways_pawn_moves()

    def next_game(self) -> None:
        self.__state = self.load_default()

    def add_piece(self, game_state: State, piece_id: int, rank: int, file: int) -> None:
        sq_id = square_index(rank, file)
        game_state[str(piece_id)] = set_bit(
            game_state[str(piece_id)], sq_id)
        game_state.board[rank, file] = piece_id

    def fenn_decode(self, fenn_string: str) -> State:
        piece_id = {
            'P': 1,
            'N': 2,
            'B': 3,
            'R': 4,
            'Q': 5,
            'K': 6,
            'p': -1,
            'n': -2,
            'b': -3,
            'r': -4,
            'q': -5,
            'k': -6,
        }
        game_state = State()
        file = 0
        rank = self.__board_size - 1

        try:
            pieces, turn, castling, en_passant, halfmove, fullmove = fenn_string.split(
                ' ')
        except ValueError:
            raise ValueError(f'Invalid FEN string {fenn_string}')

        # place pieces on board and bitboards
        for char in pieces:
            if char == '/':
                pass
            elif char.isdigit():
                file += int(char)
                r, file = np.divmod(file, self.__board_size)
                rank -= r
            elif char in piece_id.keys():
                self.add_piece(game_state, piece_id[char], rank, file)
                file += 1
                r, file = np.divmod(file, self.__board_size)
                rank -= r
            else:
                raise ValueError(f'Unsupported character {char} in FEN string')

        # set turn
        if turn == 'w':
            game_state['turn'] = 1
        elif turn == 'b':
            game_state['turn'] = -1
        else:
            raise ValueError(f'Unsupported player turn {turn} in FEN string')

        # set castling rights
        try:
            for char in castling:
                if char == '-':
                    pass
                elif char == 'q':
                    sq_num = square_index(7, 0)
                    if is_set(game_state['-4'], sq_num):
                        game_state['castle_rights'] = set_bit(
                            game_state['castle_rights'], sq_num)
                    else:
                        raise ValueError(
                            f'Rook not present on required square to castle: {char}')
                elif char == 'k':
                    sq_num = square_index(7, 7)
                    if is_set(game_state['-4'], sq_num):
                        game_state['castle_rights'] = set_bit(
                            game_state['castle_rights'], sq_num)
                    else:
                        raise ValueError(
                            f'Rook not present on required square to castle: {char}')
                elif char == 'Q':
                    sq_num = square_index(0, 0)
                    if is_set(game_state['4'], sq_num):
                        game_state['castle_rights'] = set_bit(
                            game_state['castle_rights'], sq_num)
                    else:
                        raise ValueError(
                            f'Rook not present on required square to castle: {char}')
                elif char == 'K':
                    sq_num = square_index(0, 7)
                    if is_set(game_state['4'], sq_num):
                        game_state['castle_rights'] = set_bit(
                            game_state['castle_rights'], sq_num)
                    else:
                        raise ValueError(
                            f'Rook not present on required square to castle: {char}')
                else:
                    raise ValueError(f'Unsupported castling direction: {char}')
        except ValueError as e:
            raise RuntimeError(f'Invalid castling rights') from e

        # set en passant target
        try:
            if en_passant != '-':
                file_char, rank_char = en_passant
                rank = self.int_from_char((rank_char))
                file = self.int_from_char((file_char))
                sq_id = square_index(rank, file)
                game_state['en_passant_target'] = set_bit(
                    game_state['en_passant_target'], sq_id)
        except:
            raise ValueError(f'Invalid en_passant_target {en_passant}')

        # set halfmove and fullmove
        if not halfmove.isnumeric():
            raise TypeError(f'Halfmove is not an intiger {halfmove}')
        halfmove = int(halfmove)
        if halfmove >= 100:
            raise ValueError(f'Invalid halfmove value {halfmove}')
        game_state['halfmove'] = halfmove

        if not fullmove.isnumeric():
            raise TypeError(f'Halfmove is not an intiger {fullmove}')
        fullmove = int(fullmove)
        if fullmove >= 100:
            raise ValueError(f'Invalid halfmove value {fullmove}')
        game_state['fullmove'] = fullmove
        return game_state

    def load_default(self) -> State:
        # return self.fenn_decode('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/7K w - - 0 1')
        return self.fenn_decode('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

    @property
    def action_space(self):
        return self.__action_space

    @property
    def state(self):
        return self.__state


gm = Chess()

masks = gm.get_masks(gm.state)
print(masks)
# print(timeit(lambda: gm.get_seen_squares(gm.state, 1), number=100000))
# print(timeit(lambda: gm.get_seen_squares(gm.state, 1), number=1))
