import numpy as np

from actionspace import ActionSpace
from move import Move
from state import State
import numba as nb
from numba import njit, prange
from numpy import int8, int32, uint64


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


@njit()
def get_vision_map(occupied: uint64, y: int8, x: int8) -> int8:
    # calculate maximum distance to edge of board
    y_dist = max(y, 7 - y)
    x_dist = max(x, 7 - x)

    # +1 here instead of in range() to avoid repeated addition millions of times
    ray_len_offset = max(y_dist, x_dist) + 1
    vision_board = np.uint64(0)

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
        self.generate_moves()

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
                    self.__move_maps[piece_id][rank, file] = np.uint64(
                        self.__move_functions[self.piece_id(piece_id)](rank, file, piece_id))
        for i in range(2, 7):
            self.__move_maps[-i] = self.__move_maps[i]

    def get_pseudo_legal_moves(self, game_state: State, turn: int8) -> np.ndarray:
        board = game_state.board
        enemy_or_empty = ~game_state.pieces_of(turn)
        pseudo = np.zeros((self.__board_size, self.__board_size), dtype=uint64)
        for rank in range(self.__board_size):
            for file in range(self.__board_size):
                if board[rank, file] == 0 or self.piece_color(board[rank, file]) != turn:
                    continue
                pseudo[rank, file] = self.__move_maps[
                    board[rank, file]][rank, file] & enemy_or_empty
        return pseudo

    def get_move_map(self, piece_id: int8, rank: int, file: int) -> int:
        return self.__move_maps[piece_id][rank, file]

    def get_seen_squares(self, game_state: State, turn: int8) -> np.ndarray:
        board = game_state.board
        occupied = game_state.occupied
        friendly = game_state.pieces_of(turn)

        # get indices of friendly pieces and possible moves
        friendly_idx = np.vectorize(get_rank_file)(idx_of_set_bits(friendly))
        friendly_ids = np.vectorize(self.piece_id)(board[friendly_idx])
        move_maps = np.vectorize(self.get_move_map)(
            friendly_ids, friendly_idx[0], friendly_idx[1])
        friendly_idx = np.moveaxis(friendly_idx, 0, 1)

        ray_pieces = np.where(friendly_ids > 2, friendly_idx)
        # get vision map of each piece as if it were a queen
        vision_maps = np.vectorize(get_vision_map)(
            occupied, friendly_idx[0], friendly_idx[1])

        vision_map = uint64(0)
        for map in vision_maps:
            vision_map |= map

        print_bitboard(vision_map)
        # add all possible moves of non-ray pieces
        for move_map in move_maps[~ray_pieces]:
            print_bitboard(move_map)
            vision_map |= move_map
        print_bitboard(vision_map)
        print_bitboard(occupied)
        return vision_map

    def get_masks(self, game_state: State) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        king = game_state.player_king
        opponent_pseudo_moves = self.get_pseudo_legal_moves(
            game_state, -game_state['turn'])

        attackers = np.argwhere(opponent_pseudo_moves & king)
        if attackers.size == 0:
            # no check, no pins -> all moves are legal
            np.full(3, -1).astype(uint64)

        king_rank, king_file = get_rank_file(square_index_binary(king))
        board = game_state.board
        checkmask = uint64(0)
        pinmask_hv = uint64(0)
        pinmask_diag = uint64(0)
        # TOOD: add "seen" mask, squares seen by enemy pieces

        for attacker in attackers:
            attacker_type = self.piece_id(board[attacker[0], attacker[1]])
            if attacker_type == 2:
                checkmask |= set_bit(
                    checkmask, square_index(attacker[0], attacker[1]))
                continue
            ray, ray_dir_bitrep = generate_ray(
                attacker[0], attacker[1], king_rank, king_file)
            # extend the ray to behind the king (prevent king from staying in check)
            checkmask |= ray | set_bit(uint64(1), square_index(
                king_rank, king_file) + ray_dir_bitrep)
            if attacker[0] == king_rank or attacker[1] == king_file:
                pinmask_hv |= ray
            else:
                pinmask_diag |= ray
        if pinmask_hv == 0:
            pinmask_hv = np.array(-1).astype(uint64)
        if pinmask_diag == 0:
            pinmask_diag = np.array(-1).astype(uint64)
        return checkmask, pinmask_hv, pinmask_diag

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
                if pinboard_hv[rank, file]:
                    pinmask = pinmask_hv
                elif pinboard_diag[rank, file]:
                    pinmask = pinmask_diag
                else:
                    pinmask = no_pin
                if piece_id == 6:
                    enemy_or_empty & ~checkmask & ~enemy_visionmask
                else:
                    legal_moves[rank, file] = \
                        pseudo_legal_moves[rank,
                                           file] & enemy_or_empty & checkmask & pinmask
        return legal_moves

    def generate_vision_edges(self) -> None:
        bottom8 = uint64(0)
        right8 = uint64(0)

        for i in range(8):
            bottom8 = set_bit(bottom8, uint64(i))
            right8 = set_bit(right8, uint64(i * 8))

        top8 = bottom8 << 56
        left8 = right8 << 7

        for rank in range(8):
            for file in range(8):
                move_map = self.__move_maps[5][rank, file]
                edge_mask = uint64(0)

                edge_mask |= uint64((rank != 0) * bottom8)
                edge_mask |= uint64((rank != 7) * top8)
                edge_mask |= uint64((file != 0) * right8)
                edge_mask |= uint64((file != 7) * left8)
                edge_mask &= move_map
                self.__vision_edges[rank, file] = edge_mask

    def next_game(self) -> None:
        self.__state = self.load_default()

    def add_piece(self, game_state, piece_id, rank, file) -> None:
        sq_id = square_index(rank, file)
        game_state[str(piece_id)] = set_bit(
            game_state[str(piece_id)], sq_id)
        game_state.board[rank, file] = piece_id

    def fenn_decode(self, fenn_string) -> State:
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
                    if self.is_set(game_state['-4'], sq_num):
                        game_state['castle_rights'] = set_bit(
                            game_state['castle_rights'], sq_num)
                    else:
                        raise ValueError(
                            f'Rook not present on required square to castle: {char}')
                elif char == 'k':
                    sq_num = square_index(7, 7)
                    if self.is_set(game_state['-4'], sq_num):
                        game_state['castle_rights'] = set_bit(
                            game_state['castle_rights'], sq_num)
                    else:
                        raise ValueError(
                            f'Rook not present on required square to castle: {char}')
                elif char == 'Q':
                    sq_num = square_index(0, 0)
                    if self.is_set(game_state['4'], sq_num):
                        game_state['castle_rights'] = set_bit(
                            game_state['castle_rights'], sq_num)
                    else:
                        raise ValueError(
                            f'Rook not present on required square to castle: {char}')
                elif char == 'K':
                    sq_num = square_index(0, 7)
                    if self.is_set(game_state['4'], sq_num):
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
        return self.fenn_decode('rnbkqbnr/8/8/R6N/8/8/8/R6R w - - 0 1')
        return self.fenn_decode('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

    @property
    def action_space(self):
        return self.__action_space

    @property
    def state(self):
        return self.__state


gm = Chess()

seen = gm.get_seen_squares(gm.state, 1)
gm.print_bitboard(seen)

# from timeit import timeit
# print(timeit(lambda: gm.generate_ray(uint64(0), uint64(0),uint64(5),uint64(5)), number=10000))
