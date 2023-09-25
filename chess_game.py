from copy import deepcopy
from timeit import timeit

import chess as chesslib
import numba as nb
import numpy as np
from numba import njit, prange
from numpy import int8, int32, uint8, uint64

from actionspace import ActionSpace
from move import Move
from state import State


def print_bitboard(bitboard) -> None:
    bitboard = np.binary_repr(bitboard, width=64)
    for i in range(8):
        for j in range(7, -1, -1):
            print(bitboard[i*8+j], end=' ')
        print()
    print()


@njit(nb.uint64(nb.uint64, nb.uint8))
def set_bit(bitboard: uint64, bit_num: uint8) -> uint64:
    return bitboard | (1 << bit_num)


@njit()
def clear_bit(bitboard: uint64, bit_num: uint8) -> uint64:
    return bitboard & ~(1 << bit_num)


@njit(nb.boolean(nb.uint64, nb.uint8))
def is_set(bitboard: uint64, bit_num: uint8) -> uint8:
    if bitboard & (1 << bit_num):
        return True
    else:
        return False


@njit(nb.uint8(nb.uint8, nb.uint8))
def square_index(rank: uint8, file: uint8) -> uint8:
    return rank * 8 + file


@njit(nb.types.UniTuple(nb.uint8, 2)(nb.uint8))
def get_rank_file(square_index: uint8) -> tuple[uint8, uint8]:
    return square_index // 8, square_index % 8


@njit()
def square_index_binary(bitboard: uint64) -> uint8:
    return np.log2(bitboard)


@njit()
def idx_of_set_bits(bitboard: uint64) -> list[uint8]:
    indices = []
    while bitboard:
        lsb = bitboard & -bitboard
        indices.append(int(np.log2(lsb)))
        bitboard ^= lsb
    return indices


@njit()
def is_in_bounds(y: uint8, x: uint8) -> bool:
    return 0 <= y < 8 and 0 <= x < 8


@njit()
def is_in_bounds_sq_num(sq_num: uint8) -> bool:
    return 0 <= sq_num < 64


@njit()
def generate_ray(rank: uint8, file: uint8, dst_rank: uint8,
                 dst_file: uint8) -> tuple[uint64, uint8]:
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
def get_vision_map(occupied: uint64, y: uint8, x: uint8) -> uint64:
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
        self.__action_space: ActionSpace = self.generate_actionspace()

    def bind_move(self, rank, file) -> tuple[int, int]:
        return max(0, min(rank, self.__board_size - 1)), max(0, min(file, self.__board_size - 1))

    def piece_id(self, piece) -> np.number:
        return abs(piece)

    def piece_color(self, piece) -> np.number:
        return np.sign(piece)

    def int_from_char(self, char) -> int:
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
                p = board[rank, file]
                p_id = self.piece_id(p)
                if p == 0 or self.piece_color(p) != turn:
                    continue
                # calculate normal movement for empty squares
                pseudo[rank, file] = self.__move_maps[p][rank, file] & empty
                # calculate attack moves
                pseudo[rank, file] |= self.__attack_maps[p][rank, file] & enemy
                if p_id != 2:
                    # keep pin on enemy king
                    ray_to_king = 0
                    if pseudo[rank, file] & uint64(game_state[f'{(-turn)*6}']):
                        ray_to_king, ray_dir = generate_ray(
                            rank, file, *get_rank_file(square_index_binary(uint64(game_state[f'{(-turn)*6}']))))
                    # remove unwanted pins
                    pseudo[rank, file] &= uint64(
                        get_vision_map(~empty, rank, file))
                    pseudo[rank, file] |= uint64(ray_to_king)
        return pseudo

    def get_move_map(self, piece_id: int, rank: int, file: int) -> uint64:
        if piece_id == 0:
            pass
        return self.__move_maps[piece_id][rank, file]

    def get_attack_map(self, piece_id: int, rank: int, file: int) -> uint64:
        if piece_id == 0:
            pass
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

        ray_pieces = np.where(friendly_ids > 2, True, False)
        friendly_idx = np.array(friendly_idx)
        friendly_ray_idx = friendly_idx[:, ray_pieces]
        ray_attack_maps = attack_maps[ray_pieces]

        # get vision map of each piece as if it were a queen
        # force explicit return type to uint64 to avoid sys.maxsize overflow
        vision_maps = np.vectorize(get_vision_map, otypes=[np.uint64])(
            occupied, friendly_ray_idx[0], friendly_ray_idx[1])

        vision_map = uint64(0)
        for map, attack_map in zip(vision_maps, ray_attack_maps):
            vision_map |= map & attack_map

        # add all possible attacks of non-ray pieces
        for attack_map in attack_maps[~ray_pieces]:
            vision_map |= attack_map
        return vision_map

    def get_masks(self, game_state: State) -> tuple((uint64, uint64, uint64, uint64)):
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
            # include attacker in checkmask
            checkmask |= uint64(set_bit(
                checkmask, square_index(attacker[0], attacker[1])))

            attacker_type = self.piece_id(board[attacker[0], attacker[1]])
            if attacker_type == 2:
                continue

            ray, ray_dir_bitrep = generate_ray(
                attacker[0], attacker[1], king_rank, king_file)
            checkmask |= uint64(ray)
            # print_bitboard(checkmask)

            # extend the pinmask to behind the king (prevent king from staying in check),
            # only if attacker could attack that square
            pin = uint64(0)
            sq_behind = square_index(king_rank, king_file) + ray_dir_bitrep
            if is_in_bounds_sq_num(sq_behind):
                attacking_piece = board[attacker[0], attacker[1]]
                attacker_move_map = self.get_attack_map(
                    attacking_piece, attacker[0], attacker[1])
                if is_set(attacker_move_map, sq_behind):
                    pin |= uint64(set_bit(pin, sq_behind))
            # print_bitboard(pin)

            if attacker[0] == king_rank or attacker[1] == king_file:
                # include attacker in pinmask
                pinmask_hv |= pin | uint64(
                    set_bit(ray, square_index(attacker[0], attacker[1])))
            else:
                # include attacker in pinmask
                pinmask_diag |= pin | uint64(
                    set_bit(ray, square_index(attacker[0], attacker[1])))

            # # correct checkmask to only include squares that are seen by the attacker
            # checkmask &= seen_mask
            # pinmask_hv &= seen_mask
            # pinmask_diag &= seen_mask
            # # print_bitboard(checkmask)

        if pinmask_hv == 0:
            # pinmask_hv = uint64(np.iinfo(np.uint64).max)
            pinmask_hv = uint64(18446744073709551615)
        if pinmask_diag == 0:
            pinmask_diag = uint64(18446744073709551615)
        # print_bitboard(checkmask)
        # print_bitboard(king)
        # print_bitboard(seen_mask)

        if not uint64(game_state[f'{game_state["turn"]*6}']) & seen_mask:
            # king is not in check, so all moves are legal
            checkmask = uint64(18446744073709551615)
        return (checkmask, pinmask_hv, pinmask_diag, seen_mask)

    def get_legal_moves(self, game_state: State) -> np.ndarray:
        enemy_or_empty = game_state.enemy_or_empty
        empty = game_state.empty
        pseudo_legal_moves = self.get_pseudo_legal_moves(
            game_state, game_state['turn'])
        checkmask, pinmask_hv, pinmask_diag, enemy_visionmask = self.get_masks(
            game_state)
        board = game_state.board

        self.print_board(game_state)
        print_bitboard(checkmask)
        print_bitboard(pinmask_hv)
        print_bitboard(pinmask_diag)
        print_bitboard(enemy_visionmask)
        print_bitboard(enemy_or_empty)
        print_bitboard(game_state['-2'])
        print_bitboard(game_state['en_passant_target'])

        legal_moves = []
        for rank in range(self.__board_size):
            for file in range(self.__board_size):
                piece = board[rank, file]
                if piece == 0 or self.piece_color(piece) != game_state['turn']:
                    continue
                piece_id = self.piece_id(piece)
                composite_pin = pinmask_diag & pinmask_hv

                # calculate legal moves for each piece
                move_bitboard = pseudo_legal_moves[rank,
                                                   file] & enemy_or_empty
                if rank == 7 and file == 0:
                    print_bitboard(move_bitboard)
                if piece_id > 2:
                    seen_mask = uint64(get_vision_map(~empty, rank, file)) & uint64(
                        self.__attack_maps[piece_id][rank, file])
                    move_bitboard &= seen_mask
                if rank == 7 and file == 0:
                    print_bitboard(move_bitboard)

                sq_id = uint8(square_index(rank, file))
                if piece_id == 6:
                    move_bitboard &= ~enemy_visionmask & ~(
                        composite_pin & ~checkmask)
                else:
                    if is_set(pinmask_diag, sq_id):
                        move_bitboard &= pinmask_diag
                    if is_set(pinmask_hv, sq_id):
                        move_bitboard &= pinmask_hv
                    move_bitboard &= checkmask
                
                if rank == 7 and file == 0:
                    print_bitboard(move_bitboard)
                # calculate castling moves
                if piece_id == 6 and (not is_set(checkmask, sq_id)
                                      or checkmask == np.iinfo(np.uint64).max):
                    # king side
                    if is_set(game_state['castle_rights'], square_index(rank, 7)):
                        if is_set(empty, square_index(rank, 5)) \
                                and is_set(empty, square_index(rank, 6)) \
                                and not is_set(enemy_visionmask, square_index(rank, 5)) \
                                and not is_set(enemy_visionmask, square_index(rank, 6)):
                            legal_moves.append(Move(rank, file, rank, 6, 0))
                    # queen side
                    if is_set(game_state['castle_rights'], square_index(rank, 0)):
                        if is_set(empty, square_index(rank, 1)) \
                                and is_set(empty, square_index(rank, 2)) \
                                and is_set(empty, square_index(rank, 3)) \
                                and not is_set(enemy_visionmask, square_index(rank, 2)) \
                                and not is_set(enemy_visionmask, square_index(rank, 3)):
                            legal_moves.append(
                                Move(rank, file, rank, 2, 0))

                # calculate pawn moves
                if piece_id == 1:
                    # calculate en passant moves
                    if uint64(game_state['en_passant_target']) & \
                            self.__attack_maps[piece][rank, file] != 0 \
                            and not is_set(composite_pin, sq_id):
                        legal_moves.append(
                            Move(rank, file, *get_rank_file(square_index_binary(uint64(game_state['en_passant_target'])))))

                    # calculate promotions if pawn is on last rank or normal moves
                    if (rank == 1 and game_state['turn'] == -1) or (rank == 6 and game_state['turn'] == 1):
                        for move in idx_of_set_bits(move_bitboard):
                            # enable promotion to multiple pieces
                            for promo_piece in [2, 3, 4, 5]:
                                legal_moves.append(
                                    Move(rank, file, *get_rank_file(move), promo_piece * piece))
                    else:
                        # add moves to list
                        for move in idx_of_set_bits(move_bitboard):
                            legal_moves.append(
                                Move(rank, file, *get_rank_file(move)))
                else:
                    # add moves to list
                    for move in idx_of_set_bits(move_bitboard):
                        legal_moves.append(
                            Move(rank, file, *get_rank_file(move)))
        return legal_moves

    def generate_attacks(self) -> None:
        self.__attack_maps = deepcopy(self.__move_maps)

        vert_line = 0
        for i in range(8):
            vert_line = set_bit(vert_line, uint64(i * 8))

        for file in range(8):
            tmp = uint64(vert_line << file)
            for rank in range(8):
                # distinguish pawn attacks from normal forward movement
                self.__attack_maps[1][rank, file] &= ~tmp
                self.__move_maps[1][rank, file] &= tmp
                self.__attack_maps[-1][rank, file] &= ~tmp
                self.__move_maps[-1][rank, file] &= tmp

    def generate_maps(self) -> None:
        self.generate_moves()
        self.generate_attacks()

    def generate_actionspace(self) -> None:
        action_space = ActionSpace()
        for piece_id in [5, 2, 1, -1]:
            for rank in range(self.__board_size):
                for file in range(self.__board_size):
                    move_bitboard = self.__move_maps[piece_id][rank, file] | \
                        self.__attack_maps[piece_id][rank, file]
                    for indexes in idx_of_set_bits(move_bitboard):
                        action_space.add(
                            Move(rank, file, *get_rank_file(indexes)))

                    # calculate promotions
                    if (piece_id == 1 and rank == 6) or (piece_id == -1 and rank == 1):
                        for indexes in idx_of_set_bits(move_bitboard):
                            # enable promotion to multiple pieces
                            for promo_piece in [2, 3, 4, 5]:
                                action_space.add(
                                    Move(rank, file, *get_rank_file(indexes), promo_piece * piece_id))
        return action_space

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
                rank = self.int_from_char(rank_char) - 1
                file = self.int_from_char(file_char)
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

    def fenn_encode(self, game_state: State) -> str:
        piece_char = {
            1: 'P',
            2: 'N',
            3: 'B',
            4: 'R',
            5: 'Q',
            6: 'K',
            -1: 'p',
            -2: 'n',
            -3: 'b',
            -4: 'r',
            -5: 'q',
            -6: 'k',
        }
        fenn_string = ''
        board = game_state.board
        for rank in range(self.__board_size - 1, -1, -1):
            empty = 0
            for file in range(self.__board_size):
                piece = board[rank, file]
                if piece == 0:
                    empty += 1
                else:
                    if empty > 0:
                        fenn_string += str(empty)
                        empty = 0
                    fenn_string += piece_char[piece]
            if empty > 0:
                fenn_string += str(empty)
            if rank > 0:
                fenn_string += '/'
        fenn_string += ' '
        if game_state['turn'] == 1:
            fenn_string += 'w'
        else:
            fenn_string += 'b'
        fenn_string += ' '
        if game_state['castle_rights'] == 0:
            fenn_string += '-'
        else:
            if is_set(game_state['castle_rights'], square_index(0, 7)):
                fenn_string += 'K'
            if is_set(game_state['castle_rights'], square_index(0, 0)):
                fenn_string += 'Q'
            if is_set(game_state['castle_rights'], square_index(7, 7)):
                fenn_string += 'k'
            if is_set(game_state['castle_rights'], square_index(7, 0)):
                fenn_string += 'q'
        fenn_string += ' '
        if game_state['en_passant_target'] == 0:
            fenn_string += '-'
        else:
            y, x = get_rank_file(square_index_binary(
                game_state['en_passant_target']))
            fenn_string += chr(x + 97)
            fenn_string += chr(y + 49)
        fenn_string += ' '
        fenn_string += str(game_state['halfmove'])
        fenn_string += ' '
        fenn_string += str(game_state['fullmove'])
        return fenn_string

    def load_default(self) -> State:
        # return self.fenn_decode('rnbqkbnr/pppppppp/8/8/8/8//7K w - - 0 1')
        return self.fenn_decode('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

    def move(self, src_str, dst_str) -> None:
        src_rank, src_file = self.int_from_char(
            src_str[1]), self.int_from_char(src_str[0])
        dst_rank, dst_file = self.int_from_char(
            dst_str[1]), self.int_from_char(dst_str[0])
        move = Move(src_rank, src_file, dst_rank, dst_file)
        self.__state = self.__state.make_move(move)

    def print_board_reverse(self) -> None:
        # for i in range(8):
        for i in range(7, -1, -1):
            # for j in range(8):
            for j in range(7, -1, -1):
                piece = self.__state.board[i, j]
                if piece == 0:
                    print('.', end=' ')
                elif piece == 1:
                    print('P', end=' ')
                elif piece == 2:
                    print('N', end=' ')
                elif piece == 3:
                    print('B', end=' ')
                elif piece == 4:
                    print('R', end=' ')
                elif piece == 5:
                    print('Q', end=' ')
                elif piece == 6:
                    print('K', end=' ')
                elif piece == -1:
                    print('p', end=' ')
                elif piece == -2:
                    print('n', end=' ')
                elif piece == -3:
                    print('b', end=' ')
                elif piece == -4:
                    print('r', end=' ')
                elif piece == -5:
                    print('q', end=' ')
                elif piece == -6:
                    print('k', end=' ')
            print()
        print()

    def print_board(self, game_state = None) -> None:
        if game_state is None:
            game_state = self.__state
        # for i in range(8):
        for i in range(7, -1, -1):
            for j in range(8):
                # for j in range(7, -1, -1):
                piece = game_state.board[i, j]
                if piece == 0:
                    print('.', end=' ')
                elif piece == 1:
                    print('P', end=' ')
                elif piece == 2:
                    print('N', end=' ')
                elif piece == 3:
                    print('B', end=' ')
                elif piece == 4:
                    print('R', end=' ')
                elif piece == 5:
                    print('Q', end=' ')
                elif piece == 6:
                    print('K', end=' ')
                elif piece == -1:
                    print('p', end=' ')
                elif piece == -2:
                    print('n', end=' ')
                elif piece == -3:
                    print('b', end=' ')
                elif piece == -4:
                    print('r', end=' ')
                elif piece == -5:
                    print('q', end=' ')
                elif piece == -6:
                    print('k', end=' ')
            print()
        print()

    def set_state(self, state: State) -> None:
        self.__state = state

    def perft(self, depth: int, state: State) -> int:
        if depth == 0:
            return 1
        nodes = 0
        for move in self.get_legal_moves(state):
            nodes += self.perft(depth - 1, state.get_next_state(move))
        return nodes

    def perft_debug(self, depth: int, state: State) -> int:
        if depth == 0:
            return 1
        nodes = 0
        moves = self.get_legal_moves(state)

        board = chesslib.Board(self.fenn_encode(state))
        legal_moves = list(board.legal_moves)
        if len(moves) != len(legal_moves):
            print(self.fenn_encode(state))
            print(f'legal moves: {len(legal_moves)}')
            moves = [move.uci_str() for move in moves]
            legal_moves = [move.uci() for move in legal_moves]
            for move in moves:
                if move not in legal_moves:
                    print(f'is illegal: {move}')
            for move in legal_moves:
                if move not in moves:
                    xd = self.get_legal_moves(state)
                    print(f'is legal: {move}')
            exit()

        for move in moves:
            nodes += self.perft_debug(depth - 1, state.get_next_state(move))
        return nodes

    @property
    def action_space(self):
        return self.__action_space

    @property
    def state(self):
        return self.__state
