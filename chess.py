import numpy as np

from actionspace import ActionSpace
from custom_types import GameState, Move, allocate_game_state, allocate_move, allocate_move_arr
from state import State


class Chess():
    def __init__(self, args) -> None:
        self.__args = args
        self.__board_size: int = 8
        self.__state: State = self.load_default(8)
        self.__action_space: ActionSpace = ActionSpace(8, 8)
        self.__move_maps = {
            1: {},
            -1: {},
            2: {},
            3: {},
            4: {},
            5: {},
            6: {}
        }
        self.move_functions = {
            1: self.generate_pawn,
            2: self.generate_knight,
            3: self.generate_sliding,
            4: self.generate_sliding,
            5: self.generate_sliding,
            6: self.generate_sliding
        }
        self.move_range = {
            3: 999,
            4: 999,
            5: 999,
            6: 1
        }

    def set_bit(self, bitboard, bit_num) -> None:
        return bitboard | (np.uint64(1) << bit_num)

    def clear_bit(self, bitboard, bit_num) -> None:
        return bitboard & ~(np.uint64(1) << bit_num)

    def is_set(self, bitboard, bit_num) -> np.integer:
        if bitboard & (np.uint64(1) << bit_num):
            return True
        return False

    def square_index(self, rank, file) -> np.uint8:
        return np.uint8(rank * self.__board_size + file)

    def move_piece(self, bitboard, src_sq, dst_sq) -> None:
        self.set_bit(bitboard, dst_sq)
        self.clear_bit(bitboard, src_sq)

    def is_in_bounds(self, rank, file) -> bool:
        return 0 <= rank < self.__board_size and 0 <= file < self.__board_size

    def bind_move(self, rank, file) -> tuple[int, int]:
        return max(0, min(rank, self.__board_size - 1)), max(0, min(file, self.__board_size - 1))

    def piece_id(self, piece) -> np.number:
        return abs(piece)

    def piece_color(self, piece) -> np.number:
        return np.sign(piece)

    def int_from_char(char) -> np.uint8:
        if char.isdigit():
            return np.uint8(ord(char) - 48)
        return np.uint8(ord(char) - 97)

    def generate_sliding(self, board_size: np.uint8, src_rank: np.uint8,
                         src_file: np.uint8, piece: np.int8) -> list[int]:
        piece_id = self.piece_id(piece)
        move_range = self.move_range[piece_id]
        start_rank, start_file = self.bind_move(
            src_rank - move_range,
            src_file - move_range,
            board_size)
        end_rank, end_file = self.bind_move(
            src_rank + move_range,
            src_file + move_range,
            board_size)

        directions = []
        if piece_id == 4 or piece_id == 5 or piece_id == 6:
            directions += [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if piece_id == 3 or piece_id == 5 or piece_id == 6:
            directions += [(1, 1), (-1, 1), (1, -1), (-1, -1)]

        move_bitboard = np.uint64(0)
        for rank_dir, file_dir in directions:
            rank, file = src_rank + rank_dir, src_file + file_dir
            while start_rank <= rank <= end_rank and start_file <= file <= end_file:
                self.set_bit(move_bitboard, self.square_index(rank, file))
                rank += rank_dir
                file += file_dir
        return move_bitboard

    def generate_knight(self, board: np.ndarray, src_rank: np.uint8,
                        src_file: np.uint8, piece: np.int8) -> list[int]:
        move_bitboard = np.uint64(0)
        for file_dir, rank_dir in [(2, 1), (2, -1), (-2, 1), (-2, -1),
                                   (1, 2), (1, -2), (-1, 2), (-1, -2)]:
            rank, file = src_rank + rank_dir, src_file + file_dir
            if self.is_in_bounds(rank, file, board.shape[0]):
                self.set_bit(move_bitboard, self.square_index(rank, file))
        return move_bitboard

    def generate_pawn(self, board_size: np.uint8, src_rank: np.uint8,
                      src_file: np.uint8, piece: np.int8) -> list[int]:
        rank = src_rank + piece
        move_bitboard = np.uint64
        self.set_bit(move_bitboard, self.square_index(rank, file))

        for side_dir in [-1, 1]:
            file = src_file + side_dir
            if self.is_in_bounds(rank, file, board_size):
                self.set_bit(move_bitboard, self.square_index(rank, file))
        return move_bitboard

    def generate_moves(self):
        pass  # TODO

    def next_game(self) -> None:
        self.__state = self.load_default(self.__board_size)

    def add_piece(self, game_state, char, color, piece_id, rank, file) -> None:
        square_index = self.square_index(rank, file)
        game_state[char] = self.set_bit(game_state[char], square_index)
        str_color = 'black' if color == -1 else 'white'
        game_state[str_color] = self.set_bit(
            game_state[str_color], square_index)
        game_state['occupied'] = self.set_bit(
            game_state['occupied'], square_index)
        game_state['board'][rank, file] = piece_id

    def fenn_decode(self, fenn_string) -> State:
        board_size = 1
        for char in fenn_string:
            if char == '/':
                board_size += 1
        if board_size == 0:
            raise ValueError(f'Invalid fenn string {fenn_string}')
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
        game_state = allocate_game_state()
        state = State(game_state)
        file = 0
        rank = board_size - 1

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
                r, file = np.divmod(file, board_size)
                rank -= r
            elif char in game_state.dtype.fields:
                self.add_piece(game_state, char, self.piece_color(piece_id[char]),
                               piece_id[char], rank, file)
                file += 1
                r, file = np.divmod(file, board_size)
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
                if char == 'q':
                    sq_num = self.square_index(7, 0)
                    if self.is_set(game_state['r'], sq_num):
                        game_state['castle_rights'] = self.set_bit(
                            game_state['castle_rights'], sq_num)
                    else:
                        raise ValueError(
                            f'Rook not present on required square to castle: {char}')
                elif char == 'k':
                    sq_num = self.square_index(7, 7)
                    if self.is_set(game_state['r'], sq_num):
                        game_state['castle_rights'] = self.set_bit(
                            game_state['castle_rights'], sq_num)
                    else:
                        raise ValueError(
                            f'Rook not present on required square to castle: {char}')
                elif char == 'Q':
                    sq_num = self.square_index(0, 0)
                    if self.is_set(game_state['R'], sq_num):
                        game_state['castle_rights'] = self.set_bit(
                            game_state['castle_rights'], sq_num)
                    else:
                        raise ValueError(
                            f'Rook not present on required square to castle: {char}')
                elif char == 'K':
                    sq_num = self.square_index(0, 7)
                    if self.is_set(game_state['R'], sq_num):
                        game_state['castle_rights'] = self.set_bit(
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
                sq_id = self.square_index(rank, file)
                game_state['en_passant_target'] = self.set_bit(
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
        return state

    def load_default(self, board_size: int) -> State:
        return self.fenn_decode('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

    @property
    def action_space(self):
        return self.__action_space


ch = Chess(dict())
