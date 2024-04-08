import chess


class ActionSpace():
    def __init__(self) -> None:
        self.__actionspace: dict = dict()
        self.__key_map: dict = dict()
        self.__size: int = 0
        self.__load_from_file()

    def __getitem__(self, key) -> tuple:
        return self.__actionspace[key]

    def _add(self, obj) -> None:
        if obj in self.__key_map:
            return
        self.__actionspace[self.__size] = obj
        self.__key_map[obj] = self.__size
        self.__size += 1

    def get_key(self, obj) -> int:
        return self.__key_map[obj]

    def __load_from_file(self) -> None:
        """Load the action space from file if it exists, otherwise calculate it."""

        try:
            self.__load()
        except FileNotFoundError:
            self.__calculate()
            self.__save()

    def __calculate(self) -> None:
        """Calculate the action space of chess."""

        # Create an empty chess board
        board = chess.Board()
        # Initialize an empty list to store all possible moves
        action_space = []
        # Iterate over all squares and all piece types
        for square in chess.SQUARES:
            for piece_type in chess.PIECE_TYPES:
                for color in chess.COLORS:
                    # Set player to color
                    board.turn = color
                    # Place piece on given square
                    board.set_piece_at(square, chess.Piece(piece_type, color))

                    # If the piece is a pawn, place other pawns around it to include attacks
                    if piece_type == chess.PAWN:
                        other_color = chess.WHITE if color == chess.BLACK else chess.BLACK
                        for square_offset in [-7, -9, 7, 9]:
                            if square + square_offset in chess.SQUARES:
                                board.set_piece_at(
                                    square + square_offset, chess.Piece(piece_type, other_color))

                    for move in board.legal_moves:
                        # If the move is not already in the action space, add it
                        if move not in action_space:
                            action_space.append(move)

                    # Clear the board
                    board.clear()
        action_space = sorted(action_space, key=lambda x: x.uci())
        for action in action_space:
            self._add(action)

    def __save(self) -> None:
        with open("actionspace/actionspace.txt", "w") as f:
            for i in range(self.__size):
                f.write(f"{self.__actionspace[i]}\n")

    def __load(self) -> None:
        with open("actionspace.txt", "r") as f:
            for line in f:
                self._add(chess.Move.from_uci(line.strip()))

    @property
    def size(self) -> int:
        return self.__size
