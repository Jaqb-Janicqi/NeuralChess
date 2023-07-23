import tkinter as tk


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


# basic colors
WHITE = rgb_to_hex((255, 255, 255))
GREY = rgb_to_hex((128, 128, 128))
YELLOW = rgb_to_hex((204, 204, 0))
BLUE = rgb_to_hex((50, 255, 255))
BLACK = rgb_to_hex((0, 0, 0))


class Gui:
    def __init__(self, alphazero, board_size, game, actionspace, debug_mode=False):
        self.game = game
        self.actionspace = actionspace
        self.board_size = board_size
        self.alphazero = alphazero
        self.debug_mode = debug_mode
        self.player_color = None
        self.state = None

        # initialise display
        self.window = tk.Tk()
        self.window.title('Chess')
        self.window.geometry('800x600')
        self.window.resizable(False, False)
        self.__pieces = self.load_pieces()
        self.selected_piece = None
        self.left_frame = tk.Frame(
            self.window, width=200, height=600, bg='grey')
        self.left_frame.pack(side=tk.LEFT)
        self.right_frame = tk.Frame(
            self.window, width=600, height=600, bg='white')
        self.right_frame.pack(side=tk.RIGHT)
        self.pixel = tk.PhotoImage(width=0, height=0)
        self.__button_board = [[None for file in range(
            board_size)] for rank in range(board_size)]
        self.init_complete = False
        self.init_gui()
        self.last_player_action = None

    def init_gui(self):
        text = 'Play as White'
        button_white = tk.Button(
            self.left_frame, bg='white', text=text)
        button_white.config(borderwidth=0)
        button_white.config(command=self.play_white)
        button_white.bind('<Button-1>')
        button_white.pack(side=tk.TOP, pady=10)

        text = 'Play as Black'
        button_black = tk.Button(
            self.left_frame, bg='white', text=text)
        button_black.config(borderwidth=0)
        button_black.config(command=self.play_black)
        button_black.bind('<Button-1>')
        button_black.pack(side=tk.TOP, pady=10)

        self.left_frame.pack(side=tk.LEFT)
        self.window.mainloop()

    def init_state(self):
        self.init_complete = False
        self.state = self.game.load_default(self.board_size)
        self.init_board()
        self.redraw_board()

    def play_white(self):
        self.player_color = 1
        self.init_state()
        self.init_complete = True

    def play_black(self):
        self.player_color = -1
        self.init_state()
        self.init_complete = True

        if self.state.win == 0 and self.debug_mode is False:
            self.alphazero_move()
            self.redraw_board()

    def init_board(self):
        for rank in range(self.board_size):
            for file in range(self.board_size):
                if (rank + file) % 2 == 0:
                    color = WHITE
                else:
                    color = GREY
                square = self.state.square(rank, file)
                text = str(file) + str(' ') + str(rank)
                if square == 0:
                    button = tk.Button(
                        self.right_frame, image=self.pixel, width=98, height=98, bg=color, text=text)
                    button.config(borderwidth=0)
                    button.config(command=lambda file=file,
                                  rank=rank: self.click(rank, file))
                    self.__button_board[rank][file] = button  # type: ignore
                else:
                    square_color = self.game.get_color(square)
                    image = self.__pieces[str(square_color)][str(abs(square))]
                    button = tk.Button(
                        self.right_frame, image=image, width=98, height=98, bg=color, text=text)
                    button.config(borderwidth=0)
                    button.config(command=lambda file=file,
                                  rank=rank: self.click(rank, file))
                    self.__button_board[rank][file] = button  # type: ignore
                if self.player_color == 1:
                    button.grid(column=file, row=self.board_size - rank - 1)
                else:
                    button.grid(column=self.board_size - file - 1, row=rank)
                button.bind('<Button-1>', self.click(rank, file))
        self.right_frame.pack(side=tk.RIGHT)
        self.color_board()
        self.selected_piece = None

    def color_board(self):
        for rank in range(self.board_size):
            for file in range(self.board_size):
                if (rank + file) % 2 == 0:
                    color = WHITE
                else:
                    color = GREY
                self.__button_board[rank][file].config(
                    bg=color)  # type: ignore

    def redraw_board(self):
        for rank in range(self.board_size):
            for file in range(self.board_size):
                square = self.state.square(rank, file)
                if square == 0:
                    image = self.pixel
                else:
                    square_color = self.game.get_color(square)
                    image = self.__pieces[str(square_color)][str(abs(square))]
                self.__button_board[rank][file].config(
                    image=image)  # type: ignore
        self.window.update_idletasks()
        self.window.update()

    def click(self, rank, file):
        if not self.init_complete:
            return
        if self.state.win != 0:
            return
        square = self.state.square(rank, file)
        square_color = self.game.get_color(square)
        if self.selected_piece is None:
            if square != 0 and square_color == self.state.player_turn:
                self.selected_piece = (rank, file)
                print(self.selected_piece)
                self.highlight_moves(rank, file)
        elif self.selected_piece == (rank, file):
            # deselect piece
            print('deselect piece' + str(self.selected_piece))
            self.selected_piece = None
            self.unhighlight_moves(rank, file)
        else:
            # move piece
            self.unhighlight_moves(
                self.selected_piece[0], self.selected_piece[1])  # type: ignore
            moved = self.move_piece(rank, file)
            self.selected_piece = None
            self.redraw_board()
            if self.state.win == 0 and moved is True and self.debug_mode is False:
                self.alphazero_move()
                self.redraw_board()

    def alphazero_move(self):
        best_action = self.alphazero.predict(self.state)
        self.state = self.state.next_state(best_action)

    def move_piece(self, rank, file):
        src_rank, src_file = self.selected_piece
        move_hashes = self.game.generate_one_piece(
            self.state.board, src_rank, src_file)
        moves = []
        for move_hash in move_hashes:
            moves.append(self.actionspace.get(move_hash))
        move = list(filter(lambda move: (move.dst_rank,
                    move.dst_file) == (rank, file), moves))
        if len(move) == 0:
            return False
        self.state = self.state.next_state(move[0])
        self.redraw_board()
        return True

    def highlight_moves(self, rank, file):
        square = self.state.square(rank, file)
        square_color = self.game.get_color(square)
        if square_color == self.state.player_turn:
            move_hashes = self.game.generate_one_piece(
                self.state.board, rank, file)
            moves = []
            for move_hash in move_hashes:
                moves.append(self.actionspace.get(move_hash))
            for move in moves:
                self.highlight_square(move.dst_rank, move.dst_file)

    def highlight_square(self, rank, file):
        self.__button_board[rank][file].config(bg=BLUE)

    def unhighlight_moves(self, rank, file):
        move_hashes = self.game.generate_one_piece(
            self.state.board, rank, file)
        moves = []
        for move_hash in move_hashes:
            moves.append(self.actionspace.get(move_hash))
        for move in moves:
            self.unhighlight_square(move.dst_rank, move.dst_file)

    def unhighlight_square(self, rank, file):
        if (file + rank) % 2 == 0:
            color = WHITE
        else:
            color = GREY
        self.__button_board[rank][file].config(bg=color)

    def load_pieces(self):
        pieces = {}
        pieces['1'] = {}
        pieces['-1'] = {}
        pieces['1']['1'] = tk.PhotoImage(file='assets/white_pawn.png')
        pieces['1']['2'] = tk.PhotoImage(file='assets/white_knight.png')
        pieces['1']['3'] = tk.PhotoImage(file='assets/white_bishop.png')
        pieces['1']['4'] = tk.PhotoImage(file='assets/white_rook.png')
        pieces['1']['5'] = tk.PhotoImage(file='assets/white_quuen.png')
        pieces['1']['6'] = tk.PhotoImage(file='assets/white_king.png')
        pieces['-1']['1'] = tk.PhotoImage(file='assets/black_pawn.png')
        pieces['-1']['2'] = tk.PhotoImage(file='assets/black_knight.png')
        pieces['-1']['3'] = tk.PhotoImage(file='assets/black_bishop.png')
        pieces['-1']['4'] = tk.PhotoImage(file='assets/black_rook.png')
        pieces['-1']['5'] = tk.PhotoImage(file='assets/black_quuen.png')
        pieces['-1']['6'] = tk.PhotoImage(file='assets/black_king.png')
        return pieces
