import itertools
import math
import queue
import select
import sys
import threading
from time import perf_counter

import chess
import torch

from actionspace import ActionSpace
from cache_read_priority import Cache
from mcts import MCTS
from resnet import ResNet
from state import State


class Engine():
    def __init__(self, args=None, device="cpu") -> None:
        self.__args = args
        self.__device = device
        self.__info = {
            "name": "Hubert++",
            "author": "Jakub Janicki"
        }

        # select interface
        self.__read_init()

    def __ready(self) -> None:
        """Initialize the engine"""

        # check args
        if self.__args is None:
            self.__args = {
                "num_searches": 2000,
                "c_puct": 1,
                "num_blocks": 10,
                "num_channels": 256,
                "model_path": "model.pt"
            }
        else:
            if "num_searches" not in self.__args:
                self.__args["num_searches"] = 2000
            if "c_puct" not in self.__args:
                self.__args["c_puct"] = 1
            if "num_blocks" not in self.__args:
                self.__args["num_blocks"] = 10
            if "num_channels" not in self.__args:
                self.__args["num_channels"] = 256
            if "model_path" not in self.__args:
                self.__args["model_path"] = "model.pt"

        self.__action_space = ActionSpace()
        # self.__model = ResNet(
        #     self.__args['num_blocks'], self.__args['num_channels'], self.__action_space.size, device)
        # self.__model.load_state_dict(torch.load(
        #     self.__args['model_path'], map_location=device))
        # self.__model.eval()
        # self.__mcts = MCTS(
        #     self.__args, self.__action_space, Cache(), self.__model)
        self.__mcts = MCTS(self.__args, self.__action_space,
                           Cache(max_size=1024))

        self.__default_go_args = {
            "ponder_move": None,
            "wtime": 0,
            "btime": 0,
            "winc": 0,
            "binc": 0,
            "depth": sys.maxsize ** 10,
            "nodes": sys.maxsize ** 10,
            "movestogo": 0,
            "movetime": 0,
            "infinite": False
        }
        self.__go_args = self.__default_go_args.copy()
        self.__search_interrupt = threading.Event()
        self.__search_result = queue.Queue()

        print("readyok")

    def __safe_input(self) -> str:
        """Read input and handle KeyboardInterrupt, EOFError"""

        try:
            return input()
        except (KeyboardInterrupt, EOFError):
            return "Unknown"

    def __calculate_search_time(self) -> int:
        """Calculate the time avaible for the search"""

        # get the time left for the current player
        time_left = self.__go_args["wtime"] if self.__mcts.root.state.turn else self.__go_args["btime"]
        # get the increment for the current player
        increment = self.__go_args["winc"] if self.__mcts.root.state.turn else self.__go_args["binc"]
        # return the time to search
        return math.log2(time_left) + increment

    def __perform_search(self) -> None:
        """
        Search for the best move until the stop event is set, 
        put best and ponder moves to parent class queue
        """

        # copy max_depth and nodes from go_args
        max_depth = self.__go_args["depth"]
        max_nodes = self.__go_args["nodes"]

        nodes_searched = 0
        max_depth_this_search = 0
        tic = perf_counter()

        # search until the stop event is set
        for node_num in range(max_nodes):
            depth_reached = self.__mcts.search_step(max_depth)
            # update search depth
            if depth_reached > max_depth_this_search:
                max_depth_this_search = depth_reached
            # check if the search was interrupted
            if self.__search_interrupt.is_set():
                # reset the stop event and break
                self.__search_interrupt.clear()
                nodes_searched = node_num
                break

        best_move, ponder_move = self.__mcts.best_move()
        toc = perf_counter()
        time_ms = int((toc-tic)*1000)
        nps = int(nodes_searched/(toc-tic))
        print("info depth", self.__mcts.depth, "seldepth", max_depth_this_search, "nodes",
              nodes_searched, "time", time_ms, "nps", nps)
        self.__search_result.put((best_move, ponder_move))

    def ponder(self, fen_move) -> None:
        move = chess.Move.from_uci(fen_move)
        self.__mcts.restrict_root([move])
        max_depth = self.__go_args["depth"]

        # continue ponder search until the engine is interrupted
        while True:
            # check input buffer
            input_str = self.__safe_input()
            if input_str:
                if input_str == "ponderhit":
                    # user played the ponder move, transition to normal search
                    self.__mcts.select_restricted_as_new_root()
                    break
                elif input_str.startswith("position"):
                    fen = "".join(input_str.split(maxsplit=1)[1:])
                    self.__mcts.root_from_fen(fen)
                    break
            else:
                self.__mcts.search_step(max_depth)
        self.__perform_search()

    def __search(self) -> None:
        """Manage the engine search thread and perform the search"""

        # check if the engine is already searching
        if self.__search_interrupt.is_set():
            # stop the current search
            self.__search_interrupt.set()
            # wait for the search to stop
            self.__search_thread.join()

        # start the search
        self.__search_thread = threading.Thread(target=self.__perform_search)
        self.__search_thread.start()

        # wait for the search to finish
        if self.__go_args["infinte"]:
            # wait for the stop command
            while self.__check_input_buffer() != "stop":
                pass
            self.__search_interrupt.set()
        self.__search_thread.join()

        # retreive the search result, output bestmove and ponder if available
        search_result = self.__search_result.get()
        if search_result[1] is None:
            print("bestmove", search_result[0])
        else:
            print("bestmove", search_result[0], "ponder", search_result[1])

    def __uci_go(self, go_command: str) -> None:
        """Parse the go command and perform the search"""

        self.__go_args = self.__default_go_args.copy()
        go_command_args = go_command.split()[1:]
        for i in range(0, len(go_command_args), 2):
            if go_command_args[i] == "ponder":
                self.__go_args["ponder_move"] = chess.Move.from_uci(
                    go_command_args[i+1])

            elif go_command_args[i] == "searchmoves":
                moves_str = go_command_args[i+1].split()
                moves = []
                for move_str in moves_str:
                    moves.append(chess.Move.from_uci(move_str))
                self.__mcts.restrict_root(moves)

            elif go_command_args[i] == "wtime":
                self.__go_args["wtime"] = int(go_command_args[i+1])

            elif go_command_args[i] == "btime":
                self.__go_args["btime"] = int(go_command_args[i+1])

            elif go_command_args[i] == "winc":
                self.__go_args["winc"] = int(go_command_args[i+1])

            elif go_command_args[i] == "binc":
                self.__go_args["binc"] = int(go_command_args[i+1])

            elif go_command_args[i] == "depth":
                self.__go_args["depth"] = int(go_command_args[i+1])

            elif go_command_args[i] == "nodes":
                self.__go_args["nodes"] = int(go_command_args[i+1])

            elif go_command_args[i] == "movestogo":
                self.__go_args["movestogo"] = int(go_command_args[i+1])

            elif go_command_args[i] == "movetime":
                self.__go_args["movetime"] = int(go_command_args[i+1])

            elif go_command_args[i] == "infinite":
                self.__go_args["infinite"] = True

        if self.__go_args["ponder_move"]:
            self.ponder(self.__go_args["ponder_move"])
        else:
            self.__search()

    def __uci(self) -> None:
        """Handle uci command"""

        # acknowledge uci interface
        print("id name", self.__info["name"])
        print("id author", self.__info["author"])
        print("uciok")

        while True:
            command = self.__safe_input()
            if command == "quit":
                break

            elif command == "isready":
                self.__ready()
                print("readyok")

            elif command == "ucinewgame":
                self.__mcts.reset()

            # TODO "position startpos moves e2e4 e7e5"
            elif command.startswith("position"):
                fen = "".join(command.split(maxsplit=1)[1:])
                if fen[0] == "startpos":
                    fen = "startpos"
                self.__mcts.root_from_fen(fen)

            elif command.startswith("go"):
                best_move, ponder = self.__uci_go(command)
                print("bestmove", best_move, "ponder", ponder)

    def __read_init(self) -> None:
        """Read the init command"""

        while True:
            try:
                command = input()
                if command == "quit":
                    break

                elif command == "uci":
                    self.__uci()
            except (KeyboardInterrupt, EOFError):
                break

# "position r1qr1b2/1R3pkp/3p2pN/ppnPp1Q1/bn2P3/4P2P/pBBP2P1/5RK1 w - - 0 1"
# args = {"num_searches": 2000, "c_puct": 1}
# engine = Engine(args)
# engine.uci_read()
