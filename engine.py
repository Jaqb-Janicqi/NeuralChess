import cProfile
import math
import os
import select
import sys
import threading
from contextlib import contextmanager
from time import perf_counter, sleep

import chess
import torch
import yaml

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
            "name": "RoboHub",
            "author": "Jakub Janicki"
        }
        self.__model_params = {
            "num_blocks": 19,
            "num_channels": 256
        }
        self.__debug = False
        self.__infinity = sys.maxsize ** 10
        self.__input_interrupt = threading.Event()
        self.__search_interrupt = threading.Event()

        # select interface
        self.__read_init()

    def __read_config(self) -> None:
        """Read the config file"""

        # read config.yaml
        with open("config.yaml", "r") as config_file:
            self.__args = yaml.safe_load(config_file)

    def __check_config(self) -> None:
        """Check if the config file contains all the required parameters"""

        if "num_searches" not in self.__args:
            self.__args["num_searches"] = 4000
        if "c_puct" not in self.__args:
            self.__args["c_puct"] = 1
        if "model_path" not in self.__args:
            self.__args["model_path"] = "model.pt"
        if "cache_size" not in self.__args:
            self.__args["cache_size"] = 512

    def __ready(self) -> None:
        """Initialize the engine"""

        # read the config file
        self.__read_config()
        # check if the config file contains all the required parameters
        self.__check_config()
        # initialize action space
        self.__action_space = ActionSpace()
        self.__model = None

        if os.path.exists(self.__args["model_path"]):
            # load the model from the specified path
            self.__model = ResNet(
                self.__model_params['num_blocks'],
                self.__model_params['num_channels'],
                self.__action_space.size, self.__device)
            try:
                self.__model.load_state_dict(torch.load(
                    self.__args["model_path"], map_location=self.__device))
                self.__model.eval()
            except FileNotFoundError:
                self.__model = None
                print("Model corrupted or has wrong structure")
        self.__mcts = MCTS(self.__args, self.__action_space,
                           Cache(self.__args["cache_size"]), self.__model)

        self.__default_go_args = {
            "ponder_move": None,
            "wtime": 0,
            "btime": 0,
            "winc": 0,
            "binc": 0,
            "depth": self.__infinity,
            "nodes": self.__infinity,
            "movestogo": 0,
            "movetime": 0,
            "infinite": False
        }
        self.__go_args = self.__default_go_args.copy()
        self.__nodes_searched = 0
        self.__depth_reached = 0
        self.__search_start_time = None
        # engine is ready

    def __safe_input(self) -> str:
        """Read input, handle "quit" command, KeyboardInterrupt and EOFError"""

        try:
            input_str = ""
            while not self.__input_interrupt.is_set():
                input_chr = sys.stdin.read(1)
                if input_chr == "\n":
                    break
                input_str += input_chr
            if input_str == "quit".casefold():
                raise KeyboardInterrupt
            return input_str
        except (KeyboardInterrupt, EOFError):
            # exit the engine
            self.__search_interrupt.set()
            raise SystemExit

    def __wait(self):
        sleep(0.00001)

    def __print_search_info(self, time_now: int) -> None:
        """Print the info about the search"""

        search_time = time_now - self.__search_start_time
        nps = int(
            self.__nodes_searched / search_time)
        print("info depth", self.__depth_reached, "nodes", self.__nodes_searched, "time", search_time * 1000,
              "score cp", self.__get_cp_score(), "nps", nps, "pv", self.__mcts.best_line())

    def __calculate_search_time(self) -> int:
        """Calculate the time avaible for the search"""

        if self.__go_args["movetime"]:
            # return the time to search
            return self.__go_args["movetime"]
        # get the time left for the current player
        time_left = self.__go_args["wtime"] if self.__mcts.root.state.turn else self.__go_args["btime"]
        # get the increment for the current player
        increment = self.__go_args["winc"] if self.__mcts.root.state.turn else self.__go_args["binc"]
        # return the time to search
        if time_left <= 0:
            return 0
        return math.log2(time_left) + increment

    def __get_cp_score(self) -> int:
        """Return an approximation of the cp score"""

        # Alphazero style engines do not evaluate positions using cp scores,
        # probability of victory in range {-1, 1} is used instead.
        # We can approximate a cp score, assuming a player would be
        # clearly winning having an advantage of an extra queen.
        # In Deepmind's Alphazero paper, a queen value of 9.5 is stated.
        # Approximation will be made using that value rescaled to centipawns.
        return int(self.__mcts.evaluation * 9.5 * 100)

    def __perform_search(self) -> None:
        """
        Search for the best move until the stop event is set, 
        put best and ponder moves to parent class queue
        """

        # copy max_depth and nodes from go_args
        max_depth = self.__go_args["depth"]
        max_nodes = self.__go_args["nodes"]

        # search until the stop event is set
        while self.__nodes_searched < max_nodes and not self.__search_interrupt.is_set():
            self.__mcts.search_step(max_depth)
            self.__nodes_searched += 1
        self.__depth_reached = self.__mcts.depth

    def __perform_ponder_search(self) -> None:
        """Search for the best move until the stop event is set"""

        # search until the stop event is set
        while not self.__search_interrupt.is_set():
            self.__mcts.search_step(self.__infinity)
            self.__nodes_searched += 1
        self.__depth_reached = self.__mcts.depth

    def __ponder(self, move) -> None:
        self.__mcts.restrict_root([move])

        # start search_thread
        ponder_thread = threading.Thread(
            target=self.__perform_ponder_search, daemon=True)
        ponder_thread.start()

        input_str = ""
        # continue ponder search until the engine is interrupted
        while True:
            # check input buffer
            input_str = self.__safe_input()
            if input_str == "ponderhit" or input_str.startswith("position"):
                self.__search_interrupt.set()
                break

        # stop the search
        ponder_thread.join()
        if input_str == "ponderhit":
            # ponderhit received, continue pondering
            self.__mcts.select_child_as_new_root(move)
        else:
            fen = "".join(input_str.split(maxsplit=1)[1:])
            self.__mcts.root_from_fen(fen)

        # switch to normal search
        self.__init_search()

    def __handle_timer(self) -> None:
        """Handle the timer of timed search"""

        # calculate the time to search
        search_time = self.__calculate_search_time()
        # wait for the search to finish
        while perf_counter() - self.__search_start_time < search_time:
            self.__wait()
        self.__search_interrupt.set()

    def __init_search(self) -> None:
        """Manage the engine search threads and perform the search"""

        # start tracking the search time
        self.__search_start_time = perf_counter()

        # start the search in single thread
        thread = threading.Thread(
            target=self.__perform_search, daemon=True)
        thread.start()

        # wait for the search to finish
        if self.__go_args["infinite"]:
            # wait for the stop command
            while self.__safe_input() != "stop":
                pass
            self.__search_interrupt.set()
        elif self.__go_args["nodes"] == self.__infinity or self.__go_args["movetime"] != 0:
            # wait for the search to finish by max nodes or stop command
            timer_thread = threading.Thread(
                target=self.__handle_timer, daemon=True)
            timer_thread.start()

            # await quit or search finish
            quit_thread = threading.Thread(
                target=self.__safe_input, daemon=True)
            quit_thread.start()

            while thread.is_alive():
                # wait for thread to finish or SystemExit except to be raised from subthread
                self.__wait()
                if not quit_thread.is_alive():
                    sys.exit(0)
                pass
            self.__input_interrupt.set()
            timer_thread.join()
        else:
            pass

        # wait for the search to stop
        thread.join()
        search_end = perf_counter()
        self.__search_interrupt.clear()
        self.__input_interrupt.clear()

        # retreive the search result, output bestmove and ponder if available
        bestmove, pondermove = self.__mcts.best_move()

        # end the search and update time in go_args
        self.__print_search_info(search_end)
        search_time = (search_end - self.__search_start_time) * 1000
        player = self.__mcts.root.state.turn
        if player:
            self.__go_args["wtime"] -= search_time
        else:
            self.__go_args["btime"] -= search_time

        if pondermove is None:
            print("bestmove", bestmove)
        else:
            print("bestmove", bestmove, "ponder", pondermove)

        # reset engine search stats
        self.__nodes_searched = 0
        self.__depth_reached = 0

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
                self.__go_args["movetime"] = int(go_command_args[i+1]) / 1000

            elif go_command_args[i] == "infinite":
                self.__go_args["infinite"] = True

        if self.__go_args["ponder_move"]:
            self.__ponder(self.__go_args["ponder_move"])
        else:
            self.__init_search()

    def __uci_position(self, position_command: str) -> None:
        """Parse the position command and update the root node"""

        fen = "".join(position_command.split(maxsplit=1)[1:])
        # position can be either "startpos" or fen string
        if fen[0] == "startpos":
            fen[0] = chess.STARTING_FEN
        elif fen == "startpos":
            fen = chess.STARTING_FEN

        # if the mcts has a root node, that means the engine is in the middle of the game,
        # so we should select a child node as the new root, using the opponent's move
        if self.__mcts.root:
            opponent_move = fen[-1]
            self.__mcts.select_child_as_new_root(opponent_move)
        else:
            self.__mcts.root_from_fen(fen)

    def __uci(self) -> None:
        """Handle uci command"""

        # acknowledge uci interface
        print("id name", self.__info["name"])
        print("id author", self.__info["author"])
        print("uciok")

        # start initializing the engine on a different thread
        init_thread = threading.Thread(target=self.__ready, daemon=True)
        init_thread.start()

        while True:
            command = self.__safe_input()
            if command == "isready":
                while init_thread.is_alive():
                    pass
                print("readyok")

            elif command == "debug":
                if self.__debug:
                    self.__debug = False
                else:
                    self.__debug = True

            elif command == "ucinewgame":
                self.__mcts.reset()

            elif command.startswith("position"):
                self.__uci_position(command)

            elif command.startswith("go"):
                self.__uci_go(command)

    def __read_init(self) -> None:
        """Read the init command"""

        while True:
            command = self.__safe_input()
            if command == "uci":
                self.__uci()


if __name__ == "__main__":
    engine = Engine()
