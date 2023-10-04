import math
import multiprocessing
import queue
import sys
import threading
from contextlib import contextmanager
from time import perf_counter

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
            "num_blocks": 16,
            "num_channels": 256
        }
        self.__readers: int = 0
        self.__writers: int = 0
        self.__read_ready = threading.Condition(threading.Lock())

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
        if "multi_threaded" not in self.__args:
            self.__args["multi_threaded"] = False

    def __acquire_write(self) -> None:
        """Acquire write lock"""

        self.__read_ready.acquire()
        try:
            while self.__readers > 0 or self.__writers > 0:
                self.__read_ready.wait()
            self.__writers += 1
        finally:
            self.__read_ready.release()

    def __release_write(self) -> None:
        """Release write lock"""

        self.__read_ready.acquire()
        try:
            self.__writers -= 1
            self.__read_ready.notify_all()
        finally:
            self.__read_ready.release()

    def __acquire_read(self) -> None:
        """Acquire read lock"""

        self.__read_ready.acquire()
        try:
            self.__readers += 1
            while self.__writers > 0:
                self.__read_ready.wait()
        finally:
            self.__read_ready.release()

    def __release_read(self) -> None:
        """Release read lock"""

        self.__read_ready.acquire()
        try:
            self.__readers -= 1
            if self.__readers == 0:
                self.__read_ready.notify_all()
        finally:
            self.__read_ready.release()

    @contextmanager
    def __read(self) -> None:
        """Read context manager"""

        self.__acquire_read()
        yield
        self.__release_read()

    @contextmanager
    def __write(self) -> None:
        """Write context manager"""

        self.__acquire_write()
        yield
        self.__release_write()

    def __ready(self) -> None:
        """Initialize the engine"""

        # read the config file
        self.__read_config()
        # check if the config file contains all the required parameters
        self.__check_config()
        # initialize action space
        self.__action_space = ActionSpace()
        # load the model from the specified path
        self.__model = ResNet(
            self.__model_params['num_blocks'], self.__model_params['num_channels'], self.__action_space.size, self.__device)

        try:
            self.__model.load_state_dict(torch.load(
                self.__args["model_path"], map_location=self.__device))
            self.__model.eval()
            self.__mcts = MCTS(self.__args, self.__action_space, Cache(
                self.__args["cache_size"]))
        except FileNotFoundError:
            print("Model not found")

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

        self.__num_threads = multiprocessing.cpu_count()
        self.__search_threads = []
        self.__search_interrupt = threading.Event()
        self.__search_result = queue.Queue()
        self.__nodes_searched = 0
        self.__depth_reached = 0
        self.__info_write_lock = threading.Lock()

        # engine is ready
        print("readyok")

    def __safe_input(self) -> str:
        """Read input, handle "quit" command, KeyboardInterrupt and EOFError"""

        try:
            input_str = input()
            if input_str == "quit".casefold():
                raise KeyboardInterrupt
            return input_str
        except (KeyboardInterrupt, EOFError):
            # stop the current search
            if self.__search_threads:
                self.__search_interrupt.set()
                for thread in self.__search_threads:
                    if thread.is_alive():
                        thread.join()
            # exit the engine
            sys.exit(0)

    def __print_info(self, info_tic, info_toc):
        """Print the info about the search"""

        nps = int(
            self.__nodes_searched / (info_toc - info_tic))
        print("info depth", self.__depth_reached, "nodes", self.__nodes_searched, "time", info_toc - info_tic,
              "score cp", self.__get_cp_score(self.__mcts.evaluation), "nps", nps, "pv", self.__mcts.best_line())

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
        return math.log2(time_left) + increment

    def __get_cp_score(self, score: float) -> int:
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
        while self.__nodes_searched < (max_nodes - self.__num_threads) and not self.__search_interrupt.is_set():
            depth = self.__mcts.search_step(max_depth)
            # update search depth
            if self.__depth_reached > self.__depth_reached:
                self.__depth_reached = depth

    def __start_search_threads(self) -> None:
        if self.__args["multi_threaded"]:
            # create search threads
            self.__search_threads = []
            for _ in range(self.__num_threads):
                thread = threading.Thread(target=self.__perform_search)
                self.__search_threads.append(thread)
            # start the search threads
            for thread in self.__search_threads:
                thread.start()

    def __ponder(self, fen_move) -> None:
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
                    self.__mcts.select_child_as_new_root(move)
                    break
                elif input_str.startswith("position"):
                    fen = "".join(input_str.split(maxsplit=1)[1:])
                    self.__mcts.root_from_fen(fen)
                    break
            else:
                self.__mcts.search_step(max_depth)
        self.__init_search()

    def __init_search(self) -> None:
        """Manage the engine search threads and perform the search"""

        # start tracking the search time
        search_start = perf_counter()

        # check if the engine is already searching
        if self.__search_interrupt.is_set():
            # wait for the search to stop
            for thread in self.__search_threads:
                if thread.is_alive():
                    thread.join()
            # clear the search interrupt event
            self.__search_interrupt.clear()

        # start the search in multithreaded mode
        self.__start_search_threads()

        # wait for the search to finish
        if self.__go_args["infinte"]:
            # wait for the stop command
            while self.__safe_input() != "stop":
                pass
            self.__search_interrupt.set()
        else:
            # calculate the time to search
            search_time = self.__calculate_search_time()
            # wait for the search to finish
            while perf_counter() - search_start < search_time:
                pass
            self.__search_interrupt.set()

        for thread in self.__search_threads:
            if thread.is_alive():
                thread.join()
        self.__search_interrupt.clear()

        # retreive the search result, output bestmove and ponder if available
        search_result = self.__search_result.get()
        if search_result[1] is None:
            print("bestmove", search_result[0])
        else:
            print("bestmove", search_result[0], "ponder", search_result[1])

        # end the search and update time in go_args
        search_end = perf_counter()
        search_time = search_end - search_start
        player = self.__mcts.root.state.turn
        if player:
            self.__go_args["wtime"] -= search_time
        else:
            self.__go_args["btime"] -= search_time

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
            self.__ponder(self.__go_args["ponder_move"])
        else:
            self.__init_search()

    def __uci_position(self, position_command: str) -> None:
        """Parse the position command and update the root node"""

        fen = "".join(position_command.split(maxsplit=1)[1:])
        # position can be either "startpos" or fen string
        if fen[0] == "startpos":
            fen[0] = chess.STARTING_FEN

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

        while True:
            command = self.__safe_input()
            if command == "isready":
                self.__ready()
                print("readyok")

            elif command == "ucinewgame":
                self.__mcts.reset()

            elif command.startswith("position"):
                self.__uci_position(command)

            elif command.startswith("go"):
                best_move, ponder = self.__uci_go(command)
                print("bestmove", best_move, "ponder", ponder)

    def __read_init(self) -> None:
        """Read the init command"""

        while True:
            command = self.__safe_input()
            if command == "uci":
                self.__uci()


if __name__ == "__main__":
    engine = Engine()
