from queue import Queue
import math
import os
import sys
import threading
from time import perf_counter, sleep

import chess
import torch
import torch_directml
import yaml

from actionspace.actionspace import ActionSpace
from cache.cache_read_priority import Cache
from mcts.mcts import MCTS
from resnet.resnet import ResNet


class Engine():
    def __init__(self, device=torch.device("cpu")) -> None:
        self.__args = None
        self.__device = device
        self.__info = {
            "name": "RoboHub",
            "author": "Jakub Janicki"
        }
        self.__debug = False
        self.__infinity = sys.maxsize ** 10
        self.__input_interrupt = threading.Event()
        self.__search_interrupt = threading.Event()
        self.__ponder_stoppers = []
        self.__search_stoppers = []
        self.__autoponder = False
        self.__autopush = False
        self.__input_queue = Queue()
        self.__input_thread = threading.Thread(
            target=self.__safe_input, daemon=True)
        self.__input_thread.start()

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
            self.__args["cache_size"] = 1024

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
            try:
                self.__model_dict = torch.load(
                    self.__args["model_path"])
                self.__model = ResNet(
                    num_blocks=self.__model_dict["num_blocks"],
                    num_features=self.__model_dict["num_features"],
                    num_input_features=self.__model_dict["num_input_features"],
                    squeeze_and_excitation=self.__model_dict["squeeze_and_excitation"]
                )
                self.__model.load_state_dict(
                    self.__model_dict["model_state_dict"])

                self.__model.to(self.__device)
                self.__model.eval()
            except Exception as e:
                self.__model = None
                if str(e) != '':
                    print(e)
                    print("info Incompatible model")
                else:
                    print("Model corrupted or has wrong structure")
        else: # model not found
            print("info Model not found")

        self.__mcts = MCTS(
            self.__args["c_puct"],
            self.__action_space,
            Cache(self.__args["cache_size"]),
            self.__model
        )

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

        # wait for input unless the input_interrupt is set
        while True:
            try:
                input_str = input()
            except (KeyboardInterrupt, EOFError):
                # exit the engine
                self.__search_interrupt.set()
                self.__input_interrupt.set()
                print("info Keyboard interrupt")
                os._exit(0)
            if input_str == "quit".casefold():
                os._exit(0)
            self.__input_queue.put(input_str)
            if self.__input_interrupt.is_set():
                break

    def __wait(self):
        sleep(0.0001)

    def __print_search_info(self, time_now: int) -> None:
        """Print the info about the search"""

        search_time = time_now - self.__search_start_time
        nps = int(
            self.__nodes_searched / search_time)
        best_line = self.__mcts.best_line()
        best_line_str = ""
        for move in best_line:
            best_line_str += move + " "
        print("info depth", self.__depth_reached, "nodes", self.__nodes_searched, "time", search_time * 1000,
              "score cp", self.__get_cp_score(), "nps", nps, "pv", best_line_str)

    def __calculate_search_time(self) -> int:
        """Calculate the time avaible for the search"""

        if self.__go_args["movetime"]:
            # return the time to search
            return self.__go_args["movetime"]
        # get the time left for the current player
        time_left = self.__go_args["wtime"] if self.__mcts.root.turn else self.__go_args["btime"]
        # get the increment for the current player
        increment = self.__go_args["winc"] if self.__mcts.root.turn else self.__go_args["binc"]
        # return the time to search
        if time_left <= 0:
            return 0
        return math.log2(time_left) + increment

    def __get_cp_score(self) -> int:
        """Return an approximation of the cp score"""

        # Alphazero style engines do not evaluate positions using cp scores,
        # probability of victory in range {-1, 1} is used instead.
        # We can use a recalibrated formula developed for LeelaChessZero
        # to approximate the cp score from the probability of victory:
        # cp = 111.714640912 * tan(1.5620688421 * Q), where Q is the probability of victory.
        # source: https://github.com/LeelaChessZero/lc0/pull/841
        return int(111.714640912 * math.tan(1.5620688421 * self.__mcts.evaluation))

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
        self.__mcts.restrict_root([chess.Move.from_uci(move)])

        # start search_thread
        ponder_thread = threading.Thread(
            target=self.__perform_ponder_search, daemon=True)
        ponder_thread.start()

        input_str = ""
        # continue ponder search until the engine is interrupted
        while True:
            # check input buffer
            input_str = self.__input_queue.get()
            # check if the input is a ponder stopper
            if input_str in self.__ponder_stoppers:
                self.__search_interrupt.set()
                break
            # check if command can be split
            if ' ' not in input_str:
                continue
            command = input_str.split()
            if command[0] in self.__ponder_stoppers:
                self.__search_interrupt.set()
                if command[-1].isdigit():
                    self.__go_args["movetime"] = float(command[-1])
                break

        # stop the search
        ponder_thread.join()
        if input_str == "ponderhit":
            # ponderhit received, continue pondering
            self.__mcts.select_child_as_root(move)
        elif input_str.startswith("position"):
            fen = "".join(input_str.split(maxsplit=1)[1:])
            self.__mcts.root_from_fen(fen)
        elif input_str.startswith("push"):
            move = input_str.split()[1]
            self.__push(move)
        else:
            print("Unknown command:", input_str)
            raise SystemExit

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
        search_thread = threading.Thread(
            target=self.__perform_search, daemon=True)
        search_thread.start()

        # wait for the search to finish
        if self.__go_args["infinite"]:
            # wait for the stop command
            input_str = ''
            while input_str != "stop":
                self.__input_queue.get()
            self.__search_interrupt.set()

        elif self.__go_args["nodes"] == self.__infinity or self.__go_args["movetime"] != 0:
            # wait for the search to finish by max nodes or stop command
            timer_thread = threading.Thread(
                target=self.__handle_timer, daemon=True)
            timer_thread.start()

            while search_thread.is_alive():
                # wait search finish or subthread except
                try:
                    input_str = self.__input_queue.get_nowait()
                except:
                    self.__wait()
                    continue
                if input_str in self.__search_stoppers:
                    self.__search_interrupt.set()
                    break
            timer_thread.join()
        else:
            pass

        # wait for the search to stop
        search_thread.join()
        search_end = perf_counter()
        self.__search_interrupt.clear()
        self.__input_interrupt.clear()

        # retreive the search result, output bestmove and ponder if available
        bestmove, pondermove = self.__mcts.best_move()

        # end the search and update time in go_args
        self.__print_search_info(search_end)
        search_time = (search_end - self.__search_start_time) * 1000
        player = self.__mcts.root.turn
        if player:
            self.__go_args["wtime"] -= search_time
        else:
            self.__go_args["btime"] -= search_time

        if pondermove is None:
            print("bestmove", bestmove.uci())
        else:
            print("bestmove", bestmove.uci(), "ponder", pondermove.uci())

        # reset engine search stats
        self.__nodes_searched = 0
        self.__depth_reached = 0

        # if autopush is enabled, push the bestmove to the root node
        if self.__autopush:
            print(f"info push {bestmove.uci()}")
            self.__push(bestmove.uci())

        # if autoponder is enabled, start pondering
        if self.__autoponder:
            print(f"info ponder {pondermove.uci()}")
            self.__ponder(pondermove.uci())

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
        else:
            try:
                chess.Board(fen)
            except:
                print("info Invalid fen string")
                return

        # if the mcts has a root node, that means the engine is in the middle of the game,
        # so we should select a child node as the new root, using the opponent's move
        if self.__mcts.root:
            opponent_move = fen[-1]
            self.__mcts.select_child_as_root(opponent_move)
        else:
            self.__mcts.root_from_fen(fen)

    def __push(self, move: str) -> None:
        """Push a move to the root node"""

        self.__mcts.select_child_as_root(chess.Move.from_uci(move))

    def __uci(self) -> None:
        """Handle uci command"""

        # acknowledge uci interface
        print("id name", self.__info["name"])
        print("id author", self.__info["author"])
        print("uciok")

        # set ponder stoppers
        self.__ponder_stoppers = ["position", "ponderhit"]
        # set search stoppers
        self.__search_stoppers = ["stop"]

        # start initializing the engine on a different thread
        init_thread = threading.Thread(target=self.__ready, daemon=True)
        init_thread.start()

        while True:
            command = self.__input_queue.get()
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

    def __enable_autopush(self) -> None:
        """Enable autopush"""

        self.__autopush = True
        print("info autopush enabled")

    def __disable_autopush(self) -> None:
        """Disable autopush"""

        self.__autopush = False
        print("info autopush disabled")

    def __enable_autoponder(self) -> None:
        """Enable autoponder"""

        self.__autoponder = True
        print("info autoponder enabled")
        if not self.__autopush:
            self.__enable_autopush()

    def __disable_autoponder(self) -> None:
        """Disable autoponder"""

        self.__autoponder = False
        print("info autoponder disabled")

    def __ocb(self) -> None:
        """Handle ocb command"""

        # acknowledge deepblue interface
        print("id name", self.__info["name"])
        print("id author", self.__info["author"])
        print("ocbok")

        # set ponder stoppers
        self.__ponder_stoppers = ["push", "stop"]
        # set search stoppers
        self.__search_stoppers = ["stop"]

        # set options
        self.__autoponder = True
        self.__autopush = True

        # print options
        print("option name autoponder type check default true")
        print("option name autopush type check default true")
        print("info autoponder requires autopush")

        # start initializing the engine on a different thread
        init_thread = threading.Thread(target=self.__ready, daemon=True)
        init_thread.start()

        while True:
            command = self.__input_queue.get()
            if command == "isready":
                while init_thread.is_alive():
                    pass
                print("readyok")

            elif command == "new":
                self.__mcts.reset()
                self.__uci_position("position startpos")

            elif command == "reset":
                self.__mcts.reset()

            elif command.startswith("position"):
                self.__uci_position(command)

            elif command.startswith("go"):
                # if command is like "go num"
                if len(command.split(sep=' ')) == 2:
                    command = "go movetime " + command.split()[1] + "000"
                self.__uci_go(command)

            elif command.startswith("push"):
                self.__push(command.split()[1])

            elif command.startswith("setoption"):
                if command.split()[2] == "autoponder":
                    if command.split()[4] == "true":
                        self.__enable_autoponder()
                    else:
                        self.__disable_autoponder()
                elif command.split()[2] == "autopush":
                    if command.split()[4] == "true":
                        self.__enable_autopush()
                    else:
                        self.__disable_autopush()

    def __read_init(self) -> None:
        """Read the init command"""

        while True:
            command = self.__input_queue.get()
            if command == "uci":
                self.__uci()
            if command == "ocb":
                self.__ocb()


if __name__ == "__main__":
    dml = torch_directml.device()
    engine = Engine(dml)
