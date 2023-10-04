from contextlib import contextmanager
from math import sqrt
from threading import Condition, Lock

import chess
import numpy as np

from actionspace import ActionSpace
from cache_read_priority import Cache
from resnet import ResNet
from state import State


class Node():
    def __init__(self, args, state, action_taken=None, policy_value=0) -> None:
        self.__args = args
        self.__children: list = []
        self.__state: State = state
        self.__action_taken: chess.Move = action_taken
        self.__value: float = 0
        self.__policy_value: float = policy_value
        self.__visit_count: int = 0
        self.__read_ready = Condition(Lock())
        self.__readers: int = 0
        self.__writers: int = 0

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
    def read(self) -> None:
        """Read context manager"""

        self.__acquire_read()
        yield
        self.__release_read()

    @contextmanager
    def write(self) -> None:
        """Write context manager"""

        self.__acquire_write()
        yield
        self.__release_write()

    def expand(self, policy, actions, nodes_dict) -> None:
        """
        Expand the node with the given policy and actions. 
        Add new nodes to nodes_dict.
        """

        with self.write():
            for prob, action in zip(policy, actions):
                next_state = self.__state.next_state(action)

                # to narrow the search space, connect branches that lead to the same state
                fen = next_state.fen
                if fen in nodes_dict:
                    self.__children.append(nodes_dict[fen])
                else:
                    new_node = Node(
                        self.__args, next_state, action_taken=action, parent=self, policy_value=prob)
                    nodes_dict[fen] = new_node
                    self.__children.append(new_node)

    def select(self) -> "Node":
        """Select the child node with the highest UCB value, """

        # select child with best ucb value
        with self.read():
            best_child = max(self.__children, key=lambda child: child.ucb)
        return best_child

    def backpropagate(self, value: float, select_stack: list("Node")) -> None:
        """Backpropagate the value of the node to the root"""

        with self.write():
            self.__visit_count += 1
            self.__value += value
        if select_stack:
            parent = select_stack.pop()
            parent.backpropagate(-value, select_stack)

    def simulate(self) -> int:
        """Simulate a game from the current state"""

        # since a player cannot checkmate himself,
        # only the player who has just moved can be the winner
        # therefore, distinction who won is not necessary
        # state never changes after initialization so there is no need to lock
        if self.__state.is_checkmate:
            return 1
        return 0

    def clear_children(self) -> None:
        """Clear the children of the node"""

        with self.write():
            self.__children.clear()

    def remove_child(self, child: "Node") -> None:
        """Remove a child from the node"""

        with self.write():
            self.__children.remove(child)

    def ucb(self, parent: "Node") -> float:
        """Calculate the UCB value of the node"""

        with self.read():
            q = self.__value / (1 + self.__visit_count)
            if parent is None:
                u = 0
            else:
                u = self.__policy_value * self.__args["c_puct"] * \
                    sqrt(parent.visit_count) / (1 + self.__visit_count)
        return q + u

    @property
    def action_taken(self) -> chess.Move:
        """Return the action taken to reach the node"""

        return self.__action_taken

    @property
    def visit_count(self) -> int:
        """Return the number of visits of the node"""

        return self.__visit_count

    @property
    def value(self) -> float:
        """Return the value of the node"""

        return self.__value

    @property
    def state(self) -> State:
        """Return the state of the node"""

        return self.__state

    @property
    def children(self) -> list:
        """Return the children of the node"""

        return self.__children


class MCTS():
    """Monte Carlo Tree Search with optional ResNet model and evaluation cache"""

    def __init__(self, args, action_space, cache, model=None) -> None:
        self.args = args
        self.__model: ResNet = model
        self.__root: Node = None
        self.__nodes: dict = dict()
        self.__action_space: ActionSpace = action_space
        self.__cache: Cache = cache
        self.__uniform_policy = np.ones(
            self.__action_space.size, dtype=np.float32) / self.__action_space.size
        self.__depth_reached = 0
        self.__read_ready = Condition(Lock())
        self.__readers: int = 0
        self.__writers: int = 0

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
    def read(self) -> None:
        """Read context manager"""

        self.__acquire_read()
        yield
        self.__release_read()

    @contextmanager
    def write(self) -> None:
        """Write context manager"""

        self.__acquire_write()
        yield
        self.__release_write()

    def calculate_policy(self, node: Node, legal_moves: list) -> np.ndarray:
        """Calculate the policy and value of a given node"""

        policy, policy_value = self.__model(
            self.__model.get_tensor_state(node.state.encoded))
        policy = self.__model.get_policy(policy)
        policy_value = self.__model.get_value(policy_value)

        # calculate legal policy
        legal_ids = np.array(
            [self.__action_space.get_key(move) for move in legal_moves])
        legal_policy = np.zeros(self.__action_space.size)
        legal_policy[legal_ids] = policy[legal_ids]
        legal_policy = legal_policy[legal_policy != 0]
        legal_policy = legal_policy / np.sum(legal_policy)
        return legal_policy, policy_value

    def search_step(self, max_depth: int) -> None:
        """Perform a single tree search step"""

        # initialize node to root
        node = self.__root
        # select a leaf node

        depth_reached = 0
        for depth in range(max_depth):
            if not node.children:
                depth_reached = depth
                break
            node = node.select()
        value = node.simulate()
        legal_moves = node.state.legal_moves

        # expand the node if it is not terminal
        if not value:
            # use uniform policy if no model is provided
            if self.__model:
                fen = node.state.fen
                # check cache for policy, skip calculation if found
                legal_policy = self.__cache[fen]
                if legal_policy is None:
                    legal_policy, value = self.calculate_policy(
                        node, legal_moves)
                    # add policy cache
                    self.__cache[fen] = legal_policy
            else:
                # get uniform policy
                legal_policy = self.__uniform_policy
            node.expand(legal_policy, legal_moves, self.__nodes)
        node.backpropagate(value)

        # update depth reached
        with self.write():
            if depth_reached > self.__depth_reached:
                self.__depth_reached = depth_reached
        return depth_reached

    def get_dist(self) -> np.ndarray:
        """Return the distribution of visits of leaf nodes"""

        dist = np.zeros(self.__action_space.size)
        with self.__root.read():
            for child in self.__root.children:
                dist[self.__action_space.get_key(
                    child.action_taken)] = child.visit_count
        return dist / np.sum(dist)

    def initialize_root(self, reinitialize=False) -> None:
        """Expand root node if necessary"""

        if reinitialize:
            self.__root.clear_children()
        if self.__root.state.is_terminal or self.__root.children:
            return

        if self.__model:
            legal_moves = self.__root.state.legal_moves
            legal_policy, value = self.calculate_policy(
                self.__root, legal_moves)
            self.__root.expand(legal_policy, legal_moves, self.__nodes)
        else:
            self.__root.expand(self.__uniform_policy,
                               self.__root.state.legal_moves, self.__nodes)

    def set_root(self, state: State) -> None:
        """Set the root node to the given state"""

        with self.write():
            if state.fen in self.__nodes:
                self.__root = self.__nodes[state.fen]
            else:
                self.__root = Node(self.args, state)
                self.__nodes[state.fen] = self.__root
            # reset depth reached
            self.__depth_reached = 0

    def root_from_fen(self, fen: str) -> None:
        """Set the root node to the given fen"""

        board = chess.Board(fen)
        state = State(board)
        self.set_root(state)
        self.initialize_root()

    def restrict_root(self, actions: list[chess.Move]) -> None:
        """Restrict the root nodes to the given actions"""

        # get the child node corresponding to the action
        for child in self.__root.children:
            # remove children not specified in actions
            if child.action_taken not in actions:
                self.__root.remove_child(child)

    def select_child_as_new_root(self, action: chess.Move) -> None:
        """Select the child node as new root"""

        for child in self.__root.children:
            if child.action_taken == action:
                with self.write():
                    self.set_root(child.state)
                break

    def search(self, max_depth: int, num_searches=None) -> tuple:
        """
        Perform a search for a given number of searches up to given depth. 
        num_searches=args['num_searches'] by default
        """

        if not num_searches:
            num_searches = self.args["num_searches"]

        # perform search
        for _ in range(num_searches):
            self.search_step(max_depth)
        return (self.__root.state.encoded, self.get_dist(), -self.__root.value/self.__root.visit_count)

    def best_move(self) -> tuple:
        """Return the best move and the ponder move"""

        bestchild = max(self.__root.children,
                        key=lambda child: child.visit_count)
        ponderchild = max(bestchild.children,
                          key=lambda child: child.visit_count)
        return (bestchild.action_taken.uci(), ponderchild.action_taken.uci())

    def best_line(self) -> list:
        """Return the best line of the tree"""

        line = []
        node = self.__root
        while node.children:
            node = max(node.children, key=lambda child: child.visit_count)
            line.append(node.action_taken.uci())
        return line

    def reset(self) -> None:
        """Reset the tree and associated caches"""

        with self.write():
            self.__root = None
            self.__cache.clear()
            self.__nodes.clear()

    def reset_node_cache(self) -> None:
        """Reset the node cache"""

        with self.write():
            self.__cache.clear()

    @property
    def node_count(self) -> int:
        """Return the number of nodes in the tree"""

        with self.read():
            return len(self.__nodes)

    @property
    def root_fen(self) -> str:
        """Return the fen of the root node"""

        return self.__root.state.fen

    @property
    def evaluation(self) -> float:
        """Return the evaluation of the root node as a value between -1 and 1"""
        with self.read():
            return -self.__root.value/self.__root.visit_count

    @property
    def depth(self) -> int:
        """Return the maximum depth reached in the tree"""

        with self.read():
            return self.__depth_reached

    @property
    def root(self) -> bool:
        """Return the root node"""

        with self.read():
            return self.__root
