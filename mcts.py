import numpy as np
import time

from actionspace import ActionSpace
from cache import Cache
from resnet import ResNet
from state import State


class Node():
    def __init__(self, args, state, action_taken=None, parent=None, policy_value=0) -> None:
        self.args = args
        self.parent: "Node" = parent
        self.children: list = []
        self.state: State = state
        self.action_taken: tuple = action_taken
        self.value: float = 0
        self.policy_value: float = policy_value
        self.visit_count: int = 0

    def expand(self, policy, actions) -> None:
        for prob, action in zip(policy, actions):
            next_state = self.state.get_next_state(action)
            self.children.append(
                Node(self.args, next_state, action_taken=action, parent=self, policy_value=prob))

    def select(self) -> "Node":
        return max(self.children, key=lambda child: child.ucb)

    def backpropagate(self, value) -> None:
        self.visit_count += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(-value)

    def simulate(self) -> int:
        if self.state.win != 0:
            return 1
        return 0

    @property
    def ucb(self) -> float:
        q = self.value / (1 + self.visit_count)
        if self.parent is None:
            u = 0
        else:
            u = self.policy_value * self.args["c_puct"] * \
                np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return q + u

class MCTS():
    def __init__(self, args, action_space, cache, model=None, num_searches=None) -> None:
        self.args = args
        self.model: ResNet = model
        self.root: Node = None
        self.action_space: ActionSpace = action_space
        self.cache: Cache = cache
        self.num_searches = num_searches if num_searches else args['num_searches']

    def search_step(self):
        node = self.root
        while node.children:
            node = node.select()
        value = node.simulate()
        if not node.state.is_terminal:
            legal_moves = node.state.get_legal_moves()
            if self.model:
                if node.state.byte_rep in self.cache:
                    legal_policy, value = self.cache[node.state.byte_rep]
                else:
                    policy, policy_value = self.model(self.model.get_tensor_board(node.state.encode()))
                    policy = self.model.get_policy(policy)
                    policy_value = self.model.get_value(policy_value)                        
                    legal_ids = np.array(
                        [self.action_space.get_key(move) for move in legal_moves])
                    legal_policy = np.zeros(self.action_space.size)
                    legal_policy[legal_ids] = policy[legal_ids]
                    legal_policy = legal_policy[legal_policy != 0]
                    legal_policy = legal_policy / np.sum(legal_policy)
                    self.cache[node.state.byte_rep] = (legal_policy, value)
            else:
                legal_policy = np.ones(len(legal_moves)) / len(legal_moves)
            node.expand(legal_policy, legal_moves)
        node.backpropagate(value)

    def get_dist(self):
        dist = np.zeros(self.action_space.size)
        for child in self.root.children:
            dist[self.action_space.get_key(child.action_taken)] = child.visit_count
        dist = dist / np.sum(dist)
        return dist

    def search(self, state):
        # perform a search for num_searches
        self.root = Node(self.args, state)
        for _ in range(self.num_searches):
            self.search_step(state)
        return (self.root.state.encode(), self.get_dist(), -self.root.value/self.root.visit_count)
    
    def timed_search(self, state, time_limit):
        # perform a search for time_limit ms
        self.root = Node(self.args, state)
        start = time.time() * 1000 # time in ms
        while time.time() - start < time_limit:
            self.search_step(state)
        return (self.root.state.encode(), self.get_dist(), -self.root.value/self.root.visit_count)

    def reset(self):
        self.root = None
        self.cache.clear()