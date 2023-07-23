import copy
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from cache import Cache
from chess import Chess
from mcts import MCTS
from resnet import ResNet


class AlphaZero():
    def __init__(self, args, game, device="cpu") -> None:
        self.__args = args
        self.__game: Chess = game
        self.__model = ResNet(
            self.__args['num_blocks'], self.__args['num_channels'], game.action_space.size, device)
        self.__optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args['lr'], amsgrad=True)
        self.__dataset = []
        self.__model_num = 1
        self.__loss_fn = nn.CrossEntropyLoss()
        self.__evaluation_cache = Cache()

    def step(self, probs):
        legal_moves = self.__game.get_legal_moves()
        action = self.__game.__action_space[np.argmax(probs)]
        if action not in legal_moves:
            return False
        self.__state = self.__state.get_next_state(action)
        return True

    def stochastic_step(self, probs):
        t = 1 if "pit_temp" not in self.__args else self.__args["pit_temp"]
        probs **= (1/t)
        probs = probs / np.sum(probs)
        legal_moves = self.get_legal_moves()
        action_id = np.random.choice(np.arange(len(probs)), p=probs)
        action = self.__action_space[action_id]
        if action not in legal_moves:
            raise ValueError("Illegal move")
        self.__state = self.__state.get_next_state(action)
    
    def pit(self, num_games, model1, model2, stochastic=False, verbose=False) -> float:
        p1 = MCTS(self.__args, self.__action_space, Cache(), model1, int(self.__args["pit_searches"]))
        p2 = MCTS(self.__args, self.__action_space, Cache(), model2, int(self.__args["pit_searches"]))
        players = {
            1: {
                "model": p1,
                "wins": 0,
                "draws": 0,
                "losses": 0
            },
            -1: {
                "model": p2,
                "wins": 0,
                "draws": 0,
                "losses": 0
            }
        }
        for _ in range(num_games):
            while not self.__state.is_terminal:
                if self.__state.player_turn == 1:
                    board, dist, value = players[1]["model"].search(self.__state)
                    self.stochastic_step(dist) if stochastic else self.step(dist)
                else:
                    board, dist, value = players[-1]["model"].search(self.__state)
                    self.stochastic_step(dist) if stochastic else self.step(dist)
            if verbose:   
                print("\033[H\033[J")
                print(self.__state.board)
            if self.__state.win == 1:
                players[1]["wins"] += 1
                players[-1]["losses"] += 1
            elif self.__state.win == -1:
                players[-1]["wins"] += 1
                players[1]["losses"] += 1
            else:
                players[1]["draws"] += 1
                players[-1]["draws"] += 1
            self.next_game()
            players[1], players[-1] = players[-1], players[1]
        wr = players[1]["wins"] / num_games
        if verbose:
            print(wr)
        return wr

    def execute_epoch(self, dataloader, model):
        for data in dataloader:
            state, policy, value = data
            state = state.to(self.model.device).unsqueeze(1).float()
            policy = policy.to(self.model.device).unsqueeze(1).float()
            value = value.to(self.model.device).float()
            policy_pred, value_pred = model(state)
            policy_pred = policy_pred.unsqueeze(0)
            policy = policy.squeeze(1).unsqueeze(0)
            value_pred = value_pred.squeeze(1)

            value_loss = self.loss_fn(value_pred, value)
            policy_loss = self.loss_fn(policy_pred, policy)
            loss = value_loss + policy_loss
            if loss.item() is np.nan:
                return None
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return model

    def saturate(self, dataloader):
        self.optimizer.param_groups[0]['lr'] = self.args['saturation_lr']
        new_model = copy.deepcopy(self.model)
        for i in trange(self.args['saturation_patience'], desc="Saturating"):
            new_model.train()
            for epoch in range(self.args['saturation_epochs']):
                temp = copy.deepcopy(new_model)
                temp = self.execute_epoch(dataloader, temp, self.args['saturation_lr'])
                if temp is not None:
                    new_model = temp
            new_model.eval()
            with torch.no_grad():
                winrate = self.game.pit(100, self.model, new_model)
            if winrate > 0.55:
                torch.save(new_model.state_dict(), f"model_{self.model_num}.pt")
                self.model = new_model
                i = 0
            self.model_num += 1
            with open("winratio.txt", "a") as f:
                f.write(f"saturation: {winrate}\n")

    def train(self, load_dataset=False, extend_dataset=True, only_last_model=False):
        self.optimizer.param_groups[0]['lr'] = self.args['lr']
        self.model.eval()
        # reset winratio file
        with open("winratio.txt", "w") as f:
            f.write("")
        if load_dataset:
            with open("dataset.pkl", "rb") as f:
                self.dataset = pickle.load(f)
        for episode in range(self.args['num_episodes']):
            self.model.eval()
            if extend_dataset:
                with torch.no_grad():
                    data = []
                    for _ in trange(self.args['num_games'], desc="Self Play"):
                        data.extend(self.game.self_play(self.model, self.evaluation_cache, True))
                    self.dataset.extend(data)
                with open("dataset.pkl", "wb") as f:
                    pickle.dump(self.dataset, f)
            elif self.dataset == []:
                with open("dataset.pkl", "rb") as f:
                    self.dataset = pickle.load(f)
            dataloader = DataLoader(
                self.dataset, batch_size=self.args['batch_size'], shuffle=True)
            new_model = copy.deepcopy(self.model)
            new_model.train()
            for epoch in range(self.args['num_epochs']):
                temp = copy.deepcopy(new_model)
                temp = self.execute_epoch(dataloader, temp)
                if temp is not None:
                    new_model = temp
            self.evaluation_cache.clear()
            # self.dataset = []

            new_model.eval()
            with torch.no_grad():
                winrate = self.game.pit(100, self.model, new_model, True)
            if winrate > 0.55:
                if only_last_model:
                    self.dataset = []
                # append winratio to file
                with open("winratio.txt", "a") as f:
                    f.write(f"{winrate}\n")
                torch.save(new_model.state_dict(), f"model_{self.model_num}.pt")
                self.model = new_model
            self.model_num += 1
        self.saturate(dataloader)