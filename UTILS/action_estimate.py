import pickle
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class PhyGraph:
    def load_model(self, debug=False) -> bool:
        """
        returns True and loads the model if it exists, else returns False
        """
        dirname = self.cfg["Input"]["TopicsDir"]
        filename = self.cfg["Topics"][self.choice]["InputFile"].split(".")[0]
        model_path = Path(f"{dirname}/{filename}")
        if not model_path.exists():
            return False
        with open(str(model_path), "rb") as f:
            self.action_value = pickle.load(f)
        if debug:
            for k, v in self.action_value.items():
                logger.debug(f"{k}: {v}")
            logger.debug(f"--" * 20)
        return True

    def save_model(self):
        dirname = self.cfg["Input"]["TopicsDir"]
        filename = self.cfg["Topics"][self.choice]["InputFile"].split(".")[0]
        model_path = Path(f"{dirname}/{filename}")
        with open(str(model_path), "wb") as f:
            pickle.dump(self.action_value, f)

    def __init__(
        self,
        node_names,
        graph,
        cfg: DictConfig,
        choice: int,
        epsilon_decay: float = 0.99,
        lower_epsilon: float = 0.5,
        lr: float = 0.2,
        gamma: float = 0.9
    ):
        self.cfg, self.choice = cfg, choice
        self.graph = graph
        self.node_names = node_names
        self.action_value = {}
        if not self.load_model(debug=True):
            for idx, eqn in enumerate(node_names):
                for x in eqn:
                    self.action_value[(idx, x)] = 0.0
            self.action_value["epsilon"] = 1.0
            self.action_value["epsilon_decay"] = epsilon_decay
            self.action_value["lower_epsilon"] = lower_epsilon
            self.action_value["lr"] = lr
            self.action_value["gamma"] = gamma 
        self.eqn_path = []
        self.vis = {}

    def predict(self, id: int, unk: dict, vis: dict) -> int:
        mx_value, mx_indx = None, None
        edge_indx = 0
        for eqn_id, edge in self.graph[id]:
            value = self.action_value[(id, edge)]
            if mx_value == None or mx_value < value:
                mx_value, mx_indx = value, edge_indx
            edge_indx += 1

        epsilon, actN = self.action_value["epsilon"], len(self.graph[id])
        action_dict = [epsilon / actN] * actN
        action_dict[mx_indx] += (1 - epsilon)

        _iter = 5
        while _iter:
            child_id = np.random.choice(
                list(range(actN)), p=action_dict
            )
            _next_eq, _next_edge =self.graph[id][child_id] # (eqnId, edge)
            if _next_edge not in unk and _next_eq not in vis:
                break
            _iter -= 1
        logger.debug(
            f"Parent: {id}, "
            + f"Child: ({_next_eq}, {_next_edge}), "
            + f"Value: {self.action_value[(id, _next_edge)]}, "
            + f"Prob: {action_dict}"
        )
        return child_id

    def save_trajectory(self, trajectory):
        lb, n = 0, len(trajectory)
        while lb < n:
            eqn_id, edge = trajectory[lb], trajectory[lb + 1]
            self.eqn_path.append((eqn_id, edge))
            lb += 2

    def fit(self, reward: float):
        self.eqn_path = self.eqn_path[::-1]
        prev = 0.0
        gamma, lr = self.action_value["gamma"], self.action_value["lr"]
        for s, a in self.eqn_path:
            self.action_value[(s, a)] = (
                (1 - lr) * self.action_value[(s, a)] + 
                lr *(reward + gamma * prev)
            )
            prev = self.action_value[(s, a)]
            # Current reward, we want the reward to propagate back to the first action in the trajectory
            reward = 0.0 
        # Update epsilon
        self.action_value["epsilon"] = max(
            self.action_value["lower_epsilon"],
            self.action_value["epsilon"] * self.action_value["epsilon_decay"]
        )

        self.eqn_path = []
