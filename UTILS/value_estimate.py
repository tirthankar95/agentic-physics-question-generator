import json
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class PhyGraph:
    def load_model(self) -> bool:
        """
        returns True and loads the model if it exists, else returns False
        """
        dirname = self.cfg["Input"]["TopicsDir"]
        filename = self.cfg["Topics"][self.choice]["InputFile"].split(".")[0]
        model_path = Path(f"{dirname}/{filename}")
        if not model_path.exists():
            return False
        with open(str(model_path), "r") as f:
            self.state_value = json.load(f)
        return True

    def save_model(self):
        dirname = self.cfg["Input"]["TopicsDir"]
        filename = self.cfg["Topics"][self.choice]["InputFile"].split(".")[0]
        model_path = Path(f"{dirname}/{filename}")
        with open(str(model_path), "w") as f:
            json.dump(self.state_value, f, indent=5)

    def __init__(
        self,
        node_names,
        graph,
        cfg: DictConfig,
        choice: int,
        epsilon_decay: float = 0.99,
        lower_epsilon: float = 0.5,
    ):
        self.cfg, self.choice = cfg, choice
        self.graph = graph
        self.state_value = {}
        if not self.load_model():
            for idx, x in enumerate(node_names):
                self.state_value[str(idx)] = (0.0, x)  # (value, name)
            self.state_value["epsilon"] = 1.0
            self.state_value["epsilon_decay"] = epsilon_decay
            self.state_value["lower_epsilon"] = lower_epsilon
        self.eqn_path = []

    def predict(self, id: int) -> int:
        mx_value, mx_id = None, None
        allEdges = set()
        if len(self.eqn_path) == 0:
            self.eqn_path.append(id)
        for eqn_id, _ in self.graph[id]:
            value, __ = self.state_value[str(eqn_id)]
            if mx_value == None or mx_value < value:
                mx_value, mx_id = value, eqn_id
            allEdges.add(eqn_id)

        epsilon, actN = self.state_value["epsilon"], len(allEdges)
        action_dict = {action: epsilon / actN for action in allEdges}
        action_dict[mx_id] += 1 - epsilon
        child_id = np.random.choice(
            list(action_dict.keys()), p=list(action_dict.values())
        )
        logger.debug(
            f"Parent: {self.state_value[str(id)][1]}, "
            + f"Child: {self.state_value[str(child_id)][1]}, "
            + f"Value: {self.state_value[str(child_id)][0]}, "
            + f"Prob: {action_dict}"
        )
        self.eqn_path.append(child_id)
        return child_id

    def fit(self, reward: float):
        self.eqn_path = []
