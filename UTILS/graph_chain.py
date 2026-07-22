from UTILS.action_estimate import PhyGraph
from collections import defaultdict
from omegaconf import DictConfig
from typing import List 
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GraphEquation:
    operators = ["+", "*", "=", "/", "^", "(", ")", "-", \
                 "exp(", "pow(", "sin(", "cos(", "tan(", "tanh("]

    def __init__(self, equations, cfg: DictConfig, choice: int):
        self.equation_element = []
        self.cfg, self.choice = cfg, choice
        for equation in equations:
            self.equation_element.append(self._parse(equation))
        N = len(self.equation_element)
        self.adj = defaultdict(list)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                allEdges = [
                    (j, ch)
                    for ch in set(self.equation_element[i]).intersection(
                        set(self.equation_element[j])
                    )
                ]
                self.adj[i].extend(allEdges)
        self.rl_obj = PhyGraph(self.equation_element, self.adj, cfg, choice)

    def _parse(self, equation: str) -> List[str]:
        """Extracts variables from equations by removing operators, constants, etc.

        Args:
            equation(str): The string containing the equation.

        Returns:
            List of variables in an equation.
        """
        elements_ = [
            var
            for var in equation.split(" ")
            if var not in self.operators and len(var.strip()) > 0
        ]
        elements = []
        for x in elements_:
            try: # Remove constants.
                var = float(x)
            except:
                elements.append(x)
        return elements

    def getEquation(self):
        """Traverse the topic specific equation graph create known and unkwon variables
        for the N equations.
        """
        def get_start_node():
            n = len(self.adj)
            nodes = np.array([1e-5] * n)
            state_dict = self.rl_obj.action_value
            for k, v in state_dict.items():
                if isinstance(k, tuple):
                    nodes[k[0]] += v
            epsilon = self.rl_obj.action_value["epsilon"]
            best_node = int(np.argmax(nodes))
            probabilities = np.array([epsilon / n] * n)
            probabilities[best_node] += (1 - epsilon)
            return np.random.choice(range(n), p=probabilities)
            
        qid = get_start_node().item()
        threshold, eqn = 0.0, [qid]
        self.vis, unk = defaultdict(bool), defaultdict(bool)
        self.qu = [qid]
        self.vis[qid] = True
        path = []
        while len(self.qu):
            src = self.qu.pop(0)
            path.append(src)
            # Stop Condition
            if np.random.normal() >= threshold:
                edgeId = self.rl_obj.predict(src, unk, self.vis)
                eqnId, edge = self.adj[src][edgeId]
                if edge in unk or eqnId in self.vis:
                    '''
                    After multiple tries you cannot find a new unknown variable, or
                    the equation has already been visited.
                    '''
                    break
                else:
                    unk[edge], self.vis[eqnId] = True, True
                    eqn.append(eqnId)
                    path.append(edge)
                    self.qu.append(eqnId)
                    
        while True:
            ch = np.random.choice(self.equation_element[eqn[-1]])
            if ch.item() not in unk:
                break
        unk[ch.item()] = True
        path.append(ch.item())
        self.rl_obj.save_trajectory(path)
        
        assert len(unk) == len(eqn), f"Bad Equation: {unk} {eqn}"
        known = defaultdict(bool)
        for eId in eqn:
            for ch in self.equation_element[eId]:
                if ch not in unk:
                    known[ch] = True
        unk, known = [k for k in unk.keys()], [k for k in known.keys()]
        logging.debug(f"{unk}, {known}")
        return unk, known, eqn
