from UTILS.action_estimate import PhyGraph
from collections import defaultdict
from omegaconf import DictConfig
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GraphEquation:
    operators = ["+", "*", "=", "/", "^", "(", ")", "-", "exp("]

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

    def _parse(self, equation):
        elements_ = [
            var
            for var in equation.split(" ")
            if var not in self.operators and len(var.strip()) > 0
        ]
        elements = []
        for x in elements_:
            try:
                var = float(x)
            except:
                elements.append(x)
        return elements

    def getEquation(self):
        qid = np.random.randint(len(self.adj))
        threshold, eqn = 0.25, [qid]
        self.vis, unk = defaultdict(bool), defaultdict(bool)
        self.qu = [qid]
        self.vis[qid] = True
        path = []
        while len(self.qu):
            src = self.qu.pop(0)
            path.append(src)
            # Stop Condition
            if 0.5 + np.random.normal() >= threshold:
                edgeId = self.rl_obj.predict(src, unk, self.vis)
                edge = self.adj[src][edgeId] # (eqnId, edge) 'edge is a variable'
                if edge[1] in unk or edge[0] in self.vis:
                    '''
                    After multiple tries you cannot find a new unknown variable, or
                    the equantion has already been visited.
                    '''
                    break
                else:
                    unk[edge[1]], self.vis[edge[0]] = True, True
                    eqn.append(edge[0])
                    path.append(edge[1])
                    self.qu.append(edge[0])
                    
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
