import json
import pytest
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from main import generate_question_variables

@pytest.fixture
def config(tmp_path):
    cfg = OmegaConf.load("config.yaml")
    cfg["Train"] = 1.0
    # Isolate model persistence to avoid cross-test contamination.
    topic_dir = tmp_path / "test"
    topic_dir.mkdir(parents=True, exist_ok=True)
    source = Path("TOPICS") / "test0.json"
    target = topic_dir / "test0.json"
    target.write_text(source.read_text())
    cfg["Input"]["TopicsDir"] = str(topic_dir)
    return cfg

def aggregate_node_values(state_dict):
    node = {}
    for k, v in state_dict.items():
        if isinstance(k, tuple):
            if k[0] not in node:
                node[k[0]] = 0
            node[k[0]] += v
    return node


def test_equation_set0(config, choice=4):
    combo_reward = {
        (0, 1): 1.0,
        (0, 2): 0.1,
        (1, 2): 0.5,
        (0, 1, 2): 0.3,
        (0,): 0.2,
        (1,): 0.2,
        (2,): 0.1
    }
    def inference_distribution(no_simulations: int):
        states = {k: 0 for k in combo_reward.keys()}
        reward_arr = []
        for _ in range(no_simulations):
            prompt, sol, units_p, rl_obj = generate_question_variables(data, config, choice)
            state = tuple(sorted(sol))
            states[state] += 1
            reward_arr.append(combo_reward[state])
        total = sum(states.values())
        return {k: v / total for k, v in states.items()}, reward_arr

    ITER, topic = 10000, config.Topics[choice]
    with open(f"{config['Input']['TopicsDir']}/{topic['InputFile']}") as file:
        data = json.load(file)
    all_prompts, all_solutions, all_units, all_rl_objs = True, True, True, True
    dist_before, reward_arr_before = inference_distribution(ITER)
    print(f"Distribution before training: {dist_before}")
    reward_arr = []
    
    for i in range(ITER):
        prompt, sol, units_p, rl_obj = generate_question_variables(data, config, choice)
        all_prompts &= prompt is not None
        all_solutions &= sol is not None
        all_units &= units_p is not None
        all_rl_objs &= rl_obj is not None
        # Train
        if config["Train"] != 0:
            state = tuple(sorted(sol))
            reward = combo_reward[state]
            reward_arr.append(reward)
            if 0 <= reward <= 1:
                rl_obj.fit(reward)
            rl_obj.save_model()
            
    dist_after, reward_arr_after = inference_distribution(ITER)
    print(f"Distribution after training: {dist_after}")
    assert all_prompts and all_solutions and all_units and all_rl_objs

    node_after = aggregate_node_values(rl_obj.action_value)
    print(f"V(s): {node_after}")

    # Check if reward is increasing over time.
    assert sum(reward_arr_after) / len(reward_arr_after) > sum(reward_arr_before) / len(reward_arr_before)

    # Check value functions of each equation nodes
    assert node_after[0] > node_after[2]
    
    # Check if the frequency of the worst solution decreased
    assert dist_after[(0, 2)] <= dist_before[(0, 2)]