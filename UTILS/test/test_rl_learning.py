import os 
import json
import pytest 
from omegaconf import OmegaConf
from unittest.mock import MagicMock
from main import generate_question_variables

print(f'PATH: {os.getcwd()}')

@pytest.fixture
def config():
    cfg = OmegaConf.load("config.yaml")
    cfg['Train'] = 1.0
    return cfg

def test_equation_set0(config, choice=4):
    combo_reward = {
        (0, 1): 1.0,
        (0, 2): 0.0,
        (1, 2): 0.5,
        (0, 1, 2): 0.0,
        (0,): 0.5,
        (1,): 0.5,
        (2,): 0.5
    }
    def inference_distribution(no_simulations: int):
        states = {k: 0 for k in combo_reward.keys()}
        for _ in range(no_simulations):
            prompt, sol, units_p, rl_obj = generate_question_variables(data, config, choice)
            state = tuple(sorted(sol))
            states[state] += 1
        total = sum(states.values())
        return {k: v / total for k, v in states.items()}
        
    ITER, topic = 10000, config.Topics[choice]
    with open(f"{config['Input']['TopicsDir']}/{topic['InputFile']}") as file:
        data = json.load(file)
    all_prompts, all_solutions, all_units, all_rl_objs = True, True, True, True
    dist_before = inference_distribution(ITER)
    file = open('traj0.txt', 'w')
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
            print(f'State: {state}, Reward: {reward}', file=file)
            if 0 <= reward <= 1:
                rl_obj.fit(reward)
            rl_obj.save_model()
    dist_after = inference_distribution(ITER)
    print(f"Distribution before training: {dist_before}")
    print(f"Distribution after training: {dist_after}")
    assert all_prompts and all_solutions and all_units and all_rl_objs
    
    # Check if the probability of the best solution increased
    assert dist_after[(0, 1)] > dist_before[(0, 1)]  
    
    # Check if the probability of the worst solution decreased
    assert dist_after[(0, 2)] < dist_before[(0, 2)]
    
    # Code cleanup: remove any saved model files after the test
    model_dir = config["Input"]["TopicsDir"]
    for file_name in os.listdir(model_dir):
        if file_name == "test0":
            os.remove(os.path.join(model_dir, file_name))
