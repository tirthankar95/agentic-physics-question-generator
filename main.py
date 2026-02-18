import sys
import json
import numpy as np
import pandas as pd
sys.path.insert(0, "LLM_CONFIG")
from LLM.llm import get_response
from UTILS.rag_agent import RagAgent
from colorama import Fore, Back, Style
from UTILS.utils import fix, load_env_vars
from UTILS.word_change import replace_words
from UTILS.graph_chain import GraphEquation
from omegaconf import DictConfig, OmegaConf
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

def generate_question_variables(data, cfg: DictConfig, choice: int):
    equations, units = data["equations"], {}
    obj = GraphEquation(equations, cfg, choice)
    unknown, known, eqn = obj.getEquation()
    logging.debug(f"known: {known}, unknown: {unknown}")
    problem = ""
    two_d = True if np.random.normal() >= 0 else False
    for element in known:
        name, v_range, unit, type, number_type = data["variable_names"][element]
        type = 0 if type == "S" else 1  # (0: scaler, 1: vector)
        var = fix(np.random.uniform(v_range[0], v_range[1]), number_type=number_type)
        if two_d and type == 1:
            theta = np.random.randint(0, 180)
            problem += f"{name} = {var} {unit} at an angle {theta} degrees with the horizontal, "
        else:
            problem += f"{name} = {var} {unit}, "
        units[unit] = True
    for idx, element in enumerate(unknown):
        name, v_range, unit, type, number_type = data["variable_names"][element]
        type = 0 if type == "S" else 1  # (0: scaler, 1: vector)
        if two_d and type == 1:
            problem += (
                f"horizontal component of {name} = unknown, "
                + f"vertical component of {name} = unknown, "
            )
        else:
            problem += f"{name} = unknown, "
    problem = problem[:-2] + "."
    logging.debug(problem)
    return problem, eqn, units


def beautify(filename, data):
    with open(f"{filename}", "w") as file:
        json.dump(data, file, indent=5)


def get_solution(sol, data):
    sol = set(sol)
    solution = ""
    for idx, sol_id in enumerate(sol):
        if idx == 0:
            solution = data["equations"][sol_id]
        else:
            solution += ", " + data["equations"][sol_id]
    return solution


def get_phyQ(cfg: DictConfig, choice: int):
    """
    1. Read topic configs.
    TOPICS/<topic_name>.json contains equations associated with each topic.
    2. Read llm configs.
    LLM_CONFIG/<config>.json which model to use & passwords to use.

    Generates a physics question TRIAL number of times and saves it to DATASET/<topic_name>.csv
    """
    load_env_vars()
    topic = cfg.Topics[choice]
    with open(f"{cfg['Input']['TopicsDir']}/{topic['InputFile']}") as file:
        data = json.load(file)
    with open(cfg['Input']['LLMConfig'], "r") as file:
        env = json.load(file)
    env_obj_rag = RagAgent(
        model_name=env["LLM_MODEL"]["INNER_MODEL"],
        collection_name=topic["CollectionName"],
    )
    try:
        df = pd.read_csv(f"{cfg['Output']['Dir']}/{topic['OutputFile']}")
    except Exception as e:
        print(f"{Style.BRIGHT}{Fore.RED}Creating file...{Style.RESET_ALL}")
        with open(f"{cfg['Output']['Dir']}/{topic['OutputFile']}", "w") as ofile:
            ofile.write("")

    for _ in range(topic['Trials']):
        # 0. Get set of equations to form the question.
        prompt, sol, units_p = generate_question_variables(data, cfg, choice)
        logging.info(f"\n[PROMPT] {prompt=}" + "\n" + "-" * 100)
        # 1. Change how you get the topic ~ Use an agent_ic RAG
        # topic_words, units_t = env_obj.get_topic_words()
        topic_words, units_t = env_obj_rag.get_topic_phrase(prompt), ""
        logging.info(f"\n[TOPIC WORDS] {topic_words=}" + "\n" + "-" * 100)

        # 2. Use LLMs to replace certain words.
        if np.random.normal() >= 0:
            topic_words = replace_words(topic_words, units_t, strategy="llm")
            logging.info(f"\n[R_TOPIC WORDS] {topic_words=}" + "\n" + "-" * 100)

        # 3. Pass final prompt to LLM.
        """
        Do not use LLM to check if the final question is valid or not because 
        our claim is that LLMs cannot solve complex physics questions.
        """
        ins0, ins1, ins2 = (
            "Generate a physics question using all the known and unknown variables. You must use all the variables.\n[variables] ",
            "\nAnd you may choose to use the elements from topic phrase.\n[topic phrase] ",
            ". Do not provide solution to the question, as it will be solved directly by the student.",
        )

        prompt = ins0 + prompt + ins1 + topic_words + ins2
        problem = get_response(prompt)
        logging.info(f"\n[FINAL PROMPT] {prompt=}" + "\n" + "-" * 100)
        print(f"{Style.BRIGHT}{Fore.GREEN}{problem.strip()}")
        print(f"{Fore.CYAN}[HINT] {get_solution(sol, data).strip()}{Style.RESET_ALL}")
        print("\n" + f"{Fore.BLACK}{Back.WHITE}--" * 30 + f"{Style.RESET_ALL}" + "\n")

        # 4. Push the physics question in a CSV file if it's valid.
        if cfg['Output']['BUILD']:
            with open(f"{cfg['Output']['Dir']}/{topic['OutputFile']}", "a") as file:
                new_row = {"Prompt": prompt, "Question": problem}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                print(f"{prompt}, {problem}", file=file)
        beautify(f"{cfg['Input']['TopicsDir']}/{topic['InputFile']}", data)
    df.to_csv(f"{cfg['Output']['Dir']}/{topic['OutputFile']}", index=False)


def main():
    cfg = OmegaConf.load("config.yaml")
    print(
        f"{Style.BRIGHT}{Fore.GREEN}Which topics do you want to generate questions from?\n"
    )
    for i in range(len(cfg.Topics)):
        print(f"{i}. {cfg.Topics[i].Name}")
    print(f"{Style.RESET_ALL}")
    choice = int(input(f"{Style.BRIGHT}Choose the index of the topic: "))
    if choice < 0 or choice >= len(cfg.Topics):
        print(f"{Style.BRIGHT}Wrong Choice!!\n")
    print(f"{Style.RESET_ALL}")
    get_phyQ(cfg, choice)


if __name__ == "__main__":
    main()