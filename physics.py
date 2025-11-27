import json
import numpy as np
import pandas as pd
import sys
from collections import namedtuple

sys.path.insert(0, "LLM_CONFIG")
from UTILS.utils import GraphEquation, RagAgent, fix
from LLM.llm import get_response
from colorama import Fore, Back, Style
from UTILS.word_change import replace_words
from UTILS.utils import load_env_vars
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
Topic = namedtuple("Topic", ["name", "collection_name", "in_file", "out_file"])


def generate_question_variables(data):
    equations, units = data["equations"], {}
    obj = GraphEquation(equations)
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


def get_phyQ(topic):
    """
    1. Read topic configs.
    TOPICS/<topic_name>.json contains equations associated with each topic.
    2. Read llm configs.
    LLM_CONFIG/<config>.json which model to use & passwords to use.

    Generates a physics question TRIAL number of times and saves it to DATASET/<topic_name>.csv
    """
    load_env_vars()
    with open(f"TOPICS/{topic.in_file}") as file:
        data = json.load(file)
    with open("LLM_CONFIG/config.json", "r") as file:
        env = json.load(file)
    env_obj_rag = RagAgent(
        model_name=env["LLM_MODEL"]["INNER_MODEL"],
        collection_name=topic.collection_name,
    )
    try:
        df = pd.read_csv(f"DATASET/{topic.out_file}")
    except Exception as e:
        print(f"{Style.BRIGHT}{Fore.RED}Creating file...{Style.RESET_ALL}")
        with open(f"DATASET/{topic.out_file}", "w") as ofile:
            ofile.write("")

    for _ in range(TRIALS):
        # 0. Get set of equations to form the question.
        prompt, sol, units_p = generate_question_variables(data)
        logging.info(f"\n[PROMPT] {prompt=}" + "\n" + "-" * 100)

        # 1. Change how you get the topic ~ Use Agentic RAG
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

        # 5. Push the physics question in a CSV file if it's valid.
        if env["BUILD_DATASET"]:
            with open(f"DATASET/{topic.out_file}", "a") as file:
                new_row = {"Prompt": prompt, "Question": problem}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                print(f"{prompt}, {problem}", file=file)
        beautify(f"TOPICS/{topic.in_file}", data)
    df.to_csv(f"DATASET/{topic.out_file}", index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Argparser for Physics Question Generation."
    )
    args.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of physics questions to generate per topic",
    )
    TRIALS = args.parse_args().trials
    topics = [
        Topic("SIMPLE KINEMATICS", "env_sm", "simple_motion.json", "physics_sm.csv"),
        Topic("NUCLEAR PHYSICS", "env_np", "nuclear_physics.json", "physics_np.csv"),
        Topic("GRAVITATION", "env_g", "gravitation.json", "physics_g.csv"),
        Topic("ELECTROSTATICS", "env_elec", "electrostatics.json", "physics_elec.csv"),
    ]
    print(
        f"{Style.BRIGHT}{Fore.GREEN}Which topics do you want to generate questions from?\n"
    )
    for i in range(len(topics)):
        print(f"{i}. {topics[i][0]}")
    print(f"{Style.RESET_ALL}")
    choice = int(input(f"{Style.BRIGHT}Choose the index of the topic: "))
    if choice < 0 or choice >= len(topics):
        print(f"{Style.BRIGHT}Wrong Choice!!\n")
    print(f"{Style.RESET_ALL}")
    get_phyQ(topics[choice])
