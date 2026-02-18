import os
import json
import glob
import logging
import numpy as np

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def fix(no, number_type="R"):
    """
    If the number is less than 10^5 and greater than equal 10^-1
    then we round it to 2 decimal places
    else we convert it to scientific notation.
    """
    if no == 0:
        return 0
    if number_type == "Z":
        return int(no)
    exponent = int(np.log10(abs(no)))
    if 0 <= exponent < 5:
        return round(no, 2)
    return f"{no:.2e}"


def load_env_vars():
    with open("LLM_CONFIG/config.json", "r") as file:
        env = json.load(file)
    for k, v in env.items():
        if k[-3:] == "KEY" or k[-5:] == "TOKEN":
            os.environ[k] = v


class Env:
    def __init__(self, filename):
        files = glob.glob(f"ENTITY/{filename}*.json")
        print(files)
        for file in files:
            with open(f"{file}", "r") as file:
                self.data = json.load(file)
                self.envs = [k for k in self.data]

    def get_topic_words(self):
        units = []
        if len(self.data) == 0:
            return self.prefix, units
        env = self.envs[np.random.randint(len(self.data))]
        topic_words = f"{env} it's properties and topic words - "
        if len(self.data[env]["topic_words"]):
            topic_words += np.random.choice(self.data[env]["topic_words"]) + ", "
        two_d = (
            True if np.random.normal() >= -0.5 else False
        )  # (1: Enable, 0: Disable) 2D
        for attribute, v in self.data[env].items():
            # An attribute can be skipped or not to increase the variability.
            if attribute == "topic_words" or np.random.normal() > 0:
                continue
            v_range, unit, type, number_type = v
            type = 0 if type == "S" else 1  # (0: scaler, 1: vector)
            var = fix(
                np.random.uniform(v_range[0], v_range[1]), number_type=number_type
            )
            if two_d and type == 1:
                theta = np.random.randint(0, 180)
                topic_words += f"{env} {attribute} = {var} {unit} at an angle {theta} degrees with the horizontal, "
            else:
                topic_words += f"{env} {attribute} = {var} {unit}, "
            if unit not in units:
                units.append(unit)
        return topic_words[:-2] + ".", units
