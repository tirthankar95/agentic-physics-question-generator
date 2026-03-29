if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, f"{os.getcwd()}")
from LLM.llm import get_llm_obj
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict
from pydantic import BaseModel, Field
from typing import Optional, Literal

import logging

logging.basicConfig(
    level=logging.WARN, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrustedEditingGraph(TypedDict):
    # INPUT
    physics_question: str  # Current physics question.
    equation_prompt: str  # Contains the original
    # OUTPUT 1 ~ no loop.
    verdict_frmt: str  # Whether a question can be solved.
    physics_question_frmt: str  # Store the formatted physics question.
    # OUTPUT 2 ~ loops.
    verdict: str  # Whether a question can be solved.
    solution: Optional[str]  # Store the solution if the question is solvable.
    final_question: str  # Store the final question.
    # FINAL OUTPUT
    final_question_clean: str  # Store the final question.


class QuestionFormatResponse(BaseModel):
    is_valid_format: Literal["yes", "no"] = Field(
        description="Indicates whether the physics question includes all the known and unknown variables with their proper values."
    )
    corrected_question: Optional[str] = Field(
        default=None,
        description=(
            "If the format is invalid, provide a corrected version of the physics question. "
            "The corrected question must include all known and unknown variables with proper values."
        ),
    )


class QuestionValidityResponse(BaseModel):
    is_solvable: Literal["yes", "no"] = Field(
        description="Indicates whether the physics question is logically solvable."
    )
    solution: Optional[str] = Field(
        default=None, description="Step-by-step solution if the question is solvable."
    )
    final_question: Optional[str] = Field(
        default=None,
        description="Modified version of the physics question such that it becomes solvable. This should only be provided if the original question is deemed unsolvable.",
    )


class TEditAgent:
    def __init__(self, model_name: str, max_solvability_trials: int = 3):
        # Get LLMs
        self.model_name = model_name
        self.llm = get_llm_obj(model_name).get_client()
        self.llm_frmt = self.llm.with_structured_output(QuestionFormatResponse)
        self.llm_valid = self.llm.with_structured_output(QuestionValidityResponse)
        # Nodes
        self.graph_builder = StateGraph(TrustedEditingGraph)
        self.graph_builder.add_node("validate_format", self.validate_format)
        self.graph_builder.add_node("validate_solvability", self.validate_solvability)
        self.graph_builder.add_node("clean_question", self.clean_question)
        # Edges
        self.graph_builder.add_edge(START, "validate_format")
        self.graph_builder.add_conditional_edges(
            "validate_format",
            lambda state: state["verdict_frmt"],
            {"yes": "validate_solvability", "no": END},
        )
        self.graph_builder.add_conditional_edges(
            "validate_solvability",
            lambda state: state["verdict"],
            {"yes": "clean_question", "no": "validate_solvability"},
        )
        self.graph_builder.add_edge("clean_question", END)
        self.graph = self.graph_builder.compile()
        self.solvability_trials = 0
        self.mx_trials = max_solvability_trials

    def validate_format(self, state: TrustedEditingGraph) -> TrustedEditingGraph:
        # Implement logic to validate the format of the physics question
        prompt = [
            SystemMessage(
                content=f"""You are an agent that validates the format of physics questions.

A physics question is considered valid if:
- All known variables from the 'Equation Prompt' are present with correct values.
- All unknown variables are clearly specified (without values).

If the question is valid, respond with exactly:
    yes

If the question is not valid, respond with:
    no
Corrected Question: <provide corrected version including all variables properly>
Do not include any extra explanation."""
            ),
            HumanMessage(
                content=f"""Equation Prompt:
{state['equation_prompt']}

Physics Question:
{state['physics_question']}"""
            ),
        ]
        response = self.llm_frmt.invoke(prompt)
        state["verdict_frmt"] = response.is_valid_format
        if response.corrected_question:
            state["physics_question_frmt"] = response.corrected_question
        else:
            state["physics_question_frmt"] = state["physics_question"]
        return state

    def validate_solvability(self, state: TrustedEditingGraph) -> TrustedEditingGraph:
        # Implement logic to determine if the physics question is solvable
        relevant_question = (
            state["physics_question_frmt"]
            if self.solvability_trials == 0
            else state["final_question"]
        )
        # If maximum number of iterations have been reached.
        if self.solvability_trials >= self.mx_trials:
            logger.warning(
                "Maximum solvability trials reached. Question is unsolvable."
            )
            state["verdict"] = "yes"
            state["final_question"] = relevant_question
            return state
        prompt = [
            SystemMessage(
                content="""You are an expert physics problem solver.
Your task is to determine whether a given physics question is solvable.

- If the question is solvable:
    Respond with:
        yes
    Solution: <step-by-step solution>
    Original Question: <original question>

- If the question is not solvable:
    Respond with:
        no
    Modified Question: <rewrite the question so that it becomes solvable by adding missing information>
Do not include any extra explanation."""
            ),
            HumanMessage(
                content=f"""Physics Question:
{relevant_question}
"""
            ),
        ]
        response = self.llm_valid.invoke(prompt)
        state["verdict"] = response.is_solvable
        if response.solution:
            state["solution"] = response.solution
        if response.final_question:
            state["final_question"] = response.final_question
        else:
            state["final_question"] = relevant_question
        self.solvability_trials += 1
        return state

    def clean_question(self, state: TrustedEditingGraph) -> TrustedEditingGraph:
        if self.solvability_trials >= self.mx_trials:
            return state
        prompt = [
            SystemMessage(
                content=f"""You are an agent extracts and cleans a physics question.
- The clean question shouldn't have any unnecessary sentences and preambles like 'Here is the clean code' or any solution to the problem. 
- It should only have the physics question in a clean presentable way. 
- You may also add emojis to make the question more engaging and more fun. 
- Remember don't modify the content of the actual question."""
            ),
            HumanMessage(
                content=f"""Question to clean:
{state['final_question']}"""
            ),
        ]
        response = self.llm.invoke(prompt)
        state["final_question_clean"] = response.content
        return state

    def run(self, question: str, equation_prompt: str):
        # Implement logic to run the graph with the initial question
        self.solvability_trials = 0
        state = self.graph.invoke(
            {"physics_question": question, "equation_prompt": equation_prompt}
        )
        if self.solvability_trials >= self.mx_trials:
            state["verdict"] = "no"
        return state


if __name__ == "__main__":
    agent = TEditAgent(model_name="claude-3-haiku-20240307")
    state = agent.run(
        equation_prompt="""Generate a physics question using all the known and unknown variables. You must use all the variables.[variables] train acceleration = 80ms−2 at an angle of 152 degrees with the horizontal, 
train velocity = 57 ms−1 at an angle of 8 degrees with the horizontal, train length = 544m, train height = 68m.""",
        question="""A train is traveling at a constant acceleration of 79.57ms−2 at an angle of 152 degrees with the horizontal. 
At the same time, a ball is thrown from the train with an initial velocity of 57.91ms−1 at an angle 8 degrees with the horizontal. What is the maximum height the ball will reach ?""",
    )
    logging.info("STATE:  -- >\n")
    logging.info(state)
