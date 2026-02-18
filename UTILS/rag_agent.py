import logging
from UTILS.rag_db import RagDB
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Literal, TypedDict, Optional

logger = logging.getLogger(__name__)


class RagAgent:
    class RagAgentState(TypedDict):
        question: str
        verdict: str
        vector_chunks: str
        summary_chunk: str

    class FetchAgentResponse(BaseModel):
        verdict: Literal["halt", "fetch"] = Field(
            description="If you need to fetch extra information from vector DB, send verdict as fetch else halt."
        )
        vector_sentence: Optional[str] = Field(
            description="""The sentence you need to form that will be used to fetch information from vector store.
If you do not need to fetch additional data keep the vector_sentence as empty."""
        )

    def __init__(self, model_name, collection_name):
        import os
        import sys

        sys.path.insert(0, os.getcwd())
        from LLM.llm_OpenAI import get_llm

        self.model_name = model_name
        self.llm = get_llm(model_name)
        self.llm_fetch = self.llm.with_structured_output(self.FetchAgentResponse)
        self.collection_name = collection_name
        self.rag = RagDB(collection_name)
        self.form_workflow()

    def fetch_agent(self, state: RagAgentState):
        if "vector_chunks" not in state:
            # Initially do a basic retrieval.
            chunks = self.rag.get_chunks(state["question"])
            state["vector_chunks"] = chunks + "\n"
            logger.info("[Initial Prompt:]\n" + state["question"])
            logger.debug(
                "[Initial Chunks:]\n" + chunks[-100:] + "\n" + "---" * 25 + "\n"
            )
            return {"vector_chunks": chunks, "verdict": "fetch"}
        else:
            self.mx_cycle -= 1
            prompt = [
                SystemMessage(
                    content=f"""You are an agent who is an expert at retrieving relevant information from a vector database so that the next agent can generate physics questions based on user prompts.
As input, you will receive the original user prompt and the history of retrieved vector chunk. Your task is to determine whether the retrieved information is sufficient to form a good physics question.
If it is not sufficient, you will create a prompt which will be sent to the vector database again to obtain more relevant information.
Remember chunks are retrieved after doing similarity search with the question you ask. Now create the question.
"""
                ),
                HumanMessage(
                    content=f"""Original prompt: {state['question']},\nHistory of vector chunks: {state['vector_chunks']}"""
                ),
            ]
            query = self.llm_fetch.invoke(prompt)
            if query.verdict == "halt" or self.mx_cycle == 0:
                return {"verdict": "halt"}
            state["vector_chunks"] += self.rag.get_chunks(query.vector_sentence) + "\n"
            logger.info("[VectorDB Prompt:]\n" + query.vector_sentence)
            logger.debug(
                "[Next Chunks:]\n"
                + state["vector_chunks"][-100:]
                + "\n"
                + "---" * 25
                + "\n"
            )
            return {"vector_chunks": state["vector_chunks"], "verdict": "fetch"}

    def router(self, state: RagAgentState):
        return state["verdict"]

    def summarize(self, state: RagAgentState):
        prompt = [
            SystemMessage(
                content="""You are an expert at summarizing large and complex information, even when it is not written in proper English. 
Your task is to:
1. Summarize the user’s query concisely and bring out frequently occurring topic words and phrases.
2. Correct any grammatical or language issues if needed.
3. Do not say the user wants this or wants that or do not print numbers, just mention the topic words and phrases."""
            ),
            HumanMessage(content=f"""{state['vector_chunks']}"""),
        ]
        response = self.llm.invoke(prompt)
        return {"summary_chunk": response.content}

    def form_workflow(self):
        workflow = StateGraph(self.RagAgentState)
        workflow.add_node(self.fetch_agent, "fetch_agent")
        workflow.add_node(self.summarize, "summarize")
        workflow.add_edge(START, "fetch_agent")
        workflow.add_conditional_edges(
            "fetch_agent", self.router, {"halt": "summarize", "fetch": "fetch_agent"}
        )
        workflow.add_edge("summarize", END)
        self.ai_rag = workflow.compile()

    def get_topic_phrase(self, question):
        self.mx_cycle = 5
        state = self.ai_rag.invoke({"question": question})
        return state["summary_chunk"]
