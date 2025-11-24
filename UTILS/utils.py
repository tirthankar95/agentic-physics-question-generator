import re
import os
import json
import glob
import numpy as np
from uuid import uuid4
from collections import defaultdict
from typing import Literal, TypedDict, Optional
from pydantic import BaseModel, Field
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
import logging

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

operators = ["+", "*", "=", "/", "^", "(", ")", "-", "exp("]


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


def parse(equation):
    elements_ = [
        var
        for var in equation.split(" ")
        if var not in operators and len(var.strip()) > 0
    ]
    elements = []
    for x in elements_:
        try:
            var = float(x)
        except:
            elements.append(x)
    return elements


class GraphEquation:
    def __init__(self, equations):
        self.equation_element = []
        for equation in equations:
            self.equation_element.append(parse(equation))
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

    def getEquation(self):
        qid = np.random.randint(len(self.adj))
        threshold, eqn = 0.25, [qid]
        self.vis, unk = defaultdict(bool), defaultdict(bool)
        self.qu = [qid]
        self.vis[qid] = True
        while len(self.qu):
            src = self.qu.pop(0)
            if 0.5 + np.random.normal() >= threshold:
                edgeId = np.random.randint(len(self.adj[src]))
                edge = self.adj[src][edgeId]
                if edge[-1] in unk:
                    break
                if edge[0] not in self.vis:
                    unk[edge[-1]] = True
                    eqn.append(edge[0])
                    self.qu.append(edge[0])
                    self.vis[edge[0]]
        while True:
            ch = np.random.choice(self.equation_element[eqn[-1]])
            if ch not in unk:
                break
        unk[ch] = True
        assert len(unk) == len(eqn), f"Bad Equation: {unk} {eqn}"
        known = defaultdict(bool)
        for eId in eqn:
            for ch in self.equation_element[eId]:
                if ch not in unk:
                    known[ch] = True
        unk, known = [k for k in unk.keys()], [k for k in known.keys()]
        logging.debug(f"{unk}, {known}")
        return unk, known, eqn


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


######################
### Agent Workflow ###
######################
class RagDB:
    def __init__(
        self, collection_name, model_name="sentence-transformers/all-mpnet-base-v2"
    ):
        """
        To update vector DB with new documents first delete and then create.
        """
        self.topic = collection_name
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        self.hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        if not self.collection_exists():
            self.create_collection()

    def collection_exists(self):
        client = QdrantClient(path="QdrantDB")
        collections = client.get_collections().collections
        collections = [c.name for c in collections]
        client.close()
        return True if self.topic in collections else False

    def delete_collections(self, collection_name):
        client = QdrantClient(path="QdrantDB")
        # BEFORE
        collections = client.get_collections().collections
        collections = [c.name for c in collections]
        print(f"Before Deletion: {', '.join(collections)}")
        client.delete_collection(collection_name=collection_name)
        # AFTER
        collections = client.get_collections().collections
        collections = [c.name for c in collections]
        print(f"After Deletion: {', '.join(collections)}")
        client.close()

    def create_collection(self):
        print(f"[TM] tirthankar mittra:")
        client = QdrantClient(path="QdrantDB")
        client.create_collection(
            collection_name=self.topic,
            vectors_config=VectorParams(
                size=768, distance=Distance.COSINE
            ),  # NOTE: size depends on model we are using.
        )
        qdrant = QdrantVectorStore(
            client=client,
            collection_name=self.topic,
            embedding=self.hf,
            retrieval_mode=RetrievalMode.DENSE,
        )
        files = glob.glob(f"RAW_TEXT/{self.topic}*.pdf")
        print(f"{files=}")
        for file in files:
            logger.info(f"Adding {file}")
            loader = PyPDFLoader(file)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            documents = text_splitter.split_documents(documents)
            uuids = [str(uuid4()) for _ in range(len(documents))]
            qdrant.add_documents(documents=documents, ids=uuids)
        client.close()

    def get_chunks(self, query: str, top_k: int = 10):
        client = QdrantClient(path="QdrantDB")
        qdrant = QdrantVectorStore(
            client=client,
            collection_name=self.topic,
            embedding=self.hf,
            retrieval_mode=RetrievalMode.DENSE,
        )
        found_docs = qdrant.similarity_search_with_score(query, k=top_k)
        for doc, score in found_docs:
            logger.debug(
                f"Score[{score}]\n{doc.page_content[:50]}\n------------------------\n"
            )
        docs, scores = [doc.page_content for doc, score in found_docs], [
            score for doc, score in found_docs
        ]
        scores = np.exp(np.array(scores) - max(scores))
        scores /= np.sum(scores)
        index = np.random.choice(len(docs), p=scores)
        chunks = docs[index]
        assert top_k >= len(found_docs), f"returned chunks are more than top_k."
        chunks = re.sub(r"[^\w\s=.]", " ", chunks)
        client.close()
        return chunks


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


######################
######################
######################

if __name__ == "__main__":
    ## Check Topic Word Selection.
    # obj_env = Env("env_sm.json")
    # topic_words, units = obj_env.get_topic_words()
    # print(topic_words, units)
    """
        ## Check Agentic RAG
        r_agent = RagAgent(model_name="local", collection_name="env_elec")
        resp = r_agent.get_topic_phrase("Superposition Principle: The principle is based on the property that the forces
    with which two charges attract or repel each other are not affected by the presence of a third")
        print(resp)
    """
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    hf = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    client = QdrantClient(path="QdrantDB")
    qdrant = QdrantVectorStore(
        client=client,
        collection_name="env_elec",
        embedding=hf,
        retrieval_mode=RetrievalMode.DENSE,
    )
    found_docs = qdrant.similarity_search_with_score(
        """capacitance = 6.88e-08 F, electric potential = 228.22 V, total electrostatic energy = 8.51e-06 J, 
velocity of particle = 9.65e+05 m/s, electrostatic potential energy = unknown, mass of particle = unknown.""",
        k=10,
    )
    for doc, score in found_docs:
        logger.debug(
            f"Score[{score}]\n{doc.page_content[:50]}\n------------------------\n"
        )
