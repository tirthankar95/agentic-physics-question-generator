import re
import glob
import logging
import numpy as np
from uuid import uuid4
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    ## Check Topic Word Selection.
    """
    ## Check Agentic RAG
        r_agent = RagAgent(model_name="local", collection_name="env_elec")
        resp =  r_agent.get_topic_phrase("Superposition Principle: The principle is based on the property that the forces
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
