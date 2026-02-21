import os
from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage
from UTILS.utils import load_env_vars


class BaseLLM(ABC):
    """Base class for LLM implementations."""
    
    def __init__(self):
        load_env_vars()
    
    @abstractmethod
    def get_client(self):
        """Return the underlying LLM client."""
        pass
    
    def get_response(self, prompt: str) -> str:
        """Get a response from the LLM for the given prompt."""
        client = self.get_client()
        messages = [HumanMessage(content=prompt)]
        return client.invoke(messages).content
