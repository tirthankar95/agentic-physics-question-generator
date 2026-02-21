from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


class HuggingFaceLLM:
    """HuggingFace LLM implementation with response post-processing."""
    
    def __init__(self, model_name: str, temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self._client = None
        self._chain = None
    
    def get_client(self):
        """Lazily initialize and return the HuggingFace client."""
        if self._client is None:
            self._client = HuggingFaceEndpoint(
                repo_id=self.model_name,
                temperature=self.temperature
            )
        return self._client
    
    @staticmethod
    def stop_hallucinations(response: str) -> str:
        """Post-process response to remove hallucinated follow-up questions."""
        return response.split("Question:")[0]
    
    def _build_chain(self):
        """Build the LangChain processing chain."""
        template = """
        {question}
        """
        prompt = PromptTemplate(input_variables=["question"], template=template)
        
        chain = (
            {"question": lambda x: x}
            | prompt
            | self.get_client()
            | StrOutputParser()
            | self.stop_hallucinations
        )
        return chain
    
    def get_response(self, problem: str) -> str:
        """Get a response from the model for the given problem."""
        if self._chain is None:
            self._chain = self._build_chain()
        return self._chain.invoke(problem)


# Usage:
# llm = HuggingFaceLLM("microsoft/Phi-3.5-mini-instruct")
# response = llm.get_response("What is Python?")

# Or with different temperature:
# llm = HuggingFaceLLM("HuggingFaceH4/zephyr-7b-beta", temperature=0.5)