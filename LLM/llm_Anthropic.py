import os 
from LLM.llm_Base import BaseLLM
from langchain_anthropic import ChatAnthropic


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM implementation."""
    
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self._client = None
    
    def get_client(self):
        """Lazily initialize and return the Anthropic client."""
        if self._client is None:
            self._client = ChatAnthropic(
                model_name=self.model_name,
                api_key=os.environ["ANTHROPIC_API_KEY"]
            )
        return self._client