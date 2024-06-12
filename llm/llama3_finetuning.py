import os

from llm.base_llm import BaseLLM
import requests
import openai

class LLaMA3(BaseLLM):
    """Class for the LLaMA3 turbo model from OpenAI with 4000 tokens of context"""

    def __init__(self):
        """Constructor for the LLama class
        Args:
            prompt_token_cost (float): Cost of a token in the prompt
            response_token_cost (float): Cost of a token in the response
        """
        super().__init__(0, 0, 8000, 0.7)

        self.endpoint = os.getenv("LLAMA_ENDPOINT")

       

    def _format_prompt(self, prompt: str, role: str = 'user') -> list[dict[str, str]]:
        """Format the prompt to be used by the LLaMA-3 8B
        Args:
            prompt (str): Prompt
        Returns:
            list: List of dictionaries containing the prompt and the role of the speaker
        """
        return [
            { "role": role, "content": prompt}
        ]

    def __completion(self, prompt: str, **kwargs) -> tuple[str, int, int]:
        """Completion api for the LLaMA 3 8B Instruct model
        Args:
            prompt (str): Prompt for the completion
        Returns:
            tuple(str, int, int): A tuple with the completed text, the number of tokens in the prompt and the number of tokens in the response
        """

        # Check if there is a system prompt
        if "system_prompt" in kwargs:
            system_prompt = kwargs["system_prompt"]
            del kwargs["system_prompt"]
        else:
            system_prompt = ""

        info = {"system_prompt":system_prompt,"prompt": prompt}
        response = requests.post(self.endpoint, json=info).json()

        return response["generated_text"], 0, 0
    
    def _completion(self, prompt: str, **kwargs) -> tuple[str, int, int]:
        """Wrapper for the completion api with retry and exponential backoff
        
        Args:
            prompt (str): Prompt for the completion

        Returns:
            tuple(str, int, int): A tuple with the completed text, the number of tokens in the prompt and the number of tokens in the response
        """
        wrapper = BaseLLM.retry_with_exponential_backoff(self.__completion, self.logger, errors=(openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError))
        return wrapper(prompt, **kwargs)
    
    def _calculate_tokens(self, prompt: str) -> int:
        """Calculate the number of tokens in the prompt
        Args:
            prompt (str): Prompt
        Returns:
            int: Number of tokens in the prompt
        """
        
        num_tokens = 0
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 0 #len(self.encoding.encode(prompt))
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    