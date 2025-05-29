import os
from typing import Any
import openai

class OpenAILLM:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url="https://api.metisai.ir/openai/v1"
        self.model = model
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided via argument or OPENAI_API_KEY env variable.")
        openai.api_key = self.api_key
        openai.base_url = base_url

    def __call__(self, prompt: str, **kwargs) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
            **kwargs
        )
        return response["choices"][0]["message"]["content"]
