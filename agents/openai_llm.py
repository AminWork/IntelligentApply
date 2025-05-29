from openai import OpenAI
import os

class OpenAILLM:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        self.api_key = "tpsg-b1MsaOzQ0DhJ9ULVYLxaX2j7hwmC1DJ"
        if not self.api_key:
            raise ValueError("API key must be set")
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.metisai.ir/openai/v1")
        self.model = model

    def __call__(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=256,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI call failed: {e}")
