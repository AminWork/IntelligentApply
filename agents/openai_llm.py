from openai import OpenAI
import os

class OpenAILLM:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        self.api_key = "tpsg-b1MsaOzQ0DhJ9ULVYLxaX2j7hwmC1DJ"
        if not self.api_key:
            raise ValueError("API key must be set")
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.metisai.ir/openai/v1")
        self.model = model

    def __call__(self, messages, **kwargs) -> str:
        try:
            print(f"[OpenAILLM DEBUG] Making API call to model {self.model} with messages: {messages}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=512,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR in OpenAILLM] OpenAI call failed: {e}")
            import traceback
            print(traceback.format_exc())
            # Ensure a string is raised, consistent with type hint, though it will be caught by fallback
            raise RuntimeError(f"OpenAI call failed: {e}")
