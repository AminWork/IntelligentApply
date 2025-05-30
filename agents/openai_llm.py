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

    def get_embedding(self, text: str, model="text-embedding-3-small"):
        try:
            text = text.replace("\n", " ") # OpenAI recommends replacing newlines
            print(f"[Embedding] {text}")
            print(f"[OpenAILLM DEBUG] Requesting embedding for text (first 100 chars): '{text[:100]}...' with model {model}")
            response = self.client.embeddings.create(
                input=[text],
                model=model,
                encoding_format="float"
                )
            print(f"[Embedding] {response.data[0].embedding}")
            embedding = response.data[0].embedding
            # text-embedding-3-small and ada-002 (older) produce 1536 dimensions
            # text-embedding-3-large produces 3072 dimensions
            if model == "text-embedding-3-small" and len(embedding) != 1536:
                 print(f"[OpenAILLM WARNING] Embedding dimension mismatch for {model}. Expected 1536, got {len(embedding)}. Text: '{text[:100]}...'")
            elif model == "text-embedding-3-large" and len(embedding) != 3072:
                 print(f"[OpenAILLM WARNING] Embedding dimension mismatch for {model}. Expected 3072, got {len(embedding)}. Text: '{text[:100]}...'")

            print(f"[OpenAILLM DEBUG] Embedding generated successfully. Dimensions: {len(embedding)}")
            return embedding
        except Exception as e:
            print(f"[ERROR in OpenAILLM get_embedding] OpenAI embedding call failed: {e}")
            import traceback
            print(traceback.format_exc())
            return None # Return None or an empty list on failure, to be handled by the caller
