from typing import Any
from agents.state import ChatState
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

class ChosenTool(BaseModel):
    tool_name: Literal["fallback", "intake"] = Field(
        description="The tool that was chosen by LLM in question routing stage"
    )

class DispatcherNode:
    def __init__(self, llm):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=ChosenTool)

    def forward(self, state: ChatState, **kwargs) -> ChatState:
        print(f"[LOG] Entered DispatcherNode with user_message: {state.user_message}")

        prompt_template = (
            "You are a routing assistant for an academic application system. "
            "Your task is to decide what the user wants to do based on their message.\n\n"
            "If the user is uploading their resume and preferences, respond with \"intake\".\n"
            "If the user is greeting, asking about the service, or asking general questions, respond with \"fallback\".\n\n"
            "User message: {user_message}\n\n"
            "Respond with only one word: either \"intake\" or \"fallback\".\n"
            "{format_instructions}"
        )
        prompt = prompt_template.format(
            user_message=state.user_message,
            format_instructions=self.parser.get_format_instructions()
        )

        try:
            raw_response = self.llm(prompt)
            response_text = getattr(raw_response, "content", raw_response)
            chosen = self.parser.parse(response_text)
            print("[Router] Tool chosen:", chosen.tool_name)
            state.stage = chosen.tool_name
        except Exception as e:
            print(f"[ERROR] Dispatcher routing failed: {e}")
            state.stage = "fallback"
        return state

    def __call__(self, state: ChatState, **kwargs) -> ChatState:
        return self.forward(state, **kwargs)
