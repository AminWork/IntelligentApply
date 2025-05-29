from typing import Any
from agents.state import ChatState
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

# Define the ChosenTool class for the parser
class ChosenTool(BaseModel):
    tool_name: Literal["fallback", "intake"] = Field(
        description="the tool that was chosen by LLM in question routing stage"
    )

class DispatcherNode:
    def __init__(self, llm=None):
        self.llm = llm  # Pass an LLM instance if available
        self.parser = PydanticOutputParser(pydantic_object=ChosenTool)

    def forward(self, state: ChatState, **kwargs) -> ChatState:
        msg = state.user_message.lower()
        if self.llm:
            prompt = (
                "You are a routing assistant for an academic apply system. "
                "Your task is to decide whether the user uploaded their resume with preferences "
                "or is asking questions about our service.\n"
                f"User message: {state.user_message}\n"
                f"{self.parser.get_format_instructions()}"
            )
            try:
                response = self.llm(prompt)
                chosen = self.parser.parse(response)
                print("### Router response:", chosen.tool_name)
                if chosen.tool_name == 'intake':
                    state.stage = 'intake'
                else:
                    state.stage = 'fallback'
            except Exception:
                state.stage = 'fallback'
        else:
            # Fallback: simple rules
            if "resume" in msg or "apply" in msg:
                state.stage = "intake"
            elif any(word in msg for word in ["hi", "hello", "what do you do", "who are you"]):
                state.stage = "fallback"
            else:
                state.stage = "fallback"
        return state

    def __call__(self, state: ChatState, **kwargs) -> ChatState:
        return self.forward(state, **kwargs)
