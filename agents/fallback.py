from agents.state import ChatState
from agents.openai_llm import OpenAILLM

class FallbackNode:
    def __init__(self, llm=None):
        self.llm = llm or OpenAILLM()

    def __call__(self, state: ChatState, **kwargs) -> ChatState:
        print("🤖 Entered fallback_node")
        prompt = f"""
You are an assistant that helps users apply to academic research positions.

ONLY answer questions related to:
- Academic job discovery
- Résumé or CV parsing
- Research field matching
- Writing personalized application emails
- Following up with professors
- Preparing for academic interviews

If the user asks something out of scope, politely say:
"I'm here to help with academic job applications. Could you tell me about your research interests or share your résumé?"

User message: "{state.user_message}"
"""
        try:
            response = self.llm(prompt)
            print("[Fallback LLM response] >", response)
            state.llm_response = response
            state.reply = response  # For Chainlit UI
            print("[Fallback LLM response]")
        except Exception as e:
            print(f"[ERROR in fallback_node] {e}")
            state.llm_response = "I'm here to ..."
            state.reply = state.llm_response
        state.stage = "waiting"  # a non-graph stage to pause routing
        return state
