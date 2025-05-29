from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from agents.state import ChatState
from langgraph.graph import StateGraph, END
from agents.dispatcher import DispatcherNode

# ────────────────────────────────────────────────────────────────
# 1️⃣  SHARED STATE  (everything nodes might read / write)
# ────────────────────────────────────────────────────────────────
@dataclass
class ChatState:
    stage: str = "fallback"                      # Current high-level stage
    user_message: str = ""                       # Latest user utterance
    resume: Optional[Dict[str, Any]] = None      # Parsed résumé
    preferences: Optional[Dict[str, Any]] = None # User’s filters
    positions: List[Dict[str, Any]] = field(default_factory=list)
    confirmation: Optional[bool] = None          # Did the user say “yes”?
    emails_sent: bool = False
    replies: List[Dict[str, Any]] = field(default_factory=list)
    follow_up_done: bool = False                 # After follow-up completed


# ────────────────────────────────────────────────────────────────
# 2️⃣  NODE HANDLERS  (plug your real business logic later)
# Each takes (state, **kwargs) and returns *updated* state
# ────────────────────────────────────────────────────────────────

def fallback_node(state: ChatState, **_) -> ChatState:
    """
    Handles greetings / general questions.
    """
    print("🤖 fallback:", state.user_message)
    # If user provided a résumé or intent, escalate
    if "resume" in state.user_message.lower():
        state.stage = "intake"
    return state


def intake_node(state: ChatState, **_) -> ChatState:
    """
    Parse résumé & preferences, then find and store matching positions.
    """
    print("📥 intake")
    # Simulate parsing and finding positions
    state.resume = {"parsed": True}
    state.preferences = {"dummy": 1}
    state.positions = [
        {"title": "PhD in Quantum Optics", "institution": "ETH Zürich"},
        {"title": "Postdoc in NLP", "institution": "MIT CSAIL"},
    ]
    # After showing positions, go to check_replies
    state.stage = "check_replies"
    return state


def check_replies_node(state: ChatState, **_) -> ChatState:
    """
    Poll mailbox to see if any professor replied.
    """
    print("🔎 checking replies")
    # Simulate checking for replies
    state.replies = []  # e.g., [{"prof": "Dr Smith", "positive": True}]
    if state.replies:
        state.stage = "follow_up"
    else:
        state.stage = "END"
    return state


def follow_up_node(state: ChatState, **_) -> ChatState:
    """
    Send polite follow-ups or schedule interview coach.
    """
    print("🔁 follow-up phase")
    state.follow_up_done = True
    state.stage = "END"
    return state


# ────────────────────────────────────────────────────────────────
# 3️⃣  BUILD THE GRAPH
# ────────────────────────────────────────────────────────────────
graph = StateGraph(ChatState)

from agents.openai_llm import OpenAILLM
llm = OpenAILLM(api_key="tpsg-b1MsaOzQ0DhJ9ULVYLxaX2j7hwmC1DJ")
dispatcher = DispatcherNode(llm=llm)

graph.add_node("dispatcher", dispatcher)
graph.add_node("fallback", fallback_node)
graph.add_node("intake", intake_node)
graph.add_node("check_replies", check_replies_node)
graph.add_node("follow_up", follow_up_node)

# ENTRY
graph.set_entry_point("dispatcher")

# CONDITIONAL EDGES
graph.add_conditional_edges(
    "dispatcher",
    lambda s, *_: s.stage,
    {
        "fallback": "fallback",
        "intake": "intake",
    }
)

graph.add_conditional_edges(
    "intake",
    lambda s, *_: s.stage,
    {"check_replies": "check_replies"},
)

graph.add_conditional_edges(
    "check_replies",
    lambda s, *_: "follow_up" if s.replies else END,
    {"follow_up": "follow_up"},
)

graph.add_edge("fallback", END)
graph.add_edge("follow_up", END)

lang_graph = graph.compile()

# ────────────────────────────────────────────────────────────────
# 4️⃣  MINI DRIVER  (for CLI or Chainlit integration)
# ────────────────────────────────────────────────────────────────
def plot_graph(filename: str = "figures/langgraph.png"):
    """
    Plots the compiled LangGraph as a PNG using Mermaid.
    """
    g = lang_graph.get_graph()
    try:
        png_bytes = g.draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_bytes)
        print(f"Graph saved as {filename}")
    except Exception as e:
        print(f"Error plotting graph: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_graph()
        sys.exit(0)
    state = ChatState()

    while True:
        state.user_message = input("👤 You: ")
        state = lang_graph.invoke(state)
        if state.stage == "END":
            print("🏁 Conversation finished.")
            break
