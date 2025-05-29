# graph.py
from __future__ import annotations
from langgraph.graph import StateGraph, END
from agents.state import ChatState
from agents.dispatcher import DispatcherNode
from agents.openai_llm import OpenAILLM
from agents.fallback import FallbackNode
import os

def intake_node(state: ChatState, **_) -> ChatState:
    print("📥 Intake: parsed CV + found jobs")
    state.positions = [{"title": "PhD …"}, {"title": "Postdoc …"}]
    state.stage = "confirm"
    return state

def confirm_node(state: ChatState, **_) -> ChatState:
    msg = state.user_message.lower()
    if "yes" in msg:
        state.stage = "send_emails"
    elif "no" in msg:
        state.stage = "dispatcher"  # maybe user wants to change filters
    else:
        # ask again
        print("❔ Please say yes/no")
        state.stage = "confirm"
    return state

def send_emails_node(state: ChatState, **_) -> ChatState:
    print("📧 Emails sent!")
    state.emails_sent = True
    state.stage = "check_replies"
    return state

def check_replies_node(state: ChatState, **_) -> ChatState:
    print("🔎 Checking mailbox")
    state.replies = []  # TODO real fetch
    state.stage = "follow_up" if state.replies else "dispatcher"
    return state

def follow_up_node(state: ChatState, **_) -> ChatState:
    print("🔁 Follow-up logic")
    state.follow_up_done = True
    state.stage = "dispatcher"
    return state

# ────────────── Build the graph ──────────────
graph = StateGraph(ChatState)

llm = OpenAILLM()                                     # reads key from env
dispatcher = DispatcherNode(llm=llm)
fallback = FallbackNode(llm=llm)

graph.add_node("dispatcher", dispatcher)
graph.add_node("fallback", fallback)
graph.add_node("intake", intake_node)
graph.add_node("confirm", confirm_node)
graph.add_node("send_emails", send_emails_node)
graph.add_node("check_replies", check_replies_node)
graph.add_node("follow_up", follow_up_node)

graph.set_entry_point("dispatcher")

# Dispatcher routes first
graph.add_conditional_edges(
    "dispatcher",
    lambda s, *_: s.stage,
    dict(fallback="fallback", intake="intake")
)

# Intake → confirm
graph.add_conditional_edges(
    "intake",
    lambda s, *_: s.stage,
    {"confirm": "confirm"}
)

# Confirm → send_emails or back to dispatcher
graph.add_conditional_edges(
    "confirm",
    lambda s, *_: s.stage,
    {"send_emails": "send_emails", "dispatcher": "dispatcher", "confirm": "confirm"}
)

# send_emails → check
graph.add_conditional_edges(
    "send_emails",
    lambda s, *_: "check_replies",
    {"check_replies": "check_replies"}
)

# check → follow_up or back to dispatcher
graph.add_conditional_edges(
    "check_replies",
    lambda s, *_: s.stage,
    {"follow_up": "follow_up", "dispatcher": "dispatcher"}
)

# follow_up → dispatcher
graph.add_edge("follow_up", END)

# fallback loops to dispatcher too
graph.add_edge("fallback", END)

lang_graph = graph.compile()

# ────────────── CLI test driver ──────────────
if __name__ == "__main__":
    state = ChatState()
    while True:
        state.user_message = input("👤 You: ")
        state = lang_graph.invoke(state)
