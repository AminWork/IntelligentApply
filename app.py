import chainlit as cl
from agents.state import ChatState
from agents.graph import lang_graph

@cl.on_chat_start
async def start():
    await cl.Message("👋 Hello! I'm your AI assistant for applying to PhD academic positions.").send()
    await cl.Message("You can upload your résumé, ask questions about our service, or let me help you find and apply to positions. Let's get started!").send()
    cl.user_session.set("state", ChatState())

@cl.on_message
async def on_message(msg: cl.Message):
    state: ChatState = cl.user_session.get("state")
    if not isinstance(state, ChatState):
        if state is not None:
            state = ChatState(**dict(state))
        else:
            state = ChatState()
    state.user_message = msg.content
    updated_state = lang_graph.invoke(state)
    # Ensure updated_state is a ChatState instance
    if not isinstance(updated_state, ChatState):
        updated_state = ChatState(**dict(updated_state))
    cl.user_session.set("state", updated_state)

    # Show LLM or fallback response if available
    # if hasattr(updated_state, "reply") and updated_state.reply is not None:
    #     await cl.Message(str(updated_state.reply)).send()
    await cl.Message(str(updated_state.llm_response)).send()
    # else:
    #     await cl.Message(f"Moved to stage: {getattr(updated_state, 'stage', 'unknown')}").send()
