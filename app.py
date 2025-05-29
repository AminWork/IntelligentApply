import chainlit as cl
from agents.state import ChatState
from agents.graph import lang_graph

@cl.on_chat_start
async def start():
    await cl.Message("👋 Hello! I'm your AI assistant for applying to PhD academic positions.").send()
    await cl.Message("You can upload your résumé, ask questions about our service, or let me help you find and apply to positions. Let's get started!").send()
    cl.user_session.set("state", ChatState())

@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve or initialize conversation state
    state = cl.user_session.get("state")
    if not isinstance(state, ChatState):
        if state is not None:
            state = ChatState(**dict(state))
        else:
            state = ChatState()

    # Handle file upload (if any)
    uploaded_file = None
    if message.elements:
        for element in message.elements:
            if "application/pdf" in getattr(element, "mime", ""):
                uploaded_file = element
                state.resume = {"uploaded": True, "filename": element.name}
                break

    # Update state with user message
    state.user_message = message.content

    # Run the langgraph
    state = lang_graph.invoke(state)

    # Respond based on the new state
    if state.stage == "intake":
        await cl.Message("📥 I see you're interested in applying! Please upload your résumé and let me know your preferences.").send()
    elif state.stage == "check_replies":
        await cl.Message("🔎 Checking for replies from professors...").send()
    elif state.stage == "follow_up":
        await cl.Message("🔁 Time to send polite follow-ups or schedule an interview coach.").send()
    elif state.stage == "END":
        await cl.Message("🏁 Conversation finished. Thank you for using the Academic Apply Assistant!").send()
    else:
        await cl.Message("🤖 How can I assist you today? You can upload your résumé or ask about our service.").send()

    # Save updated state
    cl.user_session.set("state", state)
