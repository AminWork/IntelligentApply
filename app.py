import chainlit as cl
import agents.openai_llm as llm
import asyncio
import io
import json
import httpx
from PyPDF2 import PdfReader
from agents.resume_parsing_agent import ResumeParsingAgent
from agents.preference_extractor_agent import PreferenceExtractorAgent, DEFAULT_PREFERENCES_JSON_STRUCTURE

client = llm.OpenAILLM()
resume_agent = ResumeParsingAgent(llm_client=client)
preference_agent = PreferenceExtractorAgent(llm_client=client)

# State constants
STATE_RECEIVE_RESUME = 1
STATE_RECEIVE_PREFERENCES = 2
STATE_RETRIEVE_POSITIONS = 3
STATE_SEND_EMAIL = 4
STATE_FOLLOW_UP = 5

async def handle_receive_resume(message: cl.Message):
    print("[STATE] handle_receive_resume")
    pdf_file_element = None
    pdf_text = None
    user_prompt = message.content

    if message.elements:
        for element in message.elements:
            if element.mime and "application/pdf" in element.mime:
                pdf_file_element = element
                if not pdf_file_element.path:
                    print("[STATE] PDF element found, but path is missing.")
                    return "Error: PDF file path not available.", STATE_RECEIVE_RESUME
                try:
                    with open(pdf_file_element.path, "rb") as f_bytes:
                        pdf_stream = io.BytesIO(f_bytes.read())
                        reader = PdfReader(pdf_stream)
                        pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
                except Exception as e:
                    print(f"[STATE] Error extracting PDF text: {e}")
                    pdf_text = f"[Error extracting PDF text: {e}]"
                break

    if not pdf_file_element:
        print("[STATE] No PDF element found in message, asking user to upload.")
        return "Please upload your résumé file (PDF) to continue.", STATE_RECEIVE_RESUME

    cl.user_session.set("pdf_file_name", pdf_file_element.name)
    cl.user_session.set("pdf_text", pdf_text)
    cl.user_session.set("pdf_prompt", user_prompt)

    print("[STATE] PDF content extracted, calling resume agent.")
    parsed_resume_json_str = await asyncio.to_thread(resume_agent.parse_resume, pdf_text or "", user_prompt)
    cl.user_session.set("parsed_resume", parsed_resume_json_str)
    
    cl.user_session.set("preferences_data", DEFAULT_PREFERENCES_JSON_STRUCTURE.copy())
    cl.user_session.set("first_preference_prompt_sent", False)

    print("[STATE] Resume parsed, moving to preferences.")
    return f"""Parsed Resume JSON:
```json
{parsed_resume_json_str}
```

Thanks for the resume! Now, could you tell me about your preferences for academic positions? For example, what type of role are you seeking (e.g., PhD, Postdoctoral Fellow, Research Scientist, Faculty position), your research fields of interest, any preferred university types or geographic locations, funding requirements, etc.?""", STATE_RECEIVE_PREFERENCES

async def handle_receive_preferences(message: cl.Message):
    print("[STATE] handle_receive_preferences")
    current_preferences = cl.user_session.get("preferences_data")
    if current_preferences is None:
        current_preferences = DEFAULT_PREFERENCES_JSON_STRUCTURE.copy()
        print("[PREFERENCES] Re-initialized preferences data as it was missing.")

    user_input_for_preferences = message.content
    print(f"[PREFERENCES] User input: {user_input_for_preferences}")

    agent_response = await asyncio.to_thread(
        preference_agent.extract_and_suggest, 
        user_input_for_preferences, 
        current_preferences
    )

    extracted_prefs = agent_response.get("extracted_preferences", DEFAULT_PREFERENCES_JSON_STRUCTURE.copy())
    is_sufficient = agent_response.get("is_sufficient", False)
    suggested_question = agent_response.get("suggested_question", "Could you tell me more about your preferences?")

    cl.user_session.set("preferences_data", extracted_prefs)
    print(f"[PREFERENCES] Updated preferences_data: {json.dumps(extracted_prefs, indent=2)}")

    if is_sufficient:
        final_preferences_json_str = json.dumps(extracted_prefs, indent=2)
        cl.user_session.set("job_preferences_json", final_preferences_json_str)
        
        print("[PREFERENCES] Preferences collection complete (sufficient). Asking for confirmation.")
        
        confirmation_prompt_message = f"""Great! I have collected the following preferences for you:
```json
{final_preferences_json_str}
```
Please type 'yes' to confirm and allow me to search for positions, or 'no' to go back and modify them."""
        await cl.Message(content=confirmation_prompt_message).send()

        # Transition to retrieve_positions state; the next user message ("yes"/"no") will be handled there.
        # Return None for the message content so on_message doesn't send an additional message.
        return None, STATE_RETRIEVE_POSITIONS
    else:
        print(f"[PREFERENCES] Preferences not yet sufficient. Asking: {suggested_question}")
        cl.user_session.set("first_preference_prompt_sent", True)
        return suggested_question, STATE_RECEIVE_PREFERENCES

async def handle_retrieve_positions(message: cl.Message):
    print("[STATE] handle_retrieve_positions")
    user_input = message.content.strip().lower() if message.content else ""

    if user_input == "yes":
        print("[RETRIEVE] User confirmed 'yes'. Proceeding to retrieve positions.")
        job_preferences_json_str = cl.user_session.get("job_preferences_json")
        if not job_preferences_json_str:
            print("[RETRIEVE] Critical: No job preferences found in session after 'yes'. Re-routing.")
            return "It seems your preferences were not saved correctly. Let's try specifying them again.", STATE_RECEIVE_PREFERENCES

        print(f"[RETRIEVE] Job preferences for embedding: {job_preferences_json_str}")

        try:
            # 1. Generate embedding for the preferences
            # Assuming the client has a method like `get_embedding`
            # You might need to adjust this based on your llm.OpenAILLM implementation
            preference_embedding = await asyncio.to_thread(client.get_embedding, job_preferences_json_str)
            
            if not preference_embedding or len(preference_embedding) != 1536:
                print(f"[RETRIEVE] Failed to generate a valid 1536-dim embedding. Embedding: {preference_embedding}")
                return "Sorry, I couldn't process your preferences to find positions at the moment. Please try again later.", STATE_RECEIVE_PREFERENCES

            # 2. Query FAISS DB
            faiss_query_payload = {
                "vector": preference_embedding,
                "k": 5  # Retrieve top 5 positions
            }
            faiss_db_url = "http://faiss-db:8080/search" 
            # Use the service name `faiss-db` as defined in docker-compose, port 8080
            
            print(f"[RETRIEVE] Querying FAISS DB at {faiss_db_url} with k=5")
            async with httpx.AsyncClient() as http_client:
                response = await http_client.post(faiss_db_url, json=faiss_query_payload, timeout=10.0)
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                faiss_results = response.json()
            
            position_ids = faiss_results.get("ids", [])
            # distances = faiss_results.get("distances", []) # We can use distances later if needed

            if not position_ids:
                print("[RETRIEVE] No position IDs found from FAISS.")
                return "I couldn't find any matching positions based on your preferences right now. You might want to try broadening your criteria.", STATE_RECEIVE_PREFERENCES

            print(f"[RETRIEVE] Retrieved position IDs from FAISS: {position_ids}")

            # For now, just return the IDs. 
            # TODO: Fetch position names/details from MongoDB using these IDs.
            positions_message = "Here are the IDs of some positions that might match your preferences:\n\n" + "\n".join([f"- ID: {pid}" for pid in position_ids])
            
            # Placeholder for actual position names
            # We will need to implement a lookup from MongoDB here
            # For example: position_names = await fetch_position_names_from_mongo(position_ids)
            # positions_message = "Here are some positions based on your preferences:\n\n" + "\n".join(position_names)

            print("[STATE] Positions retrieved, moving to send email.")
            return positions_message, STATE_SEND_EMAIL

        except httpx.HTTPStatusError as e:
            print(f"[RETRIEVE] HTTP error connecting to FAISS DB: {e}")
            return "Sorry, I'm having trouble searching for positions right now (FAISS service error). Please try again later.", STATE_RECEIVE_PREFERENCES
        except httpx.RequestError as e:
            print(f"[RETRIEVE] Request error connecting to FAISS DB: {e}")
            return "Sorry, I'm having trouble searching for positions right now (FAISS connection error). Please try again later.", STATE_RECEIVE_PREFERENCES
        except Exception as e:
            print(f"[RETRIEVE] An unexpected error occurred: {e}")
            # Log the full traceback for debugging
            import traceback
            traceback.print_exc()
            return "An unexpected error occurred while trying to find positions. Please try again.", STATE_RECEIVE_PREFERENCES
    else: # Handles "no" and any other input
        if user_input == "no":
            print("[RETRIEVE] User responded 'no'. Returning to preference collection.")
            return "Okay, please let me know what preferences you'd like to modify.", STATE_RECEIVE_PREFERENCES
        else:
            print(f"[RETRIEVE] User provided an unexpected response: '{message.content}'. Returning to preference collection.")
            # It's important to use message.content here for the raw input if user_input was empty due to no content
            return f"I understood that as a request to modify preferences, or your input was unclear ('{message.content}'). Please tell me about your preferences again.", STATE_RECEIVE_PREFERENCES

async def handle_send_email(message):
    print("[STATE] handle_send_email")
    print("[STATE] Email sent, moving to follow up.")
    return "Application email sent!", STATE_FOLLOW_UP

async def handle_follow_up(message):
    print("[STATE] handle_follow_up")
    print("[STATE] Follow-up complete, staying in follow up state.")
    return "Follow-up complete. Thank you!", STATE_FOLLOW_UP

state_handlers = {
    STATE_RECEIVE_RESUME: handle_receive_resume,
    STATE_RECEIVE_PREFERENCES: handle_receive_preferences,
    STATE_RETRIEVE_POSITIONS: handle_retrieve_positions,
    STATE_SEND_EMAIL: handle_send_email,
    STATE_FOLLOW_UP: handle_follow_up,
}

@cl.on_chat_start
async def start():
    print("[STATE] Chat started, initializing state to RECEIVE_RESUME.")
    cl.user_session.set("user_state", STATE_RECEIVE_RESUME)
    
    # Keys to clear at the start of a new chat session
    keys_to_clear = [
        "preferences_data",
        "first_preference_prompt_sent",
        "job_preferences_json",
        "parsed_resume",
        "pdf_file_name",
        "pdf_text",
        "pdf_prompt"
    ]
    
    for key in keys_to_clear:
        try:
            del cl.user_session[key]
            print(f"[SESSION] Cleared session key: {key}")
        except KeyError:
            print(f"[SESSION] Session key {key} not found, no need to clear.")
        except Exception as e:
            print(f"[SESSION] Error clearing session key {key}: {e}") # Catch any other unexpected errors
            
    await cl.Message("👋 Hello! Please upload your résumé (PDF) to get started.").send()

@cl.on_message
async def on_message(message: cl.Message):
    user_state = cl.user_session.get("user_state") or STATE_RECEIVE_RESUME
    print(f"[STATE] on_message called. Current state: {user_state}. Message elements: {message.elements}")
    handler = state_handlers.get(user_state, handle_receive_resume)
    response, next_state = await handler(message)
    print(f"[STATE] Handler returned. Next state: {next_state}")
    if response is not None:  # Only send message if there's a response
        cl.user_session.set("user_state", next_state)
        await cl.Message(content=response).send()