import chainlit as cl
from typing import List

# Sample job record (in a real version, you'd query your database or vector index)
def search_jobs(interests: str, location: str = None, remote_only: bool = False) -> List[dict]:
    # Mock data for demonstration
    jobs = [
        {"title": "PhD in Quantum Optics", "institution": "ETH Zurich", "location": "Switzerland", "remote": False},
        {"title": "Postdoc in NLP", "institution": "MIT CSAIL", "location": "USA", "remote": True},
        {"title": "PhD in AI Alignment", "institution": "Oxford", "location": "UK", "remote": False}
    ]
    # Simple filtering logic
    return [
        job for job in jobs
        if interests.lower() in job["title"].lower()
        and (location.lower() in job["location"].lower() if location else True)
        and (job["remote"] if remote_only else True)
    ]


@cl.on_chat_start
async def start():
    await cl.Message("👋 Hello! I’m your Academic Agent Assistant.").send()
    await cl.Message("Tell me your research interests and job preferences.").send()

    # Ask for research interest
    interests = await cl.AskUserMessage(content="🔍 What is your primary research interest?").send()
    
    # Ask for location (optional)
    location = await cl.AskUserMessage(content="🌍 Any preferred country or location? (Type 'any' for no preference)").send()
    location_value = location["content"].strip()
    location_filter = None if location_value.lower() == "any" else location_value

    # Ask for remote preference
    remote_resp = await cl.AskUserMessage(content="🏠 Should I only show **remote** opportunities? (yes/no)").send()
    remote_only = remote_resp["content"].lower().strip() == "yes"

    # Search for jobs
    matches = search_jobs(interests=interests["content"], location=location_filter, remote_only=remote_only)

    if not matches:
        await cl.Message("❌ No positions found. Try adjusting your filters or keywords.").send()
        return

    await cl.Message("✅ Here are some positions you might be interested in:").send()

    for job in matches:
        await cl.Message(
            content=f"**{job['title']}**\n📍 {job['institution']} ({job['location']})\n🖥 Remote: {'Yes' if job['remote'] else 'No'}"
        ).send()
