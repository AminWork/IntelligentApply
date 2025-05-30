from agents.openai_llm import OpenAILLM

RESUME_PARSING_SYSTEM_PROMPT = '''You are a Resume Parsing Agent. Your task is to extract structured information from a user's resume. The resume may be provided as pasted text or parsed from a document.

Your output should be a clean JSON object with the following fields:

{
  "full_name": "",
  "email": "",
  "website": "",
  "linkedin": "",
  "summary": "",
  "current_location": "",
  "education": "",
  "research_experience": [],
  "work_experience": "",
  "publication": "",
  "languages": [],
  "skills": [],
  "research_interests": [],
  "scores": "",
  "honors_awards": []
}

Follow these instructions:
- Keep field values short and precise.
- If a field is not found, leave it as an empty string or empty list.
- For publications, include at least 3 if possible.
- Only extract factual information—do not infer or hallucinate.

Ready to parse the user resume when available.'''

class ResumeParsingAgent:
    def __init__(self, llm_client=None):
        self.llm = llm_client or OpenAILLM()
        self.system_prompt = RESUME_PARSING_SYSTEM_PROMPT

    def parse_resume(self, resume_text: str, user_prompt: str = "") -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Resume:\n{resume_text}\n\nPrompt: {user_prompt}"}
        ]
        return self.llm(messages) 