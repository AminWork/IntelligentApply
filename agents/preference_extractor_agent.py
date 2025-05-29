from agents.openai_llm import OpenAILLM
import json

PREFERENCE_EXTRACTOR_SYSTEM_PROMPT = '''
You are a Preference Extraction Assistant. Your goal is to help a user specify their job preferences for an academic position search.
You will be given the user's text input and the current state of their preferences (which might be partially filled or empty).
Your task is to:
1.  Parse the user's latest input to identify any job preferences they've mentioned.
2.  Update the provided preference data with any new information from the user's input. Merge intelligently; don't just overwrite if the user provides partial updates.
3.  Determine if sufficient information has been collected to proceed with a job search. "Sufficient" generally means 'position_type' and 'fields_of_interest' are filled. Other fields are valuable but may not block proceeding if these primary ones are known.
4.  Respond with a JSON object ONLY, with NO other text before or after the JSON. The JSON structure MUST be:

{
  "extracted_preferences": {
    "position_type": "", 
    "fields_of_interest": [], 
    "preferred_locations": [], 
    "excluded_locations": [],
    "funding_required": null, 
    "citizenship_constraints": "", 
    "start_date": "", 
    "important_keywords": [],
    "dealbreakers": []
  },
  "is_sufficient": false, 
  "suggested_question": "What type of position are you primarily looking for (e.g., PhD, Postdoc)?"
}

Guidelines for filling `extracted_preferences`:
- Adhere strictly to the data types shown (string, list of strings, boolean/null).
- For lists (e.g., `fields_of_interest`, `preferred_locations`), append new items if the user provides more, don't just replace the list unless the user explicitly indicates they are replacing all previous entries for that field. If the user says "none" or "not applicable" for a list field, use an empty list `[]`.
- For `funding_required`: if user says "require funding", "need funding", or "yes" (in context of funding), set to `true`. If "self-funded", "don't need funding", or "no", set to `false`. If unclear or not mentioned, and not previously set, keep as `null`.
- If a field is not mentioned by the user and not already in current_preferences, leave it as its default (empty string, empty list, or null for boolean).

Guidelines for `is_sufficient`:
- `position_type` and `fields_of_interest` are highly important. If both are filled, you can lean towards `is_sufficient: true`.
- If one or both of these are missing, `is_sufficient` should generally be `false`.
- Use your judgment for other fields. If many secondary fields are missing but the primary ones are present, you might still set `is_sufficient: true`.

Guidelines for `suggested_question`:
- If `is_sufficient` is `false`, formulate a polite and concise question targeting one or two of the most important missing preferences.
- Example: If `position_type` is missing: "What type of position are you looking for (e.g., PhD, Postdoc)?"
- Example: If `fields_of_interest` is missing: "What are your main fields of interest or research areas?"

You will be given:
1.  Current collected preferences (JSON string, or an empty dict if none yet).
2.  User's latest message.

Make sure your entire response is a single JSON object.
'''

DEFAULT_PREFERENCES_JSON_STRUCTURE = {
  "position_type": "",
  "fields_of_interest": [],
  "preferred_locations": [],
  "excluded_locations": [],
  "funding_required": None, # Using None for not-yet-specified boolean
  "citizenship_constraints": "",
  "start_date": "",
  "important_keywords": [],
  "dealbreakers": []
}


class PreferenceExtractorAgent:
    def __init__(self, llm_client=None):
        self.llm = llm_client or OpenAILLM()
        self.system_prompt = PREFERENCE_EXTRACTOR_SYSTEM_PROMPT

    def extract_and_suggest(self, user_text: str, current_preferences: dict | None = None) -> dict:
        if current_preferences is None:
            current_preferences = DEFAULT_PREFERENCES_JSON_STRUCTURE.copy()
        
        current_preferences_json = json.dumps(current_preferences, indent=2)
        
        prompt_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Current Preferences:\n```json\n{current_preferences_json}\n```\n\nUser's latest message:\n```\n{user_text}\n```\n\nYour JSON response:"}
        ]
        
        try:
            print(f"[PreferenceExtractorAgent] Sending to LLM. Current Prefs: {current_preferences_json}, User Text: {user_text}")
            response_str = self.llm(prompt_messages)
            print(f"[PreferenceExtractorAgent] LLM raw response: {response_str}")
            
            # Ensure the response is treated as a JSON string
            # The LLM might sometimes add explanations or markdown around the JSON
            json_response_cleaned = response_str
            if "```json" in json_response_cleaned:
                json_response_cleaned = json_response_cleaned.split("```json")[1]
            if "```" in json_response_cleaned:
                json_response_cleaned = json_response_cleaned.split("```")[0]
            json_response_cleaned = json_response_cleaned.strip()

            parsed_response = json.loads(json_response_cleaned)
            
            # Validate structure a bit, or ensure defaults for missing top-level keys
            if "extracted_preferences" not in parsed_response:
                parsed_response["extracted_preferences"] = current_preferences # Fallback
            if "is_sufficient" not in parsed_response:
                 parsed_response["is_sufficient"] = False # Default to not sufficient if key missing
            if "suggested_question" not in parsed_response:
                parsed_response["suggested_question"] = "Could you tell me a bit more about your preferences?"


            # Ensure all keys from DEFAULT_PREFERENCES_JSON_STRUCTURE are in extracted_preferences
            # This is important if the LLM omits some keys it deems "empty"
            final_extracted_prefs = DEFAULT_PREFERENCES_JSON_STRUCTURE.copy()
            final_extracted_prefs.update(parsed_response.get("extracted_preferences", {}))
            parsed_response["extracted_preferences"] = final_extracted_prefs

            print(f"[PreferenceExtractorAgent] Parsed LLM response: {parsed_response}")
            return parsed_response
        except json.JSONDecodeError as e:
            print(f"[PreferenceExtractorAgent] Error decoding LLM JSON response: {e}")
            print(f"[PreferenceExtractorAgent] Faulty response string: {response_str}")
            # Fallback response in case of JSON error
            return {
                "extracted_preferences": current_preferences,
                "is_sufficient": False,
                "suggested_question": "I had a little trouble understanding that. Could you please rephrase or tell me more about your preferences?"
            }
        except Exception as e:
            print(f"[PreferenceExtractorAgent] Unexpected error: {e}")
            import traceback
            print(traceback.format_exc())
            return {
                "extracted_preferences": current_preferences,
                "is_sufficient": False,
                "suggested_question": "Sorry, an unexpected error occurred while processing your preferences. Could you try again?"
            } 