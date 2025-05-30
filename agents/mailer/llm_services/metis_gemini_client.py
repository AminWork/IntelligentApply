# phd_apply_agent/app_document_pipeline/llm_services/metis_gemini_client.py

"""
Provides direct client access to Google Gemini models via the Metis AI service,
using the 'google-generativeai' library.
"""
import os
import sys
import google.generativeai as genai
from google.api_core.client_options import ClientOptions
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- Dynamic sys.path modification for direct script execution ---
if __name__ == '__main__':
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    app_doc_pipeline_dir = os.path.dirname(current_script_dir)
    project_root_dir = os.path.dirname(app_doc_pipeline_dir)
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)
# --- End dynamic sys.path modification ---

if __name__ == '__main__':
    from app_document_pipeline.config import (
        METIS_API_KEY, METIS_API_BASE_URL,
        ECONOMICAL_MODEL_ID, POWERFUL_MODEL_ID,  # These will now reflect the test change
        DEFAULT_TEMPERATURE_DRAFTING, DEFAULT_TEMPERATURE_REFINEMENT,
        MAX_OUTPUT_TOKENS_DRAFT_EMAIL, MAX_OUTPUT_TOKENS_REFINE_EMAIL
    )
else:
    from ..config import (
        METIS_API_KEY, METIS_API_BASE_URL,
        ECONOMICAL_MODEL_ID, POWERFUL_MODEL_ID,
    )

_is_genai_globally_configured_for_metis = False
_genai_global_config_error_message = None

try:
    can_configure_genai = False
    if METIS_API_KEY and METIS_API_KEY != "YOUR_METIS_API_KEY_PLACEHOLDER":
        can_configure_genai = True

    if can_configure_genai:
        genai.configure(
            api_key=METIS_API_KEY,
            transport='rest',
            client_options=ClientOptions(api_endpoint=METIS_API_BASE_URL)
        )
        print(
            f"[Metis Gemini Client] Global 'google.generativeai' (genai) configured for Metis endpoint: {METIS_API_BASE_URL}")
        _is_genai_globally_configured_for_metis = True
    else:
        _genai_global_config_error_message = "[Metis Gemini Client WARNING] METIS_API_KEY is missing or a placeholder in config. Global 'genai' configuration for Metis skipped. LLM calls will fail."
        print(_genai_global_config_error_message)

except Exception as e:
    _genai_global_config_error_message = f"[Metis Gemini Client ERROR] Failed to configure global 'genai' for Metis: {e}"
    print(_genai_global_config_error_message)


def get_model_instance(model_id: str) -> genai.GenerativeModel:
    if not _is_genai_globally_configured_for_metis:
        error_message = _genai_global_config_error_message or "Global 'google.generativeai' (genai) is not configured for Metis. Cannot create model instance."
        raise ConnectionError(error_message)
    try:
        # Using BLOCK_NONE for all safety categories for debugging purposes.
        # Revert to stricter settings for production.
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        model = genai.GenerativeModel(model_id, safety_settings=safety_settings)
        return model
    except Exception as e:
        print(f"[Metis Gemini Client ERROR] Failed to create GenerativeModel instance for {model_id} via Metis: {e}")
        raise ConnectionError(f"Failed to initialize model {model_id} via Metis: {e}")


def generate_text_from_model(
        model_instance: genai.GenerativeModel,
        prompt_text: str,
        temperature: float,
        max_output_tokens: int
) -> str:
    if not model_instance:
        raise ValueError("Model instance cannot be None.")

    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens
    )
    try:
        response = model_instance.generate_content(
            contents=[{"parts": [{"text": prompt_text}], "role": "user"}],
            generation_config=generation_config
        )

        generated_text = ""
        if hasattr(response, 'text') and response.text:
            generated_text = response.text
        elif hasattr(response, 'parts') and response.parts:
            generated_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))

        if not generated_text:
            print(
                f"[Metis Gemini Client WARNING] Response from model '{model_instance.model_name}' was empty or content could not be extracted.")
            if hasattr(response, 'prompt_feedback'):
                print(f"  Prompt Feedback: {response.prompt_feedback}")
            if response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    print(
                        f"  Candidate Finish Reason: {candidate.finish_reason.name} ({candidate.finish_reason.value})")  # .name gives string, .value gives int
                if hasattr(candidate, 'safety_ratings'):
                    print(f"  Candidate Safety Ratings: {candidate.safety_ratings}")
            else:
                print(f"  No candidates found in the response.")
            print(f"  Full response object for debugging: {response}")  # Print entire response object

        return generated_text

    except Exception as e:
        print(f"[Metis Gemini Client ERROR] Error during text generation with model '{model_instance.model_name}': {e}")
        # Attempt to print response even on error, if it exists
        if 'response' in locals() and response:
            print(f"  Response object at time of error (if available): {response}")
        return ""


def get_economical_model() -> genai.GenerativeModel:
    return get_model_instance(ECONOMICAL_MODEL_ID)


def get_powerful_model() -> genai.GenerativeModel:
    return get_model_instance(POWERFUL_MODEL_ID)


if __name__ == "__main__":
    print("\nTesting Direct Metis Gemini Client (metis_gemini_client.py)...")

    # Config variables are imported via the conditional import at the top

    if not _is_genai_globally_configured_for_metis:
        print(f"Skipping direct LLM test: {_genai_global_config_error_message}")
    else:
        test_passed_eco = False
        test_passed_pow = False

        # Test Economical Model
        print(f"\n--- Testing Economical Model ({ECONOMICAL_MODEL_ID}) ---")
        try:
            eco_model = get_economical_model()
            print(f"Economical model instance created. Testing generation...")
            prompt_eco = "Briefly, what is a PhD? Respond in less than 30 words."
            response_text_eco = generate_text_from_model(
                eco_model, prompt_eco, DEFAULT_TEMPERATURE_DRAFTING, MAX_OUTPUT_TOKENS_DRAFT_EMAIL
            )
            print(f"Economical Model Response: {response_text_eco}")
            if response_text_eco and len(response_text_eco) > 5:
                test_passed_eco = True
        except ConnectionError as e:
            print(f"ConnectionError during economical model test: {e}")
        except Exception as e:
            print(f"Unexpected error during economical model test: {e}")

        # Test Powerful Model
        print(
            f"\n--- Testing Powerful Model ({POWERFUL_MODEL_ID}) ---")
        try:
            pow_model = get_powerful_model()
            print(f"Powerful model instance created. Testing generation...")
            prompt_pow = "WHAT IS YOUR NAME?"
            response_text_pow = generate_text_from_model(
                pow_model, prompt_pow, DEFAULT_TEMPERATURE_REFINEMENT, MAX_OUTPUT_TOKENS_REFINE_EMAIL
            )
            print(f"Powerful Model Response: {response_text_pow}")
            if response_text_pow and len(response_text_pow) > 5:
                test_passed_pow = True
        except ConnectionError as e:
            print(f"ConnectionError during powerful model test: {e}")
        except Exception as e:
            print(f"Unexpected error during powerful model test: {e}")

        print("\n--- Test Summary ---")
        print(f"Economical LLM Test Passed: {test_passed_eco}")
        print(f"Powerful LLM Test Passed: {test_passed_pow}")
        if test_passed_eco and test_passed_pow:
            print("Both model instance creation and basic generation tests seem OK.")
        else:
            print(
                "One or more model tests failed. Further checks needed (API key, endpoint, model ID validity via Metis, safety settings impact).")

