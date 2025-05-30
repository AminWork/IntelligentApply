# phd_apply_agent / app_document_pipeline / follow_up_email_drafter.py

"""
LLM Call 3 (New 3-Call Pipeline): Follow-up Email Drafter

Generates a follow-up email text using an ECONOMICAL LLM.
Inputs are based on details of the original application and context.
"""
import os
import sys
import logging
from typing import Dict, Any
from datetime import datetime, timedelta  # For calculating time_elapsed if needed

# --- Dynamic sys.path modification for direct script execution ---
if __name__ == '__main__':
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    app_doc_pipeline_dir = os.path.dirname(current_script_dir)
    project_root_dir = os.path.dirname(app_doc_pipeline_dir)
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)
# --- End dynamic sys.path modification ---

# Conditional import for config variables
if __name__ == '__main__':
    from app_document_pipeline.config import (
        DEFAULT_TEMPERATURE_DRAFTING,
        MAX_OUTPUT_TOKENS_FOLLOWUP_EMAIL,
        MODULE_LOG_LEVEL,
        ECONOMICAL_MODEL_ID  # For logging
    )
    from app_document_pipeline.llm_services.metis_gemini_client import (
        get_economical_model,  # Using economical model
        generate_text_from_model,
        _is_genai_globally_configured_for_metis as _is_llm_service_configured,
        _genai_global_config_error_message as _llm_service_config_error
    )
    from app_document_pipeline.prompts import TEMPLATE_EMAIL_FOLLOWUP

    logging.basicConfig(level=MODULE_LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
else:
    from .config import (
        DEFAULT_TEMPERATURE_DRAFTING,
        MAX_OUTPUT_TOKENS_FOLLOWUP_EMAIL,
        MODULE_LOG_LEVEL,
        ECONOMICAL_MODEL_ID
    )
    from .llm_services.metis_gemini_client import (
        get_economical_model,
        generate_text_from_model,
        _is_genai_globally_configured_for_metis as _is_llm_service_configured,
        _genai_global_config_error_message as _llm_service_config_error
    )
    from .prompts import TEMPLATE_EMAIL_FOLLOWUP

logger = logging.getLogger(__name__)
logger.setLevel(MODULE_LOG_LEVEL)


def draft_follow_up_email(
        original_email_subject: str,
        original_email_recipient_name: str,  # e.g., "Prof. Smith" or "Hiring Committee"
        position_title: str,
        application_date_str: str,  # Should be a string like "YYYY-MM-DD" or "Month DD, YYYY"
        user_fullname: str,
        previous_email_body_snippet: str,
        time_elapsed_str: str  # e.g., "one week", "10 days", "two weeks"
) -> str:
    """
    Generates a follow-up email text using an ECONOMICAL LLM.

    Args:
        original_email_subject (str): Subject of the initial email sent.
        original_email_recipient_name (str): Name/title used in the salutation of the original email.
        position_title (str): The title of the PhD position.
        application_date_str (str): The date the original application was sent (as a string).
        user_fullname (str): The applicant's full name.
        previous_email_body_snippet (str): A short snippet of the original email's body for context.
        time_elapsed_str (str): A human-readable string describing time elapsed.

    Returns:
        str: The generated follow-up email text, or an empty string if an error occurs.
    """
    logger.info(f"Starting follow-up email draft generation for position: {position_title}")

    if not _is_llm_service_configured:
        logger.error(f"LLM service not configured, cannot proceed: {_llm_service_config_error}")
        return ""

    if not all([original_email_subject, original_email_recipient_name, position_title,
                application_date_str, user_fullname, time_elapsed_str]):
        logger.error("Missing one or more critical inputs for follow-up email generation.")
        return ""

    try:
        prompt_inputs = {
            "original_email_subject": original_email_subject,
            "original_email_recipient_name": original_email_recipient_name,
            "position_title": position_title,
            "application_date": application_date_str,
            "time_elapsed": time_elapsed_str,
            "user_fullname": user_fullname,
            "previous_email_body_snippet": previous_email_body_snippet[:200]  # Limit snippet length
        }

        formatted_prompt = TEMPLATE_EMAIL_FOLLOWUP.format(**prompt_inputs)
        logger.debug(f"Formatted prompt for follow-up email (first 500 chars):\n{formatted_prompt[:500]}...")

        economical_model_instance = get_economical_model()
        logger.info(f"Using economical model ({economical_model_instance.model_name}) for follow-up email generation.")

        follow_up_email_text = generate_text_from_model(
            model_instance=economical_model_instance,
            prompt_text=formatted_prompt,
            temperature=DEFAULT_TEMPERATURE_DRAFTING,
            max_output_tokens=MAX_OUTPUT_TOKENS_FOLLOWUP_EMAIL
        )

        if follow_up_email_text:
            logger.info("Successfully generated follow-up email text.")
            logger.debug(f"Generated follow-up email text:\n{follow_up_email_text}")
            return follow_up_email_text.strip()
        else:
            logger.warning("Follow-up email generation resulted in empty text.")
            return ""

    except ConnectionError as e:
        logger.error(f"Connection error during follow-up email generation: {e}", exc_info=True)
        return ""
    except KeyError as e:
        logger.error(
            f"Missing key in prompt inputs for follow-up email: {e}. Available keys: {list(prompt_inputs.keys())}",
            exc_info=True)
        return ""
    except Exception as e:
        logger.error(f"An unexpected error occurred during follow-up email generation: {e}", exc_info=True)
        return ""


if __name__ == '__main__':
    logger.info("Running follow_up_email_drafter.py directly for testing...")

    # Mock Inputs
    mock_original_subject = "PhD Application: Research in AI Ethics - Chidi Anagonye"
    mock_recipient_name = "Prof. Eleanor Shellstrop"  # Or "Hiring Committee"
    mock_position_title = "PhD in AI Ethics"
    mock_app_date = "May 20, 2025"
    mock_user_fullname = "Chidi Anagonye"
    mock_prev_email_snippet = "I am writing to express my keen interest in the PhD position in AI Ethics... My CV and Cover Letter are attached."
    mock_time_elapsed = "two weeks"

    logger.info("Mock inputs prepared for follow-up email test.")

    if not _is_llm_service_configured:
        logger.error(f"LLM service (metis_gemini_client) not configured, test cannot run: {_llm_service_config_error}")
    else:
        try:
            economical_model_test_instance = get_economical_model()
            logger.info(
                f"Economical LLM service ({economical_model_test_instance.model_name}) seems configured for testing.")

            generated_follow_up_email = draft_follow_up_email(
                original_email_subject=mock_original_subject,
                original_email_recipient_name=mock_recipient_name,
                position_title=mock_position_title,
                application_date_str=mock_app_date,
                user_fullname=mock_user_fullname,
                previous_email_body_snippet=mock_prev_email_snippet,
                time_elapsed_str=mock_time_elapsed
            )

            if generated_follow_up_email:
                print("\n--- Generated Follow-up Email Text ---")
                print(generated_follow_up_email)
                print("--- End of Follow-up Email Text ---")
            else:
                print("\nFailed to generate follow-up email text. Check logs for details.")
        except ConnectionError as e:
            logger.error(f"ConnectionError during follow_up_email_drafter.py test setup or execution: {e}")
            print("\nFailed to generate follow-up email text due to ConnectionError.")
        except Exception as e:
            logger.error(f"Unexpected error during follow_up_email_drafter.py test: {e}", exc_info=True)
            print("\nFailed to generate follow-up email text due to an unexpected error.")
