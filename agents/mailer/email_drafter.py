"""
LLM Call 2 (New 3-Call Pipeline): Email Application Generator

Generates an email text using an ECONOMICAL LLM.
Inputs are based ONLY on the user-confirmed EXTRACTED fields for both Position and User CV,
respecting the exact field names (including typos) provided by the user for data sourcing,
and mapping them to the correctly spelled placeholders in the prompt template.
NO RAW TEXT (Position or CV) is provided to this LLM call.
Includes a list of additional attachment filenames.
"""
import os
import sys
import logging
from typing import Dict, Any, Union, List

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
        MAX_OUTPUT_TOKENS_DRAFT_EMAIL,
        MODULE_LOG_LEVEL,
        ECONOMICAL_MODEL_ID
    )
    from app_document_pipeline.llm_services.metis_gemini_client import (
        get_economical_model,
        generate_text_from_model,
        _is_genai_globally_configured_for_metis as _is_llm_service_configured,
        _genai_global_config_error_message as _llm_service_config_error
    )
    from app_document_pipeline.prompts import TEMPLATE_EMAIL_APPLICATION

    logging.basicConfig(level=MODULE_LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
else:
    # Relative imports for when this module is imported as part of the package
    from .config import (
        DEFAULT_TEMPERATURE_DRAFTING,
        MAX_OUTPUT_TOKENS_DRAFT_EMAIL,
        MODULE_LOG_LEVEL,
        ECONOMICAL_MODEL_ID
    )
    from .llm_services.metis_gemini_client import (
        get_economical_model,
        generate_text_from_model,
        _is_genai_globally_configured_for_metis as _is_llm_service_configured,
        _genai_global_config_error_message as _llm_service_config_error
    )
    from .prompts import TEMPLATE_EMAIL_APPLICATION

logger = logging.getLogger(__name__)
logger.setLevel(MODULE_LOG_LEVEL)


def _format_list_for_prompt(data_list: Union[list, str, None], default_val="N/A") -> str:
    """Helper to format a list (or already stringified list) for the prompt."""
    if data_list is None:
        return default_val
    if isinstance(data_list, list):
        # Filter out empty strings or None values before joining
        filtered_list = [str(item) for item in data_list if item and str(item).strip()]
        return ", ".join(filtered_list) if filtered_list else default_val
    if isinstance(data_list, str):
        return data_list.strip() if data_list.strip() else default_val
    return str(data_list)


def generate_application_email(
        position_extracted_fields: Dict[str, Any],
        user_cv_extracted_fields: Dict[str, Any],
        additional_attachments_filenames: List[str]
) -> str:
    """
    Generates an email text using an ECONOMICAL LLM and ONLY user-confirmed extracted data fields.

    Args:
        position_extracted_fields (Dict[str, Any]): Extracted fields for the position,
            using user's exact field names (e.g., 'position_summery', 'position_department_faculty').
        user_cv_extracted_fields (Dict[str, Any]): Extracted fields for the user's CV,
            using user's exact field names (e.g., 'user_cv_summery', 'user_research_interests').
        additional_attachments_filenames (List[str]): List of filenames for other attachments
            (derived from user_cv_extracted_fields['user_additional_file_names']).
    Returns:
        str: The generated email text, or an empty string if an error occurs.
    """
    logger.info(
        "Starting application email generation (economical model, strictly user-defined extracted fields only)...")

    if not _is_llm_service_configured:
        logger.error(f"LLM service not configured, cannot proceed: {_llm_service_config_error}")
        return ""

    if not all([position_extracted_fields, user_cv_extracted_fields]):
        logger.error("Missing position_extracted_fields or user_cv_extracted_fields.")
        return ""

    try:
        # Prepare prompt inputs strictly based on available fields and prompt placeholders
        # Sourcing from user's exact field names (including typos)
        contact_person_raw = position_extracted_fields.get('position_person_name', "Hiring Committee")

        position_contact_person_name_for_salutation_val = "Hiring Committee"
        if contact_person_raw and " " in contact_person_raw and not contact_person_raw == "Hiring Committee":
            parts = contact_person_raw.split()
            # Check if the first part is a known title
            if parts[0].lower().rstrip('.') in ["prof", "dr", "mr", "mrs", "ms", "mx"]:
                position_contact_person_name_for_salutation_val = f"{parts[0]} {parts[-1]}"
            else:  # Assume it's a full name or other title like "The Research Team"
                position_contact_person_name_for_salutation_val = contact_person_raw
        elif contact_person_raw:  # Handles single word names/titles or "Hiring Committee"
            position_contact_person_name_for_salutation_val = contact_person_raw

        # Mapping user's field names (with typos) to the correctly spelled placeholders in TEMPLATE_EMAIL_APPLICATION
        prompt_inputs = {
            # Position fields
            "position_title": position_extracted_fields.get("position_title", "N/A"),
            "position_university_name": position_extracted_fields.get("position_university_name", "N/A"),
            "position_department_faculty": position_extracted_fields.get("position_department_faculty", "N/A"),
            "position_summary": position_extracted_fields.get("position_summery", "N/A"),  # Source from user's typo
            "position_contact_person_name_for_salutation": position_contact_person_name_for_salutation_val,
            "application_deadline": position_extracted_fields.get("application_deadline", "N/A"),
            "position_keywords": _format_list_for_prompt(position_extracted_fields.get("position_keywords")),

            # User CV fields: Sourcing from user's exact field names (including typos),
            "user_fullname": user_cv_extracted_fields.get("user_fullname", "The Applicant"),
            "user_cv_summary": user_cv_extracted_fields.get("user_cv_summery", "a strong academic background."),
            # Maps to {user_cv_summary}
            "user_skills": _format_list_for_prompt(user_cv_extracted_fields.get("user_skills")),
            "user_research_interests": _format_list_for_prompt(user_cv_extracted_fields.get("user_research_interests")),
            # Maps to {user_research_interests}
            "user_scores": _format_list_for_prompt(user_cv_extracted_fields.get("user_scores")),

            # --- CORRECTED SECTION: Added missing fields ---
            "user_education": _format_list_for_prompt(user_cv_extracted_fields.get("user_education", "N/A")),
            "user_research_experience": _format_list_for_prompt(
                user_cv_extracted_fields.get("user_research_experience", "N/A")),
            # Ensure this key matches your data, e.g., "user_research_experiance" if that's the typo
            "user_work_experience": _format_list_for_prompt(
                user_cv_extracted_fields.get("user_work_experience", "N/A")),  # Ensure this key matches your data
            "user_publication": _format_list_for_prompt(user_cv_extracted_fields.get("user_publication", "N/A")),
            "user_honor_awards": _format_list_for_prompt(user_cv_extracted_fields.get("user_honor_awards", "N/A")),
            # --- END OF CORRECTION ---

            "additional_attachments_list_str": _format_list_for_prompt(additional_attachments_filenames,
                                                                       default_val="any other requested documents")
            # Changed default
        }

        formatted_prompt = TEMPLATE_EMAIL_APPLICATION.format(**prompt_inputs)
        logger.debug(f"Formatted prompt for email generation (first 500 chars):\n{formatted_prompt[:500]}...")

        economical_model_instance = get_economical_model()
        logger.info(f"Using economical model ({economical_model_instance.model_name}) for email generation.")

        email_text = generate_text_from_model(
            model_instance=economical_model_instance,
            prompt_text=formatted_prompt,
            temperature=DEFAULT_TEMPERATURE_DRAFTING,
            max_output_tokens=MAX_OUTPUT_TOKENS_DRAFT_EMAIL
        )

        if email_text:
            logger.info("Successfully generated email text.")
            logger.debug(f"Generated email text:\n{email_text}")
            return email_text.strip()
        else:
            logger.warning("Email generation resulted in empty text.")
            return ""

    except ConnectionError as e:
        logger.error(f"Connection error during email generation: {e}", exc_info=True)
        return ""
    except KeyError as e:
        logger.error(
            f"KeyError during prompt formatting for email generation: {e}. Available keys in prompt_inputs: {list(prompt_inputs.keys())}",
            exc_info=True)
        return ""
    except Exception as e:
        logger.error(f"An unexpected error occurred during email generation: {e}", exc_info=True)
        return ""


if __name__ == '__main__':
    logger.info("Running email_drafter.py (generate_application_email) directly for testing...")

    # Mock Inputs using user's exact field names from their definitive list (including typos)
    mock_position_extracted_fields = {
        "position_title": "PhD in AI Ethics",
        "position_summery": "This PhD program explores the ethical implications of artificial intelligence, focusing on fairness, accountability, and transparency in AI systems.",
        "position_university_name": "University of Morality",
        "position_department_faculty": "Department of Philosophy and AI",
        "position_location": "Virtue City, Utopia",
        "application_deadline": "2025-12-31",
        "position_person_name": "Prof. Eleanor Shellstrop",
        # Corrected from "Dr. Eleanor Shellstrop" to match salutation logic
        "position_keywords": ["AI Ethics", "Fairness", "Accountability", "Transparency", "PhD"]
    }

    mock_user_cv_extracted_fields = {
        "user_fullname": "Chidi Anagonye",
        "user_email_address": "chidi@example.com",
        "user_website": "chidi-ethics.example.com",
        "user_linkedin": "linkedin.com/in/chidianagonye",
        "user_cv_summery": "Moral philosopher with extensive research in ethics and decision-making. Eager to apply ethical frameworks to emerging AI technologies.",
        "user_education": "PhD in Moral Philosophy; MA in Ethics",
        # This field was missing in the previous prompt_inputs
        "user_research_experience": "Postdoctoral fellow in Applied Ethics; Research on Kantian deontology",
        # This field was missing
        "user_work_experience": "Lecturer in Ethics",  # This field was missing
        "user_publication": "Several papers on ethical theory and its applications.",  # This field was missing
        "user_skills": ["Ethical Analysis", "Critical Thinking", "Philosophical Writing", "Logic"],
        "user_research_interests": ["AI Ethics", "Deontology", "Existentialism"],
        "user_scores": "GRE: Verbal 165, Quant 160, Writing 5.0",
        "user_honor_awards": "Philosophy Scholar of the Year Award"  # This field was missing
    }
    mock_additional_attachments_filenames_from_user = ["transcript_chidi.pdf", "ethics_essay_sample.pdf"]

    logger.info("Mock inputs prepared for email generation test.")

    if not _is_llm_service_configured:
        logger.error(f"LLM service (metis_gemini_client) not configured, test cannot run: {_llm_service_config_error}")
    else:
        try:
            economical_model_test_instance = get_economical_model()
            logger.info(
                f"Economical LLM service ({economical_model_test_instance.model_name}) seems configured for testing.")

            generated_email = generate_application_email(
                position_extracted_fields=mock_position_extracted_fields,
                user_cv_extracted_fields=mock_user_cv_extracted_fields,
                additional_attachments_filenames=mock_additional_attachments_filenames_from_user
            )

            if generated_email:
                print(
                    "\n--- Generated Email Text (Using Economical Model & Strictly User-Defined Extracted Fields Only) ---")
                print(generated_email)
                print("--- End of Email Text ---")
            else:
                print("\nFailed to generate email text. Check logs for details.")
        except ConnectionError as e:
            logger.error(f"ConnectionError during email_drafter.py test setup or execution: {e}")
            print("\nFailed to generate email text due to ConnectionError.")
        except Exception as e:
            logger.error(f"Unexpected error during email_drafter.py test: {e}", exc_info=True)
            print("\nFailed to generate email text due to an unexpected error.")