# phd_apply_agent/app_document_pipeline/cover_letter_drafter.py

"""
LLM Call 1 (New 3-Call Pipeline): Cover Letter Generator

Generates a full LaTeX cover letter using an ECONOMICAL LLM.
Inputs are based ONLY on the user-confirmed EXTRACTED fields for both Position and User CV,
respecting the exact field names (including typos) provided by the user for data sourcing,
and mapping them to the correctly spelled placeholders in the prompt template.
NO RAW TEXT (Position or CV) is provided to this LLM call.
"""
import os
import sys
import logging
from typing import Dict, Any, Union
from datetime import datetime  # For current_date

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
    from mailer.config import (
        DEFAULT_TEMPERATURE_DRAFTING,
        MAX_OUTPUT_TOKENS_DRAFT_CL_BODY,
        MODULE_LOG_LEVEL,
        ECONOMICAL_MODEL_ID
    )
    from mailer.llm_services.metis_gemini_client import (
        get_economical_model,
        generate_text_from_model,
        _is_genai_globally_configured_for_metis as _is_llm_service_configured,
        _genai_global_config_error_message as _llm_service_config_error
    )
    from mailer.prompts import TEMPLATE_COVER_LETTER

    logging.basicConfig(level=MODULE_LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
else:
    from .config import (
        DEFAULT_TEMPERATURE_DRAFTING,
        MAX_OUTPUT_TOKENS_DRAFT_CL_BODY,
        MODULE_LOG_LEVEL,
        ECONOMICAL_MODEL_ID
    )
    from .llm_services.metis_gemini_client import (
        get_economical_model,
        generate_text_from_model,
        _is_genai_globally_configured_for_metis as _is_llm_service_configured,
        _genai_global_config_error_message as _llm_service_config_error
    )
    from .prompts import TEMPLATE_COVER_LETTER

logger = logging.getLogger(__name__)
logger.setLevel(MODULE_LOG_LEVEL)


def _format_list_for_prompt(data_list: Union[list, str, None], default_val="N/A") -> str:
    """Helper to format a list (or already stringified list) for the prompt."""
    if data_list is None:
        return default_val
    if isinstance(data_list, list):
        filtered_list = [str(item) for item in data_list if item and str(item).strip()]
        return ", ".join(filtered_list) if filtered_list else default_val
    if isinstance(data_list, str):
        return data_list.strip() if data_list.strip() else default_val
    return str(data_list)


def _escape_latex_special_chars(text: str) -> str:
    """Escapes LaTeX special characters in a given text string."""
    if not isinstance(text, str):
        logger.warning(f"Attempted to escape non-string value: {text} (type: {type(text)}). Returning as string.")
        return str(text)

    chars = {
        '\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '&': r'\&',
        '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
        '^': r'\^{}', '~': r'\textasciitilde{}', '<': r'\textless{}', '>': r'\textgreater{}',
    }
    escaped_text = text
    for char, replacement in chars.items():
        escaped_text = escaped_text.replace(char, replacement)
    return escaped_text


def generate_cover_letter_latex(
        position_extracted_fields: Dict[str, Any],
        user_cv_extracted_fields: Dict[str, Any]
) -> str:
    """
    Generates a full LaTeX cover letter using an ECONOMICAL LLM and
    ONLY user-confirmed extracted fields (no raw texts).

    Args:
        position_extracted_fields (Dict[str, Any]): Extracted fields for the position.
            Expected keys (user's exact names): 'position_title', 'position_summery',
                           'position_university_name', 'position_department_faculty',
                           'position_location', 'application_deadline',
                           'position_person_name', 'position_keywords'.
        user_cv_extracted_fields (Dict[str, Any]): Extracted fields for the user's CV.
            Expected keys (user's exact names): 'user_fullname', 'user_email_address',
                           'user_website', 'user_linkedin', 'user_cv_summery',
                           'user_education', 'user_research_experience', 'user_work_experience',
                           'user_publication', 'user_skills', 'user_research_interests',
                           'user_scores', 'user_honor_awards'.
    Returns:
        str: The generated full LaTeX cover letter string, or an empty string if an error occurs.
    """
    logger.info(
        "Starting full LaTeX cover letter generation (economical model, strictly user-defined extracted fields only)...")

    if not _is_llm_service_configured:
        logger.error(f"LLM service not configured, cannot proceed: {_llm_service_config_error}")
        return ""

    if not all([position_extracted_fields, user_cv_extracted_fields]):
        logger.error("Missing position_extracted_fields or user_cv_extracted_fields.")
        return ""

    try:
        # --- Prepare inputs for the prompt strictly based on available fields and prompt placeholders ---
        # User details - Sourcing from user's exact field names (including typos)
        user_fullname_val = _escape_latex_special_chars(user_cv_extracted_fields.get("user_fullname", "The Candidate"))
        user_email_address_val = _escape_latex_special_chars(user_cv_extracted_fields.get("user_email_address", "N/A"))
        user_website_val = _escape_latex_special_chars(user_cv_extracted_fields.get("user_website", ""))
        user_linkedin_val = _escape_latex_special_chars(user_cv_extracted_fields.get("user_linkedin", ""))
        user_cv_summery_val = _escape_latex_special_chars(
            user_cv_extracted_fields.get("user_cv_summery", "N/A"))  # User's spelling
        user_skills_val = _escape_latex_special_chars(
            _format_list_for_prompt(user_cv_extracted_fields.get("user_skills")))
        user_research_interests_val = _escape_latex_special_chars(
            _format_list_for_prompt(user_cv_extracted_fields.get("user_research_interests")))  # User's spelling
        user_scores_val = _escape_latex_special_chars(
            _format_list_for_prompt(user_cv_extracted_fields.get("user_scores")))
        user_education_val = _escape_latex_special_chars(
            _format_list_for_prompt(user_cv_extracted_fields.get("user_education")))
        user_research_experience_val = _escape_latex_special_chars(
            _format_list_for_prompt(user_cv_extracted_fields.get("user_research_experience")))  # User's spelling
        user_work_experience_val = _escape_latex_special_chars(
            _format_list_for_prompt(user_cv_extracted_fields.get("user_work_experience")))  # User's spelling
        user_publication_val = _escape_latex_special_chars(
            _format_list_for_prompt(user_cv_extracted_fields.get("user_publication")))
        user_honor_awards_val = _escape_latex_special_chars(
            _format_list_for_prompt(user_cv_extracted_fields.get("user_honor_awards")))

        current_date_for_letter_str = datetime.now().strftime("%B %d, %Y")

        # Position details - Sourcing from user's exact field names (including typos)
        pos_contact_person_raw = position_extracted_fields.get("position_person_name", "Hiring Committee")
        recipient_name_title_formatted_val = _escape_latex_special_chars(pos_contact_person_raw)

        salutation_derived = "Hiring Committee"
        if pos_contact_person_raw and pos_contact_person_raw != "Hiring Committee":
            parts = pos_contact_person_raw.split()
            if parts[0].lower().rstrip('.') in ["prof", "dr", "mr", "mrs", "ms", "mx"]:
                salutation_derived = f"{parts[0]} {parts[-1]}"
            else:
                salutation_derived = pos_contact_person_raw
        recipient_salutation_formatted_val = _escape_latex_special_chars(salutation_derived)

        # Mapping user's field names (with typos) to the correctly spelled placeholders in TEMPLATE_COVER_LETTER
        prompt_inputs = {
            "user_fullname": user_fullname_val,
            "user_email_address": user_email_address_val,
            "user_website": user_website_val,
            "user_linkedin": user_linkedin_val,
            "user_cv_summary": user_cv_summery_val,  # Placeholder is {user_cv_summary}
            "user_skills": user_skills_val,
            "user_research_interests": user_research_interests_val,  # Placeholder is {user_research_interests}
            "user_scores": user_scores_val,
            "user_education": user_education_val,  # Placeholder is {user_education}
            "user_research_experience": user_research_experience_val,  # Placeholder is {user_research_experience}
            "user_work_experience": user_work_experience_val,  # Placeholder is {user_work_experience}
            "user_publication": user_publication_val,  # Placeholder is {user_publication}
            "user_honor_awards": user_honor_awards_val,

            "current_date_for_letter": current_date_for_letter_str,

            "recipient_name_title_formatted": recipient_name_title_formatted_val,
            "recipient_salutation_formatted": recipient_salutation_formatted_val,

            "position_title": _escape_latex_special_chars(position_extracted_fields.get("position_title", "N/A")),
            "position_university_name": _escape_latex_special_chars(
                position_extracted_fields.get("position_university_name", "N/A")),
            "position_department_faculty": _escape_latex_special_chars(
                position_extracted_fields.get("position_department_faculty", "N/A")),  # Source from user's typo
            "position_location": _escape_latex_special_chars(position_extracted_fields.get("position_location", "N/A")),
            # Source from user's typo
            "position_person_name": _escape_latex_special_chars(
                position_extracted_fields.get("position_person_name", "N/A")),  # For context section in prompt
            "application_deadline": position_extracted_fields.get("application_deadline", "N/A"),
            "position_keywords": _escape_latex_special_chars(
                _format_list_for_prompt(position_extracted_fields.get("position_keywords"))),
            "position_summary": _escape_latex_special_chars(position_extracted_fields.get("position_summery", "N/A")),
            # Source from user's typo
        }

        formatted_prompt = TEMPLATE_COVER_LETTER.format(**prompt_inputs)
        logger.debug(f"Formatted prompt for LaTeX cover letter (first 500 chars):\n{formatted_prompt[:500]}...")

        economical_model_instance = get_economical_model()
        logger.info(f"Using economical model ({economical_model_instance.model_name}) for LaTeX cover letter.")

        full_latex_output = generate_text_from_model(
            model_instance=economical_model_instance,
            prompt_text=formatted_prompt,
            temperature=DEFAULT_TEMPERATURE_DRAFTING,
            max_output_tokens=MAX_OUTPUT_TOKENS_DRAFT_CL_BODY
        )

        if full_latex_output and "\\documentclass" in full_latex_output and "\\end{document}" in full_latex_output:
            logger.info("Successfully generated full LaTeX cover letter.")
            logger.debug(f"Generated full LaTeX cover letter (first 1000 chars):\n{full_latex_output[:1000]}...")
            return full_latex_output.strip()
        else:
            logger.warning(
                f"LaTeX cover letter generation resulted in invalid or empty text. Output snippet: {str(full_latex_output)[:500]}...")
            return ""

    except ConnectionError as e:
        logger.error(f"Connection error during LaTeX cover letter generation: {e}", exc_info=True)
        return ""
    except KeyError as e:
        logger.error(
            f"KeyError during prompt formatting for LaTeX cover letter: {e}. Available keys in prompt_inputs: {list(prompt_inputs.keys())}",
            exc_info=True)
        return ""
    except Exception as e:
        logger.error(f"An unexpected error occurred during LaTeX cover letter generation: {e}", exc_info=True)
        return ""


if __name__ == '__main__':
    logger.info("Running cover_letter_drafter.py (generate_cover_letter_latex) directly for testing...")

    # Mock Inputs using user's exact field names (including typos)
    mock_position_extracted_fields = {
        "position_title": "PhD in Computational Astrophysics",
        "position_summery": "This program focuses on developing simulations of galactic dynamics and star formation. Strong analytical and programming skills desired.",
        # User's spelling
        "position_university_name": "Galaxy University",
        "position_department_faculty": "Department of Astronomy and Physics",  # User's spelling
        "position_location": "Stellarton, Cosmos",  # User's spelling
        "application_deadline": "2025-11-15",
        "position_person_name": "Dr. Celeste Nova",
        "position_keywords": ["Astrophysics", "Simulations", "PhD", "Computational Science"],
    }
    mock_user_cv_extracted_fields = {
        "user_fullname": "Orion Stargazer",
        "user_email_address": "orion.s@example.com",
        "user_website": "orionstargazer.dev",
        "user_linkedin": "linkedin.com/in/orionstargazer",
        "user_cv_summery": "Aspiring astrophysicist with a Master's in Physics, skilled in Python and data modeling. Passionate about cosmology.",
        # User's spelling
        "user_education": "MSc Physics (Astrophysics focus); BSc Physics",
        "user_research_experience": "Thesis on dark matter simulations; Research assistant for stellar evolution project",
        # User's spelling
        "user_work_experience": "Planetarium presenter",  # User's spelling
        "user_publication": "Co-author on conference paper: 'Simulating Early Universe Expansion'",
        "user_skills": ["Python", "C++", "Data Analysis", "Numerical Methods", "Scientific Communication"],
        "user_research_interests": ["Cosmology", "Galactic Dynamics", "Dark Matter"],  # User's spelling
        "user_scores": "GRE: Q168 V165 AWA5.0; TOEFL: 110",
        "user_honor_awards": "Physics Department Scholarship; Best Poster Award at AstroMeet 2023"
    }

    logger.info("Mock inputs prepared for cover letter generation test.")

    if not _is_llm_service_configured:
        logger.error(f"LLM service (metis_gemini_client) not configured, test cannot run: {_llm_service_config_error}")
    else:
        try:
            economical_model_test_instance = get_economical_model()
            logger.info(
                f"Economical LLM service ({economical_model_test_instance.model_name}) seems configured for testing.")

            generated_latex_cl = generate_cover_letter_latex(
                position_extracted_fields=mock_position_extracted_fields,
                user_cv_extracted_fields=mock_user_cv_extracted_fields
            )

            if generated_latex_cl:
                print("\n--- Generated Full LaTeX Cover Letter (Economical Model, Strictly Extracted Fields Only) ---")
                print(generated_latex_cl[:10500] + "\n...")
                if "\\end{document}" not in generated_latex_cl[-35:]:
                    print("WARNING: Output might be truncated or incomplete LaTeX. Check full output if possible.")
                print("--- End of Generated LaTeX Cover Letter ---")
            else:
                print("\nFailed to generate full LaTeX cover letter. Check logs.")
        except ConnectionError as e:
            logger.error(f"ConnectionError during cover_letter_drafter.py test setup or execution: {e}")
            print("\nFailed to generate LaTeX cover letter due to ConnectionError.")
        except Exception as e:
            logger.error(f"Unexpected error during cover_letter_drafter.py test: {e}", exc_info=True)
            print("\nFailed to generate LaTeX cover letter due to an unexpected error.")
