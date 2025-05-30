# phd_apply_agent/app_document_pipeline/config.py

"""
Configuration settings specific to the App Document Pipeline module.
This module is responsible for analyzing supplementary materials, drafting,
and refining application documents (cover letter, email).
"""

import os

# --- Metis AI Configuration for Google Gemini ---

METIS_API_KEY = "tpsg-b1MsaOzQ0DhJ9ULVYLxaX2j7hwmC1DJ"
METIS_API_BASE_URL = "https://api.metisai.ir"

# --- Google Gemini Model Identifiers (via Metis) ---
ECONOMICAL_MODEL_ID = "gemini-2.5-flash-preview-04-17"
POWERFUL_MODEL_ID = "gemini-2.5-pro-preview-03-25"

# --- Default LLM Call Parameters ---
# These can be overridden when making specific LLM calls if needed.

# Temperature settings (creativity vs. factuality)
DEFAULT_TEMPERATURE_DRAFTING = 0.0

# Maximum output token limits for different tasks
# Adjust these based on expected output lengths and model capabilities
MAX_OUTPUT_TOKENS_DRAFT_CL_BODY = 10000
MAX_OUTPUT_TOKENS_DRAFT_EMAIL = 10000
MAX_OUTPUT_TOKENS_FOLLOWUP_EMAIL = 10000


# --- Logging Configuration for this Module ---
# Example: "INFO", "DEBUG", "WARNING", "ERROR"
MODULE_LOG_LEVEL = os.getenv("APP_DOC_PIPELINE_LOG_LEVEL", "INFO")

# --- Sanity Check for API Key ---
if METIS_API_KEY:
    print(f"[AppDocumentPipeline Config] Metis API Key loaded (first 5 chars): {METIS_API_KEY[:5]}...")
else:
    print("[AppDocumentPipeline Config ERROR] Metis API Key is not set. Please set the it via updating the config.")

print(f"[AppDocumentPipeline Config] Initialized. Log Level: {MODULE_LOG_LEVEL}")
print(f"[AppDocumentPipeline Config] Economical Model: {ECONOMICAL_MODEL_ID}")
print(f"[AppDocumentPipeline Config] Powerful Model: {POWERFUL_MODEL_ID}")
print(f"[AppDocumentPipeline Config] Metis API Endpoint for Gemini: {METIS_API_BASE_URL}")
