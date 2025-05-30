# phd_apply_agent/app_document_pipeline/prompts.py

"""
Centralized storage for prompt string templates for direct use with the
'google-generativeai' library via Metis.
This version reflects a simplified 3-LLM call pipeline with enhanced instructions
for TEMPLATE_COVER_LETTER and TEMPLATE_EMAIL_APPLICATION.
- LLMs use ONLY the user-confirmed EXTRACTED FIELDS for both Position and User CV,
  with corrected spellings.
- NO RAW TEXT (Position or CV) is provided to LLM Calls 1 & 2.
- Includes 'user_scores' and 'user_additional_file_names' (as additional_attachments_list_str).
"""

# Note: Input variables for these prompts will be formatted into the strings
# using f-strings or .format() by the calling agent functions.
# Literal curly braces in LaTeX syntax MUST be doubled (e.g., {{, }})

# --- LLM Call 1: Cover Letter Generator (Full LaTeX) ---
# Uses ONLY user-confirmed extracted fields with corrected spellings.
# 'current_date_for_letter' is injected by Python.
template_cover_letter_str = """
You are an expert academic assistant. Your task is to generate a complete, refined, and compilable LaTeX document for a cover letter.
The cover letter must be highly personalized, deeply connecting the candidate's technical background and research aspirations with the specific PhD opportunity.
Your goal is to meticulously analyze the provided extracted information to create a compelling narrative that demonstrates not only qualification but also genuine intellectual curiosity and enthusiasm for THIS specific research.
You must infer specific position requirements, potential research questions, technical challenges, and detailed research focus primarily from the "{position_summary}" and "{position_keywords}".
NO RAW CV TEXT OR RAW POSITION DESCRIPTION IS PROVIDED.

--- Candidate Details (Based ONLY on EXTRACTED CV Information - Address and Phone are NOT available) ---
Name: {user_fullname}
Email: {user_email_address}
Website: {user_website}
LinkedIn: {user_linkedin}
CV Summary: {user_cv_summary}
Key Skills: {user_skills}
Research Interests: {user_research_interests}
Test Scores: {user_scores}
Education Highlights: {user_education}
Research Experience Highlights: {user_research_experience}
Work Experience Highlights: {user_work_experience}
Publications Summary: {user_publication}
Awards and Honors: {user_honor_awards}

--- Position Details (Based ONLY on EXTRACTED Information) ---
Position Title: {position_title}
University Name: {position_university_name}
Department/Faculty: {position_department_faculty}
Location: {position_location}
Contact Person: {position_person_name}
Application Deadline: {application_deadline}
Keywords: {position_keywords}
Position Summary: {position_summary}

--- Recipient & Letter Meta-Details (Derived or Python-generated) ---
Current Date: {current_date_for_letter}
Recipient Name/Title (for letter addressing, derived from Contact Person): {recipient_name_title_formatted} 
Salutation for Letter: Dear {recipient_salutation_formatted},

--- LaTeX Document Structure to Fill & Refine ---
Generate a complete LaTeX document starting with \\documentclass[11pt,a4paper]{{article}} and ending with \\end{{document}}.
Use a standard letter format. Include standard packages: \\usepackage[utf8]{{inputenc}}, \\usepackage{{geometry}} (with \\geometry{{a4paper, margin=1in}}), \\usepackage{{hyperref}}.
Use \\usepackage{{parskip}} (and ensure paragraph separation is by whitespace, not indent).
Consider including \\usepackage{{amsmath}}, \\usepackage{{amsfonts}} if the candidate's skills/research imply mathematical notation.
The \\hypersetup block should be included for PDF metadata.

\\documentclass[11pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{geometry}}
\\geometry{{a4paper, margin=1in}}
\\usepackage{{parskip}}
\\setlength{{\\parindent}}{{0pt}} % Usually goes with parskip
\\setlength{{\\parskip}}{{1em}}   % Usual value for parskip
\\usepackage{{hyperref}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}

\\hypersetup{{
    colorlinks=true,
    linkcolor=blue,
    urlcolor=cyan,
    pdftitle={{Cover Letter - {user_fullname} - {position_title}}},
    pdfauthor={{ {user_fullname} }}
}}

\\begin{{document}}

% Sender's Information
% LLM: Construct this block using only {user_fullname} and {user_email_address}. Each on a new line (\\\\).
% If {user_website} is not an empty string or "N/A", include it on a new line, hyperlinked: \\href{{{user_website}}}{{{user_website}}}\\\\
% If {user_linkedin} is not an empty string or "N/A", include it on a new line, hyperlinked: \\href{{{user_linkedin}}}{{{user_linkedin}}}\\\\
% DO NOT include lines for address or phone number.

\\vspace{{1cm}} % Use standard LaTeX spacing like \bigskip or \medskip if preferred
\\noindent {current_date_for_letter}
\\vspace{{1cm}}

% Recipient's Information
\\noindent {recipient_name_title_formatted}\\\\
{position_department_faculty}\\\\
{position_university_name}\\\\
{position_location}
\\vspace{{1cm}}

Dear {recipient_salutation_formatted},
\\vspace{{0.5cm}}

% Narrative Body of the Cover Letter (Generate 3-4 Focused Paragraphs) 
% note that this part must be fully technical and demonstrate strong correlation between the position and the user career path  and what is done up to now
... (rest unchanged) ...

\\end{{document}}
"""
TEMPLATE_COVER_LETTER = template_cover_letter_str

# --- LLM Call 2: Email Application Generator ---
# Uses ONLY user-confirmed Extracted CV fields & Position fields with corrected spellings.
# NO RAW CV TEXT. NO RAW POSITION TEXT.
# Includes a list of additional attachment filenames.
template_email_application_str = """
You are an expert academic assistant specializing in crafting PhD application communications for science and engineering fields.
Your task is to draft an extremely concise, professional, and highly informative email (4–6 sentences) using ONLY the provided structured fields. Do NOT reference raw CV or job description text.

The candidate's CV and a Cover Letter will be attached.
Additionally, the following files will be attached: {additional_attachments_list_str} but not mention the file name understand the file purpose (e.g bach_trans is potentially bachelor transcription)

--- PhD Position Details (Using ONLY Extracted Fields) ---
Position Title: {position_title}
University: {position_university_name}
Department: {position_department_faculty}
Contact Person: {position_contact_person_name_for_salutation}
Application Deadline: {application_deadline}
Keywords: {position_keywords}
Position Summary: {position_summary}

--- Candidate Profile (Using ONLY Extracted CV Fields) ---
Name: {user_fullname}
Key Skills: {user_skills}
Research Interests: {user_research_interests}
CV Summary: {user_cv_summary}
Test Scores: {user_scores}
Education Highlights: {user_education}
Research Experience: {user_research_experience}
Work Experience: {user_work_experience}
Publications Summary: {user_publication}
Awards and Honors: {user_honor_awards}

Based on these fields, generate the full email:
1. Subject line: "PhD Application – {position_title}".
2. Salutation: "Dear Professor {position_contact_person_name_for_salutation}," or "Dear Hiring Committee,".
3. Opening sentence: state your application and reference the position (specify the position by university+lab+supervisor).
4. Two to three key factual highlights directly drawn from the profile:
   • Academic Performance or Test Scores
   • Relevant Technical Skill(s)
   • Direct Research Experience or Publications/Honors or relevant work experience or other parts suits for this task
   • other things that must be covered on cv like publications summery research and worked experience and etc..

5. Reference attachments: CV, cover letter, and additional files.
6. Closing sentence: express appreciation and availability for further details.
7. Sign-off: "Sincerely," or "Best regards," followed by {user_fullname}.

Ensure the email is direct, specific, and avoids redundant or generic filler text.
"""
TEMPLATE_EMAIL_APPLICATION = template_email_application_str

# --- LLM Call 3: Follow-up Email Drafter ---
# Expected input variables:
#   original_email_subject (str), original_email_recipient_name (str),
#   position_title (str), application_date (str), time_elapsed (str),
#   user_fullname (str), previous_email_body_snippet (str).
template_email_followup_str = """
You are an expert academic assistant. Your task is to draft a polite and professional follow-up email regarding a PhD application.
The applicant, {user_fullname}, previously applied for the position of "{position_title}".
The original application was sent on {application_date} (approximately {time_elapsed} ago).
The recipient was addressed as "Dear {original_email_recipient_name}," in the previous email.
A snippet of the previous email body: "{previous_email_body_snippet}..."

Based on this information, draft a brief follow-up email. The email should:
1. Start with an appropriate salutation to {original_email_recipient_name}.
2. Briefly and politely refer to the previous application for the "{position_title}" position, mentioning the date it was sent.
3. Reiterate continued strong interest in the opportunity.
4. Inquire if there is any update on the application status or if any further information is required from the applicant.
5. Be concise, professional, and courteous.
6. Concludes with a professional closing and the applicant's name ({user_fullname}).

Generate ONLY the full follow-up email text.
"""
TEMPLATE_EMAIL_FOLLOWUP = template_email_followup_str

if __name__ == '__main__':
    print("[AppDocumentPipeline Prompts] All prompt string templates defined for the 3-call pipeline (Enhanced V6).")

    defined_prompts_check = {
        "TEMPLATE_COVER_LETTER": 'TEMPLATE_COVER_LETTER' in globals(),
        "TEMPLATE_EMAIL_APPLICATION": 'TEMPLATE_EMAIL_APPLICATION' in globals(),
        "TEMPLATE_EMAIL_FOLLOWUP": 'TEMPLATE_EMAIL_FOLLOWUP' in globals()
    }
    # ... (rest of the __main__ block for checks remains the same) ...
    all_current_prompts_defined = True
    for name, is_defined in defined_prompts_check.items():
        if is_defined:
            print(f"Current prompt '{name}' is defined.")
        else:
            print(f"ERROR: Current prompt '{name}' is NOT defined.")
            all_current_prompts_defined = False

    obsolete_prompt_names = [
        "PROMPT_STRING_SUPPLEMENTARY_ANALYZER",
        "PROMPT_STRING_COVER_LETTER_GENERATE",
        "PROMPT_STRING_EMAIL_GENERATE_AND_REFINE",
        "PROMPT_STRING_FOLLOW_UP_EMAIL_DRAFT",
        "PROMPT_STRING_CONSISTENCY_POLISH",
        "PROMPT_STRING_COVER_LETTER_REFINE",
        "PROMPT_STRING_EMAIL_REFINE"
    ]

    no_obsolete_found = True
    for obs_name in obsolete_prompt_names:
        if obs_name in globals():
            print(f"ERROR: Obsolete prompt '{obs_name}' IS STILL DEFINED but should be removed.")
            no_obsolete_found = False

    if all_current_prompts_defined and no_obsolete_found:
        print(
            "Prompts file correctly reflects the 3-call pipeline structure with current naming convention and no obsolete prompts.")
    else:
        print("Prompts file needs review based on the 3-call pipeline structure and naming convention.")
