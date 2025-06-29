ANALYSIS_PROMPT = """\
Extract car models and other search terms from the user’s message.
Return **only** a JSON object of the form:

{{
  "<canonical model>": {{
    "aliases": ["<English spelling>", "<Farsi spelling>"],
    "keywords": ["<other keyword1>", "<keyword2>", …]
  }},
  …
}}

Generate both English & Farsi aliases for each model. Do not output anything else.
User: {user_query}
"""