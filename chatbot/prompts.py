# prompts.py
# این فایل برای سازگاری باقی مانده است؛ در نسخهٔ سریع، از fast_analyze_query استفاده می‌کنیم.
ANALYSIS_PROMPT = """\
Extract car models and other search terms from the user’s message.
Return ONLY a JSON object:

{
  "<canonical model>": {
    "aliases": ["<English spelling>", "<Farsi spelling>"],
    "keywords": ["<kw1>", "<kw2>"]
  }
}

User: {user_query}
"""
