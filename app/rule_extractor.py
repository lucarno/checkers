import json
import re
import anthropic
from .models import JournalRules


EXTRACTION_PROMPT = """You are an expert at reading academic journal submission guidelines and extracting structured formatting rules.

Given the following text scraped from a journal's submission guidelines page, extract the rules into a structured JSON object.

Return ONLY valid JSON matching this schema (use null for fields you cannot determine):
{
  "journal_name": "string or null",
  "word_limit": "integer or null",
  "page_limit": "integer or null",
  "font": "string or null (e.g., 'Times New Roman')",
  "font_size": "number or null (e.g., 12)",
  "line_spacing": "string or null (e.g., 'double', '1.5')",
  "margins": "string or null (e.g., '1 inch all sides')",
  "required_sections": ["list of required section names"],
  "abstract_word_limit": "integer or null",
  "citation_style": "one of: APA, Chicago, MLA, Harvard, Vancouver, Numbered, Other, Unknown",
  "figure_requirements": {
    "formats": ["list of accepted formats"],
    "min_resolution_dpi": "integer or null",
    "max_file_size_mb": "number or null",
    "placement": "string or null"
  },
  "reference_format": "string or null (brief description)",
  "anonymization_required": "boolean or null",
  "additional_notes": ["list of other notable requirements"]
}

Guidelines text:
---
{guidelines_text}
---

Return ONLY the JSON object, no other text."""


def extract_rules(guidelines_text: str, api_key: str) -> JournalRules:
    """Use Claude API to extract structured rules from guidelines text."""
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": EXTRACTION_PROMPT.format(guidelines_text=guidelines_text),
            }
        ],
    )

    response_text = message.content[0].text.strip()

    # Strip markdown code fences anywhere in the response
    response_text = re.sub(r"```(?:json)?\s*\n?", "", response_text).strip()

    # Extract the first JSON object from the response
    match = re.search(r"\{", response_text)
    if match:
        response_text = response_text[match.start():]

    data = json.loads(response_text)
    return JournalRules(**data)
