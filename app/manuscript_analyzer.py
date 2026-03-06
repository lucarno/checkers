"""LLM-based manuscript analysis for precise text-level metadata extraction.

When an Anthropic API key is provided, this module sends the manuscript's
extracted text to Claude for accurate section segmentation, abstract word
counting, citation style detection, and structural analysis — replacing
the heuristic approaches in the parsers and checker.
"""

import json
import re

import anthropic

from .models import ManuscriptMetadata, SectionWordCounts

ANALYSIS_PROMPT = """You are an expert academic manuscript analyst. Given the full text of a scholarly manuscript, extract precise structured metadata.

Carefully read the manuscript text and return ONLY a valid JSON object with these fields:

{{
  "sections_found": ["list of section headings found, in order"],
  "section_word_counts": {{
    "title_page": <int, words before the abstract (title, authors, affiliations, etc.)>,
    "abstract": <int, words in the abstract section only>,
    "body": <int, words from first body section (e.g. Introduction) through the last body section (e.g. Conclusion), including any in-text footnotes>,
    "footnotes": <int, words in a dedicated notes/endnotes section if separate from body, 0 if footnotes are inline>,
    "references": <int, words in the reference/bibliography list>,
    "appendix": <int, words in appendix/appendices sections>
  }},
  "abstract_word_count": <int, exact word count of the abstract text>,
  "has_abstract": <boolean>,
  "has_references": <boolean>,
  "citation_style": <string, one of: "APA", "Chicago", "Harvard", "Vancouver", "Numbered", "MLA", "Other", "Unknown">,
  "citation_style_details": <string, brief explanation of why you identified this style>,
  "contains_author_info": <boolean, whether author names, affiliations, or emails are present>,
  "title": <string or null, the manuscript title>,
  "has_figures": <boolean>,
  "figure_count": <int, number of figures referenced>,
  "acknowledgment_location": <string, one of: "after_references", "before_references", "footnote", "not_found">,
  "structural_issues": [<list of strings describing any structural problems, e.g. "Acknowledgments appear as a footnote rather than a section">]
}}

Rules for counting:
- Count words by splitting on whitespace. Hyphenated words count as one word.
- "title_page" includes everything before the abstract (or before the first body section if no abstract).
- "abstract" is ONLY the abstract paragraph(s), not including the heading "Abstract" itself.
- "body" starts at the first body section (usually Introduction) and ends at the last body section (usually Conclusion/Discussion), including section headings.
- "footnotes" counts words in a dedicated Notes/Endnotes section ONLY. If footnotes are inline (at bottom of pages), count them as 0 here — they'll be part of body text.
- "references" counts all words in the References/Bibliography section.
- "appendix" counts everything in Appendix/Appendices sections.
- All section counts should sum approximately to the total manuscript word count.

For citation_style:
- APA: (Author, Year) with comma before year; references as Author, A. B. (Year). Title.
- Chicago: (Author Year) without comma; or footnote-based citations with ibid/op. cit.
- Harvard: (Author Year) or (Author Year: page); varies by institution.
- Vancouver/Numbered: [1] or superscript numbers; numbered reference list.
- If it's a specific sub-style (e.g., APSA, AEA), still classify under the parent (Chicago, APA, etc.) but note it in citation_style_details.

Manuscript text:
---
{manuscript_text}
---

Return ONLY the JSON object."""


def analyze_manuscript(raw_text: str, api_key: str) -> dict:
    """Send manuscript text to Claude for precise analysis.

    Returns a dict with LLM-extracted metadata fields.
    """
    # Truncate very long texts to stay within token limits (~100k chars ≈ 25k tokens)
    text_to_send = raw_text[:100000] if len(raw_text) > 100000 else raw_text

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": ANALYSIS_PROMPT.format(manuscript_text=text_to_send),
            }
        ],
    )

    response_text = message.content[0].text.strip()

    # Strip markdown code fences
    response_text = re.sub(r"```(?:json)?\s*\n?", "", response_text).strip()

    # Extract JSON object
    match = re.search(r"\{", response_text)
    if match:
        response_text = response_text[match.start():]

    return json.loads(response_text)


def apply_llm_analysis(metadata: ManuscriptMetadata, analysis: dict) -> ManuscriptMetadata:
    """Override heuristic metadata fields with LLM analysis results.

    Only overrides text-level fields; binary-level fields (font, font_size,
    line_spacing, page_count) are kept from the parser since those come from
    the file format itself.
    """
    # Section word counts
    swc_data = analysis.get("section_word_counts", {})
    if swc_data:
        swc = SectionWordCounts(
            title_page=swc_data.get("title_page", 0),
            abstract=swc_data.get("abstract", 0),
            body=swc_data.get("body", 0),
            references=swc_data.get("references", 0),
            footnotes=swc_data.get("footnotes", 0),
            appendix=swc_data.get("appendix", 0),
        )
        swc.total = (swc.title_page + swc.abstract + swc.body +
                     swc.references + swc.footnotes + swc.appendix)
        metadata.section_word_counts = swc

    # Abstract
    if "has_abstract" in analysis:
        metadata.has_abstract = analysis["has_abstract"]
    if "abstract_word_count" in analysis and analysis["abstract_word_count"] is not None:
        metadata.abstract_word_count = analysis["abstract_word_count"]

    # Sections found
    if "sections_found" in analysis:
        metadata.detected_sections = analysis["sections_found"]

    # Title
    if analysis.get("title"):
        metadata.title = analysis["title"]

    # References
    if "has_references" in analysis:
        metadata.has_references = analysis["has_references"]

    # Figures
    if "has_figures" in analysis:
        metadata.has_figures = analysis["has_figures"]
    if "figure_count" in analysis:
        metadata.figure_count = analysis["figure_count"]

    # Author info
    if "contains_author_info" in analysis:
        metadata.contains_author_info = analysis["contains_author_info"]

    # Store LLM-specific analysis results
    metadata.llm_citation_style = analysis.get("citation_style")
    metadata.llm_citation_style_details = analysis.get("citation_style_details")
    metadata.llm_acknowledgment_location = analysis.get("acknowledgment_location")
    metadata.llm_structural_issues = analysis.get("structural_issues", [])
    metadata.llm_analyzed = True

    return metadata
