"""LLM-based manuscript analysis for precise text-level metadata extraction.

Hybrid approach: the LLM identifies section boundaries by quoting the exact
text at each boundary. Python then locates those strings in the manuscript
and counts words programmatically — giving exact counts rather than LLM
estimates.
"""

import json
import re
import unicodedata

import anthropic

from .models import ManuscriptMetadata, SectionWordCounts

# Common PDF text extraction artifacts: ligatures, smart quotes, dashes
_UNICODE_REPLACEMENTS = {
    '\ufb00': 'ff',
    '\ufb01': 'fi',
    '\ufb02': 'fl',
    '\ufb03': 'ffi',
    '\ufb04': 'ffl',
    '\u2018': "'",   # left single quote
    '\u2019': "'",   # right single quote
    '\u201c': '"',   # left double quote
    '\u201d': '"',   # right double quote
    '\u2013': '-',   # en-dash
    '\u2014': '-',   # em-dash
    '\u2012': '-',   # figure dash
    '\u2015': '-',   # horizontal bar
    '\u00a0': ' ',   # non-breaking space
    '\u2002': ' ',   # en space
    '\u2003': ' ',   # em space
    '\u2009': ' ',   # thin space
    '\u200a': ' ',   # hair space
    '\u200b': '',    # zero-width space
    '\ufeff': '',    # BOM / zero-width no-break space
}

ANALYSIS_PROMPT = """You are an expert academic manuscript analyst. Given the full text of a scholarly manuscript, identify the exact boundaries between sections.

Return ONLY a valid JSON object with these fields:

{{
  "title": <string or null, the manuscript title>,
  "sections": [
    {{
      "type": <string, one of: "title_page", "abstract", "body", "footnotes", "references", "appendix">,
      "heading": <string, the section heading as it appears in the text, e.g. "Abstract", "1. Introduction", "References">,
      "start_marker": <string, the EXACT first 8-12 words of this section (including the heading itself). Must be a verbatim quote from the manuscript that can be found with a text search.>
    }}
  ],
  "abstract_start_marker": <string or null, the EXACT first 8-12 words of the abstract TEXT (after the "Abstract" heading). Verbatim quote.>,
  "abstract_end_marker": <string or null, the EXACT first 8-12 words of whatever comes right after the abstract (e.g. "1. Introduction", "Keywords:", "JEL codes"). Verbatim quote.>,
  "has_abstract": <boolean>,
  "has_references": <boolean>,
  "citation_style": <string, one of: "APA", "Chicago", "Harvard", "Vancouver", "Numbered", "MLA", "Other", "Unknown">,
  "citation_style_details": <string, brief explanation of why you identified this style>,
  "contains_author_info": <boolean, whether author names, affiliations, or emails are present>,
  "has_figures": <boolean>,
  "figure_count": <int, number of distinct figures referenced (Figure 1, Figure 2, etc.)>,
  "acknowledgment_location": <string, one of: "after_references", "before_references", "footnote", "not_found">,
  "structural_issues": [<list of strings describing any structural problems>]
}}

CRITICAL rules for sections:
- List sections in the ORDER they appear in the manuscript.
- "title_page": everything before the abstract (title, authors, affiliations). Always include this as the first section.
- "abstract": the abstract section. The start_marker should include the word "Abstract" or equivalent heading.
- "body": starts at the first substantive section after abstract (e.g. "Introduction", "1. Background"). If the body has multiple sections (Introduction, Methods, Results, etc.), combine them ALL into ONE "body" entry — use the start of the FIRST body section as the start_marker.
- "footnotes": a dedicated Notes/Endnotes section (NOT inline footnotes). Only include if there's a separate section with a heading like "Notes" or "Endnotes".
- "references": the reference/bibliography list.
- "appendix": appendix/appendices sections.
- Each start_marker MUST be an exact verbatim quote (8-12 words) that appears in the manuscript text. Copy it character-for-character.
- Do NOT include sections that don't exist in the manuscript.

For abstract_start_marker: quote the first 8-12 words of the abstract CONTENT (the actual text, not the heading). For abstract_end_marker: quote the first 8-12 words of whatever section follows the abstract.

For citation_style:
- APA: (Author, Year) with comma before year
- Chicago: (Author Year) without comma; or footnote-based citations
- Harvard: (Author Year) or (Author Year: page)
- Vancouver/Numbered: [1] or superscript numbers
- If it's a sub-style (APSA, AEA), classify under the parent but note it in details.

Manuscript text:
---
{manuscript_text}
---

Return ONLY the JSON object."""


def _normalize_unicode(text: str) -> str:
    """Normalize unicode artifacts common in PDF text extraction."""
    for orig, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(orig, replacement)
    return text


def _make_flexible_pattern(marker_text: str) -> str:
    """Build a regex pattern from marker text with flexible whitespace and punctuation.

    Each word is escaped for regex, and words are joined with a pattern that
    allows any whitespace (including line breaks) and optional hyphens between
    them (to handle PDF line-break hyphenation).
    """
    words = marker_text.split()
    if not words:
        return ""
    escaped_words = [re.escape(w) for w in words]
    return r'[\s\-]*'.join(escaped_words)


def _find_marker(text: str, marker: str) -> int:
    """Find the position of a marker string in the text.

    Tries exact match first, then progressively looser matching
    with unicode normalization and flexible whitespace/punctuation.
    Returns -1 if not found.
    """
    if not marker:
        return -1

    # Exact match
    pos = text.find(marker)
    if pos >= 0:
        return pos

    # Normalize unicode in both text and marker, then try flexible matching
    norm_text = _normalize_unicode(text)
    norm_marker = _normalize_unicode(re.sub(r'\s+', ' ', marker.strip()))

    # Full marker with flexible whitespace
    pattern = _make_flexible_pattern(norm_marker)
    if pattern:
        m = re.search(pattern, norm_text, re.IGNORECASE)
        if m:
            return m.start()

    # Try with just the first 6 words
    words = norm_marker.split()
    if len(words) > 6:
        pattern = _make_flexible_pattern(' '.join(words[:6]))
        m = re.search(pattern, norm_text, re.IGNORECASE)
        if m:
            return m.start()

    # Try first 4 words
    if len(words) > 4:
        pattern = _make_flexible_pattern(' '.join(words[:4]))
        m = re.search(pattern, norm_text, re.IGNORECASE)
        if m:
            return m.start()

    # Try first 3 words
    if len(words) > 3:
        pattern = _make_flexible_pattern(' '.join(words[:3]))
        m = re.search(pattern, norm_text, re.IGNORECASE)
        if m:
            return m.start()

    return -1


def _find_heading(text: str, heading: str, sec_type: str) -> int:
    """Find a section heading in the text as a fallback when start_marker fails.

    Searches for the heading text at the beginning of a line, which is how
    section headings typically appear in extracted manuscript text.
    Returns -1 if not found.
    """
    if not heading:
        return -1

    norm_text = _normalize_unicode(text)
    norm_heading = _normalize_unicode(heading.strip())

    # Try: heading at start of line (with optional numbering/whitespace before it)
    escaped = re.escape(norm_heading)
    # Allow optional leading whitespace and numbering like "1." or "I."
    pattern = r'(?:^|\n)\s*' + escaped + r'\s*(?:\n|$|[:.])'
    m = re.search(pattern, norm_text, re.IGNORECASE)
    if m:
        # Return position of the heading text itself, not the newline
        heading_match = re.search(escaped, norm_text[m.start():m.end()], re.IGNORECASE)
        if heading_match:
            return m.start() + heading_match.start()
        return m.start()

    # For well-known section types, try common heading variants
    _HEADING_VARIANTS = {
        "abstract": [r"abstract"],
        "references": [r"references", r"bibliography", r"works\s+cited", r"literature\s+cited"],
        "appendix": [r"appendi(?:x|ces)", r"online\s+appendix", r"supplementary\s+materials?",
                      r"internet\s+appendix"],
        "footnotes": [r"notes", r"endnotes", r"footnotes"],
    }

    variants = _HEADING_VARIANTS.get(sec_type, [])
    for variant in variants:
        pattern = r'(?:^|\n)\s*' + variant + r'\s*(?:\n|$|[:.])'
        m = re.search(pattern, norm_text, re.IGNORECASE)
        if m:
            content_match = re.search(variant, norm_text[m.start():m.end()], re.IGNORECASE)
            if content_match:
                return m.start() + content_match.start()
            return m.start()

    return -1


def _wc(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())


def analyze_manuscript(raw_text: str, api_key: str) -> dict:
    """Send manuscript text to Claude for boundary detection, then count words.

    Returns a dict with LLM-extracted metadata and programmatic word counts.
    """
    # Truncate very long texts to stay within token limits
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

    analysis = json.loads(response_text)

    # === Hybrid step: use LLM boundaries to count words programmatically ===
    sections = analysis.get("sections", [])
    text = raw_text

    # Build ordered list of (position, section_type) from markers
    boundaries = []
    for sec in sections:
        marker = sec.get("start_marker", "")
        heading = sec.get("heading", "")
        sec_type = sec.get("type", "")
        pos = _find_marker(text, marker)
        # Fallback: search for the section heading itself
        if pos < 0:
            pos = _find_heading(text, heading, sec_type)
        if pos >= 0:
            boundaries.append((pos, sec_type))

    # Sort by position
    boundaries.sort(key=lambda x: x[0])

    # Compute word counts from boundaries
    section_texts = {
        "title_page": "",
        "abstract": "",
        "body": "",
        "footnotes": "",
        "references": "",
        "appendix": "",
    }

    for i, (pos, sec_type) in enumerate(boundaries):
        end_pos = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        chunk = text[pos:end_pos]
        if sec_type in section_texts:
            section_texts[sec_type] += chunk

    # If no title_page boundary was found but we have other boundaries,
    # everything before the first boundary is title_page
    if boundaries and (not boundaries[0][1] == "title_page" or boundaries[0][0] > 0):
        first_pos = boundaries[0][0]
        if first_pos > 0:
            existing_tp = section_texts["title_page"]
            section_texts["title_page"] = text[:first_pos] + existing_tp

    # Count words per section
    swc = {}
    for key, sec_text in section_texts.items():
        swc[key] = _wc(sec_text)

    analysis["section_word_counts"] = swc

    # === Abstract word count: use specific abstract markers ===
    abs_start_marker = analysis.get("abstract_start_marker")
    abs_end_marker = analysis.get("abstract_end_marker")

    if abs_start_marker:
        abs_start = _find_marker(text, abs_start_marker)
        if abs_start >= 0:
            abs_end = len(text)
            if abs_end_marker:
                end_pos = _find_marker(text, abs_end_marker)
                if end_pos > abs_start:
                    abs_end = end_pos
            elif "abstract" in section_texts and section_texts["abstract"]:
                # Fall back to the section boundary
                for i, (pos, sec_type) in enumerate(boundaries):
                    if sec_type == "abstract" and i + 1 < len(boundaries):
                        abs_end = boundaries[i + 1][0]
                        break

            abstract_text = text[abs_start:abs_end].strip()
            analysis["abstract_word_count"] = _wc(abstract_text)
        else:
            # Fallback: use the abstract section text
            analysis["abstract_word_count"] = swc.get("abstract", 0)
    else:
        analysis["abstract_word_count"] = swc.get("abstract", 0)

    # Sections found (from the LLM)
    analysis["sections_found"] = [
        sec.get("heading", sec.get("type", ""))
        for sec in sections
    ]

    return analysis


def apply_llm_analysis(metadata: ManuscriptMetadata, analysis: dict) -> ManuscriptMetadata:
    """Override heuristic metadata fields with LLM analysis results.

    Only overrides text-level fields; binary-level fields (font, font_size,
    line_spacing, page_count) are kept from the parser since those come from
    the file format itself.
    """
    # Section word counts (now computed programmatically from LLM boundaries)
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
