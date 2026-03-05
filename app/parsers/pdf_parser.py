import io
from collections import Counter

import pdfplumber

from ..models import ManuscriptMetadata

# Common section headings in academic papers
SECTION_PATTERNS = [
    "abstract", "introduction", "literature review", "methodology", "methods",
    "data", "results", "discussion", "conclusion", "conclusions",
    "references", "bibliography", "appendix", "appendices",
    "acknowledgments", "acknowledgements", "keywords", "jel codes",
    "jel classification", "data availability", "funding",
    "declaration of interest", "conflict of interest",
]

AUTHOR_INDICATORS = [
    "university", "department", "institute", "college", "school of",
    "faculty of", "email:", "e-mail:", "@", "corresponding author",
    "orcid",
]


def parse_pdf(file_bytes: bytes, filename: str) -> ManuscriptMetadata:
    """Extract metadata from a PDF manuscript."""
    pdf = pdfplumber.open(io.BytesIO(file_bytes))

    pages = pdf.pages
    page_count = len(pages)

    all_text_parts = []
    font_counter: Counter = Counter()
    size_counter: Counter = Counter()

    for page in pages:
        text = page.extract_text() or ""
        all_text_parts.append(text)

        # Collect font info from characters
        if page.chars:
            for char in page.chars:
                if char.get("fontname"):
                    font_counter[char["fontname"]] += 1
                if char.get("size"):
                    size_counter[round(char["size"], 1)] += 1

    full_text = "\n".join(all_text_parts)
    words = full_text.split()
    word_count = len(words)

    # Detect most common font and size
    detected_font = font_counter.most_common(1)[0][0] if font_counter else None
    detected_size = size_counter.most_common(1)[0][0] if size_counter else None

    # Detect sections
    text_lower = full_text.lower()
    detected_sections = []
    for section in SECTION_PATTERNS:
        # Look for section as a heading (start of line or after number)
        if section in text_lower:
            detected_sections.append(section.title())

    # Abstract detection and word count
    has_abstract = "abstract" in text_lower
    abstract_word_count = None
    if has_abstract:
        abs_start = text_lower.find("abstract")
        # Find end of abstract (next section or ~500 words)
        abs_text_after = full_text[abs_start + len("abstract"):]
        # Try to find next section heading
        abs_end = len(abs_text_after)
        for section in ["introduction", "keywords", "jel", "1."]:
            idx = abs_text_after.lower().find(section)
            if idx > 0 and idx < abs_end:
                abs_end = idx
        abstract_text = abs_text_after[:abs_end].strip()
        abstract_word_count = len(abstract_text.split())

    # References
    has_references = any(s in text_lower for s in ["references", "bibliography"])

    # Figures
    figure_count = text_lower.count("figure ") + text_lower.count("fig.")
    has_figures = figure_count > 0

    # Author info detection (check first 2 pages)
    first_pages = "\n".join(all_text_parts[:2]).lower()
    contains_author_info = any(ind in first_pages for ind in AUTHOR_INDICATORS)

    # Title (first non-empty line, heuristic)
    title = None
    for line in full_text.split("\n"):
        stripped = line.strip()
        if stripped and len(stripped) > 5:
            title = stripped
            break

    pdf.close()

    return ManuscriptMetadata(
        filename=filename,
        file_type="pdf",
        word_count=word_count,
        page_count=page_count,
        detected_font=detected_font,
        detected_font_size=detected_size,
        detected_sections=detected_sections,
        has_abstract=has_abstract,
        abstract_word_count=abstract_word_count,
        has_references=has_references,
        has_figures=has_figures,
        figure_count=figure_count,
        contains_author_info=contains_author_info,
        title=title,
        raw_text=full_text,
    )
