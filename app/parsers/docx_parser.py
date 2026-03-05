import io
from collections import Counter

from docx import Document
from docx.shared import Pt

from ..models import ManuscriptMetadata

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


def parse_docx(file_bytes: bytes, filename: str) -> ManuscriptMetadata:
    """Extract metadata from a Word document."""
    doc = Document(io.BytesIO(file_bytes))

    all_text_parts = []
    font_counter: Counter = Counter()
    size_counter: Counter = Counter()
    detected_sections = []
    heading_texts = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            all_text_parts.append(text)

        # Check if paragraph is a heading
        if para.style and para.style.name and para.style.name.startswith("Heading"):
            heading_texts.append(text.lower())

        # Collect font info from runs
        for run in para.runs:
            if run.font.name:
                font_counter[run.font.name] += 1
            if run.font.size:
                size_pt = run.font.size / Pt(1)
                size_counter[round(size_pt, 1)] += 1

    full_text = "\n".join(all_text_parts)
    word_count = len(full_text.split())

    # Detect font/size
    detected_font = font_counter.most_common(1)[0][0] if font_counter else None
    detected_size = size_counter.most_common(1)[0][0] if size_counter else None

    # Detect line spacing from first body paragraph
    detected_line_spacing = None
    for para in doc.paragraphs:
        if para.paragraph_format.line_spacing:
            spacing = para.paragraph_format.line_spacing
            if isinstance(spacing, (int, float)):
                if spacing >= 2.0:
                    detected_line_spacing = "double"
                elif spacing >= 1.5:
                    detected_line_spacing = "1.5"
                else:
                    detected_line_spacing = "single"
            break

    # Detect sections from headings and text
    text_lower = full_text.lower()
    for section in SECTION_PATTERNS:
        if section in text_lower or any(section in h for h in heading_texts):
            detected_sections.append(section.title())

    # Abstract
    has_abstract = "abstract" in text_lower
    abstract_word_count = None
    if has_abstract:
        abs_start = text_lower.find("abstract")
        abs_text_after = full_text[abs_start + len("abstract"):]
        abs_end = len(abs_text_after)
        for section in ["introduction", "keywords", "jel", "1."]:
            idx = abs_text_after.lower().find(section)
            if 0 < idx < abs_end:
                abs_end = idx
        abstract_word_count = len(abs_text_after[:abs_end].split())

    has_references = any(s in text_lower for s in ["references", "bibliography"])
    figure_count = text_lower.count("figure ") + text_lower.count("fig.")
    has_figures = figure_count > 0

    # Author info
    first_part = "\n".join(all_text_parts[:10]).lower()
    contains_author_info = any(ind in first_part for ind in AUTHOR_INDICATORS)

    # Title
    title = None
    for para in doc.paragraphs:
        if para.style and para.style.name and "Title" in para.style.name and para.text.strip():
            title = para.text.strip()
            break
    if not title:
        for part in all_text_parts:
            if len(part) > 5:
                title = part
                break

    # Page count estimate (~250 words per page)
    page_count = max(1, word_count // 250) if word_count else None

    return ManuscriptMetadata(
        filename=filename,
        file_type="docx",
        word_count=word_count,
        page_count=page_count,
        detected_font=detected_font,
        detected_font_size=detected_size,
        detected_line_spacing=detected_line_spacing,
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
