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


def _extract_abstract_by_margins(pages) -> int | None:
    """Detect abstract by finding text with wider margins (indented block) on page 1."""
    if not pages:
        return None

    first_page = pages[0]
    if not first_page.chars:
        return None

    # Group characters into lines by y-position (top coordinate)
    lines_by_y: dict[float, list] = {}
    for char in first_page.chars:
        if not char.get("text", "").strip():
            continue
        y_key = round(char["top"], 0)
        if y_key not in lines_by_y:
            lines_by_y[y_key] = []
        lines_by_y[y_key].append(char)

    if not lines_by_y:
        return None

    # Sort lines by vertical position
    sorted_ys = sorted(lines_by_y.keys())

    # Find the left margin (x0) for each line
    line_margins = []
    for y in sorted_ys:
        chars = lines_by_y[y]
        text = "".join(c["text"] for c in sorted(chars, key=lambda c: c["x0"]))
        x0 = min(c["x0"] for c in chars)
        line_margins.append((y, x0, text))

    # Find where "Abstract" heading is
    abstract_line_idx = None
    for i, (y, x0, text) in enumerate(line_margins):
        if text.strip().lower() in ("abstract", "abstract:"):
            abstract_line_idx = i
            break

    if abstract_line_idx is None:
        return None

    # Determine the body text left margin (most common x0, excluding title/heading area)
    all_x0s = [x0 for _, x0, text in line_margins if len(text.strip()) > 10]
    if not all_x0s:
        return None

    # Body margin is the most common left margin
    x0_counter: Counter = Counter(round(x, 0) for x in all_x0s)
    body_margin = x0_counter.most_common(1)[0][0]

    # Collect lines after "Abstract" that have a noticeably larger left margin
    abstract_lines = []
    margin_threshold = body_margin + 5  # abstract indented by at least 5 points

    for i in range(abstract_line_idx + 1, len(line_margins)):
        y, x0, text = line_margins[i]
        text_stripped = text.strip()
        if not text_stripped:
            continue
        if round(x0, 0) >= margin_threshold:
            abstract_lines.append(text_stripped)
        elif abstract_lines:
            # We hit a line back at body margin — abstract is done
            break

    if not abstract_lines:
        return None

    abstract_text = " ".join(abstract_lines)
    wc = len(abstract_text.split())
    # Sanity check: abstracts are 50-400 words
    if 20 < wc < 500:
        return wc
    return None


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
        # Strategy 1: Use character positions to detect indented abstract block
        # Abstracts often have wider margins (larger x0) than body text
        abstract_word_count = _extract_abstract_by_margins(pages)

        if not abstract_word_count:
            # Strategy 2: Text-based extraction with end markers
            abs_start = text_lower.find("abstract")
            abs_text_after = full_text[abs_start + len("abstract"):]
            abs_end = len(abs_text_after)
            end_markers = [
                "introduction", "keywords", "key words", "jel",
                "1.", "1 ", "i.", "i ",
                "literature review", "background", "motivation",
                "methods", "methodology", "data",
                "table of contents",
            ]
            for marker in end_markers:
                idx = abs_text_after.lower().find(marker)
                if 0 < idx < abs_end:
                    abs_end = idx
            abstract_text = abs_text_after[:abs_end].strip()
            abstract_words = abstract_text.split()

            if len(abstract_words) > 400:
                # Strategy 3: First-page extraction with footnote detection
                first_page_text = all_text_parts[0] if all_text_parts else ""
                fp_lower = first_page_text.lower()
                fp_abs_start = fp_lower.find("abstract")
                if fp_abs_start >= 0:
                    fp_abs_text = first_page_text[fp_abs_start + len("abstract"):]
                    footnote_idx = len(fp_abs_text)
                    for fn_marker in ["\n*", "\n†", "\n‡", "\n∗"]:
                        idx = fp_abs_text.find(fn_marker)
                        if 0 < idx < footnote_idx:
                            footnote_idx = idx
                    double_nl = fp_abs_text.find("\n\n")
                    if 0 < double_nl < footnote_idx:
                        footnote_idx = double_nl
                    fp_abstract = fp_abs_text[:footnote_idx].strip()
                    if 20 < len(fp_abstract.split()) < 500:
                        abstract_word_count = len(fp_abstract.split())
                    else:
                        for chunk in fp_abs_text.split("\n\n"):
                            chunk = chunk.strip()
                            if chunk and len(chunk.split()) > 20:
                                abstract_word_count = len(chunk.split())
                                break
                        else:
                            abstract_word_count = len(abstract_words)
                else:
                    abstract_word_count = len(abstract_words)
            else:
                abstract_word_count = len(abstract_words)

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
