import io
from collections import Counter

import pdfplumber

from ..models import ManuscriptMetadata
from .segmenter import segment_text

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


def _detect_line_spacing(pages) -> str | None:
    """Detect line spacing by measuring distances between text lines."""
    # Sample body text from a middle page (avoid title pages and references)
    sample_page_idx = min(2, len(pages) - 1)
    page = pages[sample_page_idx]
    if not page.chars:
        return None

    # Get the most common font size (body text size)
    sizes = Counter(round(c["size"], 1) for c in page.chars if c.get("size"))
    if not sizes:
        return None
    body_size = sizes.most_common(1)[0][0]

    # Collect y-positions (top) of lines that use body font size
    line_tops = []
    seen_ys = set()
    for char in sorted(page.chars, key=lambda c: c["top"]):
        if not char.get("text", "").strip():
            continue
        if abs(round(char["size"], 1) - body_size) > 0.5:
            continue
        y = round(char["top"], 1)
        if y not in seen_ys:
            seen_ys.add(y)
            line_tops.append(y)

    if len(line_tops) < 5:
        return None

    # Calculate consecutive line distances
    distances = [line_tops[i + 1] - line_tops[i] for i in range(len(line_tops) - 1)]
    # Filter out large gaps (paragraph breaks, section breaks)
    median_dist = sorted(distances)[len(distances) // 2]
    normal_distances = [d for d in distances if d < median_dist * 1.8]
    if not normal_distances:
        return None

    avg_distance = sum(normal_distances) / len(normal_distances)

    # The ratio of line distance to font size indicates spacing
    # Single spacing ≈ 1.0-1.2x font size
    # 1.5 spacing ≈ 1.3-1.7x font size
    # Double spacing ≈ 1.8-2.5x font size
    ratio = avg_distance / body_size

    if ratio >= 1.8:
        return "double"
    elif ratio >= 1.3:
        return "1.5"
    else:
        return "single"


def _find_page_before_appendix(all_text_parts: list[str]) -> int:
    """Find the page count up to (and including) the references section, excluding appendix."""
    import re
    for i in range(len(all_text_parts) - 1, -1, -1):
        text_lower = all_text_parts[i].lower()
        # Look for appendix heading on this page
        if re.search(r"(?:^|\n)\s*(?:appendix|appendices|online appendix|supplementary|internet appendix)\b",
                      text_lower):
            # Return the page number before this page (i is 0-indexed)
            # But only if we've already passed references
            # Check if references appeared on an earlier page
            for j in range(i):
                if re.search(r"(?:^|\n)\s*(?:references|bibliography|works cited|literature cited)\b",
                             all_text_parts[j].lower()):
                    return i  # pages 0..i-1 = i pages (the appendix page is excluded)
            # References might be on the same page as appendix starts, count that page
            if re.search(r"(?:references|bibliography)\b", text_lower):
                return i + 1
            break
    # No appendix found: return total page count
    return len(all_text_parts)


def parse_pdf(file_bytes: bytes, filename: str) -> ManuscriptMetadata:
    """Extract metadata from a PDF manuscript."""
    pdf = pdfplumber.open(io.BytesIO(file_bytes))

    pages = pdf.pages
    total_pages = len(pages)

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

    # Page count: count only up to references, excluding appendix
    page_count = _find_page_before_appendix(all_text_parts)

    # Detect most common font and size
    detected_font = font_counter.most_common(1)[0][0] if font_counter else None
    detected_size = size_counter.most_common(1)[0][0] if size_counter else None

    # Detect line spacing from inter-line distances
    detected_line_spacing = _detect_line_spacing(pages) if len(pages) >= 2 else None

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
            # Strategy 2: First-page text extraction with section-heading end markers
            # Only search first 2 pages to avoid matching body text
            first_pages_text = "\n".join(all_text_parts[:2])
            fp_lower = first_pages_text.lower()
            abs_start = fp_lower.find("abstract")
            if abs_start >= 0:
                abs_text_after = first_pages_text[abs_start + len("abstract"):]
                # Strip leading colon, dash, period, whitespace
                abs_text_after = abs_text_after.lstrip(":.- \t\n")

                abs_end = len(abs_text_after)
                # Only match end markers at the START of a line (section headings)
                import re as _re
                heading_patterns = [
                    r"\n\s*(?:1[\.\)]?\s+)?introduction\b",
                    r"\n\s*keywords?\s*:",
                    r"\n\s*key\s+words?\s*:",
                    r"\n\s*jel\s+(?:codes?|classification)",
                    r"\n\s*(?:1[\.\)])\s+[A-Z]",  # "1. Section" or "1) Section"
                    r"\n\s*i[\.\)]\s+[A-Z]",  # "I. Section"
                    r"\n\s*literature\s+review\b",
                    r"\n\s*(?:\d+[\.\)]?\s+)?background\b",
                    r"\n\s*(?:\d+[\.\)]?\s+)?motivation\b",
                    r"\n\s*(?:\d+[\.\)]?\s+)?methods?\b",
                    r"\n\s*(?:\d+[\.\)]?\s+)?methodology\b",
                    r"\n\s*table\s+of\s+contents\b",
                ]
                for pattern in heading_patterns:
                    match = _re.search(pattern, abs_text_after.lower())
                    if match and 0 < match.start() < abs_end:
                        abs_end = match.start()

                # Double newline can indicate end of abstract
                double_nl = abs_text_after.find("\n\n")
                if double_nl > 0:
                    # Only use double newline if it gives a reasonable abstract
                    candidate_wc = len(abs_text_after[:double_nl].split())
                    if 30 <= candidate_wc <= 400 and double_nl < abs_end:
                        abs_end = double_nl

                # Check for footnote markers — but only cut if the abstract
                # already has a reasonable length (avoids truncating when the
                # footnote symbol appears close to the end of the abstract)
                for fn_marker in ["\n*", "\n†", "\n‡", "\n∗"]:
                    idx = abs_text_after.find(fn_marker)
                    if 0 < idx < abs_end:
                        candidate_wc = len(abs_text_after[:idx].split())
                        if candidate_wc >= 30:
                            abs_end = idx

                abstract_text = abs_text_after[:abs_end].strip()
                abstract_words = abstract_text.split()

                # Validate: abstracts typically 30-400 words
                if 20 <= len(abstract_words) <= 500:
                    abstract_word_count = len(abstract_words)
                elif len(abstract_words) > 500:
                    # Too long — try splitting on double newlines for first paragraph
                    for chunk in abs_text_after.split("\n\n"):
                        chunk = chunk.strip()
                        if chunk and 20 <= len(chunk.split()) <= 500:
                            abstract_word_count = len(chunk.split())
                            break

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

    # Section word counts
    section_word_counts = segment_text(full_text, abstract_word_count=abstract_word_count)

    pdf.close()

    return ManuscriptMetadata(
        filename=filename,
        file_type="pdf",
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
        section_word_counts=section_word_counts,
        raw_text=full_text,
    )
