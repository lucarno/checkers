"""Text segmentation: split manuscript text into sections and count words per section."""

import re

from ..models import SectionWordCounts

# Patterns that mark section boundaries (case-insensitive, at start of line)
_ABSTRACT_START = re.compile(
    r"(?:^|\n)\s*abstract\s*[:.]?\s*\n", re.IGNORECASE
)
_ABSTRACT_INLINE = re.compile(
    r"(?:^|\n)\s*abstract\s*[:.\s—-]+", re.IGNORECASE
)

_BODY_START_PATTERNS = [
    r"(?:1[\.\)]?\s+)?introduction\b",
    r"i[\.\)]\s+introduction\b",
    r"literature\s+review\b",
    r"(?:1[\.\)]?\s+)?background\b",
    r"(?:1[\.\)]?\s+)?motivation\b",
    r"(?:1[\.\)]?\s+)?overview\b",
    r"1[\.\)]\s+[A-Z]",          # "1. Any Section Title" or "1) Any..."
    r"I[\.\)]\s+[A-Z]",          # "I. Any Section Title"
]
_BODY_START = re.compile(
    r"(?:^|\n)\s*(?:" + "|".join(_BODY_START_PATTERNS) + r")",
    re.IGNORECASE,
)

_REFERENCES_START = re.compile(
    r"(?:^|\n)\s*(?:references|bibliography|works\s+cited|literature\s+cited)\s*\n",
    re.IGNORECASE,
)

_APPENDIX_START = re.compile(
    r"(?:^|\n)\s*(?:appendix|appendices|online\s+appendix|supplementary|internet\s+appendix)\b",
    re.IGNORECASE,
)

_FOOTNOTE_PATTERNS = re.compile(
    r"(?:^|\n)\s*(?:notes|endnotes|footnotes)\s*\n",
    re.IGNORECASE,
)

# Footnote-style acknowledgments or table/figure notes in the text
_TABLE_FIGURE_NOTE = re.compile(
    r"(?:^|\n)\s*(?:note|notes|source|sources)\s*[:.]",
    re.IGNORECASE,
)


def _wc(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())


def segment_text(full_text: str, abstract_word_count: int | None = None) -> SectionWordCounts:
    """Segment manuscript text into sections and return word counts.

    The segmentation finds approximate boundaries for:
    - title_page: everything before abstract (or before body start if no abstract)
    - abstract: the abstract section
    - body: from first body section heading through end of main text
    - references: the reference list
    - footnotes: any notes/endnotes section (between body and references)
    - appendix: anything after references / appendix heading

    If abstract_word_count is provided (from the parser's more accurate
    extraction), it is used to find the abstract boundary instead of
    heuristic heading detection.
    """
    text = full_text
    text_lower = text.lower()
    length = len(text)

    # Find section boundaries (positions in text)
    abstract_pos = None
    body_pos = None
    notes_pos = None
    references_pos = None
    appendix_pos = None

    # Abstract
    m = _ABSTRACT_START.search(text)
    if not m:
        m = _ABSTRACT_INLINE.search(text)
    if m:
        abstract_pos = m.start()

    # Body start (Introduction, etc.)
    m = _BODY_START.search(text)
    if m:
        body_pos = m.start()

    # Notes/Endnotes section
    m = _FOOTNOTE_PATTERNS.search(text)
    if m:
        # Only count if it appears after body start and looks like a section heading
        if body_pos is None or m.start() > body_pos:
            notes_pos = m.start()

    # References
    m = _REFERENCES_START.search(text)
    if m:
        references_pos = m.start()

    # Appendix
    m = _APPENDIX_START.search(text)
    if m:
        appendix_pos = m.start()

    # Now assign text ranges to sections
    # Order: title_page | abstract | body | [notes] | references | [appendix]

    # Determine abstract end = body start or next section
    abstract_end = None
    if abstract_pos is not None:
        # Find end of abstract heading line to get the actual abstract text start
        heading_end = text.find("\n", abstract_pos + 1)
        if heading_end < 0:
            heading_end = abstract_pos + 10
        abstract_text_start = heading_end

        # Strategy 1: If we have a reliable abstract word count from the parser,
        # use it to find the boundary by counting words forward
        if abstract_word_count and abstract_word_count > 0:
            words_seen = 0
            # Walk character by character from abstract_text_start, counting words
            in_word = False
            for i in range(abstract_text_start, min(abstract_text_start + 8000, length)):
                c = text[i]
                if c.isspace():
                    if in_word:
                        words_seen += 1
                        if words_seen >= abstract_word_count:
                            # Find the end of the current line/paragraph
                            nl = text.find("\n", i)
                            abstract_end = nl if nl > 0 else i
                            break
                    in_word = False
                else:
                    in_word = True

        # Strategy 2: Use body start heading if found
        if abstract_end is None and body_pos is not None and body_pos > abstract_pos:
            abstract_end = body_pos

        # Strategy 3: Heuristic fallback
        if abstract_end is None:
            # Try double newline within first 3000 chars after abstract
            search_region = text[abstract_text_start:abstract_text_start + 4000]
            double_nl = search_region.find("\n\n")
            if double_nl > 0:
                candidate = search_region[:double_nl]
                wc = _wc(candidate)
                if 20 <= wc <= 500:
                    abstract_end = abstract_text_start + double_nl
            if abstract_end is None:
                for pos in [notes_pos, references_pos, appendix_pos]:
                    if pos is not None and pos > abstract_pos:
                        abstract_end = pos
                        break
                if abstract_end is None:
                    abstract_end = min(abstract_text_start + 3000, length)

    # Title page: everything before abstract (or before body if no abstract)
    title_page_end = abstract_pos or body_pos or 0
    title_page_text = text[:title_page_end] if title_page_end > 0 else ""

    # Abstract text
    abstract_text = ""
    if abstract_pos is not None and abstract_end is not None:
        abstract_text = text[abstract_pos:abstract_end]

    # Body text: from body_pos (or abstract_end) to notes/references/appendix
    body_start = body_pos
    if body_start is None and abstract_end is not None:
        body_start = abstract_end
    elif body_start is None:
        body_start = title_page_end

    # Body ends at notes, references, or appendix (whichever comes first after body)
    body_end = length
    for pos in sorted(filter(None, [notes_pos, references_pos, appendix_pos])):
        if pos > body_start:
            body_end = pos
            break

    body_text = text[body_start:body_end]

    # Notes text (between body and references)
    notes_text = ""
    if notes_pos is not None:
        notes_end = length
        for pos in sorted(filter(None, [references_pos, appendix_pos])):
            if pos > notes_pos:
                notes_end = pos
                break
        if notes_end > body_end:
            notes_text = text[notes_pos:notes_end]

    # References text
    references_text = ""
    if references_pos is not None:
        refs_end = appendix_pos if appendix_pos and appendix_pos > references_pos else length
        references_text = text[references_pos:refs_end]

    # Appendix text
    appendix_text = ""
    if appendix_pos is not None:
        appendix_text = text[appendix_pos:]

    # Count words
    counts = SectionWordCounts(
        title_page=_wc(title_page_text),
        abstract=_wc(abstract_text),
        body=_wc(body_text),
        references=_wc(references_text),
        footnotes=_wc(notes_text),
        appendix=_wc(appendix_text),
    )
    counts.total = (counts.title_page + counts.abstract + counts.body +
                    counts.references + counts.footnotes + counts.appendix)

    return counts
