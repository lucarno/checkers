import re

from ..models import ManuscriptMetadata
from .segmenter import segment_text

SECTION_COMMANDS = re.compile(
    r"\\(?:section|subsection|subsubsection)\*?\{([^}]+)\}", re.IGNORECASE
)

AUTHOR_PATTERNS = [
    r"\\author\{", r"\\affiliation\{", r"\\institute\{",
    r"\\email\{", r"\\address\{", r"\\thanks\{",
]


def parse_latex(file_bytes: bytes, filename: str) -> ManuscriptMetadata:
    """Extract metadata from a LaTeX manuscript."""
    text = file_bytes.decode("utf-8", errors="replace")

    # Strip comments (lines starting with %)
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("%"):
            # Remove inline comments (but not escaped \%)
            line = re.sub(r"(?<!\\)%.*$", "", line)
            lines.append(line)
    clean_text = "\n".join(lines)

    # Extract title
    title_match = re.search(r"\\title\{([^}]+)\}", clean_text)
    title = title_match.group(1).strip() if title_match else None

    # Extract sections
    sections = SECTION_COMMANDS.findall(clean_text)
    detected_sections = [s.strip() for s in sections]

    # Check for abstract environment
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", clean_text, re.DOTALL
    )
    has_abstract = abstract_match is not None
    abstract_word_count = None
    if abstract_match:
        abstract_text = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})?", " ", abstract_match.group(1))
        abstract_word_count = len(abstract_text.split())

    # Word count (strip LaTeX commands for rough count)
    # Extract text between \begin{document} and \end{document}
    doc_match = re.search(
        r"\\begin\{document\}(.*?)\\end\{document\}", clean_text, re.DOTALL
    )
    body = doc_match.group(1) if doc_match else clean_text

    # Strip commands for word counting
    body_text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", body)
    body_text = re.sub(r"[{}\\]", " ", body_text)
    body_text = re.sub(r"\s+", " ", body_text)
    word_count = len(body_text.split())

    # Check for references
    has_references = bool(
        re.search(r"\\bibliography\{|\\begin\{thebibliography\}|\\printbibliography", clean_text)
    )

    # Check for figures
    figure_matches = re.findall(r"\\begin\{figure\}", clean_text)
    figure_count = len(figure_matches)
    has_figures = figure_count > 0

    # Author info
    contains_author_info = any(
        re.search(pat, clean_text) for pat in AUTHOR_PATTERNS
    )

    # Detect font from document class or packages
    detected_font = None
    if re.search(r"\\usepackage.*\{times\}", clean_text):
        detected_font = "Times"
    elif re.search(r"\\usepackage.*\{palatino\}", clean_text):
        detected_font = "Palatino"
    elif re.search(r"\\usepackage.*\{helvet\}", clean_text):
        detected_font = "Helvetica"

    # Detect font size from documentclass
    detected_size = None
    size_match = re.search(r"\\documentclass\[.*?(\d+)pt", clean_text)
    if size_match:
        detected_size = float(size_match.group(1))

    # Detect line spacing
    detected_line_spacing = None
    if re.search(r"\\doublespacing|\\linespread\{1\.6\}", clean_text):
        detected_line_spacing = "double"
    elif re.search(r"\\onehalfspacing|\\linespread\{1\.3\}", clean_text):
        detected_line_spacing = "1.5"

    # Add common section names to detected sections list
    section_lower = [s.lower() for s in detected_sections]
    if has_abstract and "abstract" not in section_lower:
        detected_sections.insert(0, "Abstract")
    if has_references and not any("reference" in s.lower() or "bibliography" in s.lower() for s in detected_sections):
        detected_sections.append("References")

    # Section word counts
    section_word_counts = segment_text(body_text, abstract_word_count=abstract_word_count)

    return ManuscriptMetadata(
        filename=filename,
        file_type="latex",
        word_count=word_count,
        page_count=None,  # Cannot determine from LaTeX source
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
        raw_text=body_text,
    )
