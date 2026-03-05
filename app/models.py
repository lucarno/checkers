from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class CitationStyle(str, Enum):
    APA = "APA"
    CHICAGO = "Chicago"
    MLA = "MLA"
    HARVARD = "Harvard"
    VANCOUVER = "Vancouver"
    NUMBERED = "Numbered"
    OTHER = "Other"
    UNKNOWN = "Unknown"


class FigureRequirements(BaseModel):
    formats: list[str] = Field(default_factory=list, description="Accepted image formats (e.g., TIFF, EPS, PDF)")
    min_resolution_dpi: Optional[int] = None
    max_file_size_mb: Optional[float] = None
    placement: Optional[str] = None  # e.g., "end of document", "inline"


class JournalRules(BaseModel):
    journal_name: Optional[str] = None
    word_limit: Optional[int] = None
    page_limit: Optional[int] = None
    font: Optional[str] = None
    font_size: Optional[float] = None
    line_spacing: Optional[str] = None  # e.g., "double", "1.5", "single"
    margins: Optional[str] = None  # e.g., "1 inch all sides"
    required_sections: list[str] = Field(default_factory=list)
    abstract_word_limit: Optional[int] = None
    citation_style: Optional[CitationStyle] = None
    figure_requirements: Optional[FigureRequirements] = None
    reference_format: Optional[str] = None
    anonymization_required: Optional[bool] = None
    additional_notes: list[str] = Field(default_factory=list)


class SectionWordCounts(BaseModel):
    """Word counts for individual manuscript sections."""
    title_page: int = 0       # Title, authors, affiliations (before abstract)
    abstract: int = 0         # Abstract section
    body: int = 0             # Main text (introduction through conclusion)
    references: int = 0       # Reference list
    footnotes: int = 0        # Footnotes / endnotes / table and figure notes
    appendix: int = 0         # Appendices
    total: int = 0            # Sum of all sections


class ManuscriptMetadata(BaseModel):
    filename: str
    file_type: str  # "pdf", "docx", "latex"
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    detected_font: Optional[str] = None
    detected_font_size: Optional[float] = None
    detected_line_spacing: Optional[str] = None
    detected_sections: list[str] = Field(default_factory=list)
    has_abstract: bool = False
    abstract_word_count: Optional[int] = None
    has_references: bool = False
    has_figures: bool = False
    figure_count: int = 0
    contains_author_info: bool = False
    title: Optional[str] = None
    section_word_counts: Optional[SectionWordCounts] = None
    raw_text: Optional[str] = Field(None, exclude=True)


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIPPED = "skipped"


class CheckItem(BaseModel):
    name: str
    status: CheckStatus
    message: str
    details: Optional[str] = None


class CheckResult(BaseModel):
    total_checks: int
    passed: int
    failed: int
    warnings: int
    skipped: int
    checks: list[CheckItem]
    section_word_counts: Optional[SectionWordCounts] = None
