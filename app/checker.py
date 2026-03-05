import re
from .models import JournalRules, ManuscriptMetadata, CheckItem, CheckResult, CheckStatus


def _detect_citation_style(text: str) -> str | None:
    """Heuristic detection of citation style from manuscript text."""
    # Extract references section
    text_lower = text.lower()
    ref_start = max(text_lower.rfind("references"), text_lower.rfind("bibliography"))
    ref_section = text[ref_start:] if ref_start > 0 else ""

    # In-text citation patterns
    # Numbered: [1], [2,3], [1-3]
    numbered_cites = len(re.findall(r"\[[\d,\s\-–]+\]", text))
    # Author-year: (Author, 2020) or (Author 2020)
    author_year_cites = len(re.findall(r"\([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?,?\s*\d{4}", text))

    # Reference list patterns
    if ref_section:
        # APA: Author, A. B. (2020). Title.
        apa_refs = len(re.findall(r"\(\d{4}[a-z]?\)\.\s", ref_section))
        # Vancouver/Numbered: 1. Author... or [1] Author...
        vancouver_refs = len(re.findall(r"(?:^\s*\d+[\.\)]\s|\[\d+\]\s)", ref_section, re.MULTILINE))
        # Chicago author-date is similar to APA in list format
        # Harvard is also author-year with slight differences
    else:
        apa_refs = 0
        vancouver_refs = 0

    # Decision logic
    if numbered_cites > 5 and numbered_cites > author_year_cites * 2:
        if vancouver_refs > 3:
            return "Vancouver"
        return "Numbered"

    if author_year_cites > 3:
        if apa_refs > 3:
            return "APA"
        # Could be Harvard or Chicago — hard to distinguish without deeper analysis
        # Check for "et al." usage and format
        if re.search(r"\(\w+\s+et\s+al\.\s*,?\s*\d{4}\)", text):
            return "APA"
        return "Harvard"

    # Check for footnote-based citations (Chicago notes-bibliography)
    footnote_cites = len(re.findall(r"(?:Ibid|ibid|Op\.\s*cit)", text))
    if footnote_cites > 2:
        return "Chicago"

    return None


def check_manuscript(rules: JournalRules, metadata: ManuscriptMetadata) -> CheckResult:
    """Compare manuscript metadata against journal rules and return results."""
    checks: list[CheckItem] = []

    # Word count check
    if rules.word_limit and metadata.word_count:
        if metadata.word_count <= rules.word_limit:
            checks.append(CheckItem(
                name="Word Count",
                status=CheckStatus.PASS,
                message=f"Word count ({metadata.word_count:,}) is within the limit ({rules.word_limit:,}).",
            ))
        else:
            over = metadata.word_count - rules.word_limit
            checks.append(CheckItem(
                name="Word Count",
                status=CheckStatus.FAIL,
                message=f"Word count ({metadata.word_count:,}) exceeds the limit ({rules.word_limit:,}) by {over:,} words.",
            ))
    elif rules.word_limit and not metadata.word_count:
        checks.append(CheckItem(
            name="Word Count",
            status=CheckStatus.SKIPPED,
            message="Could not determine word count from the manuscript.",
        ))

    # Page count check
    if rules.page_limit and metadata.page_count:
        if metadata.page_count <= rules.page_limit:
            checks.append(CheckItem(
                name="Page Count",
                status=CheckStatus.PASS,
                message=f"Page count ({metadata.page_count}) is within the limit ({rules.page_limit}).",
            ))
        else:
            checks.append(CheckItem(
                name="Page Count",
                status=CheckStatus.FAIL,
                message=f"Page count ({metadata.page_count}) exceeds the limit ({rules.page_limit}).",
            ))
    elif rules.page_limit and not metadata.page_count:
        checks.append(CheckItem(
            name="Page Count",
            status=CheckStatus.SKIPPED,
            message="Could not determine page count from the manuscript.",
        ))

    # Abstract word limit
    if rules.abstract_word_limit:
        if not metadata.has_abstract:
            checks.append(CheckItem(
                name="Abstract",
                status=CheckStatus.FAIL,
                message="No abstract detected in the manuscript.",
            ))
        elif metadata.abstract_word_count:
            # Sanity check: if detected count is implausibly high, the parser
            # likely failed to find the abstract boundary
            max_plausible = max(rules.abstract_word_limit * 3, 500)
            if metadata.abstract_word_count > max_plausible:
                checks.append(CheckItem(
                    name="Abstract Word Count",
                    status=CheckStatus.WARNING,
                    message=f"Abstract detected but word count ({metadata.abstract_word_count}) seems too high — "
                            f"the parser may have included body text. Please verify manually (limit: {rules.abstract_word_limit}).",
                ))
            elif metadata.abstract_word_count <= rules.abstract_word_limit:
                checks.append(CheckItem(
                    name="Abstract Word Count",
                    status=CheckStatus.PASS,
                    message=f"Abstract ({metadata.abstract_word_count} words) is within the limit ({rules.abstract_word_limit}).",
                ))
            else:
                over = metadata.abstract_word_count - rules.abstract_word_limit
                checks.append(CheckItem(
                    name="Abstract Word Count",
                    status=CheckStatus.FAIL,
                    message=f"Abstract ({metadata.abstract_word_count} words) exceeds the limit ({rules.abstract_word_limit}) by {over} words.",
                ))
        else:
            checks.append(CheckItem(
                name="Abstract Word Count",
                status=CheckStatus.WARNING,
                message="Abstract detected but could not count words accurately.",
            ))

    # Required sections check
    if rules.required_sections:
        manuscript_sections_lower = [s.lower() for s in metadata.detected_sections]
        for req_section in rules.required_sections:
            found = any(req_section.lower() in s for s in manuscript_sections_lower)
            if found:
                checks.append(CheckItem(
                    name=f"Section: {req_section}",
                    status=CheckStatus.PASS,
                    message=f"Required section '{req_section}' found.",
                ))
            else:
                checks.append(CheckItem(
                    name=f"Section: {req_section}",
                    status=CheckStatus.FAIL,
                    message=f"Required section '{req_section}' not found in manuscript.",
                ))

    # Font check
    if rules.font and metadata.detected_font:
        rule_font_lower = rules.font.lower()
        detected_lower = metadata.detected_font.lower()
        if rule_font_lower in detected_lower or detected_lower in rule_font_lower:
            checks.append(CheckItem(
                name="Font",
                status=CheckStatus.PASS,
                message=f"Detected font '{metadata.detected_font}' matches required '{rules.font}'.",
            ))
        else:
            checks.append(CheckItem(
                name="Font",
                status=CheckStatus.WARNING,
                message=f"Detected font '{metadata.detected_font}' may not match required '{rules.font}'.",
                details="Font detection from PDFs is approximate. Please verify manually.",
            ))
    elif rules.font and not metadata.detected_font:
        checks.append(CheckItem(
            name="Font",
            status=CheckStatus.SKIPPED,
            message=f"Could not detect font. Required: '{rules.font}'.",
        ))

    # Font size check
    if rules.font_size and metadata.detected_font_size:
        if abs(metadata.detected_font_size - rules.font_size) < 0.5:
            checks.append(CheckItem(
                name="Font Size",
                status=CheckStatus.PASS,
                message=f"Detected font size ({metadata.detected_font_size}pt) matches required ({rules.font_size}pt).",
            ))
        else:
            checks.append(CheckItem(
                name="Font Size",
                status=CheckStatus.WARNING,
                message=f"Detected font size ({metadata.detected_font_size}pt) differs from required ({rules.font_size}pt).",
                details="Font size detection may be approximate.",
            ))
    elif rules.font_size and not metadata.detected_font_size:
        checks.append(CheckItem(
            name="Font Size",
            status=CheckStatus.SKIPPED,
            message=f"Could not detect font size. Required: {rules.font_size}pt.",
        ))

    # Line spacing check
    if rules.line_spacing and metadata.detected_line_spacing:
        if rules.line_spacing.lower() == metadata.detected_line_spacing.lower():
            checks.append(CheckItem(
                name="Line Spacing",
                status=CheckStatus.PASS,
                message=f"Line spacing '{metadata.detected_line_spacing}' matches requirement.",
            ))
        else:
            checks.append(CheckItem(
                name="Line Spacing",
                status=CheckStatus.WARNING,
                message=f"Detected line spacing '{metadata.detected_line_spacing}' may not match required '{rules.line_spacing}'.",
            ))
    elif rules.line_spacing and not metadata.detected_line_spacing:
        checks.append(CheckItem(
            name="Line Spacing",
            status=CheckStatus.SKIPPED,
            message=f"Could not detect line spacing. Required: '{rules.line_spacing}'.",
        ))

    # Anonymization check
    if rules.anonymization_required is True:
        if metadata.contains_author_info:
            checks.append(CheckItem(
                name="Anonymization",
                status=CheckStatus.FAIL,
                message="Author-identifying information detected. This journal requires anonymized submissions.",
                details="Found indicators such as university names, email addresses, or author affiliations.",
            ))
        else:
            checks.append(CheckItem(
                name="Anonymization",
                status=CheckStatus.PASS,
                message="No obvious author-identifying information detected.",
                details="Please double-check manually for self-citations or other identifying information.",
            ))

    # Citation style check
    if rules.citation_style and rules.citation_style.value not in ("Unknown", "Other") and metadata.raw_text:
        detected_style = _detect_citation_style(metadata.raw_text)
        required = rules.citation_style.value
        if detected_style:
            if detected_style.lower() == required.lower():
                checks.append(CheckItem(
                    name="Citation Style",
                    status=CheckStatus.PASS,
                    message=f"Citation style appears to be {detected_style}, matching required {required}.",
                ))
            else:
                checks.append(CheckItem(
                    name="Citation Style",
                    status=CheckStatus.WARNING,
                    message=f"Citation style appears to be {detected_style}, but journal requires {required}.",
                    details="Citation style detection is heuristic-based. Please verify manually.",
                ))
        else:
            checks.append(CheckItem(
                name="Citation Style",
                status=CheckStatus.SKIPPED,
                message=f"Could not detect citation style. Required: {required}.",
            ))

    # References check
    checks.append(CheckItem(
        name="References",
        status=CheckStatus.PASS if metadata.has_references else CheckStatus.WARNING,
        message="References section found." if metadata.has_references else "No references/bibliography section detected.",
    ))

    # Compute summary
    passed = sum(1 for c in checks if c.status == CheckStatus.PASS)
    failed = sum(1 for c in checks if c.status == CheckStatus.FAIL)
    warnings = sum(1 for c in checks if c.status == CheckStatus.WARNING)
    skipped = sum(1 for c in checks if c.status == CheckStatus.SKIPPED)

    return CheckResult(
        total_checks=len(checks),
        passed=passed,
        failed=failed,
        warnings=warnings,
        skipped=skipped,
        checks=checks,
    )
