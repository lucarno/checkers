import re
from .models import JournalRules, ManuscriptMetadata, CheckItem, CheckResult, CheckStatus


def _detect_citation_style(text: str) -> str | None:
    """Heuristic detection of citation style from manuscript text.

    Distinguishes between:
    - Numbered / Vancouver: [1] or superscript numbered references
    - APA: (Author, 2020) with comma before year; refs as Author, A. B. (2020). Title.
    - Chicago author-date: (Author 2020) no comma; refs as Author, First. 2020. Title.
    - Harvard: (Author 2020) or (Author 2020: 45) with colon for pages;
               refs without parens around year: Author, F. 2020. Title.
    """
    # Extract references section
    text_lower = text.lower()
    ref_start = max(text_lower.rfind("references"), text_lower.rfind("bibliography"),
                    text_lower.rfind("literature cited"))
    ref_section = text[ref_start:] if ref_start > 0 else ""

    # ── In-text citation patterns ──

    # Numbered: [1], [2,3], [1-3]
    numbered_cites = len(re.findall(r"\[[\d,\s\-–]+\]", text))

    # Author-year with comma before year — APA style: (Author, 2020) or (Author & Author, 2020)
    apa_inline = len(re.findall(
        r"\([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?(?:\s+et\s+al\.)?,\s*\d{4}", text))

    # Author-year without comma — Chicago/Harvard: (Author 2020)
    no_comma_inline = len(re.findall(
        r"\([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?(?:\s+et\s+al\.)?\s+\d{4}", text))

    # Harvard-style page refs with colon: (Author 2020: 45) or (Author 2020, 45)
    harvard_page_cites = len(re.findall(
        r"\([A-Z][a-z]+\s+\d{4}:\s*\d+", text))

    # Chicago-style page refs with comma: (Author 2020, 45) — note no colon
    chicago_page_cites = len(re.findall(
        r"\([A-Z][a-z]+\s+\d{4},\s*\d+\)", text))

    # Semicolon-separated multiple cites (common in all author-date styles)
    total_author_year = apa_inline + no_comma_inline

    # ── Reference list patterns ──

    # APA: Author, A. B. (2020). Title of work.
    # Key: year in parentheses followed by period
    apa_refs = len(re.findall(r"\(\d{4}[a-z]?\)\.\s", ref_section)) if ref_section else 0

    # Chicago author-date list: Author, First. 2020. "Title" or Author, First. 2020. Title.
    # Key: year followed by period (no parens around year), often with quoted article titles
    chicago_refs = len(re.findall(
        r"[A-Z][a-z]+,\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?\.\s+\d{4}[a-z]?\.\s", ref_section)) if ref_section else 0

    # Harvard list: Author, F. 2020. or Author, F. (2020) — varies, but often no parens
    # Harvard often uses initials: Author, F.M. 2020.
    harvard_refs = len(re.findall(
        r"[A-Z][a-z]+,?\s+[A-Z]\.?\s*[A-Z]?\.?\s+\(\d{4}\)", ref_section)) if ref_section else 0

    # Vancouver/Numbered: 1. Author... or [1] Author... or numbered list
    vancouver_refs = len(re.findall(
        r"(?:^\s*\d+[\.\)]\s|\[\d+\]\s)", ref_section, re.MULTILINE)) if ref_section else 0

    # ── Decision logic ──

    # Numbered styles (Vancouver or generic numbered)
    if numbered_cites > 5 and numbered_cites > total_author_year * 2:
        if vancouver_refs > 3:
            return "Vancouver"
        return "Numbered"

    # Clearly APA: comma before year in-text + year in parens in refs
    if apa_inline > 3 and apa_inline > no_comma_inline:
        return "APA"

    # Author-year without comma: Chicago or Harvard
    if total_author_year > 3 or no_comma_inline > 3:
        # Harvard page citations use colon (Author 2020: 45)
        if harvard_page_cites > chicago_page_cites and harvard_page_cites >= 2:
            return "Harvard"

        # Check reference list format to distinguish
        if ref_section:
            # APA refs have year in parens: (2020).
            if apa_refs > chicago_refs and apa_refs > 3:
                return "APA"
            # Chicago refs have year without parens: 2020.
            if chicago_refs > apa_refs and chicago_refs > 2:
                return "Chicago"
            # Harvard refs can look like either — use inline cues
            if harvard_page_cites >= 2:
                return "Harvard"

        # Fallback: no comma = likely Chicago (most common in econ/polisci)
        if no_comma_inline > apa_inline:
            return "Chicago"
        return "APA"

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
                    message=f"Citation style detected as {detected_style}, matching required {required}.",
                ))
            else:
                checks.append(CheckItem(
                    name="Citation Style",
                    status=CheckStatus.WARNING,
                    message=f"Citation style detected as {detected_style}, but journal requires {required}.",
                    details=f"Detected {detected_style}-style patterns in citations and reference list. "
                            f"If the journal uses a variant of {required} (e.g., APSA or AEA style based on {required}), "
                            f"this may still be correct — verify against the journal's style guide.",
                ))
        else:
            checks.append(CheckItem(
                name="Citation Style",
                status=CheckStatus.SKIPPED,
                message=f"Could not detect citation style from manuscript text. Required: {required}.",
            ))

    # References check
    checks.append(CheckItem(
        name="References",
        status=CheckStatus.PASS if metadata.has_references else CheckStatus.WARNING,
        message="References section found." if metadata.has_references else "No references/bibliography section detected.",
    ))

    # Structural checks derived from additional_notes
    if rules.additional_notes and metadata.raw_text:
        text_lower = metadata.raw_text.lower()
        notes_text = " ".join(rules.additional_notes).lower()

        # Acknowledgment placement check
        if "acknowledgment" in notes_text or "acknowledgement" in notes_text:
            ack_after_refs = "acknowledgment" in notes_text and "after reference" in notes_text
            ack_not_footnote = "not as a numbered note" in notes_text or "not as a footnote" in notes_text
            if ack_after_refs or ack_not_footnote:
                # Find positions of acknowledgments and references in manuscript
                ack_pos = max(text_lower.rfind("acknowledgments"), text_lower.rfind("acknowledgements"),
                              text_lower.rfind("acknowledgment"), text_lower.rfind("acknowledgement"))
                ref_pos = max(text_lower.rfind("references"), text_lower.rfind("bibliography"))
                has_ack = ack_pos > 0
                if has_ack and ref_pos > 0:
                    if ack_pos > ref_pos:
                        checks.append(CheckItem(
                            name="Acknowledgment Placement",
                            status=CheckStatus.PASS,
                            message="Acknowledgments section appears after references.",
                        ))
                    else:
                        checks.append(CheckItem(
                            name="Acknowledgment Placement",
                            status=CheckStatus.WARNING,
                            message="Acknowledgments appear before references. This journal requires acknowledgments after the reference list, not as a numbered note.",
                        ))
                elif not has_ack:
                    checks.append(CheckItem(
                        name="Acknowledgment Placement",
                        status=CheckStatus.WARNING,
                        message="No acknowledgments section detected. This journal expects acknowledgments at the end of the manuscript after the reference list.",
                    ))

        # Footnotes vs endnotes check
        if "endnotes" in notes_text or "footnotes" in notes_text:
            uses_footnotes_rule = "use footnotes, not endnotes" in notes_text or "footnotes at bottom" in notes_text
            uses_endnotes_rule = "uses endnotes, not footnotes" in notes_text or "endnotes, not footnotes" in notes_text
            if uses_endnotes_rule:
                # Check if manuscript has footnote markers (hard to distinguish in extracted text,
                # but we can check for "Notes" section before references)
                notes_heading = re.search(r"(?:^|\n)\s*(?:end\s*)?notes\s*\n", text_lower)
                if not notes_heading:
                    checks.append(CheckItem(
                        name="Notes Format",
                        status=CheckStatus.WARNING,
                        message="This journal requires endnotes (not footnotes). Verify that notes are placed as endnotes before the references.",
                    ))

        # Word count on title/front page
        if "word count" in notes_text and ("title page" in notes_text or "front page" in notes_text or "first page" in notes_text):
            # Check first ~500 chars for word count
            first_page_text = metadata.raw_text[:1500].lower()
            has_word_count = bool(re.search(r"word\s*count\s*[:=]?\s*[\d,]+", first_page_text)) or \
                             bool(re.search(r"[\d,]+\s*words?\b", first_page_text))
            if has_word_count:
                checks.append(CheckItem(
                    name="Word Count on Title Page",
                    status=CheckStatus.PASS,
                    message="Word count found on title/front page.",
                ))
            else:
                checks.append(CheckItem(
                    name="Word Count on Title Page",
                    status=CheckStatus.WARNING,
                    message="Word count not found on the title page. This journal requires the word count to appear on the front page.",
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
