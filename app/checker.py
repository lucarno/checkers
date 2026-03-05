from .models import JournalRules, ManuscriptMetadata, CheckItem, CheckResult, CheckStatus


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
            if metadata.abstract_word_count <= rules.abstract_word_limit:
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
