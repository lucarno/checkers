"""Tests for manuscript_analyzer: marker finding, heading fallback, and boundary building."""

import json
from unittest.mock import patch, MagicMock

import pytest

from app.manuscript_analyzer import (
    _find_marker,
    _find_heading,
    _find_body_after_abstract,
    _normalize_unicode,
    _wc,
    analyze_manuscript,
    apply_llm_analysis,
)
from app.models import ManuscriptMetadata, SectionWordCounts


# ── _normalize_unicode ──

class TestNormalizeUnicode:
    def test_ligatures(self):
        assert _normalize_unicode("e\ufb03cient") == "efficient"
        assert _normalize_unicode("\ufb01nd") == "find"
        assert _normalize_unicode("\ufb02ow") == "flow"
        assert _normalize_unicode("\ufb00ect") == "ffect"
        assert _normalize_unicode("\ufb04") == "ffl"

    def test_smart_quotes(self):
        assert _normalize_unicode("\u201chello\u201d") == '"hello"'
        assert _normalize_unicode("\u2018world\u2019") == "'world'"

    def test_dashes(self):
        assert _normalize_unicode("a\u2013b") == "a-b"
        assert _normalize_unicode("a\u2014b") == "a-b"

    def test_special_spaces(self):
        assert _normalize_unicode("a\u00a0b") == "a b"
        assert _normalize_unicode("a\u200bb") == "ab"  # zero-width space removed

    def test_plain_text_unchanged(self):
        assert _normalize_unicode("hello world") == "hello world"


# ── _find_marker ──

class TestFindMarker:
    def test_exact_match(self):
        text = "This is the abstract section of the paper."
        assert _find_marker(text, "abstract section") == 12

    def test_empty_marker(self):
        assert _find_marker("some text", "") == -1
        assert _find_marker("some text", None) == -1

    def test_case_insensitive(self):
        text = "ABSTRACT This paper examines"
        pos = _find_marker(text, "abstract this paper")
        assert pos == 0

    def test_flexible_whitespace(self):
        text = "Abstract\n\nThis paper examines the relationship"
        pos = _find_marker(text, "Abstract This paper examines")
        assert pos >= 0

    def test_ligature_in_text(self):
        """LLM quotes 'efficient' but PDF text has fi-ligature."""
        text = "This is an e\ufb03cient method for analysis"
        pos = _find_marker(text, "This is an efficient method")
        assert pos == 0

    def test_ligature_in_marker(self):
        """Marker has ligature but text has normal chars."""
        text = "This is an efficient method for analysis"
        pos = _find_marker(text, "This is an e\ufb03cient method")
        assert pos == 0

    def test_smart_quotes_in_text(self):
        text = 'He said \u201chello\u201d to the world'
        pos = _find_marker(text, 'He said "hello" to')
        assert pos == 0

    def test_em_dash_vs_hyphen(self):
        text = "This\u2014an important point\u2014shows that"
        pos = _find_marker(text, "This-an important point-shows")
        assert pos == 0

    def test_first_few_words_fallback(self):
        text = "Introduction to the study of complex systems in nature"
        marker = "Introduction to the study of complex systems and their behavior"
        # Last words differ, but first 6 should match
        pos = _find_marker(text, marker)
        assert pos == 0

    def test_first_3_words_fallback(self):
        text = "References\n\nSmith, J. (2020). The effects of climate."
        marker = "References Smith, J. (2020). The effects of climate on"
        pos = _find_marker(text, marker)
        assert pos >= 0

    def test_newline_within_marker(self):
        """PDF line breaks within a sentence."""
        text = "This paper examines\nthe relationship between\nvariables"
        pos = _find_marker(text, "This paper examines the relationship between")
        assert pos == 0

    def test_hyphenation_between_words(self):
        """PDF inserts hyphens/newlines between words."""
        text = "the relationship between-\nvariables in the model"
        pos = _find_marker(text, "the relationship between variables in")
        # The [\s\-]* joining handles hyphen-newline between words
        assert pos == 0

    def test_not_found(self):
        text = "This paper is about climate change"
        assert _find_marker(text, "quantum computing research methods") == -1


# ── _find_heading ──

class TestFindHeading:
    def test_simple_heading_on_own_line(self):
        text = "Some title text\n\nAbstract\nThis paper examines..."
        pos = _find_heading(text, "Abstract", "abstract")
        assert pos >= 0
        assert text[pos:pos+8] == "Abstract"

    def test_heading_with_number(self):
        text = "some text\n1. Introduction\nWe begin by..."
        pos = _find_heading(text, "1. Introduction", "body")
        assert pos >= 0

    def test_references_heading(self):
        text = "end of paper.\n\nReferences\nSmith (2020)..."
        pos = _find_heading(text, "References", "references")
        assert pos >= 0
        assert "References" in text[pos:pos+12]

    def test_references_variant_bibliography(self):
        text = "end of paper.\n\nBibliography\nSmith (2020)..."
        pos = _find_heading(text, "Reference List", "references")
        # Should find "Bibliography" via variant matching
        assert pos >= 0

    def test_appendix_heading(self):
        text = "refs here.\n\nAppendix A\nExtra data..."
        pos = _find_heading(text, "Appendix A", "appendix")
        assert pos >= 0

    def test_appendix_variant_online(self):
        text = "refs here.\n\nOnline Appendix\nExtra data..."
        pos = _find_heading(text, "Supplementary", "appendix")
        # Should find via "online appendix" variant
        assert pos >= 0

    def test_notes_heading(self):
        text = "body text.\n\nEndnotes\n1. Some note..."
        pos = _find_heading(text, "Endnotes", "footnotes")
        assert pos >= 0

    def test_heading_not_found(self):
        text = "This paper has no sections at all just flowing text"
        assert _find_heading(text, "References", "references") == -1

    def test_empty_heading(self):
        assert _find_heading("some text", "", "body") == -1

    def test_heading_with_colon(self):
        text = "some text\nAbstract: This paper..."
        pos = _find_heading(text, "Abstract", "abstract")
        assert pos >= 0


# ── Integration: analyze_manuscript boundary building ──

class TestAnalyzeManuscriptBoundaries:
    """Test the boundary-building logic without calling the LLM."""

    SAMPLE_TEXT = (
        "My Great Paper Title\n"
        "John Doe, University of Example\n"
        "\n"
        "Abstract\n"
        "This paper studies the effect of X on Y. We find significant results "
        "that contribute to the literature on Z. Our analysis uses data from "
        "multiple sources spanning twenty years.\n"
        "\n"
        "1. Introduction\n"
        "The study of X has a long history. Previous work by Smith (2020) "
        "established the basic framework. We extend this by considering Y. "
        "The remainder of this paper is organized as follows. Section 2 "
        "describes our data. Section 3 presents results. Section 4 concludes.\n"
        "\n"
        "2. Data\n"
        "We collected data from public archives. The dataset includes 5000 "
        "observations over 20 years. Variables include income, education, "
        "and employment status.\n"
        "\n"
        "3. Results\n"
        "Our regression analysis shows a statistically significant positive "
        "effect of X on Y with a coefficient of 0.45. Robustness checks "
        "confirm our findings across multiple specifications.\n"
        "\n"
        "4. Conclusion\n"
        "We have shown that X significantly affects Y. Future research "
        "should explore additional mechanisms.\n"
        "\n"
        "References\n"
        "Smith, J. (2020). The basics of X. Journal of Examples, 15(2), 100-120.\n"
        "Doe, A. (2019). Understanding Y. Review of Studies, 8(1), 50-75.\n"
        "\n"
        "Appendix A\n"
        "Table A1 shows additional regression results with alternative controls.\n"
    )

    def _mock_llm_response(self, sections, **extra):
        """Create a mock LLM analysis response."""
        analysis = {
            "title": "My Great Paper Title",
            "sections": sections,
            "has_abstract": True,
            "has_references": True,
            "citation_style": "APA",
            "citation_style_details": "Author-year with comma",
            "contains_author_info": True,
            "has_figures": False,
            "figure_count": 0,
            "acknowledgment_location": "not_found",
            "structural_issues": [],
            **extra,
        }
        return analysis

    @patch("app.manuscript_analyzer.anthropic")
    def test_markers_found_correctly(self, mock_anthropic):
        """When markers match exactly, sections are split correctly."""
        sections = [
            {"type": "title_page", "heading": "Title", "start_marker": "My Great Paper Title John Doe University of Example"},
            {"type": "abstract", "heading": "Abstract", "start_marker": "Abstract This paper studies the effect of X on"},
            {"type": "body", "heading": "1. Introduction", "start_marker": "1. Introduction The study of X has a long"},
            {"type": "references", "heading": "References", "start_marker": "References Smith J 2020 The basics of X"},
            {"type": "appendix", "heading": "Appendix A", "start_marker": "Appendix A Table A1 shows additional regression results"},
        ]
        analysis = self._mock_llm_response(
            sections,
            abstract_start_marker="This paper studies the effect of X on Y",
            abstract_end_marker="1. Introduction The study of X has a long",
        )

        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=json.dumps(analysis))]
        )

        result = analyze_manuscript(self.SAMPLE_TEXT, "fake-key")

        swc = result["section_word_counts"]
        assert swc["title_page"] > 0
        assert swc["abstract"] > 0
        assert swc["body"] > 0
        assert swc["references"] > 0
        assert swc["appendix"] > 0
        # Body should NOT include references or appendix
        assert swc["body"] < 200  # body is ~100 words, not 9000+

    @patch("app.manuscript_analyzer.anthropic")
    def test_heading_fallback_when_markers_fail(self, mock_anthropic):
        """When start_markers are garbage, heading fallback still finds sections."""
        sections = [
            {"type": "title_page", "heading": "Title", "start_marker": "COMPLETELY WRONG MARKER THAT WONT MATCH"},
            {"type": "abstract", "heading": "Abstract", "start_marker": "GARBAGE MARKER FOR ABSTRACT SECTION"},
            {"type": "body", "heading": "1. Introduction", "start_marker": "WRONG BODY MARKER TEXT HERE"},
            {"type": "references", "heading": "References", "start_marker": "INVALID REFERENCES START MARKER"},
            {"type": "appendix", "heading": "Appendix A", "start_marker": "FAKE APPENDIX MARKER TEXT"},
        ]
        analysis = self._mock_llm_response(
            sections,
            abstract_start_marker="WRONG ABSTRACT START MARKER TEXT",
            abstract_end_marker="WRONG ABSTRACT END MARKER TEXT",
        )

        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=json.dumps(analysis))]
        )

        result = analyze_manuscript(self.SAMPLE_TEXT, "fake-key")

        swc = result["section_word_counts"]
        # The heading fallback should find Abstract, References, Appendix
        assert swc["abstract"] > 0, f"Abstract should have words, got {swc}"
        assert swc["references"] > 0, f"References should have words, got {swc}"
        assert swc["appendix"] > 0, f"Appendix should have words, got {swc}"
        # Body should be reasonable (not include references/appendix)
        assert swc["body"] < 200, f"Body word count too high, likely includes other sections: {swc}"

    @patch("app.manuscript_analyzer.anthropic")
    def test_ligature_markers_still_found(self, mock_anthropic):
        """Markers with ligature differences still match."""
        text_with_ligatures = self.SAMPLE_TEXT.replace("effect", "e\ufb00ect")

        sections = [
            {"type": "title_page", "heading": "Title", "start_marker": "My Great Paper Title"},
            {"type": "abstract", "heading": "Abstract", "start_marker": "Abstract This paper studies the effect of X on"},
            {"type": "body", "heading": "1. Introduction", "start_marker": "1. Introduction The study of X has"},
            {"type": "references", "heading": "References", "start_marker": "References Smith J 2020"},
            {"type": "appendix", "heading": "Appendix A", "start_marker": "Appendix A Table A1"},
        ]
        analysis = self._mock_llm_response(
            sections,
            abstract_start_marker="This paper studies the effect of X on Y",
            abstract_end_marker="1. Introduction The study of X has",
        )

        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=json.dumps(analysis))]
        )

        result = analyze_manuscript(text_with_ligatures, "fake-key")

        swc = result["section_word_counts"]
        assert swc["abstract"] > 0
        assert swc["references"] > 0

    @patch("app.manuscript_analyzer.anthropic")
    def test_abstract_word_count_with_heading_fallback(self, mock_anthropic):
        """Abstract word count works even when abstract markers fail but heading is found."""
        sections = [
            {"type": "title_page", "heading": "Title", "start_marker": "WRONG"},
            {"type": "abstract", "heading": "Abstract", "start_marker": "WRONG"},
            {"type": "body", "heading": "1. Introduction", "start_marker": "WRONG"},
            {"type": "references", "heading": "References", "start_marker": "WRONG"},
        ]
        analysis = self._mock_llm_response(
            sections,
            abstract_start_marker="WRONG ABSTRACT START",
            abstract_end_marker="WRONG ABSTRACT END",
        )

        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=json.dumps(analysis))]
        )

        result = analyze_manuscript(self.SAMPLE_TEXT, "fake-key")

        # Abstract word count should come from the section boundary (heading fallback)
        # since the abstract_start_marker/end_marker both fail
        swc = result["section_word_counts"]
        assert swc["abstract"] > 10, f"Abstract should have meaningful word count: {swc}"
        assert result["abstract_word_count"] == swc["abstract"]


# ── Edge cases ──

class TestEdgeCases:
    def test_wc_empty(self):
        assert _wc("") == 0

    def test_wc_whitespace_only(self):
        assert _wc("   \n\n  ") == 0

    def test_wc_normal(self):
        assert _wc("one two three four five") == 5

    def test_wc_joined_words(self):
        """PDF joined-word tokens should be estimated as multiple words."""
        # "Thispaperexamines" is 3 words joined (17 chars) - under threshold, counts as 1
        # "Thispaperexamineshowdemocratic" is 5 words joined (30 chars) - over threshold
        text = "Thispaperexamineshowdemocraticpoliticians by securing regime trust"
        count = _wc(text)
        # The joined token should count as multiple words, not 1
        assert count > 5, f"Joined words not split: {count}"

    def test_wc_hyphenated_words_not_split(self):
        """Hyphenated words like 'difference-in-differences' should count as 1."""
        text = "using a difference-in-differences design with control variables"
        assert _wc(text) == 7  # 7 whitespace tokens, hyphenated = 1 word

    def test_find_marker_with_non_breaking_space(self):
        text = "Abstract\u00a0\u00a0This paper"
        pos = _find_marker(text, "Abstract This paper")
        assert pos == 0

    def test_find_marker_zero_width_space(self):
        text = "Abstract\u200b This paper"
        pos = _find_marker(text, "Abstract This paper")
        assert pos >= 0

    def test_find_heading_abstract_inline_colon(self):
        """Some papers use 'Abstract:' not 'Abstract' on its own line."""
        text = "Title here\nAbstract: This paper examines"
        pos = _find_heading(text, "Abstract", "abstract")
        assert pos >= 0


# ── Real-world PDF text scenarios ──

# Simulates actual pdfplumber output: words joined without spaces, footnote
# markers, page numbers, etc.
REAL_PDF_TEXT = (
    "The Autocracy Bandwagon:\nElectoral Institutions and Elite Selection in\n"
    "\u2217\nAuthoritarian Regimes\nJohn Smith and Jane Doe\n"
    "[Preliminary Draft. Please do not cite or circulate.]\n"
    "Abstract\n"
    "Thispaperexamineshowdemocraticpoliticianssurviveauthoritarianreversals\n"
    "by securing regime trust. While research shows how authoritarian institutions\n"
    "help dictators share power, we argue that these institutions also allow politi-\n"
    "cianstocrediblyabandonoppositionalambitionsbypubliclyjoiningtheruling\n"
    "party. Using individual-level data from a country's last elections before the\n"
    "coup, and combining a difference-in-differences design with a regression\n"
    "discontinuity approach, we show that left-wing incumbents were more likely\n"
    "to join the ruling party.\n"
    "\u2217We thank participants of APSA and EPSA for helpful feedback.\n"
    "During democratic breakdowns, democratic politicians occupy a paradoxical po-\n"
    "sition: they are democracy's strongest potential defenders yet also its most vulner-\n"
    "able political actors. Their ability to mobilize voters and forge alliances makes\n"
    "them central to efforts to resist authoritarian encroachment.\n"
    "2 Background\n"
    "We examine our argument in the context of a historical case. The country\n"
    "experienced a right-wing military coup that transformed democracy into a\n"
    "durable authoritarian regime lasting over two decades.\n"
    "3 Data\n"
    "We collected data from public archives covering all candidates. The dataset\n"
    "includes individual-level information on party affiliation and vote shares.\n"
    "4 Results\n"
    "Our analysis shows a statistically significant positive effect. Left-wing\n"
    "incumbents were more likely to join the ruling party than unelected leftists.\n"
    "5 Conclusion\n"
    "Political elites who oppose authoritarian leaders play a crucial role in shaping\n"
    "regime consolidation. Yet those excluded from the dictator's initial coalition\n"
    "often face a defining choice between resistance and accommodation.\n"
    "25\n"
    "References\n"
    "Acemoglu, Daron and James A Robinson. 2005. Economic origins of dictatorship\n"
    "and democracy. Cambridge University Press.\n"
    "Smith, John. 2020. Authoritarian institutions and elite behavior. Journal of\n"
    "Politics 82(3): 1045-1060.\n"
    "30\n"
    "A Summary table\n"
    "Avg Median Min Max\n"
    "Member Ruling Party 0.15 0.00 0.00 1.00\n"
    "B Additional results\n"
    "Table B1 shows additional regression results with alternative controls.\n"
)


class TestFindBodyAfterAbstract:
    def test_body_after_footnote(self):
        """Body starts after abstract footnote, not at the footnote itself."""
        abs_pos = REAL_PDF_TEXT.index("Abstract")
        body_pos = _find_body_after_abstract(REAL_PDF_TEXT, abs_pos)
        assert body_pos > 0
        # Should find "During democratic breakdowns", NOT "We thank participants"
        assert REAL_PDF_TEXT[body_pos:body_pos+6] == "During", \
            f"Expected 'During', got: {repr(REAL_PDF_TEXT[body_pos:body_pos+50])}"

    def test_body_after_keywords(self):
        """Body starts after keywords section."""
        text = (
            "Abstract\n"
            "This paper studies X and Y.\n"
            "Keywords: democracy, authoritarianism\n"
            "JEL codes: D72, P16\n"
            "Introduction to the study of X.\n"
        )
        body_pos = _find_body_after_abstract(text, 0)
        assert body_pos > 0
        assert "Introduction" in text[body_pos:body_pos+20]

    def test_body_with_explicit_heading(self):
        """Papers with '1. Introduction' heading should find it via strategy 1."""
        text = (
            "Abstract\n"
            "This paper studies the effect of X on Y.\n"
            "\n"
            "1. Introduction\n"
            "The study of X has a long history.\n"
        )
        body_pos = _find_body_after_abstract(text, 0)
        assert body_pos > 0
        assert "1." in text[body_pos:body_pos+5]

    def test_no_body_found(self):
        """Returns -1 when no body boundary can be detected."""
        text = "Abstract\nThis is all there is."
        assert _find_body_after_abstract(text, 0) == -1


class TestRealPDFBoundaries:
    """Integration tests using realistic PDF-extracted text."""

    def _mock_and_run(self, sections, **extra):
        analysis = {
            "title": "The Autocracy Bandwagon",
            "sections": sections,
            "has_abstract": True,
            "has_references": True,
            "citation_style": "Chicago",
            "citation_style_details": "Author-year without comma",
            "contains_author_info": True,
            "has_figures": False,
            "figure_count": 0,
            "acknowledgment_location": "footnote",
            "structural_issues": [],
            **extra,
        }
        with patch("app.manuscript_analyzer.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client
            mock_client.messages.create.return_value = MagicMock(
                content=[MagicMock(text=json.dumps(analysis))]
            )
            return analyze_manuscript(REAL_PDF_TEXT, "fake-key")

    def test_good_markers(self):
        """LLM returns accurate markers — all sections detected."""
        result = self._mock_and_run(
            sections=[
                {"type": "title_page", "heading": "Title", "start_marker": "The Autocracy Bandwagon Electoral Institutions and Elite Selection"},
                {"type": "abstract", "heading": "Abstract", "start_marker": "Abstract This paper examines how democratic politicians survive"},
                {"type": "body", "heading": "Body", "start_marker": "During democratic breakdowns, democratic politicians occupy a paradoxical"},
                {"type": "references", "heading": "References", "start_marker": "References Acemoglu, Daron and James A Robinson. 2005"},
                {"type": "appendix", "heading": "Appendix", "start_marker": "A Summary table Avg Median Min Max"},
            ],
            abstract_start_marker="This paper examines how democratic politicians survive",
            abstract_end_marker="During democratic breakdowns, democratic politicians occupy",
        )
        swc = result["section_word_counts"]
        assert swc["abstract"] > 0 and swc["abstract"] < 200
        assert swc["body"] > 0
        assert swc["references"] > 0
        assert swc["appendix"] > 0
        assert result["abstract_word_count"] < 200

    def test_hallucinated_markers_body_not_found(self):
        """LLM hallucinates '1. Introduction' but paper has no such heading.
        Body should be inferred from abstract end."""
        result = self._mock_and_run(
            sections=[
                {"type": "title_page", "heading": "Title", "start_marker": "The Autocracy Bandwagon Electoral Institutions"},
                {"type": "abstract", "heading": "Abstract", "start_marker": "Abstract This paper examines how democratic politicians survive"},
                {"type": "body", "heading": "Introduction", "start_marker": "1. Introduction During democratic breakdowns"},
                {"type": "references", "heading": "References", "start_marker": "References 1. Acemoglu Daron and James Robinson"},
                {"type": "appendix", "heading": "Appendix", "start_marker": "Appendix A Summary Statistics Table A1"},
            ],
            abstract_start_marker="This paper examines how democratic politicians survive",
            abstract_end_marker="1. Introduction During democratic breakdowns",
        )
        swc = result["section_word_counts"]
        # Abstract should NOT be thousands of words
        assert swc["abstract"] < 300, f"Abstract too large: {swc['abstract']}"
        # Body should be found via gap-filling
        assert swc["body"] > 100, f"Body too small: {swc['body']}"
        # References found via heading fallback
        assert swc["references"] > 0
        # Abstract word count should use section boundary fallback
        assert result["abstract_word_count"] < 300

    def test_all_markers_wrong(self):
        """Every single marker is wrong — relies entirely on heading fallback + gap-fill."""
        result = self._mock_and_run(
            sections=[
                {"type": "title_page", "heading": "Title", "start_marker": "WRONG WRONG WRONG"},
                {"type": "abstract", "heading": "Abstract", "start_marker": "WRONG WRONG WRONG"},
                {"type": "body", "heading": "Introduction", "start_marker": "WRONG WRONG WRONG"},
                {"type": "references", "heading": "References", "start_marker": "WRONG WRONG WRONG"},
            ],
            abstract_start_marker="WRONG",
            abstract_end_marker="WRONG",
        )
        swc = result["section_word_counts"]
        # Abstract and References found via heading fallback
        assert swc["abstract"] < 300
        assert swc["references"] > 0
        assert swc["body"] > 100

    def test_joined_words_in_marker(self):
        """LLM quotes text with spaces but PDF has joined words (no spaces)."""
        result = self._mock_and_run(
            sections=[
                {"type": "title_page", "heading": "Title", "start_marker": "The Autocracy Bandwagon"},
                {"type": "abstract", "heading": "Abstract", "start_marker": "Abstract This paper examines how democratic politicians survive authoritarian reversals"},
                {"type": "body", "heading": "During", "start_marker": "During democratic breakdowns, democratic politicians occupy"},
                {"type": "references", "heading": "References", "start_marker": "References Acemoglu, Daron and James A Robinson"},
            ],
            abstract_start_marker="This paper examines how democratic politicians survive authoritarian reversals",
            abstract_end_marker="During democratic breakdowns, democratic politicians",
        )
        swc = result["section_word_counts"]
        # The flexible pattern should match despite joined words
        assert swc["abstract"] > 0 and swc["abstract"] < 200
        assert swc["body"] > 0


class TestApplyLLMAnalysisValidation:
    """Test that apply_llm_analysis doesn't override good heuristic results with bad LLM results."""

    def _make_metadata(self):
        return ManuscriptMetadata(
            filename="test.pdf",
            file_type="pdf",
            word_count=10000,
            section_word_counts=SectionWordCounts(
                title_page=100, abstract=150, body=5000,
                references=2000, footnotes=0, appendix=500,
                total=7750,
            ),
            has_abstract=True,
            abstract_word_count=150,
            has_references=True,
        )

    def test_good_llm_results_override_heuristic(self):
        metadata = self._make_metadata()
        analysis = {
            "section_word_counts": {
                "title_page": 80, "abstract": 160, "body": 5200,
                "references": 1800, "footnotes": 0, "appendix": 600,
            },
            "has_abstract": True,
            "abstract_word_count": 160,
            "has_references": True,
        }
        result = apply_llm_analysis(metadata, analysis)
        assert result.section_word_counts.body == 5200  # LLM value used

    def test_bad_llm_no_body_keeps_heuristic(self):
        """LLM says body=0 (boundary failed), heuristic should be kept."""
        metadata = self._make_metadata()
        analysis = {
            "section_word_counts": {
                "title_page": 100, "abstract": 7000, "body": 0,
                "references": 0, "footnotes": 0, "appendix": 0,
            },
            "has_abstract": True,
            "abstract_word_count": 7000,
            "has_references": True,
        }
        result = apply_llm_analysis(metadata, analysis)
        # Heuristic body=5000 should be kept, not overridden with 0
        assert result.section_word_counts.body == 5000

    def test_bad_llm_huge_abstract_keeps_heuristic(self):
        """LLM says abstract=5000 (body boundary missed), heuristic should be kept."""
        metadata = self._make_metadata()
        analysis = {
            "section_word_counts": {
                "title_page": 100, "abstract": 5000, "body": 2000,
                "references": 800, "footnotes": 0, "appendix": 0,
            },
            "has_abstract": True,
            "abstract_word_count": 5000,
            "has_references": True,
        }
        result = apply_llm_analysis(metadata, analysis)
        # Heuristic abstract=150 is plausible, LLM abstract=5000 is not
        assert result.section_word_counts.abstract == 150
