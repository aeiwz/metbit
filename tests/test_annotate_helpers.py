import pytest

from metbit.annotate_peak import _sanitize_annotation_text, _sanitize_csv_cell


class TestSanitizeAnnotationText:
    def test_empty_string_returns_empty(self):
        assert _sanitize_annotation_text("") == ""

    def test_none_like_returns_empty(self):
        assert _sanitize_annotation_text(None) == ""

    def test_normal_text_unchanged(self):
        result = _sanitize_annotation_text("Alanine")
        assert result == "Alanine"

    def test_html_special_chars_escaped(self):
        result = _sanitize_annotation_text("<script>alert(1)</script>")
        assert "<" not in result
        assert ">" not in result

    def test_ampersand_escaped(self):
        result = _sanitize_annotation_text("A & B")
        assert "&" not in result or "&amp;" in result

    def test_long_text_truncated_to_max_len(self):
        long_text = "x" * 500
        result = _sanitize_annotation_text(long_text, max_len=100)
        assert len(result) <= 100

    def test_whitespace_only_returns_empty(self):
        assert _sanitize_annotation_text("   ") == ""

    def test_quotes_escaped(self):
        result = _sanitize_annotation_text('say "hello"', max_len=256)
        assert '"' not in result or "&quot;" in result


class TestSanitizeCsvCell:
    def test_normal_text_unchanged(self):
        assert _sanitize_csv_cell("Glucose") == "Glucose"

    def test_formula_starting_with_equals_prefixed(self):
        result = _sanitize_csv_cell("=SUM(A1:A10)")
        assert result.startswith("'")

    def test_formula_starting_with_plus_prefixed(self):
        result = _sanitize_csv_cell("+123")
        assert result.startswith("'")

    def test_formula_starting_with_minus_prefixed(self):
        result = _sanitize_csv_cell("-1")
        assert result.startswith("'")

    def test_at_sign_prefixed(self):
        result = _sanitize_csv_cell("@user")
        assert result.startswith("'")

    def test_tab_prefixed(self):
        result = _sanitize_csv_cell("\tcell")
        assert result.startswith("'")

    def test_empty_string_unchanged(self):
        assert _sanitize_csv_cell("") == ""

    def test_numeric_string_unchanged(self):
        assert _sanitize_csv_cell("3.14") == "3.14"

    def test_none_becomes_string(self):
        result = _sanitize_csv_cell(None)
        assert isinstance(result, str)
