from metbit.annotate_peak import _sanitize_annotation_text, _sanitize_csv_cell


def test_sanitize_annotation_text_escapes_html_payload():
    raw = '<img src=x onerror=alert("xss")>'
    safe = _sanitize_annotation_text(raw)

    assert "<" not in safe
    assert ">" not in safe
    assert "&lt;img" in safe


def test_sanitize_annotation_text_trims_and_limits_size():
    raw = "  A" * 300
    safe = _sanitize_annotation_text(raw, max_len=32)

    assert safe == safe.strip()
    assert len(safe) <= 32


def test_sanitize_csv_cell_blocks_formula_injection():
    assert _sanitize_csv_cell("=1+1") == "'=1+1"
    assert _sanitize_csv_cell("+SUM(A1:A2)") == "'+SUM(A1:A2)"
    assert _sanitize_csv_cell("-10+20") == "'-10+20"
    assert _sanitize_csv_cell("@HYPERLINK(\"x\")") == "'@HYPERLINK(\"x\")"


def test_sanitize_csv_cell_leaves_normal_text_unchanged():
    assert _sanitize_csv_cell("Peak_A") == "Peak_A"
