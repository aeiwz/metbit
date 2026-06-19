"""Covers metbit/_compat.py (backwards-compatibility re-exports)."""
import importlib


def test_compat_importable():
    mod = importlib.import_module("metbit._compat")
    assert hasattr(mod, "opls_da")
    assert hasattr(mod, "Scaler")
    assert hasattr(mod, "UnivarStats")
    assert hasattr(mod, "Normalise")
