import numpy as np
import pytest

from metbit.pca_ellipse import confidence_ellipse


class TestConfidenceEllipse:
    def test_returns_svg_path_string(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 3.0, 4.0, 5.0])

        path = confidence_ellipse(x, y)

        assert isinstance(path, str)
        assert path.startswith("M ")
        assert path.endswith("Z")

    def test_mismatched_sizes_raise_value_error(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="same size"):
            confidence_ellipse(x, y)

    def test_path_contains_line_segments(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(30)
        y = rng.standard_normal(30)

        path = confidence_ellipse(x, y, size=50)

        assert path.count("L") == 49

    def test_custom_n_std_changes_ellipse_scale(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)

        path_small = confidence_ellipse(x, y, n_std=1.0)
        path_large = confidence_ellipse(x, y, n_std=3.0)

        assert path_small != path_large

    def test_circular_data_produces_valid_path(self):
        theta = np.linspace(0, 2 * np.pi, 50)
        x = np.cos(theta)
        y = np.sin(theta)

        path = confidence_ellipse(x, y)

        assert "M " in path
        assert "Z" in path
