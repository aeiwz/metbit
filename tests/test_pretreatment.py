import numpy as np
import pytest

from metbit.pretreatment import Scaler


X2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


class TestScalerInit:
    def test_unknown_scaler_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown scaler"):
            Scaler(scaler="nonexistent")

    def test_valid_scalers_construct_without_error(self):
        for name in ("uv", "pareto", "mean", "minmax"):
            Scaler(scaler=name)


class TestAutoScaling:
    def test_uv_produces_zero_mean_unit_variance(self):
        scaler = Scaler(scaler="uv")
        result = scaler.fit(X2)

        np.testing.assert_allclose(result.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(result.std(axis=0), 1, atol=1e-12)

    def test_uv_scale_applies_stored_params(self):
        scaler = Scaler(scaler="uv")
        scaler.fit(X2)

        X_new = np.array([[3.0, 4.0]])
        out = scaler.scale(X_new)
        np.testing.assert_allclose(out.mean(), 0, atol=1e-12)


class TestParetoScaling:
    def test_pareto_centers_to_zero_mean(self):
        scaler = Scaler(scaler="pareto")
        result = scaler.fit(X2)

        np.testing.assert_allclose(result.mean(axis=0), 0, atol=1e-12)

    def test_pareto_std_is_sqrt_of_original_std(self):
        scaler = Scaler(scaler="pareto")
        result = scaler.fit(X2)

        expected_std = np.sqrt(X2.std(axis=0))
        assert np.all(result.std(axis=0) > 0)
        assert result.shape == X2.shape


class TestMeanCentering:
    def test_mean_produces_zero_mean(self):
        scaler = Scaler(scaler="mean")
        result = scaler.fit(X2)

        np.testing.assert_allclose(result.mean(axis=0), 0, atol=1e-12)

    def test_mean_scale_subtracts_center(self):
        scaler = Scaler(scaler="mean")
        scaler.fit(X2)

        X_new = np.array([[3.0, 4.0]])
        out = scaler.scale(X_new)
        expected = X_new - X2.mean(axis=0)
        np.testing.assert_allclose(out, expected, atol=1e-12)


class TestMinMaxScaling:
    def test_minmax_maps_to_zero_one(self):
        scaler = Scaler(scaler="minmax")
        result = scaler.fit(X2)

        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_minmax_constant_column_produces_nan_not_crash(self):
        X_const = np.array([[2.0, 1.0], [2.0, 2.0], [2.0, 3.0]])
        scaler = Scaler(scaler="minmax")

        with np.errstate(invalid="ignore"):
            result = scaler.fit(X_const)

        assert np.isnan(result[:, 0]).all()
        assert not np.isnan(result[:, 1]).any()
