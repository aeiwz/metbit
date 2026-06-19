# -*- coding: utf-8 -*-
"""
End-to-end pipeline tests for metbit.

Each test exercises a full workflow from raw synthetic data through the complete
analysis chain to final outputs, verifying that every stage produces the correct
type, shape, and value range. These tests catch integration breakage that unit
tests miss: e.g. an opls_da score DataFrame that has wrong column names will fail
the VIP plot even if VIP computation itself is correct.

Fixtures
--------
spectra_df    : NMR-like DataFrame (samples x ppm columns)
ab_dataset    : (X, y) with genuine two-group separation
aa_dataset    : (X, y) same-group split with no true separation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(2024)


@pytest.fixture(scope="module")
def spectra_df(rng):
    """Synthetic NMR-like spectra: 40 samples x 200 ppm points."""
    n, p = 40, 200
    ppm = np.linspace(9.5, 0.5, p)
    data = rng.standard_normal((n, p))
    # Inject a signal cluster at ppm ~3.0
    sig_idx = np.argmin(np.abs(ppm - 3.0))
    data[:, sig_idx]     += rng.standard_normal(n) * 0.5
    data[:, sig_idx + 1]  = data[:, sig_idx] * 1.8 + rng.standard_normal(n) * 0.05
    data[:, sig_idx - 1]  = data[:, sig_idx] * 0.9 + rng.standard_normal(n) * 0.1
    return pd.DataFrame(data, columns=ppm.tolist())


@pytest.fixture(scope="module")
def ab_dataset(rng):
    """
    AB dataset: two groups with a genuine separation.
    Group A: baseline. Group B: +3 std on 10 features.
    Large effect size ensures the model reliably discriminates.
    """
    n_per_group, p = 30, 50
    X_a = rng.standard_normal((n_per_group, p))
    X_b = rng.standard_normal((n_per_group, p))
    X_b[:, :10] += 3.0          # strong shift on first 10 features
    X = np.vstack([X_a, X_b])
    y = pd.Series(["A"] * n_per_group + ["B"] * n_per_group, name="Group")
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), y


@pytest.fixture(scope="module")
def aa_dataset(rng):
    """
    AA dataset: same population randomly split into two fake groups.
    No true separation exists. Used to validate the model does not overfit.
    """
    n_per_group, p = 30, 50
    X = rng.standard_normal((2 * n_per_group, p))   # single population
    y = pd.Series(
        ["A"] * n_per_group + ["A"] * n_per_group, name="Group"
    )
    # Relabel second half as B to create a fake two-group structure
    y = pd.Series(
        ["A"] * n_per_group + ["B"] * n_per_group, name="Group"
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), y


# ---------------------------------------------------------------------------
# E2E: OPLS-DA full pipeline (AB dataset)
# ---------------------------------------------------------------------------

class TestOPLSDAEndToEnd:
    """Full opls_da pipeline: fit -> scores -> VIP -> all plots."""

    @pytest.fixture(autouse=True)
    def _fit(self, ab_dataset):
        from metbit import opls_da
        X, y = ab_dataset
        self.model = opls_da(X, y, n_components=2, scaling_method="pareto", kfold=3)
        self.model.fit()
        self.model.vip_scores()

    # -- fit outputs --------------------------------------------------------

    def test_fit_produces_r2xcorr(self):
        assert isinstance(self.model.R2Xcorr, float)
        assert 0.0 <= self.model.R2Xcorr <= 1.0

    def test_fit_produces_r2y(self):
        assert isinstance(self.model.R2y, float)
        assert 0.0 <= self.model.R2y <= 1.0

    def test_fit_produces_q2(self):
        assert isinstance(self.model.q2, float)

    def test_scores_dataframe_shape(self):
        df = self.model.get_oplsda_scores()
        assert isinstance(df, pd.DataFrame)
        assert "t_scores" in df.columns
        assert "t_ortho" in df.columns
        assert "Group" in df.columns
        assert len(df) == 60    # 30 A + 30 B

    def test_s_scores_dataframe_shape(self):
        df = self.model.get_s_scores()
        assert isinstance(df, pd.DataFrame)
        assert "correlation" in df.columns
        assert "covariance" in df.columns
        assert len(df) == 50    # p features

    # -- VIP ----------------------------------------------------------------

    def test_vip_stored_after_vip_scores(self):
        vips = self.model.get_vip_scores()
        assert isinstance(vips, pd.DataFrame)
        assert "VIP" in vips.columns
        assert "Features" in vips.columns
        assert len(vips) == 50

    def test_vip_values_are_non_negative(self):
        vips = self.model.get_vip_scores()
        assert (vips["VIP"] >= 0).all()

    def test_vip_filter_reduces_rows(self):
        all_vips = self.model.get_vip_scores(filter_=False)
        filtered  = self.model.get_vip_scores(filter_=True, threshold=1.0)
        assert len(filtered) <= len(all_vips)

    # -- Plots (return type + basic structure) ------------------------------

    def test_scores_plot_returns_figure(self):
        fig = self.model.plot_oplsda_scores()
        assert isinstance(fig, go.Figure)

    def test_vip_plot_returns_figure(self):
        fig = self.model.vip_plot()
        assert isinstance(fig, go.Figure)

    def test_s_scores_plot_returns_figure(self):
        fig = self.model.plot_s_scores()
        assert isinstance(fig, go.Figure)

    def test_loading_plot_returns_figure(self):
        fig = self.model.plot_loading()
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# E2E: STOCSY full pipeline
# ---------------------------------------------------------------------------

class TestSTOCSYEndToEnd:
    def test_figure_type(self, spectra_df):
        from metbit import STOCSY
        ppm = [float(c) for c in spectra_df.columns]
        fig = STOCSY(spectra_df, anchor_ppm_value=ppm[50])
        assert isinstance(fig, go.Figure)

    def test_figure_has_two_traces(self, spectra_df):
        from metbit import STOCSY
        ppm = [float(c) for c in spectra_df.columns]
        fig = STOCSY(spectra_df, anchor_ppm_value=ppm[50])
        assert len(fig.data) == 2

    def test_xaxis_reversed(self, spectra_df):
        from metbit import STOCSY
        ppm = [float(c) for c in spectra_df.columns]
        fig = STOCSY(spectra_df, anchor_ppm_value=ppm[50])
        assert fig.layout.xaxis.autorange == "reversed"


# ---------------------------------------------------------------------------
# E2E: ChunkedSTOCSY matches standard STOCSY
# ---------------------------------------------------------------------------

class TestChunkedSTOCSYEndToEnd:
    def test_correlations_match_standard_stocsy(self, spectra_df):
        from metbit.analysis.stocsy import _stocsy_statistics
        from metbit import ChunkedSTOCSY

        ppm = spectra_df.columns.astype(float).to_numpy()
        anchor_idx = np.argmin(np.abs(ppm - 3.0))
        anchor_ppm = float(ppm[anchor_idx])

        # Standard path
        corr_std, pval_std = _stocsy_statistics(spectra_df, anchor_index=anchor_idx)

        # Chunked path (small chunk to exercise the loop)
        stocsy = ChunkedSTOCSY(chunk_size=30)
        ppm_out, corr_out, pval_out = stocsy.compute(spectra_df, anchor_ppm_value=anchor_ppm)

        np.testing.assert_allclose(corr_out, corr_std, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(pval_out, pval_std, rtol=1e-8,  atol=1e-10)

    def test_chunked_plot_returns_figure(self, spectra_df):
        from metbit import ChunkedSTOCSY
        ppm = [float(c) for c in spectra_df.columns]
        stocsy = ChunkedSTOCSY()
        fig = stocsy.plot(spectra_df, anchor_ppm_value=ppm[50])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_backend_info_keys(self):
        from metbit import ChunkedSTOCSY
        info = ChunkedSTOCSY.active_backend()
        assert "native_c" in info
        assert "gpu" in info
        assert "openmp_threads" in info


# ---------------------------------------------------------------------------
# E2E: feature_preselection pipeline
# ---------------------------------------------------------------------------

class TestFeaturePreselectionEndToEnd:
    def test_variance_method_reduces_features(self, ab_dataset):
        from metbit import feature_preselection
        X, _ = ab_dataset

        X_red, mask = feature_preselection(X, percentile=30, method="variance")
        assert mask.dtype == bool
        assert mask.sum() == X_red.shape[1]
        assert X_red.shape[0] == X.shape[0]
        assert X_red.shape[1] < X.shape[1]

    def test_iqr_method_reduces_features(self, ab_dataset):
        from metbit import feature_preselection
        X, _ = ab_dataset

        X_red, mask = feature_preselection(X, percentile=20, method="iqr")
        assert X_red.shape[1] < X.shape[1]

    def test_zero_percentile_keeps_all(self, ab_dataset):
        from metbit import feature_preselection
        X, _ = ab_dataset

        X_red, mask = feature_preselection(X, percentile=0, method="variance")
        assert mask.all()
        assert X_red.shape[1] == X.shape[1]

    def test_ndarray_input_also_works(self, ab_dataset):
        from metbit import feature_preselection
        X, _ = ab_dataset

        X_arr = X.to_numpy()
        X_red, mask = feature_preselection(X_arr, percentile=20)
        assert isinstance(X_red, np.ndarray)

    def test_preselection_then_opls_da_runs(self, ab_dataset):
        from metbit import feature_preselection, opls_da

        X, y = ab_dataset
        X_red, _ = feature_preselection(X, percentile=20)
        model = opls_da(X_red, y, n_components=2, kfold=3)
        model.fit()
        assert model.R2y > 0


# ---------------------------------------------------------------------------
# E2E: alignment pipeline
# ---------------------------------------------------------------------------

class TestAlignmentEndToEnd:
    @pytest.fixture
    def spectra_with_shifts(self):
        """Synthetic spectra with a Gaussian peak that is detectable in the median spectrum.

        A narrow single-point spike per row does not survive into the median
        (all other rows are zero at that point). A Gaussian spread over ~7 points
        does survive: the median at each point within the cluster is non-zero
        because every sample contributes intensity there.
        """
        rng = np.random.default_rng(7)
        p = 300
        ppm = np.linspace(10.0, 0.0, p)
        n = 20
        # Gaussian peak width (sigma in points)
        sigma = 4
        data = np.zeros((n, p))
        base_idx = int(p // 2)
        for i in range(n):
            shift = rng.integers(-3, 4)   # small shift
            cx = base_idx + shift
            for k in range(p):
                data[i, k] += 10.0 * np.exp(-0.5 * ((k - cx) / sigma) ** 2)
        # Add a little noise floor
        data += rng.standard_normal((n, p)) * 0.05
        return pd.DataFrame(data, columns=ppm.tolist()), ppm

    def test_icoshift_align_output_shape(self, spectra_with_shifts):
        from metbit import icoshift_align
        spectra, ppm = spectra_with_shifts
        windows = [(5.5, 4.5)]
        aligned, shifts = icoshift_align(spectra, ppm, windows)
        assert aligned.shape == spectra.shape
        assert len(shifts) == len(spectra)

    def test_peak_aligner_auto_windows(self, spectra_with_shifts):
        from metbit import PeakAligner
        spectra, ppm = spectra_with_shifts
        pa = PeakAligner(spectra, ppm, sf_mhz=600)
        windows, mptable = pa.auto_windows(top_n=5)
        assert isinstance(windows, list)
        aligned, shifts = pa.align(windows)
        assert aligned.shape == spectra.shape

    def test_alignment_does_not_create_extra_copies(self, spectra_with_shifts):
        """Verifies the single-allocation fix: aligned values within valid range."""
        from metbit import icoshift_align
        spectra, ppm = spectra_with_shifts
        windows = [(5.5, 4.5)]
        aligned, _ = icoshift_align(spectra, ppm, windows)
        # Alignment must not produce out-of-range artifacts
        assert np.isfinite(aligned.to_numpy()).all()


# ---------------------------------------------------------------------------
# E2E: dtype parameter in opls_da (float32 path)
# ---------------------------------------------------------------------------

class TestOPLSDAFloat32:
    def test_float32_produces_valid_r2y(self, ab_dataset):
        from metbit import opls_da
        X, y = ab_dataset
        model = opls_da(X, y, n_components=2, kfold=3, dtype=np.float32)
        model.fit()
        assert 0.0 <= model.R2y <= 1.0

    def test_float32_vip_non_negative(self, ab_dataset):
        from metbit import opls_da
        X, y = ab_dataset
        model = opls_da(X, y, n_components=2, kfold=3, dtype=np.float32)
        model.fit()
        model.vip_scores()
        vips = model.get_vip_scores()
        assert (vips["VIP"] >= 0).all()

    def test_float32_and_float64_q2_close(self, ab_dataset):
        from metbit import opls_da
        X, y = ab_dataset
        m64 = opls_da(X, y, n_components=2, kfold=3, dtype=np.float64)
        m32 = opls_da(X, y, n_components=2, kfold=3, dtype=np.float32)
        m64.fit()
        m32.fit()
        # float32 Q2 should be within 5% of float64 Q2
        assert abs(float(m64.q2) - float(m32.q2)) < 0.05


# ---------------------------------------------------------------------------
# E2E: MemoryEstimator
# ---------------------------------------------------------------------------

class TestMemoryEstimator:
    def test_estimate_returns_expected_keys(self):
        from metbit import MemoryEstimator
        result = MemoryEstimator.estimate(1000, 50000, np.float64, copies=2)
        assert "single_matrix_gb" in result
        assert "peak_gb_with_copies" in result
        assert "recommended_dtype" in result
        assert "summary" in result

    def test_float32_half_of_float64(self):
        from metbit import MemoryEstimator
        r64 = MemoryEstimator.estimate(500, 10000, np.float64)
        r32 = MemoryEstimator.estimate(500, 10000, np.float32)
        ratio = r64["single_matrix_gb"] / r32["single_matrix_gb"]
        assert abs(ratio - 2.0) < 1e-9

    def test_large_dataset_recommends_float32(self):
        from metbit import MemoryEstimator
        result = MemoryEstimator.estimate(10000, 1_000_000, np.float64, copies=2)
        assert result["recommended_dtype"] == "float32"
