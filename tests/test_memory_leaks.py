"""
Memory leak tests for metbit C extension kernels and Python dispatch layer.

Strategy
--------
Three complementary checks for each kernel:

1. **Reference-count check** – call once, grab result, delete it, run gc.collect(),
   then confirm that sys.getrefcount on the intermediate bytes objects returned to 1
   (owned only by the local name).  If Py_BuildValue used "O" instead of "N" this
   check catches the leaked increment.

2. **Growth check (tracemalloc)** – call the kernel N times in a burst, take before/
   after snapshots.  Divide net growth by N; if it exceeds a per-call threshold (a few
   KB) the test fails.  Genuine leaks grow linearly with N; one-off initialization costs
   do not.

3. **Repeated large allocation** – allocate a large matrix, call the kernel, del the
   result, repeat M times, assert RSS delta is bounded.  Uses tracemalloc so it works
   without psutil.

Markers
-------
@pytest.mark.slow – excluded from the default `pytest` run
Run with:  pytest -m slow --no-cov -q tests/test_memory_leaks.py
"""
from __future__ import annotations

import gc
import sys
import tracemalloc
from typing import Callable

import numpy as np
import pytest

try:
    import metbit._native_backend  # noqa: F401
    _NATIVE_AVAILABLE = True
except ImportError:
    _NATIVE_AVAILABLE = False

_skip_no_native = pytest.mark.skipif(
    not _NATIVE_AVAILABLE,
    reason="metbit._native_backend C extension not built",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _net_alloc(fn: Callable, reps: int = 200) -> float:
    """Return net bytes allocated per call (negative = freed more than allocated)."""
    gc.collect()
    tracemalloc.start()
    for _ in range(reps):
        result = fn()
        del result
    gc.collect()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # peak / reps gives worst-case per-call overhead (conservative)
    return peak / reps


def _growth_per_call(fn: Callable, reps: int = 300) -> float:
    """
    Run fn() reps times and return the slope of allocated memory vs call number
    (bytes/call).  A persistent leak produces a positive slope proportional to
    the leaked size; one-off costs flatten out.
    """
    gc.collect()
    snapshots = []
    tracemalloc.start()
    for i in range(reps):
        r = fn()
        del r
        if i % 50 == 49:
            snapshots.append(tracemalloc.get_traced_memory()[0])
    tracemalloc.stop()
    if len(snapshots) < 2:
        return 0.0
    # simple linear slope across checkpoint intervals
    dy = snapshots[-1] - snapshots[0]
    dx = (len(snapshots) - 1) * 50
    return dy / max(dx, 1)


# ---------------------------------------------------------------------------
# Reference count sanity (pure-Python, fast)
# ---------------------------------------------------------------------------

@_skip_no_native
class TestReferenceCount:
    """Verify C extension outputs are collectable after release.

    Direct sys.getrefcount checks are fragile with Python 3.14's optimised
    bytecode and coverage instrumentation, so we use tracemalloc-based
    allocation checks instead: call N times, del each result, assert that
    total allocated memory does not grow linearly (which a leaked reference
    would cause via the retained bytes buffer).
    """

    def _assert_no_per_call_leak(self, fn, reps: int = 100, max_bytes: int = 1024):
        """Run fn reps times; assert net tracemalloc growth < max_bytes total."""
        gc.collect()
        tracemalloc.start()
        snap_before = tracemalloc.take_snapshot()
        for _ in range(reps):
            r = fn()
            del r
        gc.collect()
        snap_after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        stats = snap_after.compare_to(snap_before, "lineno")
        net = sum(s.size_diff for s in stats if s.size_diff > 0)
        assert net < max_bytes, (
            f"Possible reference leak: {net} bytes grew over {reps} calls "
            f"(threshold {max_bytes} B). Top stat: {stats[0] if stats else 'none'}"
        )

    def test_nipals_full_no_leak(self):
        """nipals_full must not retain bytes buffers after result is deleted."""
        import metbit._native_backend as nb
        rng = np.random.default_rng(0)
        x = np.ascontiguousarray(rng.standard_normal((20, 15)))
        y = np.ascontiguousarray(rng.standard_normal(20))
        self._assert_no_per_call_leak(
            lambda: nb.nipals_full(memoryview(x), memoryview(y), 20, 15, 1e-10, 50),
            reps=100, max_bytes=4096,
        )

    def test_scale_transform_no_leak(self):
        """scale_transform output bytes must be freed after del."""
        import metbit._native_backend as nb
        x    = np.ascontiguousarray(np.random.default_rng(1).standard_normal((20, 30)))
        mean = np.zeros(30)
        s    = np.ones(30)
        self._assert_no_per_call_leak(
            lambda: nb.scale_transform(memoryview(x), memoryview(mean), memoryview(s), 20, 30),
            reps=100, max_bytes=4096,
        )

    def test_xcorr_max_shift_no_leak(self):
        """xcorr_max_shift returns a (int, float) tuple – verify no leak."""
        import metbit._native_backend as nb
        tmpl = np.ascontiguousarray(np.sin(np.linspace(0, 4, 200)))
        qry  = np.ascontiguousarray(np.roll(tmpl, 5))
        result = nb.xcorr_max_shift(memoryview(tmpl), memoryview(qry), 200, 20)
        assert isinstance(result, tuple) and len(result) == 2
        self._assert_no_per_call_leak(
            lambda: nb.xcorr_max_shift(memoryview(tmpl), memoryview(qry), 200, 20),
            reps=200, max_bytes=2048,
        )

    def test_pqn_median_no_leak(self):
        """pqn_median_quotient returns a float – verify no leak."""
        import metbit._native_backend as nb
        samp = np.ascontiguousarray(np.ones(100))
        ref  = np.ascontiguousarray(np.ones(100) * 2.0)
        result = nb.pqn_median_quotient(memoryview(samp), memoryview(ref), 100)
        assert isinstance(result, float)
        self._assert_no_per_call_leak(
            lambda: nb.pqn_median_quotient(memoryview(samp), memoryview(ref), 100),
            reps=200, max_bytes=2048,
        )


# ---------------------------------------------------------------------------
# Growth tests  (tracemalloc)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestMemoryGrowth:
    """Check that repeated calls do not produce linear memory growth."""

    # Threshold: allow up to 512 B of net growth per call (covers Python frame
    # overhead, numpy array bookkeeping, etc.).  A genuine leak from a 20×40
    # float64 matrix (6 400 B) would dwarf this.
    _MAX_BYTES_PER_CALL = 512

    def test_nipals_no_growth(self):
        """Repeated nipals() calls must not accumulate memory."""
        from metbit._native import nipals
        rng = np.random.default_rng(1)
        x = np.ascontiguousarray(rng.standard_normal((40, 50)))
        y = rng.standard_normal(40)
        slope = _growth_per_call(lambda: nipals(x, y), reps=300)
        assert slope < self._MAX_BYTES_PER_CALL, (
            f"nipals() leaks ~{slope:.0f} B/call"
        )

    def test_scale_transform_no_growth(self):
        """Repeated scale_transform() calls must not accumulate memory."""
        from metbit._native import scale_transform
        rng = np.random.default_rng(2)
        X    = np.ascontiguousarray(rng.standard_normal((100, 200)))
        mean = X.mean(axis=0)
        s    = X.std(axis=0)
        slope = _growth_per_call(lambda: scale_transform(X, mean, s), reps=300)
        assert slope < self._MAX_BYTES_PER_CALL, (
            f"scale_transform() leaks ~{slope:.0f} B/call"
        )

    def test_xcorr_no_growth(self):
        """Repeated xcorr_max_shift() calls must not accumulate memory."""
        from metbit._native import xcorr_max_shift
        tmpl = np.ascontiguousarray(np.sin(np.linspace(0, 10, 500)))
        qry  = np.ascontiguousarray(np.roll(tmpl, 7))
        slope = _growth_per_call(lambda: xcorr_max_shift(tmpl, qry, 20), reps=500)
        assert slope < self._MAX_BYTES_PER_CALL, (
            f"xcorr_max_shift() leaks ~{slope:.0f} B/call"
        )

    def test_pqn_no_growth(self):
        """Repeated pqn_median_quotient() calls must not accumulate memory."""
        from metbit._native import pqn_median_quotient
        samp = np.ascontiguousarray(np.random.default_rng(3).uniform(0.5, 2.0, 10_000))
        ref  = np.ascontiguousarray(np.random.default_rng(4).uniform(0.5, 2.0, 10_000))
        slope = _growth_per_call(lambda: pqn_median_quotient(samp, ref), reps=500)
        assert slope < self._MAX_BYTES_PER_CALL, (
            f"pqn_median_quotient() leaks ~{slope:.0f} B/call"
        )

    def test_pearson_columns_no_growth(self):
        """Existing pearson_columns kernel must not accumulate memory."""
        from metbit._native import pearson_columns
        rng = np.random.default_rng(5)
        mat = np.ascontiguousarray(rng.standard_normal((50, 200)), dtype=np.float64)
        slope = _growth_per_call(lambda: pearson_columns(mat, anchor_index=10), reps=300)
        assert slope < self._MAX_BYTES_PER_CALL, (
            f"pearson_columns() leaks ~{slope:.0f} B/call"
        )

    def test_vip_scores_no_growth(self):
        """Existing vip_scores kernel must not accumulate memory."""
        from metbit._native import vip_scores
        from sklearn.cross_decomposition import PLSRegression
        rng = np.random.default_rng(6)
        X = rng.standard_normal((60, 100))
        y = rng.integers(0, 2, 60).astype(float)
        pls = PLSRegression(n_components=3).fit(X, y)
        t = np.ascontiguousarray(pls.x_scores_)
        w = np.ascontiguousarray(pls.x_weights_)
        q = np.ascontiguousarray(pls.y_loadings_)
        slope = _growth_per_call(lambda: vip_scores(t, w, q), reps=300)
        assert slope < self._MAX_BYTES_PER_CALL, (
            f"vip_scores() leaks ~{slope:.0f} B/call"
        )


# ---------------------------------------------------------------------------
# Large allocation / deallocation cycle
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestLargeAllocationCycle:
    """Allocate large arrays, call kernel, release, repeat — net alloc must be bounded."""

    _MAX_NET_MB = 5.0  # allow up to 5 MB net after 20 large cycles

    def _net_mb(self, fn: Callable, reps: int = 20) -> float:
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        for _ in range(reps):
            r = fn()
            del r
        gc.collect()
        current, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return (current - base) / 1024 / 1024

    def test_nipals_large_cycle(self):
        """nipals() on a large matrix must release all temporaries."""
        from metbit._native import nipals
        rng = np.random.default_rng(10)
        def _run():
            x = np.ascontiguousarray(rng.standard_normal((100, 5000)))
            y = rng.standard_normal(100)
            w, u, c, t = nipals(x, y)
            return (w, u, c, t)
        net = self._net_mb(_run, reps=10)
        assert net < self._MAX_NET_MB, f"nipals large-cycle net alloc: {net:.2f} MB"

    def test_scale_transform_large_cycle(self):
        """scale_transform() must not retain output after del."""
        from metbit._native import scale_transform
        rng = np.random.default_rng(11)
        def _run():
            X    = np.ascontiguousarray(rng.standard_normal((500, 10_000)))
            mean = X.mean(axis=0)
            s    = X.std(axis=0)
            return scale_transform(X, mean, s)
        net = self._net_mb(_run, reps=5)
        assert net < self._MAX_NET_MB, f"scale_transform large-cycle net alloc: {net:.2f} MB"

    def test_pqn_large_cycle(self):
        """pqn_median_quotient() must free its internal sort buffer."""
        from metbit._native import pqn_median_quotient
        rng = np.random.default_rng(12)
        def _run():
            samp = np.ascontiguousarray(rng.uniform(0.5, 2.0, 100_000))
            ref  = np.ascontiguousarray(rng.uniform(0.5, 2.0, 100_000))
            return pqn_median_quotient(samp, ref)
        net = self._net_mb(_run, reps=20)
        assert net < self._MAX_NET_MB, f"pqn large-cycle net alloc: {net:.2f} MB"

    def test_xcorr_large_cycle(self):
        """xcorr_max_shift() must not retain the input buffers."""
        from metbit._native import xcorr_max_shift
        rng = np.random.default_rng(13)
        def _run():
            t = np.ascontiguousarray(rng.standard_normal(50_000))
            q = np.ascontiguousarray(np.roll(t, 10))
            return xcorr_max_shift(t, q, 50)
        net = self._net_mb(_run, reps=20)
        assert net < self._MAX_NET_MB, f"xcorr large-cycle net alloc: {net:.2f} MB"
